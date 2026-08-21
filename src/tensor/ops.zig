const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const math = @import("math.zig");
const gemm = @import("gemm.zig");
const parallel = @import("parallel.zig");

const VEC = gemm.VEC;
const Vec = gemm.Vec;

/// Output columns held in vector registers while the channel and kernel loops
/// run underneath them, so each accumulator is written to memory once.
const OW_TILE = 4 * VEC;

/// Copies `[N, C, H, W]` into `[N, C, H + 2p, W + 2p]` with a zero border, so
/// the convolution inner loop needs no bounds checks.
fn zeroPad(allocator: std.mem.Allocator, input: Tensor, pad: usize) !Tensor {
    const batch = input.shape[0];
    const channels = input.shape[1];
    const h = input.shape[2];
    const w = input.shape[3];

    const shape = [_]usize{ batch, channels, h + 2 * pad, w + 2 * pad };
    var out = try Tensor.initZeros(allocator, &shape);
    errdefer out.deinit();

    const w_pad = w + 2 * pad;
    for (0..batch * channels) |plane| {
        const src_plane = input.data[plane * h * w ..][0 .. h * w];
        const dst_plane = out.data[plane * (h + 2 * pad) * w_pad ..][0 .. (h + 2 * pad) * w_pad];
        for (0..h) |y| {
            @memcpy(dst_plane[(y + pad) * w_pad + pad ..][0..w], src_plane[y * w ..][0..w]);
        }
    }

    return out;
}

/// A 1x1 convolution is `[C_out, C_in] . [C_in, H*W]`, so it goes straight to
/// the blocked GEMM.
fn conv2dPointwise(
    allocator: std.mem.Allocator,
    input: Tensor,
    weight: Tensor,
    bias: ?Tensor,
    out: *Tensor,
) !void {
    const batch = input.shape[0];
    const c_in = input.shape[1];
    const plane = input.shape[2] * input.shape[3];
    const c_out = weight.shape[0];

    for (0..batch) |b| {
        try gemm.gemm(
            allocator,
            c_out,
            plane,
            c_in,
            weight.data,
            c_in,
            input.data[b * c_in * plane ..][0 .. c_in * plane],
            plane,
            .row_major,
            null,
            out.data[b * c_out * plane ..][0 .. c_out * plane],
        );

        if (bias) |bi| {
            for (0..c_out) |co| {
                const row = out.data[(b * c_out + co) * plane ..][0..plane];
                const bv: Vec = @splat(bi.data[co]);
                var i: usize = 0;
                while (i + VEC <= plane) : (i += VEC) {
                    const v: Vec = row[i..][0..VEC].*;
                    row[i..][0..VEC].* = v + bv;
                }
                while (i < plane) : (i += 1) row[i] += bi.data[co];
            }
        }
    }
}

pub fn conv2d(
    allocator: std.mem.Allocator,
    input: Tensor,
    weight: Tensor,
    bias: ?Tensor,
    stride: usize,
    padding: usize,
) !Tensor {
    // Input: [N, C_in, H_in, W_in]
    // Weight: [C_out, C_in, K_h, K_w]
    std.debug.assert(input.shape.len == 4);
    std.debug.assert(weight.shape.len == 4);

    const batch = input.shape[0];
    const c_in = input.shape[1];
    const h_in = input.shape[2];
    const w_in = input.shape[3];

    const c_out = weight.shape[0];
    std.debug.assert(weight.shape[1] == c_in);
    const k_h = weight.shape[2];
    const k_w = weight.shape[3];

    const h_out = (h_in + 2 * padding - k_h) / stride + 1;
    const w_out = (w_in + 2 * padding - k_w) / stride + 1;

    const out_shape = [_]usize{ batch, c_out, h_out, w_out };
    var out = try Tensor.init(allocator, &out_shape);
    errdefer out.deinit();

    if (k_h == 1 and k_w == 1 and stride == 1 and padding == 0) {
        try conv2dPointwise(allocator, input, weight, bias, &out);
        return out;
    }

    // Padding is materialised once rather than tested per tap.
    var padded: ?Tensor = if (padding > 0) try zeroPad(allocator, input, padding) else null;
    defer if (padded) |*p| p.deinit();
    const src = if (padded) |p| p else input;
    const h_src = src.shape[2];
    const w_src = src.shape[3];

    const Context = struct {
        src: Tensor,
        weight: Tensor,
        bias: ?Tensor,
        out: *Tensor,
        c_in: usize,
        h_src: usize,
        w_src: usize,
        c_out: usize,
        k_h: usize,
        k_w: usize,
        h_out: usize,
        w_out: usize,
        stride: usize,
    };

    var ctx = Context{
        .src = src,
        .weight = weight,
        .bias = bias,
        .out = &out,
        .c_in = c_in,
        .h_src = h_src,
        .w_src = w_src,
        .c_out = c_out,
        .k_h = k_h,
        .k_w = k_w,
        .h_out = h_out,
        .w_out = w_out,
        .stride = stride,
    };

    parallel.parallelFor(allocator, batch * c_out, &ctx, struct {
        fn worker(c: *Context, start: usize, end: usize) void {
            const plane_src = c.h_src * c.w_src;
            const taps = c.k_h * c.k_w;

            for (start..end) |task| {
                const b = task / c.c_out;
                const co = task % c.c_out;
                const bias_val = if (c.bias) |bi| bi.data[co] else 0.0;
                const w_base = co * c.c_in * taps;

                for (0..c.h_out) |oh| {
                    const out_row = c.out.data[(task * c.h_out + oh) * c.w_out ..][0..c.w_out];

                    if (c.stride == 1) {
                        // Tiled over output columns: the tile stays in registers
                        // across every input channel and kernel tap.
                        var ow0: usize = 0;
                        while (ow0 < c.w_out) : (ow0 += OW_TILE) {
                            const width = @min(OW_TILE, c.w_out - ow0);
                            var acc: [OW_TILE]f32 = @splat(bias_val);

                            for (0..c.c_in) |ci| {
                                const in_plane = c.src.data[(b * c.c_in + ci) * plane_src ..][0..plane_src];
                                const w_ch = c.weight.data[w_base + ci * taps ..][0..taps];

                                for (0..c.k_h) |kh| {
                                    const in_row = in_plane[(oh + kh) * c.w_src ..];
                                    for (0..c.k_w) |kw| {
                                        const wv = w_ch[kh * c.k_w + kw];
                                        if (wv == 0.0) continue;
                                        const tap = in_row[ow0 + kw ..];

                                        if (width == OW_TILE) {
                                            const wsplat: Vec = @splat(wv);
                                            inline for (0..OW_TILE / VEC) |t| {
                                                const iv: Vec = tap[t * VEC ..][0..VEC].*;
                                                const av: Vec = acc[t * VEC ..][0..VEC].*;
                                                acc[t * VEC ..][0..VEC].* = av + wsplat * iv;
                                            }
                                        } else {
                                            for (0..width) |t| acc[t] += wv * tap[t];
                                        }
                                    }
                                }
                            }

                            @memcpy(out_row[ow0..][0..width], acc[0..width]);
                        }
                    } else {
                        // Strided: contiguity over output columns is gone, so
                        // reduce along the (contiguous) kernel row instead.
                        for (0..c.w_out) |ow| {
                            var sum: f32 = bias_val;
                            for (0..c.c_in) |ci| {
                                const in_plane = c.src.data[(b * c.c_in + ci) * plane_src ..][0..plane_src];
                                const w_ch = c.weight.data[w_base + ci * taps ..][0..taps];
                                for (0..c.k_h) |kh| {
                                    const in_row = in_plane[(oh * c.stride + kh) * c.w_src + ow * c.stride ..][0..c.k_w];
                                    sum += math.dotProduct(in_row, w_ch[kh * c.k_w ..][0..c.k_w]);
                                }
                            }
                            out_row[ow] = sum;
                        }
                    }
                }
            }
        }
    }.worker);

    return out;
}


/// `convTranspose2d` for `kernel == stride`, `padding == 0`: one GEMM per kernel
/// tap, each writing a disjoint sub-lattice of the output.
fn convTranspose2dStrided(
    allocator: std.mem.Allocator,
    input: Tensor,
    weight: Tensor,
    bias: ?Tensor,
    stride: usize,
    out: *Tensor,
) !void {
    const batch = input.shape[0];
    const c_in = input.shape[1];
    const h_in = input.shape[2];
    const w_in = input.shape[3];
    const c_out = weight.shape[1];

    const plane_in = h_in * w_in;
    const taps = stride * stride;
    const w_out = w_in * stride;

    // weight is [C_in, C_out, k, k]; the GEMM wants [C_out, C_in] per tap.
    const tap_weights = try allocator.alloc(f32, c_out * c_in);
    defer allocator.free(tap_weights);

    const product = try allocator.alloc(f32, c_out * plane_in);
    defer allocator.free(product);

    for (0..batch) |b| {
        for (0..taps) |tap| {
            const kh = tap / stride;
            const kw = tap % stride;

            for (0..c_out) |co| {
                for (0..c_in) |ci| {
                    tap_weights[co * c_in + ci] = weight.data[((ci * c_out + co) * stride + kh) * stride + kw];
                }
            }

            try gemm.gemm(
                allocator,
                c_out,
                plane_in,
                c_in,
                tap_weights,
                c_in,
                input.data[b * c_in * plane_in ..][0 .. c_in * plane_in],
                plane_in,
                .row_major,
                null,
                product,
            );

            for (0..c_out) |co| {
                const bias_val = if (bias) |bi| bi.data[co] else 0.0;
                const src = product[co * plane_in ..][0..plane_in];
                for (0..h_in) |ih| {
                    const dst_row = out.data[((b * c_out + co) * h_in * stride + ih * stride + kh) * w_out ..][0..w_out];
                    const src_row = src[ih * w_in ..][0..w_in];
                    for (0..w_in) |iw| {
                        dst_row[iw * stride + kw] = src_row[iw] + bias_val;
                    }
                }
            }
        }
    }
}

pub fn convTranspose2d(
    allocator: std.mem.Allocator,
    input: Tensor,
    weight: Tensor,
    bias: ?Tensor,
    stride: usize,
    padding: usize,
) !Tensor {
    std.debug.assert(input.shape.len == 4);
    std.debug.assert(weight.shape.len == 4);

    const batch = input.shape[0];
    const c_in = input.shape[1];
    const h_in = input.shape[2];
    const w_in = input.shape[3];

    std.debug.assert(weight.shape[0] == c_in);
    const c_out = weight.shape[1];
    const k_h = weight.shape[2];
    const k_w = weight.shape[3];

    const h_out = (h_in - 1) * stride - 2 * padding + k_h;
    const w_out = (w_in - 1) * stride - 2 * padding + k_w;

    const out_shape = [_]usize{ batch, c_out, h_out, w_out };
    var out = try Tensor.initZeros(allocator, &out_shape);
    errdefer out.deinit();

    // The FPN scale layers all use kernel == stride with no padding. Each output
    // pixel then comes from exactly one input pixel and one kernel tap, so the
    // whole thing is `stride * stride` independent GEMMs of
    // `[C_out, C_in] . [C_in, H_in*W_in]` scattered into the output grid.
    if (k_h == stride and k_w == stride and padding == 0) {
        try convTranspose2dStrided(allocator, input, weight, bias, stride, &out);
        return out;
    }

    // Gathering per output pixel (rather than scattering per input pixel) keeps
    // every task's writes disjoint, so the batch/channel loop parallelises.
    const Context = struct {
        input: Tensor,
        weight: Tensor,
        bias: ?Tensor,
        out: *Tensor,
        c_in: usize,
        h_in: usize,
        w_in: usize,
        c_out: usize,
        k_h: usize,
        k_w: usize,
        h_out: usize,
        w_out: usize,
        stride: usize,
        padding: usize,
    };

    var ctx = Context{
        .input = input,
        .weight = weight,
        .bias = bias,
        .out = &out,
        .c_in = c_in,
        .h_in = h_in,
        .w_in = w_in,
        .c_out = c_out,
        .k_h = k_h,
        .k_w = k_w,
        .h_out = h_out,
        .w_out = w_out,
        .stride = stride,
        .padding = padding,
    };

    parallel.parallelFor(allocator, batch * c_out, &ctx, struct {
        fn worker(c: *Context, start: usize, end: usize) void {
            for (start..end) |task| {
                const b = task / c.c_out;
                const co = task % c.c_out;
                const b_val = if (c.bias) |bi| bi.data[co] else 0.0;

                for (0..c.h_out) |oh| {
                    for (0..c.w_out) |ow| {
                        var sum: f32 = b_val;

                        for (0..c.k_h) |kh| {
                            // oh = ih * stride + kh - padding
                            const num_h = @as(isize, @intCast(oh + c.padding)) - @as(isize, @intCast(kh));
                            if (num_h < 0) continue;
                            if (@rem(num_h, @as(isize, @intCast(c.stride))) != 0) continue;
                            const ih = @as(usize, @intCast(@divExact(num_h, @as(isize, @intCast(c.stride)))));
                            if (ih >= c.h_in) continue;

                            for (0..c.k_w) |kw| {
                                const num_w = @as(isize, @intCast(ow + c.padding)) - @as(isize, @intCast(kw));
                                if (num_w < 0) continue;
                                if (@rem(num_w, @as(isize, @intCast(c.stride))) != 0) continue;
                                const iw = @as(usize, @intCast(@divExact(num_w, @as(isize, @intCast(c.stride)))));
                                if (iw >= c.w_in) continue;

                                for (0..c.c_in) |ci| {
                                    sum += c.input.at4(b, ci, ih, iw) * c.weight.at4(ci, co, kh, kw);
                                }
                            }
                        }

                        c.out.set4(b, co, oh, ow, sum);
                    }
                }
            }
        }
    }.worker);

    return out;
}

/// Non-overlapping max pooling, matching `nn.MaxPool2d(kernel_size, stride)`.
pub fn maxPool2d(
    allocator: std.mem.Allocator,
    input: Tensor,
    kernel: usize,
    stride: usize,
) !Tensor {
    std.debug.assert(input.shape.len == 4);

    const batch = input.shape[0];
    const channels = input.shape[1];
    const h_in = input.shape[2];
    const w_in = input.shape[3];

    const h_out = (h_in - kernel) / stride + 1;
    const w_out = (w_in - kernel) / stride + 1;

    const out_shape = [_]usize{ batch, channels, h_out, w_out };
    var out = try Tensor.init(allocator, &out_shape);

    const Context = struct {
        input: Tensor,
        out: *Tensor,
        channels: usize,
        h_out: usize,
        w_out: usize,
        kernel: usize,
        stride: usize,
    };

    var ctx = Context{
        .input = input,
        .out = &out,
        .channels = channels,
        .h_out = h_out,
        .w_out = w_out,
        .kernel = kernel,
        .stride = stride,
    };

    parallel.parallelFor(allocator, batch * channels, &ctx, struct {
        fn worker(c: *Context, start: usize, end: usize) void {
            for (start..end) |task| {
                const b = task / c.channels;
                const ch = task % c.channels;

                for (0..c.h_out) |oh| {
                    for (0..c.w_out) |ow| {
                        var best: f32 = -std.math.inf(f32);
                        for (0..c.kernel) |kh| {
                            for (0..c.kernel) |kw| {
                                const v = c.input.at4(b, ch, oh * c.stride + kh, ow * c.stride + kw);
                                if (v > best) best = v;
                            }
                        }
                        c.out.set4(b, ch, oh, ow, best);
                    }
                }
            }
        }
    }.worker);

    return out;
}


pub fn bilinearUpsample(
    allocator: std.mem.Allocator,
    input: Tensor,
    target_h: usize,
    target_w: usize,
    align_corners: bool,
) !Tensor {
    std.debug.assert(input.shape.len == 4);
    const batch = input.shape[0];
    const channels = input.shape[1];
    const in_h = input.shape[2];
    const in_w = input.shape[3];

    const out_shape = [_]usize{ batch, channels, target_h, target_w };
    var out = try Tensor.init(allocator, &out_shape);

    const r_h: f32 = if (align_corners and target_h > 1)
        @as(f32, @floatFromInt(in_h - 1)) / @as(f32, @floatFromInt(target_h - 1))
    else
        @as(f32, @floatFromInt(in_h)) / @as(f32, @floatFromInt(target_h));

    const r_w: f32 = if (align_corners and target_w > 1)
        @as(f32, @floatFromInt(in_w - 1)) / @as(f32, @floatFromInt(target_w - 1))
    else
        @as(f32, @floatFromInt(in_w)) / @as(f32, @floatFromInt(target_w));

    const Context = struct {
        input: Tensor,
        out: *Tensor,
        batch: usize,
        channels: usize,
        in_h: usize,
        in_w: usize,
        target_h: usize,
        target_w: usize,
        r_h: f32,
        r_w: f32,
        align_corners: bool,
    };

    var ctx = Context{
        .input = input,
        .out = &out,
        .batch = batch,
        .channels = channels,
        .in_h = in_h,
        .in_w = in_w,
        .target_h = target_h,
        .target_w = target_w,
        .r_h = r_h,
        .r_w = r_w,
        .align_corners = align_corners,
    };

    const total_rows = batch * channels * target_h;
    parallel.parallelFor(allocator, total_rows, &ctx, struct {
        fn worker(c: *Context, start: usize, end: usize) void {
            const ch_total = c.channels;
            const t_h = c.target_h;
            const t_w = c.target_w;
            const ih_max = c.in_h;
            const iw_max = c.in_w;
            const rh = c.r_h;
            const rw = c.r_w;

            for (start..end) |task| {
                const b = task / (ch_total * t_h);
                const rem = task % (ch_total * t_h);
                const ch = rem / t_h;
                const oh = rem % t_h;

                const in_y = if (c.align_corners)
                    rh * @as(f32, @floatFromInt(oh))
                else
                    rh * (@as(f32, @floatFromInt(oh)) + 0.5) - 0.5;

                const y_low: usize = @intCast(@max(0, @as(isize, @intFromFloat(@floor(in_y)))));
                const y_high: usize = @min(ih_max - 1, y_low + 1);
                const y_weight = @max(0.0, @min(1.0, in_y - @as(f32, @floatFromInt(y_low))));

                for (0..t_w) |ow| {
                    const in_x = if (c.align_corners)
                        rw * @as(f32, @floatFromInt(ow))
                    else
                        rw * (@as(f32, @floatFromInt(ow)) + 0.5) - 0.5;

                    const x_low: usize = @intCast(@max(0, @as(isize, @intFromFloat(@floor(in_x)))));
                    const x_high: usize = @min(iw_max - 1, x_low + 1);
                    const x_weight = @max(0.0, @min(1.0, in_x - @as(f32, @floatFromInt(x_low))));

                    const v00 = c.input.at4(b, ch, y_low, x_low);
                    const v01 = c.input.at4(b, ch, y_low, x_high);
                    const v10 = c.input.at4(b, ch, y_high, x_low);
                    const v11 = c.input.at4(b, ch, y_high, x_high);

                    const top = v00 + (v01 - v00) * x_weight;
                    const bot = v10 + (v11 - v10) * x_weight;
                    const val = top + (bot - top) * y_weight;

                    c.out.set4(b, ch, oh, ow, val);
                }
            }
        }
    }.worker);

    return out;
}

pub fn sinusoidalEmbedding2D(
    allocator: std.mem.Allocator,
    h: usize,
    w: usize,
    embed_dim: usize,
    temperature: f32,
    scale: f32,
) !Tensor {
    std.debug.assert(embed_dim % 4 == 0);
    const num_pos_feats = embed_dim / 2;

    const out_shape = [_]usize{ 1, h * w, embed_dim };
    var out = try Tensor.init(allocator, &out_shape);

    var dim_t = try allocator.alloc(f32, num_pos_feats / 2);
    defer allocator.free(dim_t);

    for (0..num_pos_feats / 2) |i| {
        const exponent = 2.0 * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(num_pos_feats));
        dim_t[i] = std.math.pow(f32, temperature, exponent);
    }

    var idx: usize = 0;
    for (0..h) |y| {
        const y_val = (@as(f32, @floatFromInt(y)) + 0.5) * scale;
        for (0..w) |x| {
            const x_val = (@as(f32, @floatFromInt(x)) + 0.5) * scale;

            for (0..num_pos_feats / 2) |i| {
                const dt = dim_t[i];
                const sin_x = @sin(x_val / dt);
                const cos_x = @cos(x_val / dt);
                const sin_y = @sin(y_val / dt);
                const cos_y = @cos(y_val / dt);

                out.data[idx * embed_dim + i * 2] = sin_x;
                out.data[idx * embed_dim + i * 2 + 1] = cos_x;
                out.data[idx * embed_dim + num_pos_feats + i * 2] = sin_y;
                out.data[idx * embed_dim + num_pos_feats + i * 2 + 1] = cos_y;
            }
            idx += 1;
        }
    }

    return out;
}

pub fn multiHeadAttention(
    allocator: std.mem.Allocator,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    num_heads: usize,
    attn_mask: ?Tensor,
) !Tensor {
    std.debug.assert(q.shape.len == 3);
    std.debug.assert(k.shape.len == 3);
    std.debug.assert(v.shape.len == 3);

    const batch = q.shape[0];
    const seq_q = q.shape[1];
    const d_model = q.shape[2];
    const seq_k = k.shape[1];

    std.debug.assert(d_model % num_heads == 0);
    const d_head = d_model / num_heads;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(d_head)));

    const out_shape = [_]usize{ batch, seq_q, d_model };
    var out = try Tensor.initZeros(allocator, &out_shape);

    for (0..batch) |b| {
        for (0..num_heads) |h| {
            const head_offset = h * d_head;

            for (0..seq_q) |qi| {
                const q_vec = q.data[b * seq_q * d_model + qi * d_model + head_offset ..][0..d_head];

                var scores_buf = try allocator.alloc(f32, seq_k);
                defer allocator.free(scores_buf);

                var max_score: f32 = -std.math.inf(f32);
                for (0..seq_k) |ki| {
                    const k_vec = k.data[b * seq_k * d_model + ki * d_model + head_offset ..][0..d_head];
                    var dot: f32 = 0.0;
                    for (0..d_head) |d| {
                        dot += q_vec[d] * k_vec[d];
                    }
                    dot *= scale;
                    if (attn_mask) |m| {
                        dot += m.at3(b, qi, ki);
                    }
                    scores_buf[ki] = dot;
                    if (dot > max_score) max_score = dot;
                }

                var sum_exp: f32 = 0.0;
                for (scores_buf) |*sc| {
                    sc.* = @exp(sc.* - max_score);
                    sum_exp += sc.*;
                }
                const inv_sum = 1.0 / (sum_exp + 1e-12);
                for (scores_buf) |*sc| {
                    sc.* *= inv_sum;
                }

                for (0..d_head) |d| {
                    var weighted_sum: f32 = 0.0;
                    for (0..seq_k) |ki| {
                        const v_val = v.data[b * seq_k * d_model + ki * d_model + head_offset + d];
                        weighted_sum += scores_buf[ki] * v_val;
                    }
                    out.data[b * seq_q * d_model + qi * d_model + head_offset + d] = weighted_sum;
                }
            }
        }
    }

    return out;
}

test "Conv2D and BilinearUpsample" {
    const allocator = std.testing.allocator;

    const in_shape = [_]usize{ 1, 1, 2, 2 };
    var input = try Tensor.init(allocator, &in_shape);
    defer input.deinit();
    input.data[0] = 1; input.data[1] = 2;
    input.data[2] = 3; input.data[3] = 4;

    var up = try bilinearUpsample(allocator, input, 4, 4, false);
    defer up.deinit();

    try std.testing.expectEqual(@as(usize, 4), up.shape[2]);
    try std.testing.expectEqual(@as(usize, 4), up.shape[3]);
}

test "ConvTranspose2D matches hand-computed overlap-add" {
    const allocator = std.testing.allocator;

    const shape = [_]usize{ 1, 1, 2, 2 };
    var input = try Tensor.init(allocator, &shape);
    defer input.deinit();
    input.data[0] = 1; input.data[1] = 2;
    input.data[2] = 3; input.data[3] = 4;

    var weight = try Tensor.init(allocator, &shape);
    defer weight.deinit();
    weight.data[0] = 1; weight.data[1] = 2;
    weight.data[2] = 3; weight.data[3] = 4;

    // stride 2: each input pixel stamps a disjoint copy of the kernel.
    var strided = try convTranspose2d(allocator, input, weight, null, 2, 0);
    defer strided.deinit();
    const expect_stride2 = [_]f32{
        1, 2, 2,  4,
        3, 4, 6,  8,
        3, 6, 4,  8,
        9, 12, 12, 16,
    };
    try std.testing.expectEqual(@as(usize, 4), strided.shape[2]);
    for (expect_stride2, strided.data) |expected, actual| {
        try std.testing.expectApproxEqAbs(expected, actual, 1e-5);
    }

    // stride 1: neighbouring stamps overlap and accumulate.
    var overlapped = try convTranspose2d(allocator, input, weight, null, 1, 0);
    defer overlapped.deinit();
    const expect_stride1 = [_]f32{
        1, 4,  4,
        6, 20, 16,
        9, 24, 16,
    };
    try std.testing.expectEqual(@as(usize, 3), overlapped.shape[2]);
    for (expect_stride1, overlapped.data) |expected, actual| {
        try std.testing.expectApproxEqAbs(expected, actual, 1e-5);
    }
}

test "MaxPool2D halves a feature map" {
    const allocator = std.testing.allocator;

    const shape = [_]usize{ 1, 1, 4, 4 };
    var input = try Tensor.init(allocator, &shape);
    defer input.deinit();
    for (input.data, 0..) |*v, i| v.* = @floatFromInt(i);

    var pooled = try maxPool2d(allocator, input, 2, 2);
    defer pooled.deinit();

    try std.testing.expectEqual(@as(usize, 2), pooled.shape[2]);
    try std.testing.expectEqual(@as(usize, 2), pooled.shape[3]);
    const expected = [_]f32{ 5, 7, 13, 15 };
    for (expected, pooled.data) |e, a| {
        try std.testing.expectEqual(e, a);
    }
}

/// Textbook definition, used to pin the blocked/GEMM paths above.
fn referenceConv2d(
    allocator: std.mem.Allocator,
    input: Tensor,
    weight: Tensor,
    bias: ?Tensor,
    stride: usize,
    padding: usize,
) !Tensor {
    const batch = input.shape[0];
    const c_in = input.shape[1];
    const h_in = input.shape[2];
    const w_in = input.shape[3];
    const c_out = weight.shape[0];
    const k_h = weight.shape[2];
    const k_w = weight.shape[3];
    const h_out = (h_in + 2 * padding - k_h) / stride + 1;
    const w_out = (w_in + 2 * padding - k_w) / stride + 1;

    const shape = [_]usize{ batch, c_out, h_out, w_out };
    var out = try Tensor.init(allocator, &shape);
    errdefer out.deinit();

    for (0..batch) |b| {
        for (0..c_out) |co| {
            for (0..h_out) |oh| {
                for (0..w_out) |ow| {
                    var sum: f32 = if (bias) |bi| bi.data[co] else 0.0;
                    for (0..c_in) |ci| {
                        for (0..k_h) |kh| {
                            const ih = @as(isize, @intCast(oh * stride + kh)) - @as(isize, @intCast(padding));
                            if (ih < 0 or ih >= h_in) continue;
                            for (0..k_w) |kw| {
                                const iw = @as(isize, @intCast(ow * stride + kw)) - @as(isize, @intCast(padding));
                                if (iw < 0 or iw >= w_in) continue;
                                sum += input.at4(b, ci, @intCast(ih), @intCast(iw)) * weight.at4(co, ci, kh, kw);
                            }
                        }
                    }
                    out.set4(b, co, oh, ow, sum);
                }
            }
        }
    }
    return out;
}

fn referenceConvTranspose2d(
    allocator: std.mem.Allocator,
    input: Tensor,
    weight: Tensor,
    bias: ?Tensor,
    stride: usize,
    padding: usize,
) !Tensor {
    const batch = input.shape[0];
    const c_in = input.shape[1];
    const h_in = input.shape[2];
    const w_in = input.shape[3];
    const c_out = weight.shape[1];
    const k_h = weight.shape[2];
    const k_w = weight.shape[3];
    const h_out = (h_in - 1) * stride - 2 * padding + k_h;
    const w_out = (w_in - 1) * stride - 2 * padding + k_w;

    const shape = [_]usize{ batch, c_out, h_out, w_out };
    var out = try Tensor.initZeros(allocator, &shape);
    errdefer out.deinit();

    for (0..batch) |b| {
        for (0..c_out) |co| {
            for (0..h_out * w_out) |i| {
                out.data[(b * c_out + co) * h_out * w_out + i] = if (bias) |bi| bi.data[co] else 0.0;
            }
        }
    }

    for (0..batch) |b| {
        for (0..c_in) |ci| {
            for (0..c_out) |co| {
                for (0..h_in) |ih| {
                    for (0..w_in) |iw| {
                        const v = input.at4(b, ci, ih, iw);
                        for (0..k_h) |kh| {
                            const oh = @as(isize, @intCast(ih * stride + kh)) - @as(isize, @intCast(padding));
                            if (oh < 0 or oh >= h_out) continue;
                            for (0..k_w) |kw| {
                                const ow = @as(isize, @intCast(iw * stride + kw)) - @as(isize, @intCast(padding));
                                if (ow < 0 or ow >= w_out) continue;
                                const idx = ((b * c_out + co) * h_out + @as(usize, @intCast(oh))) * w_out + @as(usize, @intCast(ow));
                                out.data[idx] += v * weight.at4(ci, co, kh, kw);
                            }
                        }
                    }
                }
            }
        }
    }
    return out;
}

test "conv2d fast paths match the reference definition" {
    const allocator = std.testing.allocator;

    // {c_in, c_out, h, w, kernel, stride, padding}: pointwise, padded 3x3, a
    // patch-embedding-shaped strided case, and a non-square tile remainder.
    const cases = [_][7]usize{
        .{ 5, 7, 9, 11, 1, 1, 0 },
        .{ 4, 6, 10, 13, 3, 1, 1 },
        .{ 3, 8, 12, 12, 3, 1, 0 },
        .{ 3, 5, 14, 14, 7, 7, 0 },
        .{ 2, 4, 9, 9, 2, 2, 0 },
        .{ 6, 3, 7, 40, 3, 1, 1 },
    };

    for (cases) |case| {
        const c_in, const c_out, const h, const w, const k, const stride, const pad = case;

        const in_shape = [_]usize{ 1, c_in, h, w };
        var input = try Tensor.initRandom(allocator, &in_shape, 11, -1.0, 1.0);
        defer input.deinit();

        const w_shape = [_]usize{ c_out, c_in, k, k };
        var weight = try Tensor.initRandom(allocator, &w_shape, 23, -0.5, 0.5);
        defer weight.deinit();

        const b_shape = [_]usize{c_out};
        var bias = try Tensor.initRandom(allocator, &b_shape, 37, -0.2, 0.2);
        defer bias.deinit();

        var got = try conv2d(allocator, input, weight, bias, stride, pad);
        defer got.deinit();
        var want = try referenceConv2d(allocator, input, weight, bias, stride, pad);
        defer want.deinit();

        try std.testing.expectEqualSlices(usize, want.shape, got.shape);
        for (want.data, got.data) |e, g| try std.testing.expectApproxEqAbs(e, g, 1e-4);
    }
}

test "convTranspose2d fast path matches the reference definition" {
    const allocator = std.testing.allocator;

    // {c_in, c_out, h, w, kernel, stride, padding}: the FPN's kernel == stride
    // shape, plus overlapping and padded cases that take the general path.
    const cases = [_][7]usize{
        .{ 6, 4, 5, 7, 2, 2, 0 },
        .{ 3, 5, 4, 4, 3, 3, 0 },
        .{ 4, 3, 5, 5, 3, 1, 0 },
        .{ 2, 6, 6, 4, 4, 2, 1 },
    };

    for (cases) |case| {
        const c_in, const c_out, const h, const w, const k, const stride, const pad = case;

        const in_shape = [_]usize{ 1, c_in, h, w };
        var input = try Tensor.initRandom(allocator, &in_shape, 5, -1.0, 1.0);
        defer input.deinit();

        const w_shape = [_]usize{ c_in, c_out, k, k };
        var weight = try Tensor.initRandom(allocator, &w_shape, 17, -0.5, 0.5);
        defer weight.deinit();

        const b_shape = [_]usize{c_out};
        var bias = try Tensor.initRandom(allocator, &b_shape, 29, -0.2, 0.2);
        defer bias.deinit();

        var got = try convTranspose2d(allocator, input, weight, bias, stride, pad);
        defer got.deinit();
        var want = try referenceConvTranspose2d(allocator, input, weight, bias, stride, pad);
        defer want.deinit();

        try std.testing.expectEqualSlices(usize, want.shape, got.shape);
        for (want.data, got.data) |e, g| try std.testing.expectApproxEqAbs(e, g, 1e-4);
    }
}
