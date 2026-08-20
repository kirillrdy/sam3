const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const math = @import("math.zig");
const parallel = @import("parallel.zig");

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
    var out = try Tensor.initZeros(allocator, &out_shape);

    const Context = struct {
        input: Tensor,
        weight: Tensor,
        bias: ?Tensor,
        out: *Tensor,
        batch: usize,
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
        .batch = batch,
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

    const total_tasks = batch * c_out;
    parallel.parallelFor(allocator, total_tasks, &ctx, struct {
        fn worker(c: *Context, start: usize, end: usize) void {
            const c_o_total = c.c_out;
            const h_o = c.h_out;
            const w_o = c.w_out;
            const c_i_total = c.c_in;
            const h_i = c.h_in;
            const w_i = c.w_in;
            const kh_max = c.k_h;
            const kw_max = c.k_w;
            const st = c.stride;
            const pad = c.padding;

            for (start..end) |task| {
                const b = task / c_o_total;
                const co = task % c_o_total;
                const b_val = if (c.bias) |bi| bi.data[co] else 0.0;

                for (0..h_o) |oh| {
                    for (0..w_o) |ow| {
                        var sum: f32 = b_val;
                        const ih_start = @as(isize, @intCast(oh * st)) - @as(isize, @intCast(pad));
                        const iw_start = @as(isize, @intCast(ow * st)) - @as(isize, @intCast(pad));

                        for (0..c_i_total) |ci| {
                            for (0..kh_max) |kh| {
                                const ih = ih_start + @as(isize, @intCast(kh));
                                if (ih < 0 or ih >= h_i) continue;

                                for (0..kw_max) |kw| {
                                    const iw = iw_start + @as(isize, @intCast(kw));
                                    if (iw < 0 or iw >= w_i) continue;

                                    const in_val = c.input.at4(b, ci, @intCast(ih), @intCast(iw));
                                    const w_val = c.weight.at4(co, ci, kh, kw);
                                    sum += in_val * w_val;
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
