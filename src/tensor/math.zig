const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const parallel = @import("parallel.zig");

pub const VEC_SIZE = 8;
pub const F32Vec = @Vector(VEC_SIZE, f32);

pub inline fn dotProduct(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    const n = a.len;
    var sum: f32 = 0.0;
    var i: usize = 0;

    var vec_sum: F32Vec = @splat(0.0);
    while (i + VEC_SIZE <= n) : (i += VEC_SIZE) {
        const va: F32Vec = a[i..][0..VEC_SIZE].*;
        const vb: F32Vec = b[i..][0..VEC_SIZE].*;
        vec_sum += va * vb;
    }
    sum += @reduce(.Add, vec_sum);

    while (i < n) : (i += 1) {
        sum += a[i] * b[i];
    }
    return sum;
}

pub inline fn sigmoid(x: f32) f32 {
    if (x > 15.0) return 1.0;
    if (x < -15.0) return 0.0;
    return 1.0 / (1.0 + @exp(-x));
}

pub inline fn relu(x: f32) f32 {
    return if (x > 0.0) x else 0.0;
}

pub inline fn silu(x: f32) f32 {
    return x * sigmoid(x);
}

pub inline fn gelu(x: f32) f32 {
    const k: f32 = 0.7978845608; // sqrt(2/pi)
    const c: f32 = 0.044715;
    const inner = k * (x + c * x * x * x);
    const tanh_val = std.math.tanh(inner);
    return 0.5 * x * (1.0 + tanh_val);
}

pub inline fn quickGelu(x: f32) f32 {
    return x * sigmoid(1.702 * x);
}

/// Gauss error function, Abramowitz & Stegun 7.1.26 (max abs error 1.5e-7).
pub inline fn erf(x: f32) f32 {
    const sign: f32 = if (x < 0.0) -1.0 else 1.0;
    const ax = @abs(x);
    const t = 1.0 / (1.0 + 0.3275911 * ax);
    const y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * @exp(-ax * ax);
    return sign * y;
}

/// Exact GELU (`nn.GELU()` / HF `"gelu"`), as opposed to the tanh approximation.
pub inline fn geluExact(x: f32) f32 {
    return 0.5 * x * (1.0 + erf(x * std.math.sqrt1_2));
}

pub fn applyActivation(t: *Tensor, act_type: enum { gelu, gelu_exact, quick_gelu, silu, relu, sigmoid }) void {
    for (t.data) |*v| {
        v.* = switch (act_type) {
            .gelu => gelu(v.*),
            .gelu_exact => geluExact(v.*),
            .quick_gelu => quickGelu(v.*),
            .silu => silu(v.*),
            .relu => relu(v.*),
            .sigmoid => sigmoid(v.*),
        };
    }
}

pub fn softmaxLastDim(t: *Tensor) void {
    if (t.shape.len == 0) return;
    const last_dim = t.shape[t.shape.len - 1];
    const outer_count = t.numElements() / last_dim;

    for (0..outer_count) |i| {
        const slice = t.data[i * last_dim .. (i + 1) * last_dim];
        var max_val: f32 = -std.math.inf(f32);
        for (slice) |v| {
            if (v > max_val) max_val = v;
        }

        var sum_exp: f32 = 0.0;
        for (slice) |*v| {
            v.* = @exp(v.* - max_val);
            sum_exp += v.*;
        }

        const inv_sum = 1.0 / (sum_exp + 1e-12);
        for (slice) |*v| {
            v.* *= inv_sum;
        }
    }
}

pub fn layerNorm(allocator: std.mem.Allocator, input: Tensor, gamma: ?Tensor, beta: ?Tensor, eps: f32) !Tensor {
    const rank = input.shape.len;
    std.debug.assert(rank >= 1);
    const d = input.shape[rank - 1];
    const outer_count = input.numElements() / d;

    const out = try Tensor.init(allocator, input.shape);

    const Context = struct {
        in_data: []const f32,
        out_data: []f32,
        d: usize,
        gamma: ?Tensor,
        beta: ?Tensor,
        eps: f32,
    };

    const ctx = Context{
        .in_data = input.data,
        .out_data = out.data,
        .d = d,
        .gamma = gamma,
        .beta = beta,
        .eps = eps,
    };

    parallel.parallelFor(allocator, outer_count, ctx, struct {
        fn worker(c: Context, start: usize, end: usize) void {
            const dim = c.d;
            for (start..end) |i| {
                const in_slice = c.in_data[i * dim .. (i + 1) * dim];
                const out_slice = c.out_data[i * dim .. (i + 1) * dim];

                var sum: f32 = 0.0;
                for (in_slice) |v| sum += v;
                const mean = sum / @as(f32, @floatFromInt(dim));

                var var_sum: f32 = 0.0;
                for (in_slice) |v| {
                    const diff = v - mean;
                    var_sum += diff * diff;
                }
                const variance = var_sum / @as(f32, @floatFromInt(dim));
                const inv_std = 1.0 / @sqrt(variance + c.eps);

                for (0..dim) |j| {
                    const normalized = (in_slice[j] - mean) * inv_std;
                    const g = if (c.gamma) |gm| gm.data[j] else 1.0;
                    const b = if (c.beta) |bt| bt.data[j] else 0.0;
                    out_slice[j] = normalized * g + b;
                }
            }
        }
    }.worker);

    return out;
}

pub fn rmsNorm(allocator: std.mem.Allocator, input: Tensor, weight: ?Tensor, eps: f32) !Tensor {
    const rank = input.shape.len;
    std.debug.assert(rank >= 1);
    const d = input.shape[rank - 1];
    const outer_count = input.numElements() / d;

    var out = try Tensor.init(allocator, input.shape);

    for (0..outer_count) |i| {
        const in_slice = input.data[i * d .. (i + 1) * d];
        const out_slice = out.data[i * d .. (i + 1) * d];

        var sum_sq: f32 = 0.0;
        for (in_slice) |v| sum_sq += v * v;
        const rms = @sqrt(sum_sq / @as(f32, @floatFromInt(d)) + eps);
        const inv_rms = 1.0 / rms;

        for (0..d) |j| {
            const w = if (weight) |wt| wt.data[j] else 1.0;
            out_slice[j] = (in_slice[j] * inv_rms) * w;
        }
    }

    return out;
}

pub fn matmul2D(allocator: std.mem.Allocator, a: Tensor, b: Tensor) !Tensor {
    std.debug.assert(a.shape.len == 2);
    std.debug.assert(b.shape.len == 2);
    const m = a.shape[0];
    const k = a.shape[1];
    std.debug.assert(b.shape[0] == k);
    const n = b.shape[1];

    const shape = [_]usize{ m, n };
    const c = try Tensor.initZeros(allocator, &shape);

    // Transpose B into b_t (n x k) for cache-friendly contiguous SIMD dot products
    const b_t_shape = [_]usize{ n, k };
    var b_t = try Tensor.init(allocator, &b_t_shape);
    defer b_t.deinit();

    for (0..k) |row| {
        for (0..n) |col| {
            b_t.data[col * k + row] = b.data[row * n + col];
        }
    }

    const Context = struct {
        a_data: []const f32,
        b_t_data: []const f32,
        c_data: []f32,
        m: usize,
        k: usize,
        n: usize,
    };

    const ctx = Context{
        .a_data = a.data,
        .b_t_data = b_t.data,
        .c_data = c.data,
        .m = m,
        .k = k,
        .n = n,
    };

    parallel.parallelFor(allocator, m, ctx, struct {
        fn worker(cx: Context, start: usize, end: usize) void {
            const kk = cx.k;
            const nn = cx.n;
            for (start..end) |i| {
                const a_row = cx.a_data[i * kk .. (i + 1) * kk];
                const c_row = cx.c_data[i * nn .. (i + 1) * nn];
                for (0..nn) |j| {
                    const b_col = cx.b_t_data[j * kk .. (j + 1) * kk];
                    c_row[j] = dotProduct(a_row, b_col);
                }
            }
        }
    }.worker);

    return c;
}

pub fn linear(allocator: std.mem.Allocator, input: Tensor, weight: Tensor, bias: ?Tensor) !Tensor {
    std.debug.assert(weight.shape.len == 2);
    const out_features = weight.shape[0];
    const in_features = weight.shape[1];

    const in_rank = input.shape.len;
    std.debug.assert(in_rank >= 1);
    std.debug.assert(input.shape[in_rank - 1] == in_features);

    var out_shape = try allocator.alloc(usize, in_rank);
    defer allocator.free(out_shape);
    @memcpy(out_shape, input.shape);
    out_shape[in_rank - 1] = out_features;

    const out = try Tensor.init(allocator, out_shape);

    const outer_count = input.numElements() / in_features;

    const Context = struct {
        in_data: []const f32,
        w_data: []const f32,
        bias: ?Tensor,
        out_data: []f32,
        in_f: usize,
        out_f: usize,
    };

    const ctx = Context{
        .in_data = input.data,
        .w_data = weight.data,
        .bias = bias,
        .out_data = out.data,
        .in_f = in_features,
        .out_f = out_features,
    };

    parallel.parallelFor(allocator, outer_count, ctx, struct {
        fn worker(c: Context, start: usize, end: usize) void {
            const in_f = c.in_f;
            const out_f = c.out_f;
            for (start..end) |i| {
                const in_vec = c.in_data[i * in_f .. (i + 1) * in_f];
                const out_vec = c.out_data[i * out_f .. (i + 1) * out_f];

                for (0..out_f) |j| {
                    const w_row = c.w_data[j * in_f .. (j + 1) * in_f];
                    var val = dotProduct(in_vec, w_row);
                    if (c.bias) |b| {
                        val += b.data[j];
                    }
                    out_vec[j] = val;
                }
            }
        }
    }.worker);

    return out;
}

pub fn batchedMatmul(allocator: std.mem.Allocator, a: Tensor, b: Tensor) !Tensor {
    std.debug.assert(a.shape.len == 3);
    std.debug.assert(b.shape.len == 3);
    const batch = a.shape[0];
    std.debug.assert(b.shape[0] == batch);
    const m = a.shape[1];
    const k = a.shape[2];
    std.debug.assert(b.shape[1] == k);
    const n = b.shape[2];

    const out_shape = [_]usize{ batch, m, n };
    var c = try Tensor.initZeros(allocator, &out_shape);

    const b_t_shape = [_]usize{ n, k };
    var b_t = try Tensor.init(allocator, &b_t_shape);
    defer b_t.deinit();

    for (0..batch) |b_idx| {
        const a_batch = a.data[b_idx * m * k .. (b_idx + 1) * m * k];
        const b_batch = b.data[b_idx * k * n .. (b_idx + 1) * k * n];
        const c_batch = c.data[b_idx * m * n .. (b_idx + 1) * m * n];

        for (0..k) |row| {
            for (0..n) |col| {
                b_t.data[col * k + row] = b_batch[row * n + col];
            }
        }

        for (0..m) |i| {
            const a_row = a_batch[i * k .. (i + 1) * k];
            const c_row = c_batch[i * n .. (i + 1) * n];
            for (0..n) |j| {
                const b_col = b_t.data[j * k .. (j + 1) * k];
                c_row[j] = dotProduct(a_row, b_col);
            }
        }
    }

    return c;
}

test "SIMD dot product and activations" {
    const a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const b = [_]f32{ 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 };
    const dot = dotProduct(&a, &b);
    try std.testing.expectEqual(@as(f32, 110.0), dot);

    try std.testing.expect(sigmoid(0.0) == 0.5);
    try std.testing.expect(silu(0.0) == 0.0);
    try std.testing.expect(gelu(0.0) == 0.0);
}

test "Matmul and Linear" {
    const allocator = std.testing.allocator;
    const a_shape = [_]usize{ 2, 3 };
    var a = try Tensor.init(allocator, &a_shape);
    defer a.deinit();
    a.data[0] = 1; a.data[1] = 2; a.data[2] = 3;
    a.data[3] = 4; a.data[4] = 5; a.data[5] = 6;

    const b_shape = [_]usize{ 3, 2 };
    var b = try Tensor.init(allocator, &b_shape);
    defer b.deinit();
    b.data[0] = 1; b.data[1] = 2;
    b.data[2] = 3; b.data[3] = 4;
    b.data[4] = 5; b.data[5] = 6;

    var c = try matmul2D(allocator, a, b);
    defer c.deinit();

    try std.testing.expectEqual(@as(f32, 22.0), c.at2(0, 0));
    try std.testing.expectEqual(@as(f32, 28.0), c.at2(0, 1));
    try std.testing.expectEqual(@as(f32, 49.0), c.at2(1, 0));
    try std.testing.expectEqual(@as(f32, 64.0), c.at2(1, 1));
}

test "LayerNorm" {
    const allocator = std.testing.allocator;
    const shape = [_]usize{ 1, 4 };
    var input = try Tensor.init(allocator, &shape);
    defer input.deinit();
    input.data[0] = 2.0;
    input.data[1] = 4.0;
    input.data[2] = 4.0;
    input.data[3] = 2.0;

    var normed = try layerNorm(allocator, input, null, null, 1e-5);
    defer normed.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, -1.0), normed.at2(0, 0), 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), normed.at2(0, 1), 1e-3);
}

test "Exact GELU matches reference values" {
    // Reference: torch.nn.functional.gelu (erf formulation).
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), geluExact(0.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8413447), geluExact(1.0), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -0.1586553), geluExact(-1.0), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.9544997), geluExact(2.0), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -0.0040495), geluExact(-3.0), 1e-5);
}
