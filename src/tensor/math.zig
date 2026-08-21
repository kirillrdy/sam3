const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const parallel = @import("parallel.zig");
const gemm = @import("gemm.zig");

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

/// Vectorised `@exp` for f32 lanes.
///
/// `@exp` lowers to a scalar libm call, and softmax over a 5184-token attention
/// map makes tens of millions of them. This is the usual range reduction
/// (`x = n*ln2 + r`) with a degree-6 minimax polynomial on the reduced argument
/// and the power of two folded in through the exponent field - accurate to
/// about one ulp, so a normalised softmax is unchanged to well beyond f32
/// display precision.
pub inline fn expVec(x: F32Vec) F32Vec {
    const log2e: F32Vec = @splat(1.44269504088896341);
    const ln2_hi: F32Vec = @splat(0.693145751953125);
    const ln2_lo: F32Vec = @splat(1.42860682030941723e-6);

    // Clamped either side of the f32 range so the exponent assembly cannot
    // overflow; softmax only ever passes non-positive arguments anyway.
    const clamped = @min(@max(x, @as(F32Vec, @splat(-87.0))), @as(F32Vec, @splat(88.0)));

    const n = @round(clamped * log2e);
    const r = clamped - n * ln2_hi - n * ln2_lo;

    var p: F32Vec = @splat(1.9875691500e-4);
    p = p * r + @as(F32Vec, @splat(1.3981999507e-3));
    p = p * r + @as(F32Vec, @splat(8.3334519073e-3));
    p = p * r + @as(F32Vec, @splat(4.1665795894e-2));
    p = p * r + @as(F32Vec, @splat(1.6666665459e-1));
    p = p * r + @as(F32Vec, @splat(5.0000001201e-1));
    p = p * r * r + r + @as(F32Vec, @splat(1.0));

    const exponent: @Vector(VEC_SIZE, i32) = @intFromFloat(n);
    const bits = (exponent + @as(@Vector(VEC_SIZE, i32), @splat(127))) <<
        @as(@Vector(VEC_SIZE, u5), @splat(23));
    return p * @as(F32Vec, @bitCast(bits));
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

/// Vectorised `erf`, same Abramowitz & Stegun 7.1.26 rational form as `erf`.
pub inline fn erfVec(x: F32Vec) F32Vec {
    const one: F32Vec = @splat(1.0);
    const zero: F32Vec = @splat(0.0);
    const sign = @select(f32, x < zero, @as(F32Vec, @splat(-1.0)), one);
    const ax = @abs(x);

    const t = one / (one + @as(F32Vec, @splat(0.3275911)) * ax);
    var poly = @as(F32Vec, @splat(1.061405429)) * t - @as(F32Vec, @splat(1.453152027));
    poly = poly * t + @as(F32Vec, @splat(1.421413741));
    poly = poly * t - @as(F32Vec, @splat(0.284496736));
    poly = poly * t + @as(F32Vec, @splat(0.254829592));
    const y = one - poly * t * expVec(-ax * ax);
    return sign * y;
}

pub inline fn geluExactVec(x: F32Vec) F32Vec {
    const half: F32Vec = @splat(0.5);
    const one: F32Vec = @splat(1.0);
    return half * x * (one + erfVec(x * @as(F32Vec, @splat(std.math.sqrt1_2))));
}

pub const Activation = enum { gelu, gelu_exact, quick_gelu, silu, relu, sigmoid };

/// The MLP activation runs over `tokens * 4736` elements per layer, so it is
/// both parallelised and, for the exact GELU the backbone actually uses,
/// vectorised - the scalar form calls libm's `expf` once per element.
pub fn applyActivation(t: *Tensor, act_type: Activation) void {
    const Context = struct {
        data: []f32,
        kind: Activation,
    };
    const ctx = Context{ .data = t.data, .kind = act_type };

    parallel.parallelFor(std.heap.page_allocator, t.data.len, ctx, struct {
        fn worker(c: Context, start: usize, end: usize) void {
            const slice = c.data[start..end];

            if (c.kind == .gelu_exact) {
                var i: usize = 0;
                while (i + VEC_SIZE <= slice.len) : (i += VEC_SIZE) {
                    const v: F32Vec = slice[i..][0..VEC_SIZE].*;
                    slice[i..][0..VEC_SIZE].* = geluExactVec(v);
                }
                while (i < slice.len) : (i += 1) slice[i] = geluExact(slice[i]);
                return;
            }

            for (slice) |*v| {
                v.* = switch (c.kind) {
                    .gelu => gelu(v.*),
                    .gelu_exact => unreachable,
                    .quick_gelu => quickGelu(v.*),
                    .silu => silu(v.*),
                    .relu => relu(v.*),
                    .sigmoid => sigmoid(v.*),
                };
            }
        }
    }.worker);
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
    var c = try Tensor.init(allocator, &shape);
    errdefer c.deinit();

    try gemm.gemm(allocator, m, n, k, a.data, k, b.data, n, .row_major, null, c.data);
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

    var out = try Tensor.init(allocator, out_shape);
    errdefer out.deinit();

    const outer_count = input.numElements() / in_features;

    // `nn.Linear` stores its weight as [out_features, in_features], i.e. B
    // already transposed.
    try gemm.gemm(
        allocator,
        outer_count,
        out_features,
        in_features,
        input.data,
        in_features,
        weight.data,
        in_features,
        .transposed,
        if (bias) |b| b.data else null,
        out.data,
    );

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
    var c = try Tensor.init(allocator, &out_shape);
    errdefer c.deinit();

    for (0..batch) |b_idx| {
        try gemm.gemm(
            allocator,
            m,
            n,
            k,
            a.data[b_idx * m * k ..][0 .. m * k],
            k,
            b.data[b_idx * k * n ..][0 .. k * n],
            n,
            .row_major,
            null,
            c.data[b_idx * m * n ..][0 .. m * n],
        );
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

test "expVec matches scalar @exp across the softmax range" {
    // Softmax feeds it max-shifted values, so the interesting range is [-90, 0],
    // but check a little either side of that too.
    const probes = [_]f32{ -90.0, -40.0, -10.0, -1.0, -0.5, -0.001, 0.0, 0.5, 1.0, 5.0, 20.0, 80.0 };

    var i: usize = 0;
    while (i < probes.len) : (i += VEC_SIZE) {
        var lane: [VEC_SIZE]f32 = @splat(0.0);
        const take = @min(VEC_SIZE, probes.len - i);
        @memcpy(lane[0..take], probes[i..][0..take]);

        const got: [VEC_SIZE]f32 = expVec(@as(F32Vec, lane));
        for (0..take) |j| {
            const want = @exp(lane[j]);
            const tol = @max(@abs(want) * 1e-6, 1e-30);
            try std.testing.expectApproxEqAbs(want, got[j], tol);
        }
    }
}
