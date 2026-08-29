//! Every operator that runs on the GPU. Compiled for `nvptx64-cuda` and
//! embedded in the binary as PTX; the host side in `ops.zig` launches these.
//!
//! Shapes and strides arrive in a small `meta` buffer rather than as
//! parameters, so one kernel covers a tensor of any rank.

const gpu = @import("gpu");

pub const panic = gpu.panic;

pub const max_rank = 8;

const inf: f32 = @bitCast(@as(u32, 0x7f800000));
const neg_inf: f32 = -inf;

/// Elementwise binary operations, selected by `op`.
pub const Binary = enum(u32) { add, sub, mul, div, pow, min, max, equal, less, greater };

/// Elementwise unary operations, selected by `op`.
pub const Unary = enum(u32) { neg, erf, exp, sqrt, reciprocal, sigmoid, tanh, relu, abs, floor, sin, cos, log, sign, is_nan, gelu };

const Meta = [*]addrspace(.global) const u32;

/// Walks `index` through `rank` dimensions of the output, accumulating the
/// offset it maps to in an input with the given strides. A stride of zero is
/// how a broadcast dimension is spelled.
fn offsetOf(index: u32, meta: Meta, dims_at: u32, strides_at: u32, rank: u32) u32 {
    var remaining = index;
    var offset: u32 = 0;
    var axis = rank;
    while (axis > 0) {
        axis -= 1;
        const dim = meta[dims_at + axis];
        const coordinate = remaining % dim;
        remaining /= dim;
        offset +%= coordinate *% meta[strides_at + axis];
    }
    return offset;
}

/// Where an operand is the tail of the output shape -- a bias over the last
/// axis, a window shared by every batch, a single value -- the same block of
/// `period` values repeats over everything in front of it, and one wrap is the
/// whole index calculation. `period` is uniform across the block, so the power
/// of two test costs nothing and the common cases avoid the division.
fn wrap(index: u32, period: u32) u32 {
    return if (period & (period -% 1) == 0) index & (period -% 1) else index % period;
}

/// meta: [0..rank) output dims, [rank..2*rank) a strides, [2*rank..3*rank) b strides.
///
/// A period of zero is an operand whose broadcast does not have that shape, and
/// is walked stride by stride through `meta` instead. Walking costs a division
/// and a modulo per axis, and almost no arithmetic in a graph needs it.
pub fn binary(
    a: [*]addrspace(.global) const f32,
    b: [*]addrspace(.global) const f32,
    out: [*]addrspace(.global) f32,
    meta: Meta,
    rank: u32,
    count: u32,
    op: u32,
    a_period: u32,
    b_period: u32,
) callconv(.kernel) void {
    const i = gpu.globalIndex();
    if (i >= count) return;

    const x = if (a_period != 0) a[wrap(i, a_period)] else a[offsetOf(i, meta, 0, rank, rank)];
    const y = if (b_period != 0) b[wrap(i, b_period)] else b[offsetOf(i, meta, 0, 2 * rank, rank)];

    out[i] = switch (@as(Binary, @enumFromInt(op))) {
        .add => x + y,
        .sub => x - y,
        .mul => x * y,
        .div => x / y,
        .pow => power(x, y),
        .min => @min(x, y),
        .max => @max(x, y),
        .equal => if (x == y) 1 else 0,
        .less => if (x < y) 1 else 0,
        .greater => if (x > y) 1 else 0,
    };
}

pub fn unary(
    x: [*]addrspace(.global) const f32,
    out: [*]addrspace(.global) f32,
    count: u32,
    op: u32,
) callconv(.kernel) void {
    const i = gpu.globalIndex();
    if (i >= count) return;

    const v = x[i];
    out[i] = switch (@as(Unary, @enumFromInt(op))) {
        .neg => -v,
        .erf => erf(v),
        .exp => exp(v),
        .sqrt => @sqrt(v),
        .reciprocal => 1.0 / v,
        .sigmoid => 1.0 / (1.0 + exp(-v)),
        .tanh => tanh(v),
        .relu => @max(v, 0),
        .abs => @abs(v),
        .floor => @floor(v),
        .sin => sin(v),
        .cos => cos(v),
        .log => log(v),
        .sign => if (v > 0) 1 else if (v < 0) -1 else 0,
        // Comparing a value against itself is the only NaN test that survives
        // a compiler entitled to assume floats are never NaN.
        .is_nan => if (v != v) 1 else 0,
        // 1 / sqrt(2), which is what the graph divides by before the erf.
        .gelu => 0.5 * v * (1.0 + erf(v * 0.7071067811865476)),
    };
}

/// Abramowitz and Stegun 7.1.26, the approximation PyTorch's GELU is built on.
/// Accurate to about 1.5e-7, which is under a float's last bit for |x| < 1.
fn erf(x: f32) f32 {
    const sign: f32 = if (x < 0) -1 else 1;
    const v = @abs(x);
    const t = 1.0 / (1.0 + 0.3275911 * v);
    const poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    return sign * (1.0 - poly * exp(-v * v));
}

fn tanh(x: f32) f32 {
    const e = exp(2 * x);
    return (e - 1) / (e + 1);
}

fn power(x: f32, y: f32) f32 {
    if (y == 2.0) return x * x;
    if (y == 0.5) return @sqrt(x);
    return exp(y * log(x));
}

/// LLVM lowers the language builtins to libcalls for NVPTX, where there is no
/// device libc to satisfy them. PTX provides native approximate instructions;
/// their error is small relative to the model's FP32 accumulation error.
fn exp(x: f32) f32 {
    return asm ("ex2.approx.f32 %[out], %[in];"
        : [out] "=f" (-> f32),
        : [in] "f" (x * 1.4426950408889634),
    );
}

fn log(x: f32) f32 {
    const base2 = asm ("lg2.approx.f32 %[out], %[in];"
        : [out] "=f" (-> f32),
        : [in] "f" (x),
    );
    return base2 * 0.6931471805599453;
}

/// `sin.approx.f32` is only accurate near the origin, and SAM 3's positional
/// encoding takes the sine of a Fourier projection scaled by 2*pi, which is
/// not. Cody and Waite's reduction brings the argument back into one period
/// first: 2*pi is split into a head that is exact in f32 and a tail, so the
/// subtraction does not lose the low bits the way `x - k * 6.2831855` would.
const two_pi_head = 6.28125;
const two_pi_tail = 0.001935307179586477;

fn reduceTurns(x: f32) f32 {
    const turns = @round(x * 0.15915494309189535);
    return (x - turns * two_pi_head) - turns * two_pi_tail;
}

fn sin(x: f32) f32 {
    return asm ("sin.approx.f32 %[out], %[in];"
        : [out] "=f" (-> f32),
        : [in] "f" (reduceTurns(x)),
    );
}

fn cos(x: f32) f32 {
    return asm ("cos.approx.f32 %[out], %[in];"
        : [out] "=f" (-> f32),
        : [in] "f" (reduceTurns(x)),
    );
}

/// Copies a strided window of `src` into a dense `dst`. Slice, Concat, Split,
/// Pad and plain copies all reduce to this.
/// meta: [0..rank) dst dims, [rank..2*rank) src strides.
pub fn copy(
    src: [*]addrspace(.global) const f32,
    dst: [*]addrspace(.global) f32,
    meta: Meta,
    rank: u32,
    count: u32,
    src_offset: u32,
    dst_offset: u32,
) callconv(.kernel) void {
    const i = gpu.globalIndex();
    if (i >= count) return;
    dst[dst_offset + i] = src[src_offset +% offsetOf(i, meta, 0, rank, rank)];
}

pub fn fill(
    dst: [*]addrspace(.global) f32,
    value: f32,
    count: u32,
) callconv(.kernel) void {
    const i = gpu.globalIndex();
    if (i >= count) return;
    dst[i] = value;
}

/// Picks elementwise between `a` and `b`, all three broadcast to the output.
/// meta: [0..rank) output dims, then condition, a and b strides.
pub fn select(
    condition: [*]addrspace(.global) const f32,
    a: [*]addrspace(.global) const f32,
    b: [*]addrspace(.global) const f32,
    out: [*]addrspace(.global) f32,
    meta: Meta,
    rank: u32,
    count: u32,
) callconv(.kernel) void {
    const i = gpu.globalIndex();
    if (i >= count) return;
    out[i] = if (condition[offsetOf(i, meta, 0, rank, rank)] != 0)
        a[offsetOf(i, meta, 0, 2 * rank, rank)]
    else
        b[offsetOf(i, meta, 0, 3 * rank, rank)];
}

/// Repeats `src` a whole number of times along every axis. Unlike a broadcast
/// this wraps the coordinate rather than pinning it, so it cannot be spelled
/// as a stride of zero.
/// meta: [0..rank) output dims, [rank..2*rank) source dims, then source strides.
pub fn tile(
    src: [*]addrspace(.global) const f32,
    dst: [*]addrspace(.global) f32,
    meta: Meta,
    rank: u32,
    count: u32,
) callconv(.kernel) void {
    const i = gpu.globalIndex();
    if (i >= count) return;

    var remaining = i;
    var offset: u32 = 0;
    var axis = rank;
    while (axis > 0) {
        axis -= 1;
        const coordinate = remaining % meta[axis];
        remaining /= meta[axis];
        offset += (coordinate % meta[rank + axis]) * meta[2 * rank + axis];
    }
    dst[i] = src[offset];
}

/// ONNX spells Clip's bounds as scalar tensors, so they arrive as device
/// pointers rather than as values; either may be absent.
pub fn clip(
    x: [*]addrspace(.global) const f32,
    low: [*]addrspace(.global) const f32,
    high: [*]addrspace(.global) const f32,
    out: [*]addrspace(.global) f32,
    count: u32,
    has_low: u32,
    has_high: u32,
) callconv(.kernel) void {
    const i = gpu.globalIndex();
    if (i >= count) return;
    var v = x[i];
    if (has_low != 0) v = @max(v, low[0]);
    if (has_high != 0) v = @min(v, high[0]);
    out[i] = v;
}

/// Writes each `slice`-element run of `updates` at its own offset in `dst`.
/// ScatterND reduces to this once the host has flattened its index tuples.
pub fn scatter(
    updates: [*]addrspace(.global) const f32,
    offsets: [*]addrspace(.global) const u32,
    dst: [*]addrspace(.global) f32,
    slice: u32,
    count: u32,
) callconv(.kernel) void {
    const i = gpu.globalIndex();
    if (i >= count) return;
    dst[offsets[i / slice] + i % slice] = updates[i];
}

/// Copies one dense tensor into an axis window of another dense tensor. Used
/// by Concat and Split without constructing temporary tensors.
pub fn concatCopy(
    src: [*]addrspace(.global) const f32,
    dst: [*]addrspace(.global) f32,
    src_axis: u32,
    dst_axis: u32,
    inner: u32,
    dst_axis_offset: u32,
    count: u32,
) callconv(.kernel) void {
    const i = gpu.globalIndex();
    if (i >= count) return;
    const inner_index = i % inner;
    const axis_index = (i / inner) % src_axis;
    const outer_index = i / (inner * src_axis);
    dst[(outer_index * dst_axis + dst_axis_offset + axis_index) * inner + inner_index] = src[i];
}

/// Constant padding. `meta` contains output dims, input dims, input strides,
/// then leading pads, each `rank` elements long.
pub fn pad(
    src: [*]addrspace(.global) const f32,
    dst: [*]addrspace(.global) f32,
    meta: Meta,
    rank: u32,
    count: u32,
    /// ONNX spells the fill as a scalar tensor, so it arrives as a device
    /// pointer rather than a value, and may be absent.
    value: [*]addrspace(.global) const f32,
    has_value: u32,
) callconv(.kernel) void {
    const index = gpu.globalIndex();
    if (index >= count) return;
    var remaining = index;
    var source_offset: u32 = 0;
    var inside = true;
    var axis = rank;
    while (axis > 0) {
        axis -= 1;
        const coordinate = remaining % meta[axis];
        remaining /= meta[axis];
        const before = meta[3 * rank + axis];
        const input_dim = meta[rank + axis];
        if (coordinate < before or coordinate >= before + input_dim) {
            inside = false;
        } else {
            source_offset += (coordinate - before) * meta[2 * rank + axis];
        }
    }
    dst[index] = if (inside) src[source_offset] else (if (has_value != 0) value[0] else 0);
}

/// out[outer, index, inner] = src[outer, indices[index], inner]
pub fn gather(
    src: [*]addrspace(.global) const f32,
    indices: [*]addrspace(.global) const u32,
    dst: [*]addrspace(.global) f32,
    outer: u32,
    axis: u32,
    inner: u32,
    index_count: u32,
    count: u32,
) callconv(.kernel) void {
    const i = gpu.globalIndex();
    if (i >= count) return;

    const inner_index = i % inner;
    const index = (i / inner) % index_count;
    const outer_index = i / (inner * index_count);
    _ = outer;

    const source = indices[index];
    dst[i] = src[(outer_index * axis + source) * inner + inner_index];
}

/// One block per row: mean and variance over `cols`, then scale and shift.
pub fn layerNorm(
    x: [*]addrspace(.global) const f32,
    scale: [*]addrspace(.global) const f32,
    bias: [*]addrspace(.global) const f32,
    out: [*]addrspace(.global) f32,
    cols: u32,
    epsilon: f32,
    has_bias: u32,
) callconv(.kernel) void {
    const row = gpu.blockIndex();
    const lane = gpu.threadIndex();
    const width = gpu.blockSize();
    const base = row * cols;

    var sum: f32 = 0;
    var i = lane;
    while (i < cols) : (i += width) sum += x[base + i];
    const mean = reduceSum(sum) / @as(f32, @floatFromInt(cols));

    var variance: f32 = 0;
    i = lane;
    while (i < cols) : (i += width) {
        const d = x[base + i] - mean;
        variance += d * d;
    }
    const inverse = 1.0 / @sqrt(reduceSum(variance) / @as(f32, @floatFromInt(cols)) + epsilon);

    i = lane;
    while (i < cols) : (i += width) {
        const normalized = (x[base + i] - mean) * inverse;
        out[base + i] = normalized * scale[i] + (if (has_bias != 0) bias[i] else 0);
    }
}

/// One block per row, over the last axis.
///
/// The maximum and the sum come out of a single pass, the running sum being
/// rescaled whenever a larger element turns up. The obvious three-pass form
/// touches the row five times -- read, read and write, read and write -- and
/// attention rows here run to 5184 columns, so those passes are the cost.
pub fn softmax(
    x: [*]addrspace(.global) const f32,
    out: [*]addrspace(.global) f32,
    cols: u32,
) callconv(.kernel) void {
    const row = gpu.blockIndex();
    const lane = gpu.threadIndex();
    const width = gpu.blockSize();
    const base = row * cols;

    var top: f32 = neg_inf;
    var sum: f32 = 0;
    var i = lane;
    while (i < cols) : (i += width) {
        const v = x[base + i];
        if (v > top) {
            sum = if (top == neg_inf) 0 else sum * exp(top - v);
            top = v;
        }
        if (v != neg_inf and top != neg_inf) {
            sum += exp(v - top);
        }
    }

    // Every lane's running sum is against its own maximum, so it is brought
    // onto the row's maximum before the sums are added together.
    const peak = reduceMax(top);
    const local_sum = if (top == neg_inf or peak == neg_inf) 0 else (sum * exp(top - peak));
    const total = reduceSum(local_sum);

    i = lane;
    while (i < cols) : (i += width) {
        const v = x[base + i];
        out[base + i] = if (v == neg_inf or peak == neg_inf or total == 0) 0 else (exp(v - peak) / total);
    }
}

/// The one scratch array both reductions share. Its callers reduce twice in a
/// row -- a mean then a variance, a maximum then a sum -- so each reduction
/// has to leave it free for the next: without the closing barrier a thread
/// that has moved on can overwrite the result while a slower one is still
/// reading it, which shows up as a handful of wrong values per run.
var reduction: [1024]f32 addrspace(.shared) = undefined;

fn reduceSum(value: f32) f32 {
    const lane = gpu.threadIndex();
    reduction[lane] = value;
    gpu.syncThreads();

    var stride = gpu.blockSize() / 2;
    while (stride > 0) : (stride /= 2) {
        if (lane < stride) reduction[lane] += reduction[lane + stride];
        gpu.syncThreads();
    }
    const total = reduction[0];
    gpu.syncThreads();
    return total;
}

fn reduceMax(value: f32) f32 {
    const lane = gpu.threadIndex();
    reduction[lane] = value;
    gpu.syncThreads();

    var stride = gpu.blockSize() / 2;
    while (stride > 0) : (stride /= 2) {
        if (lane < stride) reduction[lane] = @max(reduction[lane], reduction[lane + stride]);
        gpu.syncThreads();
    }
    const peak = reduction[0];
    gpu.syncThreads();
    return peak;
}

// A 16x16 block of threads, each owning a `rows` x `cols` patch of C. Holding
// that patch in registers is what makes the kernel fast: one pair of shared
// reads then feeds `rows * cols` multiply-adds instead of one, so the inner
// loop stops being bound by shared memory.
//
// The patch is strided rather than contiguous -- a thread owns every 16th row
// and column of its block's tile -- so the 16 lanes reading a row of `b_tile`
// take 16 consecutive floats and no two of them land in the same bank.
const mm_threads = 16;
const mm_rows = 8;
const mm_cols = 4;
const mm_tile_m = mm_threads * mm_rows;
const mm_tile_n = mm_threads * mm_cols;
const mm_tile_k = 16;
const mm_stage = mm_threads * mm_threads;

var a_tile: [mm_tile_m][mm_tile_k]f32 addrspace(.shared) = undefined;
var b_tile: [mm_tile_k][mm_tile_n]f32 addrspace(.shared) = undefined;

/// Stages the block's slab of the left operand, which is a plain `m` x `k`
/// matrix for both of the kernels below -- weights, in the convolution's case.
inline fn stageLeft(left: [*]addrspace(.global) const f32, base: u32, row_base: u32, m: u32, k: u32, step: u32, thread: u32) void {
    inline for (0..mm_tile_m * mm_tile_k / mm_stage) |slab| {
        const index = thread + slab * mm_stage;
        const r = index / mm_tile_k;
        const column = index % mm_tile_k;
        a_tile[r][column] = if (row_base + r < m and step + column < k)
            left[base + (row_base + r) * k + step + column]
        else
            0;
    }
}

/// Folds one staged k-step into the thread's register patch. `@mulAdd` rather
/// than `acc += a * b`: Zig will not contract a multiply and an add on its
/// own, and uncontracted it issues two instructions where the hardware has one.
inline fn accumulateTiles(acc: *[mm_rows][mm_cols]f32, tx: u32, ty: u32) void {
    // Fully unrolled, and instantiated once per caller: 16 depths of 8x4
    // multiply-adds is more comptime steps than the default budget allows.
    @setEvalBranchQuota(20000);
    inline for (0..mm_tile_k) |depth| {
        var a_reg: [mm_rows]f32 = undefined;
        var b_reg: [mm_cols]f32 = undefined;
        inline for (0..mm_rows) |i| a_reg[i] = a_tile[ty + i * mm_threads][depth];
        inline for (0..mm_cols) |j| {
            b_reg[j] = b_tile[depth][tx + j * mm_threads];
        }
        inline for (0..mm_rows) |i| {
            inline for (0..mm_cols) |j| acc[i][j] = @mulAdd(f32, a_reg[i], b_reg[j], acc[i][j]);
        }
    }
}

/// C[b] = A[b] x B[b], blockIdx.z selecting the batch. A batch stride of zero
/// broadcasts that operand across the batch.
pub fn matmul(
    a: [*]addrspace(.global) const f32,
    b: [*]addrspace(.global) const f32,
    c: [*]addrspace(.global) f32,
    m: u32,
    n: u32,
    k: u32,
    a_batch: u32,
    b_batch: u32,
    c_batch: u32,
    /// B held as `n` rows of `k`, the way Gemm's `transB` spells it. Decided
    /// once here rather than tested per element: the two layouts differ only
    /// inside the staging loop, and a branch there costs the whole kernel
    /// about a tenth of its throughput.
    b_transposed: u32,
    /// Added to every element on the way out, indexed by column, where a
    /// MatMul the graph followed with an Add has been folded into this one.
    bias: [*]addrspace(.global) const f32,
    has_bias: u32,
    /// Applied to every element after the bias, where the activation that
    /// followed the product has been folded into it as well.
    activation: u32,
    /// How many tiles of C a work group block covers. Which tile each block
    /// takes decides what the resident ones share; the host sizes this to what
    /// it reckons the last level cache holds both operands of.
    block_m: u32,
    block_n: u32,
) callconv(.kernel) void {
    if (b_transposed != 0) {
        matmulTiled(a, b, c, m, n, k, a_batch, b_batch, c_batch, bias, has_bias, activation, block_m, block_n, true);
    } else {
        matmulTiled(a, b, c, m, n, k, a_batch, b_batch, c_batch, bias, has_bias, activation, block_m, block_n, false);
    }
}

inline fn matmulTiled(
    a: [*]addrspace(.global) const f32,
    b: [*]addrspace(.global) const f32,
    c: [*]addrspace(.global) f32,
    m: u32,
    n: u32,
    k: u32,
    a_batch: u32,
    b_batch: u32,
    c_batch: u32,
    bias: [*]addrspace(.global) const f32,
    has_bias: u32,
    activation: u32,
    block_m: u32,
    block_n: u32,
    comptime transposed: bool,
) void {
    const tx = gpu.threadIndex();
    const ty = gpu.threadIndexY();
    const thread = ty * mm_threads + tx;
    const tiles_n = (n + mm_tile_n - 1) / mm_tile_n;
    const tiles_m = (m + mm_tile_m - 1) / mm_tile_m;
    const blocks_n = (tiles_n + block_n - 1) / block_n;
    const taken = gpu.blockIndex();
    const at = taken / (block_m * block_n);
    const within = taken % (block_m * block_n);
    const tile_m = (at / blocks_n) * block_m + within / block_n;
    const tile_n = (at % blocks_n) * block_n + within % block_n;
    // The grid is a whole number of blocks, so the last ones overhang.
    if (tile_m >= tiles_m or tile_n >= tiles_n) return;
    const row_base = tile_m * mm_tile_m;
    const col_base = tile_n * mm_tile_n;
    const batch = gpu.blockIndexZ();
    const a_base = batch * a_batch;
    const b_base = batch * b_batch;

    var acc: [mm_rows][mm_cols]f32 = @splat(@splat(0));

    var step: u32 = 0;
    while (step < k) : (step += mm_tile_k) {
        // The 256 threads stage the block's slab of A and of B between them.
        stageLeft(a, a_base, row_base, m, k, step, thread);
        inline for (0..mm_tile_k * mm_tile_n / mm_stage) |slab| {
            const index = thread + slab * mm_stage;
            const r = index / mm_tile_n;
            const column = index % mm_tile_n;
            b_tile[r][column] = if (step + r < k and col_base + column < n)
                (if (transposed)
                    b[b_base + (col_base + column) * k + step + r]
                else
                    b[b_base + (step + r) * n + col_base + column])
            else
                0;
        }
        gpu.syncThreads();

        accumulateTiles(&acc, tx, ty);
        gpu.syncThreads();
    }

    inline for (0..mm_rows) |i| {
        const row = row_base + ty + i * mm_threads;
        if (row < m) {
            inline for (0..mm_cols) |j| {
                const col = col_base + tx + j * mm_threads;
                if (col < n) {
                    const value = acc[i][j] + if (has_bias != 0) bias[col] else 0;
                    c[batch * c_batch + row * n + col] =
                        if (activation != 0) 0.5 * value * (1.0 + erf(value * 0.7071067811865476)) else value;
                }
            }
        }
    }
}

/// Sums over a set of axes. The kept axes are walked as usual; the reduced
/// ones are a second shape that each output element sweeps.
/// meta: [0..rank) output dims with a 1 where an axis was reduced,
///       [rank..2*rank) input strides, then the reduced dims and their strides.
pub fn sumAxes(
    src: [*]addrspace(.global) const f32,
    dst: [*]addrspace(.global) f32,
    meta: Meta,
    rank: u32,
    reduced: u32,
    swept: u32,
    count: u32,
) callconv(.kernel) void {
    const i = gpu.globalIndex();
    if (i >= count) return;

    const base = offsetOf(i, meta, 0, rank, rank);
    var total: f32 = 0;
    var step: u32 = 0;
    while (step < swept) : (step += 1) {
        total += src[base + offsetOf(step, meta, 2 * rank, 2 * rank + reduced, reduced)];
    }
    dst[i] = total;
}

/// Nearest-neighbour resize, ONNX's `asymmetric` mapping with `floor`: an
/// output coordinate reads back from `coordinate * in_dim / out_dim`.
/// meta: [0..rank) output dims, [rank..2*rank) input dims, then input strides.
pub fn resizeNearest(
    src: [*]addrspace(.global) const f32,
    dst: [*]addrspace(.global) f32,
    meta: Meta,
    rank: u32,
    count: u32,
) callconv(.kernel) void {
    const i = gpu.globalIndex();
    if (i >= count) return;

    var remaining = i;
    var offset: u32 = 0;
    var axis = rank;
    while (axis > 0) {
        axis -= 1;
        const dim = meta[axis];
        const coordinate = remaining % dim;
        remaining /= dim;
        offset += (coordinate * meta[rank + axis] / dim) * meta[2 * rank + axis];
    }
    dst[i] = src[offset];
}

/// One block per plane: normalizes each plane over its own extent, then scales
/// and shifts by its channel's parameters. Unlike `layerNorm`, whose scale
/// runs along the row, this one has a single pair per row.
pub fn instanceNorm(
    x: [*]addrspace(.global) const f32,
    scale: [*]addrspace(.global) const f32,
    bias: [*]addrspace(.global) const f32,
    out: [*]addrspace(.global) f32,
    channels: u32,
    cols: u32,
    epsilon: f32,
) callconv(.kernel) void {
    const row = gpu.blockIndex();
    const lane = gpu.threadIndex();
    const width = gpu.blockSize();
    const base = row * cols;

    var sum: f32 = 0;
    var i = lane;
    while (i < cols) : (i += width) sum += x[base + i];
    const mean = reduceSum(sum) / @as(f32, @floatFromInt(cols));

    var variance: f32 = 0;
    i = lane;
    while (i < cols) : (i += width) {
        const d = x[base + i] - mean;
        variance += d * d;
    }
    const inverse = 1.0 / @sqrt(reduceSum(variance) / @as(f32, @floatFromInt(cols)) + epsilon);

    const gain = scale[row % channels];
    const shift = bias[row % channels];
    i = lane;
    while (i < cols) : (i += width) out[base + i] = (x[base + i] - mean) * inverse * gain + shift;
}

/// ONNX Runtime's MatMulNBits: A times a weight matrix held as `n` rows of
/// `k` 4-bit values, one scale per `block_size` of them along a row. A weight
/// row is an output column, so this computes A x W'.
///
/// The weights are unpacked as the tile is staged, which keeps them 4-bit in
/// memory: an eighth of the traffic of the matrix they stand for, and the only
/// way the text-prompt models fit in VRAM at all.
///
/// Same block tiling as `matmul`, and it needs the same grid.
pub fn matmulNBits(
    a: [*]addrspace(.global) const f32,
    quantized: [*]addrspace(.global) const u8,
    scales: [*]addrspace(.global) const f32,
    out: [*]addrspace(.global) f32,
    m: u32,
    n: u32,
    k: u32,
    /// log2 of the block size. ONNX requires a power of two, and the host
    /// checks it, because an integer division here would cost more than the
    /// unpacking it serves: it sits in the innermost staging loop.
    block_shift: u32,
    blocks_per_row: u32,
) callconv(.kernel) void {
    const tx = gpu.threadIndex();
    const ty = gpu.threadIndexY();
    const thread = ty * mm_threads + tx;
    const row_base = gpu.blockIndexY() * mm_tile_m;
    const col_base = gpu.blockIndex() * mm_tile_n;
    const shift: u5 = @intCast(block_shift);
    const block_mask = (@as(u32, 1) << shift) - 1;
    const blob = @as(u32, 1) << (shift - 1);
    const row_stride = blocks_per_row * blob;

    var acc: [mm_rows][mm_cols]f32 = @splat(@splat(0));

    var step: u32 = 0;
    while (step < k) : (step += mm_tile_k) {
        stageLeft(a, 0, row_base, m, k, step, thread);
        // A weight row is contiguous along `k` and this walks across rows, so
        // the reads are scattered where the plain matmul's are consecutive.
        // Staging along `k` instead, to read whole rows, measures the same:
        // what this kernel is short of is not memory but the unpacking.
        inline for (0..mm_tile_k * mm_tile_n / mm_stage) |slab| {
            const index = thread + slab * mm_stage;
            const depth = index / mm_tile_n;
            const column = index % mm_tile_n;
            const weight_row = col_base + column;
            const along = step + depth;

            var value: f32 = 0;
            if (weight_row < n and along < k) {
                const block = along >> shift;
                const within = along & block_mask;
                const byte = quantized[weight_row * row_stride + block * blob + (within >> 1)];
                const nibble: u8 = if (within & 1 == 0) byte & 0xf else byte >> 4;
                // These exports carry no zero points, which means the
                // midpoint of the 4-bit range.
                value = (@as(f32, @floatFromInt(nibble)) - 8.0) * scales[weight_row * blocks_per_row + block];
            }
            b_tile[depth][column] = value;
        }
        gpu.syncThreads();

        accumulateTiles(&acc, tx, ty);
        gpu.syncThreads();
    }

    inline for (0..mm_rows) |i| {
        const row = row_base + ty + i * mm_threads;
        if (row < m) {
            inline for (0..mm_cols) |j| {
                const col = col_base + tx + j * mm_threads;
                if (col < n) out[row * n + col] = acc[i][j];
            }
        }
    }
}

/// Running sum along one axis, one thread per line. The lines are short where
/// this is used, so walking them in order beats a parallel scan.
pub fn cumulativeSum(
    src: [*]addrspace(.global) const f32,
    dst: [*]addrspace(.global) f32,
    along: u32,
    inner: u32,
    count: u32,
) callconv(.kernel) void {
    const i = gpu.globalIndex();
    if (i >= count) return;

    const base = (i / inner) * along * inner + i % inner;
    var total: f32 = 0;
    var step: u32 = 0;
    while (step < along) : (step += 1) {
        total += src[base + step * inner];
        dst[base + step * inner] = total;
    }
}

/// meta: in_h, in_w, out_h, out_w, kernel_h, kernel_w, stride_h, stride_w,
///       pad_h, pad_w
pub fn maxPool2d(
    src: [*]addrspace(.global) const f32,
    dst: [*]addrspace(.global) f32,
    meta: Meta,
    count: u32,
) callconv(.kernel) void {
    const i = gpu.globalIndex();
    if (i >= count) return;

    const in_h = meta[0];
    const in_w = meta[1];
    const out_h = meta[2];
    const out_w = meta[3];
    const kernel_h = meta[4];
    const kernel_w = meta[5];

    const out_x = i % out_w;
    const out_y = (i / out_w) % out_h;
    const plane = i / (out_w * out_h);

    var top: f32 = -3.4e38;
    for (0..kernel_h) |ky| {
        const y = out_y * meta[6] + @as(u32, @intCast(ky));
        if (y < meta[8]) continue;
        const iy = y - meta[8];
        if (iy >= in_h) continue;
        for (0..kernel_w) |kx| {
            const x = out_x * meta[7] + @as(u32, @intCast(kx));
            if (x < meta[9]) continue;
            const ix = x - meta[9];
            if (ix >= in_w) continue;
            top = @max(top, src[(plane * in_h + iy) * in_w + ix]);
        }
    }
    dst[i] = top;
}

/// Convolution as an implicit GEMM: out[channel, pixel] is the weight matrix,
/// already laid out as `out_channels` x `in_channels * kernel_h * kernel_w`,
/// times the patch matrix that an im2col would build. The patch matrix is
/// gathered straight out of the image while staging the tile, so it never
/// exists -- which matters, since for SAM 3's neck it would be 764 MiB.
///
/// Same block tiling as `matmul`, and it needs the same grid.
/// meta: in_channels, in_h, in_w, out_h, out_w, kernel_h, kernel_w,
///       stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w
pub fn conv2dGemm(
    x: [*]addrspace(.global) const f32,
    w: [*]addrspace(.global) const f32,
    bias: [*]addrspace(.global) const f32,
    out: [*]addrspace(.global) f32,
    meta: Meta,
    m: u32,
    n: u32,
    k: u32,
    has_bias: u32,
) callconv(.kernel) void {
    const tx = gpu.threadIndex();
    const ty = gpu.threadIndexY();
    const thread = ty * mm_threads + tx;
    const row_base = gpu.blockIndexY() * mm_tile_m;
    const col_base = gpu.blockIndex() * mm_tile_n;
    const batch = gpu.blockIndexZ();

    const in_channels = meta[0];
    const in_h = meta[1];
    const in_w = meta[2];
    const out_w = meta[4];
    const kernel_w = meta[6];
    const stride_h = meta[7];
    const stride_w = meta[8];
    const pad_h = meta[9];
    const pad_w = meta[10];
    const dilation_h = meta[11];
    const dilation_w = meta[12];
    const taps = meta[5] * kernel_w;

    const x_base = batch * in_channels * in_h * in_w;

    var acc: [mm_rows][mm_cols]f32 = @splat(@splat(0));

    var step: u32 = 0;
    while (step < k) : (step += mm_tile_k) {
        stageLeft(w, 0, row_base, m, k, step, thread);
        inline for (0..mm_tile_k * mm_tile_n / mm_stage) |slab| {
            const index = thread + slab * mm_stage;
            const r = index / mm_tile_n;
            const column = index % mm_tile_n;
            const tap = step + r;
            const pixel = col_base + column;

            var value: f32 = 0;
            if (tap < k and pixel < n) {
                const channel = tap / taps;
                const ky = (tap % taps) / kernel_w;
                const kx = tap % kernel_w;
                const y = (pixel / out_w) * stride_h + ky * dilation_h;
                const p = (pixel % out_w) * stride_w + kx * dilation_w;
                if (y >= pad_h and p >= pad_w and y - pad_h < in_h and p - pad_w < in_w) {
                    value = x[x_base + (channel * in_h + y - pad_h) * in_w + p - pad_w];
                }
            }
            b_tile[r][column] = value;
        }
        gpu.syncThreads();

        accumulateTiles(&acc, tx, ty);
        gpu.syncThreads();
    }

    inline for (0..mm_rows) |i| {
        const row = row_base + ty + i * mm_threads;
        if (row < m) {
            const shift = if (has_bias != 0) bias[row] else 0;
            inline for (0..mm_cols) |j| {
                const col = col_base + tx + j * mm_threads;
                if (col < n) out[batch * m * n + row * n + col] = acc[i][j] + shift;
            }
        }
    }
}

/// Direct convolution over NCHW input with OIHW weights. Still what a grouped
/// convolution runs on: the implicit GEMM above assumes one group, since a
/// block tile of output channels would otherwise straddle two of them.
pub fn conv2d(
    x: [*]addrspace(.global) const f32,
    w: [*]addrspace(.global) const f32,
    bias: [*]addrspace(.global) const f32,
    out: [*]addrspace(.global) f32,
    meta: Meta,
    count: u32,
    has_bias: u32,
) callconv(.kernel) void {
    const i = gpu.globalIndex();
    if (i >= count) return;

    const in_channels = meta[0];
    const in_h = meta[1];
    const in_w = meta[2];
    const out_channels = meta[3];
    const out_h = meta[4];
    const out_w = meta[5];
    const kernel_h = meta[6];
    const kernel_w = meta[7];
    const stride_h = meta[8];
    const stride_w = meta[9];
    const pad_h = meta[10];
    const pad_w = meta[11];
    const dilation_h = meta[12];
    const dilation_w = meta[13];
    const groups = meta[14];

    const out_x = i % out_w;
    const out_y = (i / out_w) % out_h;
    const channel = (i / (out_w * out_h)) % out_channels;
    const batch = i / (out_w * out_h * out_channels);

    const group_size = in_channels / groups;
    const group = channel / (out_channels / groups);

    var sum: f32 = 0;
    for (0..group_size) |c| {
        const in_c = group * group_size + @as(u32, @intCast(c));
        for (0..kernel_h) |ky| {
            const in_y = out_y * stride_h + @as(u32, @intCast(ky)) * dilation_h;
            if (in_y < pad_h) continue;
            const y = in_y - pad_h;
            if (y >= in_h) continue;
            for (0..kernel_w) |kx| {
                const in_x = out_x * stride_w + @as(u32, @intCast(kx)) * dilation_w;
                if (in_x < pad_w) continue;
                const x_index = in_x - pad_w;
                if (x_index >= in_w) continue;

                const pixel = x[((batch * in_channels + in_c) * in_h + y) * in_w + x_index];
                const weight = w[
                    ((channel * group_size + @as(u32, @intCast(c))) * kernel_h +
                        @as(u32, @intCast(ky))) * kernel_w + @as(u32, @intCast(kx))
                ];
                sum += pixel * weight;
            }
        }
    }

    out[i] = sum + (if (has_bias != 0) bias[channel] else 0);
}

/// Transposed convolution, gathering rather than scattering: each output
/// pixel walks the input positions that would have written to it.
pub fn convTranspose2d(
    x: [*]addrspace(.global) const f32,
    w: [*]addrspace(.global) const f32,
    bias: [*]addrspace(.global) const f32,
    out: [*]addrspace(.global) f32,
    meta: Meta,
    count: u32,
    has_bias: u32,
) callconv(.kernel) void {
    const i = gpu.globalIndex();
    if (i >= count) return;

    const in_channels = meta[0];
    const in_h = meta[1];
    const in_w = meta[2];
    const out_channels = meta[3];
    const out_h = meta[4];
    const out_w = meta[5];
    const kernel_h = meta[6];
    const kernel_w = meta[7];
    const stride_h = meta[8];
    const stride_w = meta[9];
    const pad_h = meta[10];
    const pad_w = meta[11];

    const out_x = i % out_w;
    const out_y = (i / out_w) % out_h;
    const channel = (i / (out_w * out_h)) % out_channels;
    const batch = i / (out_w * out_h * out_channels);

    var sum: f32 = 0;
    for (0..kernel_h) |ky| {
        const shifted_y = out_y + pad_h;
        if (shifted_y < @as(u32, @intCast(ky))) continue;
        const numerator_y = shifted_y - @as(u32, @intCast(ky));
        if (numerator_y % stride_h != 0) continue;
        const in_y = numerator_y / stride_h;
        if (in_y >= in_h) continue;

        for (0..kernel_w) |kx| {
            const shifted_x = out_x + pad_w;
            if (shifted_x < @as(u32, @intCast(kx))) continue;
            const numerator_x = shifted_x - @as(u32, @intCast(kx));
            if (numerator_x % stride_w != 0) continue;
            const in_x = numerator_x / stride_w;
            if (in_x >= in_w) continue;

            for (0..in_channels) |c| {
                const in_c: u32 = @intCast(c);
                const pixel = x[((batch * in_channels + in_c) * in_h + in_y) * in_w + in_x];
                const weight = w[
                    ((in_c * out_channels + channel) * kernel_h +
                        @as(u32, @intCast(ky))) * kernel_w + @as(u32, @intCast(kx))
                ];
                sum += pixel * weight;
            }
        }
    }

    out[i] = sum + (if (has_bias != 0) bias[channel] else 0);
}

/// Non-overlapping transposed convolution as GEMM plus pixel shuffle.
/// A is the input viewed as [pixel, input_channel], B is the weight viewed as
/// [input_channel, output_channel * kernel_h * kernel_w]. Both views are
/// gathered from their native NCHW / ONNX layouts while staging each tile.
pub fn convTranspose2dGemm(
    x: [*]addrspace(.global) const f32,
    w: [*]addrspace(.global) const f32,
    bias: [*]addrspace(.global) const f32,
    out: [*]addrspace(.global) f32,
    in_h: u32,
    in_w: u32,
    out_channels: u32,
    out_h: u32,
    out_w: u32,
    kernel_h: u32,
    kernel_w: u32,
    m: u32,
    n: u32,
    k: u32,
    has_bias: u32,
) callconv(.kernel) void {
    _ = in_h;
    const tx = gpu.threadIndex();
    const ty = gpu.threadIndexY();
    const thread = ty * mm_threads + tx;
    const row_base = gpu.blockIndexY() * mm_tile_m;
    const col_base = gpu.blockIndex() * mm_tile_n;
    const batch = gpu.blockIndexZ();
    const taps = kernel_h * kernel_w;
    const x_base = batch * k * m;
    const out_base = batch * out_channels * out_h * out_w;

    var acc: [mm_rows][mm_cols]f32 = @splat(@splat(0));

    var step: u32 = 0;
    while (step < k) : (step += mm_tile_k) {
        inline for (0..mm_tile_m * mm_tile_k / mm_stage) |slab| {
            const idx = thread + slab * mm_stage;
            const pixel = row_base + idx / mm_tile_k;
            const channel = step + idx % mm_tile_k;
            a_tile[idx / mm_tile_k][idx % mm_tile_k] = if (pixel < m and channel < k)
                x[x_base + channel * m + pixel]
            else
                0;
        }

        inline for (0..mm_tile_k * mm_tile_n / mm_stage) |slab| {
            const idx = thread + slab * mm_stage;
            const channel = step + idx / mm_tile_n;
            const feature = col_base + idx % mm_tile_n;
            b_tile[idx / mm_tile_n][idx % mm_tile_n] = if (channel < k and feature < n)
                w[channel * n + feature]
            else
                0;
        }
        gpu.syncThreads();

        accumulateTiles(&acc, tx, ty);
        gpu.syncThreads();
    }

    inline for (0..mm_rows) |i| {
        const pixel = row_base + ty + i * mm_threads;
        if (pixel < m) {
            const in_y = pixel / in_w;
            const in_x = pixel % in_w;
            inline for (0..mm_cols) |j| {
                const feature = col_base + tx + j * mm_threads;
                if (feature < n) {
                    const channel = feature / taps;
                    const tap = feature % taps;
                    const out_y = in_y * kernel_h + tap / kernel_w;
                    const out_x = in_x * kernel_w + tap % kernel_w;
                    out[out_base + (channel * out_h + out_y) * out_w + out_x] =
                        acc[i][j] + (if (has_bias != 0) bias[channel] else 0);
                }
            }
        }
    }
}

export fn anchor() usize {
    return @intFromPtr(&binary) ^ @intFromPtr(&unary) ^ @intFromPtr(&copy) ^
        @intFromPtr(&fill) ^ @intFromPtr(&select) ^ @intFromPtr(&tile) ^ @intFromPtr(&conv2dGemm) ^
        @intFromPtr(&matmulNBits) ^ @intFromPtr(&cumulativeSum) ^ @intFromPtr(&maxPool2d) ^
        @intFromPtr(&sumAxes) ^ @intFromPtr(&resizeNearest) ^ @intFromPtr(&instanceNorm) ^
        @intFromPtr(&clip) ^ @intFromPtr(&scatter) ^
        @intFromPtr(&concatCopy) ^ @intFromPtr(&pad) ^
        @intFromPtr(&gather) ^ @intFromPtr(&layerNorm) ^
        @intFromPtr(&softmax) ^ @intFromPtr(&matmul) ^ @intFromPtr(&conv2d) ^
        @intFromPtr(&convTranspose2d) ^ @intFromPtr(&convTranspose2dGemm);
}
