//! Portable SAM 3 operators for Apple GPUs. Generated from kernels.cl by
//! tools/port-metal.rb; edit the OpenCL operator bodies, then regenerate.
#include <metal_stdlib>
using namespace metal;

#if SAM3_HALF
typedef half real;
typedef half4 realv;
#define TO_REALV(v) half4(v)
#else
typedef float real;
typedef float4 realv;
#define TO_REALV(v) (v)
#endif
typedef float4 floatv;
#define LANE_STEP 4
#define TO_FLOATV(v) float4(v)
#define VLOADV(i, p) (((device const realv*)(p))[i])
#define VSTOREV(v, i, p) (((device realv*)(p))[i] = (v))
#define INFINITY __builtin_inff()
#define CLK_LOCAL_MEM_FENCE 0
#define barrier(ignore) threadgroup_barrier(mem_flags::mem_threadgroup)
#define get_global_id(axis) gid[axis]
#define get_local_id(axis) lid[axis]
#define get_group_id(axis) group_id[axis]
#define get_local_size(axis) threads_per_group[axis]

enum Binary {
    BIN_ADD = 0, BIN_SUB = 1, BIN_MUL = 2, BIN_DIV = 3, BIN_POW = 4,
    BIN_MIN = 5, BIN_MAX = 6, BIN_EQUAL = 7, BIN_LESS = 8, BIN_GREATER = 9
};

enum Unary {
    UN_NEG = 0, UN_ERF = 1, UN_EXP = 2, UN_SQRT = 3, UN_RECIP = 4,
    UN_SIGMOID = 5, UN_TANH = 6, UN_RELU = 7, UN_ABS = 8, UN_FLOOR = 9,
    UN_SIN = 10, UN_COS = 11, UN_LOG = 12, UN_SIGN = 13, UN_IS_NAN = 14,
    UN_GELU = 15
};

inline uint offsetOf(uint index, device const uint* meta, uint dims_at, uint strides_at, uint rank) {
    uint remaining = index;
    uint offset = 0;
    for (uint axis = rank; axis > 0; axis--) {
        uint dim = meta[dims_at + axis - 1];
        uint coordinate = remaining % dim;
        remaining /= dim;
        offset += coordinate * meta[strides_at + axis - 1];
    }
    return offset;
}

/// Where an operand is the tail of the output shape -- a bias over the last
/// axis, a window shared by every batch, a single value -- the same block of
/// `period` values repeats over everything in front of it, and one wrap is the
/// whole index calculation. `period` is uniform across the work group, so the
/// power of two test costs nothing and the common cases avoid the division.
inline uint wrap(uint index, uint period) {
    return ((period & (period - 1u)) == 0u) ? (index & (period - 1u)) : (index % period);
}

/// exp to about eleven bits, which is more than a half holds. The hardware
/// has it as one instruction where the precise version is a dozen. Used where
/// an exponential is an interior step of something larger -- the weights a
/// softmax normalizes away, the polynomial inside erf -- rather than where the
/// graph asked for the function itself.

inline float power(float x, float y) {
    if (y == 2.0f) return x * x;
    if (y == 0.5f) return sqrt(x);
    return pow(x, y);
}

inline float erf_approx(float x) {
    float sign = (x < 0.0f) ? -1.0f : 1.0f;
    float v = fabs(x);
    float t = 1.0f / (1.0f + 0.3275911f * v);
    float poly = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
    return sign * (1.0f - poly * exp(-v * v));
}

/// The activation the graph spells as a division, an error function, an
/// addition, and two multiplications over the same tensor -- eleven passes
/// through memory where one will do. Same approximation, so same answer.
inline float gelu(float x) {
    return 0.5f * x * (1.0f + erf_approx(x * M_SQRT1_2_F));
}

// Softmax and the normalizations reduce every row twice, and a tree reduction
// through local memory costs a barrier per halving. Where the driver exposes
// sub-groups the hardware does the first log2(sub-group width) of those steps
// with no barrier at all, which leaves two for the whole reduction.

/// A whole SIMD group folds in one instruction and needs no barrier, which
/// leaves only the fold across the groups of a work group -- two barriers for
/// the entire reduction rather than one per halving. The leading barrier is
/// what makes `scratch` reusable between two reductions in a row: it retires
/// the reads the previous one ended with.
#define REDUCE_ACROSS_SIMD_GROUPS(combine, value, scratch, identity)     \
    float within = value;                                               \
    uint groups = block_size / 32u;                                      \
    if (groups <= 1u) return within;                                     \
    barrier(CLK_LOCAL_MEM_FENCE);                                        \
    if ((lane % 32u) == 0u) scratch[lane / 32u] = within;                \
    barrier(CLK_LOCAL_MEM_FENCE);                                        \
    float folded = identity;                                             \
    for (uint g = 0; g < groups; g++) folded = combine(folded, scratch[g]); \
    return folded;

inline float addf(float a, float b) { return a + b; }

inline float reduceSumLocal(float val, threadgroup float* scratch, uint lane, uint block_size) {
    REDUCE_ACROSS_SIMD_GROUPS(addf, simd_sum(val), scratch, 0.0f)
}

inline float reduceMaxLocal(float val, threadgroup float* scratch, uint lane, uint block_size) {
    REDUCE_ACROSS_SIMD_GROUPS(fmax, simd_max(val), scratch, -INFINITY)
}

kernel void binary(
    device const real* a [[buffer(0)]],
    device const real* b [[buffer(1)]],
    device real* out [[buffer(2)]],
    device const uint* meta [[buffer(3)]],
    constant uint& rank [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    constant uint& op [[buffer(6)]],
    constant uint& a_period [[buffer(7)]],
    constant uint& b_period [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    uint i = get_global_id(0);
    if (i >= count) return;
    float x = a_period ? a[wrap(i, a_period)] : a[offsetOf(i, meta, 0, rank, rank)];
    float y = b_period ? b[wrap(i, b_period)] : b[offsetOf(i, meta, 0, 2 * rank, rank)];
    float res = 0.0f;
    switch (op) {
        case BIN_ADD: res = x + y; break;
        case BIN_SUB: res = x - y; break;
        case BIN_MUL: res = x * y; break;
        case BIN_DIV: res = x / y; break;
        case BIN_POW: res = power(x, y); break;
        case BIN_MIN: res = fmin(x, y); break;
        case BIN_MAX: res = fmax(x, y); break;
        case BIN_EQUAL: res = (x == y) ? 1.0f : 0.0f; break;
        case BIN_LESS: res = (x < y) ? 1.0f : 0.0f; break;
        case BIN_GREATER: res = (x > y) ? 1.0f : 0.0f; break;
    }
    out[i] = res;
}

kernel void unary(
    device const real* x [[buffer(0)]],
    device real* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant uint& op [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    uint i = get_global_id(0);
    if (i >= count) return;
    float v = x[i];
    float res = 0.0f;
    switch (op) {
        case UN_NEG: res = -v; break;
        case UN_ERF: res = erf_approx(v); break;
        case UN_EXP: res = exp(v); break;
        case UN_SQRT: res = sqrt(v); break;
        case UN_RECIP: res = 1.0f / v; break;
        case UN_SIGMOID: res = 1.0f / (1.0f + exp(-v)); break;
        case UN_TANH: res = tanh(v); break;
        case UN_RELU: res = fmax(v, 0.0f); break;
        case UN_ABS: res = fabs(v); break;
        case UN_FLOOR: res = floor(v); break;
        case UN_SIN: res = sin(v); break;
        case UN_COS: res = cos(v); break;
        case UN_LOG: res = log(v); break;
        case UN_SIGN: res = (v > 0.0f) ? 1.0f : ((v < 0.0f) ? -1.0f : 0.0f); break;
        case UN_IS_NAN: res = isnan(v) ? 1.0f : 0.0f; break;
        case UN_GELU: res = gelu(v); break;
    }
    out[i] = res;
}

/// `binary` and `unary` again, four elements to a work item. Elementwise
/// arithmetic here is entirely a question of memory, and a lane that moves one
/// float at a time leaves most of the bus idle. The host takes these paths only
/// for the operations below and only when the element count and both repeats
/// divide by LANE_STEP, so no work item straddles the end of a repeat; a
/// period of zero is the one case a group cannot express, a single value
/// broadcast over everything.
inline floatv groupOf(device const real* p, uint index, uint period) {
    return period ? TO_FLOATV(VLOADV(wrap(index, period), p)) : floatv((float)p[0]);
}

kernel void binaryVec(
    device const real* a [[buffer(0)]],
    device const real* b [[buffer(1)]],
    device real* out [[buffer(2)]],
    constant uint& groups [[buffer(3)]],
    constant uint& op [[buffer(4)]],
    constant uint& a_period [[buffer(5)]],
    constant uint& b_period [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    uint i = get_global_id(0);
    if (i >= groups) return;
    floatv x = groupOf(a, i, a_period);
    floatv y = groupOf(b, i, b_period);
    floatv res = floatv(0.0f);
    switch (op) {
        case BIN_ADD: res = x + y; break;
        case BIN_SUB: res = x - y; break;
        case BIN_MUL: res = x * y; break;
        case BIN_DIV: res = x / y; break;
        case BIN_MIN: res = fmin(x, y); break;
        case BIN_MAX: res = fmax(x, y); break;
    }
    VSTOREV(TO_REALV(res), i, out);
}

inline floatv erfApproxVec(floatv x) {
    floatv v = fabs(x);
    floatv t = 1.0f / (1.0f + 0.3275911f * v);
    floatv poly = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
    floatv magnitude = 1.0f - poly * exp(-v * v);
    return select(magnitude, -magnitude, x < 0.0f);
}

kernel void unaryVec(
    device const real* x [[buffer(0)]],
    device real* out [[buffer(1)]],
    constant uint& groups [[buffer(2)]],
    constant uint& op [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    uint i = get_global_id(0);
    if (i >= groups) return;
    floatv v = TO_FLOATV(VLOADV(i, x));
    floatv res = floatv(0.0f);
    switch (op) {
        case UN_NEG: res = -v; break;
        case UN_ERF: res = erfApproxVec(v); break;
        case UN_EXP: res = exp(v); break;
        case UN_SQRT: res = sqrt(v); break;
        case UN_RECIP: res = 1.0f / v; break;
        case UN_SIGMOID: res = 1.0f / (1.0f + exp(-v)); break;
        case UN_TANH: res = tanh(v); break;
        case UN_RELU: res = fmax(v, 0.0f); break;
        case UN_ABS: res = fabs(v); break;
        case UN_FLOOR: res = floor(v); break;
        case UN_SIN: res = sin(v); break;
        case UN_COS: res = cos(v); break;
        case UN_LOG: res = log(v); break;
        case UN_GELU: res = 0.5f * v * (1.0f + erfApproxVec(v * M_SQRT1_2_F)); break;
    }
    VSTOREV(TO_REALV(res), i, out);
}

/// The rotary embedding these exports spell out with eleven operators, five of
/// which write the whole tensor down. The graph multiplies x by a cosine
/// table, turns each adjacent pair of x into (-second, first) through a
/// reshape, a split, a negation and a concatenation, multiplies that by a sine
/// table, and adds the two. Here it is one pass over x.
///
/// `period` is how many elements of the tables go by before they repeat, the
/// same broadcast `wrap` serves elsewhere. `scale` is the constant the graph
/// follows the sum with, or one where it follows it with nothing.
kernel void rope(
    device const real* x [[buffer(0)]],
    device const real* cosine [[buffer(1)]],
    device const real* sine [[buffer(2)]],
    device real* out [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    constant uint& period [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    uint i = get_global_id(0);
    if (i >= count) return;
    uint table = wrap(i, period);
    // Odd lanes take the element below them, even lanes the negated one above.
    float turned = (i & 1u) ? (float)x[i - 1] : -(float)x[i + 1];
    float value = (float)x[i] * (float)cosine[table] + turned * (float)sine[table];
    out[i] = (real)(value * scale);
}

/// The same four elements to a work item. A group is two whole pairs whichever
/// way it falls -- they start on even indices and LANE_STEP is even -- so the
/// turn is a shuffle between registers and never reaches across groups.
kernel void ropeVec(
    device const real* x [[buffer(0)]],
    device const real* cosine [[buffer(1)]],
    device const real* sine [[buffer(2)]],
    device real* out [[buffer(3)]],
    constant uint& groups [[buffer(4)]],
    constant uint& table_groups [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    uint i = get_global_id(0);
    if (i >= groups) return;
    uint table = table_groups ? wrap(i, table_groups) : 0;
    floatv v = TO_FLOATV(VLOADV(i, x));
    floatv turned = floatv(-v.y, v.x, -v.w, v.z);
    floatv c = table_groups ? TO_FLOATV(VLOADV(table, cosine)) : floatv((float)cosine[0]);
    floatv s = table_groups ? TO_FLOATV(VLOADV(table, sine)) : floatv((float)sine[0]);
    VSTOREV(TO_REALV((v * c + turned * s) * scale), i, out);
}

kernel void copy(
    device const real* src [[buffer(0)]],
    device real* dst [[buffer(1)]],
    device const uint* meta [[buffer(2)]],
    constant uint& rank [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    constant uint& src_offset [[buffer(5)]],
    constant uint& dst_offset [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    uint i = get_global_id(0);
    if (i >= count) return;
    dst[dst_offset + i] = src[src_offset + offsetOf(i, meta, 0, rank, rank)];
}

/// `copy` where the innermost axis is contiguous on both sides, so one work
/// item can carry a whole group: a quarter of the index arithmetic, and a
/// vector load in place of a scalar one. The host has already divided the
/// innermost extent by LANE_STEP and set its stride to match, so `offsetOf`
/// lands on the start of a group.
kernel void copyVec(
    device const real* src [[buffer(0)]],
    device real* dst [[buffer(1)]],
    device const uint* meta [[buffer(2)]],
    constant uint& rank [[buffer(3)]],
    constant uint& groups [[buffer(4)]],
    constant uint& src_offset [[buffer(5)]],
    constant uint& dst_offset [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    uint i = get_global_id(0);
    if (i >= groups) return;
    realv group = VLOADV(0, &src[src_offset + offsetOf(i, meta, 0, rank, rank)]);
    VSTOREV(group, i, &dst[dst_offset]);
}

kernel void fill(
    device real* dst [[buffer(0)]],
    constant float& value [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    uint i = get_global_id(0);
    if (i >= count) return;
    dst[i] = value;
}

kernel void select_kernel(
    device const real* condition [[buffer(0)]],
    device const real* a [[buffer(1)]],
    device const real* b [[buffer(2)]],
    device real* out [[buffer(3)]],
    device const uint* meta [[buffer(4)]],
    constant uint& rank [[buffer(5)]],
    constant uint& count [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    uint i = get_global_id(0);
    if (i >= count) return;
    float cond = condition[offsetOf(i, meta, 0, rank, rank)];
    out[i] = (cond != 0.0f) ? a[offsetOf(i, meta, 0, 2 * rank, rank)] : b[offsetOf(i, meta, 0, 3 * rank, rank)];
}

kernel void tile(
    device const real* src [[buffer(0)]],
    device real* dst [[buffer(1)]],
    device const uint* meta [[buffer(2)]],
    constant uint& rank [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    uint i = get_global_id(0);
    if (i >= count) return;
    uint remaining = i;
    uint offset = 0;
    for (uint axis = rank; axis > 0; axis--) {
        uint coordinate = remaining % meta[axis - 1];
        remaining /= meta[axis - 1];
        offset += (coordinate % meta[rank + axis - 1]) * meta[2 * rank + axis - 1];
    }
    dst[i] = src[offset];
}

kernel void clip(
    device const real* x [[buffer(0)]],
    device const real* low [[buffer(1)]],
    device const real* high [[buffer(2)]],
    device real* out [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    constant uint& has_low [[buffer(5)]],
    constant uint& has_high [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    uint i = get_global_id(0);
    if (i >= count) return;
    float v = x[i];
    if (has_low) v = fmax(v, (float)low[0]);
    if (has_high) v = fmin(v, (float)high[0]);
    out[i] = v;
}

kernel void scatter(
    device const real* updates [[buffer(0)]],
    device const uint* offsets [[buffer(1)]],
    device real* dst [[buffer(2)]],
    constant uint& slice [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    uint i = get_global_id(0);
    if (i >= count) return;
    dst[offsets[i / slice] + (i % slice)] = updates[i];
}

kernel void concatCopy(
    device const real* src [[buffer(0)]],
    device real* dst [[buffer(1)]],
    constant uint& src_axis [[buffer(2)]],
    constant uint& dst_axis [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    constant uint& dst_axis_offset [[buffer(5)]],
    constant uint& count [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    uint i = get_global_id(0);
    if (i >= count) return;
    uint inner_index = i % inner;
    uint axis_index = (i / inner) % src_axis;
    uint outer_index = i / (inner * src_axis);
    dst[(outer_index * dst_axis + dst_axis_offset + axis_index) * inner + inner_index] = src[i];
}

kernel void pad(
    device const real* src [[buffer(0)]],
    device real* dst [[buffer(1)]],
    device const uint* meta [[buffer(2)]],
    constant uint& rank [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    device const real* value [[buffer(5)]],
    constant uint& has_value [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    uint index = get_global_id(0);
    if (index >= count) return;
    uint remaining = index;
    uint source_offset = 0;
    int inside = 1;
    for (uint axis = rank; axis > 0; axis--) {
        uint coordinate = remaining % meta[axis - 1];
        remaining /= meta[axis - 1];
        uint before = meta[3 * rank + axis - 1];
        uint input_dim = meta[rank + axis - 1];
        if (coordinate < before || coordinate >= before + input_dim) {
            inside = 0;
        } else {
            source_offset += (coordinate - before) * meta[2 * rank + axis - 1];
        }
    }
    dst[index] = inside ? src[source_offset] : (has_value ? value[0] : 0.0f);
}

kernel void gather(
    device const real* src [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device real* dst [[buffer(2)]],
    constant uint& outer [[buffer(3)]],
    constant uint& axis [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    constant uint& index_count [[buffer(6)]],
    constant uint& count [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    uint i = get_global_id(0);
    if (i >= count) return;
    uint inner_index = i % inner;
    uint index = (i / inner) % index_count;
    uint outer_index = i / (inner * index_count);
    uint source = indices[index];
    dst[i] = src[(outer_index * axis + source) * inner + inner_index];
}

kernel void layerNorm(
    device const real* x [[buffer(0)]],
    device const real* scale [[buffer(1)]],
    device const real* bias [[buffer(2)]],
    device real* out [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant float& epsilon [[buffer(5)]],
    constant uint& has_bias [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    threadgroup float scratch[256];
    uint row = get_group_id(0);
    uint lane = get_local_id(0);
    uint width = get_local_size(0);
    uint base = row * cols;

    float sum = 0.0f;
    for (uint i = lane; i < cols; i += width) sum += x[base + i];
    float mean = reduceSumLocal(sum, scratch, lane, width) / (float)cols;

    float variance = 0.0f;
    for (uint i = lane; i < cols; i += width) {
        float d = x[base + i] - mean;
        variance += d * d;
    }
    float inverse = 1.0f / sqrt(reduceSumLocal(variance, scratch, lane, width) / (float)cols + epsilon);

    for (uint i = lane; i < cols; i += width) {
        float normalized = (x[base + i] - mean) * inverse;
        out[base + i] = normalized * scale[i] + (has_bias ? bias[i] : 0.0f);
    }
}

kernel void softmax(
    device const real* x [[buffer(0)]],
    device real* out [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    threadgroup float scratch[256];
    uint row = get_group_id(0);
    uint lane = get_local_id(0);
    uint width = get_local_size(0);
    uint base = row * cols;

    // The rows an attention softmax normalizes are thousands of elements wide
    // and the kernel walks each of them twice, so it is entirely a question of
    // how fast the row can be read. LANE_STEP at a time is how fast.
    float top = -INFINITY;
    float sum = 0.0f;
    if ((cols % LANE_STEP) == 0u) {
        device const realv* groups = (device const realv*)(x + base);
        for (uint i = lane; i < cols / LANE_STEP; i += width) {
            floatv v = TO_FLOATV(groups[i]);
            float highest = -INFINITY;
            for (int e = 0; e < LANE_STEP; e++) highest = fmax(highest, v[e]);
            if (highest > top) {
                sum = (top == -INFINITY) ? 0.0f : (sum * exp(top - highest));
                top = highest;
            }
            if (top != -INFINITY) {
                floatv scaled = exp(v - top);
                for (int e = 0; e < LANE_STEP; e++) {
                    if (v[e] != -INFINITY) sum += scaled[e];
                }
            }
        }
    } else {
        for (uint i = lane; i < cols; i += width) {
            float v = x[base + i];
            if (v > top) {
                sum = (top == -INFINITY) ? 0.0f : (sum * exp(top - v));
                top = v;
            }
            if (v != -INFINITY && top != -INFINITY) {
                sum += exp(v - top);
            }
        }
    }

    float peak = reduceMaxLocal(top, scratch, lane, width);
    float local_sum = (top == -INFINITY || peak == -INFINITY) ? 0.0f : (sum * exp(top - peak));
    float total_sum = reduceSumLocal(local_sum, scratch, lane, width);
    float scale = (total_sum > 0.0f) ? (1.0f / total_sum) : 0.0f;

    if ((cols % LANE_STEP) == 0u) {
        device const realv* groups = (device const realv*)(x + base);
        device realv* result = (device realv*)(out + base);
        for (uint i = lane; i < cols / LANE_STEP; i += width) {
            floatv v = TO_FLOATV(groups[i]);
            floatv res;
            for (int e = 0; e < LANE_STEP; e++) {
                float elem = v[e];
                res[e] = (elem == -INFINITY || peak == -INFINITY || scale == 0.0f) ? 0.0f : (exp(elem - peak) * scale);
            }
            result[i] = TO_REALV(res);
        }
    } else {
        for (uint i = lane; i < cols; i += width) {
            float v = x[base + i];
            out[base + i] = (v == -INFINITY || peak == -INFINITY || scale == 0.0f) ? 0.0f : (exp(v - peak) * scale);
        }
    }
}

#define MM_THREADS_X 16
#define MM_THREADS_Y 16
#define MM_ROWS 8
#define MM_COLS 4
#define MM_TILE_M (MM_THREADS_Y * MM_ROWS)
#define MM_TILE_N (MM_THREADS_X * MM_COLS)
#define MM_TILE_K 16
#define MM_STAGE (MM_THREADS_X * MM_THREADS_Y)

kernel void matmul(
    device const real* a [[buffer(0)]],
    device const real* b [[buffer(1)]],
    device real* c [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant uint& a_batch [[buffer(6)]],
    constant uint& b_batch [[buffer(7)]],
    constant uint& c_batch [[buffer(8)]],
    constant uint& b_transposed [[buffer(9)]],
    device const real* bias [[buffer(10)]],
    constant uint& has_bias [[buffer(11)]],
    constant uint& activation [[buffer(12)]],
    constant uint& block_m [[buffer(13)]],
    constant uint& block_n [[buffer(14)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    threadgroup float a_tile[MM_TILE_M][MM_TILE_K];
    threadgroup float b_tile[MM_TILE_K][MM_TILE_N];

    uint tx = get_local_id(0);
    uint ty = get_local_id(1);
    uint thread_index = ty * MM_THREADS_X + tx;
    // The same blocked walk `matmulXmx` takes, for the same reason.
    uint tiles_n = (n + MM_TILE_N - 1) / MM_TILE_N;
    uint tiles_m = (m + MM_TILE_M - 1) / MM_TILE_M;
    uint blocks_n = (tiles_n + block_n - 1) / block_n;
    uint tile = get_group_id(0);
    uint at = tile / (block_m * block_n);
    uint within = tile % (block_m * block_n);
    uint tile_m = (at / blocks_n) * block_m + within / block_n;
    uint tile_n = (at % blocks_n) * block_n + within % block_n;
    if (tile_m >= tiles_m || tile_n >= tiles_n) return;
    uint row_base = tile_m * MM_TILE_M;
    uint col_base = tile_n * MM_TILE_N;
    uint batch = get_group_id(2);
    uint a_base = batch * a_batch;
    uint b_base = batch * b_batch;

    float acc[MM_ROWS][MM_COLS];
    for (int i = 0; i < MM_ROWS; i++) {
        for (int j = 0; j < MM_COLS; j++) acc[i][j] = 0.0f;
    }

    for (uint step = 0; step < k; step += MM_TILE_K) {
        for (uint slab = 0; slab < (MM_TILE_M * MM_TILE_K) / MM_STAGE; slab++) {
            uint idx = thread_index + slab * MM_STAGE;
            uint r = idx / MM_TILE_K;
            uint column = idx % MM_TILE_K;
            a_tile[r][column] = (row_base + r < m && step + column < k) ?
                a[a_base + (row_base + r) * k + step + column] : 0.0f;
        }

        for (uint slab = 0; slab < (MM_TILE_K * MM_TILE_N) / MM_STAGE; slab++) {
            uint idx = thread_index + slab * MM_STAGE;
            uint r = idx / MM_TILE_N;
            uint column = idx % MM_TILE_N;
            float val = 0.0f;
            if (step + r < k && col_base + column < n) {
                val = b_transposed ?
                    b[b_base + (col_base + column) * k + step + r] :
                    b[b_base + (step + r) * n + col_base + column];
            }
            b_tile[r][column] = val;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (uint depth = 0; depth < MM_TILE_K; depth++) {
            float a_reg[MM_ROWS];
            float b_reg[MM_COLS];
            #pragma unroll
            for (int i = 0; i < MM_ROWS; i++) a_reg[i] = a_tile[ty + i * MM_THREADS_Y][depth];
            #pragma unroll
            for (int j = 0; j < MM_COLS; j++) b_reg[j] = b_tile[depth][tx + j * MM_THREADS_X];
            #pragma unroll
            for (int i = 0; i < MM_ROWS; i++) {
                #pragma unroll
                for (int j = 0; j < MM_COLS; j++) {
                    acc[i][j] = fma(a_reg[i], b_reg[j], acc[i][j]);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int i = 0; i < MM_ROWS; i++) {
        uint row = row_base + ty + i * MM_THREADS_Y;
        if (row < m) {
            for (int j = 0; j < MM_COLS; j++) {
                uint col = col_base + tx + j * MM_THREADS_X;
                if (col < n) {
                    float value = acc[i][j] + (has_bias ? (float)bias[col] : 0.0f);
                    c[batch * c_batch + row * n + col] = activation ? gelu(value) : value;
                }
            }
        }
    }
}

kernel void matmulNBits(
    device const real* a [[buffer(0)]],
    device const uchar* quantized [[buffer(1)]],
    device const real* scales [[buffer(2)]],
    device real* out [[buffer(3)]],
    constant uint& m [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant uint& k [[buffer(6)]],
    constant uint& block_shift [[buffer(7)]],
    constant uint& blocks_per_row [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    threadgroup float a_tile[MM_TILE_M][MM_TILE_K];
    threadgroup float b_tile[MM_TILE_K][MM_TILE_N];

    uint tx = get_local_id(0);
    uint ty = get_local_id(1);
    uint thread_index = ty * MM_THREADS_X + tx;
    uint row_base = get_group_id(1) * MM_TILE_M;
    uint col_base = get_group_id(0) * MM_TILE_N;
    uint shift = block_shift;
    uint block_mask = (1u << shift) - 1u;
    uint blob = 1u << (shift - 1u);
    uint row_stride = blocks_per_row * blob;

    float acc[MM_ROWS][MM_COLS];
    for (int i = 0; i < MM_ROWS; i++) {
        for (int j = 0; j < MM_COLS; j++) acc[i][j] = 0.0f;
    }

    for (uint step = 0; step < k; step += MM_TILE_K) {
        for (uint slab = 0; slab < (MM_TILE_M * MM_TILE_K) / MM_STAGE; slab++) {
            uint idx = thread_index + slab * MM_STAGE;
            uint r = idx / MM_TILE_K;
            uint column = idx % MM_TILE_K;
            a_tile[r][column] = (row_base + r < m && step + column < k) ?
                a[(row_base + r) * k + step + column] : 0.0f;
        }

        for (uint slab = 0; slab < (MM_TILE_K * MM_TILE_N) / MM_STAGE; slab++) {
            uint idx = thread_index + slab * MM_STAGE;
            uint depth = idx / MM_TILE_N;
            uint column = idx % MM_TILE_N;
            uint weight_row = col_base + column;
            uint along = step + depth;
            float val = 0.0f;
            if (weight_row < n && along < k) {
                uint block = along >> shift;
                uint within = along & block_mask;
                uchar byte_val = quantized[weight_row * row_stride + block * blob + (within >> 1)];
                uchar nibble = (within & 1) ? (byte_val >> 4) : (byte_val & 0x0f);
                val = ((float)nibble - 8.0f) * scales[weight_row * blocks_per_row + block];
            }
            b_tile[depth][column] = val;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (uint depth = 0; depth < MM_TILE_K; depth++) {
            float a_reg[MM_ROWS];
            float b_reg[MM_COLS];
            #pragma unroll
            for (int i = 0; i < MM_ROWS; i++) a_reg[i] = a_tile[ty + i * MM_THREADS_Y][depth];
            #pragma unroll
            for (int j = 0; j < MM_COLS; j++) b_reg[j] = b_tile[depth][tx + j * MM_THREADS_X];
            #pragma unroll
            for (int i = 0; i < MM_ROWS; i++) {
                #pragma unroll
                for (int j = 0; j < MM_COLS; j++) {
                    acc[i][j] = fma(a_reg[i], b_reg[j], acc[i][j]);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int i = 0; i < MM_ROWS; i++) {
        uint row = row_base + ty + i * MM_THREADS_Y;
        if (row < m) {
            for (int j = 0; j < MM_COLS; j++) {
                uint col = col_base + tx + j * MM_THREADS_X;
                if (col < n) out[row * n + col] = acc[i][j];
            }
        }
    }
}

kernel void conv2dGemm(
    device const real* x [[buffer(0)]],
    device const real* w [[buffer(1)]],
    device const real* bias [[buffer(2)]],
    device real* out [[buffer(3)]],
    device const uint* meta [[buffer(4)]],
    constant uint& m [[buffer(5)]],
    constant uint& n [[buffer(6)]],
    constant uint& k [[buffer(7)]],
    constant uint& has_bias [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    threadgroup float a_tile[MM_TILE_M][MM_TILE_K];
    threadgroup float b_tile[MM_TILE_K][MM_TILE_N];

    uint tx = get_local_id(0);
    uint ty = get_local_id(1);
    uint thread_index = ty * MM_THREADS_X + tx;
    uint row_base = get_group_id(1) * MM_TILE_M;
    uint col_base = get_group_id(0) * MM_TILE_N;
    uint batch = get_group_id(2);

    uint in_channels = meta[0];
    uint in_h = meta[1];
    uint in_w = meta[2];
    uint out_w = meta[4];
    uint kernel_w = meta[6];
    uint stride_h = meta[7];
    uint stride_w = meta[8];
    uint pad_h = meta[9];
    uint pad_w = meta[10];
    uint dilation_h = meta[11];
    uint dilation_w = meta[12];
    uint taps = meta[5] * kernel_w;

    uint x_base = batch * in_channels * in_h * in_w;

    float acc[MM_ROWS][MM_COLS];
    for (int i = 0; i < MM_ROWS; i++) {
        for (int j = 0; j < MM_COLS; j++) acc[i][j] = 0.0f;
    }

    for (uint step = 0; step < k; step += MM_TILE_K) {
        for (uint slab = 0; slab < (MM_TILE_M * MM_TILE_K) / MM_STAGE; slab++) {
            uint idx = thread_index + slab * MM_STAGE;
            uint r = idx / MM_TILE_K;
            uint column = idx % MM_TILE_K;
            a_tile[r][column] = (row_base + r < m && step + column < k) ?
                w[(row_base + r) * k + step + column] : 0.0f;
        }

        for (uint slab = 0; slab < (MM_TILE_K * MM_TILE_N) / MM_STAGE; slab++) {
            uint idx = thread_index + slab * MM_STAGE;
            uint r = idx / MM_TILE_N;
            uint column = idx % MM_TILE_N;
            uint tap = step + r;
            uint pixel = col_base + column;
            float val = 0.0f;
            if (tap < k && pixel < n) {
                uint channel = tap / taps;
                uint ky = (tap % taps) / kernel_w;
                uint kx = tap % kernel_w;
                uint y = (pixel / out_w) * stride_h + ky * dilation_h;
                uint p = (pixel % out_w) * stride_w + kx * dilation_w;
                if (y >= pad_h && p >= pad_w && y - pad_h < in_h && p - pad_w < in_w) {
                    val = x[x_base + (channel * in_h + y - pad_h) * in_w + p - pad_w];
                }
            }
            b_tile[r][column] = val;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (uint depth = 0; depth < MM_TILE_K; depth++) {
            float a_reg[MM_ROWS];
            float b_reg[MM_COLS];
            #pragma unroll
            for (int i = 0; i < MM_ROWS; i++) a_reg[i] = a_tile[ty + i * MM_THREADS_Y][depth];
            #pragma unroll
            for (int j = 0; j < MM_COLS; j++) b_reg[j] = b_tile[depth][tx + j * MM_THREADS_X];
            #pragma unroll
            for (int i = 0; i < MM_ROWS; i++) {
                #pragma unroll
                for (int j = 0; j < MM_COLS; j++) {
                    acc[i][j] = fma(a_reg[i], b_reg[j], acc[i][j]);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int i = 0; i < MM_ROWS; i++) {
        uint row = row_base + ty + i * MM_THREADS_Y;
        if (row < m) {
            float shift = has_bias ? bias[row] : 0.0f;
            for (int j = 0; j < MM_COLS; j++) {
                uint col = col_base + tx + j * MM_THREADS_X;
                if (col < n) out[batch * m * n + row * n + col] = acc[i][j] + shift;
            }
        }
    }
}

/// Scatters the product a non-overlapping transposed convolution reduces to
/// into the image it stands for, adding the bias on the way. Row
/// `(channel * kernel_h + dy) * kernel_w + dx` of the product holds every
/// pixel that lands at `(y * kernel_h + dy, x * kernel_w + dx)` of that
/// channel, so with the weight already in that order on the host the whole
/// operator is one matrix product -- the fast one, on the matrix engines --
/// and this pass. `convTranspose2dGemm` below does both at once instead, and
/// is what a driver without the fast product falls back to.
kernel void pixelShuffle(
    device const real* product [[buffer(0)]],
    device const real* bias [[buffer(1)]],
    device real* out [[buffer(2)]],
    constant uint& channels [[buffer(3)]],
    constant uint& height [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    constant uint& kernel_h [[buffer(6)]],
    constant uint& kernel_w [[buffer(7)]],
    constant uint& count [[buffer(8)]],
    constant uint& has_bias [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    uint i = get_global_id(0);
    if (i >= count) return;
    const uint out_w = width * kernel_w;
    const uint out_h = height * kernel_h;
    const uint pixels = height * width;
    uint x = i % out_w;
    uint rest = i / out_w;
    uint y = rest % out_h;
    rest /= out_h;
    uint channel = rest % channels;
    uint batch = rest / channels;
    uint row = (channel * kernel_h + (y % kernel_h)) * kernel_w + (x % kernel_w);
    uint at = (batch * channels * kernel_h * kernel_w + row) * pixels +
        (y / kernel_h) * width + (x / kernel_w);
    out[i] = (real)((float)product[at] + (has_bias ? (float)bias[channel] : 0.0f));
}

// Non-overlapping transposed convolution as GEMM plus pixel shuffle.
// A is the input viewed as [pixel, input_channel], B is the weight viewed as
// [input_channel, output_channel * kernel_h * kernel_w]. Both views are
// gathered from their native NCHW / ONNX layouts while staging each tile.
kernel void convTranspose2dGemm(
    device const real* x [[buffer(0)]],
    device const real* w [[buffer(1)]],
    device const real* bias [[buffer(2)]],
    device real* out [[buffer(3)]],
    constant uint& in_h [[buffer(4)]],
    constant uint& in_w [[buffer(5)]],
    constant uint& out_channels [[buffer(6)]],
    constant uint& out_h [[buffer(7)]],
    constant uint& out_w [[buffer(8)]],
    constant uint& kernel_h [[buffer(9)]],
    constant uint& kernel_w [[buffer(10)]],
    constant uint& m [[buffer(11)]],
    constant uint& n [[buffer(12)]],
    constant uint& k [[buffer(13)]],
    constant uint& has_bias [[buffer(14)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    threadgroup float a_tile[MM_TILE_M][MM_TILE_K];
    threadgroup float b_tile[MM_TILE_K][MM_TILE_N];

    uint tx = get_local_id(0);
    uint ty = get_local_id(1);
    uint thread_index = ty * MM_THREADS_X + tx;
    uint row_base = get_group_id(1) * MM_TILE_M;
    uint col_base = get_group_id(0) * MM_TILE_N;
    uint batch = get_group_id(2);
    uint taps = kernel_h * kernel_w;
    uint x_base = batch * k * m;
    uint out_base = batch * out_channels * out_h * out_w;

    float acc[MM_ROWS][MM_COLS];
    for (int i = 0; i < MM_ROWS; i++) {
        for (int j = 0; j < MM_COLS; j++) acc[i][j] = 0.0f;
    }

    for (uint step = 0; step < k; step += MM_TILE_K) {
        for (uint slab = 0; slab < (MM_TILE_M * MM_TILE_K) / MM_STAGE; slab++) {
            uint idx = thread_index + slab * MM_STAGE;
            uint pixel = row_base + idx / MM_TILE_K;
            uint channel = step + idx % MM_TILE_K;
            a_tile[idx / MM_TILE_K][idx % MM_TILE_K] = (pixel < m && channel < k) ?
                x[x_base + channel * m + pixel] : 0.0f;
        }

        for (uint slab = 0; slab < (MM_TILE_K * MM_TILE_N) / MM_STAGE; slab++) {
            uint idx = thread_index + slab * MM_STAGE;
            uint channel = step + idx / MM_TILE_N;
            uint feature = col_base + idx % MM_TILE_N;
            b_tile[idx / MM_TILE_N][idx % MM_TILE_N] = (channel < k && feature < n) ?
                w[channel * n + feature] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (uint depth = 0; depth < MM_TILE_K; depth++) {
            float a_reg[MM_ROWS];
            float b_reg[MM_COLS];
            #pragma unroll
            for (int i = 0; i < MM_ROWS; i++) a_reg[i] = a_tile[ty + i * MM_THREADS_Y][depth];
            #pragma unroll
            for (int j = 0; j < MM_COLS; j++) b_reg[j] = b_tile[depth][tx + j * MM_THREADS_X];
            #pragma unroll
            for (int i = 0; i < MM_ROWS; i++) {
                #pragma unroll
                for (int j = 0; j < MM_COLS; j++) acc[i][j] = fma(a_reg[i], b_reg[j], acc[i][j]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int i = 0; i < MM_ROWS; i++) {
        uint pixel = row_base + ty + i * MM_THREADS_Y;
        if (pixel < m) {
            uint in_y = pixel / in_w;
            uint in_x = pixel % in_w;
            for (int j = 0; j < MM_COLS; j++) {
                uint feature = col_base + tx + j * MM_THREADS_X;
                if (feature < n) {
                    uint channel = feature / taps;
                    uint tap = feature % taps;
                    uint out_y = in_y * kernel_h + tap / kernel_w;
                    uint out_x = in_x * kernel_w + tap % kernel_w;
                    out[out_base + (channel * out_h + out_y) * out_w + out_x] =
                        acc[i][j] + (has_bias ? bias[channel] : 0.0f);
                }
            }
        }
    }
}

kernel void conv2d(
    device const real* x [[buffer(0)]],
    device const real* w [[buffer(1)]],
    device const real* bias [[buffer(2)]],
    device real* out [[buffer(3)]],
    device const uint* meta [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    constant uint& has_bias [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    uint i = get_global_id(0);
    if (i >= count) return;

    uint in_channels = meta[0];
    uint in_h = meta[1];
    uint in_w = meta[2];
    uint out_channels = meta[3];
    uint out_h = meta[4];
    uint out_w = meta[5];
    uint kernel_h = meta[6];
    uint kernel_w = meta[7];
    uint stride_h = meta[8];
    uint stride_w = meta[9];
    uint pad_h = meta[10];
    uint pad_w = meta[11];
    uint dilation_h = meta[12];
    uint dilation_w = meta[13];
    uint groups = meta[14];

    uint out_x = i % out_w;
    uint out_y = (i / out_w) % out_h;
    uint channel = (i / (out_w * out_h)) % out_channels;
    uint batch = i / (out_w * out_h * out_channels);

    uint group_size = in_channels / groups;
    uint group = channel / (out_channels / groups);

    float sum = 0.0f;
    for (uint c_idx = 0; c_idx < group_size; c_idx++) {
        uint in_c = group * group_size + c_idx;
        for (uint ky = 0; ky < kernel_h; ky++) {
            uint in_y = out_y * stride_h + ky * dilation_h;
            if (in_y < pad_h) continue;
            uint y = in_y - pad_h;
            if (y >= in_h) continue;
            for (uint kx = 0; kx < kernel_w; kx++) {
                uint in_x = out_x * stride_w + kx * dilation_w;
                if (in_x < pad_w) continue;
                uint x_index = in_x - pad_w;
                if (x_index >= in_w) continue;

                float pixel = x[((batch * in_channels + in_c) * in_h + y) * in_w + x_index];
                float weight = w[((channel * group_size + c_idx) * kernel_h + ky) * kernel_w + kx];
                sum += pixel * weight;
            }
        }
    }

    out[i] = sum + (has_bias ? bias[channel] : 0.0f);
}

kernel void convTranspose2d(
    device const real* x [[buffer(0)]],
    device const real* w [[buffer(1)]],
    device const real* bias [[buffer(2)]],
    device real* out [[buffer(3)]],
    device const uint* meta [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    constant uint& has_bias [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    uint i = get_global_id(0);
    if (i >= count) return;

    uint in_channels = meta[0];
    uint in_h = meta[1];
    uint in_w = meta[2];
    uint out_channels = meta[3];
    uint out_h = meta[4];
    uint out_w = meta[5];
    uint kernel_h = meta[6];
    uint kernel_w = meta[7];
    uint stride_h = meta[8];
    uint stride_w = meta[9];
    uint pad_h = meta[10];
    uint pad_w = meta[11];

    uint out_x = i % out_w;
    uint out_y = (i / out_w) % out_h;
    uint channel = (i / (out_w * out_h)) % out_channels;
    uint batch = i / (out_w * out_h * out_channels);

    float sum = 0.0f;
    for (uint ky = 0; ky < kernel_h; ky++) {
        uint shifted_y = out_y + pad_h;
        if (shifted_y < ky) continue;
        uint numerator_y = shifted_y - ky;
        if (numerator_y % stride_h != 0) continue;
        uint in_y = numerator_y / stride_h;
        if (in_y >= in_h) continue;

        for (uint kx = 0; kx < kernel_w; kx++) {
            uint shifted_x = out_x + pad_w;
            if (shifted_x < kx) continue;
            uint numerator_x = shifted_x - kx;
            if (numerator_x % stride_w != 0) continue;
            uint in_x = numerator_x / stride_w;
            if (in_x >= in_w) continue;

            for (uint c_idx = 0; c_idx < in_channels; c_idx++) {
                float pixel = x[((batch * in_channels + c_idx) * in_h + in_y) * in_w + in_x];
                float weight = w[((c_idx * out_channels + channel) * kernel_h + ky) * kernel_w + kx];
                sum += pixel * weight;
            }
        }
    }

    out[i] = sum + (has_bias ? bias[channel] : 0.0f);
}

kernel void cumulativeSum(
    device const real* src [[buffer(0)]],
    device real* dst [[buffer(1)]],
    constant uint& along [[buffer(2)]],
    constant uint& inner [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    uint i = get_global_id(0);
    if (i >= count) return;
    uint base = (i / inner) * along * inner + (i % inner);
    float total = 0.0f;
    for (uint step = 0; step < along; step++) {
        total += src[base + step * inner];
        dst[base + step * inner] = total;
    }
}

kernel void maxPool2d(
    device const real* src [[buffer(0)]],
    device real* dst [[buffer(1)]],
    device const uint* meta [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    uint i = get_global_id(0);
    if (i >= count) return;

    uint in_h = meta[0];
    uint in_w = meta[1];
    uint out_h = meta[2];
    uint out_w = meta[3];
    uint kernel_h = meta[4];
    uint kernel_w = meta[5];
    uint stride_h = meta[6];
    uint stride_w = meta[7];
    uint pad_h = meta[8];
    uint pad_w = meta[9];

    uint out_x = i % out_w;
    uint out_y = (i / out_w) % out_h;
    uint plane = i / (out_w * out_h);

    float top = -3.4e38f;
    for (uint ky = 0; ky < kernel_h; ky++) {
        uint y = out_y * stride_h + ky;
        if (y < pad_h) continue;
        uint iy = y - pad_h;
        if (iy >= in_h) continue;
        for (uint kx = 0; kx < kernel_w; kx++) {
            uint x = out_x * stride_w + kx;
            if (x < pad_w) continue;
            uint ix = x - pad_w;
            if (ix >= in_w) continue;
            top = fmax(top, (float)src[(plane * in_h + iy) * in_w + ix]);
        }
    }
    dst[i] = top;
}

kernel void sumAxes(
    device const real* src [[buffer(0)]],
    device real* dst [[buffer(1)]],
    device const uint* meta [[buffer(2)]],
    constant uint& rank [[buffer(3)]],
    constant uint& reduced [[buffer(4)]],
    constant uint& swept [[buffer(5)]],
    constant uint& count [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    uint i = get_global_id(0);
    if (i >= count) return;
    uint base = offsetOf(i, meta, 0, rank, rank);
    float total = 0.0f;
    for (uint step = 0; step < swept; step++) {
        total += src[base + offsetOf(step, meta, 2 * rank, 2 * rank + reduced, reduced)];
    }
    dst[i] = total;
}

kernel void resizeNearest(
    device const real* src [[buffer(0)]],
    device real* dst [[buffer(1)]],
    device const uint* meta [[buffer(2)]],
    constant uint& rank [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    uint i = get_global_id(0);
    if (i >= count) return;
    uint remaining = i;
    uint offset = 0;
    for (uint axis = rank; axis > 0; axis--) {
        uint dim = meta[axis - 1];
        uint coordinate = remaining % dim;
        remaining /= dim;
        offset += (coordinate * meta[rank + axis - 1] / dim) * meta[2 * rank + axis - 1];
    }
    dst[i] = src[offset];
}

kernel void instanceNorm(
    device const real* x [[buffer(0)]],
    device const real* scale [[buffer(1)]],
    device const real* bias [[buffer(2)]],
    device real* out [[buffer(3)]],
    constant uint& channels [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant float& epsilon [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    threadgroup float scratch[256];
    uint row = get_group_id(0);
    uint lane = get_local_id(0);
    uint width = get_local_size(0);
    uint base = row * cols;

    float sum = 0.0f;
    for (uint i = lane; i < cols; i += width) sum += x[base + i];
    float mean = reduceSumLocal(sum, scratch, lane, width) / (float)cols;

    float variance = 0.0f;
    for (uint i = lane; i < cols; i += width) {
        float d = x[base + i] - mean;
        variance += d * d;
    }
    float inverse = 1.0f / sqrt(reduceSumLocal(variance, scratch, lane, width) / (float)cols + epsilon);

    float gain = scale[row % channels];
    float shift = bias[row % channels];
    for (uint i = lane; i < cols; i += width) {
        out[base + i] = (x[base + i] - mean) * inverse * gain + shift;
    }
}


// ---------------------------------------------------------------------------
// Matrix multiplication on the Apple GPU matrix units
//
// Every SIMD group on an Apple GPU can multiply a pair of 8x8 matrices in one
// instruction, the same hardware the Xe kernels reach through DPAS. This is
// `matmul` rearranged around that: the staged tiles are the same, but the
// innermost loop hands whole 8x8 blocks to `simdgroup_multiply_accumulate`
// instead of walking them one fused multiply-add at a time. Accumulation stays
// in float -- k runs into the thousands here and a half would not hold the sum.

#if SAM3_HALF
typedef simdgroup_half8x8 simdgroup_real8x8;
#else
typedef simdgroup_float8x8 simdgroup_real8x8;
#endif

// One work group covers a 64x64 tile of C with eight SIMD groups laid out two
// deep and four across, so each of them holds a 32x16 corner of it as eight
// 8x8 accumulators.
#define SG_TILE_M 64
#define SG_TILE_N 64
#define SG_TILE_K 32
#define SG_ROWS 2
#define SG_COLS 4
#define SG_ACC_M 4
#define SG_ACC_N 2
#define SG_GROUPS (SG_ROWS * SG_COLS)
#define SG_THREADS (SG_GROUPS * 32)

kernel void matmulSimd(
    device const real* a [[buffer(0)]],
    device const real* b [[buffer(1)]],
    device real* c [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant uint& a_batch [[buffer(6)]],
    constant uint& b_batch [[buffer(7)]],
    constant uint& c_batch [[buffer(8)]],
    constant uint& b_transposed [[buffer(9)]],
    device const real* bias [[buffer(10)]],
    constant uint& has_bias [[buffer(11)]],
    constant uint& activation [[buffer(12)]],
    constant uint& block_m [[buffer(13)]],
    constant uint& block_n [[buffer(14)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup real a_tile[SG_TILE_M * SG_TILE_K];
    threadgroup real b_tile[SG_TILE_K * SG_TILE_N];
    // Where the accumulators are read back one 8x8 at a time, a slot each so
    // that no SIMD group waits on another.
    threadgroup float stage[SG_GROUPS * 64];

    // The same blocked walk `matmul` takes, for the same reason.
    uint tiles_n = (n + SG_TILE_N - 1) / SG_TILE_N;
    uint tiles_m = (m + SG_TILE_M - 1) / SG_TILE_M;
    uint blocks_n = (tiles_n + block_n - 1) / block_n;
    uint tile = group_id.x;
    uint at = tile / (block_m * block_n);
    uint within = tile % (block_m * block_n);
    uint tile_m = (at / blocks_n) * block_m + within / block_n;
    uint tile_n = (at % blocks_n) * block_n + within % block_n;
    if (tile_m >= tiles_m || tile_n >= tiles_n) return;
    uint row_base = tile_m * SG_TILE_M;
    uint col_base = tile_n * SG_TILE_N;
    uint batch = group_id.z;
    uint a_base = batch * a_batch;
    uint b_base = batch * b_batch;

    uint sg_row = (simd_group / SG_COLS) * (SG_ACC_M * 8);
    uint sg_col = (simd_group % SG_COLS) * (SG_ACC_N * 8);

    simdgroup_float8x8 acc[SG_ACC_M][SG_ACC_N];
    for (uint i = 0; i < SG_ACC_M; i++) {
        for (uint j = 0; j < SG_ACC_N; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    for (uint step = 0; step < k; step += SG_TILE_K) {
        for (uint slab = 0; slab < (SG_TILE_M * SG_TILE_K) / SG_THREADS; slab++) {
            uint idx = thread_index + slab * SG_THREADS;
            uint r = idx / SG_TILE_K;
            uint column = idx % SG_TILE_K;
            a_tile[r * SG_TILE_K + column] = (row_base + r < m && step + column < k) ?
                a[a_base + (row_base + r) * k + step + column] : (real)0;
        }

        for (uint slab = 0; slab < (SG_TILE_K * SG_TILE_N) / SG_THREADS; slab++) {
            uint idx = thread_index + slab * SG_THREADS;
            // A transposed right operand runs along k, so walk the tile that
            // way as well and neighbouring lanes still read neighbouring
            // addresses.
            uint r = b_transposed ? (idx % SG_TILE_K) : (idx / SG_TILE_N);
            uint column = b_transposed ? (idx / SG_TILE_K) : (idx % SG_TILE_N);
            real value = 0;
            if (step + r < k && col_base + column < n) {
                value = b_transposed ?
                    b[b_base + (col_base + column) * k + step + r] :
                    b[b_base + (step + r) * n + col_base + column];
            }
            b_tile[r * SG_TILE_N + column] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint depth = 0; depth < SG_TILE_K; depth += 8) {
            simdgroup_real8x8 a_frag[SG_ACC_M];
            simdgroup_real8x8 b_frag[SG_ACC_N];
            for (uint i = 0; i < SG_ACC_M; i++)
                simdgroup_load(a_frag[i], a_tile + (sg_row + i * 8) * SG_TILE_K + depth, SG_TILE_K);
            for (uint j = 0; j < SG_ACC_N; j++)
                simdgroup_load(b_frag[j], b_tile + depth * SG_TILE_N + sg_col + j * 8, SG_TILE_N);
            for (uint i = 0; i < SG_ACC_M; i++) {
                for (uint j = 0; j < SG_ACC_N; j++)
                    simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* slot = stage + simd_group * 64;
    for (uint i = 0; i < SG_ACC_M; i++) {
        for (uint j = 0; j < SG_ACC_N; j++) {
            simdgroup_store(acc[i][j], slot, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < 64; e += 32) {
                uint row = row_base + sg_row + i * 8 + e / 8;
                uint col = col_base + sg_col + j * 8 + e % 8;
                if (row < m && col < n) {
                    float value = slot[e] + (has_bias ? (float)bias[col] : 0.0f);
                    c[batch * c_batch + row * n + col] = (real)(activation ? gelu(value) : value);
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// ---------------------------------------------------------------------------
// Matrix multiplication on the neural accelerators
//
// From the M5 on, each GPU core carries a matrix unit of its own, and Metal 4
// reaches it through the tensor operations rather than through the SIMD group
// matrices above -- which, on the GPUs before it, were themselves built out of
// ordinary multiply-adds. This is the same product again written against
// `matmul2d`, and it runs about six times faster than the kernel above.
//
// A compiler without the tensor types leaves no symbol behind, so a Mac whose
// GPU predates them keeps `matmulSimd`.

#if __METAL_VERSION__ >= 400 && defined(__HAVE_TENSOR__)

#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// A work group of four SIMD groups covers a 128x64 tile of C. The operation
// stages the operands itself, so unlike the kernels above there is nothing to
// size against threadgroup memory; this is simply the tile that measured
// fastest over the shapes these graphs ask for.
#define TN_TILE_M 128
#define TN_TILE_N 64
#define TN_GROUPS 4

/// One tile of C, with the bias and the activation folded into the
/// accumulators before they are written down. Where there is neither, the
/// operation writes to C itself and nothing is staged at all.
template <typename Op, typename TA, typename TB, typename TC>
static void tensorTile(
    thread Op& op,
    thread TA& tA,
    thread TB& tB,
    thread TC& tC,
    device const real* bias,
    uint has_bias,
    uint activation,
    uint col_base,
    uint n
) {
    if (has_bias == 0 && activation == 0) {
        op.run(tA, tB, tC);
        return;
    }
    // The accumulators as the operation hands them out: thread private, in an
    // order only `get_multidimensional_index` knows, which is enough to find
    // the column each one belongs to.
    auto acc = op.template get_destination_cooperative_tensor<TA, TB, real>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < acc.get_capacity(); i++) {
        if (acc.is_valid_element(i)) acc[i] = (real)0;
    }
    op.run(tA, tB, acc);
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < acc.get_capacity(); i++) {
        if (!acc.is_valid_element(i)) continue;
        uint col = col_base + acc.get_multidimensional_index(i)[0];
        float value = (float)acc[i] + ((has_bias != 0 && col < n) ? (float)bias[col] : 0.0f);
        acc[i] = (real)(activation != 0 ? gelu(value) : value);
    }
    acc.store(tC);
}

kernel void matmulTensor(
    device const real* a [[buffer(0)]],
    device const real* b [[buffer(1)]],
    device real* c [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant uint& a_batch [[buffer(6)]],
    constant uint& b_batch [[buffer(7)]],
    constant uint& c_batch [[buffer(8)]],
    constant uint& b_transposed [[buffer(9)]],
    device const real* bias [[buffer(10)]],
    constant uint& has_bias [[buffer(11)]],
    constant uint& activation [[buffer(12)]],
    constant uint& block_m [[buffer(13)]],
    constant uint& block_n [[buffer(14)]],
    uint3 group_id [[threadgroup_position_in_grid]]
) {
    // The same blocked walk `matmul` takes, for the same reason.
    uint tiles_n = (n + TN_TILE_N - 1) / TN_TILE_N;
    uint tiles_m = (m + TN_TILE_M - 1) / TN_TILE_M;
    uint blocks_n = (tiles_n + block_n - 1) / block_n;
    uint tile = group_id.x;
    uint at = tile / (block_m * block_n);
    uint within = tile % (block_m * block_n);
    uint tile_m = (at / blocks_n) * block_m + within / block_n;
    uint tile_n = (at % blocks_n) * block_n + within % block_n;
    if (tile_m >= tiles_m || tile_n >= tiles_n) return;
    uint row_base = tile_m * TN_TILE_M;
    uint col_base = tile_n * TN_TILE_N;
    uint batch = group_id.z;

    // A tensor's fastest axis is its first, so an m by k operand is (k, m).
    // The batch is a shift of the base address rather than a third axis, since
    // the graph's batches are contiguous and the product is over two of them.
    auto A = tensor(const_cast<device real*>(a) + batch * a_batch, dextents<int32_t, 2>(k, m));
    auto C = tensor(c + batch * c_batch, dextents<int32_t, 2>(n, m));
    auto tA = A.slice(0, row_base);
    auto tC = C.slice(col_base, row_base);

    // Slicing against the whole tensor is what bounds the edge tiles: the
    // operation reads and writes only what the extents allow.
    if (b_transposed != 0) {
        constexpr auto descriptor = matmul2d_descriptor(
            TN_TILE_M, TN_TILE_N, static_cast<int>(dynamic_extent), false, true);
        matmul2d<descriptor, execution_simdgroups<TN_GROUPS>> op;
        auto B = tensor(const_cast<device real*>(b) + batch * b_batch, dextents<int32_t, 2>(k, n));
        auto tB = B.slice(0, col_base);
        tensorTile(op, tA, tB, tC, bias, has_bias, activation, col_base, n);
    } else {
        constexpr auto descriptor = matmul2d_descriptor(
            TN_TILE_M, TN_TILE_N, static_cast<int>(dynamic_extent), false, false);
        matmul2d<descriptor, execution_simdgroups<TN_GROUPS>> op;
        auto B = tensor(const_cast<device real*>(b) + batch * b_batch, dextents<int32_t, 2>(n, k));
        auto tB = B.slice(col_base, 0);
        tensorTile(op, tA, tB, tC, bias, has_bias, activation, col_base, n);
    }
}

// A convolution is the same product over a window that is gathered rather than
// stored, so the operand the graph does not hold has to be built somewhere the
// operation can read it: `matmul2d` takes threadgroup tensors as well as device
// ones, and one im2col tile at a time fits in threadgroup memory. That leaves
// the accumulator to carry across the depth, which a cooperative tensor does.
//
// A work group of eight SIMD groups covers a 128x64 tile of the output, 64 taps
// of depth at a time.
#define TC_TILE_M 128
#define TC_TILE_N 64
#define TC_TILE_K 64
#define TC_GROUPS 8
#define TC_THREADS (TC_GROUPS * 32)

kernel void conv2dGemmTensor(
    device const real* x [[buffer(0)]],
    device const real* w [[buffer(1)]],
    device const real* bias [[buffer(2)]],
    device real* out [[buffer(3)]],
    device const uint* meta [[buffer(4)]],
    constant uint& m [[buffer(5)]],
    constant uint& n [[buffer(6)]],
    constant uint& k [[buffer(7)]],
    constant uint& has_bias [[buffer(8)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]
) {
    threadgroup real window[TC_TILE_K * TC_TILE_N];

    uint row_base = group_id.y * TC_TILE_M;
    uint col_base = group_id.x * TC_TILE_N;
    uint batch = group_id.z;

    uint in_channels = meta[0];
    uint in_h = meta[1];
    uint in_w = meta[2];
    uint out_w = meta[4];
    uint kernel_w = meta[6];
    uint stride_h = meta[7];
    uint stride_w = meta[8];
    uint pad_h = meta[9];
    uint pad_w = meta[10];
    uint dilation_h = meta[11];
    uint dilation_w = meta[12];
    uint taps = meta[5] * kernel_w;
    uint x_base = batch * in_channels * in_h * in_w;

    constexpr auto descriptor = matmul2d_descriptor(
        TC_TILE_M, TC_TILE_N, TC_TILE_K, false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<TC_GROUPS>> op;

    auto weights = tensor(const_cast<device real*>(w), dextents<int32_t, 2>(k, m));
    auto staged = tensor((threadgroup real*)window, dextents<int32_t, 2>(TC_TILE_N, TC_TILE_K));
    auto image = tensor(out + batch * m * n, dextents<int32_t, 2>(n, m));
    auto tW = weights.slice(0, row_base);
    auto tOut = image.slice(col_base, row_base);

    // Float, because the depth runs into the thousands and the accumulator is
    // read back and written down once for every tile of it.
    auto acc = op.get_destination_cooperative_tensor<decltype(tW), decltype(staged), float>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < acc.get_capacity(); i++) {
        if (acc.is_valid_element(i)) acc[i] = 0.0f;
    }

    // Which pixel a lane stages is fixed for the life of the thread, and so is
    // where that pixel's window starts in the image. Hoisting it out of the
    // depth loop is most of what this kernel is: the gather costs five integer
    // divisions an element, and a thread stages a tile's worth every step.
    uint column = thread_index % TC_TILE_N;
    uint pixel = col_base + column;
    uint origin_y = (pixel / out_w) * stride_h;
    uint origin_x = (pixel % out_w) * stride_w;
    bool within = pixel < n;
    uint row_first = thread_index / TC_TILE_N;
    uint row_step = TC_THREADS / TC_TILE_N;

    // The other divisions turn a tap into a channel and a position in the
    // window, and depend on nothing else. There are TC_TILE_K of them to a step.
    threadgroup uint tap_plane[TC_TILE_K];
    threadgroup uint tap_y[TC_TILE_K];
    threadgroup uint tap_x[TC_TILE_K];

    for (uint step = 0; step < k; step += TC_TILE_K) {
        // Retires the reads the previous step's product ended with.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (thread_index < TC_TILE_K) {
            uint tap = step + thread_index;
            tap_plane[thread_index] = (tap / taps) * in_h * in_w;
            tap_y[thread_index] = ((tap % taps) / kernel_w) * dilation_h;
            tap_x[thread_index] = (tap % kernel_w) * dilation_w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint r = row_first; r < TC_TILE_K; r += row_step) {
            real value = 0;
            if (within && step + r < k) {
                uint y = origin_y + tap_y[r];
                uint p = origin_x + tap_x[r];
                if (y >= pad_h && p >= pad_w && y - pad_h < in_h && p - pad_w < in_w) {
                    value = x[x_base + tap_plane[r] + (y - pad_h) * in_w + p - pad_w];
                }
            }
            window[r * TC_TILE_N + column] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto tile = weights.slice(step, row_base);
        op.run(tile, staged, acc);
    }

    // The bias is per output channel here, so it is the row that finds it.
    // Narrowing through a second cooperative tensor rather than writing the
    // elements out one at a time keeps the operation's own store.
    auto narrowed = op.get_destination_cooperative_tensor<decltype(tW), decltype(staged), real>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < acc.get_capacity(); i++) {
        if (!acc.is_valid_element(i)) continue;
        uint row = row_base + acc.get_multidimensional_index(i)[1];
        narrowed[i] = (real)(acc[i] + ((has_bias != 0 && row < m) ? (float)bias[row] : 0.0f));
    }
    narrowed.store(tOut);
}

#endif
