//! Every operator that runs on the OpenCL / Intel GPU.
//! Compiled at runtime by the OpenCL driver JIT compiler.

// What a tensor is stored as. `real` is half when the runtime defines
// SAM3_HALF, which halves the bytes every operator here moves -- and all but
// the matrix product are bound by exactly that. Arithmetic still happens in
// float: a reduction that accumulated in half would drift, and the conversion
// is free next to the load it saves.
//
// `LANE_STEP` is how many of them one work item of a vector kernel carries.
// Four, whichever the element is: a half build gains nothing from eight, which
// costs registers and halves the work items without moving more per cycle.
// `runtime.lane_step` mirrors it.
#if SAM3_HALF
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
typedef half real;
typedef half4 realv;
typedef float4 floatv;
#define LANE_STEP 4
#define TO_REALV(v) convert_half4(v)
#define TO_FLOATV(v) convert_float4(v)
#define VLOADV vload4
#define VSTOREV vstore4
#else
typedef float real;
typedef float4 realv;
typedef float4 floatv;
#define LANE_STEP 4
#define TO_REALV(v) (v)
#define TO_FLOATV(v) (v)
#define VLOADV vload4
#define VSTOREV vstore4
#endif

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

inline uint offsetOf(uint index, __global const uint* meta, uint dims_at, uint strides_at, uint rank) {
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
#define fastExp(x) native_exp(x)

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
    return sign * (1.0f - poly * fastExp(-v * v));
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
#if defined(cl_khr_subgroups)
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#define REDUCE_BY_SUB_GROUP 1
#endif

#if REDUCE_BY_SUB_GROUP
/// The leading barrier is what makes `scratch` reusable between two reductions
/// in a row: it retires the reads the previous one ended with.
#define REDUCE_ACROSS_SUB_GROUPS(combine, value, scratch, identity)         \
    do {                                                                    \
        float within = value;                                               \
        uint groups = get_num_sub_groups();                                 \
        if (groups == 1) return within;                                     \
        barrier(CLK_LOCAL_MEM_FENCE);                                       \
        if (get_sub_group_local_id() == 0) scratch[get_sub_group_id()] = within; \
        barrier(CLK_LOCAL_MEM_FENCE);                                       \
        float folded = identity;                                            \
        for (uint g = 0; g < groups; g++) folded = combine(folded, scratch[g]); \
        return folded;                                                      \
    } while (0)

inline float addf(float a, float b) { return a + b; }

inline float reduceSumLocal(float val, __local float* scratch, uint lane, uint block_size) {
    (void)lane;
    (void)block_size;
    REDUCE_ACROSS_SUB_GROUPS(addf, sub_group_reduce_add(val), scratch, 0.0f);
}

inline float reduceMaxLocal(float val, __local float* scratch, uint lane, uint block_size) {
    (void)lane;
    (void)block_size;
    REDUCE_ACROSS_SUB_GROUPS(fmax, sub_group_reduce_max(val), scratch, -INFINITY);
}
#else
inline float reduceSumLocal(float val, __local float* scratch, uint lane, uint block_size) {
    scratch[lane] = val;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = block_size / 2; stride > 0; stride /= 2) {
        if (lane < stride) scratch[lane] += scratch[lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float total = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return total;
}

inline float reduceMaxLocal(float val, __local float* scratch, uint lane, uint block_size) {
    scratch[lane] = val;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = block_size / 2; stride > 0; stride /= 2) {
        if (lane < stride) scratch[lane] = fmax(scratch[lane], scratch[lane + stride]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float peak = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return peak;
}
#endif

__kernel void binary(
    __global const real* a,
    __global const real* b,
    __global real* out,
    __global const uint* meta,
    uint rank,
    uint count,
    uint op,
    uint a_period,
    uint b_period
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

__kernel void unary(
    __global const real* x,
    __global real* out,
    uint count,
    uint op
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
inline floatv groupOf(__global const real* p, uint index, uint period) {
    return period ? TO_FLOATV(VLOADV(wrap(index, period), p)) : (floatv)((float)p[0]);
}

__kernel void binaryVec(
    __global const real* a,
    __global const real* b,
    __global real* out,
    uint groups,
    uint op,
    uint a_period,
    uint b_period
) {
    uint i = get_global_id(0);
    if (i >= groups) return;
    floatv x = groupOf(a, i, a_period);
    floatv y = groupOf(b, i, b_period);
    floatv res = (floatv)(0.0f);
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
    floatv magnitude = 1.0f - poly * fastExp(-v * v);
    return select(magnitude, -magnitude, x < 0.0f);
}

__kernel void unaryVec(
    __global const real* x,
    __global real* out,
    uint groups,
    uint op
) {
    uint i = get_global_id(0);
    if (i >= groups) return;
    floatv v = TO_FLOATV(VLOADV(i, x));
    floatv res = (floatv)(0.0f);
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
__kernel void rope(
    __global const real* x,
    __global const real* cosine,
    __global const real* sine,
    __global real* out,
    uint count,
    uint period,
    float scale
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
__kernel void ropeVec(
    __global const real* x,
    __global const real* cosine,
    __global const real* sine,
    __global real* out,
    uint groups,
    uint table_groups,
    float scale
) {
    uint i = get_global_id(0);
    if (i >= groups) return;
    uint table = table_groups ? wrap(i, table_groups) : 0;
    floatv v = TO_FLOATV(VLOADV(i, x));
    floatv turned = (floatv)(-v.s1, v.s0, -v.s3, v.s2);
    floatv c = table_groups ? TO_FLOATV(VLOADV(table, cosine)) : (floatv)((float)cosine[0]);
    floatv s = table_groups ? TO_FLOATV(VLOADV(table, sine)) : (floatv)((float)sine[0]);
    VSTOREV(TO_REALV((v * c + turned * s) * scale), i, out);
}

__kernel void copy(
    __global const real* src,
    __global real* dst,
    __global const uint* meta,
    uint rank,
    uint count,
    uint src_offset,
    uint dst_offset
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
__kernel void copyVec(
    __global const real* src,
    __global real* dst,
    __global const uint* meta,
    uint rank,
    uint groups,
    uint src_offset,
    uint dst_offset
) {
    uint i = get_global_id(0);
    if (i >= groups) return;
    realv group = VLOADV(0, &src[src_offset + offsetOf(i, meta, 0, rank, rank)]);
    VSTOREV(group, i, &dst[dst_offset]);
}

__kernel void fill(
    __global real* dst,
    float value,
    uint count
) {
    uint i = get_global_id(0);
    if (i >= count) return;
    dst[i] = value;
}

__kernel void select_kernel(
    __global const real* condition,
    __global const real* a,
    __global const real* b,
    __global real* out,
    __global const uint* meta,
    uint rank,
    uint count
) {
    uint i = get_global_id(0);
    if (i >= count) return;
    float cond = condition[offsetOf(i, meta, 0, rank, rank)];
    out[i] = (cond != 0.0f) ? a[offsetOf(i, meta, 0, 2 * rank, rank)] : b[offsetOf(i, meta, 0, 3 * rank, rank)];
}

__kernel void tile(
    __global const real* src,
    __global real* dst,
    __global const uint* meta,
    uint rank,
    uint count
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

__kernel void clip(
    __global const real* x,
    __global const real* low,
    __global const real* high,
    __global real* out,
    uint count,
    uint has_low,
    uint has_high
) {
    uint i = get_global_id(0);
    if (i >= count) return;
    float v = x[i];
    if (has_low) v = fmax(v, (float)low[0]);
    if (has_high) v = fmin(v, (float)high[0]);
    out[i] = v;
}

__kernel void scatter(
    __global const real* updates,
    __global const uint* offsets,
    __global real* dst,
    uint slice,
    uint count
) {
    uint i = get_global_id(0);
    if (i >= count) return;
    dst[offsets[i / slice] + (i % slice)] = updates[i];
}

__kernel void concatCopy(
    __global const real* src,
    __global real* dst,
    uint src_axis,
    uint dst_axis,
    uint inner,
    uint dst_axis_offset,
    uint count
) {
    uint i = get_global_id(0);
    if (i >= count) return;
    uint inner_index = i % inner;
    uint axis_index = (i / inner) % src_axis;
    uint outer_index = i / (inner * src_axis);
    dst[(outer_index * dst_axis + dst_axis_offset + axis_index) * inner + inner_index] = src[i];
}

__kernel void pad(
    __global const real* src,
    __global real* dst,
    __global const uint* meta,
    uint rank,
    uint count,
    __global const real* value,
    uint has_value
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

__kernel void gather(
    __global const real* src,
    __global const uint* indices,
    __global real* dst,
    uint outer,
    uint axis,
    uint inner,
    uint index_count,
    uint count
) {
    uint i = get_global_id(0);
    if (i >= count) return;
    uint inner_index = i % inner;
    uint index = (i / inner) % index_count;
    uint outer_index = i / (inner * index_count);
    uint source = indices[index];
    dst[i] = src[(outer_index * axis + source) * inner + inner_index];
}

__kernel void layerNorm(
    __global const real* x,
    __global const real* scale,
    __global const real* bias,
    __global real* out,
    uint cols,
    float epsilon,
    uint has_bias
) {
    __local float scratch[256];
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

__kernel void softmax(
    __global const real* x,
    __global real* out,
    uint cols
) {
    __local float scratch[256];
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
        __global const realv* groups = (__global const realv*)(x + base);
        for (uint i = lane; i < cols / LANE_STEP; i += width) {
            floatv v = TO_FLOATV(groups[i]);
            float highest = -INFINITY;
            for (int e = 0; e < LANE_STEP; e++) highest = fmax(highest, ((float*)&v)[e]);
            if (highest > top) {
                sum = (top == -INFINITY) ? 0.0f : (sum * fastExp(top - highest));
                top = highest;
            }
            if (top != -INFINITY) {
                floatv scaled = fastExp(v - top);
                for (int e = 0; e < LANE_STEP; e++) {
                    if (((float*)&v)[e] != -INFINITY) sum += ((float*)&scaled)[e];
                }
            }
        }
    } else {
        for (uint i = lane; i < cols; i += width) {
            float v = x[base + i];
            if (v > top) {
                sum = (top == -INFINITY) ? 0.0f : (sum * fastExp(top - v));
                top = v;
            }
            if (v != -INFINITY && top != -INFINITY) {
                sum += fastExp(v - top);
            }
        }
    }

    float peak = reduceMaxLocal(top, scratch, lane, width);
    float local_sum = (top == -INFINITY || peak == -INFINITY) ? 0.0f : (sum * fastExp(top - peak));
    float total_sum = reduceSumLocal(local_sum, scratch, lane, width);
    float scale = (total_sum > 0.0f) ? (1.0f / total_sum) : 0.0f;

    if ((cols % LANE_STEP) == 0u) {
        __global const realv* groups = (__global const realv*)(x + base);
        __global realv* result = (__global realv*)(out + base);
        for (uint i = lane; i < cols / LANE_STEP; i += width) {
            floatv v = TO_FLOATV(groups[i]);
            floatv res;
            for (int e = 0; e < LANE_STEP; e++) {
                float elem = ((float*)&v)[e];
                ((float*)&res)[e] = (elem == -INFINITY || peak == -INFINITY || scale == 0.0f) ? 0.0f : (fastExp(elem - peak) * scale);
            }
            result[i] = TO_REALV(res);
        }
    } else {
        for (uint i = lane; i < cols; i += width) {
            float v = x[base + i];
            out[base + i] = (v == -INFINITY || peak == -INFINITY || scale == 0.0f) ? 0.0f : (fastExp(v - peak) * scale);
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

__kernel void matmul(
    __global const real* a,
    __global const real* b,
    __global real* c,
    uint m,
    uint n,
    uint k,
    uint a_batch,
    uint b_batch,
    uint c_batch,
    uint b_transposed,
    __global const real* bias,
    uint has_bias,
    uint activation,
    uint block_m,
    uint block_n
) {
    __local float a_tile[MM_TILE_M][MM_TILE_K];
    __local float b_tile[MM_TILE_K][MM_TILE_N];

    uint tx = get_local_id(0);
    uint ty = get_local_id(1);
    uint thread = ty * MM_THREADS_X + tx;
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
            uint idx = thread + slab * MM_STAGE;
            uint r = idx / MM_TILE_K;
            uint column = idx % MM_TILE_K;
            a_tile[r][column] = (row_base + r < m && step + column < k) ?
                a[a_base + (row_base + r) * k + step + column] : 0.0f;
        }

        for (uint slab = 0; slab < (MM_TILE_K * MM_TILE_N) / MM_STAGE; slab++) {
            uint idx = thread + slab * MM_STAGE;
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

__kernel void matmulNBits(
    __global const real* a,
    __global const uchar* quantized,
    __global const real* scales,
    __global real* out,
    uint m,
    uint n,
    uint k,
    uint block_shift,
    uint blocks_per_row
) {
    __local float a_tile[MM_TILE_M][MM_TILE_K];
    __local float b_tile[MM_TILE_K][MM_TILE_N];

    uint tx = get_local_id(0);
    uint ty = get_local_id(1);
    uint thread = ty * MM_THREADS_X + tx;
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
            uint idx = thread + slab * MM_STAGE;
            uint r = idx / MM_TILE_K;
            uint column = idx % MM_TILE_K;
            a_tile[r][column] = (row_base + r < m && step + column < k) ?
                a[(row_base + r) * k + step + column] : 0.0f;
        }

        for (uint slab = 0; slab < (MM_TILE_K * MM_TILE_N) / MM_STAGE; slab++) {
            uint idx = thread + slab * MM_STAGE;
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

__kernel void conv2dGemm(
    __global const real* x,
    __global const real* w,
    __global const real* bias,
    __global real* out,
    __global const uint* meta,
    uint m,
    uint n,
    uint k,
    uint has_bias
) {
    __local float a_tile[MM_TILE_M][MM_TILE_K];
    __local float b_tile[MM_TILE_K][MM_TILE_N];

    uint tx = get_local_id(0);
    uint ty = get_local_id(1);
    uint thread = ty * MM_THREADS_X + tx;
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
            uint idx = thread + slab * MM_STAGE;
            uint r = idx / MM_TILE_K;
            uint column = idx % MM_TILE_K;
            a_tile[r][column] = (row_base + r < m && step + column < k) ?
                w[(row_base + r) * k + step + column] : 0.0f;
        }

        for (uint slab = 0; slab < (MM_TILE_K * MM_TILE_N) / MM_STAGE; slab++) {
            uint idx = thread + slab * MM_STAGE;
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
__kernel void pixelShuffle(
    __global const real* product,
    __global const real* bias,
    __global real* out,
    uint channels,
    uint height,
    uint width,
    uint kernel_h,
    uint kernel_w,
    uint count,
    uint has_bias
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
__kernel void convTranspose2dGemm(
    __global const real* x,
    __global const real* w,
    __global const real* bias,
    __global real* out,
    uint in_h,
    uint in_w,
    uint out_channels,
    uint out_h,
    uint out_w,
    uint kernel_h,
    uint kernel_w,
    uint m,
    uint n,
    uint k,
    uint has_bias
) {
    __local float a_tile[MM_TILE_M][MM_TILE_K];
    __local float b_tile[MM_TILE_K][MM_TILE_N];

    uint tx = get_local_id(0);
    uint ty = get_local_id(1);
    uint thread = ty * MM_THREADS_X + tx;
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
            uint idx = thread + slab * MM_STAGE;
            uint pixel = row_base + idx / MM_TILE_K;
            uint channel = step + idx % MM_TILE_K;
            a_tile[idx / MM_TILE_K][idx % MM_TILE_K] = (pixel < m && channel < k) ?
                x[x_base + channel * m + pixel] : 0.0f;
        }

        for (uint slab = 0; slab < (MM_TILE_K * MM_TILE_N) / MM_STAGE; slab++) {
            uint idx = thread + slab * MM_STAGE;
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

__kernel void conv2d(
    __global const real* x,
    __global const real* w,
    __global const real* bias,
    __global real* out,
    __global const uint* meta,
    uint count,
    uint has_bias
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

__kernel void convTranspose2d(
    __global const real* x,
    __global const real* w,
    __global const real* bias,
    __global real* out,
    __global const uint* meta,
    uint count,
    uint has_bias
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

__kernel void cumulativeSum(
    __global const real* src,
    __global real* dst,
    uint along,
    uint inner,
    uint count
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

__kernel void maxPool2d(
    __global const real* src,
    __global real* dst,
    __global const uint* meta,
    uint count
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

__kernel void sumAxes(
    __global const real* src,
    __global real* dst,
    __global const uint* meta,
    uint rank,
    uint reduced,
    uint swept,
    uint count
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

__kernel void resizeNearest(
    __global const real* src,
    __global real* dst,
    __global const uint* meta,
    uint rank,
    uint count
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

__kernel void instanceNorm(
    __global const real* x,
    __global const real* scale,
    __global const real* bias,
    __global real* out,
    uint channels,
    uint cols,
    float epsilon
) {
    __local float scratch[256];
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
// Matrix multiplication on the Xe matrix engines (XMX), where a GPU has them.
//
// `intel_sub_group_f16_f16_matrix_mad_k16` multiplies an 8x16 half tile of A by
// a 16x16 half tile of B into an 8x16 float accumulator, once per sub-group.
// On an Arc integrated GPU those run at about 26 TFLOP/s against 3.9 for scalar
// float, so the rest of this kernel exists to keep them fed. Tensors stay float
// in memory and are narrowed on the way into local memory, which is the
// precision an Intel GPU execution provider runs a graph at anyway.
//
// Feeding them is the whole difficulty. A mad wants lane `l` of the sub-group
// to hold column `l` of both operands, which is a transpose away from how a row
// major tile sits in memory, and doing that transpose with scalar loads costs
// more instructions than the mads themselves. Instead each tile is staged into
// local memory in sixteen-wide rows, so that `intel_sub_group_block_read_us8`
// -- which hands lane `l` the elements at `l`, `l + 16`, `l + 32` ... -- reads a
// whole fragment already transposed, one instruction per fragment. That leaves
// both the global loads and the local stores contiguous and vectorizable.
//
// `matmul_xmx_tile` in the runtime mirrors DP_TILE_M and DP_TILE_N; the two
// have to be changed together.
// ---------------------------------------------------------------------------
#if defined(cl_intel_subgroup_matrix_multiply_accumulate) && defined(cl_intel_subgroup_local_block_io)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_local_block_io : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

/// Tensors are already half when the runtime stores them that way, so the
/// staging load is then exactly the bytes a mad needs, with no conversion.
#if SAM3_HALF
#define DP_TO_HALFV(v) (v)
#else
#define DP_TO_HALFV(v) convert_half4(v)
#endif
/// How many of a fragment row of sixteen one work item stages at a time.
#define DP_PARTS (DP_SG / LANE_STEP)

/// A mad covers eight rows of A and sixteen columns of B; DP_M_BLOCKS and
/// DP_N_BLOCKS are how many of each one sub-group keeps in its accumulators,
/// and DP_SG_M and DP_SG_N how many sub-groups a work group is.
#define DP_SG 16
#define DP_KBLOCKS 2
#define DP_SG_M 8
#define DP_SG_N 4
#define DP_M_BLOCKS 4
#define DP_N_BLOCKS 2

#define DP_KSTEP (DP_KBLOCKS * DP_SG)
#define DP_TILE_M (DP_SG_M * DP_M_BLOCKS * 8)
#define DP_TILE_N (DP_SG_N * DP_N_BLOCKS * DP_SG)
#define DP_THREADS (DP_SG * DP_SG_M * DP_SG_N)
#define DP_NBLOCKS (DP_TILE_N / DP_SG)
#define DP_B_FRAGMENT (DP_SG * DP_SG)


/// Stages DP_TILE_M rows of the left operand for one K step, sixteen columns
/// to a local row, so that one block read hands each lane the column a mad
/// wants.
inline void stageXmxRows(
    __local ushort* a_tile,
    __global const real* a,
    uint a_base,
    uint row_base,
    uint m,
    uint k,
    uint step,
    uint thread,
    bool vector
) {
    #pragma unroll
    for (uint slot = thread; slot < DP_KBLOCKS * DP_TILE_M * DP_PARTS; slot += DP_THREADS) {
        uint part = slot % DP_PARTS;
        uint row = (slot / DP_PARTS) % DP_TILE_M;
        uint kb = slot / (DP_PARTS * DP_TILE_M);
        uint along = step + kb * DP_SG + part * LANE_STEP;
        __local half* out = (__local half*)&a_tile[(kb * DP_TILE_M + row) * DP_SG + part * LANE_STEP];
        if (row_base + row < m && along + LANE_STEP - 1 < k && vector) {
            realv v = VLOADV(0, &a[a_base + (row_base + row) * k + along]);
            VSTOREV(DP_TO_HALFV(v), 0, out);
        } else {
            #pragma unroll
            for (int e = 0; e < LANE_STEP; e++) {
                out[e] = (row_base + row < m && along + e < k) ?
                    (half)a[a_base + (row_base + row) * k + along + e] : (half)0.0f;
            }
        }
    }
}

/// Reads the fragments a work group has staged and issues the mads, one
/// sub-group's DP_M_BLOCKS by DP_N_BLOCKS corner of C at a time.
inline void accumulateXmx(
    __private float8* acc,
    __local const ushort* a_tile,
    __local const ushort* b_tile,
    uint sg_row,
    uint sg_col
) {
    #pragma unroll
    for (uint kb = 0; kb < DP_KBLOCKS; kb++) {
        short8 a_reg[DP_M_BLOCKS];
        #pragma unroll
        for (int i = 0; i < DP_M_BLOCKS; i++) {
            a_reg[i] = as_short8(intel_sub_group_block_read_us8(
                &a_tile[(kb * DP_TILE_M + sg_row + i * 8) * DP_SG]));
        }

        int8 b_reg[DP_N_BLOCKS];
        #pragma unroll
        for (int j = 0; j < DP_N_BLOCKS; j++) {
            __local const ushort* base = &b_tile[((kb * DP_NBLOCKS) + sg_col + j) * DP_B_FRAGMENT];
            ushort16 packed;
            packed.lo = intel_sub_group_block_read_us8(base);
            packed.hi = intel_sub_group_block_read_us8(base + DP_B_FRAGMENT / 2);
            b_reg[j] = as_int8(packed);
        }

        #pragma unroll
        for (int i = 0; i < DP_M_BLOCKS; i++) {
            #pragma unroll
            for (int j = 0; j < DP_N_BLOCKS; j++) {
                acc[i * DP_N_BLOCKS + j] =
                    intel_sub_group_f16_f16_matrix_mad_k16(a_reg[i], b_reg[j], acc[i * DP_N_BLOCKS + j]);
            }
        }
    }
}

__attribute__((intel_reqd_sub_group_size(DP_SG)))
__kernel void matmulXmx(
    __global const real* a,
    __global const real* b,
    __global real* c,
    uint m,
    uint n,
    uint k,
    uint a_batch,
    uint b_batch,
    uint c_batch,
    uint b_transposed,
    __global const real* bias,
    uint has_bias,
    uint activation,
    uint block_m,
    uint block_n
) {
    __local ushort a_tile[DP_KBLOCKS * DP_TILE_M * DP_SG];
    __local ushort b_tile[DP_KBLOCKS * DP_NBLOCKS * DP_B_FRAGMENT];

    const uint lane = get_local_id(0);
    const uint sg = get_local_id(1);
    const uint thread = sg * DP_SG + lane;
    const uint sg_row = (sg / DP_SG_N) * (DP_M_BLOCKS * 8);
    const uint sg_col = (sg % DP_SG_N) * DP_N_BLOCKS;

    // Work groups are dispatched in order, so which tile each takes decides
    // what the resident ones share. Taken in rows of C, A is read again for
    // every column of it, which for the wide products here is most of the
    // cost. `block_m` by `block_n` tiles is as much of C as the runtime
    // reckons the last level cache holds both operands of, and the work groups
    // finish one such block before starting the next.
    const uint tiles_n = (n + DP_TILE_N - 1) / DP_TILE_N;
    const uint tiles_m = (m + DP_TILE_M - 1) / DP_TILE_M;
    const uint blocks_n = (tiles_n + block_n - 1) / block_n;
    const uint tile = get_group_id(0);
    const uint at = tile / (block_m * block_n);
    const uint within = tile % (block_m * block_n);
    const uint tile_m = (at / blocks_n) * block_m + within / block_n;
    const uint tile_n = (at % blocks_n) * block_n + within % block_n;
    // The grid is a whole number of blocks, so the last ones overhang.
    if (tile_m >= tiles_m || tile_n >= tiles_n) return;
    const uint row_base = tile_m * DP_TILE_M;
    const uint col_base = tile_n * DP_TILE_N;
    const uint batch = get_group_id(2);
    const uint a_base = batch * a_batch;
    const uint b_base = batch * b_batch;

    const bool a_vector = (k % LANE_STEP) == 0u;
    const bool b_vector = !b_transposed && (n % LANE_STEP) == 0u;

    float8 acc[DP_M_BLOCKS * DP_N_BLOCKS];
    #pragma unroll
    for (int i = 0; i < DP_M_BLOCKS * DP_N_BLOCKS; i++) acc[i] = (float8)(0.0f);

    for (uint step = 0; step < k; step += DP_KSTEP) {
        stageXmxRows(a_tile, a, a_base, row_base, m, k, step, thread, a_vector);

        // B goes in as sixteen-by-sixteen blocks, also row major, which two
        // block reads turn into the sixteen values a lane holds for one mad.
        #pragma unroll
        for (uint slot = thread; slot < DP_KBLOCKS * DP_NBLOCKS * DP_SG * DP_PARTS; slot += DP_THREADS) {
            uint part = slot % DP_PARTS;
            uint row = (slot / DP_PARTS) & (DP_SG - 1);
            uint cb = (slot / (DP_PARTS * DP_SG)) % DP_NBLOCKS;
            uint kb = slot / (DP_PARTS * DP_SG * DP_NBLOCKS);
            uint along = step + kb * DP_SG + row;
            uint col = col_base + cb * DP_SG + part * LANE_STEP;
            __local ushort* out = &b_tile[((kb * DP_NBLOCKS + cb) * DP_SG + row) * DP_SG + part * LANE_STEP];
            if (along < k && col + LANE_STEP - 1 < n && b_vector) {
                realv v = VLOADV(0, &b[b_base + along * n + col]);
                VSTOREV(DP_TO_HALFV(v), 0, (__local half*)out);
            } else {
                #pragma unroll
                for (int e = 0; e < LANE_STEP; e++) {
                    float value = 0.0f;
                    if (along < k && col + e < n) {
                        value = b_transposed ?
                            b[b_base + (col + e) * k + along] :
                            b[b_base + along * n + col + e];
                    }
                    ((__local half*)out)[e] = (half)value;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        accumulateXmx(acc, a_tile, b_tile, sg_row, sg_col);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    #pragma unroll
    for (int j = 0; j < DP_N_BLOCKS; j++) {
        uint col = col_base + (sg_col + j) * DP_SG + lane;
        if (col >= n) continue;
        float shift = has_bias ? (float)bias[col] : 0.0f;
        #pragma unroll
        for (int i = 0; i < DP_M_BLOCKS; i++) {
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                uint row = row_base + sg_row + i * 8 + r;
                if (row < m) {
                    float value = ((float*)&acc[i * DP_N_BLOCKS + j])[r] + shift;
                    c[batch * c_batch + row * n + col] = activation ? gelu(value) : value;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The same product again, reading both operands straight from global memory
// with the 2D block loads Xe2 added. The hardware gathers what a mad wants --
// sixteen rows of A, or a VNNI-packed sixteen by thirty-two block of B -- in
// one message, so there is no local memory to stage through, nothing to
// synchronize, and no address arithmetic per element. What the work groups
// re-read is served by the cache rather than by a barrier, which is worth
// about twice the throughput of the staged kernel above on an Arc iGPU.
//
// Only reachable where the tensors are halves and the operands are laid out
// as the block loads require; `runtime.product` decides and falls back. It
// takes the staged kernel's arguments unchanged so that the choice is the
// only thing that differs, which leaves `b_transposed` here as a parameter
// this kernel is never handed. TD_TILE_M, TD_TILE_N and the sub-group count
// are mirrored in the runtime.
// ---------------------------------------------------------------------------
#if defined(cl_intel_subgroup_2d_block_io) && SAM3_HALF
#pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io : enable

/// A work group is TD_SG_M by TD_SG_N sub-groups, each holding TD_M_BLOCKS by
/// TD_N_BLOCKS accumulators of the eight by sixteen a mad produces. Eight is
/// as many as fit in a thread's registers; past that the compiler spills and
/// the whole gain goes away.
#define TD_SG 16
#define TD_SG_M 4
#define TD_SG_N 4
#define TD_M_BLOCKS 4
#define TD_N_BLOCKS 2

#define TD_TILE_M (TD_SG_M * TD_M_BLOCKS * 8)
#define TD_TILE_N (TD_SG_N * TD_N_BLOCKS * TD_SG)

__attribute__((intel_reqd_sub_group_size(TD_SG)))
__kernel void matmulXmxBlock(
    __global const real* a,
    __global const real* b,
    __global real* c,
    uint m,
    uint n,
    uint k,
    uint a_batch,
    uint b_batch,
    uint c_batch,
    uint b_transposed,
    __global const real* bias,
    uint has_bias,
    uint activation,
    uint block_m,
    uint block_n
) {
    const uint lane = get_local_id(0);
    const uint sg = get_local_id(1);
    const uint sg_row = (sg / TD_SG_N) * (TD_M_BLOCKS * 8);
    const uint sg_col = (sg % TD_SG_N) * (TD_N_BLOCKS * TD_SG);

    // The same walk in cache-sized blocks as the staged kernel; see there.
    const uint tiles_n = (n + TD_TILE_N - 1) / TD_TILE_N;
    const uint tiles_m = (m + TD_TILE_M - 1) / TD_TILE_M;
    const uint blocks_n = (tiles_n + block_n - 1) / block_n;
    const uint tile = get_group_id(0);
    const uint at = tile / (block_m * block_n);
    const uint within = tile % (block_m * block_n);
    const uint tile_m = (at / blocks_n) * block_m + within / block_n;
    const uint tile_n = (at % blocks_n) * block_n + within % block_n;
    if (tile_m >= tiles_m || tile_n >= tiles_n) return;

    const uint row_base = tile_m * TD_TILE_M + sg_row;
    const uint col_base = tile_n * TD_TILE_N + sg_col;
    const uint batch = get_group_id(2);
    __global const real* a_plane = a + batch * a_batch;
    __global const real* b_plane = b + batch * b_batch;

    // What the block loads are told about the two surfaces. Whatever the
    // coordinates reach outside them reads as zero, so the edges of C need no
    // special case here, only in the store below.
    const int a_width = (int)(k * sizeof(real));
    const int a_height = (int)m;
    const int b_width = (int)(n * sizeof(real));
    const int b_height = (int)k;

    float8 acc[TD_M_BLOCKS * TD_N_BLOCKS];
    #pragma unroll
    for (int i = 0; i < TD_M_BLOCKS * TD_N_BLOCKS; i++) acc[i] = (float8)(0.0f);

    for (uint step = 0; step < k; step += TD_SG) {
        // Two fragments to a message, in both operands: the same bytes as one
        // fragment at a time, half the sends, and the destination already
        // holds them in the order the mads want.
        short8 a_reg[TD_M_BLOCKS];
        #pragma unroll
        for (int i = 0; i < TD_M_BLOCKS; i += 2) {
            intel_sub_group_2d_block_read_16b_16r16x1c(
                a_plane, a_width, a_height, a_width,
                (int2)((int)step, (int)(row_base + i * 8)), (ushort*)&a_reg[i]);
        }
        int8 b_reg[TD_N_BLOCKS];
        #pragma unroll
        for (int j = 0; j < TD_N_BLOCKS; j += 2) {
            intel_sub_group_2d_block_read_transform_16b_16r16x2c(
                b_plane, b_width, b_height, b_width,
                (int2)((int)(col_base + j * TD_SG), (int)step), (uint*)&b_reg[j]);
        }
        #pragma unroll
        for (int i = 0; i < TD_M_BLOCKS; i++) {
            #pragma unroll
            for (int j = 0; j < TD_N_BLOCKS; j++) {
                acc[i * TD_N_BLOCKS + j] =
                    intel_sub_group_f16_f16_matrix_mad_k16(a_reg[i], b_reg[j], acc[i * TD_N_BLOCKS + j]);
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < TD_N_BLOCKS; j++) {
        uint col = col_base + j * TD_SG + lane;
        if (col >= n) continue;
        float shift = has_bias ? (float)bias[col] : 0.0f;
        #pragma unroll
        for (int i = 0; i < TD_M_BLOCKS; i++) {
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                uint row = row_base + i * 8 + r;
                if (row < m) {
                    float value = ((float*)&acc[i * TD_N_BLOCKS + j])[r] + shift;
                    c[batch * c_batch + row * n + col] = activation ? gelu(value) : value;
                }
            }
        }
    }
}
#endif

/// The same product with the image patches on the right: `conv2dGemm` on the
/// matrix engines. The im2col matrix is never built -- a tap index unpacks
/// into a channel and a kernel offset, a pixel index into an output row and
/// column -- so the gather happens on the way into local memory, one element
/// at a time, and only the mads see a dense tile.
__attribute__((intel_reqd_sub_group_size(DP_SG)))
__kernel void conv2dGemmXmx(
    __global const real* x,
    __global const real* w,
    __global const real* bias,
    __global real* out,
    __global const uint* meta,
    uint m,
    uint n,
    uint k,
    uint has_bias
) {
    __local ushort a_tile[DP_KBLOCKS * DP_TILE_M * DP_SG];
    __local ushort b_tile[DP_KBLOCKS * DP_NBLOCKS * DP_B_FRAGMENT];

    const uint lane = get_local_id(0);
    const uint sg = get_local_id(1);
    const uint thread = sg * DP_SG + lane;
    const uint sg_row = (sg / DP_SG_N) * (DP_M_BLOCKS * 8);
    const uint sg_col = (sg % DP_SG_N) * DP_N_BLOCKS;

    // No band walk here: `m` is a channel count, so the grid is one or two
    // tiles tall and every work group already shares the same weights.
    const uint row_base = get_group_id(1) * DP_TILE_M;
    const uint col_base = get_group_id(0) * DP_TILE_N;
    const uint batch = get_group_id(2);

    const uint in_channels = meta[0];
    const uint in_h = meta[1];
    const uint in_w = meta[2];
    const uint out_w = meta[4];
    const uint kernel_w = meta[6];
    const uint stride_h = meta[7];
    const uint stride_w = meta[8];
    const uint pad_h = meta[9];
    const uint pad_w = meta[10];
    const uint dilation_h = meta[11];
    const uint dilation_w = meta[12];
    const uint taps = meta[5] * kernel_w;
    const uint x_base = batch * in_channels * in_h * in_w;

    float8 acc[DP_M_BLOCKS * DP_N_BLOCKS];
    #pragma unroll
    for (int i = 0; i < DP_M_BLOCKS * DP_N_BLOCKS; i++) acc[i] = (float8)(0.0f);

    for (uint step = 0; step < k; step += DP_KSTEP) {
        stageXmxRows(a_tile, w, 0, row_base, m, k, step, thread, (k % LANE_STEP) == 0u);

        for (uint slot = thread; slot < DP_KBLOCKS * DP_NBLOCKS * DP_B_FRAGMENT; slot += DP_THREADS) {
            uint column = slot % DP_SG;
            uint row = (slot / DP_SG) % DP_SG;
            uint cb = (slot / DP_B_FRAGMENT) % DP_NBLOCKS;
            uint kb = slot / (DP_B_FRAGMENT * DP_NBLOCKS);
            uint tap = step + kb * DP_SG + row;
            uint pixel = col_base + cb * DP_SG + column;
            float value = 0.0f;
            if (tap < k && pixel < n) {
                uint channel = tap / taps;
                uint ky = (tap % taps) / kernel_w;
                uint kx = tap % kernel_w;
                uint y = (pixel / out_w) * stride_h + ky * dilation_h;
                uint p = (pixel % out_w) * stride_w + kx * dilation_w;
                if (y >= pad_h && p >= pad_w && y - pad_h < in_h && p - pad_w < in_w) {
                    value = x[x_base + (channel * in_h + y - pad_h) * in_w + p - pad_w];
                }
            }
            ((__local half*)b_tile)[((kb * DP_NBLOCKS + cb) * DP_SG + row) * DP_SG + column] = (half)value;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        accumulateXmx(acc, a_tile, b_tile, sg_row, sg_col);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    #pragma unroll
    for (int j = 0; j < DP_N_BLOCKS; j++) {
        uint col = col_base + (sg_col + j) * DP_SG + lane;
        if (col >= n) continue;
        #pragma unroll
        for (int i = 0; i < DP_M_BLOCKS; i++) {
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                uint row = row_base + sg_row + i * 8 + r;
                if (row >= m) continue;
                float shift = has_bias ? (float)bias[row] : 0.0f;
                out[batch * m * n + row * n + col] = ((float*)&acc[i * DP_N_BLOCKS + j])[r] + shift;
            }
        }
    }
}

/// Scaled dot product attention, whole. The three operators it replaces --
/// Q x K, softmax over the keys, and the product with V -- pass a score matrix
/// between them that is, for the encoder's global attention, 860 MB written
/// once and read three times. Nothing else in the graph looks at it, so a
/// kernel that keeps a strip of it in registers never writes it at all.
///
/// The mad layout is what makes that cheap. After Q x K, lane `l` holds
/// S[query][key0 + l] for the eight queries its sub-group owns, which is
/// exactly the layout the second product wants for its left operand -- so the
/// scores never leave the registers they were produced in, and the running
/// maximum and sum a streaming softmax needs are one sub-group reduction each.
///
/// Written for a head dimension of exactly 64, which is what these exports
/// use; `runtime.attention` checks that before choosing this path.
#define FA_SG 16
#define FA_HEAD 64
#define FA_QROWS 8
#define FA_SUBGROUPS 16
#define FA_DBLOCKS (FA_HEAD / FA_SG)
#define FA_KSTEP 16
#define FA_KBLOCKS (FA_KSTEP / FA_SG)
#define FA_QTILE (FA_QROWS * FA_SUBGROUPS)
#define FA_THREADS (FA_SG * FA_SUBGROUPS)
#define FA_FRAGMENT (FA_SG * FA_SG)
#define FA_PARTS (FA_SG / LANE_STEP)

/// Reads one 16x16 fragment a mad wants as its right operand: the block reads
/// hand lane `l` the elements at `l`, `l + 16`, ... , which is column `l`.
inline int8 fragmentOf(__local const ushort* base) {
    ushort16 packed;
    packed.lo = intel_sub_group_block_read_us8(base);
    packed.hi = intel_sub_group_block_read_us8(base + FA_FRAGMENT / 2);
    return as_int8(packed);
}

__attribute__((intel_reqd_sub_group_size(FA_SG)))
__kernel void attention(
    __global const real* q,
    __global const real* kt,
    __global const real* v,
    __global real* out,
    uint queries,
    uint keys,
    float scale
) {
    __local ushort k_tile[FA_DBLOCKS * FA_KBLOCKS * FA_FRAGMENT];
    __local ushort v_tile[FA_KBLOCKS * FA_DBLOCKS * FA_FRAGMENT];

    const uint lane = get_local_id(0);
    const uint sg = get_local_id(1);
    const uint thread = sg * FA_SG + lane;
    const uint batch = get_group_id(1);
    const uint row_base = get_group_id(0) * FA_QTILE + sg * FA_QROWS;

    const uint q_base = batch * queries * FA_HEAD;
    const uint kt_base = batch * FA_HEAD * keys;
    const uint v_base = batch * keys * FA_HEAD;

    // Q stays in registers for the whole pass over the keys: eight queries by
    // sixteen of the head dimension, per block, the lane picking the column.
    short8 q_reg[FA_DBLOCKS];
    #pragma unroll
    for (int db = 0; db < FA_DBLOCKS; db++) {
        #pragma unroll
        for (int m = 0; m < FA_QROWS; m++) {
            uint row = row_base + m;
            ((short*)&q_reg[db])[m] = row < queries ?
                as_short((half)q[q_base + row * FA_HEAD + db * FA_SG + lane]) : (short)0;
        }
    }

    float8 out_acc[FA_DBLOCKS];
    #pragma unroll
    for (int db = 0; db < FA_DBLOCKS; db++) out_acc[db] = (float8)(0.0f);
    float8 running_max = (float8)(-INFINITY);
    float8 running_sum = (float8)(0.0f);

    for (uint key_base = 0; key_base < keys; key_base += FA_KSTEP) {
        // K goes in as the right operand of the first product: sixteen of the
        // head dimension by sixteen keys, the keys contiguous in both.
        #pragma unroll
        for (uint slot = thread; slot < FA_DBLOCKS * FA_KBLOCKS * FA_SG * FA_PARTS; slot += FA_THREADS) {
            uint part = slot % FA_PARTS;
            uint depth = (slot / FA_PARTS) % FA_SG;
            uint kb = (slot / (FA_PARTS * FA_SG)) % FA_KBLOCKS;
            uint db = slot / (FA_PARTS * FA_SG * FA_KBLOCKS);
            realv group = VLOADV(0, &kt[kt_base + (db * FA_SG + depth) * keys + key_base + kb * FA_SG + part * LANE_STEP]);
            VSTOREV(DP_TO_HALFV(group), 0,
                (__local half*)&k_tile[((db * FA_KBLOCKS + kb) * FA_SG + depth) * FA_SG + part * LANE_STEP]);
        }

        // V goes in as the right operand of the second: sixteen keys by
        // sixteen of the head dimension, the head dimension contiguous.
        #pragma unroll
        for (uint slot = thread; slot < FA_KBLOCKS * FA_DBLOCKS * FA_SG * FA_PARTS; slot += FA_THREADS) {
            uint part = slot % FA_PARTS;
            uint key = (slot / FA_PARTS) % FA_SG;
            uint db = (slot / (FA_PARTS * FA_SG)) % FA_DBLOCKS;
            uint kb = slot / (FA_PARTS * FA_SG * FA_DBLOCKS);
            realv group = VLOADV(0, &v[v_base + (key_base + kb * FA_SG + key) * FA_HEAD + db * FA_SG + part * LANE_STEP]);
            VSTOREV(DP_TO_HALFV(group), 0,
                (__local half*)&v_tile[((kb * FA_DBLOCKS + db) * FA_SG + key) * FA_SG + part * LANE_STEP]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        float8 score[FA_KBLOCKS];
        #pragma unroll
        for (int kb = 0; kb < FA_KBLOCKS; kb++) {
            score[kb] = (float8)(0.0f);
            #pragma unroll
            for (int db = 0; db < FA_DBLOCKS; db++) {
                score[kb] = intel_sub_group_f16_f16_matrix_mad_k16(
                    q_reg[db], fragmentOf(&k_tile[(db * FA_KBLOCKS + kb) * FA_FRAGMENT]), score[kb]);
            }
        }

        // The scale the graph applies to Q and K before the product, folded
        // in here rather than paid as two passes over both of them.
        #pragma unroll
        for (int kb = 0; kb < FA_KBLOCKS; kb++) score[kb] *= scale;

        // One reduction per query for the whole step, rather than one per
        // block: the largest of the four blocks first, then across the lanes.
        float8 highest = score[0];
        #pragma unroll
        for (int kb = 1; kb < FA_KBLOCKS; kb++) highest = fmax(highest, score[kb]);
        #pragma unroll
        for (int m = 0; m < FA_QROWS; m++) {
            ((float*)&highest)[m] = sub_group_reduce_max(((float*)&highest)[m]);
        }
        highest = fmax(running_max, highest);

        // Everything already accumulated was scaled against the old maximum.
        float8 correction = fastExp(running_max - highest);
        running_max = highest;
        running_sum *= correction;
        #pragma unroll
        for (int db = 0; db < FA_DBLOCKS; db++) out_acc[db] *= correction;

        float8 total = (float8)(0.0f);
        #pragma unroll
        for (int kb = 0; kb < FA_KBLOCKS; kb++) {
            score[kb] = fastExp(score[kb] - highest);
            total += score[kb];
        }
        #pragma unroll
        for (int m = 0; m < FA_QROWS; m++) {
            ((float*)&running_sum)[m] += sub_group_reduce_add(((float*)&total)[m]);
        }

        #pragma unroll
        for (int kb = 0; kb < FA_KBLOCKS; kb++) {
            short8 weights;
            #pragma unroll
            for (int m = 0; m < FA_QROWS; m++) {
                ((short*)&weights)[m] = as_short((half)((float*)&score[kb])[m]);
            }
            #pragma unroll
            for (int db = 0; db < FA_DBLOCKS; db++) {
                out_acc[db] = intel_sub_group_f16_f16_matrix_mad_k16(
                    weights, fragmentOf(&v_tile[(kb * FA_DBLOCKS + db) * FA_FRAGMENT]), out_acc[db]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    #pragma unroll
    for (int m = 0; m < FA_QROWS; m++) {
        uint row = row_base + m;
        if (row >= queries) continue;
        float scale = 1.0f / ((float*)&running_sum)[m];
        #pragma unroll
        for (int db = 0; db < FA_DBLOCKS; db++) {
            out[q_base + row * FA_HEAD + db * FA_SG + lane] = ((float*)&out_acc[db])[m] * scale;
        }
    }
}

/// The same attention with its operands read straight from memory by the 2D
/// block loads, the way `matmulXmxBlock` reads a product's. Q's rows and the
/// VNNI-packed fragments of K and V each arrive in one message already in the
/// order a mad wants them, so there is no local memory to stage anything
/// through and no barrier between the two products -- and no sixteen work
/// items writing a tile that the sixteen sub-groups then read back. What they
/// share is a cache line instead.
#if defined(cl_intel_subgroup_2d_block_io) && SAM3_HALF
__attribute__((intel_reqd_sub_group_size(FA_SG)))
__kernel void attentionBlock(
    __global const real* q,
    __global const real* kt,
    __global const real* v,
    __global real* out,
    uint queries,
    uint keys,
    float scale
) {
    const uint lane = get_local_id(0);
    const uint sg = get_local_id(1);
    const uint batch = get_group_id(1);
    const uint row_base = get_group_id(0) * FA_QTILE + sg * FA_QROWS;

    const uint q_base = batch * queries * FA_HEAD;
    __global const real* q_plane = q + q_base;
    __global const real* kt_plane = kt + batch * FA_HEAD * keys;
    __global const real* v_plane = v + batch * keys * FA_HEAD;

    // Q is a row per query and a column per depth; K arrives transposed, so
    // its rows are the depth and its columns the keys; V is a row per key.
    const int head_bytes = (int)(FA_HEAD * sizeof(real));
    const int keys_bytes = (int)(keys * sizeof(real));

    // Q stays in registers for the whole pass over the keys: eight queries by
    // sixteen of the head dimension, per block, the lane picking the column.
    // Rows past the end of Q read as zero, which is what the tail wants.
    short8 q_reg[FA_DBLOCKS];
    #pragma unroll
    for (int db = 0; db < FA_DBLOCKS; db++) {
        intel_sub_group_2d_block_read_16b_8r16x1c(
            q_plane, head_bytes, (int)queries, head_bytes,
            (int2)(db * FA_SG, (int)row_base), (ushort*)&q_reg[db]);
    }

    float8 out_acc[FA_DBLOCKS];
    #pragma unroll
    for (int db = 0; db < FA_DBLOCKS; db++) out_acc[db] = (float8)(0.0f);
    float8 running_max = (float8)(-INFINITY);
    float8 running_sum = (float8)(0.0f);

    for (uint key_base = 0; key_base < keys; key_base += FA_KSTEP) {
        float8 score[FA_KBLOCKS];
        #pragma unroll
        for (int kb = 0; kb < FA_KBLOCKS; kb++) {
            score[kb] = (float8)(0.0f);
            #pragma unroll
            for (int db = 0; db < FA_DBLOCKS; db++) {
                int8 k_reg;
                intel_sub_group_2d_block_read_transform_16b_16r16x1c(
                    kt_plane, keys_bytes, (int)FA_HEAD, keys_bytes,
                    (int2)((int)(key_base + kb * FA_SG), db * FA_SG), (uint*)&k_reg);
                score[kb] = intel_sub_group_f16_f16_matrix_mad_k16(q_reg[db], k_reg, score[kb]);
            }
        }

        // The scale the graph applies to Q and K before the product, folded
        // in here rather than paid as two passes over both of them.
        #pragma unroll
        for (int kb = 0; kb < FA_KBLOCKS; kb++) score[kb] *= scale;

        // One reduction per query for the whole step, rather than one per
        // block: the largest of the blocks first, then across the lanes.
        float8 highest = score[0];
        #pragma unroll
        for (int kb = 1; kb < FA_KBLOCKS; kb++) highest = fmax(highest, score[kb]);
        #pragma unroll
        for (int m = 0; m < FA_QROWS; m++) {
            ((float*)&highest)[m] = sub_group_reduce_max(((float*)&highest)[m]);
        }
        highest = fmax(running_max, highest);

        // Everything already accumulated was scaled against the old maximum.
        float8 correction = fastExp(running_max - highest);
        running_max = highest;
        running_sum *= correction;
        #pragma unroll
        for (int db = 0; db < FA_DBLOCKS; db++) out_acc[db] *= correction;

        float8 total = (float8)(0.0f);
        #pragma unroll
        for (int kb = 0; kb < FA_KBLOCKS; kb++) {
            score[kb] = fastExp(score[kb] - highest);
            total += score[kb];
        }
        #pragma unroll
        for (int m = 0; m < FA_QROWS; m++) {
            ((float*)&running_sum)[m] += sub_group_reduce_add(((float*)&total)[m]);
        }

        #pragma unroll
        for (int kb = 0; kb < FA_KBLOCKS; kb++) {
            short8 weights;
            #pragma unroll
            for (int m = 0; m < FA_QROWS; m++) {
                ((short*)&weights)[m] = as_short((half)((float*)&score[kb])[m]);
            }
            #pragma unroll
            for (int db = 0; db < FA_DBLOCKS; db++) {
                int8 v_reg;
                intel_sub_group_2d_block_read_transform_16b_16r16x1c(
                    v_plane, head_bytes, (int)keys, head_bytes,
                    (int2)(db * FA_SG, (int)(key_base + kb * FA_SG)), (uint*)&v_reg);
                out_acc[db] = intel_sub_group_f16_f16_matrix_mad_k16(weights, v_reg, out_acc[db]);
            }
        }
    }

    #pragma unroll
    for (int m = 0; m < FA_QROWS; m++) {
        uint row = row_base + m;
        if (row >= queries) continue;
        float scale = 1.0f / ((float*)&running_sum)[m];
        #pragma unroll
        for (int db = 0; db < FA_DBLOCKS; db++) {
            out[q_base + row * FA_HEAD + db * FA_SG + lane] = ((float*)&out_acc[db])[m] * scale;
        }
    }
}
#endif
#endif
