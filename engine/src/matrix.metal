
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
