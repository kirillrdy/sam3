//! Meta's SAM 3 vision encoder (`detector_model.vision_encoder`), ported to run
//! directly off the released `facebook/sam3` SafeTensors checkpoint.
//!
//! The backbone is the Perception Encoder ViT: 32 pre-norm layers at d=1024 with
//! 16 heads, patch 14 over a 1008x1008 input (72x72 tokens), axial 2D RoPE, and
//! 24x24 windowed attention everywhere except the four global layers
//! (7, 15, 23, 31). Position embeddings are *tiled* from the 24x24 pretraining
//! grid, not interpolated. The neck is an FPN producing four feature maps at
//! scale 4x / 2x / 1x / 0.5x of the token grid, each projected to 256 channels.
//! `Sam3Model` feeds the first three (288², 144², 72² at a 1008² input) to the
//! DETR encoder and drops the 0.5x level.
//!
//! Reference: `Sam3ViTModel` / `Sam3VisionModel` in HF Transformers, and
//! `.zig-cache/sam3/config.json` (fetched by `zig build fetch-weights`) for every
//! hyper-parameter below.

const std = @import("std");
const Tensor = @import("../tensor/tensor.zig").Tensor;
const math = @import("../tensor/math.zig");
const ops = @import("../tensor/ops.zig");
const parallel = @import("../tensor/parallel.zig");
const WeightStore = @import("../weights/weight_loader.zig").WeightStore;
const ImageRGB = @import("../io/image.zig").ImageRGB;

pub const VisionConfig = struct {
    hidden_size: usize = 1024,
    num_heads: usize = 16,
    num_layers: usize = 32,
    intermediate_size: usize = 4736,
    patch_size: usize = 14,
    /// Side length of the (square) model input. 1008 in the release; smaller
    /// multiples of `patch_size` work too since RoPE and the tiled position
    /// embeddings are both resolution-agnostic.
    image_size: usize = 1008,
    /// Grid the position embeddings were trained at: 336/14 = 24.
    pretrain_image_size: usize = 336,
    window_size: usize = 24,
    global_attn_indexes: []const usize = &.{ 7, 15, 23, 31 },
    rope_theta: f32 = 10000.0,
    layer_norm_eps: f32 = 1e-6,

    fpn_hidden_size: usize = 256,
    scale_factors: []const f32 = &.{ 4.0, 2.0, 1.0, 0.5 },

    /// Checkpoint prefix for the backbone. There is only one in the release —
    /// the detector and the tracker share it.
    prefix: []const u8 = "detector_model.vision_encoder",
    /// Checkpoint prefix for the FPN neck. The detector and the tracker each
    /// have their own neck over the shared backbone: `tracker_neck` selects the
    /// tracker's.
    neck_prefix: []const u8 = "detector_model.vision_encoder.neck",

    pub inline fn grid(self: VisionConfig) usize {
        return self.image_size / self.patch_size;
    }

    pub inline fn headDim(self: VisionConfig) usize {
        return self.hidden_size / self.num_heads;
    }

    pub inline fn isGlobalLayer(self: VisionConfig, index: usize) bool {
        for (self.global_attn_indexes) |g| {
            if (g == index) return true;
        }
        return false;
    }
};

/// Cosine/sine tables for axial 2D RoPE over an `end_x` x `end_y` token grid.
/// Layout is `[end_x * end_y, head_dim]`, matching `Sam3ViTRotaryEmbedding`.
pub const RopeTable = struct {
    allocator: std.mem.Allocator,
    cos: []f32,
    sin: []f32,
    seq_len: usize,
    head_dim: usize,

    pub fn init(
        allocator: std.mem.Allocator,
        end_x: usize,
        end_y: usize,
        head_dim: usize,
        theta: f32,
        scale: f32,
    ) !RopeTable {
        std.debug.assert(head_dim % 4 == 0);
        const seq_len = end_x * end_y;
        const quarter = head_dim / 4;

        var cos = try allocator.alloc(f32, seq_len * head_dim);
        errdefer allocator.free(cos);
        var sin = try allocator.alloc(f32, seq_len * head_dim);
        errdefer allocator.free(sin);

        // freqs = 1 / theta^(arange(0, dim, 4) / dim)
        for (0..seq_len) |pos| {
            const x_pos = @as(f32, @floatFromInt(pos % end_x)) * scale;
            const y_pos = @as(f32, @floatFromInt(pos / end_x)) * scale;

            for (0..quarter) |i| {
                const exponent = @as(f32, @floatFromInt(4 * i)) / @as(f32, @floatFromInt(head_dim));
                const freq = 1.0 / std.math.pow(f32, theta, exponent);

                // cat([freqs_x, freqs_y]) then repeat_interleave(2)
                const angle_x = x_pos * freq;
                const angle_y = y_pos * freq;

                const base = pos * head_dim;
                cos[base + 2 * i] = @cos(angle_x);
                cos[base + 2 * i + 1] = @cos(angle_x);
                sin[base + 2 * i] = @sin(angle_x);
                sin[base + 2 * i + 1] = @sin(angle_x);

                const y_off = base + head_dim / 2;
                cos[y_off + 2 * i] = @cos(angle_y);
                cos[y_off + 2 * i + 1] = @cos(angle_y);
                sin[y_off + 2 * i] = @sin(angle_y);
                sin[y_off + 2 * i + 1] = @sin(angle_y);
            }
        }

        return RopeTable{
            .allocator = allocator,
            .cos = cos,
            .sin = sin,
            .seq_len = seq_len,
            .head_dim = head_dim,
        };
    }

    pub fn deinit(self: *RopeTable) void {
        self.allocator.free(self.cos);
        self.allocator.free(self.sin);
    }
};

/// In-place pairwise rotation: `x * cos + rotate_pairwise(x) * sin`, where
/// `rotate_pairwise` maps (a, b) -> (-b, a) over adjacent channel pairs.
fn applyRope(allocator: std.mem.Allocator, x: *Tensor, table: RopeTable, num_heads: usize) void {
    std.debug.assert(x.shape.len == 3);
    const batch = x.shape[0];
    const seq = x.shape[1];
    const dim = x.shape[2];
    const head_dim = table.head_dim;
    std.debug.assert(seq == table.seq_len);
    std.debug.assert(dim == num_heads * head_dim);

    const Context = struct {
        data: []f32,
        table: RopeTable,
        seq: usize,
        dim: usize,
        num_heads: usize,
        head_dim: usize,
    };

    const ctx = Context{
        .data = x.data,
        .table = table,
        .seq = seq,
        .dim = dim,
        .num_heads = num_heads,
        .head_dim = head_dim,
    };

    parallel.parallelFor(allocator, batch * seq, ctx, struct {
        fn worker(c: Context, start: usize, end: usize) void {
            const hd = c.head_dim;
            for (start..end) |task| {
                const s = task % c.seq;
                const row = c.data[task * c.dim ..][0..c.dim];
                const cos_row = c.table.cos[s * hd ..][0..hd];
                const sin_row = c.table.sin[s * hd ..][0..hd];

                for (0..c.num_heads) |h| {
                    const head = row[h * hd ..][0..hd];
                    var d: usize = 0;
                    while (d < hd) : (d += 2) {
                        const a = head[d];
                        const bb = head[d + 1];
                        head[d] = a * cos_row[d] - bb * sin_row[d];
                        head[d + 1] = bb * cos_row[d + 1] + a * sin_row[d + 1];
                    }
                }
            }
        }
    }.worker);
}

const VEC = math.VEC_SIZE;
const Vec = math.F32Vec;

/// Queries processed together, so each key and value column loaded from the
/// packed head is reused across a whole block instead of once.
const Q_BLOCK = 8;
/// Queries per value-accumulation tile, chosen with the width below so the tile
/// fits the sixteen vector registers AVX2 offers.
const Q_SUB = 4;
/// Output channels per value-accumulation tile.
const D_TILE = 4 * VEC;

/// Softmax attention over `[batch, seq, dim]` with pre-applied RoPE, running one
/// (batch, head) pair per task.
///
/// Q, K and V arrive interleaved across heads, so a single head's rows sit
/// `dim` floats apart - at d=1024 that is a 4 KiB stride, which defeats the
/// prefetcher and touches a separate page per token. Each task therefore packs
/// its head first: K transposed to `[head_dim, seq]` and V as `[seq, head_dim]`.
///
/// Softmax is not a pass of its own. `scoreBlock` tracks the row maximum while
/// the block is still in registers, `expRow` exponentiates in place, and the
/// reciprocal of the sum is handed to `accumulateValues` to fold into its
/// stores - three passes over `seq_len` scores become one.
///
/// The transpose is what makes the scores cheap. Scoring a query against a key
/// is a 64-element dot product, and computing it as one costs a horizontal
/// vector reduction per score - more cycles than the eight fused multiply-adds
/// it reduces. With K transposed, one vector instead spans eight *keys* at a
/// fixed channel, so a block of queries accumulates eight scores each with no
/// reduction at all, and the results land contiguously in the score rows.
fn attention(
    allocator: std.mem.Allocator,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    num_heads: usize,
) !Tensor {
    const batch = q.shape[0];
    const seq = q.shape[1];
    const dim = q.shape[2];
    const head_dim = dim / num_heads;

    const out_shape = [_]usize{ batch, seq, dim };
    var out = try Tensor.init(allocator, &out_shape);
    errdefer out.deinit();

    // Per lane rather than per task: at 1008x1008 a windowed layer dispatches
    // 144 tasks and a global one 64, and a buffer each would be hundreds of
    // megabytes.
    const lanes = parallel.laneCount();
    // Two extra Q_BLOCK slots per lane: the row maxima out of `scoreBlock` and
    // the softmax reciprocals `accumulateValues` folds into its stores.
    const per_lane = 2 * seq * head_dim + Q_BLOCK * seq + Q_BLOCK * head_dim + 2 * Q_BLOCK;
    const scratch = try allocator.alloc(f32, lanes * per_lane);
    defer allocator.free(scratch);

    // A task per (batch, head) leaves the four global layers with only sixteen
    // of them, which is too coarse for eight lanes of unequal speed: the join
    // waits on an efficiency core's second task while the performance cores sit
    // idle. Splitting the query range as well gives the pool enough pieces to
    // even out. The extra cost is re-packing K and V per piece, which is under a
    // tenth of a second across the whole forward pass.
    const min_tasks = 8 * lanes;
    const q_chunks = blk: {
        const heads_tasks = batch * num_heads;
        if (heads_tasks >= min_tasks) break :blk 1;
        const want = (min_tasks + heads_tasks - 1) / heads_tasks;
        const blocks = (seq + Q_BLOCK - 1) / Q_BLOCK;
        break :blk @max(1, @min(want, blocks));
    };
    const chunk_blocks = ((seq + Q_BLOCK - 1) / Q_BLOCK + q_chunks - 1) / q_chunks;

    const Context = struct {
        q: Tensor,
        k: Tensor,
        v: Tensor,
        out: *Tensor,
        scratch: []f32,
        per_lane: usize,
        num_heads: usize,
        seq: usize,
        dim: usize,
        head_dim: usize,
        scale: f32,
        q_chunks: usize,
        chunk_blocks: usize,
    };

    var ctx = Context{
        .q = q,
        .k = k,
        .v = v,
        .out = &out,
        .scratch = scratch,
        .per_lane = per_lane,
        .num_heads = num_heads,
        .seq = seq,
        .dim = dim,
        .head_dim = head_dim,
        .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
        .q_chunks = q_chunks,
        .chunk_blocks = chunk_blocks,
    };

    parallel.parallelFor(allocator, batch * num_heads * q_chunks, &ctx, struct {
        fn worker(c: *Context, start: usize, end: usize) void {
            const seq_len = c.seq;
            const model_dim = c.dim;
            const hd = c.head_dim;

            const lane = c.scratch[parallel.laneId() * c.per_lane ..][0..c.per_lane];
            const k_t = lane[0 .. seq_len * hd]; // [head_dim, seq]
            const v_pack = lane[seq_len * hd ..][0 .. seq_len * hd]; // [seq, head_dim]
            const scores = lane[2 * seq_len * hd ..][0 .. Q_BLOCK * seq_len];
            const q_pack = lane[2 * seq_len * hd + Q_BLOCK * seq_len ..][0 .. Q_BLOCK * hd];
            const tail = 2 * seq_len * hd + Q_BLOCK * seq_len + Q_BLOCK * hd;
            const row_max = lane[tail ..][0..Q_BLOCK];
            const row_inv = lane[tail + Q_BLOCK ..][0..Q_BLOCK];

            var out_rows: [Q_BLOCK][]f32 = undefined;

            for (start..end) |task| {
                const head_task = task / c.q_chunks;
                const chunk = task % c.q_chunks;
                const b = head_task / c.num_heads;
                const head_off = (head_task % c.num_heads) * hd;

                const q_first = chunk * c.chunk_blocks * Q_BLOCK;
                if (q_first >= seq_len) continue;
                const q_last = @min(seq_len, q_first + c.chunk_blocks * Q_BLOCK);

                for (0..seq_len) |i| {
                    const base = (b * seq_len + i) * model_dim + head_off;
                    const k_src = c.k.data[base..][0..hd];
                    for (0..hd) |d| k_t[d * seq_len + i] = k_src[d];
                    @memcpy(v_pack[i * hd ..][0..hd], c.v.data[base..][0..hd]);
                }

                var q0: usize = q_first;
                while (q0 < q_last) : (q0 += Q_BLOCK) {
                    const rows = @min(Q_BLOCK, q_last - q0);
                    for (0..rows) |i| {
                        const base = (b * seq_len + q0 + i) * model_dim + head_off;
                        @memcpy(q_pack[i * hd ..][0..hd], c.q.data[base..][0..hd]);
                        out_rows[i] = c.out.data[base..][0..hd];
                        @memset(out_rows[i], 0.0);
                    }

                    scoreBlock(q_pack, k_t, scores, row_max, rows, seq_len, hd, c.scale);

                    for (0..rows) |i| {
                        row_inv[i] = expRow(scores[i * seq_len ..][0..seq_len], row_max[i]);
                    }

                    accumulateValues(scores, v_pack, &out_rows, row_inv, rows, seq_len, hd);
                }
            }
        }
    }.worker);

    return out;
}

/// `scores[i, ki] = scale * dot(q_pack[i], key ki)` for a block of queries.
///
/// `k_t` is `[head_dim, seq]`, so each vector load covers `VEC` consecutive keys
/// at one channel and the accumulators are already the finished scores.
fn scoreBlock(
    q_pack: []const f32,
    k_t: []const f32,
    scores: []f32,
    row_max: []f32,
    rows: usize,
    seq_len: usize,
    hd: usize,
    scale: f32,
) void {
    const scale_v: Vec = @splat(scale);
    const neg_inf = -std.math.inf(f32);

    // The row maximum softmax needs is tracked here rather than in a pass of
    // its own: the scores are in registers exactly once, and a vector max per
    // store is far cheaper than re-reading the whole block from L2.
    var max_v: [Q_BLOCK]Vec = @splat(@as(Vec, @splat(neg_inf)));
    for (0..rows) |i| row_max[i] = neg_inf;

    var k0: usize = 0;
    if (rows == Q_BLOCK) {
        while (k0 + VEC <= seq_len) : (k0 += VEC) {
            var acc: [Q_BLOCK]Vec = @splat(@as(Vec, @splat(0.0)));

            for (0..hd) |d| {
                const kv: Vec = k_t[d * seq_len + k0 ..][0..VEC].*;
                inline for (0..Q_BLOCK) |i| {
                    acc[i] += @as(Vec, @splat(q_pack[i * hd + d])) * kv;
                }
            }

            inline for (0..Q_BLOCK) |i| {
                const scaled = acc[i] * scale_v;
                scores[i * seq_len + k0 ..][0..VEC].* = scaled;
                max_v[i] = @max(max_v[i], scaled);
            }
        }
        for (0..rows) |i| row_max[i] = @reduce(.Max, max_v[i]);
    }

    // Ragged tail, and every block when the last one is short.
    while (k0 < seq_len) : (k0 += 1) {
        for (0..rows) |i| {
            var sum: f32 = 0.0;
            for (0..hd) |d| sum += q_pack[i * hd + d] * k_t[d * seq_len + k0];
            const v = sum * scale;
            scores[i * seq_len + k0] = v;
            row_max[i] = @max(row_max[i], v);
        }
    }
}

/// `out[i] = sum_k scores[i, k] * v_pack[k]`, tiled so a `Q_SUB x D_TILE` patch
/// of the result stays in registers for the whole reduction.
fn accumulateValues(
    scores: []const f32,
    v_pack: []const f32,
    out_rows: *[Q_BLOCK][]f32,
    row_inv: []const f32,
    rows: usize,
    seq_len: usize,
    hd: usize,
) void {
    const lanes = D_TILE / VEC;

    var q0: usize = 0;
    while (q0 < rows) : (q0 += Q_SUB) {
        const sub = @min(Q_SUB, rows - q0);

        var d0: usize = 0;
        while (d0 < hd) : (d0 += D_TILE) {
            const width = @min(D_TILE, hd - d0);

            if (sub == Q_SUB and width == D_TILE) {
                var acc: [Q_SUB][lanes]Vec = @splat(@as([lanes]Vec, @splat(@as(Vec, @splat(0.0)))));

                for (0..seq_len) |ki| {
                    const v_row = v_pack[ki * hd + d0 ..];
                    inline for (0..lanes) |j| {
                        const vv: Vec = v_row[j * VEC ..][0..VEC].*;
                        inline for (0..Q_SUB) |i| {
                            acc[i][j] += @as(Vec, @splat(scores[(q0 + i) * seq_len + ki])) * vv;
                        }
                    }
                }

                inline for (0..Q_SUB) |i| {
                    const inv: Vec = @splat(row_inv[q0 + i]);
                    inline for (0..lanes) |j| {
                        out_rows[q0 + i][d0 + j * VEC ..][0..VEC].* = acc[i][j] * inv;
                    }
                }
            } else {
                for (0..sub) |i| {
                    const dst = out_rows[q0 + i][d0..][0..width];
                    @memset(dst, 0.0);
                    for (0..seq_len) |ki| {
                        const w = scores[(q0 + i) * seq_len + ki];
                        if (w == 0.0) continue;
                        const v_row = v_pack[ki * hd + d0 ..][0..width];
                        for (0..width) |j| dst[j] += w * v_row[j];
                    }
                    const inv = row_inv[q0 + i];
                    for (dst) |*x| x.* *= inv;
                }
            }
        }
    }
}

/// Exponentiates one score row in place against a maximum `scoreBlock` already
/// found, and returns the reciprocal of the sum.
///
/// The softmax denominator is not applied here. Dividing the row would be a
/// third full pass over `seq_len` scores; folding the reciprocal into the value
/// accumulation instead applies it to `head_dim` outputs, which at seq 5184 and
/// head_dim 64 is eighty times less work.
inline fn expRow(row: []f32, max_val: f32) f32 {
    const V = math.VEC_SIZE;

    var sum: f32 = 0.0;
    const max_splat: math.F32Vec = @splat(max_val);
    var sum_vec: math.F32Vec = @splat(0.0);
    var i: usize = 0;
    while (i + V <= row.len) : (i += V) {
        const v: math.F32Vec = row[i..][0..V].*;
        const e = math.expVec(v - max_splat);
        row[i..][0..V].* = e;
        sum_vec += e;
    }
    sum = @reduce(.Add, sum_vec);
    while (i < row.len) : (i += 1) {
        row[i] = @exp(row[i] - max_val);
        sum += row[i];
    }

    return 1.0 / (sum + 1e-12);
}

pub const Windows = struct {
    tensor: Tensor, // [num_windows, window*window, channels]
    padded_h: usize,
    padded_w: usize,
};

/// Splits `[1, H, W, C]` into `[num_windows, window*window, C]`, zero-padding the
/// bottom/right edges when H or W is not a multiple of `window`.
pub fn windowPartition(allocator: std.mem.Allocator, x: Tensor, window: usize) !Windows {
    std.debug.assert(x.shape.len == 4);
    const h = x.shape[1];
    const w = x.shape[2];
    const c = x.shape[3];

    const pad_h = (window - h % window) % window;
    const pad_w = (window - w % window) % window;
    const padded_h = h + pad_h;
    const padded_w = w + pad_w;

    const win_h = padded_h / window;
    const win_w = padded_w / window;
    const num_windows = win_h * win_w;

    const shape = [_]usize{ num_windows, window * window, c };
    var out = try Tensor.initZeros(allocator, &shape);
    errdefer out.deinit();

    const Context = struct {
        src: []const f32,
        dst: []f32,
        h: usize,
        w: usize,
        c: usize,
        window: usize,
        win_w: usize,
    };

    const ctx = Context{
        .src = x.data,
        .dst = out.data,
        .h = h,
        .w = w,
        .c = c,
        .window = window,
        .win_w = win_w,
    };

    // One task per window row: within a row the copied spans are contiguous.
    parallel.parallelFor(allocator, num_windows * window, ctx, struct {
        fn worker(cx: Context, start: usize, end: usize) void {
            for (start..end) |task| {
                const win_idx = task / cx.window;
                const ih = task % cx.window;
                const wh = win_idx / cx.win_w;
                const ww = win_idx % cx.win_w;

                const src_h = wh * cx.window + ih;
                if (src_h >= cx.h) continue;

                const cols = @min(cx.window, cx.w -| (ww * cx.window));
                if (cols == 0) continue;

                const src_off = (src_h * cx.w + ww * cx.window) * cx.c;
                const dst_off = (win_idx * cx.window * cx.window + ih * cx.window) * cx.c;
                @memcpy(cx.dst[dst_off..][0 .. cols * cx.c], cx.src[src_off..][0 .. cols * cx.c]);
            }
        }
    }.worker);

    return Windows{ .tensor = out, .padded_h = padded_h, .padded_w = padded_w };
}

/// Inverse of `windowPartition`, cropping the padding back off.
pub fn windowUnpartition(
    allocator: std.mem.Allocator,
    windows: Tensor,
    window: usize,
    padded_h: usize,
    padded_w: usize,
    h: usize,
    w: usize,
) !Tensor {
    // Only the padded width matters here: it fixes the window stride per row.
    _ = padded_h;

    const c = windows.shape[2];
    const win_w = padded_w / window;

    const shape = [_]usize{ 1, h, w, c };
    var out = try Tensor.init(allocator, &shape);
    errdefer out.deinit();

    const Context = struct {
        src: []const f32,
        dst: []f32,
        w: usize,
        c: usize,
        window: usize,
        win_w: usize,
    };

    const ctx = Context{
        .src = windows.data,
        .dst = out.data,
        .w = w,
        .c = c,
        .window = window,
        .win_w = win_w,
    };

    parallel.parallelFor(allocator, h, ctx, struct {
        fn worker(cx: Context, start: usize, end: usize) void {
            for (start..end) |oh| {
                const wh = oh / cx.window;
                const ih = oh % cx.window;

                var ow: usize = 0;
                while (ow < cx.w) {
                    const ww = ow / cx.window;
                    const iw = ow % cx.window;
                    const run = @min(cx.window - iw, cx.w - ow);
                    const win_idx = wh * cx.win_w + ww;

                    const src_off = (win_idx * cx.window * cx.window + ih * cx.window + iw) * cx.c;
                    const dst_off = (oh * cx.w + ow) * cx.c;
                    @memcpy(cx.dst[dst_off..][0 .. run * cx.c], cx.src[src_off..][0 .. run * cx.c]);
                    ow += run;
                }
            }
        }
    }.worker);

    return out;
}

pub const VisionEncoder = struct {
    allocator: std.mem.Allocator,
    config: VisionConfig,

    pub const Output = struct {
        allocator: std.mem.Allocator,
        /// Backbone tokens, `[1, grid*grid, hidden_size]`.
        last_hidden_state: Tensor,
        /// One `[1, fpn_hidden_size, h, w]` map per entry in `scale_factors`.
        fpn_features: []Tensor,

        pub fn deinit(self: *Output) void {
            self.last_hidden_state.deinit();
            for (self.fpn_features) |*f| f.deinit();
            self.allocator.free(self.fpn_features);
        }
    };

    pub fn init(allocator: std.mem.Allocator, config: VisionConfig) VisionEncoder {
        return VisionEncoder{ .allocator = allocator, .config = config };
    }

    /// Resizes an image to `image_size` and normalises it the way
    /// `Sam3ImageProcessorFast` does: rescale by 1/255, then (x - 0.5) / 0.5.
    pub fn preprocess(self: VisionEncoder, img: ImageRGB) !Tensor {
        const raw_shape = [_]usize{ 1, 3, img.height, img.width };
        var raw = try Tensor.init(self.allocator, &raw_shape);
        defer raw.deinit();

        for (0..img.height) |y| {
            for (0..img.width) |x| {
                const idx = (y * img.width + x) * 3;
                for (0..3) |ch| {
                    raw.set4(0, ch, y, x, @as(f32, @floatFromInt(img.data[idx + ch])) / 255.0);
                }
            }
        }

        var resized = try ops.bilinearUpsample(self.allocator, raw, self.config.image_size, self.config.image_size, false);
        errdefer resized.deinit();

        for (resized.data) |*v| {
            v.* = (v.* - 0.5) / 0.5;
        }
        return resized;
    }

    /// Runs the backbone and neck. `pixel_values` is `[1, 3, image_size, image_size]`.
    pub fn forward(self: VisionEncoder, pixel_values: Tensor, weights: *WeightStore) !Output {
        std.debug.assert(pixel_values.shape.len == 4);
        std.debug.assert(pixel_values.shape[1] == 3);

        const cfg = self.config;
        const d = cfg.hidden_size;
        const grid = cfg.grid();
        var name_buf: [256]u8 = undefined;

        // --- Patch embedding: Conv2d(3 -> d, k=patch, s=patch, bias=False) ---
        const patch_w = try weights.getOrInit(
            try std.fmt.bufPrint(&name_buf, "{s}.backbone.embeddings.patch_embeddings.projection.weight", .{cfg.prefix}),
            &[_]usize{ d, 3, cfg.patch_size, cfg.patch_size },
            .kaiming,
            1,
        );

        var patches = try ops.conv2d(self.allocator, pixel_values, patch_w, null, cfg.patch_size, 0);
        defer patches.deinit();

        // [1, d, grid, grid] -> [1, grid, grid, d]
        var x = try patches.permute(self.allocator, &[_]usize{ 0, 2, 3, 1 });
        defer x.deinit();

        // --- Tiled position embeddings ---
        const pretrain_grid = cfg.pretrain_image_size / cfg.patch_size;
        const pos = try weights.getOrInit(
            try std.fmt.bufPrint(&name_buf, "{s}.backbone.embeddings.position_embeddings", .{cfg.prefix}),
            &[_]usize{ 1, pretrain_grid * pretrain_grid, d },
            .xavier,
            2,
        );

        for (0..grid) |h| {
            for (0..grid) |w| {
                const src = pos.data[((h % pretrain_grid) * pretrain_grid + (w % pretrain_grid)) * d ..][0..d];
                const dst = x.data[(h * grid + w) * d ..][0..d];
                for (0..d) |i| dst[i] += src[i];
            }
        }

        // --- Pre-layer norm (applied once, before the stack) ---
        {
            const g = try weights.getOrInit(
                try std.fmt.bufPrint(&name_buf, "{s}.backbone.layer_norm.weight", .{cfg.prefix}),
                &[_]usize{d},
                .ones,
                3,
            );
            const b = try weights.getOrInit(
                try std.fmt.bufPrint(&name_buf, "{s}.backbone.layer_norm.bias", .{cfg.prefix}),
                &[_]usize{d},
                .zeros,
                4,
            );
            const normed = try math.layerNorm(self.allocator, x, g, b, cfg.layer_norm_eps);
            x.deinit();
            x = normed;
        }

        // RoPE tables are shared by all layers of the same kind: windowed layers
        // all see a window_size grid at scale 1, global layers the full token
        // grid at scale window_size/grid.
        var rope_window = try RopeTable.init(
            self.allocator,
            cfg.window_size,
            cfg.window_size,
            cfg.headDim(),
            cfg.rope_theta,
            1.0,
        );
        defer rope_window.deinit();

        var rope_global = try RopeTable.init(
            self.allocator,
            grid,
            grid,
            cfg.headDim(),
            cfg.rope_theta,
            @as(f32, @floatFromInt(cfg.window_size)) / @as(f32, @floatFromInt(grid)),
        );
        defer rope_global.deinit();

        for (0..cfg.num_layers) |layer| {
            try self.forwardLayer(&x, layer, weights, rope_window, rope_global, &name_buf);
        }

        // --- Outputs ---
        var last_hidden_state = try x.clone(self.allocator);
        errdefer last_hidden_state.deinit();
        try last_hidden_state.reshape(&[_]usize{ 1, grid * grid, d });

        // [1, grid, grid, d] -> [1, d, grid, grid] for the convolutional neck.
        var spatial = try x.permute(self.allocator, &[_]usize{ 0, 3, 1, 2 });
        defer spatial.deinit();

        var fpn_features = try self.allocator.alloc(Tensor, cfg.scale_factors.len);
        errdefer self.allocator.free(fpn_features);

        var built: usize = 0;
        errdefer for (fpn_features[0..built]) |*f| f.deinit();

        for (cfg.scale_factors, 0..) |scale, i| {
            fpn_features[i] = try self.fpnLayer(spatial, i, scale, weights, &name_buf);
            built += 1;
        }

        return Output{
            .allocator = self.allocator,
            .last_hidden_state = last_hidden_state,
            .fpn_features = fpn_features,
        };
    }

    /// One pre-norm ViT layer: RoPE attention (windowed or global) then MLP.
    fn forwardLayer(
        self: VisionEncoder,
        x: *Tensor,
        layer: usize,
        weights: *WeightStore,
        rope_window: RopeTable,
        rope_global: RopeTable,
        name_buf: []u8,
    ) !void {
        const cfg = self.config;
        const d = cfg.hidden_size;
        const grid = cfg.grid();
        const global = cfg.isGlobalLayer(layer);

        const ln1_g = try weights.getOrInit(
            try std.fmt.bufPrint(name_buf, "{s}.backbone.layers.{d}.layer_norm1.weight", .{ cfg.prefix, layer }),
            &[_]usize{d},
            .ones,
            100 + layer,
        );
        const ln1_b = try weights.getOrInit(
            try std.fmt.bufPrint(name_buf, "{s}.backbone.layers.{d}.layer_norm1.bias", .{ cfg.prefix, layer }),
            &[_]usize{d},
            .zeros,
            200 + layer,
        );

        var normed = try math.layerNorm(self.allocator, x.*, ln1_g, ln1_b, cfg.layer_norm_eps);
        defer normed.deinit();

        // Reshape into the sequence layout attention wants: windowed layers get
        // one sequence per window, global layers a single grid*grid sequence.
        var seq: Tensor = undefined;
        var padded_h: usize = grid;
        var padded_w: usize = grid;
        if (global) {
            seq = try normed.clone(self.allocator);
            try seq.reshape(&[_]usize{ 1, grid * grid, d });
        } else {
            const win = try windowPartition(self.allocator, normed, cfg.window_size);
            seq = win.tensor;
            padded_h = win.padded_h;
            padded_w = win.padded_w;
        }
        defer seq.deinit();

        var attn_out = try self.attentionBlock(
            seq,
            layer,
            weights,
            if (global) rope_global else rope_window,
            name_buf,
        );
        defer attn_out.deinit();

        // Back to [1, grid, grid, d] and add the residual.
        if (global) {
            try attn_out.reshape(&[_]usize{ 1, grid, grid, d });
            x.addInPlace(attn_out);
        } else {
            var merged = try windowUnpartition(
                self.allocator,
                attn_out,
                cfg.window_size,
                padded_h,
                padded_w,
                grid,
                grid,
            );
            defer merged.deinit();
            x.addInPlace(merged);
        }

        // --- MLP ---
        const ln2_g = try weights.getOrInit(
            try std.fmt.bufPrint(name_buf, "{s}.backbone.layers.{d}.layer_norm2.weight", .{ cfg.prefix, layer }),
            &[_]usize{d},
            .ones,
            300 + layer,
        );
        const ln2_b = try weights.getOrInit(
            try std.fmt.bufPrint(name_buf, "{s}.backbone.layers.{d}.layer_norm2.bias", .{ cfg.prefix, layer }),
            &[_]usize{d},
            .zeros,
            400 + layer,
        );

        var normed2 = try math.layerNorm(self.allocator, x.*, ln2_g, ln2_b, cfg.layer_norm_eps);
        defer normed2.deinit();

        const fc1_w = try weights.getOrInit(
            try std.fmt.bufPrint(name_buf, "{s}.backbone.layers.{d}.mlp.fc1.weight", .{ cfg.prefix, layer }),
            &[_]usize{ cfg.intermediate_size, d },
            .kaiming,
            500 + layer,
        );
        const fc1_b = try weights.getOrInit(
            try std.fmt.bufPrint(name_buf, "{s}.backbone.layers.{d}.mlp.fc1.bias", .{ cfg.prefix, layer }),
            &[_]usize{cfg.intermediate_size},
            .zeros,
            600 + layer,
        );

        var hidden = try math.linear(self.allocator, normed2, fc1_w, fc1_b);
        defer hidden.deinit();
        math.applyActivation(&hidden, .gelu_exact);

        const fc2_w = try weights.getOrInit(
            try std.fmt.bufPrint(name_buf, "{s}.backbone.layers.{d}.mlp.fc2.weight", .{ cfg.prefix, layer }),
            &[_]usize{ d, cfg.intermediate_size },
            .kaiming,
            700 + layer,
        );
        const fc2_b = try weights.getOrInit(
            try std.fmt.bufPrint(name_buf, "{s}.backbone.layers.{d}.mlp.fc2.bias", .{ cfg.prefix, layer }),
            &[_]usize{d},
            .zeros,
            800 + layer,
        );

        var mlp_out = try math.linear(self.allocator, hidden, fc2_w, fc2_b);
        defer mlp_out.deinit();
        x.addInPlace(mlp_out);
    }

    /// q/k/v projections, 2D RoPE, softmax attention, output projection.
    fn attentionBlock(
        self: VisionEncoder,
        seq: Tensor,
        layer: usize,
        weights: *WeightStore,
        rope: RopeTable,
        name_buf: []u8,
    ) !Tensor {
        const cfg = self.config;
        const d = cfg.hidden_size;

        const proj = struct {
            fn get(
                w: *WeightStore,
                buf: []u8,
                prefix: []const u8,
                l: usize,
                which: []const u8,
                kind: []const u8,
                shape: []const usize,
                seed: u64,
            ) !Tensor {
                const name = try std.fmt.bufPrint(buf, "{s}.backbone.layers.{d}.attention.{s}.{s}", .{ prefix, l, which, kind });
                return w.getOrInit(name, shape, if (std.mem.eql(u8, kind, "weight")) .xavier else .zeros, seed);
            }
        };

        const w_shape = [_]usize{ d, d };
        const b_shape = [_]usize{d};

        const q_w = try proj.get(weights, name_buf, cfg.prefix, layer, "q_proj", "weight", &w_shape, 900 + layer);
        const q_b = try proj.get(weights, name_buf, cfg.prefix, layer, "q_proj", "bias", &b_shape, 1000 + layer);
        const k_w = try proj.get(weights, name_buf, cfg.prefix, layer, "k_proj", "weight", &w_shape, 1100 + layer);
        const k_b = try proj.get(weights, name_buf, cfg.prefix, layer, "k_proj", "bias", &b_shape, 1200 + layer);
        const v_w = try proj.get(weights, name_buf, cfg.prefix, layer, "v_proj", "weight", &w_shape, 1300 + layer);
        const v_b = try proj.get(weights, name_buf, cfg.prefix, layer, "v_proj", "bias", &b_shape, 1400 + layer);
        const o_w = try proj.get(weights, name_buf, cfg.prefix, layer, "o_proj", "weight", &w_shape, 1500 + layer);
        const o_b = try proj.get(weights, name_buf, cfg.prefix, layer, "o_proj", "bias", &b_shape, 1600 + layer);

        var q = try math.linear(self.allocator, seq, q_w, q_b);
        defer q.deinit();
        var k = try math.linear(self.allocator, seq, k_w, k_b);
        defer k.deinit();
        var v = try math.linear(self.allocator, seq, v_w, v_b);
        defer v.deinit();

        applyRope(self.allocator, &q, rope, cfg.num_heads);
        applyRope(self.allocator, &k, rope, cfg.num_heads);

        var attn = try attention(self.allocator, q, k, v, cfg.num_heads);
        defer attn.deinit();

        return math.linear(self.allocator, attn, o_w, o_b);
    }

    /// One FPN level: rescale the token grid, then 1x1 and 3x3 projections to
    /// `fpn_hidden_size` channels.
    fn fpnLayer(
        self: VisionEncoder,
        spatial: Tensor,
        index: usize,
        scale: f32,
        weights: *WeightStore,
        name_buf: []u8,
    ) !Tensor {
        const cfg = self.config;
        const d = cfg.hidden_size;
        const fpn_d = cfg.fpn_hidden_size;

        var current: Tensor = undefined;
        var intermediate_channels: usize = d;

        if (scale == 4.0) {
            const w0 = try weights.getOrInit(
                try std.fmt.bufPrint(name_buf, "{s}.fpn_layers.{d}.scale_layers.0.weight", .{ cfg.neck_prefix, index }),
                &[_]usize{ d, d / 2, 2, 2 },
                .kaiming,
                1700 + index,
            );
            const b0 = try weights.getOrInit(
                try std.fmt.bufPrint(name_buf, "{s}.fpn_layers.{d}.scale_layers.0.bias", .{ cfg.neck_prefix, index }),
                &[_]usize{d / 2},
                .zeros,
                1800 + index,
            );
            var up1 = try ops.convTranspose2d(self.allocator, spatial, w0, b0, 2, 0);
            defer up1.deinit();
            math.applyActivation(&up1, .gelu_exact);

            // scale_layers.1 is the GELU above, so the second deconv is index 2.
            const w2 = try weights.getOrInit(
                try std.fmt.bufPrint(name_buf, "{s}.fpn_layers.{d}.scale_layers.2.weight", .{ cfg.neck_prefix, index }),
                &[_]usize{ d / 2, d / 4, 2, 2 },
                .kaiming,
                1900 + index,
            );
            const b2 = try weights.getOrInit(
                try std.fmt.bufPrint(name_buf, "{s}.fpn_layers.{d}.scale_layers.2.bias", .{ cfg.neck_prefix, index }),
                &[_]usize{d / 4},
                .zeros,
                2000 + index,
            );
            current = try ops.convTranspose2d(self.allocator, up1, w2, b2, 2, 0);
            intermediate_channels = d / 4;
        } else if (scale == 2.0) {
            const w0 = try weights.getOrInit(
                try std.fmt.bufPrint(name_buf, "{s}.fpn_layers.{d}.scale_layers.0.weight", .{ cfg.neck_prefix, index }),
                &[_]usize{ d, d / 2, 2, 2 },
                .kaiming,
                2100 + index,
            );
            const b0 = try weights.getOrInit(
                try std.fmt.bufPrint(name_buf, "{s}.fpn_layers.{d}.scale_layers.0.bias", .{ cfg.neck_prefix, index }),
                &[_]usize{d / 2},
                .zeros,
                2200 + index,
            );
            current = try ops.convTranspose2d(self.allocator, spatial, w0, b0, 2, 0);
            intermediate_channels = d / 2;
        } else if (scale == 1.0) {
            current = try spatial.clone(self.allocator);
        } else if (scale == 0.5) {
            current = try ops.maxPool2d(self.allocator, spatial, 2, 2);
        } else {
            return error.UnsupportedFPNScale;
        }
        defer current.deinit();

        const proj1_w = try weights.getOrInit(
            try std.fmt.bufPrint(name_buf, "{s}.fpn_layers.{d}.proj1.weight", .{ cfg.neck_prefix, index }),
            &[_]usize{ fpn_d, intermediate_channels, 1, 1 },
            .kaiming,
            2300 + index,
        );
        const proj1_b = try weights.getOrInit(
            try std.fmt.bufPrint(name_buf, "{s}.fpn_layers.{d}.proj1.bias", .{ cfg.neck_prefix, index }),
            &[_]usize{fpn_d},
            .zeros,
            2400 + index,
        );

        var projected = try ops.conv2d(self.allocator, current, proj1_w, proj1_b, 1, 0);
        defer projected.deinit();

        const proj2_w = try weights.getOrInit(
            try std.fmt.bufPrint(name_buf, "{s}.fpn_layers.{d}.proj2.weight", .{ cfg.neck_prefix, index }),
            &[_]usize{ fpn_d, fpn_d, 3, 3 },
            .kaiming,
            2500 + index,
        );
        const proj2_b = try weights.getOrInit(
            try std.fmt.bufPrint(name_buf, "{s}.fpn_layers.{d}.proj2.bias", .{ cfg.neck_prefix, index }),
            &[_]usize{fpn_d},
            .zeros,
            2600 + index,
        );

        return ops.conv2d(self.allocator, projected, proj2_w, proj2_b, 1, 1);
    }
};

test "window partition round-trips" {
    const allocator = std.testing.allocator;

    // 5x7 grid with 2 channels, window 3: both dimensions need padding.
    const shape = [_]usize{ 1, 5, 7, 2 };
    var x = try Tensor.init(allocator, &shape);
    defer x.deinit();
    for (x.data, 0..) |*v, i| v.* = @floatFromInt(i);

    var win = try windowPartition(allocator, x, 3);
    defer win.tensor.deinit();
    try std.testing.expectEqual(@as(usize, 6), win.tensor.shape[0]); // 2x3 windows
    try std.testing.expectEqual(@as(usize, 6), win.padded_h);
    try std.testing.expectEqual(@as(usize, 9), win.padded_w);

    var back = try windowUnpartition(allocator, win.tensor, 3, win.padded_h, win.padded_w, 5, 7);
    defer back.deinit();

    for (x.data, back.data) |expected, actual| {
        try std.testing.expectEqual(expected, actual);
    }
}

test "RoPE preserves per-head vector norms" {
    const allocator = std.testing.allocator;

    const head_dim = 8;
    const num_heads = 2;
    var table = try RopeTable.init(allocator, 3, 3, head_dim, 10000.0, 1.0);
    defer table.deinit();

    const shape = [_]usize{ 1, 9, num_heads * head_dim };
    var x = try Tensor.initRandom(allocator, &shape, 7, -1.0, 1.0);
    defer x.deinit();

    var before = try x.clone(allocator);
    defer before.deinit();

    applyRope(allocator, &x, table, num_heads);

    for (0..9) |s| {
        for (0..num_heads) |h| {
            var norm_before: f32 = 0.0;
            var norm_after: f32 = 0.0;
            for (0..head_dim) |d| {
                const idx = s * num_heads * head_dim + h * head_dim + d;
                norm_before += before.data[idx] * before.data[idx];
                norm_after += x.data[idx] * x.data[idx];
            }
            try std.testing.expectApproxEqAbs(norm_before, norm_after, 1e-4);
        }
    }

    // Position 0 has zero rotation angle, so it must come back unchanged.
    for (0..num_heads * head_dim) |i| {
        try std.testing.expectApproxEqAbs(before.data[i], x.data[i], 1e-6);
    }
}

test "vision encoder runs end to end at a small resolution" {
    const allocator = std.testing.allocator;

    const cfg = VisionConfig{
        .hidden_size = 32,
        .num_heads = 2,
        .num_layers = 2,
        .intermediate_size = 64,
        .patch_size = 14,
        .image_size = 14 * 6,
        .pretrain_image_size = 14 * 3,
        .window_size = 4,
        .global_attn_indexes = &.{1},
        .fpn_hidden_size = 8,
        .prefix = "test_vision",
        .neck_prefix = "test_vision.neck",
    };

    var weights = WeightStore.init(allocator);
    defer weights.deinit();

    const img_shape = [_]usize{ 1, 3, cfg.image_size, cfg.image_size };
    var pixels = try Tensor.initRandom(allocator, &img_shape, 42, -1.0, 1.0);
    defer pixels.deinit();

    const encoder = VisionEncoder.init(allocator, cfg);
    var out = try encoder.forward(pixels, &weights);
    defer out.deinit();

    try std.testing.expectEqual(@as(usize, 6 * 6), out.last_hidden_state.shape[1]);
    try std.testing.expectEqual(@as(usize, 32), out.last_hidden_state.shape[2]);
    try std.testing.expectEqual(@as(usize, 4), out.fpn_features.len);

    // 4x / 2x / 1x / 0.5x of the 6x6 token grid.
    try std.testing.expectEqual(@as(usize, 24), out.fpn_features[0].shape[2]);
    try std.testing.expectEqual(@as(usize, 12), out.fpn_features[1].shape[2]);
    try std.testing.expectEqual(@as(usize, 6), out.fpn_features[2].shape[2]);
    try std.testing.expectEqual(@as(usize, 3), out.fpn_features[3].shape[2]);

    for (out.fpn_features) |f| {
        try std.testing.expectEqual(@as(usize, 8), f.shape[1]);
        for (f.data) |v| try std.testing.expect(std.math.isFinite(v));
    }
}
