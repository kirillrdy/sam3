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
//! `assets/sam3/config.json` for every hyper-parameter below.

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
fn applyRope(x: *Tensor, table: RopeTable, num_heads: usize) void {
    std.debug.assert(x.shape.len == 3);
    const batch = x.shape[0];
    const seq = x.shape[1];
    const dim = x.shape[2];
    const head_dim = table.head_dim;
    std.debug.assert(seq == table.seq_len);
    std.debug.assert(dim == num_heads * head_dim);

    for (0..batch) |b| {
        for (0..seq) |s| {
            const row = x.data[(b * seq + s) * dim ..][0..dim];
            const cos_row = table.cos[s * head_dim ..][0..head_dim];
            const sin_row = table.sin[s * head_dim ..][0..head_dim];

            for (0..num_heads) |h| {
                const head = row[h * head_dim ..][0..head_dim];
                var d: usize = 0;
                while (d < head_dim) : (d += 2) {
                    const a = head[d];
                    const bb = head[d + 1];
                    head[d] = a * cos_row[d] - bb * sin_row[d];
                    head[d + 1] = bb * cos_row[d + 1] + a * sin_row[d + 1];
                }
            }
        }
    }
}

/// Softmax attention over `[batch, seq, dim]` with pre-applied RoPE, running one
/// (batch, head) pair per task.
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

    const num_tasks = batch * num_heads;
    const scratch = try allocator.alloc(f32, num_tasks * seq);
    defer allocator.free(scratch);

    const Context = struct {
        q: Tensor,
        k: Tensor,
        v: Tensor,
        out: *Tensor,
        scratch: []f32,
        num_heads: usize,
        seq: usize,
        dim: usize,
        head_dim: usize,
        scale: f32,
    };

    var ctx = Context{
        .q = q,
        .k = k,
        .v = v,
        .out = &out,
        .scratch = scratch,
        .num_heads = num_heads,
        .seq = seq,
        .dim = dim,
        .head_dim = head_dim,
        .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
    };

    parallel.parallelFor(allocator, num_tasks, &ctx, struct {
        fn worker(c: *Context, start: usize, end: usize) void {
            const seq_len = c.seq;
            const model_dim = c.dim;
            const hd = c.head_dim;

            for (start..end) |task| {
                const b = task / c.num_heads;
                const h = task % c.num_heads;
                const head_off = h * hd;
                const scores = c.scratch[task * seq_len ..][0..seq_len];

                for (0..seq_len) |qi| {
                    const q_vec = c.q.data[(b * seq_len + qi) * model_dim + head_off ..][0..hd];

                    var max_score: f32 = -std.math.inf(f32);
                    for (0..seq_len) |ki| {
                        const k_vec = c.k.data[(b * seq_len + ki) * model_dim + head_off ..][0..hd];
                        const score = math.dotProduct(q_vec, k_vec) * c.scale;
                        scores[ki] = score;
                        if (score > max_score) max_score = score;
                    }

                    var sum_exp: f32 = 0.0;
                    for (scores) |*s| {
                        s.* = @exp(s.* - max_score);
                        sum_exp += s.*;
                    }
                    const inv_sum = 1.0 / (sum_exp + 1e-12);

                    const out_vec = c.out.data[(b * seq_len + qi) * model_dim + head_off ..][0..hd];
                    @memset(out_vec, 0.0);
                    for (0..seq_len) |ki| {
                        const w = scores[ki] * inv_sum;
                        if (w == 0.0) continue;
                        const v_vec = c.v.data[(b * seq_len + ki) * model_dim + head_off ..][0..hd];
                        for (0..hd) |d| {
                            out_vec[d] += w * v_vec[d];
                        }
                    }
                }
            }
        }
    }.worker);

    return out;
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

    for (0..win_h) |wh| {
        for (0..win_w) |ww| {
            const win_idx = wh * win_w + ww;
            for (0..window) |ih| {
                const src_h = wh * window + ih;
                if (src_h >= h) continue;
                for (0..window) |iw| {
                    const src_w = ww * window + iw;
                    if (src_w >= w) continue;

                    const src = x.data[((src_h * w) + src_w) * c ..][0..c];
                    const dst = out.data[(win_idx * window * window + ih * window + iw) * c ..][0..c];
                    @memcpy(dst, src);
                }
            }
        }
    }

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

    for (0..h) |oh| {
        const wh = oh / window;
        const ih = oh % window;
        for (0..w) |ow| {
            const ww = ow / window;
            const iw = ow % window;
            const win_idx = wh * win_w + ww;

            const src = windows.data[(win_idx * window * window + ih * window + iw) * c ..][0..c];
            const dst = out.data[((oh * w) + ow) * c ..][0..c];
            @memcpy(dst, src);
        }
    }

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

        applyRope(&q, rope, cfg.num_heads);
        applyRope(&k, rope, cfg.num_heads);

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

    applyRope(&x, table, num_heads);

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
