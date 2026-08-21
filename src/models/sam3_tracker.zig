//! SAM 3's tracker branch (`tracker_model.*`), ported to run off the released
//! `facebook/sam3` checkpoint: the geometric prompt encoder and the two-way
//! mask decoder that turn a point or box into a mask.
//!
//! This is the SAM 2-lineage head. It consumes the shared vision backbone
//! (`vision_encoder.zig`) run through the `tracker_neck` FPN, and is the same
//! path the browser playground exports as `onnx-community/sam3-tracker-ONNX`.
//! The concept (text) path goes through `detector_model.*` instead and is not
//! ported yet.
//!
//! Reference: `Sam3TrackerPromptEncoder` / `Sam3TrackerMaskDecoder` /
//! `Sam3TrackerModel` in HF Transformers.

const std = @import("std");
const Tensor = @import("../tensor/tensor.zig").Tensor;
const math = @import("../tensor/math.zig");
const ops = @import("../tensor/ops.zig");
const weight_loader = @import("../weights/weight_loader.zig");
const WeightStore = weight_loader.WeightStore;
const InitType = weight_loader.InitType;

pub const TrackerConfig = struct {
    hidden_size: usize = 256,
    num_heads: usize = 8,
    /// Cross-attention projects down to hidden_size/2; self-attention does not.
    attention_downsample_rate: usize = 2,
    /// Two-way transformer blocks.
    num_layers: usize = 2,
    mlp_dim: usize = 2048,
    num_multimask_outputs: usize = 3,
    iou_head_depth: usize = 3,

    image_size: usize = 1008,
    patch_size: usize = 14,

    /// `nn.LayerNorm` default, used by every norm inside the two-way transformer.
    layer_norm_eps: f32 = 1e-5,
    /// `Sam3TrackerLayerNorm` default, used by the mask upscaler.
    upscale_layer_norm_eps: f32 = 1e-6,

    dynamic_multimask_stability_delta: f32 = 0.05,
    dynamic_multimask_stability_thresh: f32 = 0.98,

    prefix: []const u8 = "tracker_model",

    pub inline fn numMaskTokens(self: TrackerConfig) usize {
        return self.num_multimask_outputs + 1;
    }

    pub inline fn grid(self: TrackerConfig) usize {
        return self.image_size / self.patch_size;
    }
};

/// A prompt point in pixels of the *model input* (i.e. already scaled to
/// `image_size`), with SAM's label convention: 1 foreground, 0 background,
/// -1 "not a point" padding.
pub const PromptPoint = struct {
    x: f32,
    y: f32,
    label: i32,
};

pub const SegmentResult = struct {
    allocator: std.mem.Allocator,
    /// Mask logits at a quarter of the model input, `[num_masks, low_res, low_res]`.
    masks: Tensor,
    /// Predicted IoU per mask, already through the head's sigmoid.
    iou_scores: []f32,
    /// Presence logit for the prompted object.
    object_score: f32,
    /// Index into `masks` with the highest predicted IoU.
    best_index: usize,

    pub fn deinit(self: *SegmentResult) void {
        self.masks.deinit();
        self.allocator.free(self.iou_scores);
    }

    /// One mask as a `[1, 1, H, W]` logit tensor, ready for `overlayMask`.
    pub fn maskAt(self: SegmentResult, index: usize, allocator: std.mem.Allocator) !Tensor {
        const h = self.masks.shape[1];
        const w = self.masks.shape[2];
        const shape = [_]usize{ 1, 1, h, w };
        const out = try Tensor.init(allocator, &shape);
        @memcpy(out.data, self.masks.data[index * h * w ..][0 .. h * w]);
        return out;
    }
};

fn param(
    weights: *WeightStore,
    buf: []u8,
    comptime fmt: []const u8,
    args: anytype,
    shape: []const usize,
    kind: InitType,
    seed: u64,
) !Tensor {
    const name = try std.fmt.bufPrint(buf, fmt, args);
    return weights.getOrInit(name, shape, kind, seed);
}

/// Random Fourier positional encoding of coordinates in [0, 1]:
/// `cat(sin, cos)(2*pi * (2c - 1) @ positional_embedding)`.
/// `coords` is `[n, 2]` as (x, y); the result is `[n, 2 * num_features]`.
fn encodeCoordinates(
    allocator: std.mem.Allocator,
    coords: []const [2]f32,
    positional_embedding: Tensor,
) !Tensor {
    const num_features = positional_embedding.shape[1];
    const shape = [_]usize{ coords.len, 2 * num_features };
    var out = try Tensor.init(allocator, &shape);
    errdefer out.deinit();

    for (coords, 0..) |c, i| {
        const cx = 2.0 * c[0] - 1.0;
        const cy = 2.0 * c[1] - 1.0;
        for (0..num_features) |f| {
            const dot = cx * positional_embedding.at2(0, f) + cy * positional_embedding.at2(1, f);
            const angle = 2.0 * std.math.pi * dot;
            out.set2(i, f, @sin(angle));
            out.set2(i, f + num_features, @cos(angle));
        }
    }
    return out;
}

/// The image-wide positional encoding the decoder adds to every image token:
/// pixel centres of the token grid, encoded by `shared_image_embedding`.
/// Returns `[1, hidden_size, grid, grid]`.
pub fn imagePositionalEmbeddings(
    allocator: std.mem.Allocator,
    cfg: TrackerConfig,
    weights: *WeightStore,
) !Tensor {
    var name_buf: [128]u8 = undefined;
    const shared = try param(
        weights,
        &name_buf,
        "{s}.shared_image_embedding.positional_embedding",
        .{cfg.prefix},
        &[_]usize{ 2, cfg.hidden_size / 2 },
        .xavier,
        10,
    );

    const grid = cfg.grid();
    var coords = try allocator.alloc([2]f32, grid * grid);
    defer allocator.free(coords);

    const scale = 1.0 / @as(f32, @floatFromInt(grid));
    for (0..grid) |y| {
        for (0..grid) |x| {
            coords[y * grid + x] = .{
                (@as(f32, @floatFromInt(x)) + 0.5) * scale,
                (@as(f32, @floatFromInt(y)) + 0.5) * scale,
            };
        }
    }

    var encoded = try encodeCoordinates(allocator, coords, shared);
    defer encoded.deinit();

    // [grid*grid, hidden] -> [1, hidden, grid, grid]
    const shape = [_]usize{ 1, cfg.hidden_size, grid, grid };
    var out = try Tensor.init(allocator, &shape);
    errdefer out.deinit();

    for (0..grid * grid) |p| {
        for (0..cfg.hidden_size) |c| {
            out.data[c * grid * grid + p] = encoded.at2(p, c);
        }
    }
    return out;
}

/// Sparse prompt tokens for a set of points. Mirrors `_embed_points`: shift to
/// pixel centres, encode, swap in `not_a_point_embed` for label -1, and add the
/// learned per-label embedding for labels >= 0.
pub fn embedPoints(
    allocator: std.mem.Allocator,
    cfg: TrackerConfig,
    points: []const PromptPoint,
    weights: *WeightStore,
) !Tensor {
    var name_buf: [128]u8 = undefined;
    const d = cfg.hidden_size;

    const shared = try param(
        weights,
        &name_buf,
        "{s}.prompt_encoder.shared_embedding.positional_embedding",
        .{cfg.prefix},
        &[_]usize{ 2, d / 2 },
        .xavier,
        11,
    );
    const point_embed = try param(
        weights,
        &name_buf,
        "{s}.prompt_encoder.point_embed.weight",
        .{cfg.prefix},
        &[_]usize{ 4, d },
        .xavier,
        12,
    );
    const not_a_point = try param(
        weights,
        &name_buf,
        "{s}.prompt_encoder.not_a_point_embed.weight",
        .{cfg.prefix},
        &[_]usize{ 1, d },
        .xavier,
        13,
    );

    var coords = try allocator.alloc([2]f32, points.len);
    defer allocator.free(coords);

    const size: f32 = @floatFromInt(cfg.image_size);
    for (points, 0..) |p, i| {
        // +0.5 shifts to the pixel centre, then normalise by the input size.
        coords[i] = .{ (p.x + 0.5) / size, (p.y + 0.5) / size };
    }

    var encoded = try encodeCoordinates(allocator, coords, shared);
    defer encoded.deinit();

    const shape = [_]usize{ 1, points.len, d };
    var out = try Tensor.init(allocator, &shape);
    errdefer out.deinit();

    for (points, 0..) |p, i| {
        const dst = out.data[i * d ..][0..d];
        if (p.label == -1) {
            @memcpy(dst, not_a_point.data[0..d]);
        } else {
            for (0..d) |c| dst[c] = encoded.at2(i, c);
            if (p.label >= 0) {
                const row = @as(usize, @intCast(p.label));
                for (0..d) |c| dst[c] += point_embed.at2(row, c);
            }
        }
    }
    return out;
}

/// Dense prompt when no mask is supplied: `no_mask_embed` broadcast over the grid.
fn embedNoMask(
    allocator: std.mem.Allocator,
    cfg: TrackerConfig,
    weights: *WeightStore,
) !Tensor {
    var name_buf: [128]u8 = undefined;
    const no_mask = try param(
        weights,
        &name_buf,
        "{s}.prompt_encoder.no_mask_embed.weight",
        .{cfg.prefix},
        &[_]usize{ 1, cfg.hidden_size },
        .xavier,
        14,
    );

    const grid = cfg.grid();
    const shape = [_]usize{ 1, cfg.hidden_size, grid, grid };
    var out = try Tensor.init(allocator, &shape);
    errdefer out.deinit();

    for (0..cfg.hidden_size) |c| {
        const v = no_mask.data[c];
        @memset(out.data[c * grid * grid ..][0 .. grid * grid], v);
    }
    return out;
}

/// One attention module of the decoder: project q/k/v to `internal_dim`,
/// attend, project back to `hidden_size`.
fn attentionBlock(
    allocator: std.mem.Allocator,
    cfg: TrackerConfig,
    q_in: Tensor,
    k_in: Tensor,
    v_in: Tensor,
    weights: *WeightStore,
    name_buf: []u8,
    comptime fmt: []const u8,
    args: anytype,
    downsample_rate: usize,
    seed: u64,
) !Tensor {
    const d = cfg.hidden_size;
    const internal = d / downsample_rate;

    var sub_buf: [160]u8 = undefined;
    const base = try std.fmt.bufPrint(&sub_buf, fmt, args);

    const q_w = try param(weights, name_buf, "{s}.q_proj.weight", .{base}, &[_]usize{ internal, d }, .xavier, seed);
    const q_b = try param(weights, name_buf, "{s}.q_proj.bias", .{base}, &[_]usize{internal}, .zeros, seed + 1);
    const k_w = try param(weights, name_buf, "{s}.k_proj.weight", .{base}, &[_]usize{ internal, d }, .xavier, seed + 2);
    const k_b = try param(weights, name_buf, "{s}.k_proj.bias", .{base}, &[_]usize{internal}, .zeros, seed + 3);
    const v_w = try param(weights, name_buf, "{s}.v_proj.weight", .{base}, &[_]usize{ internal, d }, .xavier, seed + 4);
    const v_b = try param(weights, name_buf, "{s}.v_proj.bias", .{base}, &[_]usize{internal}, .zeros, seed + 5);
    const o_w = try param(weights, name_buf, "{s}.o_proj.weight", .{base}, &[_]usize{ d, internal }, .xavier, seed + 6);
    const o_b = try param(weights, name_buf, "{s}.o_proj.bias", .{base}, &[_]usize{d}, .zeros, seed + 7);

    var q = try math.linear(allocator, q_in, q_w, q_b);
    defer q.deinit();
    var k = try math.linear(allocator, k_in, k_w, k_b);
    defer k.deinit();
    var v = try math.linear(allocator, v_in, v_w, v_b);
    defer v.deinit();

    var attn = try ops.multiHeadAttention(allocator, q, k, v, cfg.num_heads, null);
    defer attn.deinit();

    return math.linear(allocator, attn, o_w, o_b);
}

/// `Sam3TrackerFeedForward`: proj_in -> ReLU -> (layers -> ReLU)* -> proj_out.
fn feedForward(
    allocator: std.mem.Allocator,
    x: Tensor,
    hidden_dim: usize,
    output_dim: usize,
    num_inner_layers: usize,
    sigmoid_output: bool,
    weights: *WeightStore,
    name_buf: []u8,
    comptime fmt: []const u8,
    args: anytype,
    seed: u64,
) !Tensor {
    var sub_buf: [160]u8 = undefined;
    const base = try std.fmt.bufPrint(&sub_buf, fmt, args);
    const in_dim = x.shape[x.shape.len - 1];

    const in_w = try param(weights, name_buf, "{s}.proj_in.weight", .{base}, &[_]usize{ hidden_dim, in_dim }, .xavier, seed);
    const in_b = try param(weights, name_buf, "{s}.proj_in.bias", .{base}, &[_]usize{hidden_dim}, .zeros, seed + 1);

    var hidden = try math.linear(allocator, x, in_w, in_b);
    errdefer hidden.deinit();
    math.applyActivation(&hidden, .relu);

    for (0..num_inner_layers) |i| {
        const w = try param(weights, name_buf, "{s}.layers.{d}.weight", .{ base, i }, &[_]usize{ hidden_dim, hidden_dim }, .xavier, seed + 2 + i);
        const b = try param(weights, name_buf, "{s}.layers.{d}.bias", .{ base, i }, &[_]usize{hidden_dim}, .zeros, seed + 20 + i);

        var next = try math.linear(allocator, hidden, w, b);
        math.applyActivation(&next, .relu);
        hidden.deinit();
        hidden = next;
    }

    const out_w = try param(weights, name_buf, "{s}.proj_out.weight", .{base}, &[_]usize{ output_dim, hidden_dim }, .xavier, seed + 40);
    const out_b = try param(weights, name_buf, "{s}.proj_out.bias", .{base}, &[_]usize{output_dim}, .zeros, seed + 41);

    var out = try math.linear(allocator, hidden, out_w, out_b);
    hidden.deinit();
    if (sigmoid_output) math.applyActivation(&out, .sigmoid);
    return out;
}

fn addTensors(allocator: std.mem.Allocator, a: Tensor, b: Tensor) !Tensor {
    var out = try a.clone(allocator);
    out.addInPlace(b);
    return out;
}

/// Layer norm over the channel dimension of `[1, C, H, W]` (`data_format="channels_first"`).
fn layerNormChannelsFirst(
    allocator: std.mem.Allocator,
    x: Tensor,
    gamma: Tensor,
    beta: Tensor,
    eps: f32,
) !Tensor {
    const channels = x.shape[1];
    const spatial = x.shape[2] * x.shape[3];

    var out = try Tensor.init(allocator, x.shape);
    errdefer out.deinit();

    for (0..spatial) |p| {
        var mean: f32 = 0.0;
        for (0..channels) |c| mean += x.data[c * spatial + p];
        mean /= @floatFromInt(channels);

        var variance: f32 = 0.0;
        for (0..channels) |c| {
            const d = x.data[c * spatial + p] - mean;
            variance += d * d;
        }
        variance /= @floatFromInt(channels);
        const inv_std = 1.0 / @sqrt(variance + eps);

        for (0..channels) |c| {
            out.data[c * spatial + p] = (x.data[c * spatial + p] - mean) * inv_std * gamma.data[c] + beta.data[c];
        }
    }
    return out;
}

/// The two-way transformer plus the mask/IoU/object heads.
///
/// `image_embedding` is FPN level 2 (`[1, 256, grid, grid]`), `feat_s0` and
/// `feat_s1` are levels 0 and 1 after `conv_s0` / `conv_s1`.
pub fn maskDecoder(
    allocator: std.mem.Allocator,
    cfg: TrackerConfig,
    image_embedding: Tensor,
    image_positional: Tensor,
    sparse_prompt: Tensor,
    feat_s0: Tensor,
    feat_s1: Tensor,
    multimask_output: bool,
    weights: *WeightStore,
) !SegmentResult {
    var name_buf: [256]u8 = undefined;
    const d = cfg.hidden_size;
    const grid = image_embedding.shape[2];
    const num_mask_tokens = cfg.numMaskTokens();
    const prefix = cfg.prefix;

    // --- Output tokens: [obj_score, iou, mask_0..3] then the prompt tokens ---
    const obj_token = try param(weights, &name_buf, "{s}.mask_decoder.obj_score_token.weight", .{prefix}, &[_]usize{ 1, d }, .xavier, 100);
    const iou_token = try param(weights, &name_buf, "{s}.mask_decoder.iou_token.weight", .{prefix}, &[_]usize{ 1, d }, .xavier, 101);
    const mask_tokens = try param(weights, &name_buf, "{s}.mask_decoder.mask_tokens.weight", .{prefix}, &[_]usize{ num_mask_tokens, d }, .xavier, 102);

    const num_prompt = sparse_prompt.shape[1];
    const num_tokens = 2 + num_mask_tokens + num_prompt;

    const token_shape = [_]usize{ 1, num_tokens, d };
    var tokens = try Tensor.init(allocator, &token_shape);
    defer tokens.deinit();

    @memcpy(tokens.data[0..d], obj_token.data[0..d]);
    @memcpy(tokens.data[d .. 2 * d], iou_token.data[0..d]);
    @memcpy(tokens.data[2 * d ..][0 .. num_mask_tokens * d], mask_tokens.data[0 .. num_mask_tokens * d]);
    @memcpy(tokens.data[(2 + num_mask_tokens) * d ..][0 .. num_prompt * d], sparse_prompt.data[0 .. num_prompt * d]);

    // --- Image tokens: level 2 features plus the dense prompt, flattened ---
    var keys = try flattenSpatial(allocator, image_embedding);
    defer keys.deinit();
    var key_pe = try flattenSpatial(allocator, image_positional);
    defer key_pe.deinit();

    var queries = try tokens.clone(allocator);
    defer queries.deinit();

    for (0..cfg.num_layers) |layer| {
        const skip_first_pe = layer == 0;

        // (1) self-attention over the tokens
        {
            var attn_out: Tensor = undefined;
            if (skip_first_pe) {
                attn_out = try attentionBlock(allocator, cfg, queries, queries, queries, weights, &name_buf, "{s}.mask_decoder.transformer.layers.{d}.self_attn", .{ prefix, layer }, 1, 200 + layer * 10);
                queries.deinit();
                queries = attn_out;
            } else {
                var q = try addTensors(allocator, queries, tokens);
                defer q.deinit();
                attn_out = try attentionBlock(allocator, cfg, q, q, queries, weights, &name_buf, "{s}.mask_decoder.transformer.layers.{d}.self_attn", .{ prefix, layer }, 1, 200 + layer * 10);
                defer attn_out.deinit();
                queries.addInPlace(attn_out);
            }
        }
        try normInPlace(allocator, &queries, weights, &name_buf, "{s}.mask_decoder.transformer.layers.{d}.layer_norm1", .{ prefix, layer }, d, cfg.layer_norm_eps, 300 + layer * 10);

        // (2) tokens attend to the image
        {
            var q = try addTensors(allocator, queries, tokens);
            defer q.deinit();
            var k = try addTensors(allocator, keys, key_pe);
            defer k.deinit();

            var attn_out = try attentionBlock(allocator, cfg, q, k, keys, weights, &name_buf, "{s}.mask_decoder.transformer.layers.{d}.cross_attn_token_to_image", .{ prefix, layer }, cfg.attention_downsample_rate, 400 + layer * 10);
            defer attn_out.deinit();
            queries.addInPlace(attn_out);
        }
        try normInPlace(allocator, &queries, weights, &name_buf, "{s}.mask_decoder.transformer.layers.{d}.layer_norm2", .{ prefix, layer }, d, cfg.layer_norm_eps, 500 + layer * 10);

        // (3) token MLP
        {
            var mlp_out = try feedForward(allocator, queries, cfg.mlp_dim, d, 0, false, weights, &name_buf, "{s}.mask_decoder.transformer.layers.{d}.mlp", .{ prefix, layer }, 600 + layer * 10);
            defer mlp_out.deinit();
            queries.addInPlace(mlp_out);
        }
        try normInPlace(allocator, &queries, weights, &name_buf, "{s}.mask_decoder.transformer.layers.{d}.layer_norm3", .{ prefix, layer }, d, cfg.layer_norm_eps, 700 + layer * 10);

        // (4) image attends back to the tokens
        {
            var q = try addTensors(allocator, queries, tokens);
            defer q.deinit();
            var k = try addTensors(allocator, keys, key_pe);
            defer k.deinit();

            var attn_out = try attentionBlock(allocator, cfg, k, q, queries, weights, &name_buf, "{s}.mask_decoder.transformer.layers.{d}.cross_attn_image_to_token", .{ prefix, layer }, cfg.attention_downsample_rate, 800 + layer * 10);
            defer attn_out.deinit();
            keys.addInPlace(attn_out);
        }
        try normInPlace(allocator, &keys, weights, &name_buf, "{s}.mask_decoder.transformer.layers.{d}.layer_norm4", .{ prefix, layer }, d, cfg.layer_norm_eps, 900 + layer * 10);
    }

    // --- Final token -> image attention ---
    {
        var q = try addTensors(allocator, queries, tokens);
        defer q.deinit();
        var k = try addTensors(allocator, keys, key_pe);
        defer k.deinit();

        var attn_out = try attentionBlock(allocator, cfg, q, k, keys, weights, &name_buf, "{s}.mask_decoder.transformer.final_attn_token_to_image", .{prefix}, cfg.attention_downsample_rate, 1000);
        defer attn_out.deinit();
        queries.addInPlace(attn_out);
    }
    try normInPlace(allocator, &queries, weights, &name_buf, "{s}.mask_decoder.transformer.layer_norm_final_attn", .{prefix}, d, cfg.layer_norm_eps, 1100);

    // --- Upscale the image tokens back to a quarter of the input resolution ---
    var image_out = try unflattenSpatial(allocator, keys, grid);
    defer image_out.deinit();

    const up1_w = try param(weights, &name_buf, "{s}.mask_decoder.upscale_conv1.weight", .{prefix}, &[_]usize{ d, d / 4, 2, 2 }, .kaiming, 1200);
    const up1_b = try param(weights, &name_buf, "{s}.mask_decoder.upscale_conv1.bias", .{prefix}, &[_]usize{d / 4}, .zeros, 1201);

    var upscaled = try ops.convTranspose2d(allocator, image_out, up1_w, up1_b, 2, 0);
    defer upscaled.deinit();
    upscaled.addInPlace(feat_s1);

    const up_ln_g = try param(weights, &name_buf, "{s}.mask_decoder.upscale_layer_norm.weight", .{prefix}, &[_]usize{d / 4}, .ones, 1202);
    const up_ln_b = try param(weights, &name_buf, "{s}.mask_decoder.upscale_layer_norm.bias", .{prefix}, &[_]usize{d / 4}, .zeros, 1203);

    var normed = try layerNormChannelsFirst(allocator, upscaled, up_ln_g, up_ln_b, cfg.upscale_layer_norm_eps);
    defer normed.deinit();
    math.applyActivation(&normed, .gelu_exact);

    const up2_w = try param(weights, &name_buf, "{s}.mask_decoder.upscale_conv2.weight", .{prefix}, &[_]usize{ d / 4, d / 8, 2, 2 }, .kaiming, 1204);
    const up2_b = try param(weights, &name_buf, "{s}.mask_decoder.upscale_conv2.bias", .{prefix}, &[_]usize{d / 8}, .zeros, 1205);

    var upscaled2 = try ops.convTranspose2d(allocator, normed, up2_w, up2_b, 2, 0);
    defer upscaled2.deinit();
    upscaled2.addInPlace(feat_s0);
    math.applyActivation(&upscaled2, .gelu_exact);

    // --- Hypernetwork MLPs produce one 32-dim kernel per mask token ---
    const mask_dim = d / 8;
    const hyper_shape = [_]usize{ num_mask_tokens, mask_dim };
    var hyper = try Tensor.init(allocator, &hyper_shape);
    defer hyper.deinit();

    for (0..num_mask_tokens) |i| {
        const token_shape_1 = [_]usize{ 1, 1, d };
        var token = try Tensor.init(allocator, &token_shape_1);
        defer token.deinit();
        @memcpy(token.data, queries.data[(2 + i) * d ..][0..d]);

        var kernel = try feedForward(allocator, token, d, mask_dim, 1, false, weights, &name_buf, "{s}.mask_decoder.output_hypernetworks_mlps.{d}", .{ prefix, i }, 1300 + i * 10);
        defer kernel.deinit();
        @memcpy(hyper.data[i * mask_dim ..][0..mask_dim], kernel.data[0..mask_dim]);
    }

    // masks = hyper @ upscaled, reshaped back to a spatial map
    const low_res = upscaled2.shape[2];
    const flat_shape = [_]usize{ mask_dim, low_res * low_res };
    var flat = upscaled2;
    const saved_shape = try allocator.dupe(usize, flat.shape);
    defer allocator.free(saved_shape);
    try flat.reshape(&flat_shape);

    var masks_2d = try math.matmul2D(allocator, hyper, flat);
    errdefer masks_2d.deinit();
    try flat.reshape(saved_shape);

    // --- Quality heads ---
    const iou_token_shape = [_]usize{ 1, 1, d };
    var iou_in = try Tensor.init(allocator, &iou_token_shape);
    defer iou_in.deinit();
    @memcpy(iou_in.data, queries.data[d .. 2 * d]);

    var iou_pred = try feedForward(allocator, iou_in, d, num_mask_tokens, 1, true, weights, &name_buf, "{s}.mask_decoder.iou_prediction_head", .{prefix}, 1400);
    defer iou_pred.deinit();

    var obj_in = try Tensor.init(allocator, &iou_token_shape);
    defer obj_in.deinit();
    @memcpy(obj_in.data, queries.data[0..d]);

    var obj_pred = try feedForward(allocator, obj_in, d, 1, 1, false, weights, &name_buf, "{s}.mask_decoder.pred_obj_score_head", .{prefix}, 1500);
    defer obj_pred.deinit();

    // --- Select which masks to return ---
    const first = if (multimask_output) @as(usize, 1) else 0;
    const count = if (multimask_output) num_mask_tokens - 1 else 1;

    const out_shape = [_]usize{ count, low_res, low_res };
    var masks = try Tensor.init(allocator, &out_shape);
    errdefer masks.deinit();
    @memcpy(masks.data, masks_2d.data[first * low_res * low_res ..][0 .. count * low_res * low_res]);
    masks_2d.deinit();

    var scores = try allocator.alloc(f32, count);
    errdefer allocator.free(scores);
    var best: usize = 0;
    for (0..count) |i| {
        scores[i] = iou_pred.data[first + i];
        if (scores[i] > scores[best]) best = i;
    }

    return SegmentResult{
        .allocator = allocator,
        .masks = masks,
        .iou_scores = scores,
        .object_score = obj_pred.data[0],
        .best_index = best,
    };
}

/// `[1, C, H, W]` -> `[1, H*W, C]`.
fn flattenSpatial(allocator: std.mem.Allocator, x: Tensor) !Tensor {
    const channels = x.shape[1];
    const spatial = x.shape[2] * x.shape[3];

    const shape = [_]usize{ 1, spatial, channels };
    var out = try Tensor.init(allocator, &shape);
    errdefer out.deinit();

    for (0..spatial) |p| {
        for (0..channels) |c| {
            out.data[p * channels + c] = x.data[c * spatial + p];
        }
    }
    return out;
}

/// `[1, H*W, C]` -> `[1, C, H, W]`.
fn unflattenSpatial(allocator: std.mem.Allocator, x: Tensor, grid: usize) !Tensor {
    const channels = x.shape[2];
    const spatial = grid * grid;

    const shape = [_]usize{ 1, channels, grid, grid };
    var out = try Tensor.init(allocator, &shape);
    errdefer out.deinit();

    for (0..spatial) |p| {
        for (0..channels) |c| {
            out.data[c * spatial + p] = x.data[p * channels + c];
        }
    }
    return out;
}

fn normInPlace(
    allocator: std.mem.Allocator,
    x: *Tensor,
    weights: *WeightStore,
    name_buf: []u8,
    comptime fmt: []const u8,
    args: anytype,
    dim: usize,
    eps: f32,
    seed: u64,
) !void {
    var sub_buf: [160]u8 = undefined;
    const base = try std.fmt.bufPrint(&sub_buf, fmt, args);

    const g = try param(weights, name_buf, "{s}.weight", .{base}, &[_]usize{dim}, .ones, seed);
    const b = try param(weights, name_buf, "{s}.bias", .{base}, &[_]usize{dim}, .zeros, seed + 1);

    const normed = try math.layerNorm(allocator, x.*, g, b, eps);
    x.deinit();
    x.* = normed;
}

/// Projects FPN levels 0 and 1 into the channel counts the upscaler adds them to.
pub fn projectHighResFeatures(
    allocator: std.mem.Allocator,
    cfg: TrackerConfig,
    level0: Tensor,
    level1: Tensor,
    weights: *WeightStore,
) !struct { s0: Tensor, s1: Tensor } {
    var name_buf: [128]u8 = undefined;
    const d = cfg.hidden_size;

    const s0_w = try param(weights, &name_buf, "{s}.mask_decoder.conv_s0.weight", .{cfg.prefix}, &[_]usize{ d / 8, d, 1, 1 }, .kaiming, 1600);
    const s0_b = try param(weights, &name_buf, "{s}.mask_decoder.conv_s0.bias", .{cfg.prefix}, &[_]usize{d / 8}, .zeros, 1601);
    const s1_w = try param(weights, &name_buf, "{s}.mask_decoder.conv_s1.weight", .{cfg.prefix}, &[_]usize{ d / 4, d, 1, 1 }, .kaiming, 1602);
    const s1_b = try param(weights, &name_buf, "{s}.mask_decoder.conv_s1.bias", .{cfg.prefix}, &[_]usize{d / 4}, .zeros, 1603);

    var s0 = try ops.conv2d(allocator, level0, s0_w, s0_b, 1, 0);
    errdefer s0.deinit();
    const s1 = try ops.conv2d(allocator, level1, s1_w, s1_b, 1, 0);

    return .{ .s0 = s0, .s1 = s1 };
}

/// Full point-prompted segmentation given the tracker FPN levels 0/1/2.
pub fn segmentWithPoints(
    allocator: std.mem.Allocator,
    cfg: TrackerConfig,
    fpn_levels: []const Tensor,
    points: []const PromptPoint,
    multimask_output: bool,
    weights: *WeightStore,
) !SegmentResult {
    std.debug.assert(fpn_levels.len >= 3);

    var high_res = try projectHighResFeatures(allocator, cfg, fpn_levels[0], fpn_levels[1], weights);
    defer high_res.s0.deinit();
    defer high_res.s1.deinit();

    var sparse = try embedPoints(allocator, cfg, points, weights);
    defer sparse.deinit();

    var dense = try embedNoMask(allocator, cfg, weights);
    defer dense.deinit();

    var image_embedding = try fpn_levels[2].clone(allocator);
    defer image_embedding.deinit();

    // A frame with no memory bank behind it gets the learned "no memory"
    // embedding added to the coarsest level, exactly as the reference does
    // before running the SAM heads on the first frame.
    {
        var name_buf: [128]u8 = undefined;
        const no_memory = try param(
            weights,
            &name_buf,
            "{s}.no_memory_embedding",
            .{cfg.prefix},
            &[_]usize{ 1, 1, cfg.hidden_size },
            .zeros,
            15,
        );

        const plane = image_embedding.shape[2] * image_embedding.shape[3];
        for (0..cfg.hidden_size) |c| {
            const v = no_memory.data[c];
            for (image_embedding.data[c * plane ..][0..plane]) |*x| x.* += v;
        }
    }

    image_embedding.addInPlace(dense);

    var positional = try imagePositionalEmbeddings(allocator, cfg, weights);
    defer positional.deinit();

    return maskDecoder(
        allocator,
        cfg,
        image_embedding,
        positional,
        sparse,
        high_res.s0,
        high_res.s1,
        multimask_output,
        weights,
    );
}

test "coordinate encoding is sine/cosine of a linear projection" {
    const allocator = std.testing.allocator;

    const pe_shape = [_]usize{ 2, 2 };
    var pe = try Tensor.init(allocator, &pe_shape);
    defer pe.deinit();
    pe.data[0] = 1.0; pe.data[1] = 0.0; // x row
    pe.data[2] = 0.0; pe.data[3] = 1.0; // y row

    const coords = [_][2]f32{.{ 0.5, 0.5 }};
    var encoded = try encodeCoordinates(allocator, &coords, pe);
    defer encoded.deinit();

    // (0.5, 0.5) maps to (0, 0) after 2c-1, so sin -> 0 and cos -> 1.
    try std.testing.expectEqual(@as(usize, 4), encoded.shape[1]);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), encoded.at2(0, 0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), encoded.at2(0, 1), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), encoded.at2(0, 2), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), encoded.at2(0, 3), 1e-6);
}

test "spatial flatten round-trips" {
    const allocator = std.testing.allocator;

    const shape = [_]usize{ 1, 3, 2, 2 };
    var x = try Tensor.init(allocator, &shape);
    defer x.deinit();
    for (x.data, 0..) |*v, i| v.* = @floatFromInt(i);

    var flat = try flattenSpatial(allocator, x);
    defer flat.deinit();
    try std.testing.expectEqual(@as(usize, 4), flat.shape[1]);
    try std.testing.expectEqual(@as(usize, 3), flat.shape[2]);

    var back = try unflattenSpatial(allocator, flat, 2);
    defer back.deinit();
    for (x.data, back.data) |e, a| try std.testing.expectEqual(e, a);
}

test "tracker head runs end to end at a small resolution" {
    const allocator = std.testing.allocator;

    const cfg = TrackerConfig{
        .hidden_size = 32,
        .num_heads = 2,
        .num_layers = 2,
        .mlp_dim = 64,
        .image_size = 14 * 8,
        .patch_size = 14,
        .prefix = "test_tracker",
    };

    var weights = WeightStore.init(allocator);
    defer weights.deinit();

    const grid = cfg.grid(); // 8
    var levels: [3]Tensor = undefined;
    levels[0] = try Tensor.initRandom(allocator, &[_]usize{ 1, cfg.hidden_size, grid * 4, grid * 4 }, 1, -1.0, 1.0);
    defer levels[0].deinit();
    levels[1] = try Tensor.initRandom(allocator, &[_]usize{ 1, cfg.hidden_size, grid * 2, grid * 2 }, 2, -1.0, 1.0);
    defer levels[1].deinit();
    levels[2] = try Tensor.initRandom(allocator, &[_]usize{ 1, cfg.hidden_size, grid, grid }, 3, -1.0, 1.0);
    defer levels[2].deinit();

    const points = [_]PromptPoint{
        .{ .x = 56.0, .y = 56.0, .label = 1 },
        .{ .x = 0.0, .y = 0.0, .label = -1 },
    };

    var result = try segmentWithPoints(allocator, cfg, &levels, &points, true, &weights);
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 3), result.masks.shape[0]);
    try std.testing.expectEqual(grid * 4, result.masks.shape[1]);
    try std.testing.expectEqual(@as(usize, 3), result.iou_scores.len);
    for (result.iou_scores) |s| {
        try std.testing.expect(s >= 0.0 and s <= 1.0);
    }
    for (result.masks.data) |v| try std.testing.expect(std.math.isFinite(v));
}
