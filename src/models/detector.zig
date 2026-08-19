const std = @import("std");
const Tensor = @import("../tensor/tensor.zig").Tensor;
const math = @import("../tensor/math.zig");
const ops = @import("../tensor/ops.zig");
const SAM3Config = @import("config.zig").SAM3Config;
const WeightStore = @import("../weights/weight_loader.zig").WeightStore;
const Box = @import("prompt_encoder.zig").Box;

pub const Detection = struct {
    box: Box,
    score: f32,
    presence_score: f32,
    query_idx: usize,
};

pub const Detector = struct {
    config: SAM3Config,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: SAM3Config) Detector {
        return Detector{
            .config = config,
            .allocator = allocator,
        };
    }

    pub const Output = struct {
        presence_score: f32,
        is_present: bool,
        detections: []Detection,
        query_features: Tensor, // [1, N_queries, D]

        pub fn deinit(self: *Output, allocator: std.mem.Allocator) void {
            allocator.free(self.detections);
            self.query_features.deinit();
        }
    };

    pub fn forward(
        self: Detector,
        image_embeddings: Tensor,
        concept_tokens: ?Tensor,
        presence_token: ?Tensor,
        weights: *WeightStore,
    ) !Output {
        const d = self.config.encoder_embed_dim;
        const num_queries = self.config.num_object_queries;
        const grid_h = self.config.gridH();
        const grid_w = self.config.gridW();
        const num_patches = grid_h * grid_w;

        // 1. Presence Head evaluation
        var presence_score: f32 = 1.0;
        if (presence_token) |pt| {
            // pt is [1, 1, D]
            const p_fc1_shape = [_]usize{ self.config.presence_head_dim, d };
            const p_fc1 = try weights.getOrInit("detector.presence_head.fc1.weight", &p_fc1_shape, .kaiming, 601);
            const p_fc2_shape = [_]usize{ 1, self.config.presence_head_dim };
            const p_fc2 = try weights.getOrInit("detector.presence_head.fc2.weight", &p_fc2_shape, .xavier, 602);

            var h = try math.linear(self.allocator, pt, p_fc1, null);
            defer h.deinit();
            math.applyActivation(&h, .gelu);

            var logit = try math.linear(self.allocator, h, p_fc2, null);
            defer logit.deinit();

            presence_score = math.sigmoid(logit.data[0]);
        }

        const is_present = presence_score >= self.config.presence_threshold;

        // 2. Object Queries initialization: [1, num_queries, d]
        const query_shape = [_]usize{ 1, num_queries, d };
        const query_embed = try weights.getOrInit("detector.query_embed.weight", &query_shape, .xavier, 603);
        var queries = try query_embed.clone(self.allocator);
        errdefer queries.deinit();

        // 3. Flatten image embeddings: [1, d, grid_h, grid_w] -> [1, num_patches, d]
        const perm_dims = [_]usize{ 0, 2, 3, 1 };
        var img_flat = try image_embeddings.permute(self.allocator, &perm_dims);
        defer img_flat.deinit();
        const flat_shape = [_]usize{ 1, num_patches, d };
        try img_flat.reshape(&flat_shape);

        // 4. Cross-Attention: Queries attend to Image Embeddings
        const q_proj_shape = [_]usize{ d, d };
        const q_proj = try weights.getOrInit("detector.cross_attn.q_proj.weight", &q_proj_shape, .xavier, 604);
        const k_proj = try weights.getOrInit("detector.cross_attn.k_proj.weight", &q_proj_shape, .xavier, 605);
        const v_proj = try weights.getOrInit("detector.cross_attn.v_proj.weight", &q_proj_shape, .xavier, 606);

        var q = try math.linear(self.allocator, queries, q_proj, null);
        defer q.deinit();
        var k = try math.linear(self.allocator, img_flat, k_proj, null);
        defer k.deinit();
        var v = try math.linear(self.allocator, img_flat, v_proj, null);
        defer v.deinit();

        var attended = try ops.multiHeadAttention(self.allocator, q, k, v, self.config.mask_decoder_num_heads, null);
        defer attended.deinit();

        queries.addInPlace(attended);

        // If concept tokens provided, cross-attend queries with concept tokens
        if (concept_tokens) |ct| {
            var cq = try math.linear(self.allocator, queries, q_proj, null);
            defer cq.deinit();
            var ck = try math.linear(self.allocator, ct, k_proj, null);
            defer ck.deinit();
            var cv = try math.linear(self.allocator, ct, v_proj, null);
            defer cv.deinit();

            var concept_attn = try ops.multiHeadAttention(self.allocator, cq, ck, cv, self.config.mask_decoder_num_heads, null);
            defer concept_attn.deinit();

            queries.addInPlace(concept_attn);
        }

        // 5. Box Prediction Head: [1, num_queries, 4] -> (cx, cy, w, h)
        const box_head_shape = [_]usize{ 4, d };
        const box_head_w = try weights.getOrInit("detector.box_head.weight", &box_head_shape, .xavier, 607);
        var box_preds = try math.linear(self.allocator, queries, box_head_w, null);
        defer box_preds.deinit();

        // 6. Classification / Score Head: [1, num_queries, 1]
        const score_head_shape = [_]usize{ 1, d };
        const score_head_w = try weights.getOrInit("detector.score_head.weight", &score_head_shape, .xavier, 608);
        var score_preds = try math.linear(self.allocator, queries, score_head_w, null);
        defer score_preds.deinit();

        var detection_list: std.ArrayList(Detection) = .empty;
        errdefer detection_list.deinit(self.allocator);

        for (0..num_queries) |q_idx| {
            const raw_score = score_preds.at3(0, q_idx, 0);
            const score = math.sigmoid(raw_score) * presence_score;

            const cx = math.sigmoid(box_preds.at3(0, q_idx, 0));
            const cy = math.sigmoid(box_preds.at3(0, q_idx, 1));
            const w = math.sigmoid(box_preds.at3(0, q_idx, 2));
            const h = math.sigmoid(box_preds.at3(0, q_idx, 3));

            const x1 = std.math.clamp(cx - w / 2.0, 0.0, 1.0);
            const y1 = std.math.clamp(cy - h / 2.0, 0.0, 1.0);
            const x2 = std.math.clamp(cx + w / 2.0, 0.0, 1.0);
            const y2 = std.math.clamp(cy + h / 2.0, 0.0, 1.0);

            try detection_list.append(self.allocator, Detection{
                .box = .{ .x1 = x1, .y1 = y1, .x2 = x2, .y2 = y2 },
                .score = score,
                .presence_score = presence_score,
                .query_idx = q_idx,
            });
        }

        return Output{
            .presence_score = presence_score,
            .is_present = is_present,
            .detections = try detection_list.toOwnedSlice(self.allocator),
            .query_features = queries,
        };
    }
};
