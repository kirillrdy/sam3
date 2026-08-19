const std = @import("std");
const Tensor = @import("../tensor/tensor.zig").Tensor;
const math = @import("../tensor/math.zig");
const ops = @import("../tensor/ops.zig");
const SAM3Config = @import("config.zig").SAM3Config;
const WeightStore = @import("../weights/weight_loader.zig").WeightStore;

pub const Point = struct {
    x: f32, // normalized [0, 1]
    y: f32, // normalized [0, 1]
    label: i32, // 0: bg, 1: fg, 2: top-left box, 3: bottom-right box, -1: pad
};

pub const Box = struct {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
};

pub const Prompt = struct {
    points: ?[]const Point = null,
    box: ?Box = null,
    mask: ?Tensor = null,
    text: ?[]const u8 = null,
};

pub const PromptEncoder = struct {
    config: SAM3Config,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: SAM3Config) PromptEncoder {
        return PromptEncoder{
            .config = config,
            .allocator = allocator,
        };
    }

    pub const PromptEmbeddings = struct {
        sparse_embeddings: Tensor,   // [1, N_tokens, D]
        dense_embeddings: Tensor,    // [1, D, H_grid, W_grid]
        presence_token: ?Tensor,     // [1, 1, D]

        pub fn deinit(self: *PromptEmbeddings) void {
            self.sparse_embeddings.deinit();
            self.dense_embeddings.deinit();
            if (self.presence_token) |*pt| {
                pt.deinit();
            }
        }
    };

    pub fn tokenizeSimple(text: []const u8, vocab_size: usize, max_len: usize, allocator: std.mem.Allocator) ![]usize {
        var tokens: std.ArrayList(usize) = .empty;
        errdefer tokens.deinit(allocator);

        // Token 0 is reserved for Presence token / [CLS]
        try tokens.append(allocator, 0);

        var it = std.mem.tokenizeAny(u8, text, " ,.-_/!?:;'\t\n\r");
        while (it.next()) |word| {
            if (tokens.items.len >= max_len) break;
            var h: u32 = 2166136261;
            for (word) |c| {
                const lower = std.ascii.toLower(c);
                h ^= lower;
                h *%= 16777619;
            }
            const token_id = 1 + (h % @as(u32, @intCast(vocab_size - 2)));
            try tokens.append(allocator, token_id);
        }

        return tokens.toOwnedSlice(allocator);
    }

    pub fn forwardText(self: PromptEncoder, text: []const u8, weights: *WeightStore) !struct { tokens: Tensor, presence: Tensor } {
        const d = self.config.prompt_embed_dim;
        const max_len = self.config.text_max_seq_len;
        const vocab_size = self.config.text_vocab_size;

        const token_ids = try tokenizeSimple(text, vocab_size, max_len, self.allocator);
        defer self.allocator.free(token_ids);

        const seq_len = token_ids.len;

        // Embedding table: [vocab_size, d]
        const embed_w_shape = [_]usize{ vocab_size, d };
        const embed_w = try weights.getOrInit("text_encoder.embed_tokens.weight", &embed_w_shape, .kaiming, 401);

        // Positional embedding: [max_len, d]
        const pos_w_shape = [_]usize{ max_len, d };
        const pos_w = try weights.getOrInit("text_encoder.pos_embed.weight", &pos_w_shape, .xavier, 402);

        const out_shape = [_]usize{ 1, seq_len, d };
        var x = try Tensor.init(self.allocator, &out_shape);
        errdefer x.deinit();

        for (token_ids, 0..) |tid, idx| {
            const token_embed = embed_w.data[tid * d .. (tid + 1) * d];
            const pos_embed = pos_w.data[idx * d .. (idx + 1) * d];
            const out_slice = x.data[idx * d .. (idx + 1) * d];
            for (0..d) |j| {
                out_slice[j] = token_embed[j] + pos_embed[j];
            }
        }

        // Text Transformer Layers
        for (0..self.config.text_num_layers) |layer_idx| {
            var buf: [64]u8 = undefined;

            const ln1_name = try std.fmt.bufPrint(&buf, "text_encoder.layers.{d}.norm1.weight", .{layer_idx});
            const ln_shape = [_]usize{d};
            const ln1_w = try weights.getOrInit(ln1_name, &ln_shape, .ones, 410 + layer_idx);

            var norm1 = try math.layerNorm(self.allocator, x, ln1_w, null, 1e-6);
            defer norm1.deinit();

            // Self-Attention
            const qkv_name = try std.fmt.bufPrint(&buf, "text_encoder.layers.{d}.attn.qkv.weight", .{layer_idx});
            const qkv_shape = [_]usize{ 3 * d, d };
            const qkv_w = try weights.getOrInit(qkv_name, &qkv_shape, .xavier, 420 + layer_idx);

            var qkv = try math.linear(self.allocator, norm1, qkv_w, null);
            defer qkv.deinit();

            var q = try qkv.slice(self.allocator, 2, 0, d);
            defer q.deinit();
            var k = try qkv.slice(self.allocator, 2, d, 2 * d);
            defer k.deinit();
            var v = try qkv.slice(self.allocator, 2, 2 * d, 3 * d);
            defer v.deinit();

            var attn = try ops.multiHeadAttention(self.allocator, q, k, v, self.config.mask_decoder_num_heads, null);
            defer attn.deinit();

            x.addInPlace(attn);

            // Feed-Forward
            const ln2_name = try std.fmt.bufPrint(&buf, "text_encoder.layers.{d}.norm2.weight", .{layer_idx});
            const ln2_w = try weights.getOrInit(ln2_name, &ln_shape, .ones, 430 + layer_idx);

            var norm2 = try math.layerNorm(self.allocator, x, ln2_w, null, 1e-6);
            defer norm2.deinit();

            const mlp_dim = d * 4;
            const mlp1_name = try std.fmt.bufPrint(&buf, "text_encoder.layers.{d}.mlp.fc1.weight", .{layer_idx});
            const mlp1_shape = [_]usize{ mlp_dim, d };
            const mlp1_w = try weights.getOrInit(mlp1_name, &mlp1_shape, .kaiming, 440 + layer_idx);

            var mlp1_out = try math.linear(self.allocator, norm2, mlp1_w, null);
            defer mlp1_out.deinit();
            math.applyActivation(&mlp1_out, .gelu);

            const mlp2_name = try std.fmt.bufPrint(&buf, "text_encoder.layers.{d}.mlp.fc2.weight", .{layer_idx});
            const mlp2_shape = [_]usize{ d, mlp_dim };
            const mlp2_w = try weights.getOrInit(mlp2_name, &mlp2_shape, .kaiming, 450 + layer_idx);

            var mlp2_out = try math.linear(self.allocator, mlp1_out, mlp2_w, null);
            defer mlp2_out.deinit();

            x.addInPlace(mlp2_out);
        }

        // Presence token is token 0
        const presence_tok = try x.slice(self.allocator, 1, 0, 1);

        return .{
            .tokens = x,
            .presence = presence_tok,
        };
    }

    pub fn encodePoints(
        self: PromptEncoder,
        points: []const Point,
        weights: *WeightStore,
    ) !Tensor {
        const d = self.config.prompt_embed_dim;
        const n_pts = points.len;
        const out_shape = [_]usize{ 1, n_pts, d };
        var out = try Tensor.initZeros(self.allocator, &out_shape);

        // Point type embeddings: [4, d] (bg=0, fg=1, tl=2, br=3)
        const pt_type_shape = [_]usize{ 4, d };
        const pt_type_embed = try weights.getOrInit("prompt_encoder.point_embeddings.weight", &pt_type_shape, .xavier, 501);

        for (points, 0..) |pt, i| {
            const out_slice = out.data[i * d .. (i + 1) * d];

            // 2D Sinusoidal embedding of coordinates (x, y)
            const half_d = d / 2;
            for (0..half_d) |k| {
                const freq = @exp(-2.0 * @as(f32, @floatFromInt(k)) / @as(f32, @floatFromInt(half_d)) * @log(10000.0));
                const sin_x = @sin(pt.x * 2.0 * std.math.pi * freq);
                const cos_x = @cos(pt.y * 2.0 * std.math.pi * freq);
                out_slice[k] = sin_x;
                out_slice[half_d + k] = cos_x;
            }

            // Add label embedding
            if (pt.label >= 0 and pt.label < 4) {
                const label_u: usize = @intCast(pt.label);
                const type_slice = pt_type_embed.data[label_u * d .. (label_u + 1) * d];
                for (0..d) |j| {
                    out_slice[j] += type_slice[j];
                }
            }
        }

        return out;
    }

    pub fn forward(
        self: PromptEncoder,
        prompt: Prompt,
        weights: *WeightStore,
    ) !PromptEmbeddings {
        const d = self.config.prompt_embed_dim;
        const grid_h = self.config.gridH();
        const grid_w = self.config.gridW();

        var sparse_list: std.ArrayList(Tensor) = .empty;
        defer {
            for (sparse_list.items) |*t| t.deinit();
            sparse_list.deinit(self.allocator);
        }

        var presence_token: ?Tensor = null;

        // 1. Text Prompt
        if (prompt.text) |text| {
            const text_res = try self.forwardText(text, weights);
            try sparse_list.append(self.allocator, text_res.tokens);
            presence_token = text_res.presence;
        }

        // 2. Point Prompts
        if (prompt.points) |pts| {
            if (pts.len > 0) {
                const pt_tokens = try self.encodePoints(pts, weights);
                try sparse_list.append(self.allocator, pt_tokens);
            }
        }

        // 3. Box Prompt (converted to 2 corner points)
        if (prompt.box) |box| {
            const box_pts = [_]Point{
                .{ .x = box.x1, .y = box.y1, .label = 2 }, // Top-Left
                .{ .x = box.x2, .y = box.y2, .label = 3 }, // Bottom-Right
            };
            const box_tokens = try self.encodePoints(&box_pts, weights);
            try sparse_list.append(self.allocator, box_tokens);
        }

        // Default empty prompt token if nothing supplied
        if (sparse_list.items.len == 0) {
            const not_a_point_shape = [_]usize{ 1, 1, d };
            const not_a_point = try weights.getOrInit("prompt_encoder.not_a_point_embed", &not_a_point_shape, .xavier, 502);
            const empty_token = try not_a_point.clone(self.allocator);
            try sparse_list.append(self.allocator, empty_token);
        }

        // Concatenate all sparse prompt tokens along seq_len axis (dim 1)
        var sparse_embeddings = try Tensor.concat(self.allocator, sparse_list.items, 1);
        errdefer sparse_embeddings.deinit();

        // 4. Dense Mask Prompt
        const dense_shape = [_]usize{ 1, d, grid_h, grid_w };
        var dense_embeddings = try Tensor.initZeros(self.allocator, &dense_shape);
        errdefer dense_embeddings.deinit();

        if (prompt.mask) |in_mask| {
            var mask_resized = try ops.bilinearUpsample(self.allocator, in_mask, grid_h, grid_w, false);
            defer mask_resized.deinit();

            const mask_conv_shape = [_]usize{ d, 1, 1, 1 };
            const mask_conv_w = try weights.getOrInit("prompt_encoder.mask_downsampler.weight", &mask_conv_shape, .kaiming, 503);
            var mask_dense = try ops.conv2d(self.allocator, mask_resized, mask_conv_w, null, 1, 0);
            defer mask_dense.deinit();

            dense_embeddings.copyFrom(mask_dense);
        } else {
            const no_mask_shape = [_]usize{ 1, d, 1, 1 };
            const no_mask_embed = try weights.getOrInit("prompt_encoder.no_mask_embed", &no_mask_shape, .xavier, 504);
            for (0..grid_h) |y| {
                for (0..grid_w) |x| {
                    for (0..d) |c| {
                        dense_embeddings.set4(0, c, y, x, no_mask_embed.at4(0, c, 0, 0));
                    }
                }
            }
        }

        return PromptEmbeddings{
            .sparse_embeddings = sparse_embeddings,
            .dense_embeddings = dense_embeddings,
            .presence_token = presence_token,
        };
    }
};
