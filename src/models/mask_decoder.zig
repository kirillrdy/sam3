const std = @import("std");
const Tensor = @import("../tensor/tensor.zig").Tensor;
const math = @import("../tensor/math.zig");
const ops = @import("../tensor/ops.zig");
const SAM3Config = @import("config.zig").SAM3Config;
const WeightStore = @import("../weights/weight_loader.zig").WeightStore;

pub const MaskDecoder = struct {
    config: SAM3Config,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: SAM3Config) MaskDecoder {
        return MaskDecoder{
            .config = config,
            .allocator = allocator,
        };
    }

    pub const Output = struct {
        masks: Tensor,       // [1, num_masks, H, W] (logits)
        iou_scores: Tensor,  // [1, num_masks]

        pub fn deinit(self: *Output) void {
            self.masks.deinit();
            self.iou_scores.deinit();
        }
    };

    pub fn forward(
        self: MaskDecoder,
        image_embeddings: Tensor,
        image_pe: Tensor,
        sparse_prompt_embeddings: Tensor,
        dense_prompt_embeddings: Tensor,
        high_res_features: Tensor,
        multimask_output: bool,
        weights: *WeightStore,
    ) !Output {
        const d = self.config.encoder_embed_dim;
        const grid_h = self.config.gridH();
        const grid_w = self.config.gridW();
        const num_patches = grid_h * grid_w;
        const num_mask_tokens = self.config.mask_decoder_num_multimask_outputs + 1; // 1 IoU + 3 multimasks

        // 1. Fuse image embeddings with dense prompt embeddings
        var src_img = try image_embeddings.clone(self.allocator);
        defer src_img.deinit();
        src_img.addInPlace(dense_prompt_embeddings);

        // 2. Prepare Tokens: [1, 1 (iou) + num_mask_tokens + N_prompts, d]
        const iou_token_shape = [_]usize{ 1, 1, d };
        const iou_token = try weights.getOrInit("mask_decoder.iou_token.weight", &iou_token_shape, .xavier, 701);

        const mask_tokens_shape = [_]usize{ 1, num_mask_tokens, d };
        const mask_tokens = try weights.getOrInit("mask_decoder.mask_tokens.weight", &mask_tokens_shape, .xavier, 702);

        const token_list = [_]Tensor{ iou_token, mask_tokens, sparse_prompt_embeddings };
        var tokens = try Tensor.concat(self.allocator, &token_list, 1);
        defer tokens.deinit();

        // 3. Prepare Image Feature sequence: [1, num_patches, d]
        const perm_dims = [_]usize{ 0, 2, 3, 1 };
        var img_flat = try src_img.permute(self.allocator, &perm_dims);
        defer img_flat.deinit();
        const flat_shape = [_]usize{ 1, num_patches, d };
        try img_flat.reshape(&flat_shape);

        var pe_flat = if (image_pe.shape.len == 4) blk: {
            var p = try image_pe.permute(self.allocator, &perm_dims);
            try p.reshape(&flat_shape);
            break :blk p;
        } else try image_pe.clone(self.allocator);
        defer pe_flat.deinit();

        // 4. Two-Way Attention Transformer Layers
        for (0..self.config.mask_decoder_depth) |layer_idx| {
            var buf: [64]u8 = undefined;

            // Token Self-Attention
            const t_sa_qkv_name = try std.fmt.bufPrint(&buf, "mask_decoder.layers.{d}.self_attn.qkv.weight", .{layer_idx});
            const qkv_shape = [_]usize{ 3 * d, d };
            const t_sa_qkv_w = try weights.getOrInit(t_sa_qkv_name, &qkv_shape, .xavier, 710 + layer_idx);

            var t_qkv = try math.linear(self.allocator, tokens, t_sa_qkv_w, null);
            defer t_qkv.deinit();

            var t_q = try t_qkv.slice(self.allocator, 2, 0, d);
            defer t_q.deinit();
            var t_k = try t_qkv.slice(self.allocator, 2, d, 2 * d);
            defer t_k.deinit();
            var t_v = try t_qkv.slice(self.allocator, 2, 2 * d, 3 * d);
            defer t_v.deinit();

            var t_sa_out = try ops.multiHeadAttention(self.allocator, t_q, t_k, t_v, self.config.mask_decoder_num_heads, null);
            defer t_sa_out.deinit();
            tokens.addInPlace(t_sa_out);

            // Cross-Attention 1: Tokens attend to Image
            const ca1_q_name = try std.fmt.bufPrint(&buf, "mask_decoder.layers.{d}.cross_attn_token_to_img.q_proj.weight", .{layer_idx});
            const d_by_d = [_]usize{ d, d };
            const ca1_q_w = try weights.getOrInit(ca1_q_name, &d_by_d, .xavier, 720 + layer_idx);
            const ca1_k_name = try std.fmt.bufPrint(&buf, "mask_decoder.layers.{d}.cross_attn_token_to_img.k_proj.weight", .{layer_idx});
            const ca1_k_w = try weights.getOrInit(ca1_k_name, &d_by_d, .xavier, 721 + layer_idx);
            const ca1_v_name = try std.fmt.bufPrint(&buf, "mask_decoder.layers.{d}.cross_attn_token_to_img.v_proj.weight", .{layer_idx});
            const ca1_v_w = try weights.getOrInit(ca1_v_name, &d_by_d, .xavier, 722 + layer_idx);

            var q1 = try math.linear(self.allocator, tokens, ca1_q_w, null);
            defer q1.deinit();
            var k1 = try math.linear(self.allocator, img_flat, ca1_k_w, null);
            defer k1.deinit();
            var v1 = try math.linear(self.allocator, img_flat, ca1_v_w, null);
            defer v1.deinit();

            var ca1_out = try ops.multiHeadAttention(self.allocator, q1, k1, v1, self.config.mask_decoder_num_heads, null);
            defer ca1_out.deinit();
            tokens.addInPlace(ca1_out);

            // Token MLP
            const t_mlp_name = try std.fmt.bufPrint(&buf, "mask_decoder.layers.{d}.mlp.fc1.weight", .{layer_idx});
            const mlp_dim = d * 4;
            const mlp1_shape = [_]usize{ mlp_dim, d };
            const t_mlp_w = try weights.getOrInit(t_mlp_name, &mlp1_shape, .kaiming, 730 + layer_idx);
            const t_mlp2_name = try std.fmt.bufPrint(&buf, "mask_decoder.layers.{d}.mlp.fc2.weight", .{layer_idx});
            const mlp2_shape = [_]usize{ d, mlp_dim };
            const t_mlp2_w = try weights.getOrInit(t_mlp2_name, &mlp2_shape, .kaiming, 731 + layer_idx);

            var t_mlp_h = try math.linear(self.allocator, tokens, t_mlp_w, null);
            defer t_mlp_h.deinit();
            math.applyActivation(&t_mlp_h, .gelu);

            var t_mlp_out = try math.linear(self.allocator, t_mlp_h, t_mlp2_w, null);
            defer t_mlp_out.deinit();
            tokens.addInPlace(t_mlp_out);

            // Cross-Attention 2: Image attends to Tokens
            const ca2_q_name = try std.fmt.bufPrint(&buf, "mask_decoder.layers.{d}.cross_attn_img_to_token.q_proj.weight", .{layer_idx});
            const ca2_q_w = try weights.getOrInit(ca2_q_name, &d_by_d, .xavier, 740 + layer_idx);
            const ca2_k_name = try std.fmt.bufPrint(&buf, "mask_decoder.layers.{d}.cross_attn_img_to_token.k_proj.weight", .{layer_idx});
            const ca2_k_w = try weights.getOrInit(ca2_k_name, &d_by_d, .xavier, 741 + layer_idx);
            const ca2_v_name = try std.fmt.bufPrint(&buf, "mask_decoder.layers.{d}.cross_attn_img_to_token.v_proj.weight", .{layer_idx});
            const ca2_v_w = try weights.getOrInit(ca2_v_name, &d_by_d, .xavier, 742 + layer_idx);

            var q2 = try math.linear(self.allocator, img_flat, ca2_q_w, null);
            defer q2.deinit();
            var k2 = try math.linear(self.allocator, tokens, ca2_k_w, null);
            defer k2.deinit();
            var v2 = try math.linear(self.allocator, tokens, ca2_v_w, null);
            defer v2.deinit();

            var ca2_out = try ops.multiHeadAttention(self.allocator, q2, k2, v2, self.config.mask_decoder_num_heads, null);
            defer ca2_out.deinit();
            img_flat.addInPlace(ca2_out);
        }

        // 5. IoU Head: MLP on Token 0 (iou_token)
        var iou_tok_out = try tokens.slice(self.allocator, 1, 0, 1);
        defer iou_tok_out.deinit();

        const iou_mlp1_shape = [_]usize{ d, d };
        const iou_mlp1_w = try weights.getOrInit("mask_decoder.iou_head.fc1.weight", &iou_mlp1_shape, .kaiming, 750);
        const iou_mlp2_shape = [_]usize{ num_mask_tokens, d };
        const iou_mlp2_w = try weights.getOrInit("mask_decoder.iou_head.fc2.weight", &iou_mlp2_shape, .xavier, 751);

        var iou_h = try math.linear(self.allocator, iou_tok_out, iou_mlp1_w, null);
        defer iou_h.deinit();
        math.applyActivation(&iou_h, .gelu);

        var iou_pred = try math.linear(self.allocator, iou_h, iou_mlp2_w, null);
        defer iou_pred.deinit();

        const num_masks_to_return: usize = if (multimask_output) self.config.mask_decoder_num_multimask_outputs else 1;
        const iou_start: usize = if (multimask_output) 1 else 0;

        const iou_out_shape = [_]usize{ 1, num_masks_to_return };
        var iou_scores = try Tensor.init(self.allocator, &iou_out_shape);
        errdefer iou_scores.deinit();

        for (0..num_masks_to_return) |m_idx| {
            iou_scores.set2(0, m_idx, math.sigmoid(iou_pred.at3(0, 0, iou_start + m_idx)));
        }

        // 6. Hypernetwork MLP: maps mask tokens (tokens 1 .. 1+num_mask_tokens) -> dynamic mask weights
        var mask_tokens_out = try tokens.slice(self.allocator, 1, 1, 1 + num_mask_tokens);
        defer mask_tokens_out.deinit();

        const hyper_d = @max(1, d / 8);
        const hyper_w1_shape = [_]usize{ d, d };
        const hyper_w1 = try weights.getOrInit("mask_decoder.hyper_mlp.fc1.weight", &hyper_w1_shape, .kaiming, 760);
        const hyper_w2_shape = [_]usize{ hyper_d, d };
        const hyper_w2 = try weights.getOrInit("mask_decoder.hyper_mlp.fc2.weight", &hyper_w2_shape, .xavier, 761);

        var hyper_h = try math.linear(self.allocator, mask_tokens_out, hyper_w1, null);
        defer hyper_h.deinit();
        math.applyActivation(&hyper_h, .gelu);

        var hyper_weights = try math.linear(self.allocator, hyper_h, hyper_w2, null);
        defer hyper_weights.deinit();

        // 7. Feature Upsampling: [1, num_patches, d] -> [1, d, grid_h, grid_w] -> upsample to [1, hyper_d, grid_h*4, grid_w*4]
        const neck_shape = [_]usize{ 1, grid_h, grid_w, d };
        var reshaped_img = try img_flat.clone(self.allocator);
        defer reshaped_img.deinit();
        try reshaped_img.reshape(&neck_shape);

        const perm_back = [_]usize{ 0, 3, 1, 2 };
        var img_2d = try reshaped_img.permute(self.allocator, &perm_back);
        defer img_2d.deinit();

        // Upsample 4x
        var upsampled_img = try ops.bilinearUpsample(self.allocator, img_2d, grid_h * 4, grid_w * 4, false);
        defer upsampled_img.deinit();

        // Project upsampled_img (d channels) to hyper_d channels
        const up_proj_shape = [_]usize{ hyper_d, d, 1, 1 };
        const up_proj_w = try weights.getOrInit("mask_decoder.upsample_proj.weight", &up_proj_shape, .kaiming, 770);
        var up_proj = try ops.conv2d(self.allocator, upsampled_img, up_proj_w, null, 1, 0);
        defer up_proj.deinit();

        // Add high_res_features projection
        const hr_d = high_res_features.shape[1];
        const hr_proj_shape = [_]usize{ hyper_d, hr_d, 1, 1 };
        const hr_proj_w = try weights.getOrInit("mask_decoder.hr_proj.weight", &hr_proj_shape, .kaiming, 771);
        var hr_proj = try ops.conv2d(self.allocator, high_res_features, hr_proj_w, null, 1, 0);
        defer hr_proj.deinit();

        up_proj.addInPlace(hr_proj);

        // 8. Mask computation: Dot product of dynamic hypernetwork weights with spatial upsampled features
        const h_out = grid_h * 4;
        const w_out = grid_w * 4;

        const masks_shape = [_]usize{ 1, num_masks_to_return, h_out, w_out };
        var masks = try Tensor.init(self.allocator, &masks_shape);
        errdefer masks.deinit();

        for (0..num_masks_to_return) |m_idx| {
            const token_idx = iou_start + m_idx;
            const hw_slice = hyper_weights.data[token_idx * hyper_d .. (token_idx + 1) * hyper_d];

            for (0..h_out) |y| {
                for (0..w_out) |x| {
                    var val: f32 = 0.0;
                    for (0..hyper_d) |c| {
                        val += up_proj.at4(0, c, y, x) * hw_slice[c];
                    }
                    masks.set4(0, m_idx, y, x, val);
                }
            }
        }

        return Output{
            .masks = masks,
            .iou_scores = iou_scores,
        };
    }
};
