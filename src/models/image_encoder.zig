const std = @import("std");
const Tensor = @import("../tensor/tensor.zig").Tensor;
const math = @import("../tensor/math.zig");
const ops = @import("../tensor/ops.zig");
const SAM3Config = @import("config.zig").SAM3Config;
const WeightStore = @import("../weights/weight_loader.zig").WeightStore;

pub const ImageEncoder = struct {
    config: SAM3Config,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: SAM3Config) ImageEncoder {
        return ImageEncoder{
            .config = config,
            .allocator = allocator,
        };
    }

    pub const Output = struct {
        image_embeddings: Tensor,     // [1, D, H_grid, W_grid]
        high_res_features: Tensor,    // [1, D/4, H_grid*4, W_grid*4]

        pub fn deinit(self: *Output) void {
            self.image_embeddings.deinit();
            self.high_res_features.deinit();
        }
    };

    pub fn forward(self: ImageEncoder, image: Tensor, weights: *WeightStore) !Output {
        // Image: [1, 3, H, W]
        std.debug.assert(image.shape.len == 4);
        std.debug.assert(image.shape[0] == 1 and image.shape[1] == 3);

        const d = self.config.encoder_embed_dim;
        const p = self.config.patch_size;
        const grid_h = self.config.gridH();
        const grid_w = self.config.gridW();
        const num_patches = grid_h * grid_w;

        // 1. Patch Embedding: Conv2D(3, d, kernel_size=p, stride=p)
        const patch_w_shape = [_]usize{ d, 3, p, p };
        const patch_w = try weights.getOrInit("image_encoder.patch_embed.weight", &patch_w_shape, .kaiming, 101);
        const patch_b_shape = [_]usize{d};
        const patch_b = try weights.getOrInit("image_encoder.patch_embed.bias", &patch_b_shape, .zeros, 102);

        var patches = try ops.conv2d(self.allocator, image, patch_w, patch_b, p, 0);
        defer patches.deinit();

        // 2. Add 2D Positional Embeddings: [1, d, grid_h, grid_w]
        const pos_shape = [_]usize{ 1, d, grid_h, grid_w };
        const pos_embed = try weights.getOrInit("image_encoder.pos_embed", &pos_shape, .xavier, 103);
        patches.addInPlace(pos_embed);

        // 3. Reshape/Permute to [1, num_patches, d]
        // patches is [1, d, grid_h, grid_w] -> permute to [1, grid_h, grid_w, d] -> reshape to [1, num_patches, d]
        const perm_dims = [_]usize{ 0, 2, 3, 1 };
        var seq_patches = try patches.permute(self.allocator, &perm_dims);
        defer seq_patches.deinit();

        const seq_shape = [_]usize{ 1, num_patches, d };
        try seq_patches.reshape(&seq_shape);

        // 4. Transformer Blocks
        var x = try seq_patches.clone(self.allocator);
        defer x.deinit();

        const mlp_dim = @as(usize, @intFromFloat(@as(f32, @floatFromInt(d)) * self.config.encoder_mlp_ratio));

        for (0..self.config.encoder_depth) |layer_idx| {
            var buf: [64]u8 = undefined;

            // --- Pre-LN 1 ---
            const ln1_g_name = try std.fmt.bufPrint(&buf, "image_encoder.blocks.{d}.norm1.weight", .{layer_idx});
            const ln1_g_shape = [_]usize{d};
            const ln1_g = try weights.getOrInit(ln1_g_name, &ln1_g_shape, .ones, 200 + layer_idx);
            const ln1_b_name = try std.fmt.bufPrint(&buf, "image_encoder.blocks.{d}.norm1.bias", .{layer_idx});
            const ln1_b = try weights.getOrInit(ln1_b_name, &ln1_g_shape, .zeros, 201 + layer_idx);

            var norm1_out = try math.layerNorm(self.allocator, x, ln1_g, ln1_b, 1e-6);
            defer norm1_out.deinit();

            // Self-Attention Q, K, V
            const qkv_w_name = try std.fmt.bufPrint(&buf, "image_encoder.blocks.{d}.attn.qkv.weight", .{layer_idx});
            const qkv_w_shape = [_]usize{ 3 * d, d };
            const qkv_w = try weights.getOrInit(qkv_w_name, &qkv_w_shape, .xavier, 202 + layer_idx);
            const qkv_b_name = try std.fmt.bufPrint(&buf, "image_encoder.blocks.{d}.attn.qkv.bias", .{layer_idx});
            const qkv_b_shape = [_]usize{3 * d};
            const qkv_b = try weights.getOrInit(qkv_b_name, &qkv_b_shape, .zeros, 203 + layer_idx);

            var qkv = try math.linear(self.allocator, norm1_out, qkv_w, qkv_b);
            defer qkv.deinit();

            // Slice Q, K, V
            var q = try qkv.slice(self.allocator, 2, 0, d);
            defer q.deinit();
            var k = try qkv.slice(self.allocator, 2, d, 2 * d);
            defer k.deinit();
            var v = try qkv.slice(self.allocator, 2, 2 * d, 3 * d);
            defer v.deinit();

            var attn_out = try ops.multiHeadAttention(self.allocator, q, k, v, self.config.encoder_num_heads, null);
            defer attn_out.deinit();

            // Attn Output Projection
            const proj_w_name = try std.fmt.bufPrint(&buf, "image_encoder.blocks.{d}.attn.proj.weight", .{layer_idx});
            const proj_w_shape = [_]usize{ d, d };
            const proj_w = try weights.getOrInit(proj_w_name, &proj_w_shape, .xavier, 204 + layer_idx);
            const proj_b_name = try std.fmt.bufPrint(&buf, "image_encoder.blocks.{d}.attn.proj.bias", .{layer_idx});
            const proj_b = try weights.getOrInit(proj_b_name, &ln1_g_shape, .zeros, 205 + layer_idx);

            var attn_proj = try math.linear(self.allocator, attn_out, proj_w, proj_b);
            defer attn_proj.deinit();

            // Residual 1
            x.addInPlace(attn_proj);

            // --- Pre-LN 2 ---
            const ln2_g_name = try std.fmt.bufPrint(&buf, "image_encoder.blocks.{d}.norm2.weight", .{layer_idx});
            const ln2_g = try weights.getOrInit(ln2_g_name, &ln1_g_shape, .ones, 206 + layer_idx);
            const ln2_b_name = try std.fmt.bufPrint(&buf, "image_encoder.blocks.{d}.norm2.bias", .{layer_idx});
            const ln2_b = try weights.getOrInit(ln2_b_name, &ln1_g_shape, .zeros, 207 + layer_idx);

            var norm2_out = try math.layerNorm(self.allocator, x, ln2_g, ln2_b, 1e-6);
            defer norm2_out.deinit();

            // MLP: Linear(d -> mlp_dim) -> GELU -> Linear(mlp_dim -> d)
            const mlp1_w_name = try std.fmt.bufPrint(&buf, "image_encoder.blocks.{d}.mlp.fc1.weight", .{layer_idx});
            const mlp1_w_shape = [_]usize{ mlp_dim, d };
            const mlp1_w = try weights.getOrInit(mlp1_w_name, &mlp1_w_shape, .kaiming, 208 + layer_idx);
            const mlp1_b_name = try std.fmt.bufPrint(&buf, "image_encoder.blocks.{d}.mlp.fc1.bias", .{layer_idx});
            const mlp1_b_shape = [_]usize{mlp_dim};
            const mlp1_b = try weights.getOrInit(mlp1_b_name, &mlp1_b_shape, .zeros, 209 + layer_idx);

            var mlp1_out = try math.linear(self.allocator, norm2_out, mlp1_w, mlp1_b);
            defer mlp1_out.deinit();
            math.applyActivation(&mlp1_out, .gelu);

            const mlp2_w_name = try std.fmt.bufPrint(&buf, "image_encoder.blocks.{d}.mlp.fc2.weight", .{layer_idx});
            const mlp2_w_shape = [_]usize{ d, mlp_dim };
            const mlp2_w = try weights.getOrInit(mlp2_w_name, &mlp2_w_shape, .kaiming, 210 + layer_idx);
            const mlp2_b_name = try std.fmt.bufPrint(&buf, "image_encoder.blocks.{d}.mlp.fc2.bias", .{layer_idx});
            const mlp2_b = try weights.getOrInit(mlp2_b_name, &ln1_g_shape, .zeros, 211 + layer_idx);

            var mlp2_out = try math.linear(self.allocator, mlp1_out, mlp2_w, mlp2_b);
            defer mlp2_out.deinit();

            // Residual 2
            x.addInPlace(mlp2_out);
        }

        // 5. Neck: Reshape [1, num_patches, d] -> [1, grid_h, grid_w, d] -> permute to [1, d, grid_h, grid_w]
        const neck_shape = [_]usize{ 1, grid_h, grid_w, d };
        var neck_tensor = try x.clone(self.allocator);
        defer neck_tensor.deinit();
        try neck_tensor.reshape(&neck_shape);

        const neck_perm = [_]usize{ 0, 3, 1, 2 };
        var image_embeddings = try neck_tensor.permute(self.allocator, &neck_perm);
        errdefer image_embeddings.deinit();

        // 6. High-Res Features for edge detail: Bilinear upsampling 4x + conv [1, d/4, grid_h*4, grid_w*4]
        const high_res_d = @max(1, d / 4);
        var up_features = try ops.bilinearUpsample(self.allocator, image_embeddings, grid_h * 4, grid_w * 4, false);
        defer up_features.deinit();

        const hr_conv_w_shape = [_]usize{ high_res_d, d, 1, 1 };
        const hr_conv_w = try weights.getOrInit("image_encoder.neck.hr_conv.weight", &hr_conv_w_shape, .kaiming, 301);
        const hr_conv_b_shape = [_]usize{high_res_d};
        const hr_conv_b = try weights.getOrInit("image_encoder.neck.hr_conv.bias", &hr_conv_b_shape, .zeros, 302);

        var high_res_features = try ops.conv2d(self.allocator, up_features, hr_conv_w, hr_conv_b, 1, 0);
        errdefer high_res_features.deinit();

        return Output{
            .image_embeddings = image_embeddings,
            .high_res_features = high_res_features,
        };
    }
};
