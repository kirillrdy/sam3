const std = @import("std");
const Tensor = @import("../tensor/tensor.zig").Tensor;
const math = @import("../tensor/math.zig");
const ops = @import("../tensor/ops.zig");
const SAM3Config = @import("config.zig").SAM3Config;
const WeightStore = @import("../weights/weight_loader.zig").WeightStore;

pub const FrameMemory = struct {
    frame_idx: usize,
    is_keyframe: bool,
    memory_embedding: Tensor, // [1, D_mem, grid_h, grid_w]
    obj_id: usize,

    pub fn deinit(self: *FrameMemory) void {
        self.memory_embedding.deinit();
    }
};

pub const MemoryEncoder = struct {
    config: SAM3Config,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: SAM3Config) MemoryEncoder {
        return MemoryEncoder{
            .config = config,
            .allocator = allocator,
        };
    }

    pub fn encode(
        self: MemoryEncoder,
        image_embeddings: Tensor,
        mask_logits: Tensor,
        weights: *WeightStore,
    ) !Tensor {
        const d = self.config.encoder_embed_dim;
        const d_mem = self.config.memory_dim;
        const grid_h = self.config.gridH();
        const grid_w = self.config.gridW();

        // 1. Sigmoid mask
        var mask_prob = try mask_logits.clone(self.allocator);
        defer mask_prob.deinit();
        math.applyActivation(&mask_prob, .sigmoid);

        // 2. Downsample mask to [1, d_mem/2, grid_h, grid_w] via Conv2d
        const mask_conv_shape = [_]usize{ d_mem / 2, 1, 4, 4 };
        const mask_conv_w = try weights.getOrInit("memory_encoder.mask_downsampler.weight", &mask_conv_shape, .kaiming, 801);
        var mask_feats = try ops.conv2d(self.allocator, mask_prob, mask_conv_w, null, 4, 0);
        defer mask_feats.deinit();

        var mask_feats_aligned = if (mask_feats.shape[2] != grid_h or mask_feats.shape[3] != grid_w)
            try ops.bilinearUpsample(self.allocator, mask_feats, grid_h, grid_w, false)
        else
            try mask_feats.clone(self.allocator);
        defer mask_feats_aligned.deinit();

        // 3. Project image embeddings [1, d, grid_h, grid_w] -> [1, d_mem/2, grid_h, grid_w]
        const img_proj_shape = [_]usize{ d_mem / 2, d, 1, 1 };
        const img_proj_w = try weights.getOrInit("memory_encoder.img_proj.weight", &img_proj_shape, .kaiming, 802);
        var img_feats = try ops.conv2d(self.allocator, image_embeddings, img_proj_w, null, 1, 0);
        defer img_feats.deinit();

        // 4. Concat along channel axis (dim 1) -> [1, d_mem, grid_h, grid_w]
        const feat_list = [_]Tensor{ img_feats, mask_feats_aligned };
        var fused = try Tensor.concat(self.allocator, &feat_list, 1);
        defer fused.deinit();

        // 5. Fusion 3x3 Conv
        const fuse_conv_shape = [_]usize{ d_mem, d_mem, 3, 3 };
        const fuse_conv_w = try weights.getOrInit("memory_encoder.fuse_conv.weight", &fuse_conv_shape, .kaiming, 803);
        const memory_embedding = try ops.conv2d(self.allocator, fused, fuse_conv_w, null, 1, 1);

        return memory_embedding;
    }
};

pub const MemoryBank = struct {
    allocator: std.mem.Allocator,
    max_recent_frames: usize,
    keyframes: std.ArrayList(FrameMemory),
    recent_frames: std.ArrayList(FrameMemory),

    pub fn init(allocator: std.mem.Allocator, max_recent: usize) MemoryBank {
        return MemoryBank{
            .allocator = allocator,
            .max_recent_frames = max_recent,
            .keyframes = .empty,
            .recent_frames = .empty,
        };
    }

    pub fn deinit(self: *MemoryBank) void {
        for (self.keyframes.items) |*m| m.deinit();
        self.keyframes.deinit(self.allocator);
        for (self.recent_frames.items) |*m| m.deinit();
        self.recent_frames.deinit(self.allocator);
    }

    pub fn clear(self: *MemoryBank) void {
        for (self.keyframes.items) |*m| m.deinit();
        self.keyframes.clearRetainingCapacity();
        for (self.recent_frames.items) |*m| m.deinit();
        self.recent_frames.clearRetainingCapacity();
    }

    pub fn addMemory(
        self: *MemoryBank,
        frame_idx: usize,
        is_keyframe: bool,
        obj_id: usize,
        memory_emb: Tensor,
    ) !void {
        const mem_clone = try memory_emb.clone(self.allocator);
        const mem = FrameMemory{
            .frame_idx = frame_idx,
            .is_keyframe = is_keyframe,
            .obj_id = obj_id,
            .memory_embedding = mem_clone,
        };

        if (is_keyframe) {
            try self.keyframes.append(self.allocator, mem);
        } else {
            if (self.recent_frames.items.len >= self.max_recent_frames) {
                var oldest = self.recent_frames.orderedRemove(0);
                oldest.deinit();
            }
            try self.recent_frames.append(self.allocator, mem);
        }
    }

    pub fn getMemoriesForObject(self: MemoryBank, obj_id: usize, allocator: std.mem.Allocator) ![]FrameMemory {
        var list: std.ArrayList(FrameMemory) = .empty;
        errdefer list.deinit(allocator);

        for (self.keyframes.items) |m| {
            if (m.obj_id == obj_id) try list.append(allocator, m);
        }
        for (self.recent_frames.items) |m| {
            if (m.obj_id == obj_id) try list.append(allocator, m);
        }
        return list.toOwnedSlice(allocator);
    }
};

pub const MemoryAttention = struct {
    config: SAM3Config,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: SAM3Config) MemoryAttention {
        return MemoryAttention{
            .config = config,
            .allocator = allocator,
        };
    }

    pub fn forward(
        self: MemoryAttention,
        curr_frame_idx: usize,
        image_embeddings: Tensor,
        memories: []const FrameMemory,
        weights: *WeightStore,
    ) !Tensor {
        if (memories.len == 0) {
            return image_embeddings.clone(self.allocator);
        }

        const d = self.config.encoder_embed_dim;
        const d_mem = self.config.memory_dim;
        const grid_h = self.config.gridH();
        const grid_w = self.config.gridW();
        const num_patches = grid_h * grid_w;

        // Flatten current image tokens: [1, num_patches, d]
        const perm_dims = [_]usize{ 0, 2, 3, 1 };
        var q_tensor = try image_embeddings.permute(self.allocator, &perm_dims);
        defer q_tensor.deinit();
        const q_shape = [_]usize{ 1, num_patches, d };
        try q_tensor.reshape(&q_shape);

        // Collect all memory tokens: [1, memories.len * num_patches, d_mem]
        var mem_tokens_list: std.ArrayList(Tensor) = .empty;
        defer {
            for (mem_tokens_list.items) |*t| t.deinit();
            mem_tokens_list.deinit(self.allocator);
        }

        for (memories) |mem| {
            var mem_flat = try mem.memory_embedding.permute(self.allocator, &perm_dims);
            defer mem_flat.deinit();
            const m_shape = [_]usize{ 1, num_patches, d_mem };
            try mem_flat.reshape(&m_shape);

            const dt = @abs(@as(isize, @intCast(curr_frame_idx)) - @as(isize, @intCast(mem.frame_idx)));
            const temporal_factor = 1.0 / (1.0 + 0.05 * @as(f32, @floatFromInt(dt)));

            var weighted_mem = try mem_flat.clone(self.allocator);
            weighted_mem.scaleInPlace(temporal_factor);
            try mem_tokens_list.append(self.allocator, weighted_mem);
        }

        var all_memories = try Tensor.concat(self.allocator, mem_tokens_list.items, 1);
        defer all_memories.deinit();

        // Cross Attention: Q(image) attends to K, V(memories)
        const q_proj_shape = [_]usize{ d, d };
        const q_proj = try weights.getOrInit("memory_attn.q_proj.weight", &q_proj_shape, .xavier, 810);
        const k_proj_shape = [_]usize{ d, d_mem };
        const k_proj = try weights.getOrInit("memory_attn.k_proj.weight", &k_proj_shape, .xavier, 811);
        const v_proj_shape = [_]usize{ d, d_mem };
        const v_proj = try weights.getOrInit("memory_attn.v_proj.weight", &v_proj_shape, .xavier, 812);

        var q = try math.linear(self.allocator, q_tensor, q_proj, null);
        defer q.deinit();
        var k = try math.linear(self.allocator, all_memories, k_proj, null);
        defer k.deinit();
        var v = try math.linear(self.allocator, all_memories, v_proj, null);
        defer v.deinit();

        var attended = try ops.multiHeadAttention(self.allocator, q, k, v, self.config.mask_decoder_num_heads, null);
        defer attended.deinit();

        // Output projection + residual
        const out_proj_shape = [_]usize{ d, d };
        const out_proj = try weights.getOrInit("memory_attn.out_proj.weight", &out_proj_shape, .xavier, 813);
        var proj_out = try math.linear(self.allocator, attended, out_proj, null);
        defer proj_out.deinit();

        q_tensor.addInPlace(proj_out);

        // Reshape back to [1, d, grid_h, grid_w]
        const neck_shape = [_]usize{ 1, grid_h, grid_w, d };
        var reshaped = try q_tensor.clone(self.allocator);
        defer reshaped.deinit();
        try reshaped.reshape(&neck_shape);

        const perm_back = [_]usize{ 0, 3, 1, 2 };
        const conditioned_img = try reshaped.permute(self.allocator, &perm_back);
        return conditioned_img;
    }
};
