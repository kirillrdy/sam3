const std = @import("std");
const Tensor = @import("../tensor/tensor.zig").Tensor;
const math = @import("../tensor/math.zig");
const ops = @import("../tensor/ops.zig");
const SAM3 = @import("../models/sam3.zig").SAM3;
const Prompt = @import("../models/prompt_encoder.zig").Prompt;
const Point = @import("../models/prompt_encoder.zig").Point;
const Box = @import("../models/prompt_encoder.zig").Box;

pub const ObjectTrack = struct {
    obj_id: usize,
    label: []const u8,
    active: bool,
    last_seen_frame: usize,
    initial_prompt: ?Prompt,
};

pub const TrackedMask = struct {
    obj_id: usize,
    mask: Tensor, // [1, 1, H, W]
    iou_score: f32,
    presence_score: f32,
    box: Box,

    pub fn deinit(self: *TrackedMask) void {
        self.mask.deinit();
    }
};

pub const FrameTrackingResult = struct {
    frame_idx: usize,
    objects: []TrackedMask,

    pub fn deinit(self: *FrameTrackingResult, allocator: std.mem.Allocator) void {
        for (self.objects) |*obj| {
            obj.deinit();
        }
        allocator.free(self.objects);
    }
};

pub const SAM3VideoPredictor = struct {
    allocator: std.mem.Allocator,
    sam3: *SAM3,
    objects: std.ArrayList(ObjectTrack),

    pub fn init(allocator: std.mem.Allocator, sam3: *SAM3) SAM3VideoPredictor {
        return SAM3VideoPredictor{
            .allocator = allocator,
            .sam3 = sam3,
            .objects = .empty,
        };
    }

    pub fn deinit(self: *SAM3VideoPredictor) void {
        for (self.objects.items) |*obj| {
            self.allocator.free(obj.label);
        }
        self.objects.deinit(self.allocator);
    }

    pub fn addObject(self: *SAM3VideoPredictor, obj_id: usize, label: []const u8) !void {
        const label_copy = try self.allocator.dupe(u8, label);
        try self.objects.append(self.allocator, ObjectTrack{
            .obj_id = obj_id,
            .label = label_copy,
            .active = true,
            .last_seen_frame = 0,
            .initial_prompt = null,
        });
    }

    pub fn addPrompt(
        self: *SAM3VideoPredictor,
        frame_idx: usize,
        obj_id: usize,
        prompt: Prompt,
        image: Tensor,
    ) !void {
        // Run segmentation on keyframe
        var res = try self.sam3.segmentImage(image, prompt, false);
        defer res.deinit();

        // Encode image for memory bank
        var enc_out = try self.sam3.image_encoder.forward(image, &self.sam3.weights);
        defer enc_out.deinit();

        // Encode mask memory
        var mem_emb = try self.sam3.memory_encoder.encode(enc_out.image_embeddings, res.masks, &self.sam3.weights);
        defer mem_emb.deinit();

        // Store memory in memory bank
        try self.sam3.memory_bank.addMemory(frame_idx, true, obj_id, mem_emb);

        // Update object state
        for (self.objects.items) |*obj| {
            if (obj.obj_id == obj_id) {
                obj.last_seen_frame = frame_idx;
                obj.initial_prompt = prompt;
                break;
            }
        }
    }

    pub fn trackFrame(
        self: *SAM3VideoPredictor,
        frame_idx: usize,
        image: Tensor,
    ) !FrameTrackingResult {
        var result_list: std.ArrayList(TrackedMask) = .empty;
        errdefer {
            for (result_list.items) |*obj| obj.deinit();
            result_list.deinit(self.allocator);
        }

        // 1. Encode current image frame
        var enc_out = try self.sam3.image_encoder.forward(image, &self.sam3.weights);
        defer enc_out.deinit();

        const grid_h = self.sam3.config.gridH();
        const grid_w = self.sam3.config.gridW();
        var img_pe = try ops.sinusoidalEmbedding2D(self.allocator, grid_h, grid_w, self.sam3.config.encoder_embed_dim, 10000.0, 1.0);
        defer img_pe.deinit();

        // 2. Track each active object using memory attention
        for (self.objects.items) |*obj| {
            if (!obj.active) continue;

            const memories = try self.sam3.memory_bank.getMemoriesForObject(obj.obj_id, self.allocator);
            defer self.allocator.free(memories);

            var conditioned_img = try self.sam3.memory_attention.forward(
                frame_idx,
                enc_out.image_embeddings,
                memories,
                &self.sam3.weights,
            );
            defer conditioned_img.deinit();

            const prompt = obj.initial_prompt orelse Prompt{ .text = obj.label };
            var prompt_emb = try self.sam3.prompt_encoder.forward(prompt, &self.sam3.weights);
            defer prompt_emb.deinit();

            var presence_score: f32 = 1.0;
            if (prompt.text != null and prompt_emb.presence_token != null) {
                var det_out = try self.sam3.detector.forward(
                    conditioned_img,
                    prompt_emb.sparse_embeddings,
                    prompt_emb.presence_token,
                    &self.sam3.weights,
                );
                presence_score = det_out.presence_score;
                det_out.deinit(self.allocator);
            }

            var dec_out = try self.sam3.mask_decoder.forward(
                conditioned_img,
                img_pe,
                prompt_emb.sparse_embeddings,
                prompt_emb.dense_embeddings,
                enc_out.high_res_features,
                false,
                &self.sam3.weights,
            );
            defer dec_out.deinit();

            const iou = dec_out.iou_scores.at2(0, 0);

            const h_m = dec_out.masks.shape[2];
            const w_m = dec_out.masks.shape[3];
            var min_x: usize = w_m;
            var min_y: usize = h_m;
            var max_x: usize = 0;
            var max_y: usize = 0;
            var has_mask_pixels = false;

            for (0..h_m) |y| {
                for (0..w_m) |x| {
                    if (dec_out.masks.at4(0, 0, y, x) > 0.0) {
                        has_mask_pixels = true;
                        if (x < min_x) min_x = x;
                        if (x > max_x) max_x = x;
                        if (y < min_y) min_y = y;
                        if (y > max_y) max_y = y;
                    }
                }
            }

            const box = if (has_mask_pixels) Box{
                .x1 = @as(f32, @floatFromInt(min_x)) / @as(f32, @floatFromInt(w_m)),
                .y1 = @as(f32, @floatFromInt(min_y)) / @as(f32, @floatFromInt(h_m)),
                .x2 = @as(f32, @floatFromInt(max_x)) / @as(f32, @floatFromInt(w_m)),
                .y2 = @as(f32, @floatFromInt(max_y)) / @as(f32, @floatFromInt(h_m)),
            } else Box{ .x1 = 0, .y1 = 0, .x2 = 0, .y2 = 0 };

            var mask_clone = try dec_out.masks.clone(self.allocator);
            errdefer mask_clone.deinit();

            var mem_emb = try self.sam3.memory_encoder.encode(conditioned_img, dec_out.masks, &self.sam3.weights);
            defer mem_emb.deinit();
            try self.sam3.memory_bank.addMemory(frame_idx, false, obj.obj_id, mem_emb);

            try result_list.append(self.allocator, TrackedMask{
                .obj_id = obj.obj_id,
                .mask = mask_clone,
                .iou_score = iou,
                .presence_score = presence_score,
                .box = box,
            });
        }

        return FrameTrackingResult{
            .frame_idx = frame_idx,
            .objects = try result_list.toOwnedSlice(self.allocator),
        };
    }
};
