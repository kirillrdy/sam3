const std = @import("std");
const Tensor = @import("../tensor/tensor.zig").Tensor;
const math = @import("../tensor/math.zig");
const ops = @import("../tensor/ops.zig");
const SAM3Config = @import("config.zig").SAM3Config;
const WeightStore = @import("../weights/weight_loader.zig").WeightStore;
const ImageEncoder = @import("image_encoder.zig").ImageEncoder;
const PromptEncoder = @import("prompt_encoder.zig").PromptEncoder;
const Prompt = @import("prompt_encoder.zig").Prompt;
const Detector = @import("detector.zig").Detector;
const Detection = @import("detector.zig").Detection;
const MaskDecoder = @import("mask_decoder.zig").MaskDecoder;
const MemoryEncoder = @import("memory.zig").MemoryEncoder;
const MemoryBank = @import("memory.zig").MemoryBank;
const MemoryAttention = @import("memory.zig").MemoryAttention;

pub const SegmentationResult = struct {
    masks: Tensor,              // [1, num_masks, H, W]
    iou_scores: Tensor,         // [1, num_masks]
    presence_score: f32,
    is_present: bool,
    best_mask_idx: usize,

    pub fn deinit(self: *SegmentationResult) void {
        self.masks.deinit();
        self.iou_scores.deinit();
    }
};

pub const SAM3 = struct {
    config: SAM3Config,
    allocator: std.mem.Allocator,
    weights: WeightStore,

    image_encoder: ImageEncoder,
    prompt_encoder: PromptEncoder,
    detector: Detector,
    mask_decoder: MaskDecoder,
    memory_encoder: MemoryEncoder,
    memory_bank: MemoryBank,
    memory_attention: MemoryAttention,

    pub fn init(allocator: std.mem.Allocator, config: SAM3Config) SAM3 {
        return SAM3{
            .config = config,
            .allocator = allocator,
            .weights = WeightStore.init(allocator),
            .image_encoder = ImageEncoder.init(allocator, config),
            .prompt_encoder = PromptEncoder.init(allocator, config),
            .detector = Detector.init(allocator, config),
            .mask_decoder = MaskDecoder.init(allocator, config),
            .memory_encoder = MemoryEncoder.init(allocator, config),
            .memory_bank = MemoryBank.init(allocator, config.memory_bank_size),
            .memory_attention = MemoryAttention.init(allocator, config),
        };
    }

    pub fn deinit(self: *SAM3) void {
        self.weights.deinit();
        self.memory_bank.deinit();
    }

    pub fn loadWeights(self: *SAM3, file_path: []const u8) !void {
        try self.weights.loadFromSafeTensors(file_path);
    }

    pub fn segmentImage(
        self: *SAM3,
        image: Tensor,
        prompt: Prompt,
        multimask_output: bool,
    ) !SegmentationResult {
        // Automatically resize image to model input image_size if needed
        var resized_img: Tensor = if (image.shape[2] != self.config.image_size or image.shape[3] != self.config.image_size)
            try ops.bilinearUpsample(self.allocator, image, self.config.image_size, self.config.image_size, false)
        else
            try image.clone(self.allocator);
        defer resized_img.deinit();

        // 1. Encode Image
        var enc_out = try self.image_encoder.forward(resized_img, &self.weights);
        defer enc_out.deinit();

        // 2. Encode Prompts
        var prompt_emb = try self.prompt_encoder.forward(prompt, &self.weights);
        defer prompt_emb.deinit();

        // 3. Evaluate Concept Detector & Presence Head if text prompt is present
        var presence_score: f32 = 1.0;
        var is_present = true;
        if (prompt.text != null and prompt_emb.presence_token != null) {
            var det_out = try self.detector.forward(
                enc_out.image_embeddings,
                prompt_emb.sparse_embeddings,
                prompt_emb.presence_token,
                &self.weights,
            );
            presence_score = det_out.presence_score;
            is_present = det_out.is_present;
            det_out.deinit(self.allocator);
        }

        // 4. Generate Positional Embeddings for image grid
        const grid_h = self.config.gridH();
        const grid_w = self.config.gridW();
        var img_pe = try ops.sinusoidalEmbedding2D(self.allocator, grid_h, grid_w, self.config.encoder_embed_dim, 10000.0, 1.0);
        defer img_pe.deinit();

        // 5. Decode Masks
        var dec_out = try self.mask_decoder.forward(
            enc_out.image_embeddings,
            img_pe,
            prompt_emb.sparse_embeddings,
            prompt_emb.dense_embeddings,
            enc_out.high_res_features,
            multimask_output,
            &self.weights,
        );

        // Find best mask based on IoU score
        var best_idx: usize = 0;
        var best_score: f32 = -1.0;
        const num_masks = dec_out.iou_scores.shape[1];
        for (0..num_masks) |i| {
            const score = dec_out.iou_scores.at2(0, i);
            if (score > best_score) {
                best_score = score;
                best_idx = i;
            }
        }

        return SegmentationResult{
            .masks = dec_out.masks,
            .iou_scores = dec_out.iou_scores,
            .presence_score = presence_score,
            .is_present = is_present,
            .best_mask_idx = best_idx,
        };
    }
};
