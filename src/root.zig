const std = @import("std");

pub const tensor = @import("tensor/tensor.zig");
pub const math = @import("tensor/math.zig");
pub const ops = @import("tensor/ops.zig");
pub const safetensors = @import("weights/safetensors.zig");
pub const weight_loader = @import("weights/weight_loader.zig");
pub const config = @import("models/config.zig");
pub const image_encoder = @import("models/image_encoder.zig");
pub const prompt_encoder = @import("models/prompt_encoder.zig");
pub const detector = @import("models/detector.zig");
pub const mask_decoder = @import("models/mask_decoder.zig");
pub const memory = @import("models/memory.zig");
pub const sam3 = @import("models/sam3.zig");
pub const tracker = @import("video/tracker.zig");
pub const image = @import("io/image.zig");
pub const visualization = @import("io/visualization.zig");
pub const video_io = @import("io/video_io.zig");
pub const cli = @import("cli/cli.zig");

// Re-exports
pub const Tensor = tensor.Tensor;
pub const SAM3 = sam3.SAM3;
pub const SAM3Config = config.SAM3Config;
pub const Prompt = prompt_encoder.Prompt;
pub const Point = prompt_encoder.Point;
pub const Box = prompt_encoder.Box;
pub const SAM3VideoPredictor = tracker.SAM3VideoPredictor;
pub const SafeTensors = safetensors.SafeTensors;
pub const WeightStore = weight_loader.WeightStore;
pub const ImageRGB = image.ImageRGB;
pub const RGB = image.RGB;

test "Full SAM3 End-to-End Image and Video Test" {
    const allocator = std.testing.allocator;

    const cfg = SAM3Config.sam3_tiny();
    var model = SAM3.init(allocator, cfg);
    defer model.deinit();

    // 1. Test Image Forward Pass with Points and Concept Text
    const img_shape = [_]usize{ 1, 3, cfg.image_size, cfg.image_size };
    var img = try Tensor.initConstant(allocator, &img_shape, 0.5);
    defer img.deinit();

    const pts = [_]Point{
        .{ .x = 0.5, .y = 0.5, .label = 1 },
    };
    const prompt = Prompt{
        .points = &pts,
        .text = "circle",
    };

    var res = try model.segmentImage(img, prompt, true);
    defer res.deinit();

    try std.testing.expectEqual(@as(usize, 1), res.masks.shape[0]);
    try std.testing.expectEqual(cfg.mask_decoder_num_multimask_outputs, res.masks.shape[1]);
    try std.testing.expectEqual(cfg.gridH() * 4, res.masks.shape[2]);
    try std.testing.expectEqual(cfg.gridW() * 4, res.masks.shape[3]);
    try std.testing.expect(res.presence_score >= 0.0 and res.presence_score <= 1.0);

    // 2. Test Video Predictor Multi-Frame Tracking
    var predictor = SAM3VideoPredictor.init(allocator, &model);
    defer predictor.deinit();

    try predictor.addObject(0, "moving object");

    // Add prompt at frame 0
    try predictor.addPrompt(0, 0, prompt, img);

    // Track frame 1
    var frame1_res = try predictor.trackFrame(1, img);
    defer frame1_res.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1), frame1_res.objects.len);
    try std.testing.expectEqual(@as(usize, 0), frame1_res.objects[0].obj_id);
}
