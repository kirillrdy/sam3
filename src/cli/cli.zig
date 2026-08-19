const std = @import("std");
const Tensor = @import("../tensor/tensor.zig").Tensor;
const SAM3 = @import("../models/sam3.zig").SAM3;
const SAM3Config = @import("../models/config.zig").SAM3Config;
const Prompt = @import("../models/prompt_encoder.zig").Prompt;
const Point = @import("../models/prompt_encoder.zig").Point;
const Box = @import("../models/prompt_encoder.zig").Box;
const SAM3VideoPredictor = @import("../video/tracker.zig").SAM3VideoPredictor;
const ImageRGB = @import("../io/image.zig").ImageRGB;
const RGB = @import("../io/image.zig").RGB;
const overlayMask = @import("../io/visualization.zig").overlayMask;
const drawBoundingBox = @import("../io/visualization.zig").drawBoundingBox;
const drawPointMarker = @import("../io/visualization.zig").drawPointMarker;
const getColorForId = @import("../io/visualization.zig").getColorForId;
const VideoSequence = @import("../io/video_io.zig").VideoSequence;

inline fn getMonotonicNs() u64 {
    var ts: std.os.linux.timespec = undefined;
    _ = std.os.linux.clock_gettime(std.os.linux.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

pub fn runDemo(allocator: std.mem.Allocator) !void {
    std.debug.print("\n=== SAM 3 (Segment Anything Model 3 in Pure Zig) Demo ===\n\n", .{});

    const config = SAM3Config.sam3_tiny();
    var sam3 = SAM3.init(allocator, config);
    defer sam3.deinit();

    // 1. Image Concept & Point Segmentation Demo
    std.debug.print("[1/3] Running Image Segmentation Demo (Image Size: {d}x{d})...\n", .{ config.image_size, config.image_size });

    var input_img = try ImageRGB.init(allocator, config.image_size, config.image_size);
    defer input_img.deinit();

    // Draw a prominent synthetic foreground object (circle in center)
    const cx: isize = @intCast(config.image_size / 2);
    const cy: isize = @intCast(config.image_size / 2);
    const r: isize = @intCast(config.image_size / 4);

    var dy = -r;
    while (dy <= r) : (dy += 1) {
        var dx = -r;
        while (dx <= r) : (dx += 1) {
            if (dx * dx + dy * dy <= r * r) {
                const px = cx + dx;
                const py = cy + dy;
                if (px >= 0 and py >= 0 and px < @as(isize, @intCast(config.image_size)) and py < @as(isize, @intCast(config.image_size))) {
                    input_img.setPixel(@intCast(px), @intCast(py), RGB{ .r = 240, .g = 80, .b = 80 });
                }
            }
        }
    }

    var img_tensor = try input_img.toTensor(allocator);
    defer img_tensor.deinit();

    // Prompt 1: Point prompt in the center
    const pts = [_]Point{
        .{ .x = 0.5, .y = 0.5, .label = 1 },
    };
    const prompt = Prompt{
        .points = &pts,
        .text = "red circle",
    };

    var seg_res = try sam3.segmentImage(img_tensor, prompt, true);
    defer seg_res.deinit();

    const best_mask_idx = seg_res.best_mask_idx;
    const iou = seg_res.iou_scores.at2(0, best_mask_idx);
    std.debug.print("  -> Mask generated successfully!\n", .{});
    std.debug.print("  -> Best Mask Index: {d}, Predicted IoU: {d:.4}\n", .{ best_mask_idx, iou });
    std.debug.print("  -> Concept Presence Score: {d:.4} (is_present: {any})\n", .{ seg_res.presence_score, seg_res.is_present });

    // Slice best mask
    var best_mask = try seg_res.masks.slice(allocator, 1, best_mask_idx, best_mask_idx + 1);
    defer best_mask.deinit();

    // Overlay mask and markers
    var vis_img = try ImageRGB.init(allocator, config.image_size, config.image_size);
    defer vis_img.deinit();
    @memcpy(vis_img.data, input_img.data);

    overlayMask(&vis_img, best_mask, RGB{ .r = 0, .g = 220, .b = 100 }, 0.5);
    drawPointMarker(&vis_img, pts[0], 4);

    try vis_img.savePPM("demo_image_segmented.ppm");
    try vis_img.saveBMP("demo_image_segmented.bmp");
    std.debug.print("  -> Saved output images to 'demo_image_segmented.ppm' and 'demo_image_segmented.bmp'\n\n", .{});

    // 2. Video Tracking Demo
    std.debug.print("[2/3] Running SAM 3 Video Tracking Demo (5 frames)...\n", .{});
    var video = try VideoSequence.generateSyntheticVideo(allocator, 5, config.image_size, config.image_size);
    defer video.deinit();

    var predictor = SAM3VideoPredictor.init(allocator, &sam3);
    defer predictor.deinit();

    try predictor.addObject(0, "yellow object");
    try predictor.addObject(1, "cyan vehicle");

    // Add initial prompt on Frame 0
    var f0_tensor = try video.frames[0].image.toTensor(allocator);
    defer f0_tensor.deinit();

    const f0_pts = [_]Point{
        .{ .x = 30.0 / @as(f32, @floatFromInt(config.image_size)), .y = 0.33, .label = 1 },
    };
    try predictor.addPrompt(0, 0, Prompt{ .points = &f0_pts, .text = "yellow object" }, f0_tensor);

    // Propagate across frames
    for (video.frames, 0..) |frame, f_idx| {
        var f_tensor = try frame.image.toTensor(allocator);
        defer f_tensor.deinit();

        var track_res = try predictor.trackFrame(f_idx, f_tensor);
        defer track_res.deinit(allocator);

        std.debug.print("  -> Frame {d}: Tracked {d} objects\n", .{ f_idx, track_res.objects.len });
        for (track_res.objects) |obj| {
            std.debug.print("     * Obj {d}: IoU={d:.3}, Presence={d:.3}, Box=[{d:.2}, {d:.2}, {d:.2}, {d:.2}]\n", .{
                obj.obj_id,
                obj.iou_score,
                obj.presence_score,
                obj.box.x1,
                obj.box.y1,
                obj.box.x2,
                obj.box.y2,
            });
        }

        // Render visualization
        var frame_vis = try ImageRGB.init(allocator, config.image_size, config.image_size);
        defer frame_vis.deinit();
        @memcpy(frame_vis.data, frame.image.data);

        for (track_res.objects) |obj| {
            const col = getColorForId(obj.obj_id);
            overlayMask(&frame_vis, obj.mask, col, 0.45);
            drawBoundingBox(&frame_vis, obj.box, col, 2);
        }

        var out_name: [64]u8 = undefined;
        const fname = try std.fmt.bufPrint(&out_name, "demo_video_frame_{d}.ppm", .{f_idx});
        try frame_vis.savePPM(fname);
    }
    std.debug.print("  -> Saved video frames to 'demo_video_frame_*.ppm'\n\n", .{});

    // 3. Presence Head Concept Filtering Demo
    std.debug.print("[3/3] Testing SAM 3 Presence Head Discrimination...\n", .{});
    const prompt_concept_present = Prompt{ .text = "yellow school bus" };
    var res_present = try sam3.segmentImage(img_tensor, prompt_concept_present, false);
    defer res_present.deinit();

    std.debug.print("  -> Query 'yellow school bus' Presence Score: {d:.4}\n", .{res_present.presence_score});
    std.debug.print("\n=== Demo Complete! Pure Zig SAM 3 is fully functional. ===\n\n", .{});
}

pub fn runBenchmark(allocator: std.mem.Allocator) !void {
    std.debug.print("\n=== SAM 3 Pure Zig Performance Benchmark ===\n\n", .{});

    // 1. Matrix Multiplication SIMD Benchmark
    const m: usize = 256;
    const k: usize = 256;
    const n: usize = 256;
    const shape_a = [_]usize{ m, k };
    const shape_b = [_]usize{ k, n };

    var a = try Tensor.initRandom(allocator, &shape_a, 1, -1.0, 1.0);
    defer a.deinit();
    var b = try Tensor.initRandom(allocator, &shape_b, 2, -1.0, 1.0);
    defer b.deinit();

    const iters: usize = 100;
    const t0 = getMonotonicNs();

    for (0..iters) |_| {
        var c = try @import("../tensor/math.zig").matmul2D(allocator, a, b);
        c.deinit();
    }
    const t1 = getMonotonicNs();
    const elapsed_ns = t1 - t0;
    const gflops = (2.0 * @as(f64, @floatFromInt(m * k * n)) * @as(f64, @floatFromInt(iters))) / (@as(f64, @floatFromInt(elapsed_ns)) * 1e-9) / 1e9;

    std.debug.print("[1] GEMM (256x256x256) x {d} iterations: {d:.2} ms ({d:.2} GFLOPS)\n", .{
        iters,
        @as(f64, @floatFromInt(elapsed_ns)) / 1e6,
        gflops,
    });

    // 2. Full SAM3 Tiny End-to-End Image Pass
    const config = SAM3Config.sam3_tiny();
    var sam3 = SAM3.init(allocator, config);
    defer sam3.deinit();

    const img_shape = [_]usize{ 1, 3, config.image_size, config.image_size };
    var img = try Tensor.initRandom(allocator, &img_shape, 3, 0.0, 1.0);
    defer img.deinit();

    const pts = [_]Point{ .{ .x = 0.5, .y = 0.5, .label = 1 } };
    const prompt = Prompt{ .points = &pts, .text = "car" };

    const t2 = getMonotonicNs();
    const img_iters: usize = 10;
    for (0..img_iters) |_| {
        var res = try sam3.segmentImage(img, prompt, false);
        res.deinit();
    }
    const t3 = getMonotonicNs();
    const img_elapsed_ns = t3 - t2;
    const fps = @as(f64, @floatFromInt(img_iters)) / (@as(f64, @floatFromInt(img_elapsed_ns)) * 1e-9);

    std.debug.print("[2] Full SAM3 Image Segmentation ({d}x{d}): {d:.2} ms/pass ({d:.2} FPS)\n", .{
        config.image_size,
        config.image_size,
        @as(f64, @floatFromInt(img_elapsed_ns)) / 1e6 / @as(f64, @floatFromInt(img_iters)),
        fps,
    });

    std.debug.print("\n=== Benchmark Complete ===\n\n", .{});
}
