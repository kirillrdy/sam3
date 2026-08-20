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
const image_io = @import("../io/image.zig");
const overlayMask = @import("../io/visualization.zig").overlayMask;
const drawBoundingBox = @import("../io/visualization.zig").drawBoundingBox;
const drawPointMarker = @import("../io/visualization.zig").drawPointMarker;
const getColorForId = @import("../io/visualization.zig").getColorForId;
const VideoSequence = @import("../io/video_io.zig").VideoSequence;
const safetensors = @import("../weights/safetensors.zig");
const WeightStore = @import("../weights/weight_loader.zig").WeightStore;
const VisionEncoder = @import("../models/vision_encoder.zig").VisionEncoder;
const VisionConfig = @import("../models/vision_encoder.zig").VisionConfig;
const sam3_tracker = @import("../models/sam3_tracker.zig");
const ops = @import("../tensor/ops.zig");

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

/// Returns the module path of a tensor: its first two dot-separated components
/// (e.g. "detector_model.vision_encoder"), which is how the SAM 3 checkpoint
/// groups its submodules.
fn moduleOf(name: []const u8) []const u8 {
    var dots: usize = 0;
    for (name, 0..) |ch, i| {
        if (ch == '.') {
            dots += 1;
            if (dots == 2) return name[0..i];
        }
    }
    return name;
}

const ModuleStat = struct {
    name: []const u8,
    tensors: usize = 0,
    params: usize = 0,
};

/// Prints the layout of a SafeTensors checkpoint (module breakdown, dtypes,
/// parameter counts) by reading only its header, so a multi-gigabyte file such
/// as the full SAM 3 release is inspected without loading any weights.
pub fn runInspectWeights(
    allocator: std.mem.Allocator,
    file_path: []const u8,
    filter: ?[]const u8,
    verify: bool,
) !void {
    var header = try safetensors.readHeader(allocator, file_path);
    defer header.deinit();

    const total_params = header.totalElements();

    std.debug.print("\n=== SafeTensors Checkpoint: {s} ===\n\n", .{file_path});
    std.debug.print("  File size:   {d:.2} GiB ({d} bytes)\n", .{
        @as(f64, @floatFromInt(header.file_size)) / (1024.0 * 1024.0 * 1024.0),
        header.file_size,
    });
    std.debug.print("  Tensors:     {d}\n", .{header.entries.len});
    std.debug.print("  Parameters:  {d} ({d:.1}M)\n", .{
        total_params,
        @as(f64, @floatFromInt(total_params)) / 1.0e6,
    });

    // Dtype histogram
    var dtype_counts = [_]usize{0} ** 5;
    for (header.entries) |entry| {
        dtype_counts[@intFromEnum(entry.dtype)] += 1;
    }
    std.debug.print("  Dtypes:     ", .{});
    for (dtype_counts, 0..) |count, i| {
        if (count == 0) continue;
        std.debug.print(" {t} x{d}", .{ @as(safetensors.DType, @enumFromInt(i)), count });
    }
    std.debug.print("\n\n", .{});

    // Module breakdown
    var modules: std.ArrayList(ModuleStat) = .empty;
    defer modules.deinit(allocator);

    for (header.entries) |entry| {
        const module = moduleOf(entry.name);
        const stat = for (modules.items) |*existing| {
            if (std.mem.eql(u8, existing.name, module)) break existing;
        } else blk: {
            try modules.append(allocator, ModuleStat{ .name = module });
            break :blk &modules.items[modules.items.len - 1];
        };
        stat.tensors += 1;
        stat.params += entry.numElements();
    }

    std.debug.print("Modules ({d}):\n", .{modules.items.len});
    for (modules.items) |stat| {
        std.debug.print("  {s: <56} {d: >5} tensors  {d: >9.2}M params\n", .{
            stat.name,
            stat.tensors,
            @as(f64, @floatFromInt(stat.params)) / 1.0e6,
        });
    }

    if (filter) |pattern| {
        std.debug.print("\nTensors matching '{s}':\n", .{pattern});
        var matches: usize = 0;
        for (header.entries) |entry| {
            if (std.mem.indexOf(u8, entry.name, pattern) == null) continue;
            matches += 1;
            std.debug.print("  {s: <76} {t: <5} [", .{ entry.name, entry.dtype });
            for (entry.shape, 0..) |dim, i| {
                if (i > 0) std.debug.print(", ", .{});
                std.debug.print("{d}", .{dim});
            }
            std.debug.print("]\n", .{});
        }
        std.debug.print("  ({d} matching tensors)\n", .{matches});
    }

    if (verify) {
        std.debug.print("\n", .{});
        const failures = verifyFullLayout(&header);
        if (failures > 0) return error.CheckpointLayoutMismatch;
    }

    std.debug.print("\n", .{});
}


/// Cross-checks `SAM3Config.sam3_full()` against the tensor shapes actually
/// present in a checkpoint, so the declared architecture cannot silently drift
/// away from the weights it claims to describe.
const LayoutVerifier = struct {
    header: *const safetensors.Header,
    checks: usize = 0,
    failures: usize = 0,

    fn find(self: LayoutVerifier, name: []const u8) ?safetensors.TensorEntry {
        for (self.header.entries) |entry| {
            if (std.mem.eql(u8, entry.name, name)) return entry;
        }
        return null;
    }

    /// Number of `<prefix><n>.` submodules present, i.e. the depth of a stack.
    fn layerCount(self: LayoutVerifier, prefix: []const u8) usize {
        var count: usize = 0;
        for (self.header.entries) |entry| {
            if (!std.mem.startsWith(u8, entry.name, prefix)) continue;
            const rest = entry.name[prefix.len..];
            const dot = std.mem.indexOfScalar(u8, rest, '.') orelse continue;
            const index = std.fmt.parseInt(usize, rest[0..dot], 10) catch continue;
            count = @max(count, index + 1);
        }
        return count;
    }

    fn report(self: *LayoutVerifier, label: []const u8, ok: bool) void {
        self.checks += 1;
        if (!ok) self.failures += 1;
        std.debug.print("  [{s}] {s}\n", .{ if (ok) "ok " else "BAD", label });
    }

    fn expectShape(self: *LayoutVerifier, label: []const u8, name: []const u8, expected: []const usize) void {
        const entry = self.find(name) orelse {
            self.checks += 1;
            self.failures += 1;
            std.debug.print("  [BAD] {s}: tensor '{s}' is missing\n", .{ label, name });
            return;
        };
        const ok = std.mem.eql(usize, entry.shape, expected);
        self.report(label, ok);
        if (!ok) {
            std.debug.print("        expected {any}, checkpoint has {any}\n", .{ expected, entry.shape });
        }
    }

    fn expectCount(self: *LayoutVerifier, label: []const u8, prefix: []const u8, expected: usize) void {
        const actual = self.layerCount(prefix);
        const ok = actual == expected;
        self.report(label, ok);
        if (!ok) {
            std.debug.print("        expected {d}, checkpoint has {d}\n", .{ expected, actual });
        }
    }
};

/// Verifies that the `sam3_full` preset describes the checkpoint on disk.
/// Returns the number of mismatches.
pub fn verifyFullLayout(header: *const safetensors.Header) usize {
    const cfg = SAM3Config.sam3_full();
    var v = LayoutVerifier{ .header = header };

    std.debug.print("Checking checkpoint against SAM3Config.sam3_full():\n", .{});

    const patch_shape = [_]usize{ cfg.encoder_embed_dim, 3, cfg.patch_size, cfg.patch_size };
    v.expectShape(
        "vision patch embedding (embed_dim, patch_size)",
        "detector_model.vision_encoder.backbone.embeddings.patch_embeddings.projection.weight",
        &patch_shape,
    );

    const mlp_hidden: usize = @intFromFloat(@round(@as(f32, @floatFromInt(cfg.encoder_embed_dim)) * cfg.encoder_mlp_ratio));
    const mlp_shape = [_]usize{ mlp_hidden, cfg.encoder_embed_dim };
    v.expectShape(
        "vision MLP ratio",
        "detector_model.vision_encoder.backbone.layers.0.mlp.fc1.weight",
        &mlp_shape,
    );

    v.expectCount(
        "vision backbone depth",
        "detector_model.vision_encoder.backbone.layers.",
        cfg.encoder_depth,
    );

    const token_embed_shape = [_]usize{ cfg.text_vocab_size, cfg.encoder_embed_dim };
    v.expectShape(
        "text vocabulary",
        "detector_model.text_encoder.text_model.embeddings.token_embedding.weight",
        &token_embed_shape,
    );

    const text_pos_shape = [_]usize{ cfg.text_max_seq_len, cfg.encoder_embed_dim };
    v.expectShape(
        "text context length",
        "detector_model.text_encoder.text_model.embeddings.position_embedding.weight",
        &text_pos_shape,
    );

    v.expectCount(
        "text encoder depth",
        "detector_model.text_encoder.text_model.encoder.layers.",
        cfg.text_num_layers,
    );

    const query_shape = [_]usize{ cfg.num_object_queries, cfg.prompt_embed_dim };
    v.expectShape(
        "DETR object queries",
        "detector_model.detr_decoder.query_embed.weight",
        &query_shape,
    );

    v.expectCount(
        "DETR decoder depth",
        "detector_model.detr_decoder.layers.",
        cfg.mask_decoder_depth,
    );

    const presence_shape = [_]usize{ 1, cfg.presence_head_dim };
    v.expectShape(
        "presence token",
        "detector_model.detr_decoder.presence_token.weight",
        &presence_shape,
    );

    std.debug.print("\n  {d}/{d} checks passed.\n", .{ v.checks - v.failures, v.checks });
    return v.failures;
}

/// Runs Meta's real SAM 3 vision encoder over an image, straight off the
/// released checkpoint, and reports what the backbone and FPN neck produced.
pub fn runVisionEncoder(
    allocator: std.mem.Allocator,
    image_path: []const u8,
    weights_path: []const u8,
    image_size: usize,
) !void {
    var cfg = VisionConfig{};
    cfg.image_size = image_size;
    if (image_size % cfg.patch_size != 0) return error.ImageSizeNotMultipleOfPatch;

    std.debug.print("\n=== SAM 3 Vision Encoder ({s}) ===\n\n", .{weights_path});

    var img = try image_io.load(allocator, image_path);
    defer img.deinit();
    std.debug.print("  Image:       {s} ({d}x{d})\n", .{ image_path, img.width, img.height });
    std.debug.print("  Model input: {d}x{d} -> {d}x{d} tokens, d={d}, {d} layers\n\n", .{
        cfg.image_size, cfg.image_size, cfg.grid(), cfg.grid(), cfg.hidden_size, cfg.num_layers,
    });

    var weights = WeightStore.init(allocator);
    defer weights.deinit();

    const t_load = getMonotonicNs();
    try weights.loadFromSafeTensors(weights_path);
    std.debug.print("  Loaded {d} checkpoint tensors in {d:.1} s\n", .{
        weights.checkpoint_tensors,
        @as(f64, @floatFromInt(getMonotonicNs() - t_load)) / 1e9,
    });

    const encoder = VisionEncoder.init(allocator, cfg);

    var pixels = try encoder.preprocess(img);
    defer pixels.deinit();

    const t_fwd = getMonotonicNs();
    var out = try encoder.forward(pixels, &weights);
    defer out.deinit();
    const fwd_s = @as(f64, @floatFromInt(getMonotonicNs() - t_fwd)) / 1e9;

    std.debug.print("  Forward pass: {d:.1} s\n\n", .{fwd_s});

    std.debug.print("  Backbone tokens: [{d}, {d}, {d}]  {s}\n", .{
        out.last_hidden_state.shape[0],
        out.last_hidden_state.shape[1],
        out.last_hidden_state.shape[2],
        summariseStats(out.last_hidden_state),
    });

    for (out.fpn_features, 0..) |f, i| {
        std.debug.print("  FPN level {d} ({d:.1}x): [{d}, {d}, {d}, {d}]  {s}\n", .{
            i,
            cfg.scale_factors[i],
            f.shape[0],
            f.shape[1],
            f.shape[2],
            f.shape[3],
            summariseStats(f),
        });
    }

    const cov = weights.coverage();
    std.debug.print("\n  Checkpoint coverage: {d}/{d} weights resolved ({d:.1}%), {d} randomly initialised\n\n", .{
        cov.resolved,
        cov.requested(),
        cov.fraction() * 100.0,
        cov.synthesised,
    });

    if (cov.synthesised > 0) return error.IncompleteCheckpointCoverage;
}

var stats_buf: [128]u8 = undefined;

/// mean / std / range of a tensor, for eyeballing that a stage produced signal
/// rather than zeros or NaNs.
fn summariseStats(t: Tensor) []const u8 {
    var sum: f64 = 0.0;
    var min_v: f32 = std.math.inf(f32);
    var max_v: f32 = -std.math.inf(f32);
    for (t.data) |v| {
        sum += v;
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
    }
    const n = @as(f64, @floatFromInt(t.data.len));
    const mean = sum / n;

    var var_sum: f64 = 0.0;
    for (t.data) |v| {
        const dv = @as(f64, v) - mean;
        var_sum += dv * dv;
    }
    const stddev = @sqrt(var_sum / n);

    return std.fmt.bufPrint(&stats_buf, "mean {d: >8.4}  std {d: >7.4}  range [{d: >8.3}, {d: >7.3}]", .{
        mean, stddev, min_v, max_v,
    }) catch "<stats unavailable>";
}

/// Point-prompted segmentation through the real SAM 3 tracker branch: shared
/// vision backbone -> `tracker_neck` FPN -> prompt encoder -> two-way mask
/// decoder, then the winning mask is overlaid on the input image.
pub fn runSegment(
    allocator: std.mem.Allocator,
    image_path: []const u8,
    weights_path: []const u8,
    output_path: []const u8,
    points_in: []const Point,
    image_size: usize,
    multimask: bool,
) !void {
    var vision_cfg = VisionConfig{};
    vision_cfg.image_size = image_size;
    vision_cfg.neck_prefix = "tracker_neck";
    if (image_size % vision_cfg.patch_size != 0) return error.ImageSizeNotMultipleOfPatch;

    var tracker_cfg = sam3_tracker.TrackerConfig{};
    tracker_cfg.image_size = image_size;

    std.debug.print("\n=== SAM 3 Point-Prompted Segmentation ===\n\n", .{});

    var img = try image_io.load(allocator, image_path);
    defer img.deinit();
    std.debug.print("  Image:       {s} ({d}x{d})\n", .{ image_path, img.width, img.height });
    std.debug.print("  Model input: {d}x{d} -> {d}x{d} tokens\n", .{
        image_size, image_size, vision_cfg.grid(), vision_cfg.grid(),
    });

    // Prompt points arrive normalised; the model wants pixels of its own input.
    var prompt_points: std.ArrayList(sam3_tracker.PromptPoint) = .empty;
    defer prompt_points.deinit(allocator);

    const size_f: f32 = @floatFromInt(image_size);
    for (points_in) |p| {
        try prompt_points.append(allocator, .{ .x = p.x * size_f, .y = p.y * size_f, .label = p.label });
        std.debug.print("  Prompt:      ({d:.3}, {d:.3}) label {d}\n", .{ p.x, p.y, p.label });
    }
    // The reference implementation pads with a single "not a point" when no box
    // prompt is present, and the decoder is trained with it there.
    try prompt_points.append(allocator, .{ .x = 0.0, .y = 0.0, .label = -1 });

    var weights = WeightStore.init(allocator);
    defer weights.deinit();

    const t_load = getMonotonicNs();
    try weights.loadFromSafeTensors(weights_path);
    std.debug.print("\n  Loaded {d} checkpoint tensors in {d:.1} s\n", .{
        weights.checkpoint_tensors,
        @as(f64, @floatFromInt(getMonotonicNs() - t_load)) / 1e9,
    });

    const encoder = VisionEncoder.init(allocator, vision_cfg);

    var pixels = try encoder.preprocess(img);
    defer pixels.deinit();

    const t_vision = getMonotonicNs();
    var vision_out = try encoder.forward(pixels, &weights);
    defer vision_out.deinit();
    std.debug.print("  Vision encoder: {d:.1} s\n", .{
        @as(f64, @floatFromInt(getMonotonicNs() - t_vision)) / 1e9,
    });

    const t_dec = getMonotonicNs();
    var result = try sam3_tracker.segmentWithPoints(
        allocator,
        tracker_cfg,
        vision_out.fpn_features[0..3],
        prompt_points.items,
        multimask,
        &weights,
    );
    defer result.deinit();
    std.debug.print("  Mask decoder:   {d:.1} s\n\n", .{
        @as(f64, @floatFromInt(getMonotonicNs() - t_dec)) / 1e9,
    });

    std.debug.print("  Object score logit: {d:.4}\n", .{result.object_score});

    // Mask logits live at a quarter of the model input; resize to the original
    // image and threshold at zero, as `post_process_masks` does. Every mask of a
    // multimask prediction is written out — they are the part / subpart / whole
    // hypotheses for the same click.
    var path_buf: [512]u8 = undefined;
    for (0..result.masks.shape[0]) |i| {
        var mask = try result.maskAt(i, allocator);
        defer mask.deinit();

        var full_mask = try ops.bilinearUpsample(allocator, mask, img.height, img.width, false);
        defer full_mask.deinit();

        var covered: usize = 0;
        for (full_mask.data) |v| {
            if (v > 0.0) covered += 1;
        }
        std.debug.print("  Mask {d}: predicted IoU {d:.4}, covers {d: >5.1}% of the image{s}\n", .{
            i,
            result.iou_scores[i],
            100.0 * @as(f64, @floatFromInt(covered)) / @as(f64, @floatFromInt(full_mask.data.len)),
            if (i == result.best_index) "  <- highest IoU" else "",
        });

        var frame = try ImageRGB.init(allocator, img.width, img.height);
        defer frame.deinit();
        @memcpy(frame.data, img.data);

        overlayMask(&frame, full_mask, RGB{ .r = 0, .g = 220, .b = 100 }, 0.5);
        for (points_in) |p| {
            drawPointMarker(&frame, p, 7);
        }

        const path = if (result.masks.shape[0] == 1)
            output_path
        else
            try insertSuffix(&path_buf, output_path, i);

        if (std.mem.endsWith(u8, path, ".ppm")) {
            try frame.savePPM(path);
        } else {
            try frame.saveBMP(path);
        }
        std.debug.print("           -> {s}\n", .{path});
    }

    const cov = weights.coverage();
    std.debug.print("\n  Checkpoint coverage: {d}/{d} weights resolved ({d:.1}%), {d} randomly initialised\n", .{
        cov.resolved,
        cov.requested(),
        cov.fraction() * 100.0,
        cov.synthesised,
    });
    std.debug.print("  -> Saved '{s}'\n\n", .{output_path});

    if (cov.synthesised > 0) return error.IncompleteCheckpointCoverage;
}

/// "out.bmp" + 1 -> "out_1.bmp", so a multimask prediction writes one file per
/// mask instead of overwriting itself.
fn insertSuffix(buf: []u8, path: []const u8, index: usize) ![]const u8 {
    const dot = std.mem.lastIndexOfScalar(u8, path, '.') orelse path.len;
    return std.fmt.bufPrint(buf, "{s}_{d}{s}", .{ path[0..dot], index, path[dot..] });
}

test "insertSuffix places the index before the extension" {
    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings("out_2.bmp", try insertSuffix(&buf, "out.bmp", 2));
    try std.testing.expectEqualStrings("dir/x_0.ppm", try insertSuffix(&buf, "dir/x.ppm", 0));
    try std.testing.expectEqualStrings("noext_1", try insertSuffix(&buf, "noext", 1));
}
