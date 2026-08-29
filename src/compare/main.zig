//! Segments one image twice -- preprocessing on the CPU, then on the GPU --
//! and reports how far the two runs diverge. Both runs share the same ONNX
//! Runtime sessions, so preprocessing is the only thing that differs.

const std = @import("std");
const sam3 = @import("sam3");
const build_options = @import("build_options");
const Io = std.Io;

const vision_encoder_path = terminate(build_options.vision_encoder_path);
const decoder_path = terminate(build_options.decoder_path);
const concept_vision_path = terminate(build_options.concept_vision_path);
const concept_text_path = terminate(build_options.concept_text_path);
const concept_decoder_path = terminate(build_options.concept_decoder_path);
const concept_tokenizer_path = terminate(build_options.concept_tokenizer_path);

const prompt = [_]sam3.Point{.{ .x = 0.5, .y = 0.5, .label = 1 }};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();
    const image_path = args.next() orelse build_options.example_path;

    const png = try Io.Dir.cwd().readFileAlloc(io, image_path, allocator, .limited(32 * 1024 * 1024));
    defer allocator.free(png);
    var img = try sam3.image.decode(allocator, png);
    defer img.deinit();

    const tokenizer_json = try Io.Dir.cwd().readFileAlloc(
        io,
        concept_tokenizer_path,
        allocator,
        .limited(8 * 1024 * 1024),
    );
    defer allocator.free(tokenizer_json);

    std.debug.print("\n=== CPU vs GPU preprocessing ===\n\n", .{});
    std.debug.print("  image:        {s} ({d}x{d})\n", .{ image_path, img.width, img.height });

    var model = try sam3.Model.open(allocator, io, .{
        .vision_encoder = vision_encoder_path,
        .decoder = decoder_path,
        .concept_vision_encoder = concept_vision_path,
        .concept_text_encoder = concept_text_path,
        .concept_decoder = concept_decoder_path,
        .concept_tokenizer_json = tokenizer_json,
    }, .{ .cuda = true, .gpu = true });
    defer model.deinit();

    const preprocessor = model.preprocessor orelse {
        std.debug.print("\n  No GPU preprocessing to compare against. Build with -Ddevice=gpu or -Dcuda.\n\n", .{});
        std.process.exit(1);
    };

    // The preprocessed tensor itself.
    const gpu_start = Io.Timestamp.now(io, .awake);
    const gpu_pixels = try model.preprocess(img);
    defer allocator.free(gpu_pixels);
    const gpu_seconds = secondsSince(io, gpu_start);

    model.preprocessor = null;
    const cpu_start = Io.Timestamp.now(io, .awake);
    const cpu_pixels = try model.preprocess(img);
    defer allocator.free(cpu_pixels);
    const cpu_seconds = secondsSince(io, cpu_start);

    std.debug.print("\n  preprocess:   CPU {d:.1} ms, GPU {d:.1} ms ({d:.1}x)\n", .{
        cpu_seconds * 1000,
        gpu_seconds * 1000,
        cpu_seconds / gpu_seconds,
    });
    report("  pixel tensor", cpu_pixels, gpu_pixels);

    // The masks that come out the far end of both graphs.
    const cpu_segment_start = Io.Timestamp.now(io, .awake);
    var cpu_masks = try segment(&model, img);
    const cpu_segment_seconds = secondsSince(io, cpu_segment_start);
    defer cpu_masks.deinit();

    model.preprocessor = preprocessor;
    const gpu_segment_start = Io.Timestamp.now(io, .awake);
    var gpu_masks = try segment(&model, img);
    const gpu_segment_seconds = secondsSince(io, gpu_segment_start);
    defer gpu_masks.deinit();

    std.debug.print("\n  masks:        CPU {d}, GPU {d}, both {d}x{d}\n", .{
        cpu_masks.count,
        gpu_masks.count,
        cpu_masks.width,
        cpu_masks.height,
    });
    std.debug.print("  segment:      CPU prep {d:.2} s, GPU prep {d:.2} s\n", .{
        cpu_segment_seconds,
        gpu_segment_seconds,
    });
    if (cpu_masks.count != gpu_masks.count or cpu_masks.logits.len != gpu_masks.logits.len) {
        std.debug.print("  ! the two runs disagree on how many masks there are\n\n", .{});
        std.process.exit(1);
    }

    report("  mask logits", cpu_masks.logits, gpu_masks.logits);
    report("  mask scores", cpu_masks.scores, gpu_masks.scores);

    // The logits are thresholded at zero to draw the mask, so a difference
    // only shows up on screen if it moves a pixel across that line.
    var flipped: usize = 0;
    for (cpu_masks.logits, gpu_masks.logits) |a, b| {
        if ((a > 0) != (b > 0)) flipped += 1;
    }
    const percent = 100 * @as(f64, @floatFromInt(flipped)) / @as(f64, @floatFromInt(cpu_masks.logits.len));
    std.debug.print("  mask pixels:  {d} of {d} land on the other side of the threshold ({d:.4}%)\n", .{
        flipped,
        cpu_masks.logits.len,
        percent,
    });
    std.debug.print("  best mask:    CPU {d}, GPU {d}\n\n", .{ cpu_masks.best(), gpu_masks.best() });
}

fn segment(model: *sam3.Model, img: sam3.ImageRGB) !sam3.Masks {
    var embedding = try model.encode(img);
    defer embedding.deinit();
    return model.decode(embedding, &prompt);
}

fn report(name: []const u8, cpu: []const f32, gpu: []const f32) void {
    var max_delta: f32 = 0;
    var total: f64 = 0;
    var identical: usize = 0;
    for (cpu, gpu) |a, b| {
        const delta = @abs(a - b);
        max_delta = @max(max_delta, delta);
        total += delta;
        if (a == b) identical += 1;
    }
    std.debug.print("{s}:  max |delta| {e:.3}, mean |delta| {e:.3}, {d} of {d} bit-identical\n", .{
        name,
        max_delta,
        total / @as(f64, @floatFromInt(cpu.len)),
        identical,
        cpu.len,
    });
}

fn secondsSince(io: Io, started: Io.Timestamp) f64 {
    const elapsed = started.durationTo(Io.Timestamp.now(io, .awake));
    return @as(f64, @floatFromInt(elapsed.nanoseconds)) / 1e9;
}

fn terminate(comptime path: []const u8) [:0]const u8 {
    return (path ++ "\x00")[0..path.len :0];
}
