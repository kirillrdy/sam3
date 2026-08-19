const std = @import("std");
const sam3 = @import("sam3");

fn printUsage() void {
    std.debug.print(
        \\SAM 3 - Segment Anything Model 3 (Pure Zig Implementation)
        \\
        \\Usage:
        \\  sam3 demo                      Run synthetic image & video segmentation demo
        \\  sam3 benchmark                 Run SIMD GEMM and inference performance benchmarks
        \\  sam3 image [options]           Segment an image using points, boxes, or text concepts
        \\  sam3 video [options]           Track objects across video frames
        \\
        \\Image Options:
        \\  --image <path>                 Input image file (.ppm or .bmp)
        \\  --text <concept>               Promptable Concept Segmentation text (e.g. "yellow bus")
        \\  --point <x,y,label>            Point prompt (e.g. 0.5,0.5,1) [label 1=fg, 0=bg]
        \\  --box <x1,y1,x2,y2>            Bounding box prompt (e.g. 0.1,0.1,0.9,0.9)
        \\  --preset <tiny|base>           Model size preset (default: base)
        \\  --weights <path.safetensors>   Optional model weights in SafeTensors format
        \\  --output <path>                Output segmented image with mask overlay (.ppm or .bmp)
        \\
        \\Video Options:
        \\  --frames <dir>                 Directory containing video frames (frame_0000.ppm, etc.)
        \\  --text <concept>               Concept prompt to track across frames
        \\  --weights <path.safetensors>   Optional model weights
        \\  --output <dir>                 Directory to save output segmented video frames
        \\
    , .{});
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var args_it = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args_it.deinit();

    _ = args_it.skip(); // skip executable name

    const command = args_it.next() orelse {
        printUsage();
        return;
    };

    if (std.mem.eql(u8, command, "demo")) {
        try sam3.cli.runDemo(allocator);
    } else if (std.mem.eql(u8, command, "benchmark")) {
        try sam3.cli.runBenchmark(allocator);
    } else if (std.mem.eql(u8, command, "image")) {
        var image_path: ?[]const u8 = null;
        var text_prompt: ?[]const u8 = null;
        var point_str: ?[]const u8 = null;
        var output_path: ?[]const u8 = null;
        var weights_path: ?[]const u8 = null;
        var preset_str: []const u8 = "base";

        while (args_it.next()) |arg| {
            if (std.mem.eql(u8, arg, "--image")) {
                image_path = args_it.next();
            } else if (std.mem.eql(u8, arg, "--text")) {
                text_prompt = args_it.next();
            } else if (std.mem.eql(u8, arg, "--point")) {
                point_str = args_it.next();
            } else if (std.mem.eql(u8, arg, "--output")) {
                output_path = args_it.next();
            } else if (std.mem.eql(u8, arg, "--weights")) {
                weights_path = args_it.next();
            } else if (std.mem.eql(u8, arg, "--preset")) {
                preset_str = args_it.next() orelse "base";
            }
        }

        if (image_path == null or output_path == null) {
            std.debug.print("Error: --image and --output are required for image subcommand.\n\n", .{});
            printUsage();
            return;
        }

        var img = if (std.mem.endsWith(u8, image_path.?, ".ppm"))
            try sam3.ImageRGB.loadPPM(allocator, image_path.?)
        else
            return error.UnsupportedImageFormat;
        defer img.deinit();

        std.debug.print("Loaded image '{s}' (Resolution: {d}x{d})\n", .{ image_path.?, img.width, img.height });

        var img_tensor = try img.toTensor(allocator);
        defer img_tensor.deinit();

        const config = if (std.mem.eql(u8, preset_str, "tiny"))
            sam3.SAM3Config.sam3_tiny()
        else
            sam3.SAM3Config.sam3_base();

        var model = sam3.SAM3.init(allocator, config);
        defer model.deinit();

        if (weights_path) |wpath| {
            try model.loadWeights(wpath);
        }

        var pts_list: std.ArrayList(sam3.Point) = .empty;
        defer pts_list.deinit(allocator);

        if (point_str) |pstr| {
            var it = std.mem.splitScalar(u8, pstr, ',');
            const x_str = it.next() orelse "0.5";
            const y_str = it.next() orelse "0.5";
            const l_str = it.next() orelse "1";
            const px = try std.fmt.parseFloat(f32, x_str);
            const py = try std.fmt.parseFloat(f32, y_str);
            const pl = try std.fmt.parseInt(i32, l_str, 10);
            try pts_list.append(allocator, sam3.Point{ .x = px, .y = py, .label = pl });
        }

        const prompt = sam3.Prompt{
            .points = if (pts_list.items.len > 0) pts_list.items else null,
            .text = text_prompt,
        };

        var res = try model.segmentImage(img_tensor, prompt, false);
        defer res.deinit();

        std.debug.print("Segmentation completed successfully:\n", .{});
        std.debug.print("  -> Model Preset: {s} (Input size: {d}x{d})\n", .{ preset_str, config.image_size, config.image_size });
        std.debug.print("  -> Predicted IoU Quality Score: {d:.4}\n", .{res.iou_scores.at2(0, 0)});
        std.debug.print("  -> Concept Presence Score: {d:.4}\n", .{res.presence_score});

        sam3.visualization.overlayMask(&img, res.masks, sam3.RGB{ .r = 0, .g = 220, .b = 100 }, 0.5);
        if (pts_list.items.len > 0) {
            sam3.visualization.drawPointMarker(&img, pts_list.items[0], 6);
        }

        if (std.mem.endsWith(u8, output_path.?, ".ppm")) {
            try img.savePPM(output_path.?);
        } else if (std.mem.endsWith(u8, output_path.?, ".bmp")) {
            try img.saveBMP(output_path.?);
        } else {
            try img.saveBMP(output_path.?);
        }
        std.debug.print("  -> Saved segmented mask overlay to '{s}'\n\n", .{output_path.?});
    } else {
        printUsage();
    }
}
