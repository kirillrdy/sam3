const std = @import("std");
const onnx = @import("../onnx.zig");
const sam3 = @import("../sam3.zig");
const resample = @import("../resample.zig");
const image_io = @import("../io/image.zig");
const ImageRGB = image_io.ImageRGB;
const RGB = image_io.RGB;
const overlayMask = @import("../io/visualization.zig").overlayMask;
const drawPointMarker = @import("../io/visualization.zig").drawPointMarker;

inline fn monotonicNs() u64 {
    var ts: std.os.linux.timespec = undefined;
    _ = std.os.linux.clock_gettime(std.os.linux.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn secondsSince(start: u64) f64 {
    return @as(f64, @floatFromInt(monotonicNs() - start)) / 1e9;
}

/// Point-prompted segmentation: the frame goes through SAM 3's vision encoder,
/// the click and the resulting feature pyramid through its prompt encoder and
/// mask decoder, and every mask hypothesis that comes back is written out over
/// the original image.
pub fn runSegment(
    allocator: std.mem.Allocator,
    image_path: []const u8,
    paths: sam3.Paths,
    output_path: []const u8,
    points: []const sam3.Point,
    target: sam3.Target,
) !void {
    std.debug.print("\n=== SAM 3 Point-Prompted Segmentation ===\n\n", .{});

    var img = try image_io.load(allocator, image_path);
    defer img.deinit();
    std.debug.print("  Image:        {s} ({d}x{d})\n", .{ image_path, img.width, img.height });
    std.debug.print("  Model input:  {d}x{d}\n", .{ sam3.image_size, sam3.image_size });
    std.debug.print("  ONNX Runtime: {s}\n\n", .{onnx.version()});

    for (points) |p| {
        std.debug.print("  Prompt:       ({d:.3}, {d:.3}) label {d}\n", .{ p.x, p.y, p.label });
    }

    const t_open = monotonicNs();
    var model = try sam3.Model.open(allocator, paths, target);
    defer model.deinit();
    std.debug.print("\n  Loaded both graphs in {d:.1} s\n", .{secondsSince(t_open)});
    std.debug.print("    vision encoder -> {t}\n", .{model.vision.device});
    std.debug.print("    mask decoder   -> {t}\n\n", .{model.decoder.device});

    const t_run = monotonicNs();
    var masks = try model.segment(img, points);
    defer masks.deinit();
    std.debug.print("  Segmentation:       {d:.2} s\n", .{secondsSince(t_run)});
    std.debug.print("  Object score logit: {d:.4}\n\n", .{masks.object_score});

    // The decoder answers at its own resolution, so each hypothesis is
    // resampled up to the frame before it is thresholded at zero -- the order
    // `post_process_masks` uses, and the one that keeps the edges smooth.
    var path_buf: [512]u8 = undefined;
    const best = masks.best();

    for (0..masks.count) |i| {
        const mask = try resample.bilinear(
            allocator,
            masks.plane(i),
            masks.width,
            masks.height,
            img.width,
            img.height,
        );
        defer allocator.free(mask);

        var covered: usize = 0;
        for (mask) |logit| {
            if (logit > 0.0) covered += 1;
        }
        std.debug.print("  Mask {d}: predicted IoU {d:.4}, covers {d: >5.1}% of the image{s}\n", .{
            i,
            masks.scores[i],
            100.0 * @as(f64, @floatFromInt(covered)) / @as(f64, @floatFromInt(mask.len)),
            if (i == best) "  <- highest IoU" else "",
        });

        var frame = try ImageRGB.init(allocator, img.width, img.height);
        defer frame.deinit();
        @memcpy(frame.data, img.data);

        overlayMask(&frame, mask, RGB{ .r = 0, .g = 220, .b = 100 }, 0.5);
        for (points) |p| drawPointMarker(&frame, p, 7);

        const path = if (masks.count == 1)
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
    std.debug.print("\n", .{});
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
