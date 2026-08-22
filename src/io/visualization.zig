const std = @import("std");
const ImageRGB = @import("image.zig").ImageRGB;
const RGB = @import("image.zig").RGB;
const Point = @import("../sam3.zig").Point;

/// Tints every pixel the mask claims. `mask` holds logits at the image's own
/// resolution -- the decoder's output resampled -- and the threshold is zero,
/// which is where `post_process_masks` puts it too.
pub fn overlayMask(image: *ImageRGB, mask: []const f32, color: RGB, alpha: f32) void {
    std.debug.assert(mask.len == image.width * image.height);

    const tint = [3]f32{
        @floatFromInt(color.r),
        @floatFromInt(color.g),
        @floatFromInt(color.b),
    };
    const keep = 1.0 - alpha;

    for (mask, 0..) |logit, i| {
        if (logit <= 0.0) continue;
        for (0..3) |channel| {
            const original: f32 = @floatFromInt(image.data[i * 3 + channel]);
            image.data[i * 3 + channel] = @intFromFloat(original * keep + tint[channel] * alpha);
        }
    }
}

/// Marks where the prompt was clicked: green for a point the mask should
/// include, red for one it should exclude.
pub fn drawPointMarker(image: *ImageRGB, point: Point, radius: usize) void {
    const cx: isize = @intFromFloat(point.x * @as(f32, @floatFromInt(image.width)));
    const cy: isize = @intFromFloat(point.y * @as(f32, @floatFromInt(image.height)));
    const color = if (point.label == 1)
        RGB{ .r = 0, .g = 255, .b = 0 }
    else
        RGB{ .r = 255, .g = 0, .b = 0 };

    const r: isize = @intCast(radius);
    var dy = -r;
    while (dy <= r) : (dy += 1) {
        var dx = -r;
        while (dx <= r) : (dx += 1) {
            if (dx * dx + dy * dy > r * r) continue;
            const px = cx + dx;
            const py = cy + dy;
            if (px < 0 or py < 0) continue;
            if (px >= @as(isize, @intCast(image.width)) or py >= @as(isize, @intCast(image.height))) continue;
            image.setPixel(@intCast(px), @intCast(py), color);
        }
    }
}
