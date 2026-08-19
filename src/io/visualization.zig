const std = @import("std");
const ImageRGB = @import("image.zig").ImageRGB;
const RGB = @import("image.zig").RGB;
const Tensor = @import("../tensor/tensor.zig").Tensor;
const Box = @import("../models/prompt_encoder.zig").Box;
const Point = @import("../models/prompt_encoder.zig").Point;
const parallel = @import("../tensor/parallel.zig");

pub const PALETTE = [_]RGB{
    .{ .r = 31, .g = 119, .b = 180 },  // Blue
    .{ .r = 255, .g = 127, .b = 14 },  // Orange
    .{ .r = 44, .g = 160, .b = 44 },   // Green
    .{ .r = 214, .g = 39, .b = 40 },   // Red
    .{ .r = 148, .g = 103, .b = 189 }, // Purple
    .{ .r = 140, .g = 86, .b = 75 },   // Brown
    .{ .r = 227, .g = 119, .b = 194 }, // Pink
    .{ .r = 127, .g = 127, .b = 127 }, // Gray
    .{ .r = 188, .g = 189, .b = 34 },  // Yellow-Green
    .{ .r = 23, .g = 190, .b = 207 },  // Cyan
};

pub fn getColorForId(id: usize) RGB {
    return PALETTE[id % PALETTE.len];
}

pub fn overlayMask(
    image: *ImageRGB,
    mask: Tensor, // [1, 1, H, W] or [1, H, W]
    color: RGB,
    alpha: f32,
) void {
    const mask_h = if (mask.shape.len == 4) mask.shape[2] else mask.shape[1];
    const mask_w = if (mask.shape.len == 4) mask.shape[3] else mask.shape[2];

    const Context = struct {
        image: *ImageRGB,
        mask: Tensor,
        color: RGB,
        alpha: f32,
        mask_h: usize,
        mask_w: usize,
    };

    var ctx = Context{
        .image = image,
        .mask = mask,
        .color = color,
        .alpha = alpha,
        .mask_h = mask_h,
        .mask_w = mask_w,
    };

    parallel.parallelFor(image.allocator, image.height, &ctx, struct {
        fn worker(c: *Context, start_y: usize, end_y: usize) void {
            const img_w = c.image.width;
            const img_h = c.image.height;
            const mh = c.mask_h;
            const mw = c.mask_w;
            const cr = @as(f32, @floatFromInt(c.color.r));
            const cg = @as(f32, @floatFromInt(c.color.g));
            const cb = @as(f32, @floatFromInt(c.color.b));
            const a = c.alpha;
            const inv_a = 1.0 - a;
            const is_4d = c.mask.shape.len == 4;

            for (start_y..end_y) |y| {
                const my = y * mh / img_h;
                const row_offset = y * img_w * 3;

                for (0..img_w) |x| {
                    const mx = x * mw / img_w;
                    const logit = if (is_4d) c.mask.at4(0, 0, my, mx) else c.mask.at3(0, my, mx);

                    if (logit > 0.0) {
                        const idx = row_offset + x * 3;
                        const orig_r = @as(f32, @floatFromInt(c.image.data[idx]));
                        const orig_g = @as(f32, @floatFromInt(c.image.data[idx + 1]));
                        const orig_b = @as(f32, @floatFromInt(c.image.data[idx + 2]));

                        c.image.data[idx] = @intFromFloat(orig_r * inv_a + cr * a);
                        c.image.data[idx + 1] = @intFromFloat(orig_g * inv_a + cg * a);
                        c.image.data[idx + 2] = @intFromFloat(orig_b * inv_a + cb * a);
                    }
                }
            }
        }
    }.worker);
}

pub fn drawBoundingBox(
    image: *ImageRGB,
    box: Box,
    color: RGB,
    thickness: usize,
) void {
    const x1: usize = @intFromFloat(box.x1 * @as(f32, @floatFromInt(image.width)));
    const y1: usize = @intFromFloat(box.y1 * @as(f32, @floatFromInt(image.height)));
    const x2: usize = @min(image.width - 1, @as(usize, @intFromFloat(box.x2 * @as(f32, @floatFromInt(image.width)))));
    const y2: usize = @min(image.height - 1, @as(usize, @intFromFloat(box.y2 * @as(f32, @floatFromInt(image.height)))));

    for (0..thickness) |t| {
        // Top and bottom borders
        for (x1..x2 + 1) |x| {
            if (y1 + t < image.height) image.setPixel(x, y1 + t, color);
            if (y2 >= t) image.setPixel(x, y2 - t, color);
        }
        // Left and right borders
        for (y1..y2 + 1) |y| {
            if (x1 + t < image.width) image.setPixel(x1 + t, y, color);
            if (x2 >= t) image.setPixel(x2 - t, y, color);
        }
    }
}

pub fn drawPointMarker(
    image: *ImageRGB,
    pt: Point,
    radius: usize,
) void {
    const cx: isize = @intFromFloat(pt.x * @as(f32, @floatFromInt(image.width)));
    const cy: isize = @intFromFloat(pt.y * @as(f32, @floatFromInt(image.height)));
    const color = if (pt.label == 1) RGB{ .r = 0, .g = 255, .b = 0 } else RGB{ .r = 255, .g = 0, .b = 0 };

    const r_signed: isize = @intCast(radius);
    var dy = -r_signed;
    while (dy <= r_signed) : (dy += 1) {
        var dx = -r_signed;
        while (dx <= r_signed) : (dx += 1) {
            if (dx * dx + dy * dy <= r_signed * r_signed) {
                const px = cx + dx;
                const py = cy + dy;
                if (px >= 0 and py >= 0 and px < @as(isize, @intCast(image.width)) and py < @as(isize, @intCast(image.height))) {
                    image.setPixel(@intCast(px), @intCast(py), color);
                }
            }
        }
    }
}
