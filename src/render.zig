const std = @import("std");
const zigimg = @import("zigimg");
pub const Point = @import("point.zig").Point;

const Rgb24 = zigimg.color.Rgb24;

const mask_color: Rgb24 = .{ .r = 0, .g = 220, .b = 100 };
const mask_alpha: f32 = 0.5;
const marker_radius: usize = 7;

fn bilinear(
    allocator: std.mem.Allocator,
    src: []const f32,
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) ![]f32 {
    std.debug.assert(src.len == src_w * src_h);
    const dst = try allocator.alloc(f32, dst_w * dst_h);
    errdefer allocator.free(dst);

    const ratio_y = @as(f32, @floatFromInt(src_h)) / @as(f32, @floatFromInt(dst_h));
    const ratio_x = @as(f32, @floatFromInt(src_w)) / @as(f32, @floatFromInt(dst_w));

    for (0..dst_h) |y| {
        const in_y = ratio_y * (@as(f32, @floatFromInt(y)) + 0.5) - 0.5;
        const y0 = clampIndex(in_y, src_h);
        const y1 = @min(y0 + 1, src_h - 1);
        const wy = @max(0.0, in_y - @as(f32, @floatFromInt(y0)));

        const row0 = src[y0 * src_w ..][0..src_w];
        const row1 = src[y1 * src_w ..][0..src_w];
        const out = dst[y * dst_w ..][0..dst_w];

        for (0..dst_w) |x| {
            const in_x = ratio_x * (@as(f32, @floatFromInt(x)) + 0.5) - 0.5;
            const x0 = clampIndex(in_x, src_w);
            const x1 = @min(x0 + 1, src_w - 1);
            const wx = @max(0.0, in_x - @as(f32, @floatFromInt(x0)));

            const top = row0[x0] + (row0[x1] - row0[x0]) * wx;
            const bottom = row1[x0] + (row1[x1] - row1[x0]) * wx;
            out[x] = top + (bottom - top) * wy;
        }
    }
    return dst;
}

fn clampIndex(coordinate: f32, limit: usize) usize {
    if (coordinate <= 0.0) return 0;
    const floored: usize = @intFromFloat(@floor(coordinate));
    return @min(floored, limit - 1);
}

fn overlayMask(img: *zigimg.Image, mask: []const f32, color: Rgb24, alpha: f32) void {
    const pixels = img.pixels.rgb24;
    std.debug.assert(mask.len == pixels.len);

    const tint = [3]f32{
        @floatFromInt(color.r),
        @floatFromInt(color.g),
        @floatFromInt(color.b),
    };
    const keep = 1.0 - alpha;

    for (mask, pixels) |logit, *px| {
        if (logit <= 0.0) continue;
        const r: f32 = @floatFromInt(px.r);
        const g: f32 = @floatFromInt(px.g);
        const b: f32 = @floatFromInt(px.b);
        px.r = @intFromFloat(r * keep + tint[0] * alpha);
        px.g = @intFromFloat(g * keep + tint[1] * alpha);
        px.b = @intFromFloat(b * keep + tint[2] * alpha);
    }
}

fn drawPointMarker(img: *zigimg.Image, point: Point, radius: usize) void {
    const cx: isize = @intFromFloat(point.x * @as(f32, @floatFromInt(img.width)));
    const cy: isize = @intFromFloat(point.y * @as(f32, @floatFromInt(img.height)));
    const color: Rgb24 = if (point.label == .positive)
        .{ .r = 0, .g = 255, .b = 0 }
    else
        .{ .r = 255, .g = 0, .b = 0 };

    const r: isize = @intCast(radius);
    var dy = -r;
    while (dy <= r) : (dy += 1) {
        var dx = -r;
        while (dx <= r) : (dx += 1) {
            if (dx * dx + dy * dy > r * r) continue;
            const px = cx + dx;
            const py = cy + dy;
            if (px < 0 or py < 0) continue;
            if (px >= @as(isize, @intCast(img.width)) or py >= @as(isize, @intCast(img.height))) continue;
            const idx: usize = @intCast(py * @as(isize, @intCast(img.width)) + px);
            img.pixels.rgb24[idx] = color;
        }
    }
}

pub fn scoreMasks(logits: []const f32, scores: []const f32, count: usize, width: usize, height: usize, coverages_out: []f32) usize {
    const stride = width * height;
    var best_idx: usize = 0;
    for (0..count) |i| {
        const plane = logits[i * stride ..][0..stride];
        var covered: usize = 0;
        for (plane) |logit| {
            if (logit > 0.0) covered += 1;
        }
        if (coverages_out.len > i) {
            coverages_out[i] = @as(f32, @floatFromInt(covered)) / @as(f32, @floatFromInt(stride));
        }
        if (scores[i] > scores[best_idx]) {
            best_idx = i;
        }
    }
    return best_idx;
}

pub fn compositeRgba(
    allocator: std.mem.Allocator,
    img: zigimg.Image,
    frame_rgba: []u8,
    mask_plane: ?[]const f32,
    mask_width: usize,
    mask_height: usize,
    points: []const Point,
) void {
    var canvas = zigimg.Image.create(allocator, img.width, img.height, .rgb24) catch return;
    defer canvas.deinit(allocator);
    @memcpy(canvas.pixels.rgb24, img.pixels.rgb24);

    if (mask_plane) |plane| {
        if (bilinear(allocator, plane, mask_width, mask_height, img.width, img.height)) |resampled| {
            defer allocator.free(resampled);
            overlayMask(&canvas, resampled, mask_color, mask_alpha);
        } else |_| {}
    }

    for (points) |p| {
        drawPointMarker(&canvas, p, marker_radius);
    }

    for (canvas.pixels.rgb24, 0..) |px, i| {
        frame_rgba[i * 4 + 0] = px.r;
        frame_rgba[i * 4 + 1] = px.g;
        frame_rgba[i * 4 + 2] = px.b;
        frame_rgba[i * 4 + 3] = 255;
    }
}

test "bilinear resample identity" {
    const allocator = std.testing.allocator;
    const src = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const out = try bilinear(allocator, &src, 2, 2, 2, 2);
    defer allocator.free(out);
    try std.testing.expectEqualSlices(f32, &src, out);
}

test "overlayMask tints positive logit pixels" {
    const allocator = std.testing.allocator;
    var img = try zigimg.Image.create(allocator, 2, 1, .rgb24);
    defer img.deinit(allocator);

    img.pixels.rgb24[0] = .{ .r = 100, .g = 100, .b = 100 };
    img.pixels.rgb24[1] = .{ .r = 100, .g = 100, .b = 100 };

    const mask = [_]f32{ 1.0, -1.0 };
    overlayMask(&img, &mask, .{ .r = 0, .g = 200, .b = 100 }, 0.5);

    try std.testing.expectEqual(@as(u8, 50), img.pixels.rgb24[0].r);
    try std.testing.expectEqual(@as(u8, 150), img.pixels.rgb24[0].g);
    try std.testing.expectEqual(@as(u8, 100), img.pixels.rgb24[0].b);

    try std.testing.expectEqual(@as(u8, 100), img.pixels.rgb24[1].r);
    try std.testing.expectEqual(@as(u8, 100), img.pixels.rgb24[1].g);
    try std.testing.expectEqual(@as(u8, 100), img.pixels.rgb24[1].b);
}

test "drawPointMarker draws positive green and negative red markers" {
    const allocator = std.testing.allocator;
    var img = try zigimg.Image.create(allocator, 10, 10, .rgb24);
    defer img.deinit(allocator);

    @memset(img.pixels.rgb24, .{ .r = 0, .g = 0, .b = 0 });

    drawPointMarker(&img, .{ .x = 0.5, .y = 0.5, .label = .positive }, 2);
    const center_idx = 5 * 10 + 5;
    try std.testing.expectEqual(@as(u8, 0), img.pixels.rgb24[center_idx].r);
    try std.testing.expectEqual(@as(u8, 255), img.pixels.rgb24[center_idx].g);
    try std.testing.expectEqual(@as(u8, 0), img.pixels.rgb24[center_idx].b);

    drawPointMarker(&img, .{ .x = 0.1, .y = 0.1, .label = .negative }, 1);
    const neg_idx = 1 * 10 + 1;
    try std.testing.expectEqual(@as(u8, 255), img.pixels.rgb24[neg_idx].r);
    try std.testing.expectEqual(@as(u8, 0), img.pixels.rgb24[neg_idx].g);
    try std.testing.expectEqual(@as(u8, 0), img.pixels.rgb24[neg_idx].b);
}

test "scoreMasks selects highest score and calculates coverage" {
    const logits = [_]f32{
        1.0, 1.0, -1.0, -1.0, // mask 0: 50%
        1.0, 1.0, 1.0, -1.0, // mask 1: 75%
    };
    const scores = [_]f32{ 0.7, 0.9 };
    var coverages: [2]f32 = undefined;
    const best = scoreMasks(&logits, &scores, 2, 2, 2, &coverages);
    try std.testing.expectEqual(@as(usize, 1), best);
    try std.testing.expectEqual(@as(f32, 0.5), coverages[0]);
    try std.testing.expectEqual(@as(f32, 0.75), coverages[1]);
}
