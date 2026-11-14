const std = @import("std");
const zigimg = @import("zigimg");
const Point = @import("point.zig").Point;

const Rgb24 = zigimg.color.Rgb24;

const mask_color: Rgb24 = .{ .r = 0, .g = 220, .b = 100 };
const mask_alpha: f32 = 0.5;
const marker_radius: usize = 7;

pub const AxisSample = struct {
    lo: usize,
    hi: usize,
    w: f32,
};

pub fn axisSample(dst_index: usize, ratio: f32, limit: usize) AxisSample {
    const coord = ratio * (@as(f32, @floatFromInt(dst_index)) + 0.5) - 0.5;
    const lo: usize = if (coord <= 0.0) 0 else @min(@as(usize, @intFromFloat(@floor(coord))), limit - 1);
    return .{
        .lo = lo,
        .hi = @min(lo + 1, limit - 1),
        .w = @max(0.0, coord - @as(f32, @floatFromInt(lo))),
    };
}

pub fn bilerp(a: f32, b: f32, c: f32, d: f32, wx: f32, wy: f32) f32 {
    const top = a + (b - a) * wx;
    const bottom = c + (d - c) * wx;
    return top + (bottom - top) * wy;
}

fn blendChannel(base: u8, tint: u8) u8 {
    return @intFromFloat(@as(f32, @floatFromInt(base)) * (1.0 - mask_alpha) + @as(f32, @floatFromInt(tint)) * mask_alpha);
}

fn drawPointMarker(frame_rgba: []u8, width: usize, height: usize, point: Point, radius: usize) void {
    const width_i: isize = @intCast(width);
    const height_i: isize = @intCast(height);
    const cx: isize = @intFromFloat(point.x * @as(f32, @floatFromInt(width)));
    const cy: isize = @intFromFloat(point.y * @as(f32, @floatFromInt(height)));
    const color: [3]u8 = if (point.label == .positive) .{ 0, 255, 0 } else .{ 255, 0, 0 };

    const r: isize = @intCast(radius);
    var dy = -r;
    while (dy <= r) : (dy += 1) {
        var dx = -r;
        while (dx <= r) : (dx += 1) {
            if (dx * dx + dy * dy > r * r) continue;
            const px = cx + dx;
            const py = cy + dy;
            if (px < 0 or py < 0 or px >= width_i or py >= height_i) continue;
            const idx: usize = @intCast(py * width_i + px);
            frame_rgba[idx * 4 ..][0..3].* = color;
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
    img: zigimg.Image,
    frame_rgba: []u8,
    mask_plane: ?[]const f32,
    mask_width: usize,
    mask_height: usize,
    points: []const Point,
) void {
    if (img.width == 0 or img.height == 0) return;
    const pixels = img.pixels.rgb24;
    std.debug.assert(frame_rgba.len >= pixels.len * 4);

    const active_mask = if (mask_plane) |plane| blk: {
        if (mask_width == 0 or mask_height == 0) break :blk null;
        std.debug.assert(plane.len == mask_width * mask_height);
        break :blk plane;
    } else null;

    const ratio_y = if (active_mask != null) @as(f32, @floatFromInt(mask_height)) / @as(f32, @floatFromInt(img.height)) else 0;
    const ratio_x = if (active_mask != null) @as(f32, @floatFromInt(mask_width)) / @as(f32, @floatFromInt(img.width)) else 0;

    for (0..img.height) |y| {
        const sy = if (active_mask != null) axisSample(y, ratio_y, mask_height) else undefined;
        for (0..img.width) |x| {
            const idx = y * img.width + x;
            const px = pixels[idx];
            const tint = if (active_mask) |plane| blk: {
                const sx = axisSample(x, ratio_x, mask_width);
                const row0 = plane[sy.lo * mask_width ..][0..mask_width];
                const row1 = plane[sy.hi * mask_width ..][0..mask_width];
                break :blk bilerp(row0[sx.lo], row0[sx.hi], row1[sx.lo], row1[sx.hi], sx.w, sy.w) > 0.0;
            } else false;
            frame_rgba[idx * 4 ..][0..4].* = if (tint)
                .{ blendChannel(px.r, mask_color.r), blendChannel(px.g, mask_color.g), blendChannel(px.b, mask_color.b), 255 }
            else
                .{ px.r, px.g, px.b, 255 };
        }
    }

    for (points) |p| {
        drawPointMarker(frame_rgba, img.width, img.height, p, marker_radius);
    }
}

test "bilinear sampling preserves identity grid" {
    const src = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var out: [4]f32 = undefined;
    for (0..2) |y| {
        const sy = axisSample(y, 1.0, 2);
        for (0..2) |x| {
            const sx = axisSample(x, 1.0, 2);
            out[y * 2 + x] = bilerp(src[sy.lo * 2 + sx.lo], src[sy.lo * 2 + sx.hi], src[sy.hi * 2 + sx.lo], src[sy.hi * 2 + sx.hi], sx.w, sy.w);
        }
    }
    try std.testing.expectEqualSlices(f32, &src, &out);
}

test "compositeRgba tints positive logit pixels" {
    const allocator = std.testing.allocator;
    var img = try zigimg.Image.create(allocator, 2, 1, .rgb24);
    defer img.deinit(allocator);
    @memset(img.pixels.rgb24, .{ .r = 100, .g = 100, .b = 100 });

    const mask = [_]f32{ 1.0, -1.0 };
    var rgba: [8]u8 = undefined;
    compositeRgba(img, &rgba, &mask, 2, 1, &.{});

    try std.testing.expectEqualSlices(u8, &.{ 50, 160, 100, 255, 100, 100, 100, 255 }, &rgba);
}

test "compositeRgba draws positive green and negative red markers" {
    const allocator = std.testing.allocator;
    var img = try zigimg.Image.create(allocator, 20, 20, .rgb24);
    defer img.deinit(allocator);
    @memset(img.pixels.rgb24, .{ .r = 0, .g = 0, .b = 0 });

    var rgba: [20 * 20 * 4]u8 = undefined;
    compositeRgba(img, &rgba, null, 0, 0, &.{
        .{ .x = 0.2, .y = 0.2, .label = .positive },
        .{ .x = 0.8, .y = 0.8, .label = .negative },
    });

    const pos_off = (4 * 20 + 4) * 4;
    try std.testing.expectEqualSlices(u8, &.{ 0, 255, 0, 255 }, rgba[pos_off..][0..4]);

    const neg_off = (16 * 20 + 16) * 4;
    try std.testing.expectEqualSlices(u8, &.{ 255, 0, 0, 255 }, rgba[neg_off..][0..4]);
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
