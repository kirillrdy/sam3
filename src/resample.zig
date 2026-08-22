//! Bilinear resampling of a single-channel f32 plane.
//!
//! Two places need it and they are the two ends of the pass: the image has to
//! reach the encoder at 1008x1008, and the mask logits come back at 288x288 and
//! have to be laid over the original frame. Both are the `align_corners=false`
//! half-pixel mapping torch uses, which is what the checkpoint was exported
//! against.

const std = @import("std");

/// Resamples `src` (`src_w` x `src_h`) into a newly allocated `dst_w` x `dst_h`
/// plane. The caller owns the result.
pub fn bilinear(
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

/// The floor of a source coordinate, clamped into the plane. Negative
/// coordinates happen at the first output pixel of an upsample, where the
/// half-pixel shift reaches back past the edge.
fn clampIndex(coordinate: f32, limit: usize) usize {
    if (coordinate <= 0.0) return 0;
    const floored: usize = @intFromFloat(@floor(coordinate));
    return @min(floored, limit - 1);
}

test "resampling a plane to its own size leaves it alone" {
    const allocator = std.testing.allocator;
    const src = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    const out = try bilinear(allocator, &src, 2, 2, 2, 2);
    defer allocator.free(out);

    try std.testing.expectEqualSlices(f32, &src, out);
}

test "upsampling interpolates between the source samples" {
    const allocator = std.testing.allocator;
    const src = [_]f32{ 0.0, 4.0 };

    const out = try bilinear(allocator, &src, 2, 1, 4, 1);
    defer allocator.free(out);

    // Half-pixel centres at 0.25/0.5/0.75 of the way across, clamped at the
    // edges: 0, 1, 3, 4.
    try std.testing.expectEqualSlices(f32, &.{ 0.0, 1.0, 3.0, 4.0 }, out);
}

test "downsampling averages towards the middle" {
    const allocator = std.testing.allocator;
    const src = [_]f32{ 0.0, 0.0, 8.0, 8.0 };

    const out = try bilinear(allocator, &src, 4, 1, 2, 1);
    defer allocator.free(out);

    try std.testing.expectEqualSlices(f32, &.{ 0.0, 8.0 }, out);
}
