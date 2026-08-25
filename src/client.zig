const std = @import("std");
const image_io = @import("io/image.zig");
const resample = @import("resample.zig");
const visualization = @import("io/visualization.zig");
const protocol = @import("web/protocol.zig");
const Point = @import("point.zig").Point;
const ImageRGB = image_io.ImageRGB;
const RGB = image_io.RGB;

const gpa = std.heap.page_allocator;

const mask_color: RGB = .{ .r = 0, .g = 220, .b = 100 };
const mask_alpha = 0.5;
const marker_radius = 7;

const max_points = 32;

var image: ?ImageRGB = null;

var frame: []u8 = &.{};

var points: [max_points]Point = undefined;
var points_len: usize = 0;
var query_buffer: [max_points * 32]u8 = undefined;
var query_len: usize = 0;

var masks: Masks = .{};

const Masks = struct {
    logits: []f32 = &.{},
    scores: []f32 = &.{},

    coverage: []f32 = &.{},
    count: usize = 0,
    width: usize = 0,
    height: usize = 0,
    object_score: f32 = 0,

    fn plane(self: Masks, index: usize) []const f32 {
        const stride = self.width * self.height;
        return self.logits[index * stride ..][0..stride];
    }

    fn clear(self: *Masks) void {
        gpa.free(self.logits);
        gpa.free(self.scores);
        gpa.free(self.coverage);
        self.* = .{};
    }
};

export fn alloc(len: u32) ?[*]u8 {
    const buffer = gpa.alloc(u8, len) catch return null;
    return buffer.ptr;
}

export fn dealloc(ptr: [*]u8, len: u32) void {
    gpa.free(ptr[0..len]);
}

export fn loadImage(ptr: [*]const u8, len: u32) i32 {
    const decoded = image_io.decode(gpa, ptr[0..len]) catch return -1;

    if (image) |*old| old.deinit();
    gpa.free(frame);
    masks.clear();
    points_len = 0;
    rebuildQuery();

    image = decoded;
    frame = gpa.alloc(u8, decoded.width * decoded.height * 4) catch {
        image.?.deinit();
        image = null;
        frame = &.{};
        return -1;
    };
    render(-1);
    return 0;
}

export fn imageWidth() u32 {
    return if (image) |img| @intCast(img.width) else 0;
}

export fn imageHeight() u32 {
    return if (image) |img| @intCast(img.height) else 0;
}

export fn framePtr() [*]const u8 {
    return frame.ptr;
}

export fn addPoint(x: f32, y: f32, width: f32, height: f32, label: i32) void {
    if (points_len == max_points or width <= 0 or height <= 0) return;
    points[points_len] = .{
        .x = std.math.clamp(x / width, 0.0, 1.0),
        .y = std.math.clamp(y / height, 0.0, 1.0),
        .label = label,
    };
    points_len += 1;
    rebuildQuery();
}

export fn clearPoints() void {
    points_len = 0;
    masks.clear();
    rebuildQuery();
    render(-1);
}

export fn pointCount() u32 {
    return @intCast(points_len);
}

export fn queryPtr() [*]const u8 {
    return &query_buffer;
}

export fn queryLen() u32 {
    return @intCast(query_len);
}

fn rebuildQuery() void {
    var writer = std.Io.Writer.fixed(&query_buffer);
    for (points[0..points_len], 0..) |p, i| {
        writer.print("{s}p={d:.5},{d:.5},{d}", .{
            if (i == 0) "" else "&",
            p.x,
            p.y,
            p.label,
        }) catch break;
    }
    query_len = writer.end;
}

export fn loadMasks(ptr: [*]const u8, len: u32) i32 {
    const bytes = ptr[0..len];
    const header = protocol.Header.parse(bytes) catch return -1;
    if (bytes.len < header.responseSize()) return -1;

    const count: usize = header.count;
    const stride = @as(usize, header.width) * header.height;

    var loaded: Masks = .{
        .count = count,
        .width = header.width,
        .height = header.height,
        .object_score = header.object_score,
    };
    loaded.scores = gpa.alloc(f32, count) catch return -1;
    loaded.coverage = gpa.alloc(f32, count) catch {
        gpa.free(loaded.scores);
        return -1;
    };
    loaded.logits = gpa.alloc(f32, count * stride) catch {
        gpa.free(loaded.scores);
        gpa.free(loaded.coverage);
        return -1;
    };

    var offset: usize = protocol.Header.size;
    for (loaded.scores) |*score| {
        score.* = readF32(bytes, &offset);
    }
    for (loaded.logits) |*logit| {
        logit.* = readF32(bytes, &offset);
    }

    for (loaded.coverage, 0..) |*share, i| {
        var covered: usize = 0;
        for (loaded.plane(i)) |logit| {
            if (logit > 0.0) covered += 1;
        }
        share.* = @as(f32, @floatFromInt(covered)) / @as(f32, @floatFromInt(stride));
    }

    masks.clear();
    masks = loaded;
    return @intCast(count);
}

fn readF32(bytes: []const u8, offset: *usize) f32 {
    const value: f32 = @bitCast(std.mem.readInt(u32, bytes[offset.*..][0..4], .little));
    offset.* += 4;
    return value;
}

export fn maskCount() u32 {
    return @intCast(masks.count);
}

export fn maskScore(index: u32) f32 {
    return if (index < masks.count) masks.scores[index] else 0;
}

export fn maskCoverage(index: u32) f32 {
    return if (index < masks.count) masks.coverage[index] else 0;
}

export fn objectScore() f32 {
    return masks.object_score;
}

export fn bestMask() u32 {
    var winner: usize = 0;
    for (masks.scores, 0..) |score, i| {
        if (score > masks.scores[winner]) winner = i;
    }
    return @intCast(winner);
}

export fn render(index: i32) void {
    const img = image orelse return;

    var canvas: ImageRGB = .{
        .width = img.width,
        .height = img.height,
        .data = gpa.alloc(u8, img.data.len) catch return,
        .allocator = gpa,
    };
    defer canvas.deinit();
    @memcpy(canvas.data, img.data);

    if (index >= 0 and @as(usize, @intCast(index)) < masks.count) {
        if (resample.bilinear(
            gpa,
            masks.plane(@intCast(index)),
            masks.width,
            masks.height,
            img.width,
            img.height,
        )) |mask| {
            defer gpa.free(mask);
            visualization.overlayMask(&canvas, mask, mask_color, mask_alpha);
        } else |_| {}
    }

    for (points[0..points_len]) |p| visualization.drawPointMarker(&canvas, p, marker_radius);

    for (0..img.width * img.height) |i| {
        frame[i * 4 + 0] = canvas.data[i * 3 + 0];
        frame[i * 4 + 1] = canvas.data[i * 3 + 1];
        frame[i * 4 + 2] = canvas.data[i * 3 + 2];
        frame[i * 4 + 3] = 255;
    }
}

fn testPng(allocator: std.mem.Allocator, buffer: []u8, width: usize, height: usize, color: RGB) ![]u8 {
    const zigimg = @import("zigimg");

    var source = try zigimg.Image.create(allocator, width, height, .rgb24);
    defer source.deinit(allocator);
    for (source.pixels.rgb24) |*pixel| pixel.* = .{ .r = color.r, .g = color.g, .b = color.b };

    return source.writeToMemory(allocator, buffer, .{ .png = .{} });
}

fn testOpen(png: []const u8) !void {
    const ptr = alloc(@intCast(png.len)) orelse return error.OutOfMemory;
    defer dealloc(ptr, @intCast(png.len));
    @memcpy(ptr[0..png.len], png);
    try std.testing.expectEqual(@as(i32, 0), loadImage(ptr, @intCast(png.len)));
}

test "a click is normalised against the canvas it landed on, not the image" {
    clearPoints();

    addPoint(50, 25, 100, 50, 1);
    addPoint(0, 0, 100, 50, 0);

    addPoint(200, -10, 100, 50, 1);

    try std.testing.expectEqual(@as(u32, 3), pointCount());
    try std.testing.expectEqualStrings(
        "p=0.50000,0.50000,1&p=0.00000,0.00000,0&p=1.00000,0.00000,1",
        queryPtr()[0..queryLen()],
    );

    clearPoints();
    try std.testing.expectEqual(@as(u32, 0), pointCount());
    try std.testing.expectEqual(@as(u32, 0), queryLen());
}

test "an image the page opens reaches the canvas as RGBA" {
    var buffer: [8192]u8 = undefined;
    const png = try testPng(std.testing.allocator, &buffer, 3, 2, .{ .r = 10, .g = 20, .b = 30 });

    clearPoints();
    try testOpen(png);

    try std.testing.expectEqual(@as(u32, 3), imageWidth());
    try std.testing.expectEqual(@as(u32, 2), imageHeight());

    const pixels = framePtr()[0 .. 3 * 2 * 4];
    for (0..6) |i| {
        try std.testing.expectEqualSlices(u8, &.{ 10, 20, 30, 255 }, pixels[i * 4 ..][0..4]);
    }
}

test "a file that is not an image leaves the canvas alone" {
    var buffer: [8192]u8 = undefined;
    const png = try testPng(std.testing.allocator, &buffer, 3, 2, .{ .r = 10, .g = 20, .b = 30 });
    try testOpen(png);

    const junk = "not an image at all";
    const ptr = alloc(junk.len).?;
    defer dealloc(ptr, junk.len);
    @memcpy(ptr[0..junk.len], junk);

    try std.testing.expectEqual(@as(i32, -1), loadImage(ptr, junk.len));
    try std.testing.expectEqual(@as(u32, 3), imageWidth());
    try std.testing.expectEqual(@as(u32, 2), imageHeight());
}

test "a response becomes hypotheses, and the chosen one is tinted over the frame" {
    var buffer: [8192]u8 = undefined;
    const png = try testPng(std.testing.allocator, &buffer, 1, 1, .{ .r = 200, .g = 10, .b = 10 });

    clearPoints();
    try testOpen(png);

    const header: protocol.Header = .{ .count = 2, .width = 1, .height = 1, .object_score = 2.5 };
    var response: [protocol.Header.size + 2 * 4 + 2 * 4]u8 = undefined;
    header.write(response[0..protocol.Header.size]);
    for ([_]f32{ 0.25, 0.75, -1.0, 1.0 }, 0..) |value, i| {
        std.mem.writeInt(u32, response[protocol.Header.size + i * 4 ..][0..4], @bitCast(value), .little);
    }

    const ptr = alloc(response.len).?;
    defer dealloc(ptr, response.len);
    @memcpy(ptr[0..response.len], &response);
    try std.testing.expectEqual(@as(i32, 2), loadMasks(ptr, response.len));

    try std.testing.expectEqual(@as(u32, 2), maskCount());
    try std.testing.expectEqual(@as(f32, 0.25), maskScore(0));
    try std.testing.expectEqual(@as(f32, 2.5), objectScore());
    try std.testing.expectEqual(@as(u32, 1), bestMask());
    try std.testing.expectEqual(@as(f32, 0.0), maskCoverage(0));
    try std.testing.expectEqual(@as(f32, 1.0), maskCoverage(1));

    render(@intCast(bestMask()));
    try std.testing.expectEqualSlices(u8, &.{ 100, 115, 55, 255 }, framePtr()[0..4]);

    render(0);
    try std.testing.expectEqualSlices(u8, &.{ 200, 10, 10, 255 }, framePtr()[0..4]);
    render(-1);
    try std.testing.expectEqualSlices(u8, &.{ 200, 10, 10, 255 }, framePtr()[0..4]);
}

test "a response that is not one is refused rather than half read" {
    const truncated = "SAM3" ++ [_]u8{0} ** 8;
    const ptr = alloc(truncated.len).?;
    defer dealloc(ptr, truncated.len);
    @memcpy(ptr[0..truncated.len], truncated);
    try std.testing.expectEqual(@as(i32, -1), loadMasks(ptr, truncated.len));

    const header: protocol.Header = .{ .count = 3, .width = 288, .height = 288, .object_score = 0 };
    var short: [protocol.Header.size]u8 = undefined;
    header.write(&short);
    const short_ptr = alloc(short.len).?;
    defer dealloc(short_ptr, short.len);
    @memcpy(short_ptr[0..short.len], &short);
    try std.testing.expectEqual(@as(i32, -1), loadMasks(short_ptr, short.len));
}
