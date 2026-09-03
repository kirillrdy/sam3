const std = @import("std");
const zigimg = @import("zigimg");
const protocol = @import("protocol.zig");

const Point = struct {
    x: f32,
    y: f32,
    label: i64 = 1,
};

const Rgb24 = zigimg.color.Rgb24;

const gpa = std.heap.page_allocator;

const mask_alpha = 0.5;
const marker_radius = 7;
const box_thickness = 2;

const max_points = 32;

var image: ?zigimg.Image = null;

var frame: []u8 = &.{};

var points: [max_points]Point = undefined;
var points_len: usize = 0;
var query_buffer: [max_points * 32]u8 = undefined;
var query_len: usize = 0;

var masks: Masks = .{};

const Masks = struct {
    logits: []f32 = &.{},
    instances: []protocol.Instance = &.{},

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
        gpa.free(self.instances);
        gpa.free(self.coverage);
        self.* = .{};
    }
};

/// The colour an object is drawn in for as long as it is followed.
///
/// Identity is the whole point of a track, so the colour has to come from the
/// number and nothing else: the same object stays the same colour while other
/// objects come and go around it. Neighbouring numbers are thrown far apart
/// around the hue circle so that two objects side by side never look alike.
fn colorFor(id: u32) Rgb24 {
    // 137.5 degrees, the turn that spreads any run of numbers most evenly.
    const hue = @as(f32, @floatFromInt(id % 360)) * 137.5;
    return fromHsv(hue - @floor(hue / 360.0) * 360.0, 0.85, 1.0);
}

fn fromHsv(hue: f32, saturation: f32, value: f32) Rgb24 {
    const sector = hue / 60.0;
    const offset = sector - @floor(sector / 2.0) * 2.0;
    const chroma = value * saturation;
    const middle = chroma * (1.0 - @abs(offset - 1.0));
    const bottom = value - chroma;

    const rgb: [3]f32 = switch (@as(u32, @intFromFloat(sector)) % 6) {
        0 => .{ chroma, middle, 0 },
        1 => .{ middle, chroma, 0 },
        2 => .{ 0, chroma, middle },
        3 => .{ 0, middle, chroma },
        4 => .{ middle, 0, chroma },
        else => .{ chroma, 0, middle },
    };
    return .{
        .r = @intFromFloat(@round((rgb[0] + bottom) * 255.0)),
        .g = @intFromFloat(@round((rgb[1] + bottom) * 255.0)),
        .b = @intFromFloat(@round((rgb[2] + bottom) * 255.0)),
    };
}

export fn alloc(len: u32) ?[*]u8 {
    const buffer = gpa.alloc(u8, len) catch return null;
    return buffer.ptr;
}

export fn dealloc(ptr: [*]u8, len: u32) void {
    gpa.free(ptr[0..len]);
}

export fn loadImage(ptr: [*]const u8, len: u32) i32 {
    var decoded = zigimg.Image.fromMemory(gpa, ptr[0..len]) catch return -1;
    errdefer decoded.deinit(gpa);
    decoded.convert(gpa, .rgb24) catch return -1;

    if (image) |*old| old.deinit(gpa);
    gpa.free(frame);
    masks.clear();
    points_len = 0;
    rebuildQuery();

    image = decoded;
    frame = gpa.alloc(u8, decoded.width * decoded.height * 4) catch {
        image.?.deinit(gpa);
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

/// Drops the prompt without dropping what is already on screen. Naming a
/// second object in a video starts a new prompt over the objects already being
/// followed, rather than adding to the one that found the first.
export fn forgetPoints() void {
    points_len = 0;
    rebuildQuery();
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
    loaded.instances = gpa.alloc(protocol.Instance, count) catch return -1;
    loaded.coverage = gpa.alloc(f32, count) catch {
        gpa.free(loaded.instances);
        return -1;
    };
    loaded.logits = gpa.alloc(f32, count * stride) catch {
        gpa.free(loaded.instances);
        gpa.free(loaded.coverage);
        return -1;
    };

    for (loaded.instances, 0..) |*instance, i| {
        instance.* = protocol.Instance.parse(
            bytes[protocol.Header.instanceOffset(i)..][0..protocol.Instance.size],
        );
    }

    var offset: usize = header.planeOffset(0);
    for (loaded.logits) |*logit| {
        logit.* = @bitCast(std.mem.readInt(u32, bytes[offset..][0..4], .little));
        offset += 4;
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

export fn maskCount() u32 {
    return @intCast(masks.count);
}

export fn maskScore(index: u32) f32 {
    return if (index < masks.count) masks.instances[index].score else 0;
}

/// The number the object keeps for as long as it is followed. On a still image
/// it is just the mask's place in the reply.
export fn maskId(index: u32) u32 {
    return if (index < masks.count) masks.instances[index].id else 0;
}

/// One edge of the object's box over the frame, in the order x0, y0, x1, y1.
export fn maskBox(index: u32, edge: u32) f32 {
    if (index >= masks.count or edge >= 4) return 0;
    return masks.instances[index].box[edge];
}

/// The object's colour, packed as 0xRRGGBB so the page can label it to match.
export fn maskColor(index: u32) u32 {
    const color = colorFor(maskId(index));
    return (@as(u32, color.r) << 16) | (@as(u32, color.g) << 8) | color.b;
}

export fn maskCoverage(index: u32) f32 {
    return if (index < masks.count) masks.coverage[index] else 0;
}

export fn objectScore() f32 {
    return masks.object_score;
}

export fn bestMask() u32 {
    var winner: usize = 0;
    for (masks.instances, 0..) |instance, i| {
        if (instance.score > masks.instances[winner].score) winner = i;
    }
    return @intCast(winner);
}

/// Draws the frame with one hypothesis on it, or with none when `index` is
/// negative. This is how a still image is looked at: the decoder's readings of
/// a click are alternatives, so only one of them is true at a time.
export fn render(index: i32) void {
    if (index < 0) return draw(&.{});
    const only: [1]usize = .{@intCast(index)};
    draw(&only);
}

/// Draws the frame with everything on it at once, each object in its own
/// colour and inside its own box. This is how a video frame is looked at: the
/// objects are all there together, and telling them apart is the point.
export fn renderAll() void {
    var chosen: [max_instances]usize = undefined;
    const count = @min(masks.count, max_instances);
    for (0..count) |i| chosen[i] = i;
    draw(chosen[0..count]);
}

/// As many objects as one frame is drawn with. A frame with more than this
/// many on it is unreadable long before it is reached.
const max_instances = 64;

fn draw(chosen: []const usize) void {
    const img = image orelse return;

    var canvas = zigimg.Image.create(gpa, img.width, img.height, .rgb24) catch return;
    defer canvas.deinit(gpa);
    @memcpy(canvas.pixels.rgb24, img.pixels.rgb24);

    for (chosen) |index| {
        if (index >= masks.count) continue;
        const color = colorFor(masks.instances[index].id);
        if (bilinear(
            gpa,
            masks.plane(index),
            masks.width,
            masks.height,
            img.width,
            img.height,
        )) |mask| {
            defer gpa.free(mask);
            overlayMask(&canvas, mask, color, mask_alpha);
        } else |_| {}
        // Only worth outlining when there is more than one object to tell
        // apart; on a single mask the box is just clutter over the tint.
        if (chosen.len > 1) drawBox(&canvas, masks.instances[index].box, color, box_thickness);
    }

    for (points[0..points_len]) |p| drawPointMarker(&canvas, p, marker_radius);

    for (canvas.pixels.rgb24, 0..) |px, i| {
        frame[i * 4 + 0] = px.r;
        frame[i * 4 + 1] = px.g;
        frame[i * 4 + 2] = px.b;
        frame[i * 4 + 3] = 255;
    }
}

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

/// Outlines where an object is, so that objects sharing a colour family or
/// lying on top of one another can still be told apart.
fn drawBox(img: *zigimg.Image, box: [4]f32, color: Rgb24, thickness: usize) void {
    const x0 = @max(0, scaled(box[0], img.width));
    const y0 = @max(0, scaled(box[1], img.height));
    const x1 = @min(@as(isize, @intCast(img.width)), scaled(box[2], img.width));
    const y1 = @min(@as(isize, @intCast(img.height)), scaled(box[3], img.height));
    if (x1 <= x0 or y1 <= y0) return;

    // Four bands rather than a walk over the whole box, each held inside the
    // box so that one thinner than the outline is filled instead of drawn over
    // twice.
    const thick: isize = @intCast(thickness);
    fillRect(img, x0, y0, x1, @min(y1, y0 + thick), color);
    fillRect(img, x0, @max(y0, y1 - thick), x1, y1, color);
    fillRect(img, x0, y0, @min(x1, x0 + thick), y1, color);
    fillRect(img, @max(x0, x1 - thick), y0, x1, y1, color);
}

fn fillRect(img: *zigimg.Image, x0: isize, y0: isize, x1: isize, y1: isize, color: Rgb24) void {
    const width: isize = @intCast(img.width);
    var y = y0;
    while (y < y1) : (y += 1) {
        const row = img.pixels.rgb24[@intCast(y * width)..][@intCast(x0)..@intCast(x1)];
        @memset(row, color);
    }
}

fn scaled(coordinate: f32, extent: usize) isize {
    return @intFromFloat(@round(coordinate * @as(f32, @floatFromInt(extent))));
}

fn drawPointMarker(img: *zigimg.Image, point: Point, radius: usize) void {
    const cx: isize = @intFromFloat(point.x * @as(f32, @floatFromInt(img.width)));
    const cy: isize = @intFromFloat(point.y * @as(f32, @floatFromInt(img.height)));
    const color: Rgb24 = if (point.label == 1)
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

fn testPpm(buffer: []u8, width: usize, height: usize, color: Rgb24) []const u8 {
    var writer = std.Io.Writer.fixed(buffer);
    writer.print("P6\n{d} {d}\n255\n", .{ width, height }) catch unreachable;
    const header_len = writer.end;
    for (0..width * height) |i| {
        buffer[header_len + i * 3 + 0] = color.r;
        buffer[header_len + i * 3 + 1] = color.g;
        buffer[header_len + i * 3 + 2] = color.b;
    }
    return buffer[0 .. header_len + width * height * 3];
}

fn testOpen(image_bytes: []const u8) !void {
    const ptr = alloc(@intCast(image_bytes.len)) orelse return error.OutOfMemory;
    defer dealloc(ptr, @intCast(image_bytes.len));
    @memcpy(ptr[0..image_bytes.len], image_bytes);
    try std.testing.expectEqual(@as(i32, 0), loadImage(ptr, @intCast(image_bytes.len)));
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
    const ppm = testPpm(&buffer, 3, 2, .{ .r = 10, .g = 20, .b = 30 });

    clearPoints();
    try testOpen(ppm);

    try std.testing.expectEqual(@as(u32, 3), imageWidth());
    try std.testing.expectEqual(@as(u32, 2), imageHeight());

    const pixels = framePtr()[0 .. 3 * 2 * 4];
    for (0..6) |i| {
        try std.testing.expectEqualSlices(u8, &.{ 10, 20, 30, 255 }, pixels[i * 4 ..][0..4]);
    }
}

test "a file that is not an image leaves the canvas alone" {
    var buffer: [8192]u8 = undefined;
    const ppm = testPpm(&buffer, 3, 2, .{ .r = 10, .g = 20, .b = 30 });
    try testOpen(ppm);

    const junk = "not an image at all";
    const ptr = alloc(junk.len).?;
    defer dealloc(ptr, junk.len);
    @memcpy(ptr[0..junk.len], junk);

    try std.testing.expectEqual(@as(i32, -1), loadImage(ptr, junk.len));
    try std.testing.expectEqual(@as(u32, 3), imageWidth());
    try std.testing.expectEqual(@as(u32, 2), imageHeight());
}

/// Lays out a reply the way the server does, and hands it to the client.
fn testLoad(
    buffer: []u8,
    width: u32,
    height: u32,
    instances: []const protocol.Instance,
    planes: []const []const f32,
) !i32 {
    const header: protocol.Header = .{
        .count = @intCast(instances.len),
        .width = width,
        .height = height,
        .object_score = 2.5,
    };
    const response = buffer[0..header.responseSize()];
    header.write(response[0..protocol.Header.size]);
    for (instances, 0..) |instance, i| {
        instance.write(response[protocol.Header.instanceOffset(i)..][0..protocol.Instance.size]);
    }
    for (planes, 0..) |plane, i| {
        var offset = header.planeOffset(i);
        for (plane) |logit| {
            std.mem.writeInt(u32, response[offset..][0..4], @bitCast(logit), .little);
            offset += 4;
        }
    }

    const ptr = alloc(@intCast(response.len)) orelse return error.OutOfMemory;
    defer dealloc(ptr, @intCast(response.len));
    @memcpy(ptr[0..response.len], response);
    return loadMasks(ptr, @intCast(response.len));
}

fn testFramePixel(index: usize) [4]u8 {
    return framePtr()[index * 4 ..][0..4].*;
}

test "a response becomes hypotheses, and the chosen one is tinted over the frame" {
    var buffer: [8192]u8 = undefined;
    const ppm = testPpm(&buffer, 1, 1, .{ .r = 200, .g = 10, .b = 10 });

    clearPoints();
    try testOpen(ppm);

    var response: [8192]u8 = undefined;
    const first = [_]f32{-1.0};
    const second = [_]f32{1.0};
    try std.testing.expectEqual(@as(i32, 2), try testLoad(&response, 1, 1, &.{
        .{ .id = 0, .score = 0.25, .box = .{ 0, 0, 0, 0 } },
        .{ .id = 1, .score = 0.75, .box = .{ 0, 0, 1, 1 } },
    }, &.{ &first, &second }));

    try std.testing.expectEqual(@as(u32, 2), maskCount());
    try std.testing.expectEqual(@as(f32, 0.25), maskScore(0));
    try std.testing.expectEqual(@as(u32, 1), maskId(1));
    try std.testing.expectEqual(@as(f32, 1.0), maskBox(1, 2));
    try std.testing.expectEqual(@as(f32, 2.5), objectScore());
    try std.testing.expectEqual(@as(u32, 1), bestMask());
    try std.testing.expectEqual(@as(f32, 0.0), maskCoverage(0));
    try std.testing.expectEqual(@as(f32, 1.0), maskCoverage(1));

    // Half the object's colour laid over the pixel that is in it.
    const tint = colorFor(1);
    render(@intCast(bestMask()));
    try std.testing.expectEqualSlices(u8, &.{
        @intFromFloat(200.0 * 0.5 + @as(f32, @floatFromInt(tint.r)) * 0.5),
        @intFromFloat(10.0 * 0.5 + @as(f32, @floatFromInt(tint.g)) * 0.5),
        @intFromFloat(10.0 * 0.5 + @as(f32, @floatFromInt(tint.b)) * 0.5),
        255,
    }, &testFramePixel(0));

    render(0);
    try std.testing.expectEqualSlices(u8, &.{ 200, 10, 10, 255 }, &testFramePixel(0));
    render(-1);
    try std.testing.expectEqualSlices(u8, &.{ 200, 10, 10, 255 }, &testFramePixel(0));
}

test "every object on a video frame is drawn at once, each in its own colour" {
    const size = 12;
    var buffer: [8192]u8 = undefined;
    const ppm = testPpm(&buffer, size, size, .{ .r = 0, .g = 0, .b = 0 });

    clearPoints();
    try testOpen(ppm);

    // Two objects on one frame: a block in the top-left corner, and another in
    // the bottom-right, each with the box that bounds it.
    var top_left: [size * size]f32 = @splat(-1.0);
    var bottom_right: [size * size]f32 = @splat(-1.0);
    for (0..6) |y| {
        for (0..6) |x| {
            top_left[y * size + x] = 1.0;
            bottom_right[(y + 6) * size + x + 6] = 1.0;
        }
    }

    var response: [8192]u8 = undefined;
    try std.testing.expectEqual(@as(i32, 2), try testLoad(&response, size, size, &.{
        .{ .id = 3, .score = 0.9, .box = .{ 0.0, 0.0, 0.5, 0.5 } },
        .{ .id = 8, .score = 0.8, .box = .{ 0.5, 0.5, 1.0, 1.0 } },
    }, &.{ &top_left, &bottom_right }));

    renderAll();

    // Each object carries the colour its number gives it, and the two of them
    // do not share it. Inside the box the mask is a tint; on the box itself it
    // is the colour outright.
    const three = colorFor(3);
    const eight = colorFor(8);
    try std.testing.expect(!std.meta.eql(three, eight));
    try std.testing.expectEqualSlices(
        u8,
        &.{ three.r / 2, three.g / 2, three.b / 2, 255 },
        &testFramePixel(2 * size + 2),
    );
    try std.testing.expectEqualSlices(u8, &.{ three.r, three.g, three.b, 255 }, &testFramePixel(0));
    try std.testing.expectEqualSlices(
        u8,
        &.{ eight.r / 2, eight.g / 2, eight.b / 2, 255 },
        &testFramePixel(8 * size + 8),
    );
    // Nothing claims the corner neither object is in.
    try std.testing.expectEqualSlices(u8, &.{ 0, 0, 0, 255 }, &testFramePixel(2 * size + 9));

    // And the page can label them to match.
    try std.testing.expectEqual(
        (@as(u32, three.r) << 16) | (@as(u32, three.g) << 8) | three.b,
        maskColor(0),
    );
}

test "an object keeps its colour however many others come and go" {
    // Nothing but the number decides the colour.
    try std.testing.expectEqual(colorFor(12), colorFor(12));
    try std.testing.expect(!std.meta.eql(colorFor(12), colorFor(13)));

    // And every number lands on a colour that is actually visible.
    for (0..512) |id| {
        const color = colorFor(@intCast(id));
        const brightest = @max(color.r, @max(color.g, color.b));
        try std.testing.expect(brightest > 200);
    }
}

test "a response that is not one is refused rather than half read" {
    const truncated = "SAM3" ++ [_]u8{0} ** 8;
    const ptr = alloc(truncated.len).?;
    defer dealloc(ptr, truncated.len);
    @memcpy(ptr[0..truncated.len], truncated);
    try std.testing.expectEqual(@as(i32, -1), loadMasks(ptr, truncated.len));

    const header: protocol.Header = .{ .count = 3, .width = 288, .height = 288, .object_score = 0 };
    var short: [protocol.Header.size]u8 = undefined;
    // A header promising three masks, with none of them behind it.
    header.write(&short);
    const short_ptr = alloc(short.len).?;
    defer dealloc(short_ptr, short.len);
    @memcpy(short_ptr[0..short.len], &short);
    try std.testing.expectEqual(@as(i32, -1), loadMasks(short_ptr, short.len));
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

    try std.testing.expectEqualSlices(f32, &.{ 0.0, 1.0, 3.0, 4.0 }, out);
}

test "downsampling averages towards the middle" {
    const allocator = std.testing.allocator;
    const src = [_]f32{ 0.0, 0.0, 8.0, 8.0 };

    const out = try bilinear(allocator, &src, 4, 1, 2, 1);
    defer allocator.free(out);

    try std.testing.expectEqualSlices(f32, &.{ 0.0, 8.0 }, out);
}
