const std = @import("std");

pub const magic = [4]u8{ 'S', 'A', 'M', '3' };

pub const Header = extern struct {
    magic: [4]u8 = magic,

    count: u32,
    width: u32,
    height: u32,

    object_score: f32,

    pub const size = 20;

    comptime {
        std.debug.assert(@sizeOf(Header) == size);
    }

    pub fn write(header: Header, buffer: *[size]u8) void {
        buffer[0..4].* = header.magic;
        std.mem.writeInt(u32, buffer[4..8], header.count, .little);
        std.mem.writeInt(u32, buffer[8..12], header.width, .little);
        std.mem.writeInt(u32, buffer[12..16], header.height, .little);
        std.mem.writeInt(u32, buffer[16..20], @bitCast(header.object_score), .little);
    }

    pub fn parse(bytes: []const u8) error{ Truncated, BadMagic }!Header {
        if (bytes.len < size) return error.Truncated;
        if (!std.mem.eql(u8, bytes[0..4], &magic)) return error.BadMagic;
        return .{
            .magic = bytes[0..4].*,
            .count = std.mem.readInt(u32, bytes[4..8], .little),
            .width = std.mem.readInt(u32, bytes[8..12], .little),
            .height = std.mem.readInt(u32, bytes[12..16], .little),
            .object_score = @bitCast(std.mem.readInt(u32, bytes[16..20], .little)),
        };
    }

    pub fn responseSize(header: Header) usize {
        const instances = @as(usize, header.count) * Instance.size;
        const planes = @as(usize, header.count) * header.width * header.height * @sizeOf(f32);
        return size + instances + planes;
    }

    /// Where one instance record sits, and where its mask plane does.
    pub fn instanceOffset(index: usize) usize {
        return size + index * Instance.size;
    }

    pub fn planeOffset(header: Header, index: usize) usize {
        const stride = @as(usize, header.width) * header.height * @sizeOf(f32);
        return size + @as(usize, header.count) * Instance.size + index * stride;
    }
};

/// What the server found, one record per mask plane that follows.
///
/// The id is what makes a video a video rather than a run of unrelated frames:
/// it is the same number on every frame the object is on. On a still image
/// there is nothing to follow, so it is just the mask's position in the reply.
pub const Instance = struct {
    id: u32 = 0,
    score: f32 = 0,
    /// Corners over the frame, which is the unit square: x0, y0, x1, y1.
    box: [4]f32 = @splat(0),

    pub const size = 24;

    pub fn write(instance: Instance, buffer: *[size]u8) void {
        std.mem.writeInt(u32, buffer[0..4], instance.id, .little);
        std.mem.writeInt(u32, buffer[4..8], @bitCast(instance.score), .little);
        for (instance.box, 0..) |edge, i| {
            std.mem.writeInt(u32, buffer[8 + i * 4 ..][0..4], @bitCast(edge), .little);
        }
    }

    pub fn parse(bytes: *const [size]u8) Instance {
        var instance: Instance = .{
            .id = std.mem.readInt(u32, bytes[0..4], .little),
            .score = @bitCast(std.mem.readInt(u32, bytes[4..8], .little)),
        };
        for (&instance.box, 0..) |*edge, i| {
            edge.* = @bitCast(std.mem.readInt(u32, bytes[8 + i * 4 ..][0..4], .little));
        }
        return instance;
    }
};

test "a header survives the round trip" {
    var buffer: [Header.size]u8 = undefined;
    const written: Header = .{ .count = 3, .width = 288, .height = 288, .object_score = -1.25 };
    written.write(&buffer);

    const read = try Header.parse(&buffer);
    try std.testing.expectEqual(written.count, read.count);
    try std.testing.expectEqual(written.width, read.width);
    try std.testing.expectEqual(written.height, read.height);
    try std.testing.expectEqual(written.object_score, read.object_score);
    try std.testing.expectEqual(@as(usize, 20 + 3 * 24 + 3 * 288 * 288 * 4), read.responseSize());
}

test "an instance survives the round trip" {
    var buffer: [Instance.size]u8 = undefined;
    const written: Instance = .{ .id = 7, .score = 0.75, .box = .{ 0.1, 0.2, 0.3, 0.4 } };
    written.write(&buffer);

    const read = Instance.parse(&buffer);
    try std.testing.expectEqual(written.id, read.id);
    try std.testing.expectEqual(written.score, read.score);
    try std.testing.expectEqualSlices(f32, &written.box, &read.box);
}

test "the records come before the planes, and each has its own place" {
    const header: Header = .{ .count = 2, .width = 4, .height = 2, .object_score = 0 };

    try std.testing.expectEqual(@as(usize, 20), Header.instanceOffset(0));
    try std.testing.expectEqual(@as(usize, 44), Header.instanceOffset(1));
    try std.testing.expectEqual(@as(usize, 68), header.planeOffset(0));
    try std.testing.expectEqual(@as(usize, 100), header.planeOffset(1));
    try std.testing.expectEqual(@as(usize, 132), header.responseSize());
}

test "anything that is not a response is rejected" {
    try std.testing.expectError(error.Truncated, Header.parse("SAM3"));
    try std.testing.expectError(error.BadMagic, Header.parse("not a response at all"));
}
