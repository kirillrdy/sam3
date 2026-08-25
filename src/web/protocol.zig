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
        const scores = @as(usize, header.count) * @sizeOf(f32);
        const planes = @as(usize, header.count) * header.width * header.height * @sizeOf(f32);
        return size + scores + planes;
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
    try std.testing.expectEqual(@as(usize, 20 + 12 + 3 * 288 * 288 * 4), read.responseSize());
}

test "anything that is not a response is rejected" {
    try std.testing.expectError(error.Truncated, Header.parse("SAM3"));
    try std.testing.expectError(error.BadMagic, Header.parse("not a response at all"));
}
