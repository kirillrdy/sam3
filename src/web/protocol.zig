//! What the server sends back when it has segmented a frame.
//!
//! One definition, compiled into both ends: the server writes it from native
//! code and the client reads it from wasm. Everything is little-endian and
//! unaligned-safe on purpose -- the fields are read out of a byte buffer rather
//! than pointed at -- so the two ends agree without either of them byte
//! swapping, which is true of every target this is built for anyway.
//!
//! A response is
//!
//!     Header
//!     count f32   predicted IoU, one per hypothesis
//!     count * width * height f32   mask logits, plane after plane
//!
//! The logits are the decoder's own 288x288 output, not the frame's resolution:
//! upsampling them is the client's job, so a click costs a quarter of a
//! megabyte over the wire whatever the image is.

const std = @import("std");

pub const magic = [4]u8{ 'S', 'A', 'M', '3' };

pub const Header = extern struct {
    magic: [4]u8 = magic,
    /// How many mask hypotheses the decoder returned; three, in every export
    /// seen so far.
    count: u32,
    width: u32,
    height: u32,
    /// The model's confidence that the click landed on an object at all, as a
    /// logit: positive is present.
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

    /// Bytes a whole response with this header takes.
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
