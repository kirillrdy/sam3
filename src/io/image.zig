const std = @import("std");
const zigimg = @import("zigimg");

pub const RGB = struct {
    r: u8,
    g: u8,
    b: u8,
};

pub const ImageRGB = struct {
    width: usize,
    height: usize,
    data: []u8,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, width: usize, height: usize) !ImageRGB {
        const data = try allocator.alloc(u8, width * height * 3);
        @memset(data, 0);
        return ImageRGB{
            .width = width,
            .height = height,
            .data = data,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ImageRGB) void {
        self.allocator.free(self.data);
    }

    pub inline fn setPixel(self: *ImageRGB, x: usize, y: usize, color: RGB) void {
        if (x >= self.width or y >= self.height) return;
        const idx = (y * self.width + x) * 3;
        self.data[idx] = color.r;
        self.data[idx + 1] = color.g;
        self.data[idx + 2] = color.b;
    }
};

pub fn decode(allocator: std.mem.Allocator, bytes: []const u8) !ImageRGB {
    var decoded = try zigimg.Image.fromMemory(allocator, bytes);
    defer decoded.deinit(allocator);

    try decoded.convert(allocator, .rgb24);

    var out = try ImageRGB.init(allocator, decoded.width, decoded.height);
    errdefer out.deinit();

    for (decoded.pixels.rgb24, 0..) |px, i| {
        out.data[i * 3] = px.r;
        out.data[i * 3 + 1] = px.g;
        out.data[i * 3 + 2] = px.b;
    }
    return out;
}

test "decodes a PNG round-tripped through zigimg" {
    const allocator = std.testing.allocator;

    var source = try zigimg.Image.create(allocator, 4, 3, .rgb24);
    defer source.deinit(allocator);

    for (source.pixels.rgb24, 0..) |*px, i| {
        px.* = .{ .r = @intCast(i * 5), .g = @intCast(255 - i * 5), .b = @intCast(i * 2) };
    }

    var encode_buf: [8192]u8 = undefined;
    const png = try source.writeToMemory(allocator, &encode_buf, .{ .png = .{} });

    var img = try decode(allocator, png);
    defer img.deinit();

    try std.testing.expectEqual(@as(usize, 4), img.width);
    try std.testing.expectEqual(@as(usize, 3), img.height);
    for (source.pixels.rgb24, 0..) |px, i| {
        try std.testing.expectEqual(px.r, img.data[i * 3]);
        try std.testing.expectEqual(px.g, img.data[i * 3 + 1]);
        try std.testing.expectEqual(px.b, img.data[i * 3 + 2]);
    }
}

test "rejects data that is not an image" {
    const allocator = std.testing.allocator;

    try std.testing.expect(std.meta.isError(decode(allocator, "not an image at all")));
}
