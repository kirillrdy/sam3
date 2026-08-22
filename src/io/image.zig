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
    data: []u8, // 3 * width * height
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

    pub inline fn getPixel(self: ImageRGB, x: usize, y: usize) RGB {
        const idx = (y * self.width + x) * 3;
        return RGB{
            .r = self.data[idx],
            .g = self.data[idx + 1],
            .b = self.data[idx + 2],
        };
    }

    pub inline fn setPixel(self: *ImageRGB, x: usize, y: usize, color: RGB) void {
        if (x >= self.width or y >= self.height) return;
        const idx = (y * self.width + x) * 3;
        self.data[idx] = color.r;
        self.data[idx + 1] = color.g;
        self.data[idx + 2] = color.b;
    }

    pub fn savePPM(self: ImageRGB, file_path: []const u8) !void {
        var path_z: [1024]u8 = undefined;
        if (file_path.len >= path_z.len) return error.PathTooLong;
        @memcpy(path_z[0..file_path.len], file_path);
        path_z[file_path.len] = 0;
        const p_slice: [:0]const u8 = path_z[0..file_path.len :0];

        const fd_res = std.os.linux.open(p_slice, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, 0o644);
        const signed: isize = @bitCast(fd_res);
        if (signed < 0) return error.FileCreationFailed;
        const fd: i32 = @intCast(signed);
        defer _ = std.os.linux.close(fd);

        var header_buf: [128]u8 = undefined;
        const header = try std.fmt.bufPrint(&header_buf, "P6\n{d} {d}\n255\n", .{ self.width, self.height });
        _ = std.os.linux.write(fd, header.ptr, header.len);
        _ = std.os.linux.write(fd, self.data.ptr, self.data.len);
    }

    pub fn loadPPM(allocator: std.mem.Allocator, file_path: []const u8) !ImageRGB {
        var path_z: [1024]u8 = undefined;
        if (file_path.len >= path_z.len) return error.PathTooLong;
        @memcpy(path_z[0..file_path.len], file_path);
        path_z[file_path.len] = 0;
        const p_slice: [:0]const u8 = path_z[0..file_path.len :0];

        const fd_res = std.os.linux.open(p_slice, .{ .ACCMODE = .RDONLY }, 0);
        const signed: isize = @bitCast(fd_res);
        if (signed < 0) return error.FileNotFound;
        const fd: i32 = @intCast(signed);
        defer _ = std.os.linux.close(fd);

        const size = std.os.linux.lseek(fd, 0, 2);
        _ = std.os.linux.lseek(fd, 0, 0);

        const buf = try allocator.alloc(u8, size);
        defer allocator.free(buf);

        _ = std.os.linux.read(fd, buf.ptr, buf.len);

        var idx: usize = 0;
        while (idx < buf.len and (buf[idx] == ' ' or buf[idx] == '\n' or buf[idx] == '\r')) : (idx += 1) {}
        if (idx + 2 > buf.len or buf[idx] != 'P' or buf[idx + 1] != '6') {
            return error.InvalidPPMHeader;
        }
        idx += 2;

        var in_comment = false;
        var header_tokens: [3]usize = undefined;
        var token_idx: usize = 0;

        while (idx < buf.len and token_idx < 3) {
            const c = buf[idx];
            if (c == '#') {
                in_comment = true;
                idx += 1;
                continue;
            }
            if (in_comment) {
                if (c == '\n' or c == '\r') in_comment = false;
                idx += 1;
                continue;
            }
            if (std.ascii.isWhitespace(c)) {
                idx += 1;
                continue;
            }
            var val: usize = 0;
            while (idx < buf.len and std.ascii.isDigit(buf[idx])) : (idx += 1) {
                val = val * 10 + (buf[idx] - '0');
            }
            header_tokens[token_idx] = val;
            token_idx += 1;
        }

        if (token_idx < 3) return error.InvalidPPMHeader;
        const w = header_tokens[0];
        const h = header_tokens[1];

        if (idx < buf.len and std.ascii.isWhitespace(buf[idx])) idx += 1;

        var img = try ImageRGB.init(allocator, w, h);
        errdefer img.deinit();

        const data_len = w * h * 3;
        if (idx + data_len > buf.len) return error.UnexpectedEOF;
        @memcpy(img.data, buf[idx .. idx + data_len]);

        return img;
    }

    pub fn saveBMP(self: ImageRGB, file_path: []const u8) !void {
        var path_z: [1024]u8 = undefined;
        if (file_path.len >= path_z.len) return error.PathTooLong;
        @memcpy(path_z[0..file_path.len], file_path);
        path_z[file_path.len] = 0;
        const p_slice: [:0]const u8 = path_z[0..file_path.len :0];

        const fd_res = std.os.linux.open(p_slice, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, 0o644);
        const signed: isize = @bitCast(fd_res);
        if (signed < 0) return error.FileCreationFailed;
        const fd: i32 = @intCast(signed);
        defer _ = std.os.linux.close(fd);

        const row_stride = (self.width * 3 + 3) & ~@as(usize, 3);
        const img_size: u32 = @intCast(row_stride * self.height);
        const file_size: u32 = 54 + img_size;

        // Allocate whole file buffer in memory for a single fast buffered write
        var file_buf = try self.allocator.alloc(u8, file_size);
        defer self.allocator.free(file_buf);
        @memset(file_buf, 0);

        file_buf[0] = 'B';
        file_buf[1] = 'M';
        std.mem.writeInt(u32, file_buf[2..6], file_size, .little);
        std.mem.writeInt(u32, file_buf[6..10], 0, .little);
        std.mem.writeInt(u32, file_buf[10..14], 54, .little);
        std.mem.writeInt(u32, file_buf[14..18], 40, .little);
        std.mem.writeInt(i32, file_buf[18..22], @intCast(self.width), .little);
        std.mem.writeInt(i32, file_buf[22..26], @intCast(self.height), .little);
        std.mem.writeInt(u16, file_buf[26..28], 1, .little);
        std.mem.writeInt(u16, file_buf[28..30], 24, .little);
        std.mem.writeInt(u32, file_buf[30..34], 0, .little);
        std.mem.writeInt(u32, file_buf[34..38], img_size, .little);
        std.mem.writeInt(i32, file_buf[38..42], 2835, .little);
        std.mem.writeInt(i32, file_buf[42..46], 2835, .little);
        std.mem.writeInt(u32, file_buf[46..50], 0, .little);
        std.mem.writeInt(u32, file_buf[50..54], 0, .little);

        var dst_offset: usize = 54;
        var y: usize = self.height;

        while (y > 0) {
            y -= 1;
            const src_row_offset = y * self.width * 3;
            for (0..self.width) |x| {
                const src_idx = src_row_offset + x * 3;
                const pr = self.data[src_idx];
                const pg = self.data[src_idx + 1];
                const pb = self.data[src_idx + 2];

                file_buf[dst_offset] = pb;
                file_buf[dst_offset + 1] = pg;
                file_buf[dst_offset + 2] = pr;
                dst_offset += 3;
            }
            const padding = row_stride - self.width * 3;
            dst_offset += padding;
        }

        // Single atomic syscall write for entire BMP file
        _ = std.os.linux.write(fd, file_buf.ptr, file_buf.len);
    }
};

/// Reads an image file. PPM goes through the loader in this file; everything
/// else (PNG, JPEG, BMP, ...) is decoded by zigimg.
pub fn load(allocator: std.mem.Allocator, file_path: []const u8) !ImageRGB {
    if (std.mem.endsWith(u8, file_path, ".ppm")) return ImageRGB.loadPPM(allocator, file_path);

    const bytes = try readFileAlloc(allocator, file_path);
    defer allocator.free(bytes);
    return decode(allocator, bytes);
}

fn readFileAlloc(allocator: std.mem.Allocator, file_path: []const u8) ![]u8 {
    var path_z: [1024]u8 = undefined;
    if (file_path.len >= path_z.len) return error.PathTooLong;
    @memcpy(path_z[0..file_path.len], file_path);
    path_z[file_path.len] = 0;
    const p_slice: [:0]const u8 = path_z[0..file_path.len :0];

    const fd_res = std.os.linux.open(p_slice, .{ .ACCMODE = .RDONLY }, 0);
    const signed: isize = @bitCast(fd_res);
    if (signed < 0) return error.FileNotFound;
    const fd: i32 = @intCast(signed);
    defer _ = std.os.linux.close(fd);

    const size = std.os.linux.lseek(fd, 0, 2);
    _ = std.os.linux.lseek(fd, 0, 0);

    const buf = try allocator.alloc(u8, size);
    errdefer allocator.free(buf);

    var read_total: usize = 0;
    while (read_total < buf.len) {
        const n = std.os.linux.read(fd, buf.ptr + read_total, buf.len - read_total);
        if (n == 0) break;
        read_total += n;
    }
    return buf[0..read_total];
}
/// Decodes any format zigimg understands (PNG, JPEG, BMP, TGA, QOI, GIF, ...)
/// and flattens it to 8-bit RGB.
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
    // Which error depends on how far zigimg's format sniffing gets before it
    // runs out of bytes, and that differs between build modes. What matters
    // here is that nothing is decoded.
    try std.testing.expect(std.meta.isError(decode(allocator, "not an image at all")));
}
