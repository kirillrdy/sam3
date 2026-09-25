// Tool to extract 24x24 Adwaita cursors into linux/adwaita_cursors.bin
// Run: zig run -lc linux/generate_cursors.zig
const std = @import("std");

extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn read(fd: c_int, buf: [*]u8, count: usize) isize;
extern "c" fn close(fd: c_int) c_int;

const CursorDef = struct {
    width: u32,
    height: u32,
    hotspot_x: i32,
    hotspot_y: i32,
    pixels: [24 * 24]u32,
};

fn findCursorFile(names: []const []const u8) ?[*:0]const u8 {
    const search_dirs = [_][]const u8{
        "/run/current-system/sw/share/icons/Adwaita/cursors",
        "/usr/share/icons/Adwaita/cursors",
        "/usr/local/share/icons/Adwaita/cursors",
        "/usr/share/icons/default/cursors",
    };
    var path_buf: [512]u8 = undefined;
    for (search_dirs) |dir| {
        for (names) |name| {
            const path = std.fmt.bufPrintZ(&path_buf, "{s}/{s}", .{ dir, name }) catch continue;
            const fd = open(path, 0);
            if (fd >= 0) {
                _ = close(fd);
                return path;
            }
        }
    }
    return null;
}

fn loadCursor(names: []const []const u8) !CursorDef {
    const path = findCursorFile(names) orelse return error.NotFound;
    const fd = open(path, 0);
    if (fd < 0) return error.OpenFailed;
    defer _ = close(fd);

    var buf: [128 * 1024]u8 = undefined;
    var total: usize = 0;
    while (total < buf.len) {
        const rc = read(fd, buf[total..].ptr, buf.len - total);
        if (rc <= 0) break;
        total += @intCast(rc);
    }
    const data = buf[0..total];
    if (!std.mem.startsWith(u8, data, "Xcur")) return error.InvalidMagic;
    const ntoc = std.mem.readInt(u32, data[12..16], .little);
    for (0..ntoc) |i| {
        const offset = 16 + i * 12;
        const c_type = std.mem.readInt(u32, data[offset..][0..4], .little);
        const c_size = std.mem.readInt(u32, data[offset + 4 ..][0..4], .little);
        const c_pos = std.mem.readInt(u32, data[offset + 8 ..][0..4], .little);
        if (c_type == 0xfffd0002 and c_size == 24) {
            const chunk = data[c_pos..];
            const w = std.mem.readInt(u32, chunk[16..20], .little);
            const h = std.mem.readInt(u32, chunk[20..24], .little);
            const xhot = std.mem.readInt(u32, chunk[24..28], .little);
            const yhot = std.mem.readInt(u32, chunk[28..32], .little);
            if (w != 24 or h != 24) return error.UnexpectedDimensions;
            var def = CursorDef{
                .width = w,
                .height = h,
                .hotspot_x = @intCast(xhot),
                .hotspot_y = @intCast(yhot),
                .pixels = undefined,
            };
            const pixel_bytes = chunk[36 .. 36 + 24 * 24 * 4];
            for (0..24 * 24) |p| {
                def.pixels[p] = std.mem.readInt(u32, pixel_bytes[p * 4 ..][0..4], .little);
            }
            return def;
        }
    }
    return error.NotFound;
}

pub fn main() !void {
    const arrow = try loadCursor(&.{ "default", "left_ptr" });
    const text = try loadCursor(&.{ "text", "xterm" });
    const crosshair = try loadCursor(&.{ "crosshair", "cross" });
    const resize_ns = try loadCursor(&.{ "ns-resize", "n-resize", "s-resize" });
    const resize_ew = try loadCursor(&.{ "ew-resize", "e-resize", "w-resize" });
    const resize_nwse = try loadCursor(&.{ "nwse-resize", "nw-resize", "se-resize" });
    const resize_nesw = try loadCursor(&.{ "nesw-resize", "ne-resize", "sw-resize" });

    const out_file = std.c.fopen("linux/adwaita_cursors.bin", "wb") orelse return error.CantCreateOut;
    defer _ = std.c.fclose(out_file);

    const cursors = [_]CursorDef{ arrow, text, crosshair, resize_ns, resize_ew, resize_nwse, resize_nesw };
    for (cursors) |c| {
        const header: [4]u32 = .{ c.width, c.height, @bitCast(c.hotspot_x), @bitCast(c.hotspot_y) };
        _ = std.c.fwrite(@ptrCast(&header), 4, 4, out_file);
        _ = std.c.fwrite(@ptrCast(&c.pixels), 4, c.pixels.len, out_file);
    }
    std.debug.print("Successfully updated linux/adwaita_cursors.bin with 7 cursors!\n", .{});
}
