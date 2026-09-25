const std = @import("std");

// Source Code Pro Regular, 15 px, rasterized into grayscale cells.
pub const font_width: usize = 9;
pub const font_height: usize = 18;
const atlas = @embedFile("font_atlas.bin");

fn blend(background: u32, foreground: u32, alpha: u8) u32 {
    if (alpha == 255) return foreground;
    const a: u32 = alpha;
    const inv = 255 - a;
    const rb = (((background & 0x00ff00ff) * inv + (foreground & 0x00ff00ff) * a + 0x00800080) >> 8) & 0x00ff00ff;
    const g = (((background & 0x0000ff00) * inv + (foreground & 0x0000ff00) * a + 0x00008000) >> 8) & 0x0000ff00;
    return rb | g;
}

pub fn drawChar(pixels: []u32, stride: usize, ch: u8, x: usize, y: usize, color: u32) void {
    if (ch < 32 or ch > 126 or stride == 0) return;
    const glyph_offset = @as(usize, ch - 32) * font_width * font_height;
    for (0..font_height) |dy| {
        const py = y + dy;
        if (py >= pixels.len / stride) break;
        for (0..font_width) |dx| {
            const px = x + dx;
            if (px >= stride) break;
            const alpha = atlas[glyph_offset + dy * font_width + dx];
            if (alpha != 0) {
                const index = py * stride + px;
                pixels[index] = blend(pixels[index], color, alpha);
            }
        }
    }
}

fn nextChar(text: []const u8, index: *usize) u8 {
    const ch = text[index.*];
    if (ch < 128) {
        index.* += 1;
        return if (ch >= 32 and ch <= 126) ch else '?';
    }
    const len: usize = if (ch < 0xe0) 2 else if (ch < 0xf0) 3 else 4;
    const end = @min(text.len, index.* + len);
    const bytes = text[index.*..end];
    index.* = end;
    if (std.mem.eql(u8, bytes, "…")) return '.';
    if (std.mem.eql(u8, bytes, "×")) return 'x';
    if (std.mem.eql(u8, bytes, "“") or std.mem.eql(u8, bytes, "”")) return '"';
    if (std.mem.eql(u8, bytes, "—") or std.mem.eql(u8, bytes, "–")) return '-';
    return '?';
}

pub fn textWidth(value: []const u8) usize {
    var chars: usize = 0;
    var i: usize = 0;
    while (i < value.len) {
        _ = nextChar(value, &i);
        chars += 1;
    }
    return chars * font_width;
}

pub fn drawText(pixels: []u32, stride: usize, value: []const u8, x: usize, y: usize, color: u32) void {
    var cur_x = x;
    var i: usize = 0;
    while (i < value.len) {
        drawChar(pixels, stride, nextChar(value, &i), cur_x, y, color);
        cur_x += font_width;
    }
}

pub fn fillRect(pixels: []u32, stride: usize, x: usize, y: usize, w: usize, h: usize, color: u32) void {
    if (stride == 0) return;
    const rows = pixels.len / stride;
    for (0..h) |dy| {
        const py = y + dy;
        if (py >= rows) break;
        for (0..w) |dx| {
            const px = x + dx;
            if (px >= stride) break;
            pixels[py * stride + px] = color;
        }
    }
}

pub fn strokeRect(pixels: []u32, stride: usize, x: usize, y: usize, w: usize, h: usize, color: u32) void {
    if (w == 0 or h == 0) return;
    fillRect(pixels, stride, x, y, w, 1, color);
    fillRect(pixels, stride, x, y + h - 1, w, 1, color);
    fillRect(pixels, stride, x, y, 1, h, color);
    fillRect(pixels, stride, x + w - 1, y, 1, h, color);
}

pub fn drawButton(
    pixels: []u32,
    stride: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    value: []const u8,
    is_hover: bool,
    is_active: bool,
    accent_color: u32,
) void {
    const bg_color: u32 = if (is_active) 0x002c3038 else if (is_hover) 0x00262a32 else 0x001c1f25;
    const border_color: u32 = if (is_active) accent_color else if (is_hover) 0x005c6068 else 0x002c3038;
    const text_color: u32 = if (is_active) accent_color else 0x00e6e8ec;

    fillRect(pixels, stride, x, y, w, h, bg_color);
    strokeRect(pixels, stride, x, y, w, h, border_color);

    const text_w = textWidth(value);
    const text_x = if (w > text_w) x + (w - text_w) / 2 else x + 4;
    const text_y = if (h > font_height) y + (h - font_height) / 2 else y + 2;
    drawText(pixels, stride, value, text_x, text_y, text_color);
}

test "font blends glyphs over the background" {
    var pixels: [100 * 50]u32 = undefined;
    @memset(&pixels, 0x001c1f25);
    drawText(&pixels, 100, "SAM 3", 10, 10, 0x00ffffff);
    var changed = false;
    for (pixels) |p| {
        if (p != 0x001c1f25) changed = true;
    }
    try std.testing.expect(changed);
    try std.testing.expectEqual(@as(usize, 45), textWidth("SAM 3"));
}

test "drawButton renders background and border" {
    var pixels: [100 * 50]u32 = undefined;
    @memset(&pixels, 0);
    drawButton(&pixels, 100, 10, 10, 80, 30, "Test", false, true, 0x0000dc64);
    try std.testing.expectEqual(@as(u32, 0x0000dc64), pixels[10 * 100 + 10]);
}
