//! Device code for the image preprocessing that runs before the vision
//! encoder. Compiled for `nvptx64-cuda` and embedded in the binary as PTX.

const gpu = @import("gpu");

pub const panic = gpu.panic;

/// One output pixel per thread: samples the interleaved RGB8 source
/// bilinearly and writes the normalized plane the encoder expects.
///
/// This mirrors `resample.bilinear` followed by the normalize step in
/// `sam3.preprocessCpu`, expression for expression, so the two paths agree to
/// within the fused multiply-adds ptxas contracts on its own.
pub fn resamplePlane(
    src: [*]addrspace(.global) const u8,
    src_w: u32,
    src_h: u32,
    channel: u32,
    out: [*]addrspace(.global) f32,
    dst_w: u32,
    dst_h: u32,
    mean: f32,
    deviation: f32,
) callconv(.kernel) void {
    const i = gpu.globalIndex();
    if (i >= dst_w * dst_h) return;

    const x = i % dst_w;
    const y = i / dst_w;

    const ratio_y = @as(f32, @floatFromInt(src_h)) / @as(f32, @floatFromInt(dst_h));
    const ratio_x = @as(f32, @floatFromInt(src_w)) / @as(f32, @floatFromInt(dst_w));

    const in_y = ratio_y * (@as(f32, @floatFromInt(y)) + 0.5) - 0.5;
    const y0 = clampIndex(in_y, src_h);
    const y1 = @min(y0 + 1, src_h - 1);
    const wy = @max(0.0, in_y - @as(f32, @floatFromInt(y0)));

    const in_x = ratio_x * (@as(f32, @floatFromInt(x)) + 0.5) - 0.5;
    const x0 = clampIndex(in_x, src_w);
    const x1 = @min(x0 + 1, src_w - 1);
    const wx = @max(0.0, in_x - @as(f32, @floatFromInt(x0)));

    const top_left = sample(src, src_w, channel, x0, y0);
    const top_right = sample(src, src_w, channel, x1, y0);
    const bottom_left = sample(src, src_w, channel, x0, y1);
    const bottom_right = sample(src, src_w, channel, x1, y1);

    const top = top_left + (top_right - top_left) * wx;
    const bottom = bottom_left + (bottom_right - bottom_left) * wx;

    out[i] = (top + (bottom - top) * wy - mean) / deviation;
}

fn sample(src: [*]addrspace(.global) const u8, src_w: u32, channel: u32, x: u32, y: u32) f32 {
    const value = src[(y * src_w + x) * 3 + channel];
    return @as(f32, @floatFromInt(value)) / 255.0;
}

fn clampIndex(coordinate: f32, limit: u32) u32 {
    if (coordinate <= 0.0) return 0;
    const floored: u32 = @intFromFloat(@floor(coordinate));
    return @min(floored, limit - 1);
}

/// Keeps the kernels alive; see the note in cuda/src/kernels.zig.
export fn anchor() usize {
    return @intFromPtr(&resamplePlane);
}
