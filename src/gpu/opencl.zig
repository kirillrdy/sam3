//! Image preprocessing on an Intel Arc / OpenCL GPU.
//!
//! The whole stage is one kernel: the source image goes up as interleaved
//! RGB8 (a megabyte or so), the GPU resamples and normalizes all three planes,
//! and only the finished tensor comes back.

const std = @import("std");
const opencl = @import("opencl");

const kernels_src =
    \\inline uint clampCoord(float coordinate, uint limit) {
    \\    if (coordinate <= 0.0f) return 0;
    \\    uint floored = (uint)floor(coordinate);
    \\    return (floored < limit - 1) ? floored : (limit - 1);
    \\}
    \\
    \\inline float samplePixel(__global const uchar* src, uint src_w, uint channel, uint x, uint y) {
    \\    uchar val = src[(y * src_w + x) * 3 + channel];
    \\    return (float)val / 255.0f;
    \\}
    \\
    \\__kernel void resamplePlane(
    \\    __global const uchar* src,
    \\    uint src_w,
    \\    uint src_h,
    \\    uint channel,
    \\    __global float* out,
    \\    uint dst_offset,
    \\    uint dst_w,
    \\    uint dst_h,
    \\    float mean,
    \\    float deviation
    \\) {
    \\    uint i = get_global_id(0);
    \\    if (i >= dst_w * dst_h) return;
    \\
    \\    uint x = i % dst_w;
    \\    uint y = i / dst_w;
    \\
    \\    float ratio_y = (float)src_h / (float)dst_h;
    \\    float ratio_x = (float)src_w / (float)dst_w;
    \\
    \\    float in_y = ratio_y * ((float)y + 0.5f) - 0.5f;
    \\    uint y0 = clampCoord(in_y, src_h);
    \\    uint y1 = (y0 + 1 < src_h) ? (y0 + 1) : (src_h - 1);
    \\    float wy = fmax(0.0f, in_y - (float)y0);
    \\
    \\    float in_x = ratio_x * ((float)x + 0.5f) - 0.5f;
    \\    uint x0 = clampCoord(in_x, src_w);
    \\    uint x1 = (x0 + 1 < src_w) ? (x0 + 1) : (src_w - 1);
    \\    float wx = fmax(0.0f, in_x - (float)x0);
    \\
    \\    float top_left = samplePixel(src, src_w, channel, x0, y0);
    \\    float top_right = samplePixel(src, src_w, channel, x1, y0);
    \\    float bottom_left = samplePixel(src, src_w, channel, x0, y1);
    \\    float bottom_right = samplePixel(src, src_w, channel, x1, y1);
    \\
    \\    float top = top_left + (top_right - top_left) * wx;
    \\    float bottom = bottom_left + (bottom_right - bottom_left) * wx;
    \\
    \\    out[dst_offset + i] = (top + (bottom - top) * wy - mean) / deviation;
    \\}
;

pub const available = true;

/// Message for the most recent failure.
pub fn lastError() []const u8 {
    return opencl.lastError();
}

pub const Preprocessor = struct {
    context: opencl.Context,
    module: opencl.Module,
    resample_plane: opencl.Function,

    size: usize,
    source: opencl.Buffer(u8),
    output: opencl.Buffer(f32),

    /// Sets up the GPU for square `size` x `size` output planes.
    pub fn init(size: usize) !Preprocessor {
        try opencl.init();

        const context = try opencl.Context.init(0);
        errdefer context.deinit();

        const module = try opencl.Module.loadWithContext(context, kernels_src);
        errdefer module.unload();

        const output = try opencl.Buffer(f32).allocWithContext(context, 3 * size * size);
        errdefer output.free();

        return .{
            .context = context,
            .module = module,
            .resample_plane = try module.function("resamplePlane"),
            .size = size,
            .source = .{ .ptr = null, .len = 0, .context = context },
            .output = output,
        };
    }

    pub fn deinit(self: *Preprocessor) void {
        if (self.source.len != 0) self.source.free();
        self.output.free();
        self.module.unload();
        self.context.deinit();
        self.* = undefined;
    }

    pub fn deviceName(self: *Preprocessor, buf: []u8) ![]const u8 {
        return self.context.name(buf);
    }

    /// Fills `out` with the normalized CHW tensor for one interleaved RGB8
    /// image.
    pub fn run(
        self: *Preprocessor,
        pixels: []const u8,
        width: usize,
        height: usize,
        mean: [3]f32,
        deviation: [3]f32,
        out: []f32,
    ) !void {
        try self.context.makeCurrent();

        const plane = self.size * self.size;
        std.debug.assert(out.len == 3 * plane);

        try self.ensureSource(pixels.len);
        try self.source.upload(pixels);

        const block = 256;
        const grid: opencl.Dim = .{ .x = @intCast((plane + block - 1) / block) };
        const size: u32 = @intCast(self.size);
        for (0..3) |channel| {
            const dst_offset: u32 = @intCast(channel * plane);
            try self.resample_plane.launch(grid, .{ .x = block }, .{
                self.source.ptr,
                @as(u32, @intCast(width)),
                @as(u32, @intCast(height)),
                @as(u32, @intCast(channel)),
                self.output.ptr,
                dst_offset,
                size,
                size,
                mean[channel],
                deviation[channel],
            });
        }

        try self.context.synchronize();
        try self.output.download(out);
    }

    fn ensureSource(self: *Preprocessor, bytes: usize) !void {
        if (self.source.len >= bytes) return;
        if (self.source.len != 0) self.source.free();
        self.source = try opencl.Buffer(u8).allocWithContext(self.context, bytes);
    }
};
