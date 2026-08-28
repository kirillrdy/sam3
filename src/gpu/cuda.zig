//! Image preprocessing on an NVIDIA GPU, through the CUDA driver API.
//!
//! The whole stage is one kernel: the source image goes up as interleaved
//! RGB8 (a megabyte or so), the GPU resamples and normalizes all three planes,
//! and only the finished tensor comes back. Selected with -Dcuda; the build
//! swaps in `disabled.zig` otherwise, so this file is the only place in sam3
//! that knows CUDA exists.

const std = @import("std");
const cuda = @import("cuda");

const ptx = @embedFile("kernels.ptx");

pub const available = true;

/// Message for the most recent failure.
pub fn lastError() []const u8 {
    return cuda.lastError();
}

pub const Preprocessor = struct {
    context: cuda.Context,
    module: cuda.Module,
    resample_plane: cuda.Function,

    size: usize,
    source: cuda.Buffer(u8),
    output: cuda.Buffer(f32),

    /// Sets up the GPU for square `size` x `size` output planes.
    pub fn init(size: usize) !Preprocessor {
        try cuda.init();

        const context = try cuda.Context.init(0);
        errdefer context.deinit();

        const module = try cuda.Module.load(ptx);
        errdefer module.unload();

        const output = try cuda.Buffer(f32).alloc(3 * size * size);
        errdefer output.free();

        return .{
            .context = context,
            .module = module,
            .resample_plane = try module.function("resamplePlane"),
            .size = size,
            .source = .{ .ptr = 0, .len = 0 },
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
    /// image. Takes the image apart rather than a sam3 type: this module is
    /// compiled on its own, so a file of sam3 cannot also belong to it.
    pub fn run(
        self: *Preprocessor,
        pixels: []const u8,
        width: usize,
        height: usize,
        mean: [3]f32,
        deviation: [3]f32,
        out: []f32,
    ) !void {
        // The web server handles inference on worker threads. CUDA context
        // selection is thread-local, so initialization on the main thread is
        // not enough.
        try self.context.makeCurrent();

        const plane = self.size * self.size;
        std.debug.assert(out.len == 3 * plane);

        try self.ensureSource(pixels.len);
        try self.source.upload(pixels);

        const block = 256;
        const grid: cuda.Dim = .{ .x = @intCast((plane + block - 1) / block) };
        const size: u32 = @intCast(self.size);
        for (0..3) |channel| {
            try self.resample_plane.launch(grid, .{ .x = block }, .{
                self.source.ptr,
                @as(u32, @intCast(width)),
                @as(u32, @intCast(height)),
                @as(u32, @intCast(channel)),
                self.output.slice(channel * plane, plane).ptr,
                size,
                size,
                mean[channel],
                deviation[channel],
            });
        }

        // Launches are asynchronous; a kernel fault surfaces here.
        try self.context.synchronize();
        try self.output.download(out);
    }

    /// The source buffer follows whatever image arrives, and images keep their
    /// size across a session, so it is reused rather than freed each frame.
    fn ensureSource(self: *Preprocessor, bytes: usize) !void {
        if (self.source.len >= bytes) return;
        if (self.source.len != 0) self.source.free();
        self.source = try cuda.Buffer(u8).alloc(bytes);
    }
};
