//! Stands in for `cuda.zig` in a build without -Dcuda, so that nothing else in
//! sam3 has to know whether the CUDA path was compiled in.

pub const available = false;

pub fn lastError() []const u8 {
    return "this build has no CUDA support; build with -Dcuda";
}

pub const Preprocessor = struct {
    pub fn init(size: usize) !Preprocessor {
        _ = size;
        return error.CudaUnavailable;
    }

    pub fn deinit(self: *Preprocessor) void {
        _ = self;
    }

    pub fn deviceName(self: *Preprocessor, buf: []u8) ![]const u8 {
        _ = self;
        return buf[0..0];
    }

    pub fn run(
        self: *Preprocessor,
        pixels: []const u8,
        width: usize,
        height: usize,
        mean: [3]f32,
        deviation: [3]f32,
        out: []f32,
    ) !void {
        _ = .{ self, pixels, width, height, mean, deviation, out };
        return error.CudaUnavailable;
    }
};
