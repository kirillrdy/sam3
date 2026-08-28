//! Native CUDA execution context for the SAM 3 graph executor. This talks to
//! the NVIDIA driver directly; it does not link ONNX Runtime or the CUDA SDK.

const cuda = @import("cuda");

const ptx = @embedFile("kernels.ptx");

pub const Device = struct {
    context: cuda.Context,
    module: cuda.Module,

    binary: cuda.Function,
    unary: cuda.Function,
    copy: cuda.Function,
    fill: cuda.Function,
    select: cuda.Function,
    tile: cuda.Function,
    clip: cuda.Function,
    scatter: cuda.Function,
    concat_copy: cuda.Function,
    pad: cuda.Function,
    gather: cuda.Function,
    layer_norm: cuda.Function,
    softmax: cuda.Function,
    matmul: cuda.Function,
    conv2d: cuda.Function,
    conv2d_gemm: cuda.Function,
    matmul_nbits: cuda.Function,
    cumulative_sum: cuda.Function,
    max_pool2d: cuda.Function,
    sum_axes: cuda.Function,
    resize_nearest: cuda.Function,
    instance_norm: cuda.Function,
    conv_transpose2d: cuda.Function,

    pub fn init(ordinal: u32) !Device {
        try cuda.init();
        const context = try cuda.Context.init(ordinal);
        errdefer context.deinit();
        const module = try cuda.Module.load(ptx);
        errdefer module.unload();

        return .{
            .context = context,
            .module = module,
            .binary = try module.function("binary"),
            .unary = try module.function("unary"),
            .copy = try module.function("copy"),
            .fill = try module.function("fill"),
            .select = try module.function("select"),
            .tile = try module.function("tile"),
            .clip = try module.function("clip"),
            .scatter = try module.function("scatter"),
            .concat_copy = try module.function("concatCopy"),
            .pad = try module.function("pad"),
            .gather = try module.function("gather"),
            .layer_norm = try module.function("layerNorm"),
            .softmax = try module.function("softmax"),
            .matmul = try module.function("matmul"),
            .conv2d = try module.function("conv2d"),
            .conv2d_gemm = try module.function("conv2dGemm"),
            .matmul_nbits = try module.function("matmulNBits"),
            .cumulative_sum = try module.function("cumulativeSum"),
            .max_pool2d = try module.function("maxPool2d"),
            .sum_axes = try module.function("sumAxes"),
            .resize_nearest = try module.function("resizeNearest"),
            .instance_norm = try module.function("instanceNorm"),
            .conv_transpose2d = try module.function("convTranspose2d"),
        };
    }

    pub fn deinit(self: *Device) void {
        self.module.unload();
        self.context.deinit();
        self.* = undefined;
    }

    pub fn name(self: Device, buffer: []u8) ![]const u8 {
        return self.context.name(buffer);
    }

    pub fn capability(self: Device) ![2]u32 {
        return self.context.computeCapability();
    }

    pub fn synchronize(self: Device) !void {
        return self.context.synchronize();
    }

    pub fn makeCurrent(self: Device) !void {
        return self.context.makeCurrent();
    }

    pub fn alloc(comptime T: type, count: usize) !cuda.Buffer(T) {
        return cuda.Buffer(T).alloc(count);
    }
};
