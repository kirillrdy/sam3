//! Native GPU execution context for the SAM 3 graph executor. This talks to
//! the driver directly (CUDA, OpenCL, or Apple Metal); it does not link ONNX Runtime.

const std = @import("std");
const build_options = @import("build_options");
pub const driver = @import("gpu_driver");

/// What a float tensor is stored as on the device. Every operator but the
/// matrix product moves more bytes than it does arithmetic, so halving the
/// element halves most of what a graph costs -- and it is the precision an
/// Intel GPU execution provider runs these graphs at anyway. Graph inputs and
/// outputs stay float; the conversion happens at the boundary.
pub const Element = if (build_options.half) f16 else f32;

const kernels_data = @embedFile("kernels");

/// The OpenCL source is compiled by the driver at load, so the element type is
/// a define rather than a second copy of the file.
const kernels_source: [:0]const u8 = if (build_options.half)
    "#define SAM3_HALF 1\n" ++ kernels_data
else
    kernels_data;

pub const Device = struct {
    context: driver.Context,
    module: driver.Module,

    binary: driver.Function,
    unary: driver.Function,
    /// Four elements to a work item, for the shapes and operations that allow
    /// it. Written against OpenCL vector types, which the Metal port carries
    /// over, so only a CUDA build reaches the scalar kernels instead.
    binary_vec: ?driver.Function,
    unary_vec: ?driver.Function,
    /// The rotary position embedding, whose eleven operators one pass does.
    /// Absent on CUDA, like the vector kernels; `findFusions` looks for the
    /// pattern only where the kernel exists to run it.
    rope: ?driver.Function,
    rope_vec: ?driver.Function,
    copy: driver.Function,
    /// `copy` a group at a time, for the strides that allow it.
    copy_vec: ?driver.Function,
    fill: driver.Function,
    select: driver.Function,
    tile: driver.Function,
    clip: driver.Function,
    scatter: driver.Function,
    concat_copy: driver.Function,
    pad: driver.Function,
    gather: driver.Function,
    layer_norm: driver.Function,
    softmax: driver.Function,
    /// The three operators a scaled dot product attention is spelled as, in
    /// one kernel, on the matrix engines.
    attention: ?driver.Function,
    /// The same reading its operands with the 2D block loads rather than
    /// staging them through local memory, where the driver has them.
    attention_block: ?driver.Function,
    matmul: driver.Function,
    matmul_post: ?driver.Function,
    /// The same product on the neural accelerators the Apple GPUs carry from
    /// the M5 on, which Metal 4 reaches through its tensor operations.
    matmul_tensor: ?driver.Function,
    /// The same product on the Apple GPU matrix units, where the driver is
    /// Metal. Absent everywhere else.
    matmul_simd: ?driver.Function,
    /// The same product on the Xe matrix engines, where the driver has them.
    /// Absent on a GPU whose OpenCL implementation lacks the extensions it is
    /// written against, and on CUDA, which reaches its own tensor cores through
    /// `matmul`.
    matmul_xmx: ?driver.Function,
    /// The same product again, reading its operands with the 2D block loads
    /// Xe2 added rather than staging them through local memory. About twice as
    /// fast, and used wherever the shapes meet what those loads require.
    matmul_xmx_block: ?driver.Function,
    conv2d: driver.Function,
    conv2d_gemm: driver.Function,
    /// The same on the matrix engines, where the driver has them.
    conv2d_gemm_xmx: ?driver.Function,
    /// The same on the Apple neural accelerators, which read the window this
    /// gathers for them out of threadgroup memory.
    conv2d_gemm_tensor: ?driver.Function,
    matmul_nbits: driver.Function,
    /// Dequantizes one weight tile into threadgroup memory and takes its
    /// product on the Apple neural accelerators.
    matmul_nbits_tensor: ?driver.Function,
    cumulative_sum: driver.Function,
    max_pool2d: driver.Function,
    sum_axes: driver.Function,
    resize_nearest: driver.Function,
    instance_norm: driver.Function,
    conv_transpose2d: driver.Function,
    conv_transpose2d_gemm: driver.Function,
    /// Scatters a non-overlapping transposed convolution's product into the
    /// image, so that the product itself can be the one on the matrix engines.
    pixel_shuffle: ?driver.Function,

    pub fn init(io: std.Io, ordinal: u32) !Device {
        _ = io;
        try driver.init();
        const context = try driver.Context.init(ordinal);
        errdefer context.deinit();

        const module = try driver.Module.load(kernels_source);
        errdefer module.unload();

        const select_name = if (@hasDecl(driver, "is_opencl") or @hasDecl(driver, "is_metal")) "select_kernel" else "select";

        return .{
            .context = context,
            .module = module,
            .binary = try module.function("binary"),
            .unary = try module.function("unary"),
            .binary_vec = module.function("binaryVec") catch null,
            .unary_vec = module.function("unaryVec") catch null,
            .rope = module.function("rope") catch null,
            .rope_vec = module.function("ropeVec") catch null,
            .copy = try module.function("copy"),
            .copy_vec = module.function("copyVec") catch null,
            .fill = try module.function("fill"),
            .select = try module.function(select_name),
            .tile = try module.function("tile"),
            .clip = try module.function("clip"),
            .scatter = try module.function("scatter"),
            .concat_copy = try module.function("concatCopy"),
            .pad = try module.function("pad"),
            .gather = try module.function("gather"),
            .layer_norm = try module.function("layerNorm"),
            .softmax = try module.function("softmax"),
            .attention = if (@hasDecl(driver, "is_metal")) null else module.function("attention") catch null,
            .attention_block = if (@hasDecl(driver, "is_metal")) null else module.function("attentionBlock") catch null,
            .matmul = try module.function("matmul"),
            .matmul_post = module.function("matmulPost") catch null,
            // A build that compiled the kernel out leaves no symbol behind, so
            // failing to find it is the answer rather than an error.
            .matmul_tensor = module.function("matmulTensor") catch null,
            .matmul_simd = module.function("matmulSimd") catch null,
            .matmul_xmx = module.function("matmulXmx") catch null,
            .matmul_xmx_block = module.function("matmulXmxBlock") catch null,
            .conv2d = try module.function("conv2d"),
            .conv2d_gemm = try module.function("conv2dGemm"),
            .conv2d_gemm_xmx = module.function("conv2dGemmXmx") catch null,
            .conv2d_gemm_tensor = module.function("conv2dGemmTensor") catch null,
            .matmul_nbits = try module.function("matmulNBits"),
            .matmul_nbits_tensor = module.function("matmulNBitsTensor") catch null,
            .cumulative_sum = try module.function("cumulativeSum"),
            .max_pool2d = try module.function("maxPool2d"),
            .sum_axes = try module.function("sumAxes"),
            .resize_nearest = try module.function("resizeNearest"),
            .instance_norm = try module.function("instanceNorm"),
            .conv_transpose2d = try module.function("convTranspose2d"),
            .conv_transpose2d_gemm = try module.function("convTranspose2dGemm"),
            .pixel_shuffle = module.function("pixelShuffle") catch null,
        };
    }

    pub fn deinit(self: *Device) void {
        self.module.unload();
        self.context.deinit();
        self.* = undefined;
    }

    pub fn synchronize(self: Device) !void {
        return self.context.synchronize();
    }

    pub fn makeCurrent(self: Device) !void {
        return self.context.makeCurrent();
    }

    pub fn alloc(comptime T: type, count: usize) !driver.Buffer(T) {
        return driver.Buffer(T).alloc(count);
    }
};
