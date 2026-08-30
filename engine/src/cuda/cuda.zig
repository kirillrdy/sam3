//! Host-side bindings for the CUDA driver API (libcuda), the interface the
//! NVIDIA driver exports directly -- no CUDA toolkit, no nvcc, no runtime API.
//! Kernels arrive as PTX text and are JIT-compiled by the driver on load.

const std = @import("std");

pub const Error = error{Cuda};

/// An address in device memory. Kernel pointer parameters take this.
pub const DevicePtr = u64;
pub const null_ptr: DevicePtr = 0;
pub const Device = c_int;

const Result = c_uint;
const CUcontext = *opaque {};
const CUmodule = *opaque {};
const CUfunction = *opaque {};
const CUstream = *opaque {};

// The driver exports the post-CUDA-4.0 entry points under `_v2` names; the
// unsuffixed ones in cuda.h are macros that redirect to these.
extern fn cuInit(flags: c_uint) Result;
extern fn cuGetErrorString(err: Result, str: *[*:0]const u8) Result;
extern fn cuDeviceGet(device: *Device, ordinal: c_int) Result;
extern fn cuDeviceGetName(name: [*]u8, len: c_int, dev: Device) Result;
extern fn cuDeviceGetAttribute(value: *c_int, attribute: c_uint, dev: Device) Result;
extern fn cuDevicePrimaryCtxRetain(ctx: *CUcontext, dev: Device) Result;
extern fn cuDevicePrimaryCtxRelease_v2(dev: Device) Result;
extern fn cuCtxSetCurrent(ctx: CUcontext) Result;
extern fn cuCtxSynchronize() Result;
extern fn cuModuleLoadDataEx(
    module: *CUmodule,
    image: *const anyopaque,
    count: c_uint,
    options: [*]const c_uint,
    values: [*]?*anyopaque,
) Result;
extern fn cuModuleUnload(module: CUmodule) Result;
extern fn cuModuleGetFunction(func: *CUfunction, module: CUmodule, name: [*:0]const u8) Result;
extern fn cuMemAlloc_v2(dptr: *DevicePtr, bytes: usize) Result;
extern fn cuMemFree_v2(dptr: DevicePtr) Result;
extern fn cuMemcpyHtoD_v2(dst: DevicePtr, src: *const anyopaque, bytes: usize) Result;
extern fn cuMemcpyDtoH_v2(dst: *anyopaque, src: DevicePtr, bytes: usize) Result;
extern fn cuLaunchKernel(
    f: CUfunction,
    grid_x: c_uint,
    grid_y: c_uint,
    grid_z: c_uint,
    block_x: c_uint,
    block_y: c_uint,
    block_z: c_uint,
    shared_bytes: c_uint,
    stream: ?CUstream,
    params: ?[*]?*anyopaque,
    extra: ?[*]?*anyopaque,
) Result;

const JitOption = enum(c_uint) {
    error_log_buffer = 5,
    error_log_buffer_size_bytes = 6,
};

const Attribute = enum(c_uint) {
    max_threads_per_block = 1,
    compute_capability_major = 75,
    compute_capability_minor = 76,
};

var error_buf: [512]u8 = undefined;
var error_len: usize = 0;

/// Message for the most recent `error.Cuda`.
pub fn lastError() []const u8 {
    return error_buf[0..error_len];
}

fn check(result: Result) Error!void {
    if (result == 0) return;
    var message: [*:0]const u8 = "unknown CUDA error";
    _ = cuGetErrorString(result, &message);
    const text = std.mem.span(message);
    error_len = @min(text.len, error_buf.len);
    @memcpy(error_buf[0..error_len], text[0..error_len]);
    return Error.Cuda;
}

pub fn init() Error!void {
    try check(cuInit(0));
}

/// Grid and block extents, in blocks and in threads respectively.
pub const Dim = struct { x: u32 = 1, y: u32 = 1, z: u32 = 1 };

pub const Context = struct {
    device: Device,
    ptr: CUcontext,

    /// Binds the device's primary context to this thread -- the same context
    /// the runtime API would use, so it costs nothing to retain repeatedly.
    pub fn init(ordinal: u32) Error!Context {
        var device: Device = undefined;
        try check(cuDeviceGet(&device, @intCast(ordinal)));
        var ptr: CUcontext = undefined;
        try check(cuDevicePrimaryCtxRetain(&ptr, device));
        try check(cuCtxSetCurrent(ptr));
        return .{ .device = device, .ptr = ptr };
    }

    pub fn deinit(self: Context) void {
        _ = cuDevicePrimaryCtxRelease_v2(self.device);
    }

    pub fn name(self: Context, buf: []u8) Error![]const u8 {
        try check(cuDeviceGetName(buf.ptr, @intCast(buf.len), self.device));
        return std.mem.sliceTo(buf, 0);
    }

    /// Compute capability as {major, minor} -- `sm_89` for an Ada RTX 4070.
    pub fn computeCapability(self: Context) Error![2]u32 {
        var major: c_int = 0;
        var minor: c_int = 0;
        try check(cuDeviceGetAttribute(&major, @intFromEnum(Attribute.compute_capability_major), self.device));
        try check(cuDeviceGetAttribute(&minor, @intFromEnum(Attribute.compute_capability_minor), self.device));
        return .{ @intCast(major), @intCast(minor) };
    }

    pub fn maxThreadsPerBlock(self: Context) Error!u32 {
        var value: c_int = 0;
        try check(cuDeviceGetAttribute(&value, @intFromEnum(Attribute.max_threads_per_block), self.device));
        return @intCast(value);
    }

    /// CUDA's current context is thread-local. A retained primary context may
    /// be shared by the server, but each worker must bind it before issuing
    /// allocations, copies, launches, or synchronization calls.
    pub fn makeCurrent(self: Context) Error!void {
        try check(cuCtxSetCurrent(self.ptr));
    }

    /// Waits for every launch on this context to finish. Launches are
    /// asynchronous, so kernel errors surface here rather than at `launch`.
    pub fn synchronize(_: Context) Error!void {
        try check(cuCtxSynchronize());
    }
};

pub const Module = struct {
    ptr: CUmodule,
    prefix: []const u8,

    /// JIT-compiles PTX text for the current context's GPU. A compile failure
    /// leaves the JIT's own diagnostics in `lastError`.
    pub fn load(ptx: [:0]const u8) Error!Module {
        const options = [_]c_uint{
            @intFromEnum(JitOption.error_log_buffer),
            @intFromEnum(JitOption.error_log_buffer_size_bytes),
        };
        var values = [_]?*anyopaque{
            @ptrCast(&error_buf),
            @ptrFromInt(error_buf.len),
        };
        error_buf[0] = 0;

        var ptr: CUmodule = undefined;
        const result = cuModuleLoadDataEx(&ptr, ptx.ptr, options.len, &options, &values);
        if (result != 0) {
            // The JIT's log names the offending instruction; the result code
            // only ever says "a PTX JIT compilation failed".
            const log = std.mem.sliceTo(&error_buf, 0);
            if (log.len == 0) {
                try check(result);
                return Error.Cuda;
            }
            error_len = log.len;
            return Error.Cuda;
        }
        return .{ .ptr = ptr, .prefix = manglePrefix(ptx) };
    }

    pub fn unload(self: Module) void {
        _ = cuModuleUnload(self.ptr);
    }

    /// Looks a kernel up by its Zig name, e.g. "vecAdd".
    pub fn function(self: Module, name: []const u8) Error!Function {
        var buf: [256]u8 = undefined;
        const symbol = std.fmt.bufPrintZ(&buf, "{s}{s}", .{ self.prefix, name }) catch return Error.Cuda;
        var ptr: CUfunction = undefined;
        try check(cuModuleGetFunction(&ptr, self.ptr, symbol.ptr));
        return .{ .ptr = ptr };
    }

    /// Zig mangles kernel symbols as `<root module>_$_<function>`, so read the
    /// prefix back off the first `.entry` in the PTX instead of hard-coding
    /// the name of the file the kernels were built from.
    fn manglePrefix(ptx: [:0]const u8) []const u8 {
        const entry = std.mem.indexOf(u8, ptx, ".entry ") orelse return "";
        const start = entry + ".entry ".len;
        const line = ptx[start .. std.mem.indexOfScalarPos(u8, ptx, start, '\n') orelse ptx.len];
        const marker = std.mem.indexOf(u8, line, "_$_") orelse return "";
        return line[0 .. marker + "_$_".len];
    }
};

pub const Function = struct {
    ptr: CUfunction,

    /// Launches `grid` blocks of `block` threads. `args` is a tuple whose
    /// fields match the kernel's parameters -- `DevicePtr` for pointers.
    /// Returns as soon as the launch is queued; call `Context.synchronize`.
    pub fn launch(self: Function, grid: Dim, block: Dim, args: anytype) Error!void {
        // An empty grid means there is no work -- an operator over an empty
        // tensor. The driver rejects a zero extent, so the no-op belongs here
        // rather than at every call site that divides a count into blocks.
        if (grid.x == 0 or grid.y == 0 or grid.z == 0) return;
        if (block.x == 0 or block.y == 0 or block.z == 0) return;

        const fields = @typeInfo(@TypeOf(args)).@"struct".fields;
        comptime var types: [fields.len]type = undefined;
        inline for (fields, 0..) |field, i| types[i] = field.type;

        // The driver wants an array of pointers to the argument values, so the
        // values need runtime storage even when the caller passed literals.
        var values: std.meta.Tuple(&types) = args;
        var params: [fields.len]?*anyopaque = undefined;
        inline for (fields, 0..) |field, i| {
            params[i] = @ptrCast(&@field(values, field.name));
        }

        try check(cuLaunchKernel(
            self.ptr,
            grid.x,
            grid.y,
            grid.z,
            block.x,
            block.y,
            block.z,
            0,
            null,
            if (fields.len == 0) null else &params,
            null,
        ));
    }
};

/// A `len`-element array in device memory.
pub fn Buffer(comptime T: type) type {
    return struct {
        const Self = @This();

        ptr: DevicePtr,
        len: usize,

        /// An empty tensor is a legitimate value -- SAM 3's decoder takes a
        /// 1x0x4 box prompt -- but the driver rejects a zero-byte request, so
        /// a zero-length buffer is a null pointer rather than an allocation.
        pub fn alloc(len: usize) Error!Self {
            if (len == 0) return .{ .ptr = 0, .len = 0 };
            var ptr: DevicePtr = 0;
            try check(cuMemAlloc_v2(&ptr, len * @sizeOf(T)));
            return .{ .ptr = ptr, .len = len };
        }

        pub fn free(self: Self) void {
            if (self.ptr == 0) return;
            _ = cuMemFree_v2(self.ptr);
        }

        /// A view of `len` elements starting at `offset`, for handing one
        /// plane of a larger allocation to a kernel.
        pub fn slice(self: Self, offset: usize, len: usize) Self {
            std.debug.assert(offset + len <= self.len);
            return .{ .ptr = self.ptr + offset * @sizeOf(T), .len = len };
        }

        pub fn upload(self: Self, host: []const T) Error!void {
            std.debug.assert(host.len <= self.len);
            if (host.len == 0) return;
            try check(cuMemcpyHtoD_v2(self.ptr, host.ptr, host.len * @sizeOf(T)));
        }

        pub fn download(self: Self, host: []T) Error!void {
            std.debug.assert(host.len <= self.len);
            if (host.len == 0) return;
            try check(cuMemcpyDtoH_v2(host.ptr, self.ptr, host.len * @sizeOf(T)));
        }
    };
}
