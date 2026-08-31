//! Host-side bindings for the OpenCL driver API (ICD loader / Intel Compute Runtime),
//! talking directly to libigdrcl.so / libOpenCL.so with zero external SDK dependencies.
//! Kernels arrive as OpenCL C source and are JIT-compiled by the driver on load.

const std = @import("std");
const opencl = @This();

pub const is_opencl = true;

pub const Error = error{
    OpenCL,
    DriverNotFound,
    OutOfMemory,
};

pub const cl_int = i32;
pub const cl_uint = u32;
pub const cl_bitfield = u64;
pub const cl_device_type = cl_bitfield;
pub const cl_mem_flags = cl_bitfield;
pub const cl_command_queue_properties = cl_bitfield;
pub const cl_context_properties = isize;

pub const CL_SUCCESS: cl_int = 0;
pub const CL_DEVICE_TYPE_GPU: cl_device_type = 1 << 2;
pub const CL_PROGRAM_BUILD_LOG: cl_uint = 0x1183;
pub const CL_MEM_READ_WRITE: cl_mem_flags = 1 << 0;

pub const DevicePtr = ?*anyopaque;
pub const null_ptr: DevicePtr = null;
pub const cl_platform_id = ?*anyopaque;
pub const cl_device_id = ?*anyopaque;
pub const cl_context = ?*anyopaque;
pub const cl_command_queue = ?*anyopaque;
pub const cl_mem = ?*anyopaque;
pub const cl_program = ?*anyopaque;
pub const cl_kernel = ?*anyopaque;

pub const Dispatch = extern struct {
    clGetPlatformIDs: ?*const fn (cl_uint, ?[*]cl_platform_id, ?*cl_uint) callconv(.c) cl_int,
    clGetPlatformInfo: ?*const fn (cl_platform_id, cl_uint, usize, ?*anyopaque, ?*usize) callconv(.c) cl_int,
    clGetDeviceIDs: ?*const fn (cl_platform_id, cl_device_type, cl_uint, ?[*]cl_device_id, ?*cl_uint) callconv(.c) cl_int,
    clGetDeviceInfo: ?*const fn (cl_device_id, cl_uint, usize, ?*anyopaque, ?*usize) callconv(.c) cl_int,
    clCreateContext: ?*const fn (?[*]const cl_context_properties, cl_uint, [*]const cl_device_id, ?*const fn ([*:0]const u8, ?*const anyopaque, usize, ?*anyopaque) callconv(.c) void, ?*anyopaque, ?*cl_int) callconv(.c) cl_context,
    clCreateContextFromType: ?*const anyopaque,
    clRetainContext: ?*const fn (cl_context) callconv(.c) cl_int,
    clReleaseContext: ?*const fn (cl_context) callconv(.c) cl_int,
    clGetContextInfo: ?*const anyopaque,
    clCreateCommandQueue: ?*const fn (cl_context, cl_device_id, cl_command_queue_properties, ?*cl_int) callconv(.c) cl_command_queue,
    clRetainCommandQueue: ?*const anyopaque,
    clReleaseCommandQueue: ?*const fn (cl_command_queue) callconv(.c) cl_int,
    clGetCommandQueueInfo: ?*const anyopaque,
    clSetCommandQueueProperty: ?*const anyopaque,
    clCreateBuffer: ?*const fn (cl_context, cl_mem_flags, usize, ?*const anyopaque, ?*cl_int) callconv(.c) cl_mem,
    clCreateImage2D: ?*const anyopaque,
    clCreateImage3D: ?*const anyopaque,
    clRetainMemObject: ?*const fn (cl_mem) callconv(.c) cl_int,
    clReleaseMemObject: ?*const fn (cl_mem) callconv(.c) cl_int,
    clGetSupportedImageFormats: ?*const anyopaque,
    clGetMemObjectInfo: ?*const anyopaque,
    clGetImageInfo: ?*const anyopaque,
    clCreateSampler: ?*const anyopaque,
    clRetainSampler: ?*const anyopaque,
    clReleaseSampler: ?*const anyopaque,
    clGetSamplerInfo: ?*const anyopaque,
    clCreateProgramWithSource: ?*const fn (cl_context, cl_uint, [*]const [*:0]const u8, ?[*]const usize, ?*cl_int) callconv(.c) cl_program,
    clCreateProgramWithBinary: ?*const fn (cl_context, cl_uint, [*]const cl_device_id, [*]const usize, [*]const [*]const u8, ?[*]cl_int, ?*cl_int) callconv(.c) cl_program,
    clRetainProgram: ?*const anyopaque,
    clReleaseProgram: ?*const fn (cl_program) callconv(.c) cl_int,
    clBuildProgram: ?*const fn (cl_program, cl_uint, ?[*]const cl_device_id, ?[*:0]const u8, ?*const fn (cl_program, ?*anyopaque) callconv(.c) void, ?*anyopaque) callconv(.c) cl_int,
    clUnloadCompiler: ?*const anyopaque,
    clGetProgramInfo: ?*const anyopaque,
    clGetProgramBuildInfo: ?*const fn (cl_program, cl_device_id, cl_uint, usize, ?*anyopaque, ?*usize) callconv(.c) cl_int,
    clCreateKernel: ?*const fn (cl_program, [*:0]const u8, ?*cl_int) callconv(.c) cl_kernel,
    clCreateKernelsInProgram: ?*const anyopaque,
    clRetainKernel: ?*const fn (cl_kernel) callconv(.c) cl_int,
    clReleaseKernel: ?*const fn (cl_kernel) callconv(.c) cl_int,
    clSetKernelArg: ?*const fn (cl_kernel, cl_uint, usize, ?*const anyopaque) callconv(.c) cl_int,
    clGetKernelInfo: ?*const anyopaque,
    clGetKernelWorkGroupInfo: ?*const anyopaque,
    clWaitForEvents: ?*const anyopaque,
    clGetEventInfo: ?*const anyopaque,
    clRetainEvent: ?*const anyopaque,
    clReleaseEvent: ?*const anyopaque,
    clGetEventProfilingInfo: ?*const anyopaque,
    clFlush: ?*const fn (cl_command_queue) callconv(.c) cl_int,
    clFinish: ?*const fn (cl_command_queue) callconv(.c) cl_int,
    clEnqueueReadBuffer: ?*const fn (cl_command_queue, cl_mem, cl_uint, usize, usize, ?*anyopaque, cl_uint, ?[*]const ?*anyopaque, ?*?*anyopaque) callconv(.c) cl_int,
    clEnqueueWriteBuffer: ?*const fn (cl_command_queue, cl_mem, cl_uint, usize, usize, ?*const anyopaque, cl_uint, ?[*]const ?*anyopaque, ?*?*anyopaque) callconv(.c) cl_int,
    clEnqueueCopyBuffer: ?*const fn (cl_command_queue, cl_mem, cl_mem, usize, usize, usize, cl_uint, ?[*]const ?*anyopaque, ?*?*anyopaque) callconv(.c) cl_int,
    clEnqueueReadImage: ?*const anyopaque,
    clEnqueueWriteImage: ?*const anyopaque,
    clEnqueueCopyImage: ?*const anyopaque,
    clEnqueueCopyImageToBuffer: ?*const anyopaque,
    clEnqueueCopyBufferToImage: ?*const anyopaque,
    clEnqueueMapBuffer: ?*const anyopaque,
    clEnqueueMapImage: ?*const anyopaque,
    clEnqueueUnmapMemObject: ?*const anyopaque,
    clEnqueueNDRangeKernel: ?*const fn (cl_command_queue, cl_kernel, cl_uint, ?[*]const usize, [*]const usize, ?[*]const usize, cl_uint, ?[*]const ?*anyopaque, ?*?*anyopaque) callconv(.c) cl_int,
};

const c = @cImport({
    @cInclude("dlfcn.h");
});

var global_dispatch: ?*const Dispatch = null;
var global_platform: cl_platform_id = null;
var global_driver_handle: ?*anyopaque = null;
var global_context: ?Context = null;

var error_buf: [1024]u8 = undefined;
var error_len: usize = 0;

pub fn lastError() []const u8 {
    return error_buf[0..error_len];
}

fn setLastError(msg: []const u8) void {
    error_len = @min(msg.len, error_buf.len);
    @memcpy(error_buf[0..error_len], msg[0..error_len]);
}

fn check(err: cl_int) Error!void {
    if (err == CL_SUCCESS) return;
    var buf: [64]u8 = undefined;
    const msg = std.fmt.bufPrint(&buf, "OpenCL error code {d}", .{err}) catch "OpenCL error";
    setLastError(msg);
    return Error.OpenCL;
}

const driver_search_paths = [_][]const u8{
    "/run/opengl-driver/lib/intel-opencl/libigdrcl.so",
    "/run/opengl-driver/lib/libOpenCL.so",
    "/usr/lib64/libOpenCL.so.1",
    "/usr/lib/x86_64-linux-gnu/libOpenCL.so.1",
    "/usr/lib/libOpenCL.so",
    "libOpenCL.so.1",
    "libOpenCL.so",
};

pub fn init() Error!void {
    if (global_dispatch != null) return;

    var handle: ?*anyopaque = null;
    for (driver_search_paths) |path| {
        var zpath: [256:0]u8 = undefined;
        if (path.len >= zpath.len) continue;
        @memcpy(zpath[0..path.len], path);
        zpath[path.len] = 0;
        handle = c.dlopen(&zpath, c.RTLD_NOW | c.RTLD_GLOBAL);
        if (handle != null) break;
    }

    if (handle == null) {
        setLastError("Could not locate OpenCL driver / ICD loader (libigdrcl.so or libOpenCL.so)");
        return Error.DriverNotFound;
    }
    global_driver_handle = handle;

    const clIcdGetPlatformIDsKHR_sym = c.dlsym(handle, "clIcdGetPlatformIDsKHR");
    const clGetPlatformIDs_sym = c.dlsym(handle, "clGetPlatformIDs");

    var plat: cl_platform_id = null;
    var dispatch_ptr: *const Dispatch = undefined;

    if (clIcdGetPlatformIDsKHR_sym != null) {
        const clIcdGetPlatformIDsKHR: *const fn (cl_uint, ?[*]cl_platform_id, ?*cl_uint) callconv(.c) cl_int = @ptrCast(clIcdGetPlatformIDsKHR_sym);
        var num_platforms: cl_uint = 0;
        _ = clIcdGetPlatformIDsKHR(0, null, &num_platforms);
        if (num_platforms == 0) {
            setLastError("No OpenCL platforms found via ICD");
            return Error.OpenCL;
        }
        var platforms: [8]cl_platform_id = undefined;
        _ = clIcdGetPlatformIDsKHR(num_platforms, &platforms, null);
        plat = platforms[0];
        const raw_ptr: *const *const Dispatch = @ptrCast(@alignCast(plat.?));
        dispatch_ptr = raw_ptr.*;
    } else if (clGetPlatformIDs_sym != null) {
        const clGetPlatformIDs: *const fn (cl_uint, ?[*]cl_platform_id, ?*cl_uint) callconv(.c) cl_int = @ptrCast(clGetPlatformIDs_sym);
        var num_platforms: cl_uint = 0;
        _ = clGetPlatformIDs(0, null, &num_platforms);
        if (num_platforms == 0) {
            setLastError("No OpenCL platforms found");
            return Error.OpenCL;
        }
        var platforms: [8]cl_platform_id = undefined;
        _ = clGetPlatformIDs(num_platforms, &platforms, null);
        plat = platforms[0];
        const raw_ptr: *const *const Dispatch = @ptrCast(@alignCast(plat.?));
        dispatch_ptr = raw_ptr.*;
    } else {
        setLastError("Driver library does not export clIcdGetPlatformIDsKHR or clGetPlatformIDs");
        return Error.DriverNotFound;
    }

    global_platform = plat;
    global_dispatch = dispatch_ptr;
}

fn getDispatch() *const Dispatch {
    return global_dispatch.?;
}

fn getPlatform() cl_platform_id {
    return global_platform;
}

pub const Dim = struct { x: u32 = 1, y: u32 = 1, z: u32 = 1 };

pub const Context = struct {
    device: cl_device_id,
    context: cl_context,
    queue: cl_command_queue,
    dispatch: *const Dispatch,

    pub fn init(ordinal: u32) Error!Context {
        try opencl.init();
        const dispatch = getDispatch();
        const plat = getPlatform();

        var num_devices: cl_uint = 0;
        try check(dispatch.clGetDeviceIDs.?(plat, CL_DEVICE_TYPE_GPU, 0, null, &num_devices));
        if (num_devices == 0 or ordinal >= num_devices) {
            setLastError("Requested GPU device ordinal not found");
            return Error.OpenCL;
        }

        var devices: [16]cl_device_id = undefined;
        try check(dispatch.clGetDeviceIDs.?(plat, CL_DEVICE_TYPE_GPU, num_devices, &devices, null));
        const dev = devices[ordinal];

        const dev_arr = [_]cl_device_id{dev};
        var status: cl_int = 0;
        const ctx = dispatch.clCreateContext.?(null, 1, &dev_arr, null, null, &status);
        try check(status);
        errdefer _ = dispatch.clReleaseContext.?(ctx);

        const queue = dispatch.clCreateCommandQueue.?(ctx, dev, 0, &status);
        try check(status);
        errdefer _ = dispatch.clReleaseCommandQueue.?(queue);

        const result: Context = .{
            .device = dev,
            .context = ctx,
            .queue = queue,
            .dispatch = dispatch,
        };
        global_context = result;
        return result;
    }

    pub fn deinit(self: Context) void {
        _ = self.dispatch.clReleaseCommandQueue.?(self.queue);
        _ = self.dispatch.clReleaseContext.?(self.context);
        if (global_context != null and global_context.?.context == self.context) {
            global_context = null;
        }
    }

    pub fn makeCurrent(self: Context) Error!void {
        global_context = self;
    }

    pub fn synchronize(self: Context) Error!void {
        try check(self.dispatch.clFinish.?(self.queue));
    }
};

pub const Module = struct {
    program: cl_program,
    context: Context,

    pub fn load(source: [:0]const u8) Error!Module {
        const ctx = global_context orelse try Context.init(0);
        return loadWithContext(ctx, source);
    }

    pub fn loadWithContext(context: Context, source: [:0]const u8) Error!Module {
        const dispatch = context.dispatch;
        var status: cl_int = 0;
        const sources = [_][*:0]const u8{source.ptr};
        const lengths = [_]usize{source.len};
        const prog = dispatch.clCreateProgramWithSource.?(context.context, 1, &sources, &lengths, &status);
        try check(status);
        errdefer _ = dispatch.clReleaseProgram.?(prog);

        const devices = [_]cl_device_id{context.device};
        const build_err = dispatch.clBuildProgram.?(prog, 1, &devices, null, null, null);
        if (build_err != CL_SUCCESS) {
            var log_buf: [16384]u8 = undefined;
            var log_size: usize = 0;
            _ = dispatch.clGetProgramBuildInfo.?(prog, context.device, CL_PROGRAM_BUILD_LOG, log_buf.len, &log_buf, &log_size);
            if (log_size > 0) {
                setLastError(log_buf[0..@min(log_size, log_buf.len)]);
            } else {
                setLastError("OpenCL program build failed");
            }
            return Error.OpenCL;
        }

        return .{
            .program = prog,
            .context = context,
        };
    }

    pub fn unload(self: Module) void {
        _ = self.context.dispatch.clReleaseProgram.?(self.program);
    }

    pub fn function(self: Module, fn_name: []const u8) Error!Function {
        var buf: [256:0]u8 = undefined;
        if (fn_name.len >= buf.len) return Error.OpenCL;
        @memcpy(buf[0..fn_name.len], fn_name);
        buf[fn_name.len] = 0;

        var status: cl_int = 0;
        const kern = self.context.dispatch.clCreateKernel.?(self.program, &buf, &status);
        try check(status);
        return .{
            .kernel = kern,
            .context = self.context,
        };
    }
};

pub const Function = struct {
    kernel: cl_kernel,
    context: Context,

    pub fn launch(self: Function, grid: Dim, block: Dim, args: anytype) Error!void {
        if (grid.x == 0 or grid.y == 0 or grid.z == 0) return;
        if (block.x == 0 or block.y == 0 or block.z == 0) return;

        const dispatch = self.context.dispatch;
        const fields = @typeInfo(@TypeOf(args)).@"struct".fields;

        inline for (fields, 0..) |field, i| {
            const val = @field(args, field.name);
            const T = field.type;
            if (T == DevicePtr or T == cl_mem or T == ?*anyopaque or @typeInfo(T) == .pointer or @typeInfo(T) == .optional) {
                var mem_val: cl_mem = @ptrCast(val);
                try check(dispatch.clSetKernelArg.?(self.kernel, @intCast(i), @sizeOf(cl_mem), @ptrCast(&mem_val)));
            } else {
                try check(dispatch.clSetKernelArg.?(self.kernel, @intCast(i), @sizeOf(T), @ptrCast(&val)));
            }
        }

        const work_dim: cl_uint = if (grid.z > 1 or block.z > 1) 3 else (if (grid.y > 1 or block.y > 1) 2 else 1);
        const global_work_size = [3]usize{
            @as(usize, grid.x) * @as(usize, block.x),
            @as(usize, grid.y) * @as(usize, block.y),
            @as(usize, grid.z) * @as(usize, block.z),
        };
        const local_work_size = [3]usize{
            @as(usize, block.x),
            @as(usize, block.y),
            @as(usize, block.z),
        };

        try check(dispatch.clEnqueueNDRangeKernel.?(
            self.context.queue,
            self.kernel,
            work_dim,
            null,
            &global_work_size,
            &local_work_size,
            0,
            null,
            null,
        ));
    }
};

pub fn Buffer(comptime T: type) type {
    return struct {
        const Self = @This();

        ptr: DevicePtr,
        len: usize,
        context: ?Context = null,

        pub fn alloc(len: usize) Error!Self {
            const ctx = global_context orelse try Context.init(0);
            return allocWithContext(ctx, len);
        }

        pub fn allocWithContext(ctx_opt: ?Context, len: usize) Error!Self {
            if (len == 0) return .{ .ptr = null, .len = 0, .context = ctx_opt };
            const context = ctx_opt orelse global_context orelse try Context.init(0);
            var status: cl_int = 0;
            const mem = context.dispatch.clCreateBuffer.?(
                context.context,
                CL_MEM_READ_WRITE,
                len * @sizeOf(T),
                null,
                &status,
            );
            try check(status);
            return .{ .ptr = mem, .len = len, .context = context };
        }

        pub fn free(self: Self) void {
            if (self.ptr == null) return;
            if (self.context) |ctx| {
                _ = ctx.dispatch.clReleaseMemObject.?(@ptrCast(self.ptr));
            } else if (global_dispatch) |dispatch| {
                _ = dispatch.clReleaseMemObject.?(@ptrCast(self.ptr));
            }
        }

        pub fn slice(self: Self, offset: usize, len: usize) Self {
            std.debug.assert(offset + len <= self.len);
            if (offset == 0 and len == self.len) return self;
            if (self.ptr == null) return .{ .ptr = null, .len = 0, .context = self.context };
            const byte_offset = offset * @sizeOf(T);
            const byte_size = len * @sizeOf(T);
            const region = extern struct { origin: usize, size: usize }{ .origin = byte_offset, .size = byte_size };
            if (byte_offset % 128 == 0 and global_driver_handle != null) {
                const clCreateSubBuffer_fn: ?*const fn (cl_mem, cl_mem_flags, cl_uint, *const anyopaque, ?*cl_int) callconv(.c) cl_mem = @ptrCast(c.dlsym(global_driver_handle.?, "clCreateSubBuffer"));
                if (clCreateSubBuffer_fn) |create_sub| {
                    var status: cl_int = 0;
                    const sub = create_sub(@ptrCast(self.ptr), CL_MEM_READ_WRITE, 0x4200, &region, &status);
                    if (status == CL_SUCCESS) {
                        return .{ .ptr = sub, .len = len, .context = self.context };
                    }
                }
            }
            return self;
        }

        pub fn upload(self: Self, host: []const T) Error!void {
            std.debug.assert(host.len <= self.len);
            if (host.len == 0 or self.ptr == null) return;
            const ctx = self.context orelse global_context orelse return Error.OpenCL;
            try check(ctx.dispatch.clEnqueueWriteBuffer.?(
                ctx.queue,
                @ptrCast(self.ptr),
                1,
                0,
                host.len * @sizeOf(T),
                host.ptr,
                0,
                null,
                null,
            ));
        }

        /// Queues a host-to-device copy without waiting for earlier GPU work.
        /// The caller must keep `host` unchanged until the queue is synchronized.
        pub fn uploadAsync(self: Self, host: []const T) Error!void {
            std.debug.assert(host.len <= self.len);
            if (host.len == 0 or self.ptr == null) return;
            const ctx = self.context orelse global_context orelse return Error.OpenCL;
            try check(ctx.dispatch.clEnqueueWriteBuffer.?(
                ctx.queue,
                @ptrCast(self.ptr),
                0,
                0,
                host.len * @sizeOf(T),
                host.ptr,
                0,
                null,
                null,
            ));
        }

        pub fn download(self: Self, host: []T) Error!void {
            std.debug.assert(host.len <= self.len);
            if (host.len == 0 or self.ptr == null) return;
            const ctx = self.context orelse global_context orelse return Error.OpenCL;
            try check(ctx.dispatch.clEnqueueReadBuffer.?(
                ctx.queue,
                @ptrCast(self.ptr),
                1,
                0,
                host.len * @sizeOf(T),
                host.ptr,
                0,
                null,
                null,
            ));
        }
    };
}
