//! Minimal Metal compute driver used by the native graph runtime. The public
//! surface deliberately matches the CUDA and OpenCL packages.

const std = @import("std");

pub const is_metal = true;
pub const Error = error{Metal};

const c = @cImport(@cInclude("bridge.h"));

pub const DevicePtr = extern struct {
    buffer: ?*anyopaque,
    offset: usize,
};
pub const null_ptr: DevicePtr = .{ .buffer = null, .offset = 0 };
pub const Dim = struct { x: u32 = 1, y: u32 = 1, z: u32 = 1 };

threadlocal var current_context: ?Context = null;

pub fn lastError() []const u8 {
    return std.mem.span(c.sam_metal_last_error());
}

pub fn init() Error!void {}

pub const Context = struct {
    ptr: *c.SamMetalContext,

    pub fn init(ordinal: u32) Error!Context {
        const result: Context = .{ .ptr = c.sam_metal_context_create(ordinal) orelse return Error.Metal };
        current_context = result;
        return result;
    }

    pub fn deinit(self: Context) void {
        c.sam_metal_context_destroy(self.ptr);
        if (current_context != null and current_context.?.ptr == self.ptr) current_context = null;
    }

    pub fn name(self: Context, buf: []u8) Error![]const u8 {
        const device_name = std.mem.span(c.sam_metal_context_name(self.ptr));
        if (device_name.len > buf.len) return Error.Metal;
        @memcpy(buf[0..device_name.len], device_name);
        return buf[0..device_name.len];
    }

    pub fn computeCapability(_: Context) Error![2]u32 {
        return .{ 3, 0 };
    }
    pub fn makeCurrent(self: Context) Error!void {
        current_context = self;
    }
    pub fn synchronize(self: Context) Error!void {
        if (c.sam_metal_context_synchronize(self.ptr) == 0) return Error.Metal;
    }
};

pub const Module = struct {
    ptr: *c.SamMetalModule,
    context: Context,

    pub fn load(source: [:0]const u8) Error!Module {
        const context = current_context orelse return Error.Metal;
        return .{ .ptr = c.sam_metal_module_create(context.ptr, source.ptr) orelse return Error.Metal, .context = context };
    }

    pub fn unload(self: Module) void {
        c.sam_metal_module_destroy(self.ptr);
    }

    pub fn function(self: Module, name: []const u8) Error!Function {
        var symbol: [256:0]u8 = undefined;
        if (name.len >= symbol.len) return Error.Metal;
        @memcpy(symbol[0..name.len], name);
        symbol[name.len] = 0;
        return .{ .ptr = c.sam_metal_function_create(self.ptr, &symbol) orelse return Error.Metal };
    }
};

pub const Function = struct {
    ptr: *c.SamMetalFunction,

    pub fn launch(self: Function, grid: Dim, block: Dim, args: anytype) Error!void {
        if (grid.x == 0 or grid.y == 0 or grid.z == 0) return;
        if (block.x == 0 or block.y == 0 or block.z == 0) return;
        const fields = @typeInfo(@TypeOf(args)).@"struct".fields;
        comptime var types: [fields.len]type = undefined;
        inline for (fields, 0..) |field, i| types[i] = field.type;
        var values: std.meta.Tuple(&types) = args;
        var encoded_args: [fields.len]c.SamMetalArg = undefined;
        inline for (fields, 0..) |field, i| {
            if (field.type == DevicePtr) {
                encoded_args[i] = .{ .kind = c.SAM_METAL_BUFFER, .buffer = @bitCast(@field(values, field.name)), .bytes = null, .size = 0 };
            } else {
                encoded_args[i] = .{ .kind = c.SAM_METAL_BYTES, .buffer = @bitCast(null_ptr), .bytes = &@field(values, field.name), .size = @sizeOf(field.type) };
            }
        }
        const metal_grid: c.SamMetalDim = .{ .x = grid.x, .y = grid.y, .z = grid.z };
        const metal_block: c.SamMetalDim = .{ .x = block.x, .y = block.y, .z = block.z };
        if (c.sam_metal_launch(self.ptr, metal_grid, metal_block, &encoded_args, encoded_args.len) == 0) return Error.Metal;
    }
};

pub fn Buffer(comptime T: type) type {
    return struct {
        const Self = @This();
        ptr: DevicePtr,
        len: usize,

        pub fn alloc(len: usize) Error!Self {
            if (len == 0) return .{ .ptr = null_ptr, .len = 0 };
            const context = current_context orelse return Error.Metal;
            return .{ .ptr = .{ .buffer = c.sam_metal_buffer_create(context.ptr, len * @sizeOf(T)) orelse return Error.Metal, .offset = 0 }, .len = len };
        }

        pub fn free(self: Self) void {
            c.sam_metal_buffer_destroy(self.ptr.buffer);
        }

        pub fn slice(self: Self, offset: usize, len: usize) Self {
            std.debug.assert(offset + len <= self.len);
            return .{ .ptr = .{ .buffer = self.ptr.buffer, .offset = self.ptr.offset + offset * @sizeOf(T) }, .len = len };
        }

        pub fn upload(self: Self, host: []const T) Error!void {
            std.debug.assert(host.len <= self.len);
            if (host.len == 0) return;
            // A direct shared-memory write must not overtake kernels already
            // reading this buffer. Synchronous uploads are uncommon graph
            // boundaries, so retire the queue before exposing the write.
            if (current_context) |context| try context.synchronize();
            c.sam_metal_buffer_upload(@bitCast(self.ptr), host.ptr, host.len * @sizeOf(T));
        }
        pub fn uploadAsync(self: Self, host: []const T) Error!void {
            std.debug.assert(host.len <= self.len);
            if (host.len == 0) return;
            const context = current_context orelse return Error.Metal;
            if (c.sam_metal_buffer_upload_async(context.ptr, @bitCast(self.ptr), host.ptr, host.len * @sizeOf(T)) == 0)
                return Error.Metal;
        }
        pub fn download(self: Self, host: []T) Error!void {
            std.debug.assert(host.len <= self.len);
            if (current_context) |context| try context.synchronize();
            if (host.len != 0) c.sam_metal_buffer_download(@bitCast(self.ptr), host.ptr, host.len * @sizeOf(T));
        }
    };
}
