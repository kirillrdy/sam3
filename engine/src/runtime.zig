//! Session boundary used by the SAM 3 application. Graph parsing and CUDA
//! ownership live here so the application does not know which executor backs
//! a model. The node executor is deliberately kept behind `Session.run`.

const std = @import("std");
const cuda = @import("cuda");
const onnx = @import("onnx.zig");
const cuda_device = @import("device.zig");

pub const Error = error{
    NativeRuntime,
    UnsupportedOperator,
    MissingValue,
    InvalidShape,
    UnsupportedDataType,
    UnsupportedSliceStep,
    UnsupportedSliceType,
    RankTooLarge,
};

/// The two elementwise kernels select their operation by number. `kernels.zig`
/// is compiled for the GPU and cannot be imported here, so these mirror the
/// enums it declares and must be kept in step with them.
const Binary = enum(u32) { add, sub, mul, div, pow, min, max, equal, less, greater };
const Unary = enum(u32) { neg, erf, exp, sqrt, reciprocal, sigmoid, tanh, relu, abs, floor, sin, cos, log, sign, is_nan };

/// Comparisons run on the host, over the i64 tensors a graph does its shape
/// arithmetic with, so they are their own list rather than `Binary` entries.
const Compare = enum { equal, greater, greater_equal, less, less_equal };

/// Matches `kernels.max_rank`: the stride arrays a launch carries are fixed.
const max_rank = 8;

/// The block tile `kernels.matmul` is written around, mirrored here so the
/// host can size the grid. A block of 16x16 threads covers this much of C.
const matmul_tile_m = 128;
const matmul_tile_n = 64;

var error_buffer: [1024]u8 = undefined;
var error_length: usize = 0;

fn setError(comptime format: []const u8, args: anytype) void {
    const text = std.fmt.bufPrint(&error_buffer, format, args) catch "native runtime error";
    error_length = text.len;
}

pub fn lastError() []const u8 {
    // Not every failure comes from a graph node: allocations, copies and the
    // end-of-run synchronize fail with `error.Cuda` and only the driver has
    // anything to say about them.
    if (error_length == 0) return cuda.lastError();
    return error_buffer[0..error_length];
}

pub fn init(_: std.mem.Allocator, _: std.Io) !void {}

pub fn version() []const u8 {
    return "native CUDA";
}

pub const DeviceKind = enum {
    npu,
    gpu,
    webgpu,
    cpu,

    pub fn openvinoName(self: DeviceKind) [:0]const u8 {
        return switch (self) {
            .npu => "NPU",
            .gpu => "GPU",
            .webgpu => "WEBGPU",
            .cpu => "CPU",
        };
    }
};

pub const Device = struct {
    kind: DeviceKind,
    id: u32,
};

pub const Accelerator = union(enum) {
    device: Device,
    coreml: DeviceKind,
};

pub const Option = struct {
    key: [*:0]const u8,
    value: [*:0]const u8,
};

/// Device memory a graph is done with, kept for the next node that wants the
/// same size rather than handed back to the driver. `cuMemFree` synchronizes
/// the whole device, so freeing per node serializes the graph: nothing after a
/// release can be queued until everything before it has finished. Reuse is
/// safe without that synchronize because every launch is on the same stream,
/// which already orders a buffer's last read before its next write.
///
/// Buckets are exact sizes. A graph runs the same shapes over and over, so the
/// pool converges on that working set instead of growing.
const Pool = struct {
    buckets: std.AutoHashMapUnmanaged(usize, std.ArrayListUnmanaged(cuda.DevicePtr)) = .empty,

    fn take(self: *Pool, count: usize) !cuda.Buffer(f32) {
        if (self.buckets.getPtr(count)) |bucket| {
            if (bucket.items.len != 0) {
                const ptr = bucket.items[bucket.items.len - 1];
                bucket.items.len -= 1;
                return .{ .ptr = ptr, .len = count };
            }
        }
        return cuda.Buffer(f32).alloc(count);
    }

    /// Returning a buffer must not fail, so a pool that cannot record it just
    /// gives it back to the driver.
    fn give(self: *Pool, allocator: std.mem.Allocator, buffer: cuda.Buffer(f32)) void {
        if (buffer.len == 0) return;
        const entry = self.buckets.getOrPut(allocator, buffer.len) catch return buffer.free();
        if (!entry.found_existing) entry.value_ptr.* = .empty;
        entry.value_ptr.append(allocator, buffer.ptr) catch buffer.free();
    }

    fn deinit(self: *Pool, allocator: std.mem.Allocator) void {
        var buckets = self.buckets.iterator();
        while (buckets.next()) |entry| {
            for (entry.value_ptr.items) |ptr| (cuda.Buffer(f32){ .ptr = ptr, .len = entry.key_ptr.* }).free();
            entry.value_ptr.deinit(allocator);
        }
        self.buckets.deinit(allocator);
    }
};

const State = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    gpu: cuda_device.Device,
    /// Shapes and strides for the current launch. Bounded by the maximum rank,
    /// so one small allocation serves every node.
    metadata: cuda.Buffer(u32),
    /// Index arrays for Gather and ScatterND. Unlike shapes these have no
    /// fixed bound, so the buffer grows to the largest a graph has asked for.
    indices: cuda.Buffer(u32) = .{ .ptr = 0, .len = 0 },
    /// Intermediate tensors, recycled between nodes.
    pool: Pool = .{},
};

pub const Env = struct {
    state: *State,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, _: [:0]const u8) !Env {
        const state = try allocator.create(State);
        errdefer allocator.destroy(state);
        var gpu = try cuda_device.Device.init(0);
        errdefer gpu.deinit();
        const metadata = try cuda.Buffer(u32).alloc(128);
        errdefer metadata.free();
        state.* = .{
            .allocator = allocator,
            .io = io,
            .gpu = gpu,
            .metadata = metadata,
        };
        return .{ .state = state };
    }

    pub fn deinit(self: Env) void {
        const allocator = self.state.allocator;
        self.state.pool.deinit(allocator);
        self.state.indices.free();
        self.state.metadata.free();
        self.state.gpu.deinit();
        allocator.destroy(self.state);
    }

    pub fn registerProvider(_: Env, _: [:0]const u8, _: [:0]const u8) !void {}

    pub fn find(_: Env, kind: DeviceKind) !?Device {
        return if (kind == .gpu) .{ .kind = .gpu, .id = 0 } else null;
    }
};

pub const Session = struct {
    state: *SessionState,
    device: DeviceKind = .gpu,

    pub fn open(
        env: Env,
        model_path: [:0]const u8,
        _: ?Accelerator,
        _: []const Option,
        _: ?*const fn ([]const u8) void,
    ) !Session {
        const state = try env.state.allocator.create(SessionState);
        errdefer env.state.allocator.destroy(state);
        state.* = .{
            .env = env.state,
            .graph = try onnx.Graph.open(env.state.allocator, env.state.io, model_path),
        };
        return .{ .state = state };
    }

    pub fn deinit(self: Session) void {
        const allocator = self.state.env.allocator;
        var buffers = self.state.constants.valueIterator();
        while (buffers.next()) |buffer| buffer.free();
        self.state.constants.deinit(allocator);
        self.state.graph.deinit();
        allocator.destroy(self.state);
    }

    pub fn run(
        self: Session,
        input_names: []const [*:0]const u8,
        inputs: []const Value,
        output_names: []const [*:0]const u8,
        outputs: []Value,
    ) !void {
        return self.state.run(input_names, inputs, output_names, outputs);
    }
};

const Storage = struct {
    buffer: cuda.Buffer(f32),
    references: usize = 1,
};

const Tensor = struct {
    dtype: onnx.DataType,
    dims: []const i64,
    data: union(enum) {
        host: []const u8,
        gpu: *Storage,
        constant_gpu: cuda.Buffer(f32),
    },

    fn count(self: Tensor) !usize {
        var result: usize = 1;
        for (self.dims) |dim| {
            if (dim < 0) return Error.InvalidShape;
            result = try std.math.mul(usize, result, @intCast(dim));
        }
        return result;
    }

    fn i64s(self: Tensor) ![]const i64 {
        if (self.dtype != .i64) return Error.UnsupportedDataType;
        return @alignCast(std.mem.bytesAsSlice(i64, self.data.host));
    }

    /// ONNX spells a boolean as one byte per element.
    fn bools(self: Tensor) ![]const u8 {
        if (self.dtype != .bool or !self.onHost()) return Error.UnsupportedDataType;
        return self.data.host;
    }

    /// True for the tensors a graph does its shape arithmetic with. Those stay
    /// on the host: they are a handful of elements each, and the operators
    /// that consume them -- Reshape, Expand, Slice -- need to read the values.
    fn onHost(self: Tensor) bool {
        return switch (self.data) {
            .host => true,
            .gpu, .constant_gpu => false,
        };
    }

    /// One element of a host tensor, widened to the type shape arithmetic is
    /// done in. Booleans read back as 0 or 1.
    fn element(self: Tensor, index: usize) !i64 {
        return switch (self.dtype) {
            .i64 => (try self.i64s())[index],
            .bool => (try self.bools())[index],
            else => Error.UnsupportedDataType,
        };
    }

    fn gpuBuffer(self: Tensor) !cuda.Buffer(f32) {
        if (self.dtype != .f32) return Error.UnsupportedDataType;
        return switch (self.data) {
            .gpu => |storage| storage.buffer,
            .constant_gpu => |buffer| buffer,
            .host => Error.NativeRuntime,
        };
    }
};

const SessionState = struct {
    env: *State,
    graph: onnx.Graph,
    constants: std.StringHashMapUnmanaged(cuda.Buffer(f32)) = .empty,

    fn run(
        self: *SessionState,
        input_names: []const [*:0]const u8,
        inputs: []const Value,
        output_names: []const [*:0]const u8,
        outputs: []Value,
    ) !void {
        if (input_names.len != inputs.len or output_names.len != outputs.len) return Error.NativeRuntime;
        // Otherwise a failure with nothing to say about itself reports the
        // previous run's message.
        error_length = 0;
        try self.env.gpu.makeCurrent();

        var arena_state: std.heap.ArenaAllocator = .init(self.env.allocator);
        defer arena_state.deinit();
        const arena = arena_state.allocator();

        var values: std.StringHashMapUnmanaged(*Tensor) = .empty;
        defer values.deinit(arena);

        var uses: std.StringHashMapUnmanaged(usize) = .empty;
        defer uses.deinit(arena);
        for (self.graph.nodes) |node| for (node.inputs) |name| {
            if (name.len == 0) continue;
            const entry = try uses.getOrPut(arena, name);
            entry.value_ptr.* = if (entry.found_existing) entry.value_ptr.* + 1 else 1;
        };
        for (output_names) |name_z| {
            const name = std.mem.span(name_z);
            const entry = try uses.getOrPut(arena, name);
            entry.value_ptr.* = if (entry.found_existing) entry.value_ptr.* + 1 else 1;
        }

        for (input_names, inputs) |name_z, value| {
            const tensor = try arena.create(Tensor);
            if (value.dtype == .f32) {
                const storage = try self.newStorage(value.bytes.len / @sizeOf(f32));
                try storage.buffer.upload(@alignCast(std.mem.bytesAsSlice(f32, value.bytes)));
                tensor.* = .{ .dtype = .f32, .dims = value.dims, .data = .{ .gpu = storage } };
            } else {
                tensor.* = .{ .dtype = value.dtype, .dims = value.dims, .data = .{ .host = value.bytes } };
            }
            try values.put(arena, std.mem.span(name_z), tensor);
        }

        for (self.graph.nodes, 0..) |node, node_index| {
            self.execute(arena, &values, node) catch |err| {
                const detail = if (err == error.Cuda) cuda.lastError() else "";
                setError("node {d} {s} ({s}): {t}{s}{s}", .{
                    node_index,
                    node.op_type,
                    node.name,
                    err,
                    if (detail.len == 0) "" else ": ",
                    detail,
                });
                return err;
            };
            for (node.inputs) |name| {
                if (name.len == 0) continue;
                const remaining = uses.getPtr(name) orelse continue;
                remaining.* -= 1;
                if (remaining.* == 0) if (values.get(name)) |tensor| self.release(tensor);
            }
        }

        try self.env.gpu.synchronize();
        for (output_names, outputs) |name_z, *output| {
            const name = std.mem.span(name_z);
            const tensor = values.get(name) orelse return Error.MissingValue;
            output.* = try Value.take(self.env.allocator, tensor.*);
            if (uses.getPtr(name)) |remaining| {
                remaining.* -= 1;
                if (remaining.* == 0) self.release(tensor);
            }
        }
    }

    fn execute(
        self: *SessionState,
        arena: std.mem.Allocator,
        values: *std.StringHashMapUnmanaged(*Tensor),
        node: onnx.Node,
    ) !void {
        if (std.mem.eql(u8, node.op_type, "Conv")) return self.conv(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "ConvTranspose")) return self.convTranspose(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "Split")) return self.split(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "Shape")) return self.shape(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "Gather")) return self.gather(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "Reshape")) return self.reshape(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "Unsqueeze")) return self.unsqueeze(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "Squeeze")) return self.squeeze(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "Transpose")) return self.transpose(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "Concat")) return self.concat(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "Slice")) return self.slice(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "Pad")) return self.pad(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "MatMul")) return self.matmul(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "Add")) return self.binary(arena, values, node, .add);
        if (std.mem.eql(u8, node.op_type, "Sub")) return self.binary(arena, values, node, .sub);
        if (std.mem.eql(u8, node.op_type, "Mul")) return self.binary(arena, values, node, .mul);
        if (std.mem.eql(u8, node.op_type, "Div")) return self.binary(arena, values, node, .div);
        if (std.mem.eql(u8, node.op_type, "Mod")) return self.binary(arena, values, node, .div);
        if (std.mem.eql(u8, node.op_type, "Neg")) return self.unary(arena, values, node, .neg);
        if (std.mem.eql(u8, node.op_type, "Erf")) return self.unary(arena, values, node, .erf);
        if (std.mem.eql(u8, node.op_type, "Relu")) return self.unary(arena, values, node, .relu);
        if (std.mem.eql(u8, node.op_type, "Sigmoid")) return self.unary(arena, values, node, .sigmoid);
        if (std.mem.eql(u8, node.op_type, "Sqrt")) return self.unary(arena, values, node, .sqrt);
        if (std.mem.eql(u8, node.op_type, "Exp")) return self.unary(arena, values, node, .exp);
        if (std.mem.eql(u8, node.op_type, "Tanh")) return self.unary(arena, values, node, .tanh);
        if (std.mem.eql(u8, node.op_type, "Abs")) return self.unary(arena, values, node, .abs);
        if (std.mem.eql(u8, node.op_type, "Floor")) return self.unary(arena, values, node, .floor);
        if (std.mem.eql(u8, node.op_type, "Sin")) return self.unary(arena, values, node, .sin);
        if (std.mem.eql(u8, node.op_type, "Cos")) return self.unary(arena, values, node, .cos);
        if (std.mem.eql(u8, node.op_type, "Equal")) return self.compare(arena, values, node, .equal);
        if (std.mem.eql(u8, node.op_type, "Greater")) return self.compare(arena, values, node, .greater);
        if (std.mem.eql(u8, node.op_type, "GreaterOrEqual")) return self.compare(arena, values, node, .greater_equal);
        if (std.mem.eql(u8, node.op_type, "Less")) return self.compare(arena, values, node, .less);
        if (std.mem.eql(u8, node.op_type, "LessOrEqual")) return self.compare(arena, values, node, .less_equal);
        if (std.mem.eql(u8, node.op_type, "Not")) return self.not(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "Where")) return self.where(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "Cast")) return self.cast(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "Expand")) return self.expand(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "Tile")) return self.tile(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "Range")) return self.range(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "ConstantOfShape")) return self.constantOfShape(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "Clip")) return self.clip(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "OneHot")) return self.oneHot(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "ScatterND")) return self.scatterNd(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "Log")) return self.unary(arena, values, node, .log);
        if (std.mem.eql(u8, node.op_type, "Sign")) return self.unary(arena, values, node, .sign);
        if (std.mem.eql(u8, node.op_type, "IsNaN")) return self.unary(arena, values, node, .is_nan);
        if (std.mem.eql(u8, node.op_type, "Max")) return self.binary(arena, values, node, .max);
        if (std.mem.eql(u8, node.op_type, "Min")) return self.binary(arena, values, node, .min);
        if (std.mem.eql(u8, node.op_type, "Pow")) return self.binary(arena, values, node, .pow);
        if (std.mem.eql(u8, node.op_type, "And")) return self.logical(arena, values, node, true);
        if (std.mem.eql(u8, node.op_type, "Or")) return self.logical(arena, values, node, false);
        if (std.mem.eql(u8, node.op_type, "MatMulNBits")) return self.matmulNBits(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "Constant")) return self.constant(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "GatherND")) return self.gatherNd(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "ReduceSum")) return self.reduceSum(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "Gemm")) return self.gemm(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "Einsum")) return self.einsum(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "Resize")) return self.resize(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "InstanceNormalization")) return self.instanceNorm(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "CumSum")) return self.cumulativeSum(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "MaxPool")) return self.maxPool(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "Flatten")) return self.flatten(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "Identity")) return self.identity(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "Softmax")) return self.softmax(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "LayerNormalization")) return self.layerNorm(arena, values, node);
        return Error.UnsupportedOperator;
    }

    fn input(
        self: *SessionState,
        arena: std.mem.Allocator,
        values: *std.StringHashMapUnmanaged(*Tensor),
        name: []const u8,
    ) !*Tensor {
        if (values.get(name)) |tensor| return tensor;
        return self.materialize(arena, values, name, self.graph.constant(name) orelse return Error.MissingValue);
    }

    /// Turns a graph constant into a tensor under `name`, uploading it once if
    /// it is float data. `Constant` nodes take this path too: they carry their
    /// value as an attribute rather than as an initializer, but a graph that
    /// spells its constants that way has thousands of them and the upload
    /// should not be repeated per run either.
    fn materialize(
        self: *SessionState,
        arena: std.mem.Allocator,
        values: *std.StringHashMapUnmanaged(*Tensor),
        name: []const u8,
        source: onnx.Tensor,
    ) !*Tensor {
        const tensor = try arena.create(Tensor);
        if (source.dtype == .f32) {
            const entry = try self.constants.getOrPut(self.env.allocator, name);
            if (!entry.found_existing) {
                entry.value_ptr.* = try cuda.Buffer(f32).alloc(source.elementCount());
                errdefer _ = self.constants.remove(name);
                try entry.value_ptr.upload(source.f32s());
            }
            tensor.* = .{ .dtype = .f32, .dims = source.dims, .data = .{ .constant_gpu = entry.value_ptr.* } };
        } else {
            tensor.* = .{ .dtype = source.dtype, .dims = source.dims, .data = .{ .host = source.data } };
        }
        try values.put(arena, name, tensor);
        return tensor;
    }

    /// A quantized weight matrix is not float data and never becomes a tensor,
    /// so it goes to the device as raw bytes, once, and is addressed directly.
    fn constantBytes(self: *SessionState, name: []const u8) !cuda.DevicePtr {
        if (self.constants.get(name)) |buffer| return buffer.ptr;
        const source = self.graph.constant(name) orelse return Error.MissingValue;
        const buffer = try cuda.Buffer(f32).alloc((source.data.len + 3) / 4);
        errdefer buffer.free();
        const bytes: cuda.Buffer(u8) = .{ .ptr = buffer.ptr, .len = buffer.len * 4 };
        try bytes.upload(source.data);
        try self.constants.put(self.env.allocator, name, buffer);
        return buffer.ptr;
    }

    fn put(
        _: *SessionState,
        arena: std.mem.Allocator,
        values: *std.StringHashMapUnmanaged(*Tensor),
        name: []const u8,
        value: Tensor,
    ) !void {
        const tensor = try arena.create(Tensor);
        tensor.* = value;
        try values.put(arena, name, tensor);
    }

    fn newStorage(self: *SessionState, count: usize) !*Storage {
        const storage = try self.env.allocator.create(Storage);
        errdefer self.env.allocator.destroy(storage);
        storage.* = .{ .buffer = try self.env.pool.take(count) };
        return storage;
    }

    fn release(self: *SessionState, tensor: *Tensor) void {
        switch (tensor.data) {
            .gpu => |storage| {
                storage.references -= 1;
                if (storage.references == 0) {
                    self.env.pool.give(self.env.allocator, storage.buffer);
                    self.env.allocator.destroy(storage);
                }
            },
            else => {},
        }
    }

    fn alias(self: *SessionState, source: *Tensor, dims: []const i64) !Tensor {
        _ = self;
        switch (source.data) {
            .gpu => |storage| {
                storage.references += 1;
                return .{ .dtype = source.dtype, .dims = dims, .data = .{ .gpu = storage } };
            },
            .constant_gpu => |buffer| return .{ .dtype = source.dtype, .dims = dims, .data = .{ .constant_gpu = buffer } },
            .host => |bytes| return .{ .dtype = source.dtype, .dims = dims, .data = .{ .host = bytes } },
        }
    }

    fn conv(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node) !void {
        const x = try self.input(arena, values, node.inputs[0]);
        const weight = try self.input(arena, values, node.inputs[1]);
        if (x.dims.len != 4 or weight.dims.len != 4) return Error.InvalidShape;
        const strides = node.ints("strides");
        const pads = node.ints("pads");
        const dilations = node.ints("dilations");
        const sh: i64 = if (strides.len == 0) 1 else strides[0];
        const sw: i64 = if (strides.len == 0) 1 else strides[1];
        const ph0: i64 = if (pads.len == 0) 0 else pads[0];
        const pw0: i64 = if (pads.len == 0) 0 else pads[1];
        const ph1: i64 = if (pads.len < 4) ph0 else pads[2];
        const pw1: i64 = if (pads.len < 4) pw0 else pads[3];
        const dh: i64 = if (dilations.len == 0) 1 else dilations[0];
        const dw: i64 = if (dilations.len == 0) 1 else dilations[1];
        const out_h = @divFloor(x.dims[2] + ph0 + ph1 - dh * (weight.dims[2] - 1) - 1, sh) + 1;
        const out_w = @divFloor(x.dims[3] + pw0 + pw1 - dw * (weight.dims[3] - 1) - 1, sw) + 1;
        const dims = try arena.dupe(i64, &.{ x.dims[0], weight.dims[0], out_h, out_w });
        const count = try elementCount(dims);
        const storage = try self.newStorage(count);
        errdefer self.releaseStorage(storage);
        const groups = node.int("group", 1);
        const xb = try x.gpuBuffer();
        const wb = try weight.gpuBuffer();
        const bias = if (node.inputs.len > 2 and node.inputs[2].len != 0) try self.input(arena, values, node.inputs[2]) else null;
        const bias_buffer = if (bias) |b| try b.gpuBuffer() else xb;
        const has_bias: u32 = if (bias != null) 1 else 0;

        if (groups == 1) {
            // One group makes the convolution a matrix product, and the tiled
            // kernel is an order of magnitude faster than walking the window
            // one tap at a time.
            const meta = [_]u32{
                @intCast(x.dims[1]),      @intCast(x.dims[2]), @intCast(x.dims[3]),
                @intCast(out_h),          @intCast(out_w),     @intCast(weight.dims[2]),
                @intCast(weight.dims[3]), @intCast(sh),        @intCast(sw),
                @intCast(ph0),            @intCast(pw0),       @intCast(dh),
                @intCast(dw),
            };
            try self.env.metadata.upload(&meta);
            const rows = weight.dims[0];
            const pixels = out_h * out_w;
            const depth = weight.dims[1] * weight.dims[2] * weight.dims[3];
            try self.env.gpu.conv2d_gemm.launch(.{
                .x = @intCast(@divFloor(pixels + matmul_tile_n - 1, matmul_tile_n)),
                .y = @intCast(@divFloor(rows + matmul_tile_m - 1, matmul_tile_m)),
                .z = @intCast(x.dims[0]),
            }, .{ .x = 16, .y = 16 }, .{
                xb.ptr,
                wb.ptr,
                bias_buffer.ptr,
                storage.buffer.ptr,
                self.env.metadata.ptr,
                @as(u32, @intCast(rows)),
                @as(u32, @intCast(pixels)),
                @as(u32, @intCast(depth)),
                has_bias,
            });
            return self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
        }

        const meta = [_]u32{
            @intCast(x.dims[1]), @intCast(x.dims[2]), @intCast(x.dims[3]),      @intCast(weight.dims[0]),
            @intCast(out_h),     @intCast(out_w),     @intCast(weight.dims[2]), @intCast(weight.dims[3]),
            @intCast(sh),        @intCast(sw),        @intCast(ph0),            @intCast(pw0),
            @intCast(dh),        @intCast(dw),        @intCast(groups),
        };
        try self.env.metadata.upload(&meta);
        const block: u32 = 256;
        try self.env.gpu.conv2d.launch(.{ .x = @intCast((count + block - 1) / block) }, .{ .x = block }, .{
            xb.ptr,                    wb.ptr,   bias_buffer.ptr, storage.buffer.ptr, self.env.metadata.ptr,
            @as(u32, @intCast(count)), has_bias,
        });
        try self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
    }

    fn convTranspose(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node) !void {
        const x = try self.input(arena, values, node.inputs[0]);
        const weight = try self.input(arena, values, node.inputs[1]);
        if (x.dims.len != 4 or weight.dims.len != 4) return Error.InvalidShape;
        const strides = node.ints("strides");
        const pads = node.ints("pads");
        const sh: i64 = if (strides.len == 0) 1 else strides[0];
        const sw: i64 = if (strides.len == 0) 1 else strides[1];
        const ph0: i64 = if (pads.len == 0) 0 else pads[0];
        const pw0: i64 = if (pads.len == 0) 0 else pads[1];
        const ph1: i64 = if (pads.len < 4) ph0 else pads[2];
        const pw1: i64 = if (pads.len < 4) pw0 else pads[3];
        const out_h = (x.dims[2] - 1) * sh - ph0 - ph1 + weight.dims[2];
        const out_w = (x.dims[3] - 1) * sw - pw0 - pw1 + weight.dims[3];
        const dims = try arena.dupe(i64, &.{ x.dims[0], weight.dims[1], out_h, out_w });
        const count = try elementCount(dims);
        const storage = try self.newStorage(count);
        errdefer self.releaseStorage(storage);
        const meta = [_]u32{
            @intCast(x.dims[1]),      @intCast(x.dims[2]),      @intCast(x.dims[3]),
            @intCast(weight.dims[1]), @intCast(out_h),          @intCast(out_w),
            @intCast(weight.dims[2]), @intCast(weight.dims[3]), @intCast(sh),
            @intCast(sw),             @intCast(ph0),            @intCast(pw0),
        };
        try self.env.metadata.upload(&meta);
        const xb = try x.gpuBuffer();
        const wb = try weight.gpuBuffer();
        const bias = if (node.inputs.len > 2 and node.inputs[2].len != 0) try self.input(arena, values, node.inputs[2]) else null;
        const bias_buffer = if (bias) |b| try b.gpuBuffer() else xb;
        const block: u32 = 256;
        try self.env.gpu.conv_transpose2d.launch(.{ .x = @intCast((count + block - 1) / block) }, .{ .x = block }, .{
            xb.ptr,                    wb.ptr,                               bias_buffer.ptr, storage.buffer.ptr, self.env.metadata.ptr,
            @as(u32, @intCast(count)), @as(u32, if (bias != null) 1 else 0),
        });
        try self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
    }

    fn split(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node) !void {
        const x = try self.input(arena, values, node.inputs[0]);
        const axis = normalizeAxis(node.int("axis", 0), x.dims.len);
        var splits: []const i64 = &.{};
        if (node.inputs.len > 1 and node.inputs[1].len != 0) {
            splits = try (try self.input(arena, values, node.inputs[1])).i64s();
        } else if (node.ints("split").len != 0) {
            splits = node.ints("split");
        } else {
            const num = node.outputs.len;
            if (num == 0) return Error.InvalidShape;
            const even = try arena.alloc(i64, num);
            const part_len = @divExact(x.dims[axis], @as(i64, @intCast(num)));
            @memset(even, part_len);
            splits = even;
        }
        if (splits.len != node.outputs.len) return Error.InvalidShape;

        var dense: [8]u32 = @splat(1);
        if (x.dims.len > 1) {
            var i = x.dims.len - 1;
            while (i > 0) : (i -= 1) dense[i - 1] = dense[i] * @as(u32, @intCast(x.dims[i]));
        }

        var axis_offset: usize = 0;
        for (splits, node.outputs) |part_len, output_name| {
            const part_dims = try arena.dupe(i64, x.dims);
            part_dims[axis] = part_len;
            const part_count = try elementCount(part_dims);

            if (x.dtype == .i64) {
                const source = try x.i64s();
                const out = try arena.alloc(i64, part_count);
                for (out, 0..) |*value, output_index| {
                    var remaining = output_index;
                    var source_index: usize = 0;
                    var d = part_dims.len;
                    while (d > 0) {
                        d -= 1;
                        const coordinate = remaining % @as(usize, @intCast(part_dims[d]));
                        remaining /= @as(usize, @intCast(part_dims[d]));
                        const coord_on_x = if (d == axis) axis_offset + coordinate else coordinate;
                        source_index += coord_on_x * @as(usize, @intCast(dense[d]));
                    }
                    value.* = source[source_index];
                }
                try self.put(arena, values, output_name, .{ .dtype = .i64, .dims = part_dims, .data = .{ .host = std.mem.sliceAsBytes(out) } });
            } else if (x.dtype == .f32) {
                const storage = try self.newStorage(part_count);
                errdefer self.releaseStorage(storage);
                var metadata: [16]u32 = @splat(0);
                for (part_dims, 0..) |dim, d| {
                    metadata[d] = @intCast(dim);
                    metadata[x.dims.len + d] = dense[d];
                }
                try self.env.metadata.upload(metadata[0 .. 2 * x.dims.len]);
                const source_offset = axis_offset * dense[axis];
                const xb = try x.gpuBuffer();
                const block: u32 = 256;
                try self.env.gpu.copy.launch(.{ .x = @intCast((part_count + block - 1) / block) }, .{ .x = block }, .{
                    xb.ptr, storage.buffer.ptr, self.env.metadata.ptr, @as(u32, @intCast(x.dims.len)), @as(u32, @intCast(part_count)), @as(u32, @intCast(source_offset)), @as(u32, 0),
                });
                try self.put(arena, values, output_name, .{ .dtype = .f32, .dims = part_dims, .data = .{ .gpu = storage } });
            } else return Error.UnsupportedDataType;

            axis_offset += @intCast(part_len);
        }
    }

    fn releaseStorage(self: *SessionState, storage: *Storage) void {
        self.env.pool.give(self.env.allocator, storage.buffer);
        self.env.allocator.destroy(storage);
    }

    fn shape(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node) !void {
        const x = try self.input(arena, values, node.inputs[0]);
        const start = normalizeAxis(node.int("start", 0), x.dims.len);
        const end = normalizeAxis(node.int("end", @intCast(x.dims.len)), x.dims.len);
        const dims = try arena.dupe(i64, &.{@as(i64, @intCast(end - start))});
        const data = try arena.dupe(i64, x.dims[start..end]);
        try self.put(arena, values, node.outputs[0], .{ .dtype = .i64, .dims = dims, .data = .{ .host = std.mem.sliceAsBytes(data) } });
    }

    /// out[outer, index..., inner] = data[outer, indices[index...], inner].
    /// The indices themselves are shape arithmetic and so always on the host;
    /// the data being gathered can be on either side.
    fn gather(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node) !void {
        const data = try self.input(arena, values, node.inputs[0]);
        const indices = try self.input(arena, values, node.inputs[1]);
        if (data.dims.len == 0) return Error.InvalidShape;
        const axis = normalizeAxis(node.int("axis", 0), data.dims.len);
        const ids = try indices.i64s();

        const axis_len: usize = @intCast(data.dims[axis]);
        const outer = try elementCount(data.dims[0..axis]);
        const inner = try elementCount(data.dims[axis + 1 ..]);

        // A scalar index contributes no dimensions, which is how Gather is
        // usually spelled when it is picking one entry out of a Shape.
        const dims = try arena.alloc(i64, data.dims.len - 1 + indices.dims.len);
        @memcpy(dims[0..axis], data.dims[0..axis]);
        @memcpy(dims[axis..][0..indices.dims.len], indices.dims);
        @memcpy(dims[axis + indices.dims.len ..], data.dims[axis + 1 ..]);
        const count = try elementCount(dims);

        // Resolve the negative indices once: both paths want them folded away,
        // and the kernel takes them unsigned.
        const resolved = try arena.alloc(u32, ids.len);
        for (ids, resolved) |index, *out| {
            const normalized = if (index < 0) @as(i64, @intCast(axis_len)) + index else index;
            if (normalized < 0 or normalized >= axis_len) return Error.InvalidShape;
            out.* = @intCast(normalized);
        }

        if (data.onHost()) {
            const sources = try arena.alloc(usize, count);
            for (sources, 0..) |*source, i| {
                const inner_index = i % inner;
                const index = (i / inner) % resolved.len;
                const outer_index = i / (inner * resolved.len);
                source.* = (outer_index * axis_len + resolved[index]) * inner + inner_index;
            }
            const out = try hostSelect(arena, data.*, sources);
            return self.put(arena, values, node.outputs[0], .{ .dtype = data.dtype, .dims = dims, .data = .{ .host = out } });
        }

        const storage = try self.newStorage(count);
        errdefer self.releaseStorage(storage);
        try self.ensureIndices(resolved.len);
        try self.env.indices.upload(resolved);
        const db = try data.gpuBuffer();
        const block: u32 = 256;
        try self.env.gpu.gather.launch(.{ .x = @intCast((count + block - 1) / block) }, .{ .x = block }, .{
            db.ptr,                           self.env.indices.ptr,         storage.buffer.ptr,
            @as(u32, @intCast(outer)),        @as(u32, @intCast(axis_len)), @as(u32, @intCast(inner)),
            @as(u32, @intCast(resolved.len)), @as(u32, @intCast(count)),
        });
        try self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
    }

    /// Index arrays are not bounded the way shapes are, so the device-side
    /// buffer they are staged in follows the largest one seen so far.
    fn ensureIndices(self: *SessionState, count: usize) !void {
        if (self.env.indices.len >= count) return;
        if (self.env.indices.len != 0) self.env.indices.free();
        self.env.indices = try cuda.Buffer(u32).alloc(count);
    }

    fn reshape(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node) !void {
        const x = try self.input(arena, values, node.inputs[0]);
        const requested = try (try self.input(arena, values, node.inputs[1])).i64s();
        const dims = try arena.alloc(i64, requested.len);
        var known: usize = 1;
        var inferred: ?usize = null;
        for (requested, dims, 0..) |dim, *out, i| {
            if (dim == -1) {
                if (inferred != null) return Error.InvalidShape;
                inferred = i;
                out.* = 1;
            } else if (dim == 0 and node.int("allowzero", 0) == 0) {
                if (i >= x.dims.len) return Error.InvalidShape;
                out.* = x.dims[i];
                known *= @intCast(out.*);
            } else {
                if (dim < 0) return Error.InvalidShape;
                out.* = dim;
                known *= @intCast(dim);
            }
        }
        if (inferred) |i| dims[i] = @intCast(try x.count() / known);
        if (try elementCount(dims) != try x.count()) return Error.InvalidShape;
        try self.put(arena, values, node.outputs[0], try self.alias(x, dims));
    }

    fn unsqueeze(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node) !void {
        const x = try self.input(arena, values, node.inputs[0]);
        const axes = if (node.inputs.len > 1) try (try self.input(arena, values, node.inputs[1])).i64s() else node.ints("axes");
        const rank = x.dims.len + axes.len;
        const dims = try arena.alloc(i64, rank);
        var source: usize = 0;
        for (dims, 0..) |*dim, i| {
            var inserted = false;
            for (axes) |axis| if (normalizeAxisInsert(axis, rank) == i) {
                inserted = true;
                break;
            };
            if (inserted) dim.* = 1 else {
                dim.* = x.dims[source];
                source += 1;
            }
        }
        try self.put(arena, values, node.outputs[0], try self.alias(x, dims));
    }

    fn squeeze(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node) !void {
        const x = try self.input(arena, values, node.inputs[0]);
        const axes = if (node.inputs.len > 1) try (try self.input(arena, values, node.inputs[1])).i64s() else node.ints("axes");
        var dims_list: std.ArrayList(i64) = .empty;
        for (x.dims, 0..) |dim, i| {
            var remove = axes.len == 0 and dim == 1;
            for (axes) |axis| {
                if (normalizeAxis(axis, x.dims.len) == i) remove = true;
            }
            if (remove) {
                if (dim != 1) return Error.InvalidShape;
            } else try dims_list.append(arena, dim);
        }
        try self.put(arena, values, node.outputs[0], try self.alias(x, dims_list.items));
    }

    fn transpose(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node) !void {
        const x = try self.input(arena, values, node.inputs[0]);
        if (x.dims.len > 8) return Error.RankTooLarge;
        const perm_attr = node.ints("perm");
        const rank = x.dims.len;
        const dims = try arena.alloc(i64, rank);
        var source_strides: [8]usize = @splat(0);
        var dense: [8]usize = @splat(1);
        if (rank > 1) {
            var i = rank - 1;
            while (i > 0) : (i -= 1) dense[i - 1] = dense[i] * @as(usize, @intCast(x.dims[i]));
        }
        for (0..rank) |i| {
            const source_axis: usize = if (perm_attr.len == 0) rank - 1 - i else @intCast(perm_attr[i]);
            dims[i] = x.dims[source_axis];
            source_strides[i] = dense[source_axis];
        }
        const count = try x.count();
        if (x.dtype == .i64) {
            const source = try x.i64s();
            const out = try arena.alloc(i64, count);
            for (out, 0..) |*value, output_index| {
                var remaining = output_index;
                var source_index: usize = 0;
                var axis = rank;
                while (axis > 0) {
                    axis -= 1;
                    const coordinate = remaining % @as(usize, @intCast(dims[axis]));
                    remaining /= @as(usize, @intCast(dims[axis]));
                    source_index += coordinate * source_strides[axis];
                }
                value.* = source[source_index];
            }
            return self.put(arena, values, node.outputs[0], .{ .dtype = .i64, .dims = dims, .data = .{ .host = std.mem.sliceAsBytes(out) } });
        }
        if (x.dtype != .f32) return Error.UnsupportedSliceType;
        const storage = try self.newStorage(count);
        errdefer self.releaseStorage(storage);
        var metadata: [16]u32 = @splat(0);
        for (dims, 0..) |dim, i| metadata[i] = @intCast(dim);
        for (0..rank) |i| metadata[rank + i] = @intCast(source_strides[i]);
        try self.env.metadata.upload(metadata[0 .. 2 * rank]);
        const xb = try x.gpuBuffer();
        const block: u32 = 256;
        try self.env.gpu.copy.launch(.{ .x = @intCast((count + block - 1) / block) }, .{ .x = block }, .{
            xb.ptr,                    storage.buffer.ptr, self.env.metadata.ptr, @as(u32, @intCast(rank)),
            @as(u32, @intCast(count)), @as(u32, 0),        @as(u32, 0),
        });
        try self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
    }

    fn concat(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node) !void {
        if (node.inputs.len == 0) return Error.InvalidShape;
        const first = try self.input(arena, values, node.inputs[0]);
        const axis = normalizeAxis(node.int("axis", 0), first.dims.len);
        if (first.onHost()) {
            const dims = try arena.dupe(i64, first.dims);
            dims[axis] = 0;
            for (node.inputs) |name| {
                const part = try self.input(arena, values, name);
                if (part.dims.len != dims.len) return Error.InvalidShape;
                dims[axis] += part.dims[axis];
            }
            const outer = try elementCount(dims[0..axis]);
            const inner = try elementCount(dims[axis + 1 ..]);
            const total_axis: usize = @intCast(dims[axis]);

            // Whole runs of `inner` elements move at once, which makes the
            // copy independent of what the elements are.
            const size = first.dtype.size();
            if (size == 0) return Error.UnsupportedDataType;
            const out = try arena.alignedAlloc(u8, .@"8", try elementCount(dims) * size);
            var axis_offset: usize = 0;
            for (node.inputs) |name| {
                const part = try self.input(arena, values, name);
                if (part.dtype != first.dtype) return Error.UnsupportedDataType;
                for (part.dims, 0..) |dim, i| if (i != axis and dim != dims[i]) return Error.InvalidShape;
                const part_axis: usize = @intCast(part.dims[axis]);
                const stripe = inner * size;
                for (0..outer) |o| for (0..part_axis) |a| {
                    const from = (o * part_axis + a) * stripe;
                    const to = (o * total_axis + axis_offset + a) * stripe;
                    @memcpy(out[to..][0..stripe], part.data.host[from..][0..stripe]);
                };
                axis_offset += part_axis;
            }
            return self.put(arena, values, node.outputs[0], .{ .dtype = first.dtype, .dims = dims, .data = .{ .host = out } });
        }
        if (first.dtype != .f32) return Error.UnsupportedDataType;
        const dims = try arena.dupe(i64, first.dims);
        dims[axis] = 0;
        for (node.inputs) |name| {
            const part = try self.input(arena, values, name);
            if (part.dtype != .f32 or part.dims.len != dims.len) return Error.InvalidShape;
            for (part.dims, 0..) |dim, i| if (i != axis and dim != first.dims[i]) return Error.InvalidShape;
            dims[axis] += part.dims[axis];
        }
        const count = try elementCount(dims);
        const storage = try self.newStorage(count);
        errdefer self.releaseStorage(storage);
        var inner: usize = 1;
        for (dims[axis + 1 ..]) |dim| inner *= @intCast(dim);
        var axis_offset: usize = 0;
        for (node.inputs) |name| {
            const part = try self.input(arena, values, name);
            const part_count = try part.count();
            const part_buffer = try part.gpuBuffer();
            const block: u32 = 256;
            try self.env.gpu.concat_copy.launch(.{ .x = @intCast((part_count + block - 1) / block) }, .{ .x = block }, .{
                part_buffer.ptr,                storage.buffer.ptr,        @as(u32, @intCast(part.dims[axis])),
                @as(u32, @intCast(dims[axis])), @as(u32, @intCast(inner)), @as(u32, @intCast(axis_offset)),
                @as(u32, @intCast(part_count)),
            });
            axis_offset += @intCast(part.dims[axis]);
        }
        try self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
    }

    fn slice(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node) !void {
        const x = try self.input(arena, values, node.inputs[0]);
        const starts = try (try self.input(arena, values, node.inputs[1])).i64s();
        const ends = try (try self.input(arena, values, node.inputs[2])).i64s();
        const axes = if (node.inputs.len > 3 and node.inputs[3].len != 0) try (try self.input(arena, values, node.inputs[3])).i64s() else node.ints("axes");
        const steps = if (node.inputs.len > 4 and node.inputs[4].len != 0) try (try self.input(arena, values, node.inputs[4])).i64s() else node.ints("steps");
        if (starts.len != ends.len) return Error.InvalidShape;
        const dims = try arena.dupe(i64, x.dims);
        var offsets: [8]i64 = @splat(0);
        var step_by_axis: [8]i64 = @splat(1);
        for (starts, ends, 0..) |raw_start, raw_end, i| {
            const axis = if (axes.len == 0) i else normalizeAxis(axes[i], x.dims.len);
            const step = if (steps.len == 0) 1 else steps[i];
            if (step == 0) return Error.UnsupportedSliceStep;
            const dim = x.dims[axis];
            if (step > 0) {
                const start = @max(0, @min(dim, if (raw_start < 0) dim + raw_start else raw_start));
                const end = @max(0, @min(dim, if (raw_end < 0) dim + raw_end else raw_end));
                offsets[axis] = start;
                step_by_axis[axis] = step;
                dims[axis] = if (end <= start) 0 else @divFloor(end - start + step - 1, step);
            } else {
                const start = @max(-1, @min(dim - 1, if (raw_start < 0) dim + raw_start else raw_start));
                const end = @max(-1, @min(dim - 1, if (raw_end < 0) dim + raw_end else raw_end));
                offsets[axis] = start;
                step_by_axis[axis] = step;
                dims[axis] = if (start <= end) 0 else @divFloor(start - end + (-step) - 1, -step);
            }
        }
        if (x.dtype == .i64) {
            const source = try x.i64s();
            const out = try arena.alloc(i64, try elementCount(dims));
            var dense: [8]usize = @splat(1);
            if (x.dims.len > dense.len) return Error.RankTooLarge;
            if (x.dims.len > 1) {
                var i = x.dims.len - 1;
                while (i > 0) : (i -= 1) dense[i - 1] = dense[i] * @as(usize, @intCast(x.dims[i]));
            }
            for (out, 0..) |*value, output_index| {
                var remaining = output_index;
                var source_index: usize = 0;
                var axis = dims.len;
                while (axis > 0) {
                    axis -= 1;
                    const coordinate = remaining % @as(usize, @intCast(dims[axis]));
                    remaining /= @as(usize, @intCast(dims[axis]));
                    const idx = offsets[axis] + @as(i64, @intCast(coordinate)) * step_by_axis[axis];
                    source_index += @as(usize, @intCast(idx)) * dense[axis];
                }
                value.* = source[source_index];
            }
            return self.put(arena, values, node.outputs[0], .{ .dtype = .i64, .dims = dims, .data = .{ .host = std.mem.sliceAsBytes(out) } });
        }
        if (x.dtype != .f32 or x.dims.len > 8) return Error.UnsupportedOperator;
        var dense: [8]u32 = @splat(1);
        if (x.dims.len > 1) {
            var i = x.dims.len - 1;
            while (i > 0) : (i -= 1) dense[i - 1] = dense[i] * @as(u32, @intCast(x.dims[i]));
        }
        var source_offset: usize = 0;
        var metadata: [16]u32 = @splat(0);
        for (dims, 0..) |dim, i| {
            metadata[i] = @intCast(dim);
            const stride: i64 = @as(i64, @intCast(dense[i])) * step_by_axis[i];
            metadata[x.dims.len + i] = @bitCast(@as(i32, @intCast(stride)));
            source_offset += @as(usize, @intCast(offsets[i])) * dense[i];
        }
        const count = try elementCount(dims);
        const storage = try self.newStorage(count);
        errdefer self.releaseStorage(storage);
        try self.env.metadata.upload(metadata[0 .. 2 * x.dims.len]);
        const xb = try x.gpuBuffer();
        const block: u32 = 256;
        try self.env.gpu.copy.launch(.{ .x = @intCast((count + block - 1) / block) }, .{ .x = block }, .{
            xb.ptr, storage.buffer.ptr, self.env.metadata.ptr, @as(u32, @intCast(x.dims.len)), @as(u32, @intCast(count)), @as(u32, @intCast(source_offset)), @as(u32, 0),
        });
        try self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
    }

    fn pad(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node) !void {
        const x = try self.input(arena, values, node.inputs[0]);
        const pads = try (try self.input(arena, values, node.inputs[1])).i64s();
        if (x.dtype != .f32 or pads.len != 2 * x.dims.len or x.dims.len > 8) return Error.UnsupportedOperator;
        const dims = try arena.alloc(i64, x.dims.len);
        var metadata: [32]u32 = @splat(0);
        var stride: u32 = 1;
        var axis = x.dims.len;
        while (axis > 0) {
            axis -= 1;
            metadata[2 * x.dims.len + axis] = stride;
            stride *= @intCast(x.dims[axis]);
        }
        for (x.dims, 0..) |dim, i| {
            dims[i] = dim + pads[i] + pads[x.dims.len + i];
            if (dims[i] < 0) return Error.InvalidShape;
            metadata[i] = @intCast(dims[i]);
            metadata[x.dims.len + i] = @intCast(dim);
            metadata[3 * x.dims.len + i] = @intCast(pads[i]);
        }
        const count = try elementCount(dims);
        const storage = try self.newStorage(count);
        errdefer self.releaseStorage(storage);
        try self.env.metadata.upload(metadata[0 .. 4 * x.dims.len]);
        const xb = try x.gpuBuffer();
        const block: u32 = 256;
        const fill = if (node.inputs.len > 2 and node.inputs[2].len != 0)
            try (try self.input(arena, values, node.inputs[2])).gpuBuffer()
        else
            null;
        // The kernel ignores an absent fill, but still wants a valid pointer.
        try self.env.gpu.pad.launch(.{ .x = @intCast((count + block - 1) / block) }, .{ .x = block }, .{
            xb.ptr,
            storage.buffer.ptr,
            self.env.metadata.ptr,
            @as(u32, @intCast(x.dims.len)),
            @as(u32, @intCast(count)),
            if (fill) |buffer| buffer.ptr else xb.ptr,
            @as(u32, if (fill != null) 1 else 0),
        });
        try self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
    }

    fn matmul(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node) !void {
        const a = try self.input(arena, values, node.inputs[0]);
        const b = try self.input(arena, values, node.inputs[1]);
        if (a.dtype != .f32 or b.dtype != .f32 or a.dims.len < 2 or b.dims.len < 2) return Error.UnsupportedOperator;
        const m = a.dims[a.dims.len - 2];
        const k = a.dims[a.dims.len - 1];
        const bk = b.dims[b.dims.len - 2];
        const n = b.dims[b.dims.len - 1];
        if (k != bk) return Error.InvalidShape;
        const batch_dims = try broadcastShape(arena, a.dims[0 .. a.dims.len - 2], b.dims[0 .. b.dims.len - 2]);
        const dims = try arena.alloc(i64, batch_dims.len + 2);
        @memcpy(dims[0..batch_dims.len], batch_dims);
        dims[dims.len - 2] = m;
        dims[dims.len - 1] = n;
        const batches = try elementCount(batch_dims);
        const count = try elementCount(dims);
        const storage = try self.newStorage(count);
        errdefer self.releaseStorage(storage);
        const ab = try a.gpuBuffer();
        const bb = try b.gpuBuffer();
        const a_batches = try elementCount(a.dims[0 .. a.dims.len - 2]);
        const b_batches = try elementCount(b.dims[0 .. b.dims.len - 2]);
        if (a_batches != 1 and a_batches != batches or b_batches != 1 and b_batches != batches) return Error.UnsupportedOperator;

        // A batch that only repeats over A is a taller matrix, not a batch:
        // the batches sit end to end in both A and C, so stacking them costs
        // nothing and gains a lot. SAM 3's windowed attention projects 24 or
        // 72 rows at a time against a shared weight, and rows that few leave
        // most of a block tile idle and re-read the weight once per window.
        var rows = m;
        var grid_batches = batches;
        if (b_batches == 1 and a_batches == batches) {
            rows = m * @as(i64, @intCast(batches));
            grid_batches = 1;
        }

        try self.env.gpu.matmul.launch(.{
            .x = @intCast(@divFloor(n + matmul_tile_n - 1, matmul_tile_n)),
            .y = @intCast(@divFloor(rows + matmul_tile_m - 1, matmul_tile_m)),
            .z = @intCast(grid_batches),
        }, .{ .x = 16, .y = 16 }, .{
            ab.ptr,
            bb.ptr,
            storage.buffer.ptr,
            @as(u32, @intCast(rows)),
            @as(u32, @intCast(n)),
            @as(u32, @intCast(k)),
            @as(u32, if (a_batches == 1) 0 else @intCast(m * k)),
            @as(u32, if (b_batches == 1) 0 else @intCast(k * n)),
            @as(u32, @intCast(m * n)),
            @as(u32, 0),
        });
        try self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
    }

    fn unary(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node, op: Unary) !void {
        const x = try self.input(arena, values, node.inputs[0]);
        if (x.dtype != .f32) return Error.UnsupportedDataType;
        const count = try x.count();
        const storage = try self.newStorage(count);
        errdefer self.releaseStorage(storage);
        const xb = try x.gpuBuffer();
        const block: u32 = 256;
        try self.env.gpu.unary.launch(.{ .x = @intCast((count + block - 1) / block) }, .{ .x = block }, .{ xb.ptr, storage.buffer.ptr, @as(u32, @intCast(count)), @intFromEnum(op) });
        const dims = try arena.dupe(i64, x.dims);
        try self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
    }

    fn softmax(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node) !void {
        const x = try self.input(arena, values, node.inputs[0]);
        const axis = normalizeAxis(node.int("axis", -1), x.dims.len);
        if (x.dtype != .f32 or axis + 1 != x.dims.len) return Error.UnsupportedOperator;
        const cols: usize = @intCast(x.dims[axis]);
        const count = try x.count();
        const rows = count / cols;
        const storage = try self.newStorage(count);
        errdefer self.releaseStorage(storage);
        const xb = try x.gpuBuffer();
        // A power of two, since the tree reduction halves it, and capped well
        // below the hardware maximum: a row is reduced twice, and each step of
        // that costs a barrier, so a wide block spends more on the reduction
        // than on the row. A narrow one just loops over more of the row.
        var block: u32 = 32;
        while (block < cols and block < 256) block *= 2;
        try self.env.gpu.softmax.launch(.{ .x = @intCast(rows) }, .{ .x = block }, .{ xb.ptr, storage.buffer.ptr, @as(u32, @intCast(cols)) });
        const dims = try arena.dupe(i64, x.dims);
        try self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
    }

    fn binary(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node, op: Binary) !void {
        const a = try self.input(arena, values, node.inputs[0]);
        const b = try self.input(arena, values, node.inputs[1]);
        if (a.dtype == .i64 and b.dtype == .i64) {
            const dims = try broadcastShape(arena, a.dims, b.dims);
            if (dims.len > max_rank) return Error.RankTooLarge;
            var astrides: [max_rank]u32 = @splat(0);
            var bstrides: [max_rank]u32 = @splat(0);
            makeBroadcastStrides(a.dims, dims, &astrides);
            makeBroadcastStrides(b.dims, dims, &bstrides);
            const av = try a.i64s();
            const bv = try b.i64s();
            const out = try arena.alloc(i64, try elementCount(dims));
            const modulo = std.mem.eql(u8, node.op_type, "Mod");
            for (out, 0..) |*value, i| {
                const x = av[hostOffset(i, dims, &astrides)];
                const y = bv[hostOffset(i, dims, &bstrides)];
                value.* = switch (op) {
                    .add => x + y,
                    .sub => x - y,
                    .mul => x * y,
                    .div => if (modulo) @mod(x, y) else @divFloor(x, y),
                    .min => @min(x, y),
                    .max => @max(x, y),
                    else => return Error.UnsupportedOperator,
                };
            }
            return self.put(arena, values, node.outputs[0], .{ .dtype = .i64, .dims = dims, .data = .{ .host = std.mem.sliceAsBytes(out) } });
        }
        if (a.dtype != .f32 or b.dtype != .f32) return Error.UnsupportedDataType;
        const rank = @max(a.dims.len, b.dims.len);
        if (rank > 8) return Error.InvalidShape;
        const dims = try broadcastShape(arena, a.dims, b.dims);
        const count = try elementCount(dims);
        const storage = try self.newStorage(count);
        errdefer self.releaseStorage(storage);
        var metadata: [24]u32 = @splat(0);
        var astrides: [8]u32 = @splat(0);
        var bstrides: [8]u32 = @splat(0);
        makeBroadcastStrides(a.dims, dims, &astrides);
        makeBroadcastStrides(b.dims, dims, &bstrides);
        for (dims, 0..) |dim, i| metadata[i] = @intCast(dim);
        @memcpy(metadata[rank .. 2 * rank], astrides[0..rank]);
        @memcpy(metadata[2 * rank .. 3 * rank], bstrides[0..rank]);
        try self.env.metadata.upload(metadata[0 .. 3 * rank]);
        // Neither side broadcast means the kernel can index straight through.
        const walk: u32 = if (std.mem.eql(i64, a.dims, dims) and std.mem.eql(i64, b.dims, dims)) 0 else @intCast(rank);
        const ab = try a.gpuBuffer();
        const bb = try b.gpuBuffer();
        const block: u32 = 256;
        try self.env.gpu.binary.launch(.{ .x = @intCast((count + block - 1) / block) }, .{ .x = block }, .{
            ab.ptr, bb.ptr, storage.buffer.ptr, self.env.metadata.ptr, walk, @as(u32, @intCast(count)), @intFromEnum(op),
        });
        try self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
    }

    fn layerNorm(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node) !void {
        const x = try self.input(arena, values, node.inputs[0]);
        const scale = try self.input(arena, values, node.inputs[1]);
        const bias = if (node.inputs.len > 2 and node.inputs[2].len != 0) try self.input(arena, values, node.inputs[2]) else null;
        const axis = normalizeAxis(node.int("axis", -1), x.dims.len);
        var cols: usize = 1;
        for (x.dims[axis..]) |dim| cols *= @intCast(dim);
        const count = try x.count();
        const rows = count / cols;
        const storage = try self.newStorage(count);
        errdefer self.releaseStorage(storage);
        const xb = try x.gpuBuffer();
        const sb = try scale.gpuBuffer();
        const bb = if (bias) |b| try b.gpuBuffer() else xb;
        // A power of two, since the tree reduction halves it, and capped well
        // below the hardware maximum: a row is reduced twice, and each step of
        // that costs a barrier, so a wide block spends more on the reduction
        // than on the row. A narrow one just loops over more of the row.
        var block: u32 = 32;
        while (block < cols and block < 256) block *= 2;
        try self.env.gpu.layer_norm.launch(.{ .x = @intCast(rows) }, .{ .x = block }, .{
            xb.ptr, sb.ptr, bb.ptr, storage.buffer.ptr, @as(u32, @intCast(cols)), node.float("epsilon", 1e-5), @as(u32, if (bias != null) 1 else 0),
        });
        const dims = try arena.dupe(i64, x.dims);
        try self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
    }

    /// out[i...] = data[indices[i..., :]], the read that ScatterND writes.
    /// The index tuples are shape arithmetic here, so this stays on the host.
    fn gatherNd(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node) !void {
        const data = try self.input(arena, values, node.inputs[0]);
        const indices = try self.input(arena, values, node.inputs[1]);
        if (node.int("batch_dims", 0) != 0) return Error.UnsupportedOperator;
        if (!data.onHost() or indices.dims.len == 0 or data.dims.len > max_rank) return Error.UnsupportedDataType;

        const ids = try indices.i64s();
        const tuple: usize = @intCast(indices.dims[indices.dims.len - 1]);
        if (tuple > data.dims.len) return Error.InvalidShape;
        const tuples = try elementCount(indices.dims[0 .. indices.dims.len - 1]);
        const span = try elementCount(data.dims[tuple..]);

        var strides: [max_rank]u32 = @splat(0);
        denseStrides(data.dims, &strides);
        const sources = try arena.alloc(usize, tuples * span);
        for (0..tuples) |t| {
            var flat: usize = 0;
            for (0..tuple) |d| {
                const raw = ids[t * tuple + d];
                const normalized = if (raw < 0) data.dims[d] + raw else raw;
                if (normalized < 0 or normalized >= data.dims[d]) return Error.InvalidShape;
                flat += @as(usize, @intCast(normalized)) * strides[d];
            }
            for (0..span) |s| sources[t * span + s] = flat + s;
        }

        const dims = try arena.alloc(i64, indices.dims.len - 1 + data.dims.len - tuple);
        @memcpy(dims[0 .. indices.dims.len - 1], indices.dims[0 .. indices.dims.len - 1]);
        @memcpy(dims[indices.dims.len - 1 ..], data.dims[tuple..]);
        const out = try hostSelect(arena, data.*, sources);
        try self.put(arena, values, node.outputs[0], .{ .dtype = data.dtype, .dims = dims, .data = .{ .host = out } });
    }

    fn reduceSum(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node) !void {
        const x = try self.input(arena, values, node.inputs[0]);
        if (x.dims.len > max_rank) return Error.RankTooLarge;
        const axes = if (node.inputs.len > 1 and node.inputs[1].len != 0)
            try (try self.input(arena, values, node.inputs[1])).i64s()
        else
            node.ints("axes");
        if (axes.len == 0) return Error.UnsupportedOperator;

        var strides: [max_rank]u32 = @splat(0);
        denseStrides(x.dims, &strides);

        // The kept axes keep their extent; a reduced one collapses to a single
        // output element that sweeps the extent it had.
        const kept = try arena.dupe(i64, x.dims);
        var collapsed: [max_rank]bool = @splat(false);
        var swept_dims: [max_rank]u32 = @splat(0);
        var swept_strides: [max_rank]u32 = @splat(0);
        var reduced: usize = 0;
        var swept: usize = 1;
        for (axes) |raw| {
            const axis = normalizeAxis(raw, x.dims.len);
            if (collapsed[axis]) return Error.InvalidShape;
            collapsed[axis] = true;
            swept_dims[reduced] = @intCast(x.dims[axis]);
            swept_strides[reduced] = strides[axis];
            swept *= @intCast(x.dims[axis]);
            kept[axis] = 1;
            reduced += 1;
        }

        // The reduced shape follows the kept one, however many axes each
        // turned out to have.
        const rank = x.dims.len;
        var metadata: [4 * max_rank]u32 = @splat(0);
        for (kept, 0..) |dim, i| metadata[i] = @intCast(dim);
        @memcpy(metadata[rank..][0..rank], strides[0..rank]);
        @memcpy(metadata[2 * rank ..][0..reduced], swept_dims[0..reduced]);
        @memcpy(metadata[2 * rank + reduced ..][0..reduced], swept_strides[0..reduced]);
        try self.env.metadata.upload(metadata[0 .. 2 * rank + 2 * reduced]);

        const count = try elementCount(kept);
        const storage = try self.newStorage(count);
        errdefer self.releaseStorage(storage);
        const xb = try x.gpuBuffer();
        const block: u32 = 256;
        try self.env.gpu.sum_axes.launch(.{ .x = @intCast((count + block - 1) / block) }, .{ .x = block }, .{
            xb.ptr,
            storage.buffer.ptr,
            self.env.metadata.ptr,
            @as(u32, @intCast(rank)),
            @as(u32, @intCast(reduced)),
            @as(u32, @intCast(swept)),
            @as(u32, @intCast(count)),
        });

        var dims = kept;
        if (node.int("keepdims", 1) == 0) {
            var shrunk: std.ArrayList(i64) = .empty;
            for (kept, 0..) |dim, axis| if (!collapsed[axis]) try shrunk.append(arena, dim);
            dims = shrunk.items;
        }
        try self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
    }

    /// Y = alpha * A x B + beta * C, which these exports only ever spell with
    /// both factors at one and B transposed.
    fn gemm(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node) !void {
        const a = try self.input(arena, values, node.inputs[0]);
        const b = try self.input(arena, values, node.inputs[1]);
        if (node.float("alpha", 1) != 1 or node.float("beta", 1) != 1) return Error.UnsupportedOperator;
        if (node.int("transA", 0) != 0) return Error.UnsupportedOperator;
        if (a.dims.len != 2 or b.dims.len != 2) return Error.InvalidShape;

        const transposed = node.int("transB", 0) != 0;
        const m = a.dims[0];
        const k = a.dims[1];
        const n = if (transposed) b.dims[0] else b.dims[1];
        if ((if (transposed) b.dims[1] else b.dims[0]) != k) return Error.InvalidShape;

        const dims = try arena.dupe(i64, &.{ m, n });
        const count = try elementCount(dims);
        const storage = try self.newStorage(count);
        errdefer self.releaseStorage(storage);
        const ab = try a.gpuBuffer();
        const bb = try b.gpuBuffer();
        try self.env.gpu.matmul.launch(.{
            .x = @intCast(@divFloor(n + matmul_tile_n - 1, matmul_tile_n)),
            .y = @intCast(@divFloor(m + matmul_tile_m - 1, matmul_tile_m)),
        }, .{ .x = 16, .y = 16 }, .{
            ab.ptr,                             bb.ptr,                storage.buffer.ptr,
            @as(u32, @intCast(m)),              @as(u32, @intCast(n)), @as(u32, @intCast(k)),
            @as(u32, 0),                        @as(u32, 0),           @as(u32, @intCast(m * n)),
            @as(u32, if (transposed) 1 else 0),
        });

        if (node.inputs.len > 2 and node.inputs[2].len != 0) {
            const c = try self.input(arena, values, node.inputs[2]);
            var cstrides: [max_rank]u32 = @splat(0);
            makeBroadcastStrides(c.dims, dims, &cstrides);
            var metadata: [3 * max_rank]u32 = @splat(0);
            metadata[0] = @intCast(m);
            metadata[1] = @intCast(n);
            metadata[2] = @intCast(n);
            metadata[3] = 1;
            metadata[4] = cstrides[0];
            metadata[5] = cstrides[1];
            try self.env.metadata.upload(metadata[0..6]);
            // The sum lands back where the product is: each thread reads and
            // writes the one element it owns.
            const cb = try c.gpuBuffer();
            const block: u32 = 256;
            try self.env.gpu.binary.launch(.{ .x = @intCast((count + block - 1) / block) }, .{ .x = block }, .{
                storage.buffer.ptr, cb.ptr,                    storage.buffer.ptr,       self.env.metadata.ptr,
                @as(u32, 2),        @as(u32, @intCast(count)), @intFromEnum(Binary.add),
            });
        }
        try self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
    }

    /// Einsum, for the equations that are a batched matrix product: one index
    /// summed away, ending the left operand and following the batch on the
    /// right, with the batch leading all three in the same order. Anything
    /// else -- a transpose written into the equation, more than one contracted
    /// index -- would not reduce to the kernel this hands off to.
    fn einsum(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node) !void {
        const attribute = node.attribute("equation") orelse return Error.UnsupportedOperator;
        if (node.inputs.len != 2) return Error.UnsupportedOperator;

        var buffer: [64]u8 = undefined;
        var length: usize = 0;
        for (attribute.s) |letter| {
            if (letter == ' ') continue;
            if (length == buffer.len) return Error.UnsupportedOperator;
            buffer[length] = letter;
            length += 1;
        }
        const equation = buffer[0..length];
        const arrow = std.mem.indexOf(u8, equation, "->") orelse return Error.UnsupportedOperator;
        const comma = std.mem.indexOfScalar(u8, equation[0..arrow], ',') orelse return Error.UnsupportedOperator;
        const left = equation[0..comma];
        const right = equation[comma + 1 .. arrow];
        const result = equation[arrow + 2 ..];

        const a = try self.input(arena, values, node.inputs[0]);
        const b = try self.input(arena, values, node.inputs[1]);
        if (a.dims.len != left.len or b.dims.len != right.len) return Error.InvalidShape;

        // The summed index is the one both operands carry and the result drops.
        var contracted: ?u8 = null;
        for (left) |letter| {
            if (std.mem.indexOfScalar(u8, right, letter) == null) continue;
            if (std.mem.indexOfScalar(u8, result, letter) != null) continue;
            if (contracted != null) return Error.UnsupportedOperator;
            contracted = letter;
        }
        const summed = contracted orelse return Error.UnsupportedOperator;
        if (left[left.len - 1] != summed) return Error.UnsupportedOperator;

        const batch = std.mem.indexOfScalar(u8, right, summed).?;
        const free_left = left[batch .. left.len - 1];
        const free_right = right[batch + 1 ..];
        if (!std.mem.eql(u8, left[0..batch], right[0..batch])) return Error.UnsupportedOperator;
        if (result.len != batch + free_left.len + free_right.len) return Error.UnsupportedOperator;
        if (!std.mem.eql(u8, result[0..batch], left[0..batch])) return Error.UnsupportedOperator;
        if (!std.mem.eql(u8, result[batch..][0..free_left.len], free_left)) return Error.UnsupportedOperator;
        if (!std.mem.eql(u8, result[batch + free_left.len ..], free_right)) return Error.UnsupportedOperator;

        for (a.dims[0..batch], b.dims[0..batch]) |x, y| if (x != y) return Error.InvalidShape;
        const batches = try elementCount(a.dims[0..batch]);
        const m = try elementCount(a.dims[batch .. a.dims.len - 1]);
        const k = a.dims[a.dims.len - 1];
        if (b.dims[batch] != k) return Error.InvalidShape;
        const n = try elementCount(b.dims[batch + 1 ..]);

        const dims = try arena.alloc(i64, result.len);
        @memcpy(dims[0..batch], a.dims[0..batch]);
        @memcpy(dims[batch..][0..free_left.len], a.dims[batch .. a.dims.len - 1]);
        @memcpy(dims[batch + free_left.len ..], b.dims[batch + 1 ..]);

        const storage = try self.newStorage(try elementCount(dims));
        errdefer self.releaseStorage(storage);
        const ab = try a.gpuBuffer();
        const bb = try b.gpuBuffer();
        try self.env.gpu.matmul.launch(.{
            .x = @intCast(@divFloor(n + matmul_tile_n - 1, matmul_tile_n)),
            .y = @intCast(@divFloor(m + matmul_tile_m - 1, matmul_tile_m)),
            .z = @intCast(batches),
        }, .{ .x = 16, .y = 16 }, .{
            ab.ptr,
            bb.ptr,
            storage.buffer.ptr,
            @as(u32, @intCast(m)),
            @as(u32, @intCast(n)),
            @as(u32, @intCast(k)),
            @as(u32, @intCast(m * @as(usize, @intCast(k)))),
            @as(u32, @intCast(@as(usize, @intCast(k)) * n)),
            @as(u32, @intCast(m * n)),
            @as(u32, 0),
        });
        try self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
    }

    fn resize(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node) !void {
        const x = try self.input(arena, values, node.inputs[0]);
        // The kernel implements exactly one of Resize's many spellings, so
        // every attribute that would change the mapping is checked here.
        if (!attributeIs(node, "mode", "nearest", "nearest")) return Error.UnsupportedOperator;
        if (!attributeIs(node, "nearest_mode", "round_prefer_floor", "floor")) return Error.UnsupportedOperator;
        if (!attributeIs(node, "coordinate_transformation_mode", "half_pixel", "asymmetric")) return Error.UnsupportedOperator;
        if (node.inputs.len < 4 or node.inputs[3].len == 0) return Error.UnsupportedOperator;
        const rank = x.dims.len;
        if (rank > max_rank) return Error.RankTooLarge;

        const dims = try arena.dupe(i64, try (try self.input(arena, values, node.inputs[3])).i64s());
        if (dims.len != rank) return Error.InvalidShape;
        const count = try elementCount(dims);

        var strides: [max_rank]u32 = @splat(0);
        denseStrides(x.dims, &strides);
        var metadata: [3 * max_rank]u32 = @splat(0);
        for (dims, 0..) |dim, i| metadata[i] = @intCast(dim);
        for (x.dims, 0..) |dim, i| metadata[rank + i] = @intCast(dim);
        @memcpy(metadata[2 * rank ..][0..rank], strides[0..rank]);
        try self.env.metadata.upload(metadata[0 .. 3 * rank]);

        const storage = try self.newStorage(count);
        errdefer self.releaseStorage(storage);
        const xb = try x.gpuBuffer();
        const block: u32 = 256;
        try self.env.gpu.resize_nearest.launch(.{ .x = @intCast((count + block - 1) / block) }, .{ .x = block }, .{
            xb.ptr, storage.buffer.ptr, self.env.metadata.ptr, @as(u32, @intCast(rank)), @as(u32, @intCast(count)),
        });
        try self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
    }

    fn instanceNorm(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node) !void {
        const x = try self.input(arena, values, node.inputs[0]);
        const scale = try self.input(arena, values, node.inputs[1]);
        const bias = try self.input(arena, values, node.inputs[2]);
        if (x.dims.len < 3) return Error.InvalidShape;

        const channels: usize = @intCast(x.dims[1]);
        const cols = try elementCount(x.dims[2..]);
        const count = try x.count();
        const rows = count / @max(cols, 1);

        const storage = try self.newStorage(count);
        errdefer self.releaseStorage(storage);
        const xb = try x.gpuBuffer();
        const sb = try scale.gpuBuffer();
        const bb = try bias.gpuBuffer();
        var block: u32 = 32;
        while (block < cols and block < 256) block *= 2;
        try self.env.gpu.instance_norm.launch(.{ .x = @intCast(rows) }, .{ .x = block }, .{
            xb.ptr,
            sb.ptr,
            bb.ptr,
            storage.buffer.ptr,
            @as(u32, @intCast(channels)),
            @as(u32, @intCast(cols)),
            node.float("epsilon", 1e-5),
        });
        const dims = try arena.dupe(i64, x.dims);
        try self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
    }

    fn constant(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node) !void {
        const attribute = node.attribute("value") orelse return Error.UnsupportedOperator;
        _ = try self.materialize(arena, values, node.outputs[0], attribute.tensor orelse return Error.UnsupportedOperator);
    }

    /// Reshapes to a matrix, everything before `axis` against everything from
    /// it on. Only the shape changes, so the buffer is shared.
    fn flatten(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node) !void {
        const x = try self.input(arena, values, node.inputs[0]);
        const axis = normalizeAxis(node.int("axis", 1), x.dims.len);
        const dims = try arena.dupe(i64, &.{
            @as(i64, @intCast(try elementCount(x.dims[0..axis]))),
            @as(i64, @intCast(try elementCount(x.dims[axis..]))),
        });
        try self.put(arena, values, node.outputs[0], try self.alias(x, dims));
    }

    /// And and Or over the booleans a graph's control flow is written in.
    fn logical(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node, conjunction: bool) !void {
        const a = try self.input(arena, values, node.inputs[0]);
        const b = try self.input(arena, values, node.inputs[1]);
        if (!a.onHost() or !b.onHost()) return Error.UnsupportedDataType;
        const dims = try broadcastShape(arena, a.dims, b.dims);
        if (dims.len > max_rank) return Error.RankTooLarge;
        var astrides: [max_rank]u32 = @splat(0);
        var bstrides: [max_rank]u32 = @splat(0);
        makeBroadcastStrides(a.dims, dims, &astrides);
        makeBroadcastStrides(b.dims, dims, &bstrides);

        const out = try arena.alloc(u8, try elementCount(dims));
        for (out, 0..) |*value, i| {
            const x = try a.element(hostOffset(i, dims, &astrides)) != 0;
            const y = try b.element(hostOffset(i, dims, &bstrides)) != 0;
            value.* = @intFromBool(if (conjunction) x and y else x or y);
        }
        try self.put(arena, values, node.outputs[0], .{ .dtype = .bool, .dims = dims, .data = .{ .host = out } });
    }

    /// ONNX Runtime's MatMulNBits. The weight matrix is 4-bit and transposed,
    /// so it neither goes through `input` nor through the plain matmul.
    fn matmulNBits(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node) !void {
        const a = try self.input(arena, values, node.inputs[0]);
        // Zero points, a group index or a bias would each need their own
        // handling, and none of SAM 3's exports carry them.
        if (node.int("bits", 4) != 4 or node.inputs.len > 3) return Error.UnsupportedOperator;

        const depth = node.int("K", 0);
        const width = node.int("N", 0);
        const block_size = node.int("block_size", 0);
        // ONNX requires a power of two no smaller than 16; the kernel relies
        // on it to unpack with a shift rather than a division.
        if (depth <= 0 or width <= 0 or block_size < 2) return Error.InvalidShape;
        if (!std.math.isPowerOfTwo(@as(u64, @intCast(block_size)))) return Error.UnsupportedOperator;
        if (a.dims.len == 0 or a.dims[a.dims.len - 1] != depth) return Error.InvalidShape;
        const blocks_per_row = @divFloor(depth + block_size - 1, block_size);

        const dims = try arena.dupe(i64, a.dims);
        dims[dims.len - 1] = width;
        const rows = try elementCount(a.dims[0 .. a.dims.len - 1]);

        const storage = try self.newStorage(try elementCount(dims));
        errdefer self.releaseStorage(storage);
        const ab = try a.gpuBuffer();
        const weights = try self.constantBytes(node.inputs[1]);
        const scales = try (try self.input(arena, values, node.inputs[2])).gpuBuffer();

        try self.env.gpu.matmul_nbits.launch(.{
            .x = @intCast(@divFloor(width + matmul_tile_n - 1, matmul_tile_n)),
            .y = @intCast(@divFloor(@as(i64, @intCast(rows)) + matmul_tile_m - 1, matmul_tile_m)),
        }, .{ .x = 16, .y = 16 }, .{
            ab.ptr,
            weights,
            scales.ptr,
            storage.buffer.ptr,
            @as(u32, @intCast(rows)),
            @as(u32, @intCast(width)),
            @as(u32, @intCast(depth)),
            @as(u32, @ctz(@as(u32, @intCast(block_size)))),
            @as(u32, @intCast(blocks_per_row)),
        });
        try self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
    }

    fn cumulativeSum(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node) !void {
        const x = try self.input(arena, values, node.inputs[0]);
        if (node.int("exclusive", 0) != 0 or node.int("reverse", 0) != 0) return Error.UnsupportedOperator;
        const axis = normalizeAxis(try (try self.input(arena, values, node.inputs[1])).element(0), x.dims.len);
        const inner = try elementCount(x.dims[axis + 1 ..]);
        const outer = try elementCount(x.dims[0..axis]);
        const along: usize = @intCast(x.dims[axis]);

        const count = try x.count();
        const storage = try self.newStorage(count);
        errdefer self.releaseStorage(storage);
        const xb = try x.gpuBuffer();
        const lines = outer * inner;
        const block: u32 = 256;
        try self.env.gpu.cumulative_sum.launch(.{ .x = @intCast((lines + block - 1) / block) }, .{ .x = block }, .{
            xb.ptr, storage.buffer.ptr, @as(u32, @intCast(along)), @as(u32, @intCast(inner)), @as(u32, @intCast(lines)),
        });
        const dims = try arena.dupe(i64, x.dims);
        try self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
    }

    fn maxPool(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node) !void {
        const x = try self.input(arena, values, node.inputs[0]);
        if (x.dims.len != 4 or node.outputs.len > 1) return Error.UnsupportedOperator;
        const kernel = node.ints("kernel_shape");
        if (kernel.len != 2) return Error.UnsupportedOperator;
        const strides = node.ints("strides");
        const pads = node.ints("pads");
        const dilations = node.ints("dilations");
        for (dilations) |dilation| if (dilation != 1) return Error.UnsupportedOperator;
        const sh: i64 = if (strides.len == 0) 1 else strides[0];
        const sw: i64 = if (strides.len == 0) 1 else strides[1];
        const ph0: i64 = if (pads.len == 0) 0 else pads[0];
        const pw0: i64 = if (pads.len == 0) 0 else pads[1];
        const ph1: i64 = if (pads.len < 4) ph0 else pads[2];
        const pw1: i64 = if (pads.len < 4) pw0 else pads[3];
        const out_h = @divFloor(x.dims[2] + ph0 + ph1 - kernel[0], sh) + 1;
        const out_w = @divFloor(x.dims[3] + pw0 + pw1 - kernel[1], sw) + 1;

        const dims = try arena.dupe(i64, &.{ x.dims[0], x.dims[1], out_h, out_w });
        const count = try elementCount(dims);
        const storage = try self.newStorage(count);
        errdefer self.releaseStorage(storage);
        const meta = [_]u32{
            @intCast(x.dims[2]), @intCast(x.dims[3]), @intCast(out_h), @intCast(out_w),
            @intCast(kernel[0]), @intCast(kernel[1]), @intCast(sh),    @intCast(sw),
            @intCast(ph0),       @intCast(pw0),
        };
        try self.env.metadata.upload(&meta);
        const xb = try x.gpuBuffer();
        const block: u32 = 256;
        try self.env.gpu.max_pool2d.launch(.{ .x = @intCast((count + block - 1) / block) }, .{ .x = block }, .{
            xb.ptr, storage.buffer.ptr, self.env.metadata.ptr, @as(u32, @intCast(count)),
        });
        try self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
    }

    fn identity(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node) !void {
        const x = try self.input(arena, values, node.inputs[0]);
        try self.put(arena, values, node.outputs[0], try self.alias(x, try arena.dupe(i64, x.dims)));
    }

    /// Comparisons only ever appear in the shape arithmetic of this export, so
    /// they stay on the host and produce ONNX booleans.
    fn compare(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node, op: Compare) !void {
        const a = try self.input(arena, values, node.inputs[0]);
        const b = try self.input(arena, values, node.inputs[1]);
        if (!a.onHost() or !b.onHost()) return Error.UnsupportedDataType;
        const dims = try broadcastShape(arena, a.dims, b.dims);
        if (dims.len > max_rank) return Error.RankTooLarge;
        var astrides: [max_rank]u32 = @splat(0);
        var bstrides: [max_rank]u32 = @splat(0);
        makeBroadcastStrides(a.dims, dims, &astrides);
        makeBroadcastStrides(b.dims, dims, &bstrides);

        const out = try arena.alloc(u8, try elementCount(dims));
        for (out, 0..) |*value, i| {
            const x = try a.element(hostOffset(i, dims, &astrides));
            const y = try b.element(hostOffset(i, dims, &bstrides));
            value.* = @intFromBool(switch (op) {
                .equal => x == y,
                .greater => x > y,
                .greater_equal => x >= y,
                .less => x < y,
                .less_equal => x <= y,
            });
        }
        try self.put(arena, values, node.outputs[0], .{ .dtype = .bool, .dims = dims, .data = .{ .host = out } });
    }

    fn not(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node) !void {
        const x = try self.input(arena, values, node.inputs[0]);
        const source = try x.bools();
        const out = try arena.alloc(u8, source.len);
        for (out, source) |*value, v| value.* = @intFromBool(v == 0);
        const dims = try arena.dupe(i64, x.dims);
        try self.put(arena, values, node.outputs[0], .{ .dtype = .bool, .dims = dims, .data = .{ .host = out } });
    }

    /// Where picks between two shapes in this graph's control flow and between
    /// two feature tensors in its data flow, so it has to serve both.
    fn where(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node) !void {
        const condition = try self.input(arena, values, node.inputs[0]);
        const a = try self.input(arena, values, node.inputs[1]);
        const b = try self.input(arena, values, node.inputs[2]);
        const dims = try broadcastShape(arena, try broadcastShape(arena, condition.dims, a.dims), b.dims);
        const rank = dims.len;
        if (rank > max_rank) return Error.RankTooLarge;
        const count = try elementCount(dims);

        var cstrides: [max_rank]u32 = @splat(0);
        var astrides: [max_rank]u32 = @splat(0);
        var bstrides: [max_rank]u32 = @splat(0);
        makeBroadcastStrides(condition.dims, dims, &cstrides);
        makeBroadcastStrides(a.dims, dims, &astrides);
        makeBroadcastStrides(b.dims, dims, &bstrides);

        if (a.dtype == .i64 and b.dtype == .i64) {
            const out = try arena.alloc(i64, count);
            for (out, 0..) |*value, i| {
                value.* = if (try condition.element(hostOffset(i, dims, &cstrides)) != 0)
                    try a.element(hostOffset(i, dims, &astrides))
                else
                    try b.element(hostOffset(i, dims, &bstrides));
            }
            return self.put(arena, values, node.outputs[0], .{ .dtype = .i64, .dims = dims, .data = .{ .host = std.mem.sliceAsBytes(out) } });
        }

        // The condition is a host boolean but the branches are on the device,
        // so it goes up as the 0/1 floats the kernel tests against.
        var mask: ?*Storage = null;
        defer if (mask) |storage| self.releaseStorage(storage);
        const cb = if (condition.dtype == .f32) try condition.gpuBuffer() else blk: {
            mask = try self.deviceMask(arena, condition.*);
            break :blk mask.?.buffer;
        };
        const ab = try a.gpuBuffer();
        const bb = try b.gpuBuffer();

        var metadata: [4 * max_rank]u32 = @splat(0);
        for (dims, 0..) |dim, i| metadata[i] = @intCast(dim);
        @memcpy(metadata[rank..][0..rank], cstrides[0..rank]);
        @memcpy(metadata[2 * rank ..][0..rank], astrides[0..rank]);
        @memcpy(metadata[3 * rank ..][0..rank], bstrides[0..rank]);
        try self.env.metadata.upload(metadata[0 .. 4 * rank]);

        const storage = try self.newStorage(count);
        errdefer self.releaseStorage(storage);
        const block: u32 = 256;
        try self.env.gpu.select.launch(.{ .x = @intCast((count + block - 1) / block) }, .{ .x = block }, .{
            cb.ptr,                ab.ptr,                   bb.ptr,                    storage.buffer.ptr,
            self.env.metadata.ptr, @as(u32, @intCast(rank)), @as(u32, @intCast(count)),
        });
        try self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
    }

    /// Uploads a host integer or boolean tensor as the floats the elementwise
    /// kernels work in.
    fn deviceMask(self: *SessionState, arena: std.mem.Allocator, tensor: Tensor) !*Storage {
        const count = try tensor.count();
        const host = try arena.alloc(f32, count);
        for (host, 0..) |*value, i| value.* = @floatFromInt(try tensor.element(i));
        const storage = try self.newStorage(count);
        errdefer self.releaseStorage(storage);
        try storage.buffer.upload(host);
        return storage;
    }

    fn cast(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node) !void {
        const x = try self.input(arena, values, node.inputs[0]);
        const to: onnx.DataType = @enumFromInt(@as(u32, @intCast(node.int("to", @intFromEnum(x.dtype)))));
        const dims = try arena.dupe(i64, x.dims);
        if (to == x.dtype) return self.put(arena, values, node.outputs[0], try self.alias(x, dims));

        const count = try x.count();
        if (to == .f32) {
            if (!x.onHost()) return Error.UnsupportedDataType;
            const storage = try self.deviceMask(arena, x.*);
            errdefer self.releaseStorage(storage);
            return self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
        }

        // Anything but f32 lives on the host already; f32 has to come back
        // down. The copy is stream-ordered, so it sees the finished tensor.
        const downloaded: ?[]f32 = if (x.onHost()) null else blk: {
            const host = try arena.alloc(f32, count);
            try (try x.gpuBuffer()).download(host);
            break :blk host;
        };
        switch (to) {
            .bool => {
                const out = try arena.alloc(u8, count);
                for (out, 0..) |*value, i| {
                    value.* = @intFromBool(if (downloaded) |f| f[i] != 0 else (try x.element(i)) != 0);
                }
                try self.put(arena, values, node.outputs[0], .{ .dtype = .bool, .dims = dims, .data = .{ .host = out } });
            },
            .i64 => {
                const out = try arena.alloc(i64, count);
                for (out, 0..) |*value, i| {
                    value.* = if (downloaded) |f| @intFromFloat(@trunc(f[i])) else try x.element(i);
                }
                try self.put(arena, values, node.outputs[0], .{ .dtype = .i64, .dims = dims, .data = .{ .host = std.mem.sliceAsBytes(out) } });
            },
            else => return Error.UnsupportedDataType,
        }
    }

    /// Broadcasts to a shape given as a tensor. On the device this is the copy
    /// kernel: a broadcast axis is spelled as a stride of zero.
    fn expand(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node) !void {
        const x = try self.input(arena, values, node.inputs[0]);
        const target = try (try self.input(arena, values, node.inputs[1])).i64s();
        const dims = try broadcastShape(arena, x.dims, target);
        const rank = dims.len;
        if (rank > max_rank) return Error.RankTooLarge;
        const count = try elementCount(dims);
        var strides: [max_rank]u32 = @splat(0);
        makeBroadcastStrides(x.dims, dims, &strides);

        if (x.onHost()) {
            const sources = try arena.alloc(usize, count);
            for (sources, 0..) |*source, i| source.* = hostOffset(i, dims, &strides);
            const out = try hostSelect(arena, x.*, sources);
            return self.put(arena, values, node.outputs[0], .{ .dtype = x.dtype, .dims = dims, .data = .{ .host = out } });
        }

        var metadata: [2 * max_rank]u32 = @splat(0);
        for (dims, 0..) |dim, i| metadata[i] = @intCast(dim);
        @memcpy(metadata[rank..][0..rank], strides[0..rank]);
        try self.env.metadata.upload(metadata[0 .. 2 * rank]);

        const storage = try self.newStorage(count);
        errdefer self.releaseStorage(storage);
        const xb = try x.gpuBuffer();
        const block: u32 = 256;
        try self.env.gpu.copy.launch(.{ .x = @intCast((count + block - 1) / block) }, .{ .x = block }, .{
            xb.ptr,                    storage.buffer.ptr, self.env.metadata.ptr, @as(u32, @intCast(rank)),
            @as(u32, @intCast(count)), @as(u32, 0),        @as(u32, 0),
        });
        try self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
    }

    fn tile(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node) !void {
        const x = try self.input(arena, values, node.inputs[0]);
        const repeats = try (try self.input(arena, values, node.inputs[1])).i64s();
        const rank = x.dims.len;
        if (repeats.len != rank or rank > max_rank) return Error.InvalidShape;
        const dims = try arena.alloc(i64, rank);
        for (dims, x.dims, repeats) |*dim, source, times| {
            if (times < 0) return Error.InvalidShape;
            dim.* = source * times;
        }
        const count = try elementCount(dims);
        var strides: [max_rank]u32 = @splat(0);
        denseStrides(x.dims, &strides);

        if (x.onHost()) {
            const sources = try arena.alloc(usize, count);
            for (sources, 0..) |*source, index| {
                var remaining = index;
                var offset: usize = 0;
                var axis = rank;
                while (axis > 0) {
                    axis -= 1;
                    const dim: usize = @intCast(dims[axis]);
                    const coordinate = remaining % dim;
                    remaining /= dim;
                    offset += (coordinate % @as(usize, @intCast(x.dims[axis]))) * strides[axis];
                }
                source.* = offset;
            }
            const out = try hostSelect(arena, x.*, sources);
            return self.put(arena, values, node.outputs[0], .{ .dtype = x.dtype, .dims = dims, .data = .{ .host = out } });
        }

        var metadata: [3 * max_rank]u32 = @splat(0);
        for (dims, 0..) |dim, i| metadata[i] = @intCast(dim);
        for (x.dims, 0..) |dim, i| metadata[rank + i] = @intCast(dim);
        @memcpy(metadata[2 * rank ..][0..rank], strides[0..rank]);
        try self.env.metadata.upload(metadata[0 .. 3 * rank]);

        const storage = try self.newStorage(count);
        errdefer self.releaseStorage(storage);
        const xb = try x.gpuBuffer();
        const block: u32 = 256;
        try self.env.gpu.tile.launch(.{ .x = @intCast((count + block - 1) / block) }, .{ .x = block }, .{
            xb.ptr, storage.buffer.ptr, self.env.metadata.ptr, @as(u32, @intCast(rank)), @as(u32, @intCast(count)),
        });
        try self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
    }

    fn range(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node) !void {
        const start = try (try self.input(arena, values, node.inputs[0])).element(0);
        const limit = try (try self.input(arena, values, node.inputs[1])).element(0);
        const delta = try (try self.input(arena, values, node.inputs[2])).element(0);
        if (delta == 0) return Error.InvalidShape;

        var count: usize = 0;
        if (delta > 0 and limit > start) count = @intCast(@divFloor(limit - start + delta - 1, delta));
        if (delta < 0 and limit < start) count = @intCast(@divFloor(start - limit - delta - 1, -delta));

        const out = try arena.alloc(i64, count);
        for (out, 0..) |*value, i| value.* = start + @as(i64, @intCast(i)) * delta;
        const dims = try arena.dupe(i64, &.{@as(i64, @intCast(count))});
        try self.put(arena, values, node.outputs[0], .{ .dtype = .i64, .dims = dims, .data = .{ .host = std.mem.sliceAsBytes(out) } });
    }

    fn constantOfShape(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node) !void {
        const requested = try (try self.input(arena, values, node.inputs[0])).i64s();
        const dims = try arena.dupe(i64, requested);
        const count = try elementCount(dims);

        // The attribute carries a one-element tensor that fixes both the fill
        // value and the output type; without it ONNX fills with a float zero.
        const value = if (node.attribute("value")) |attribute| attribute.tensor else null;
        const dtype: onnx.DataType = if (value) |tensor| tensor.dtype else .f32;
        switch (dtype) {
            .i64 => {
                const filler = if (value.?.elementCount() != 0) value.?.i64s()[0] else 0;
                const out = try arena.alloc(i64, count);
                @memset(out, filler);
                try self.put(arena, values, node.outputs[0], .{ .dtype = .i64, .dims = dims, .data = .{ .host = std.mem.sliceAsBytes(out) } });
            },
            .bool => {
                const out = try arena.alloc(u8, count);
                @memset(out, if (value.?.data.len != 0) value.?.data[0] else 0);
                try self.put(arena, values, node.outputs[0], .{ .dtype = .bool, .dims = dims, .data = .{ .host = out } });
            },
            .f32 => {
                const filler: f32 = if (value) |tensor|
                    (if (tensor.elementCount() != 0) tensor.f32s()[0] else 0)
                else
                    0;
                const storage = try self.newStorage(count);
                errdefer self.releaseStorage(storage);
                const block: u32 = 256;
                try self.env.gpu.fill.launch(.{ .x = @intCast((count + block - 1) / block) }, .{ .x = block }, .{
                    storage.buffer.ptr, filler, @as(u32, @intCast(count)),
                });
                try self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
            },
            else => return Error.UnsupportedDataType,
        }
    }

    fn clip(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node) !void {
        const x = try self.input(arena, values, node.inputs[0]);
        const low = if (node.inputs.len > 1 and node.inputs[1].len != 0) try self.input(arena, values, node.inputs[1]) else null;
        const high = if (node.inputs.len > 2 and node.inputs[2].len != 0) try self.input(arena, values, node.inputs[2]) else null;
        const dims = try arena.dupe(i64, x.dims);
        const count = try x.count();

        if (x.onHost()) {
            if (x.dtype != .i64) return Error.UnsupportedDataType;
            const out = try arena.alloc(i64, count);
            for (out, 0..) |*value, i| {
                var v = try x.element(i);
                if (low) |bound| v = @max(v, try bound.element(0));
                if (high) |bound| v = @min(v, try bound.element(0));
                value.* = v;
            }
            return self.put(arena, values, node.outputs[0], .{ .dtype = .i64, .dims = dims, .data = .{ .host = std.mem.sliceAsBytes(out) } });
        }

        const storage = try self.newStorage(count);
        errdefer self.releaseStorage(storage);
        const xb = try x.gpuBuffer();
        // The kernel ignores an absent bound, but still wants a valid pointer.
        const lb = if (low) |bound| try bound.gpuBuffer() else xb;
        const hb = if (high) |bound| try bound.gpuBuffer() else xb;
        const block: u32 = 256;
        try self.env.gpu.clip.launch(.{ .x = @intCast((count + block - 1) / block) }, .{ .x = block }, .{
            xb.ptr,                               lb.ptr,
            hb.ptr,                               storage.buffer.ptr,
            @as(u32, @intCast(count)),            @as(u32, if (low != null) 1 else 0),
            @as(u32, if (high != null) 1 else 0),
        });
        try self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
    }

    fn oneHot(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node) !void {
        const indices = try self.input(arena, values, node.inputs[0]);
        const depth_tensor = try self.input(arena, values, node.inputs[1]);
        const pair = try self.input(arena, values, node.inputs[2]);
        if (pair.dtype != .i64) return Error.UnsupportedDataType;
        const ids = try indices.i64s();
        const off_on = try pair.i64s();
        if (off_on.len != 2) return Error.InvalidShape;

        const depth = try depth_tensor.element(0);
        if (depth < 0) return Error.InvalidShape;
        const rank = indices.dims.len + 1;
        const axis = normalizeAxis(node.int("axis", -1), rank);
        const dims = try arena.alloc(i64, rank);
        @memcpy(dims[0..axis], indices.dims[0..axis]);
        dims[axis] = depth;
        @memcpy(dims[axis + 1 ..], indices.dims[axis..]);

        const inner = try elementCount(dims[axis + 1 ..]);
        const slots: usize = @intCast(depth);
        const out = try arena.alloc(i64, try elementCount(dims));
        for (out, 0..) |*value, i| {
            const inner_index = i % inner;
            const slot = (i / inner) % slots;
            const outer_index = i / (inner * slots);
            const index = ids[outer_index * inner + inner_index];
            const normalized = if (index < 0) depth + index else index;
            value.* = if (normalized == @as(i64, @intCast(slot))) off_on[1] else off_on[0];
        }
        try self.put(arena, values, node.outputs[0], .{ .dtype = .i64, .dims = dims, .data = .{ .host = std.mem.sliceAsBytes(out) } });
    }

    /// Overwrites slices of `data` at the positions `indices` names. The index
    /// tuples are shape arithmetic, so the host flattens them to plain offsets
    /// and the kernel only has to copy.
    fn scatterNd(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node) !void {
        if (node.attribute("reduction")) |attribute| {
            if (attribute.s.len != 0 and !std.mem.eql(u8, attribute.s, "none")) return Error.UnsupportedOperator;
        }
        const data = try self.input(arena, values, node.inputs[0]);
        const indices = try self.input(arena, values, node.inputs[1]);
        const updates = try self.input(arena, values, node.inputs[2]);
        if (indices.dims.len == 0 or data.dims.len > max_rank) return Error.InvalidShape;

        const ids = try indices.i64s();
        const tuple: usize = @intCast(indices.dims[indices.dims.len - 1]);
        if (tuple > data.dims.len) return Error.InvalidShape;
        const tuples = try elementCount(indices.dims[0 .. indices.dims.len - 1]);
        const span = try elementCount(data.dims[tuple..]);

        var strides: [max_rank]u32 = @splat(0);
        denseStrides(data.dims, &strides);
        const offsets = try arena.alloc(u32, tuples);
        for (offsets, 0..) |*offset, t| {
            var flat: usize = 0;
            for (0..tuple) |d| {
                const raw = ids[t * tuple + d];
                const normalized = if (raw < 0) data.dims[d] + raw else raw;
                if (normalized < 0 or normalized >= data.dims[d]) return Error.InvalidShape;
                flat += @as(usize, @intCast(normalized)) * strides[d];
            }
            offset.* = @intCast(flat);
        }

        // The graph can still refer to `data` after this node, so the scatter
        // lands in a copy rather than in place.
        const dims = try arena.dupe(i64, data.dims);
        const count = try elementCount(dims);
        if (data.onHost()) {
            if (data.dtype != .i64) return Error.UnsupportedDataType;
            const out = try arena.dupe(i64, try data.i64s());
            const patch = try updates.i64s();
            for (offsets, 0..) |offset, t| @memcpy(out[offset..][0..span], patch[t * span ..][0..span]);
            return self.put(arena, values, node.outputs[0], .{ .dtype = .i64, .dims = dims, .data = .{ .host = std.mem.sliceAsBytes(out) } });
        }

        const rank = dims.len;
        var metadata: [2 * max_rank]u32 = @splat(0);
        for (dims, 0..) |dim, i| metadata[i] = @intCast(dim);
        @memcpy(metadata[rank..][0..rank], strides[0..rank]);
        try self.env.metadata.upload(metadata[0 .. 2 * rank]);

        const storage = try self.newStorage(count);
        errdefer self.releaseStorage(storage);
        const db = try data.gpuBuffer();
        const block: u32 = 256;
        try self.env.gpu.copy.launch(.{ .x = @intCast((count + block - 1) / block) }, .{ .x = block }, .{
            db.ptr,                    storage.buffer.ptr, self.env.metadata.ptr, @as(u32, @intCast(rank)),
            @as(u32, @intCast(count)), @as(u32, 0),        @as(u32, 0),
        });

        try self.ensureIndices(offsets.len);
        try self.env.indices.upload(offsets);
        const ub = try updates.gpuBuffer();
        const written = tuples * span;
        try self.env.gpu.scatter.launch(.{ .x = @intCast((written + block - 1) / block) }, .{ .x = block }, .{
            ub.ptr, self.env.indices.ptr, storage.buffer.ptr, @as(u32, @intCast(span)), @as(u32, @intCast(written)),
        });
        try self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
    }
};

/// Compares a string attribute against what a kernel actually implements,
/// standing ONNX's own default in when the attribute is absent. Several of
/// those defaults are not what is implemented here, so they have to be named.
fn attributeIs(node: onnx.Node, name: []const u8, default: []const u8, expected: []const u8) bool {
    const attribute = node.attribute(name) orelse return std.mem.eql(u8, default, expected);
    return std.mem.eql(u8, if (attribute.s.len == 0) default else attribute.s, expected);
}

fn elementCount(dims: []const i64) !usize {
    var count: usize = 1;
    for (dims) |dim| {
        if (dim < 0) return Error.InvalidShape;
        count = try std.math.mul(usize, count, @intCast(dim));
    }
    return count;
}
fn normalizeAxis(axis: i64, rank: usize) usize {
    return @intCast(if (axis < 0) @as(i64, @intCast(rank)) + axis else axis);
}
fn normalizeAxisInsert(axis: i64, rank: usize) usize {
    return normalizeAxis(axis, rank);
}

fn broadcastShape(arena: std.mem.Allocator, a: []const i64, b: []const i64) ![]i64 {
    const rank = @max(a.len, b.len);
    const out = try arena.alloc(i64, rank);
    for (0..rank) |i| {
        const ai = if (i < rank - a.len) 1 else a[i - (rank - a.len)];
        const bi = if (i < rank - b.len) 1 else b[i - (rank - b.len)];
        if (ai != bi and ai != 1 and bi != 1) return Error.InvalidShape;
        // Not `@max`: an axis of length zero broadcast against one stays
        // empty, which is how SAM 3 spells "no box prompt".
        out[i] = if (ai == 1) bi else ai;
    }
    return out;
}

/// Row-major strides for a dense tensor, the layout every buffer here is in.
fn denseStrides(dims: []const i64, result: *[max_rank]u32) void {
    var stride: u32 = 1;
    var axis = dims.len;
    while (axis > 0) {
        axis -= 1;
        result[axis] = stride;
        stride *= @intCast(dims[axis]);
    }
}

/// The host counterpart of the kernels' `offsetOf`: walks a flat index through
/// the output shape and back to an offset in an input with the given strides.
fn hostOffset(index: usize, dims: []const i64, strides: *const [max_rank]u32) usize {
    var remaining = index;
    var offset: usize = 0;
    var axis = dims.len;
    while (axis > 0) {
        axis -= 1;
        const dim: usize = @intCast(dims[axis]);
        offset += (remaining % dim) * strides[axis];
        remaining /= dim;
    }
    return offset;
}

/// Builds a host tensor by reading `sources` elements out of `tensor`, keeping
/// its element type. Gather, Expand and Tile all reduce to this once their
/// index map is known.
fn hostSelect(arena: std.mem.Allocator, tensor: Tensor, sources: []const usize) ![]const u8 {
    switch (tensor.dtype) {
        .i64 => {
            const source = try tensor.i64s();
            const out = try arena.alloc(i64, sources.len);
            for (out, sources) |*value, index| value.* = source[index];
            return std.mem.sliceAsBytes(out);
        },
        .bool => {
            const source = try tensor.bools();
            const out = try arena.alloc(u8, sources.len);
            for (out, sources) |*value, index| value.* = source[index];
            return out;
        },
        else => return Error.UnsupportedDataType,
    }
}

fn makeBroadcastStrides(input: []const i64, output: []const i64, result: *[8]u32) void {
    var stride: u32 = 1;
    var source = input.len;
    var out_axis = output.len;
    while (out_axis > 0) {
        out_axis -= 1;
        if (source == 0) {
            result[out_axis] = 0;
            continue;
        }
        source -= 1;
        result[out_axis] = if (input[source] == 1 and output[out_axis] != 1) 0 else stride;
        stride *= @intCast(input[source]);
    }
}

pub const Value = struct {
    dtype: onnx.DataType,
    bytes: []const u8,
    dims: []const i64,
    owned_allocator: ?std.mem.Allocator = null,
    owned_f32: ?[]f32 = null,
    owned_i64: ?[]i64 = null,

    pub fn deinit(self: Value) void {
        if (self.owned_allocator) |allocator| {
            allocator.free(self.dims);
            if (self.owned_f32) |data| allocator.free(data);
            if (self.owned_i64) |data| allocator.free(data);
        }
    }

    pub fn borrowF32(data: []const f32, dims: []const i64) !Value {
        return .{ .dtype = .f32, .bytes = std.mem.sliceAsBytes(data), .dims = dims };
    }

    pub fn borrowI64(data: []const i64, dims: []const i64) !Value {
        return .{ .dtype = .i64, .bytes = std.mem.sliceAsBytes(data), .dims = dims };
    }

    pub fn dataF32(self: Value) ![]const f32 {
        if (self.dtype != .f32) return Error.NativeRuntime;
        return @alignCast(std.mem.bytesAsSlice(f32, self.bytes));
    }

    pub fn shape(self: Value, buffer: []i64) ![]const i64 {
        if (self.dims.len > buffer.len) return Error.NativeRuntime;
        @memcpy(buffer[0..self.dims.len], self.dims);
        return buffer[0..self.dims.len];
    }

    fn take(allocator: std.mem.Allocator, tensor: Tensor) !Value {
        const dims = try allocator.dupe(i64, tensor.dims);
        errdefer allocator.free(dims);
        switch (tensor.dtype) {
            .f32 => {
                const data = try allocator.alloc(f32, try tensor.count());
                errdefer allocator.free(data);
                try (try tensor.gpuBuffer()).download(data);
                return .{ .dtype = .f32, .bytes = std.mem.sliceAsBytes(data), .dims = dims, .owned_allocator = allocator, .owned_f32 = data };
            },
            .i64 => {
                const source = try tensor.i64s();
                const data = try allocator.dupe(i64, source);
                return .{ .dtype = .i64, .bytes = std.mem.sliceAsBytes(data), .dims = dims, .owned_allocator = allocator, .owned_i64 = data };
            },
            else => return Error.UnsupportedDataType,
        }
    }
};
