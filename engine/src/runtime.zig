//! Session boundary used by the SAM 3 application. Graph parsing and GPU
//! ownership live here so the application does not know which executor backs
//! a model. The node executor is deliberately kept behind `Session.run`.

const std = @import("std");
const onnx = @import("onnx.zig");
const device_mod = @import("device.zig");
pub const Device = device_mod.Device;
pub const driver = device_mod.driver;
/// What a float tensor is stored as on the device; see `device.Element`.
pub const Element = device_mod.Element;

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
/// is compiled for the GPU and cannot be imported here, so these values mirror
/// its enums and must be kept in step with them. ONNX spellings are used where
/// possible so `execute` can decode them directly.
const Binary = enum(u32) { Add, Sub, Mul, Div, Pow, Min, Max };
const Unary = enum(u32) { Neg, Erf, Exp, Sqrt, reciprocal, Sigmoid, Tanh, Relu, Abs, Floor, Sin, Cos, Log, Sign, IsNaN, gelu };

/// Comparisons run on the host, over the i64 tensors a graph does its shape
/// arithmetic with, so they are their own list rather than `Binary` entries.
const Compare = enum { Equal, Greater, GreaterOrEqual, Less, LessOrEqual };

/// Matches `kernels.max_rank`: the stride arrays a launch carries are fixed.
const max_rank = 8;
const metadata_words = 4 * max_rank;
// The largest bundled graph has about 1,500 nodes. Keeping one slot per
// metadata-producing launch avoids wrapping (and therefore synchronizing)
// during a normal run while costing only 512 KiB of host memory.
const metadata_slots = 4096;

/// The block tile `kernels.matmul` is written around, mirrored here so the
/// host can size the grid. CUDA carries eight columns per thread while the
/// OpenCL and Metal kernels carry four, so their tile widths differ.
const matmul_tile_m = 128;
const matmul_tile_n = if (@hasDecl(driver, "is_opencl") or @hasDecl(driver, "is_metal")) 64 else 128;

/// The same for `kernels.matmulXmx`, whose work group is 16 sub-groups of 16
/// lanes. Kept in step with DP_TILE_M and DP_TILE_N there.
const matmul_xmx_tile_m = 256;
const matmul_xmx_tile_n = 128;
const matmul_xmx_subgroups = 32;

/// The same for `kernels.matmulTensor`, whose work group is four SIMD groups of
/// 32 lanes. Kept in step with TN_TILE_M and TN_TILE_N there.
const matmul_tensor_tile_m = 128;
const matmul_tensor_tile_n = 64;
const matmul_tensor_subgroups = 8;

/// The same for `kernels.conv2dGemmTensor`, whose work group is eight SIMD
/// groups of 32 lanes. Kept in step with TC_TILE_M and TC_TILE_N there.
const conv_tensor_tile_m = 128;
const conv_tensor_tile_n = 64;
const conv_tensor_subgroups = 16;

/// The same for `kernels.matmulNBitsTensor`.
const matmul_nbits_tensor_tile_m = 128;
const matmul_nbits_tensor_tile_n = 64;
const matmul_nbits_tensor_subgroups = 16;

/// The same for `kernels.matmulSimd`, whose work group is eight SIMD groups of
/// 32 lanes. Kept in step with SG_TILE_M and SG_TILE_N there.
const matmul_simd_tile_m = 64;
const matmul_simd_tile_n = 64;
const matmul_simd_subgroups = 16;

/// The same for `kernels.matmulXmxBlock`, whose work group is 16 sub-groups of
/// 16 lanes. Kept in step with TD_TILE_M and TD_TILE_N there.
const matmul_block_tile_m = 128;
const matmul_block_tile_n = 128;
const matmul_block_subgroups = 16;

/// How much cache the work group swizzle plans a block around. Well under the
/// 8 MiB this GPU has: C streams through it at the same time, nothing pins
/// either operand there, and measured end to end anything from 1 to 12 MiB
/// lands within the run to run spread, so the small end is the honest guess.
const matmul_cache_budget = 2 << 20;

/// The shape `kernels.attention` is written for, mirroring FA_HEAD, FA_KSTEP,
/// FA_SUBGROUPS and FA_QTILE there.
const attention_head = 64;
const attention_key_step = 16;
const attention_subgroups = 16;
const attention_query_tile = if (@hasDecl(driver, "is_opencl")) (8 * attention_subgroups) else 128;

/// How many elements one work item of a vector kernel carries, mirroring
/// LANE_STEP in `kernels.cl`.
const lane_step = 4;

var error_buffer: [1024]u8 = undefined;
var error_length: usize = 0;

fn setError(comptime format: []const u8, args: anytype) void {
    const text = std.fmt.bufPrint(&error_buffer, format, args) catch "native runtime error";
    error_length = text.len;
}

pub fn lastError() []const u8 {
    // Not every failure comes from a graph node: allocations, copies and the
    // end-of-run synchronize fail with driver error and only the driver has
    // anything to say about them.
    if (error_length == 0) return driver.lastError();
    return error_buffer[0..error_length];
}

pub fn version() []const u8 {
    return if (@hasDecl(driver, "is_opencl"))
        "native OpenCL"
    else if (@hasDecl(driver, "is_metal"))
        "native Metal"
    else
        "native CUDA";
}

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
    buckets: std.AutoHashMapUnmanaged(usize, std.ArrayListUnmanaged(driver.DevicePtr)) = .empty,

    fn take(self: *Pool, count: usize) !driver.Buffer(Element) {
        if (self.buckets.getPtr(count)) |bucket| {
            if (bucket.items.len != 0) {
                const ptr = bucket.items[bucket.items.len - 1];
                bucket.items.len -= 1;
                return .{ .ptr = ptr, .len = count };
            }
        }
        return driver.Buffer(Element).alloc(count);
    }

    /// Returning a buffer must not fail, so a pool that cannot record it just
    /// gives it back to the driver.
    fn give(self: *Pool, allocator: std.mem.Allocator, buffer: driver.Buffer(Element)) void {
        if (buffer.len == 0) return;
        const entry = self.buckets.getOrPut(allocator, buffer.len) catch return buffer.free();
        if (!entry.found_existing) entry.value_ptr.* = .empty;
        entry.value_ptr.append(allocator, buffer.ptr) catch buffer.free();
    }

    fn deinit(self: *Pool, allocator: std.mem.Allocator) void {
        var buckets = self.buckets.iterator();
        while (buckets.next()) |entry| {
            for (entry.value_ptr.items) |ptr| (driver.Buffer(Element){ .ptr = ptr, .len = entry.key_ptr.* }).free();
            entry.value_ptr.deinit(allocator);
        }
        self.buckets.deinit(allocator);
    }
};

const State = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    gpu: device_mod.Device,
    /// Shapes and strides for the current launch. Bounded by the maximum rank,
    /// so one small allocation serves every node.
    metadata: driver.Buffer(u32),
    /// Stable host storage for queued metadata writes. OpenCL and Metal require
    /// the source of a non-blocking write to remain unchanged until it completes.
    metadata_staging: []u32,
    metadata_cursor: usize = 0,
    /// Index arrays for Gather and ScatterND. Unlike shapes these have no
    /// fixed bound, so the buffer grows to the largest a graph has asked for.
    indices: driver.Buffer(u32) = .{ .ptr = driver.null_ptr, .len = 0 },
    /// Intermediate tensors, recycled between nodes.
    pool: Pool = .{},

    fn uploadMetadata(self: *State, values: []const u32) !void {
        if (values.len > metadata_words) return Error.RankTooLarge;
        if (self.metadata_cursor == metadata_slots) {
            // Extremely large graphs can wrap the staging area safely. Normal
            // SAM runs fit without this fallback synchronization.
            try self.gpu.synchronize();
            self.metadata_cursor = 0;
        }
        const start = self.metadata_cursor * metadata_words;
        const staging = self.metadata_staging[start..][0..values.len];
        @memcpy(staging, values);
        if (@hasDecl(driver, "is_opencl") or @hasDecl(driver, "is_metal")) {
            try self.metadata.uploadAsync(staging);
        } else {
            try self.metadata.upload(staging);
        }
        self.metadata_cursor += 1;
    }

    fn metadataComplete(self: *State) void {
        self.metadata_cursor = 0;
    }
};

pub const Env = struct {
    state: *State,

    pub fn init(allocator: std.mem.Allocator, io: std.Io) !Env {
        const state = try allocator.create(State);
        errdefer allocator.destroy(state);
        var gpu = try device_mod.Device.init(io, 0);
        errdefer gpu.deinit();
        const metadata = try driver.Buffer(u32).alloc(metadata_words);
        errdefer metadata.free();
        const metadata_staging = try allocator.alloc(u32, metadata_words * metadata_slots);
        errdefer allocator.free(metadata_staging);
        state.* = .{
            .allocator = allocator,
            .io = io,
            .gpu = gpu,
            .metadata = metadata,
            .metadata_staging = metadata_staging,
        };
        return .{ .state = state };
    }

    pub fn deinit(self: Env) void {
        const allocator = self.state.allocator;
        self.state.pool.deinit(allocator);
        self.state.indices.free();
        self.state.metadata.free();
        allocator.free(self.state.metadata_staging);
        self.state.gpu.deinit();
        allocator.destroy(self.state);
    }
};

pub const Session = struct {
    state: *SessionState,

    pub fn open(
        env: Env,
        model_path: []const u8,
    ) !Session {
        const state = try env.state.allocator.create(SessionState);
        errdefer env.state.allocator.destroy(state);
        state.* = .{
            .env = env.state,
            .graph = try onnx.Graph.open(env.state.allocator, env.state.io, model_path),
        };
        errdefer {
            var buffers = state.constants.valueIterator();
            while (buffers.next()) |buffer| buffer.free();
            state.constants.deinit(env.state.allocator);
            var shuffled = state.shuffled.valueIterator();
            while (shuffled.next()) |buffer| buffer.free();
            state.shuffled.deinit(env.state.allocator);
            state.graph.deinit();
        }
        try env.state.gpu.makeCurrent();
        try state.preloadInitializers();
        errdefer {
            state.fused.deinit(env.state.allocator);
            state.folded.deinit(env.state.allocator);
        }
        try state.findFusions();
        return .{ .state = state };
    }

    pub fn deinit(self: Session) void {
        const allocator = self.state.env.allocator;
        var buffers = self.state.constants.valueIterator();
        while (buffers.next()) |buffer| buffer.free();
        self.state.constants.deinit(allocator);
        var shuffled = self.state.shuffled.valueIterator();
        while (shuffled.next()) |buffer| buffer.free();
        self.state.shuffled.deinit(allocator);
        self.state.fused.deinit(allocator);
        self.state.folded.deinit(allocator);
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
    buffer: driver.Buffer(Element),
    references: usize = 1,
};

const Tensor = struct {
    dtype: onnx.DataType,
    dims: []const i64,
    data: union(enum) {
        host: []const u8,
        gpu: *Storage,
        constant_gpu: driver.Buffer(Element),
    },

    fn count(self: Tensor) !usize {
        return elementCount(self.dims);
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
            .i32 => @as([]const i32, @alignCast(std.mem.bytesAsSlice(i32, self.data.host)))[index],
            .u8 => self.data.host[index],
            .i8 => @as(i8, @bitCast(self.data.host[index])),
            .bool => (try self.bools())[index],
            .f32 => @intFromFloat(@trunc(@as([]const f32, @alignCast(std.mem.bytesAsSlice(f32, self.data.host)))[index])),
            else => Error.UnsupportedDataType,
        };
    }

    fn gpuBuffer(self: Tensor) !driver.Buffer(Element) {
        if (self.dtype != .f32) return Error.UnsupportedDataType;
        return switch (self.data) {
            .gpu => |storage| storage.buffer,
            .constant_gpu => |buffer| buffer,
            .host => Error.NativeRuntime,
        };
    }
};

/// Several operators the graph spells out that one kernel does whole,
/// recognized once when the graph is opened. It is recorded against the last
/// node of the group -- the one whose output the rest of the graph reads --
/// and the others go on `folded`, so the tensors they passed between them are
/// never allocated.
const Fusion = struct {
    kind: Kind,
    /// What the surviving node reads in place of what the graph says.
    operands: [3][]const u8,
    arity: u8,
    /// The two products a fused attention falls back to. The shapes are not
    /// known until the graph runs, so one whose shapes turn out not to suit
    /// the kernel runs these after all. Unused by the others.
    product: usize = 0,
    softmax: usize = 0,
    /// What a fused attention multiplies its scores by, where the graph
    /// scaled Q and K before the product and the kernel can do it after.
    scale: f32 = 1,
    /// The operators a fused rotary embedding stands for, in graph order, for
    /// the same reason: shapes that turn out not to suit its kernel run them
    /// after all. Unused by the others.
    swallowed: [15]u32 = @splat(0),
    swallowed_len: u8 = 0,

    const Kind = enum {
        attention,
        gelu,
        rotary,
        biased_product,
        biased_product_gelu,
    };

    fn inputs(self: *const Fusion) []const []const u8 {
        return self.operands[0..self.arity];
    }
};

const SessionState = struct {
    env: *State,
    graph: onnx.Graph,
    arena: std.mem.Allocator = undefined,
    constants: std.StringHashMapUnmanaged(driver.Buffer(Element)) = .empty,
    /// Transposed convolution weights, reordered once into the row per output
    /// channel and tap that the matrix product wants.
    shuffled: std.StringHashMapUnmanaged(driver.Buffer(Element)) = .empty,
    initializers_preloaded: bool = false,
    /// Node index of the surviving operator to what it stands for.
    fused: std.AutoHashMapUnmanaged(usize, Fusion) = .empty,
    /// Node indices the fusion swallowed, which `run` walks past.
    folded: std.AutoHashMapUnmanaged(usize, void) = .empty,

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
        errdefer {
            self.env.gpu.synchronize() catch {};
            self.env.metadataComplete();
        }

        // Session.open normally did this already. Keep the guard here so a
        // future construction path cannot restore lazy, mid-graph uploads.
        try self.preloadInitializers();

        var arena_state: std.heap.ArenaAllocator = .init(self.env.allocator);
        defer arena_state.deinit();
        const arena = arena_state.allocator();
        self.arena = arena;

        var values: std.StringHashMapUnmanaged(*Tensor) = .empty;
        defer values.deinit(arena);

        var uses: std.StringHashMapUnmanaged(usize) = .empty;
        defer uses.deinit(arena);
        // A folded node reads nothing and produces nothing; the node that
        // swallowed it reads what it read.
        for (self.graph.nodes, 0..) |node, index| {
            if (self.folded.contains(index)) continue;
            for (self.effectiveInputs(index, node)) |name| {
                if (name.len == 0) continue;
                const entry = try uses.getOrPut(arena, name);
                entry.value_ptr.* = if (entry.found_existing) entry.value_ptr.* + 1 else 1;
            }
        }
        for (output_names) |name_z| {
            const name = std.mem.span(name_z);
            const entry = try uses.getOrPut(arena, name);
            entry.value_ptr.* = if (entry.found_existing) entry.value_ptr.* + 1 else 1;
        }

        for (input_names, inputs) |name_z, value| {
            const tensor = try arena.create(Tensor);
            if (value.dtype == .f32) {
                const storage = try self.newStorage(value.bytes.len / @sizeOf(f32));
                try uploadFloats(arena, storage.buffer, @alignCast(std.mem.bytesAsSlice(f32, value.bytes)));
                tensor.* = .{ .dtype = .f32, .dims = value.dims, .data = .{ .gpu = storage } };
            } else {
                tensor.* = .{ .dtype = value.dtype, .dims = value.dims, .data = .{ .host = value.bytes } };
            }
            try values.put(arena, std.mem.span(name_z), tensor);
        }

        for (self.graph.nodes, 0..) |node, node_index| {
            if (self.folded.contains(node_index)) continue;
            self.execute(arena, &values, node, node_index) catch |err| {
                const detail = if (err == error.Cuda or err == error.OpenCL or err == error.Metal) driver.lastError() else "";
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
            for (self.effectiveInputs(node_index, node)) |name| {
                if (name.len == 0) continue;
                const remaining = uses.getPtr(name) orelse continue;
                remaining.* -= 1;
                if (remaining.* == 0) if (values.get(name)) |tensor| self.release(tensor);
            }
        }

        try self.env.gpu.synchronize();
        self.env.metadataComplete();
        for (output_names, outputs) |name_z, *output| {
            const name = std.mem.span(name_z);
            const tensor = values.get(name) orelse return Error.MissingValue;
            output.* = try Value.take(self.env.allocator, tensor.*);
            if (uses.getPtr(name)) |remaining| {
                remaining.* -= 1;
                if (remaining.* == 0) self.release(tensor);
            }
        }

        var it = values.valueIterator();
        while (it.next()) |tensor_ptr| {
            self.release(tensor_ptr.*);
        }
    }

    /// Finds every `MatMul -> Softmax -> MatMul` an attention is spelled as,
    /// and records it against the second product. The scale these exports
    /// apply is already folded into Q and K by the time the first product sees
    /// them, so the three nodes are the whole of it.
    ///
    /// The shapes are not known until the graph runs, so this only decides
    /// which nodes could fuse; `attention` checks the shapes it was handed and
    /// falls back to the three operators if they are not what the kernel is
    /// written for.
    fn findFusions(self: *SessionState) !void {
        const allocator = self.env.allocator;

        var producer: std.StringHashMapUnmanaged(usize) = .empty;
        defer producer.deinit(allocator);
        var uses: std.StringHashMapUnmanaged(usize) = .empty;
        defer uses.deinit(allocator);
        for (self.graph.nodes, 0..) |node, index| {
            for (node.outputs) |name| if (name.len != 0) try producer.put(allocator, name, index);
            for (node.inputs) |name| {
                if (name.len == 0) continue;
                const entry = try uses.getOrPut(allocator, name);
                entry.value_ptr.* = if (entry.found_existing) entry.value_ptr.* + 1 else 1;
            }
        }
        // A graph output read by nobody else still has to be produced.
        for (self.graph.outputs) |output| {
            const entry = try uses.getOrPut(allocator, output.name);
            entry.value_ptr.* = if (entry.found_existing) entry.value_ptr.* + 1 else 1;
        }

        for (self.graph.nodes, 0..) |node, index| {
            if (!std.mem.eql(u8, node.op_type, "MatMul") or node.inputs.len != 2) continue;
            const scores = node.inputs[0];
            const softmax_at = producer.get(scores) orelse continue;
            const middle = self.graph.nodes[softmax_at];
            if (!std.mem.eql(u8, middle.op_type, "Softmax")) continue;
            if (middle.int("axis", -1) != -1) continue;
            if ((uses.get(scores) orelse 0) != 1) continue;

            const paired = middle.inputs[0];
            const product_at = producer.get(paired) orelse continue;
            const first = self.graph.nodes[product_at];
            if (!std.mem.eql(u8, first.op_type, "MatMul") or first.inputs.len != 2) continue;
            if ((uses.get(paired) orelse 0) != 1) continue;

            try self.fused.put(allocator, index, .{
                .kind = .attention,
                .operands = .{ first.inputs[0], first.inputs[1], node.inputs[1] },
                .arity = 3,
                .product = product_at,
                .softmax = softmax_at,
            });
            try self.folded.put(allocator, product_at, {});
            try self.folded.put(allocator, softmax_at, {});
        }

        // A product the graph follows with a bias, which the product can add on
        // the way out instead of a second pass reading and writing all of it.
        for (self.graph.nodes, 0..) |node, index| {
            if (!std.mem.eql(u8, node.op_type, "Add") or node.inputs.len != 2) continue;
            for (0..2) |side| {
                const bias = node.inputs[side];
                const sum = node.inputs[1 - side];
                if ((uses.get(sum) orelse 0) != 1) continue;
                const tensor = self.graph.constant(bias) orelse continue;
                if (tensor.dtype != .f32 or tensor.dims.len != 1) continue;
                const at = producer.get(sum) orelse continue;
                if (self.folded.contains(at) or self.fused.contains(at)) continue;
                const dot = self.graph.nodes[at];
                if (!std.mem.eql(u8, dot.op_type, "MatMul") or dot.inputs.len != 2) continue;
                // The bias runs along the columns the product writes.
                const weights = self.graph.constant(dot.inputs[1]) orelse continue;
                if (weights.dims.len == 0 or weights.dims[weights.dims.len - 1] != tensor.dims[0]) continue;

                try self.fused.put(allocator, index, .{
                    .kind = .biased_product,
                    .operands = .{ dot.inputs[0], dot.inputs[1], bias },
                    .arity = 3,
                });
                try self.folded.put(allocator, at, {});
                break;
            }
        }

        for (self.graph.nodes, 0..) |_, index| {
            var swallowed: [4]usize = undefined;
            const source = self.geluAt(index, producer, uses, &swallowed) orelse continue;
            try self.fused.put(allocator, index, .{
                .kind = .gelu,
                .operands = .{ source, "", "" },
                .arity = 1,
            });
            for (swallowed) |at| try self.folded.put(allocator, at, {});
        }

        // These exports split the attention scale between Q and K, a pass over
        // each of them for one multiplication the scores can carry instead.
        var scaled = self.fused.iterator();
        while (scaled.next()) |entry| {
            const plan = entry.value_ptr;
            if (plan.kind != .attention) continue;
            for (plan.operands[0..2], 0..) |name, side| {
                if ((uses.get(name) orelse 0) != 1) continue;
                const at = producer.get(name) orelse continue;
                if (self.folded.contains(at) or self.fused.contains(at)) continue;
                const scale = self.scalarOperand(self.graph.nodes[at], "Mul") orelse continue;
                plan.operands[side] = scale.tensor;
                plan.scale *= scale.value;
                plan.swallowed[plan.swallowed_len] = @intCast(at);
                plan.swallowed_len += 1;
                try self.folded.put(allocator, at, {});
            }
        }

        // The rotary embedding, where there is a kernel for it. A CUDA build
        // has none, and a pattern recorded there would have nothing to run.
        if (self.env.gpu.rope != null) {
            for (self.graph.nodes, 0..) |_, index| {
                const plan = self.rotaryAt(index, producer, uses) orelse continue;
                try self.fused.put(allocator, index, plan);
                for (plan.swallowed[0..plan.swallowed_len]) |at| try self.folded.put(allocator, at, {});
            }
        }

        // A product whose bias is followed by nothing but the activation can
        // do that on the way out as well, which is a whole tensor neither
        // written nor read again -- 49 MB a layer in the encoder MLP.
        for (self.graph.nodes, 0..) |_, index| {
            const activation = self.fused.getPtr(index) orelse continue;
            if (activation.kind != .gelu) continue;
            const source = activation.operands[0];
            const at = producer.get(source) orelse continue;
            const shifted = self.fused.get(at) orelse continue;
            if (shifted.kind != .biased_product) continue;
            // The activation reads it twice and is the only thing that does,
            // which `geluAt` has already established.
            activation.* = .{
                .kind = .biased_product_gelu,
                .operands = shifted.operands,
                .arity = shifted.arity,
            };
            _ = self.fused.remove(at);
            try self.folded.put(allocator, at, {});
        }
    }

    const Producers = std.StringHashMapUnmanaged(usize);
    const Uses = std.StringHashMapUnmanaged(usize);

    /// The one value a scalar constant holds, if `name` names one.
    fn scalarConstant(self: *SessionState, name: []const u8) ?f32 {
        const tensor = self.graph.constant(name) orelse return null;
        if (tensor.dtype != .f32 or tensor.elementCount() != 1) return null;
        return tensor.f32s()[0];
    }

    /// A commutative binary node against a scalar constant: which input is the
    /// tensor, and what the constant is.
    fn scalarOperand(self: *SessionState, node: onnx.Node, op: []const u8) ?struct { tensor: []const u8, value: f32 } {
        if (!std.mem.eql(u8, node.op_type, op) or node.inputs.len != 2) return null;
        for (0..2) |side| {
            if (self.scalarConstant(node.inputs[side])) |value| {
                return .{ .tensor = node.inputs[1 - side], .value = value };
            }
        }
        return null;
    }

    /// Whether node `index` is the last of the five operators these exports
    /// spell GELU as -- `0.5 * t * (1 + erf(t / sqrt 2))` -- and if so, what
    /// `t` is. The shape is a diamond rather than a chain: `t` feeds both the
    /// division and the multiplication that closes it, so every intermediate
    /// has to be read once and `t` itself exactly twice, or something outside
    /// wants a value the fused kernel would not leave behind.
    fn geluAt(
        self: *SessionState,
        index: usize,
        producer: Producers,
        uses: Uses,
        swallowed: *[4]usize,
    ) ?[]const u8 {
        const half = self.scalarOperand(self.graph.nodes[index], "Mul") orelse return null;
        if (@abs(half.value - 0.5) > 1e-6) return null;
        if ((uses.get(half.tensor) orelse 0) != 1) return null;

        const product_at = producer.get(half.tensor) orelse return null;
        const closing = self.graph.nodes[product_at];
        if (!std.mem.eql(u8, closing.op_type, "Mul") or closing.inputs.len != 2) return null;

        for (0..2) |side| {
            const shifted = closing.inputs[side];
            const source = closing.inputs[1 - side];
            if ((uses.get(shifted) orelse 0) != 1) continue;
            if ((uses.get(source) orelse 0) != 2) continue;

            const shift_at = producer.get(shifted) orelse continue;
            const one = self.scalarOperand(self.graph.nodes[shift_at], "Add") orelse continue;
            if (@abs(one.value - 1.0) > 1e-6) continue;
            if ((uses.get(one.tensor) orelse 0) != 1) continue;

            const erf_at = producer.get(one.tensor) orelse continue;
            const erf = self.graph.nodes[erf_at];
            if (!std.mem.eql(u8, erf.op_type, "Erf") or erf.inputs.len != 1) continue;
            if ((uses.get(erf.inputs[0]) orelse 0) != 1) continue;

            const scale_at = producer.get(erf.inputs[0]) orelse continue;
            const scale = self.graph.nodes[scale_at];
            if (!std.mem.eql(u8, scale.op_type, "Div") or scale.inputs.len != 2) continue;
            if (!std.mem.eql(u8, scale.inputs[0], source)) continue;
            const root = self.scalarConstant(scale.inputs[1]) orelse continue;
            if (@abs(root - std.math.sqrt2) > 1e-5) continue;

            swallowed.* = .{ product_at, shift_at, erf_at, scale_at };
            return source;
        }
        return null;
    }

    /// The eleven operators a rotary embedding is spelled as, if `index` is
    /// the Add that ends one:
    ///
    ///     Add(Mul(x, cos),
    ///         Mul(Reshape(Concat(Unsqueeze(Neg(Squeeze(Split(Reshape(x))[1]))),
    ///                            Unsqueeze(Squeeze(Split(Reshape(x))[0])))),
    ///             sin))
    ///
    /// The reshapes and the split turn the last axis into pairs, and the
    /// concatenation puts each pair back with the two swapped and the first
    /// negated. Every intermediate has to be read by exactly one node, or
    /// folding it away would take a value something else still wants; `x`
    /// itself is read twice and stays.
    fn rotaryAt(self: *SessionState, index: usize, producer: Producers, uses: Uses) ?Fusion {
        if (self.fused.contains(index) or self.folded.contains(index)) return null;
        const node = self.graph.nodes[index];
        if (!std.mem.eql(u8, node.op_type, "Add") or node.inputs.len != 2) return null;

        for (0..2) |side| {
            const straight_at = self.onlyProducer(node.inputs[side], "Mul", producer, uses) orelse continue;
            const turned_at = self.onlyProducer(node.inputs[1 - side], "Mul", producer, uses) orelse continue;
            const straight = self.tableProduct(straight_at) orelse continue;
            const turned = self.tableProduct(turned_at) orelse continue;
            const back_at = self.onlyProducer(turned.value, "Reshape", producer, uses) orelse continue;
            // The reshape that ends the turn is the only thing that reads the
            // concatenation's value. The graph also asks it for its shape, to
            // build that very reshape's target -- arithmetic that has nothing
            // left to answer once the group is gone, so `deadShapeChain`
            // collects it rather than leaving it behind reading a tensor that
            // is no longer written.
            const seam = self.graph.nodes[back_at].inputs[0];
            const join_at = producer.get(seam) orelse continue;
            const join = self.graph.nodes[join_at];
            if (!std.mem.eql(u8, join.op_type, "Concat")) continue;
            if (join.inputs.len != 2 or join.int("axis", 0) != -1) continue;
            var asked: [4]u32 = undefined;
            const asked_len: u8 = switch (uses.get(seam) orelse 0) {
                1 => 0,
                2 => self.deadShapeChain(seam, back_at, uses, &asked) orelse continue,
                else => continue,
            };

            const lift_negated_at = self.onlyProducer(join.inputs[0], "Unsqueeze", producer, uses) orelse continue;
            const lift_at = self.onlyProducer(join.inputs[1], "Unsqueeze", producer, uses) orelse continue;
            if (!self.lastAxis(self.graph.nodes[lift_negated_at]) or !self.lastAxis(self.graph.nodes[lift_at])) continue;

            const negate_at = self.onlyProducer(self.graph.nodes[lift_negated_at].inputs[0], "Neg", producer, uses) orelse continue;
            const second_at = self.onlyProducer(self.graph.nodes[negate_at].inputs[0], "Squeeze", producer, uses) orelse continue;
            const first_at = self.onlyProducer(self.graph.nodes[lift_at].inputs[0], "Squeeze", producer, uses) orelse continue;
            if (!self.lastAxis(self.graph.nodes[second_at]) or !self.lastAxis(self.graph.nodes[first_at])) continue;

            const second = self.graph.nodes[second_at].inputs[0];
            const first = self.graph.nodes[first_at].inputs[0];
            const split_at = producer.get(first) orelse continue;
            if (split_at != (producer.get(second) orelse continue)) continue;
            const halves = self.graph.nodes[split_at];
            if (!std.mem.eql(u8, halves.op_type, "Split") or halves.outputs.len != 2) continue;
            if (halves.int("axis", 0) != -1) continue;
            // The two halves in the order the turn wants them: the second of
            // each pair negated in front of the first.
            if (!std.mem.eql(u8, halves.outputs[0], first) or !std.mem.eql(u8, halves.outputs[1], second)) continue;

            const pairs_at = self.onlyProducer(halves.inputs[0], "Reshape", producer, uses) orelse continue;
            if (!std.mem.eql(u8, self.graph.nodes[pairs_at].inputs[0], straight.value)) continue;

            var swallowed: [15]u32 = @splat(0);
            const core = [_]u32{
                @intCast(straight_at), @intCast(turned_at),       @intCast(back_at),
                @intCast(join_at),     @intCast(lift_negated_at), @intCast(lift_at),
                @intCast(negate_at),   @intCast(second_at),       @intCast(first_at),
                @intCast(split_at),    @intCast(pairs_at),
            };
            @memcpy(swallowed[0..core.len], &core);
            @memcpy(swallowed[core.len..][0..asked_len], asked[0..asked_len]);
            const length: u8 = @intCast(core.len + asked_len);
            for (swallowed[0..length]) |at| {
                if (self.fused.contains(at) or self.folded.contains(at)) return null;
            }
            // In graph order, so the fallback can run them as the graph would.
            std.mem.sort(u32, swallowed[0..length], {}, std.sort.asc(u32));

            return .{
                .kind = .rotary,
                .operands = .{ straight.value, straight.table, turned.table },
                .arity = 3,
                .swallowed = swallowed,
                .swallowed_len = length,
            };
        }
        return null;
    }

    /// The `Shape -> ... -> Reshape` an export builds a reshape's target with,
    /// walked from the reader of `name` that is not `sink` to `sink` itself.
    /// Every step has to produce one value that only the next step reads, so
    /// that folding the group away leaves none of it behind. Returns how many
    /// nodes it wrote into `steps`.
    fn deadShapeChain(
        self: *SessionState,
        name: []const u8,
        sink: usize,
        uses: Uses,
        steps: *[4]u32,
    ) ?u8 {
        var at = self.readerBesides(name, sink) orelse return null;
        var count: u8 = 0;
        while (count < steps.len) {
            const node = self.graph.nodes[at];
            const shaping = std.mem.eql(u8, node.op_type, "Shape") or
                std.mem.eql(u8, node.op_type, "Slice") or
                std.mem.eql(u8, node.op_type, "Concat") or
                std.mem.eql(u8, node.op_type, "Gather") or
                std.mem.eql(u8, node.op_type, "Unsqueeze");
            if (!shaping or node.outputs.len != 1 or node.outputs[0].len == 0) return null;
            if ((uses.get(node.outputs[0]) orelse 0) != 1) return null;
            steps[count] = @intCast(at);
            count += 1;
            const next = self.readerBesides(node.outputs[0], at) orelse return null;
            if (next == sink) return count;
            at = next;
        }
        return null;
    }

    /// The one node that reads `name` other than `besides`, if there is
    /// exactly one. Walking the graph for it is affordable because only a
    /// fusion asks, and only once, when the graph is opened.
    fn readerBesides(self: *SessionState, name: []const u8, besides: usize) ?usize {
        var found: ?usize = null;
        for (self.graph.nodes, 0..) |node, index| {
            if (index == besides) continue;
            for (node.inputs) |reads| {
                if (!std.mem.eql(u8, reads, name)) continue;
                if (found != null) return null;
                found = index;
                break;
            }
        }
        return found;
    }
    /// The node that produced `name`, if it is one of `op` and nothing else
    /// reads what it produced.
    fn onlyProducer(
        self: *SessionState,
        name: []const u8,
        op: []const u8,
        producer: Producers,
        uses: Uses,
    ) ?usize {
        if ((uses.get(name) orelse 0) != 1) return null;
        const at = producer.get(name) orelse return null;
        if (!std.mem.eql(u8, self.graph.nodes[at].op_type, op)) return null;
        return at;
    }

    /// A product of one tensor by one constant table, which is what both
    /// halves of a rotary embedding are.
    fn tableProduct(self: *SessionState, at: usize) ?struct { value: []const u8, table: []const u8 } {
        const node = self.graph.nodes[at];
        if (node.inputs.len != 2) return null;
        for (0..2) |side| {
            if (self.graph.constant(node.inputs[1 - side]) != null) continue;
            const table = self.graph.constant(node.inputs[side]) orelse continue;
            if (table.dtype != .f32) continue;
            return .{ .value = node.inputs[1 - side], .table = node.inputs[side] };
        }
        return null;
    }

    /// Whether a Squeeze or an Unsqueeze names the last axis, which is the
    /// only one a rotary embedding touches.
    fn lastAxis(self: *SessionState, node: onnx.Node) bool {
        if (node.inputs.len < 2) return false;
        const axes = self.graph.constant(node.inputs[1]) orelse return false;
        if (axes.dtype != .i64 or axes.elementCount() != 1) return false;
        return axes.i64s()[0] == -1;
    }

    /// What a fused node reads, which is not what the graph says it reads.
    fn effectiveInputs(self: *SessionState, index: usize, node: onnx.Node) []const []const u8 {
        const plan = self.fused.getPtr(index) orelse return node.inputs;
        return plan.inputs();
    }

    fn preloadInitializers(self: *SessionState) !void {
        if (self.initializers_preloaded) return;
        for (self.graph.initializers) |source| {
            if (source.dtype != .f32 or source.name.len == 0) continue;
            const entry = try self.constants.getOrPut(self.env.allocator, source.name);
            if (entry.found_existing) continue;
            entry.value_ptr.* = try driver.Buffer(Element).alloc(source.elementCount());
            errdefer {
                entry.value_ptr.free();
                _ = self.constants.remove(source.name);
            }
            try uploadFloats(self.env.allocator, entry.value_ptr.*, source.f32s());
        }
        self.initializers_preloaded = true;
    }

    fn execute(
        self: *SessionState,
        arena: std.mem.Allocator,
        values: *std.StringHashMapUnmanaged(*Tensor),
        node: onnx.Node,
        index: usize,
    ) !void {
        if (self.fused.get(index)) |plan| return switch (plan.kind) {
            .attention => self.attention(arena, values, node, plan),
            .gelu => self.unaryOf(arena, values, try self.input(arena, values, plan.operands[0]), node.outputs[0], .gelu),
            .rotary => self.rotary(arena, values, node, plan),
            .biased_product => self.matmulOf(arena, values, plan.operands[0], plan.operands[1], plan.operands[2], .none, node.outputs[0]),
            .biased_product_gelu => self.matmulOf(arena, values, plan.operands[0], plan.operands[1], plan.operands[2], .gelu, node.outputs[0]),
        };
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
        if (std.meta.stringToEnum(Binary, node.op_type)) |op| return self.binary(arena, values, node, op);
        if (std.mem.eql(u8, node.op_type, "Mod")) return self.binary(arena, values, node, .Div);
        if (std.meta.stringToEnum(Unary, node.op_type)) |op| return self.unary(arena, values, node, op);
        if (std.meta.stringToEnum(Compare, node.op_type)) |op| return self.compare(arena, values, node, op);
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
        if (std.mem.eql(u8, node.op_type, "DynamicQuantizeLinear")) return self.dynamicQuantizeLinear(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "MatMulInteger")) return self.matmulInteger(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "DequantizeLinear")) return self.dequantizeLinear(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "QuantizeLinear")) return self.quantizeLinear(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "ReduceMean")) return self.reduceMean(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "ReduceMax")) return self.reduceMax(arena, values, node);
        if (std.mem.eql(u8, node.op_type, "LSTM")) return self.lstm(arena, values, node);
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
                entry.value_ptr.* = try driver.Buffer(Element).alloc(source.elementCount());
                errdefer _ = self.constants.remove(name);
                try uploadFloats(self.env.allocator, entry.value_ptr.*, source.f32s());
            }
            tensor.* = .{ .dtype = .f32, .dims = source.dims, .data = .{ .constant_gpu = entry.value_ptr.* } };
        } else {
            tensor.* = .{ .dtype = source.dtype, .dims = source.dims, .data = .{ .host = source.data } };
        }
        try values.put(arena, name, tensor);
        return tensor;
    }

    /// A transposed convolution's weight in the order the matrix product wants
    /// it: a row per output channel and tap, by input channel, where the graph
    /// stores it input channel first. Reordered on the host once and kept for
    /// the life of the session, like any other constant.
    fn shuffledWeight(self: *SessionState, name: []const u8) !driver.Buffer(Element) {
        if (self.shuffled.get(name)) |buffer| return buffer;
        const source = self.graph.constant(name) orelse return Error.MissingValue;
        if (source.dtype != .f32 or source.dims.len != 4) return Error.UnsupportedDataType;
        const in_channels: usize = @intCast(source.dims[0]);
        const rows: usize = @intCast(source.dims[1] * source.dims[2] * source.dims[3]);
        const values = source.f32s();
        if (values.len != in_channels * rows) return Error.InvalidShape;

        const reordered = try self.env.allocator.alloc(Element, values.len);
        defer self.env.allocator.free(reordered);
        for (0..in_channels) |channel| {
            const from = values[channel * rows ..][0..rows];
            for (from, 0..) |value, row| reordered[row * in_channels + channel] = @floatCast(value);
        }

        const buffer = try driver.Buffer(Element).alloc(values.len);
        errdefer buffer.free();
        try buffer.upload(reordered);
        try self.shuffled.put(self.env.allocator, name, buffer);
        return buffer;
    }

    /// A quantized weight matrix is not float data and never becomes a tensor,
    /// so it goes to the device as raw bytes, once, and is addressed directly.
    fn constantBytes(self: *SessionState, name: []const u8) !driver.DevicePtr {
        if (self.constants.get(name)) |buffer| return buffer.ptr;
        const source = self.graph.constant(name) orelse return Error.MissingValue;
        const stride = @sizeOf(Element);
        const buffer = try driver.Buffer(Element).alloc((source.data.len + stride - 1) / stride);
        errdefer buffer.free();
        const bytes: driver.Buffer(u8) = .{ .ptr = buffer.ptr, .len = buffer.len * stride };
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
        const storage = try self.arena.create(Storage);
        storage.* = .{ .buffer = try self.env.pool.take(count) };
        return storage;
    }

    fn release(self: *SessionState, tensor: *Tensor) void {
        switch (tensor.data) {
            .gpu => |storage| {
                storage.references -= 1;
                if (storage.references == 0) {
                    self.env.pool.give(self.env.allocator, storage.buffer);
                }
                tensor.data = .{ .host = &.{} };
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
        const is_1d = x.dims.len == 3 and weight.dims.len == 3;
        if (!is_1d and (x.dims.len != 4 or weight.dims.len != 4)) return Error.InvalidShape;

        const x_d0 = x.dims[0];
        const x_c = x.dims[1];
        const x_h: i64 = if (is_1d) 1 else x.dims[2];
        const x_w: i64 = if (is_1d) x.dims[2] else x.dims[3];

        const w_out = weight.dims[0];
        const w_in = weight.dims[1];
        const w_h: i64 = if (is_1d) 1 else weight.dims[2];
        const w_w: i64 = if (is_1d) weight.dims[2] else weight.dims[3];

        const strides = node.ints("strides");
        const pads = node.ints("pads");
        const dilations = node.ints("dilations");

        const sh: i64 = if (is_1d) 1 else (if (strides.len == 0) 1 else strides[0]);
        const sw: i64 = if (is_1d) (if (strides.len == 0) 1 else strides[0]) else (if (strides.len < 2) 1 else strides[1]);

        const ph0: i64 = if (is_1d) 0 else (if (pads.len == 0) 0 else pads[0]);
        const pw0: i64 = if (is_1d) (if (pads.len == 0) 0 else pads[0]) else (if (pads.len < 2) 0 else pads[1]);
        const ph1: i64 = if (is_1d) 0 else (if (pads.len < 4) ph0 else pads[2]);
        const pw1: i64 = if (is_1d) (if (pads.len < 2) pw0 else pads[1]) else (if (pads.len < 4) pw0 else pads[3]);

        const dh: i64 = if (is_1d) 1 else (if (dilations.len == 0) 1 else dilations[0]);
        const dw: i64 = if (is_1d) (if (dilations.len == 0) 1 else dilations[0]) else (if (dilations.len < 2) 1 else dilations[1]);

        const out_h = @divFloor(x_h + ph0 + ph1 - dh * (w_h - 1) - 1, sh) + 1;
        const out_w = @divFloor(x_w + pw0 + pw1 - dw * (w_w - 1) - 1, sw) + 1;

        const dims = if (is_1d)
            try arena.dupe(i64, &.{ x_d0, w_out, out_w })
        else
            try arena.dupe(i64, &.{ x_d0, w_out, out_h, out_w });

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
                @intCast(x_c),   @intCast(x_h),   @intCast(x_w),
                @intCast(out_h), @intCast(out_w), @intCast(w_h),
                @intCast(w_w),   @intCast(sh),    @intCast(sw),
                @intCast(ph0),   @intCast(pw0),   @intCast(dh),
                @intCast(dw),
            };
            try self.env.uploadMetadata(&meta);
            const rows = w_out;
            const pixels = out_h * out_w;
            const depth = w_in * w_h * w_w;
            const engines: enum { tensor, xmx, staged } = if (self.env.gpu.conv2d_gemm_tensor != null)
                .tensor
            else if (self.env.gpu.conv2d_gemm_xmx != null)
                .xmx
            else
                .staged;
            const kernel = switch (engines) {
                .tensor => self.env.gpu.conv2d_gemm_tensor.?,
                .xmx => self.env.gpu.conv2d_gemm_xmx.?,
                .staged => self.env.gpu.conv2d_gemm,
            };
            const tile_m: i64 = switch (engines) {
                .tensor => conv_tensor_tile_m,
                .xmx => matmul_xmx_tile_m,
                .staged => matmul_tile_m,
            };
            const tile_n: i64 = switch (engines) {
                .tensor => conv_tensor_tile_n,
                .xmx => matmul_xmx_tile_n,
                .staged => matmul_tile_n,
            };
            try kernel.launch(.{
                .x = @intCast(@divFloor(pixels + tile_n - 1, tile_n)),
                .y = @intCast(@divFloor(rows + tile_m - 1, tile_m)),
                .z = @intCast(x_d0),
            }, .{ .x = 16, .y = switch (engines) {
                .tensor => conv_tensor_subgroups,
                .xmx => matmul_xmx_subgroups,
                .staged => 16,
            } }, .{
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
            @intCast(x_c),   @intCast(x_h),   @intCast(x_w), @intCast(w_out),
            @intCast(out_h), @intCast(out_w), @intCast(w_h), @intCast(w_w),
            @intCast(sh),    @intCast(sw),    @intCast(ph0), @intCast(pw0),
            @intCast(dh),    @intCast(dw),    @intCast(groups),
        };
        try self.env.uploadMetadata(&meta);
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
        try self.env.uploadMetadata(&meta);
        const xb = try x.gpuBuffer();
        const wb = try weight.gpuBuffer();
        const bias = if (node.inputs.len > 2 and node.inputs[2].len != 0) try self.input(arena, values, node.inputs[2]) else null;
        const bias_buffer = if (bias) |b| try b.gpuBuffer() else xb;
        if (sh == weight.dims[2] and sw == weight.dims[3] and
            ph0 == 0 and pw0 == 0 and ph1 == 0 and pw1 == 0)
        {
            const pixels = x.dims[2] * x.dims[3];
            const features = weight.dims[1] * weight.dims[2] * weight.dims[3];

            // A non-overlapping transposed convolution is a matrix product
            // followed by a pixel shuffle. Where the weight can be reordered
            // into the product's terms, that product is the fast one -- on the
            // matrix engines, reading whole rows -- and the shuffle is one
            // pass; `convTranspose2dGemm` below gathers both operands as it
            // stages them instead, and cannot use either.
            if (self.env.gpu.pixel_shuffle) |shuffle| taken: {
                if (self.graph.constant(node.inputs[1]) == null) break :taken;
                const reordered = self.shuffledWeight(node.inputs[1]) catch break :taken;
                const staged = try self.newStorage(@intCast(x.dims[0] * features * pixels));
                defer self.releaseStorage(staged);
                try self.product(.{
                    .a = reordered.ptr,
                    .b = xb.ptr,
                    .c = staged.buffer.ptr,
                    .m = @intCast(features),
                    .n = @intCast(pixels),
                    .k = @intCast(x.dims[1]),
                    .b_batch = @intCast(x.dims[1] * pixels),
                    .c_batch = @intCast(features * pixels),
                    .batches = @intCast(x.dims[0]),
                });
                const block: u32 = 256;
                try shuffle.launch(.{ .x = @intCast((count + block - 1) / block) }, .{ .x = block }, .{
                    staged.buffer.ptr,
                    bias_buffer.ptr,
                    storage.buffer.ptr,
                    @as(u32, @intCast(weight.dims[1])),
                    @as(u32, @intCast(x.dims[2])),
                    @as(u32, @intCast(x.dims[3])),
                    @as(u32, @intCast(weight.dims[2])),
                    @as(u32, @intCast(weight.dims[3])),
                    @as(u32, @intCast(count)),
                    @as(u32, if (bias != null) 1 else 0),
                });
                return self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
            }

            // Computing a whole output tile per workgroup avoids repeating the
            // channel reduction for every scalar output in the generic kernel.
            try self.env.gpu.conv_transpose2d_gemm.launch(.{
                .x = @intCast(@divFloor(features + matmul_tile_n - 1, matmul_tile_n)),
                .y = @intCast(@divFloor(pixels + matmul_tile_m - 1, matmul_tile_m)),
                .z = @intCast(x.dims[0]),
            }, .{ .x = 16, .y = 16 }, .{
                xb.ptr,
                wb.ptr,
                bias_buffer.ptr,
                storage.buffer.ptr,
                @as(u32, @intCast(x.dims[2])),
                @as(u32, @intCast(x.dims[3])),
                @as(u32, @intCast(weight.dims[1])),
                @as(u32, @intCast(out_h)),
                @as(u32, @intCast(out_w)),
                @as(u32, @intCast(weight.dims[2])),
                @as(u32, @intCast(weight.dims[3])),
                @as(u32, @intCast(pixels)),
                @as(u32, @intCast(features)),
                @as(u32, @intCast(x.dims[1])),
                @as(u32, if (bias != null) 1 else 0),
            });
            return self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
        }
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
                const source_offset = axis_offset * dense[axis];
                const xb = try x.gpuBuffer();
                try self.strided(xb.ptr, storage.buffer.ptr, metadata[0 .. 2 * x.dims.len], part_count, source_offset, 0);
                try self.put(arena, values, output_name, .{ .dtype = .f32, .dims = part_dims, .data = .{ .gpu = storage } });
            } else return Error.UnsupportedDataType;

            axis_offset += @intCast(part_len);
        }
    }

    /// The copy behind Transpose, Split, Slice and Expand: output element `i`
    /// comes from `src` walked by the strides `metadata` carries after its
    /// dims. Where the innermost axis is contiguous on both sides -- which is
    /// nearly every permutation these graphs ask for, the head dimension
    /// staying put -- one work item carries a whole group instead of one
    /// element. `metadata` is rewritten to say so.
    fn strided(
        self: *SessionState,
        src: driver.DevicePtr,
        dst: driver.DevicePtr,
        given: []u32,
        count: usize,
        src_offset: usize,
        dst_offset: usize,
    ) !void {
        const metadata = collapseAxes(given);
        const rank = metadata.len / 2;
        const block: u32 = 256;
        grouped: {
            const kernel = self.env.gpu.copy_vec orelse break :grouped;
            if (rank == 0) break :grouped;
            if (metadata[2 * rank - 1] != 1) break :grouped;
            if (metadata[rank - 1] % lane_step != 0) break :grouped;
            if (src_offset % lane_step != 0 or dst_offset % lane_step != 0) break :grouped;
            // A stride that is not a whole number of groups would put a work
            // item astride two of them. Slice spells a reversed axis as a
            // negative stride, whose bit pattern fails this too.
            for (metadata[rank .. 2 * rank - 1]) |stride| {
                if (stride % lane_step != 0) break :grouped;
            }
            metadata[rank - 1] /= lane_step;
            metadata[2 * rank - 1] = lane_step;
            try self.env.uploadMetadata(metadata);
            const groups = count / lane_step;
            return kernel.launch(.{ .x = @intCast((groups + block - 1) / block) }, .{ .x = block }, .{
                src,
                dst,
                self.env.metadata.ptr,
                @as(u32, @intCast(rank)),
                @as(u32, @intCast(groups)),
                @as(u32, @intCast(src_offset)),
                @as(u32, @intCast(dst_offset)),
            });
        }
        try self.env.uploadMetadata(metadata);
        try self.env.gpu.copy.launch(.{ .x = @intCast((count + block - 1) / block) }, .{ .x = block }, .{
            src,
            dst,
            self.env.metadata.ptr,
            @as(u32, @intCast(rank)),
            @as(u32, @intCast(count)),
            @as(u32, @intCast(src_offset)),
            @as(u32, @intCast(dst_offset)),
        });
    }

    fn releaseStorage(self: *SessionState, storage: *Storage) void {
        storage.references -= 1;
        if (storage.references == 0) {
            self.env.pool.give(self.env.allocator, storage.buffer);
        }
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
        const ids_count = try indices.count();
        const resolved = try arena.alloc(u32, ids_count);
        for (0..ids_count) |i| {
            const raw_id: i64 = switch (indices.dtype) {
                .i64 => (try indices.i64s())[i],
                .i32 => @as([]const i32, @alignCast(std.mem.bytesAsSlice(i32, indices.data.host)))[i],
                .u8 => indices.data.host[i],
                .i8 => @as(i8, @bitCast(indices.data.host[i])),
                else => return Error.UnsupportedDataType,
            };
            const normalized = if (raw_id < 0) @as(i64, @intCast(axis_len)) + raw_id else raw_id;
            if (normalized < 0 or normalized >= axis_len) return Error.InvalidShape;
            resolved[i] = @intCast(normalized);
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
        self.env.indices = try driver.Buffer(u32).alloc(count);
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
            for (axes) |axis| if (normalizeAxis(axis, rank) == i) {
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
        const xb = try x.gpuBuffer();
        try self.strided(xb.ptr, storage.buffer.ptr, metadata[0 .. 2 * rank], count, 0, 0);
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
        const xb = try x.gpuBuffer();
        try self.strided(xb.ptr, storage.buffer.ptr, metadata[0 .. 2 * x.dims.len], count, source_offset, 0);
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
        try self.env.uploadMetadata(metadata[0 .. 4 * x.dims.len]);
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

    /// C = A x B, on the matrix engines when the GPU has them. Both kernels
    /// take the same arguments and differ only in how much of C a work group
    /// covers, so the choice is confined to this one place.
    const Product = struct {
        a: driver.DevicePtr,
        b: driver.DevicePtr,
        c: driver.DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        a_batch: u32 = 0,
        b_batch: u32 = 0,
        c_batch: u32,
        batches: u32 = 1,
        b_transposed: bool = false,
        /// Added by column on the way out, where an Add the graph put after
        /// the product has been folded into it.
        bias: driver.DevicePtr = driver.null_ptr,
        has_bias: bool = false,
        /// Applied after the bias, where the activation that followed the
        /// product has been folded in as well.
        activation: Activation = .none,
    };

    const Activation = enum(u32) { none, gelu };

    fn product(self: *SessionState, args: Product) !void {
        // A 2D block load reads a plane whose base is 64 byte aligned, in rows
        // a multiple of 16 bytes wide and at least 64 wide, and it has nowhere
        // to put a transposed right operand. Where all of that holds the
        // blocked kernel runs about twice as fast; where it does not, the
        // staged one takes the product unchanged.
        const blocked: ?driver.Function = suited: {
            if (Element != f16 or args.b_transposed) break :suited null;
            const kernel = self.env.gpu.matmul_xmx_block orelse break :suited null;
            if (args.k % 8 != 0 or args.n % 8 != 0) break :suited null;
            if (args.k < 32 or args.n < 32) break :suited null;
            if (args.a_batch % 32 != 0 or args.b_batch % 32 != 0) break :suited null;
            break :suited kernel;
        };

        const engines: enum { tensor, blocked, xmx, simd, staged } = if (self.env.gpu.matmul_tensor != null)
            .tensor
        else if (blocked != null)
            .blocked
        else if (self.env.gpu.matmul_xmx != null)
            .xmx
        else if (self.env.gpu.matmul_simd != null)
            .simd
        else
            .staged;
        const kernel = switch (engines) {
            .tensor => self.env.gpu.matmul_tensor.?,
            .blocked => blocked.?,
            .xmx => self.env.gpu.matmul_xmx.?,
            .simd => self.env.gpu.matmul_simd.?,
            .staged => self.env.gpu.matmul,
        };
        const tile_m: u32 = switch (engines) {
            .tensor => matmul_tensor_tile_m,
            .blocked => matmul_block_tile_m,
            .xmx => matmul_xmx_tile_m,
            .simd => matmul_simd_tile_m,
            .staged => matmul_tile_m,
        };
        const tile_n: u32 = switch (engines) {
            .tensor => matmul_tensor_tile_n,
            .blocked => matmul_block_tile_n,
            .xmx => matmul_xmx_tile_n,
            .simd => matmul_simd_tile_n,
            .staged => matmul_tile_n,
        };
        const subgroups: u32 = switch (engines) {
            .tensor => matmul_tensor_subgroups,
            .blocked => matmul_block_subgroups,
            .xmx => matmul_xmx_subgroups,
            .simd => matmul_simd_subgroups,
            .staged => 16,
        };
        const tiles_m = (args.m + tile_m - 1) / tile_m;
        const tiles_n = (args.n + tile_n - 1) / tile_n;

        // Both operands span k, so the rows and columns of C a work group
        // block can cover with its operands still in cache fall as k rises.
        // Split what fits evenly between the two, and never go below one tile.
        const span = matmul_cache_budget / (@sizeOf(Element) * @max(args.k, 1));
        const block_m = std.math.clamp(span / (2 * tile_m), 1, tiles_m);
        const block_n = std.math.clamp(span / (2 * tile_n), 1, tiles_n);
        const blocks_m = (tiles_m + block_m - 1) / block_m;
        const blocks_n = (tiles_n + block_n - 1) / block_n;

        try kernel.launch(.{
            // A whole number of blocks: the work groups that overhang C on the
            // far edges return before they touch anything.
            .x = blocks_m * block_m * blocks_n * block_n,
            .z = args.batches,
        }, .{ .x = 16, .y = subgroups }, .{
            args.a,
            args.b,
            args.c,
            args.m,
            args.n,
            args.k,
            args.a_batch,
            args.b_batch,
            args.c_batch,
            @as(u32, if (args.b_transposed) 1 else 0),
            if (args.has_bias) args.bias else args.a,
            @as(u32, if (args.has_bias) 1 else 0),
            @intFromEnum(args.activation),
            block_m,
            block_n,
        });
    }

    /// x turned by the rotary embedding tables, in one pass over it rather
    /// than the eleven operators the graph spells it with. `node` is the Add
    /// those end in, the one the rest of the graph reads.
    fn rotary(
        self: *SessionState,
        arena: std.mem.Allocator,
        values: *std.StringHashMapUnmanaged(*Tensor),
        node: onnx.Node,
        plan: Fusion,
    ) !void {
        const x = try self.input(arena, values, plan.operands[0]);
        const cosine = try self.input(arena, values, plan.operands[1]);
        const sine = try self.input(arena, values, plan.operands[2]);

        if (self.env.gpu.rope) |kernel| suited: {
            if (x.dtype != .f32 or cosine.dtype != .f32 or sine.dtype != .f32) break :suited;
            if (x.dims.len == 0 or @rem(x.dims[x.dims.len - 1], 2) != 0) break :suited;
            if (!std.mem.eql(i64, cosine.dims, sine.dims)) break :suited;
            // Both tables have to be the tail of x, repeating over everything
            // in front of it; anything else is not the embedding this stands
            // for whatever the operators around it looked like.
            const period = repeatingPeriod(cosine.dims, x.dims) orelse break :suited;
            const count = try elementCount(x.dims);

            const dims = try arena.dupe(i64, x.dims);
            const storage = try self.newStorage(count);
            errdefer self.releaseStorage(storage);
            const xb = try x.gpuBuffer();
            const cb = try cosine.gpuBuffer();
            const sb = try sine.gpuBuffer();
            const block: u32 = 256;
            if (self.env.gpu.rope_vec) |vec| taken: {
                const table_groups = groupPeriod(count, period) orelse break :taken;
                const groups: u32 = @intCast(count / lane_step);
                try vec.launch(.{ .x = (groups + block - 1) / block }, .{ .x = block }, .{
                    xb.ptr, cb.ptr,       sb.ptr,      storage.buffer.ptr,
                    groups, table_groups, @as(f32, 1),
                });
                return self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
            }
            try kernel.launch(.{ .x = @intCast((count + block - 1) / block) }, .{ .x = block }, .{
                xb.ptr,
                cb.ptr,
                sb.ptr,
                storage.buffer.ptr,
                @as(u32, @intCast(count)),
                @as(u32, @intCast(period)),
                @as(f32, 1),
            });
            return self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
        }

        // Not what the kernel was written for: run the operators the fusion
        // swallowed, in the order the graph has them, and then this one. They
        // are named rather than dispatched through `execute` because that
        // would make the two functions' error sets depend on each other.
        for (plan.swallowed[0..plan.swallowed_len]) |at| {
            const step = self.graph.nodes[at];
            if (std.mem.eql(u8, step.op_type, "Mul")) {
                try self.binary(arena, values, step, .Mul);
            } else if (std.mem.eql(u8, step.op_type, "Neg")) {
                try self.unary(arena, values, step, .Neg);
            } else if (std.mem.eql(u8, step.op_type, "Reshape")) {
                try self.reshape(arena, values, step);
            } else if (std.mem.eql(u8, step.op_type, "Split")) {
                try self.split(arena, values, step);
            } else if (std.mem.eql(u8, step.op_type, "Squeeze")) {
                try self.squeeze(arena, values, step);
            } else if (std.mem.eql(u8, step.op_type, "Unsqueeze")) {
                try self.unsqueeze(arena, values, step);
            } else if (std.mem.eql(u8, step.op_type, "Concat")) {
                try self.concat(arena, values, step);
            } else return Error.UnsupportedOperator;
        }
        try self.binary(arena, values, node, .Add);
        for (plan.swallowed[0..plan.swallowed_len]) |at| {
            for (self.graph.nodes[at].outputs) |name| {
                if (name.len == 0) continue;
                if (values.get(name)) |tensor| self.release(tensor);
            }
        }
    }

    /// Scaled dot product attention: the product against the keys, the softmax
    /// over them and the product against the values, as one kernel that never
    /// writes the score matrix down. `node` is the second product, the one the
    /// rest of the graph reads.
    fn attention(
        self: *SessionState,
        arena: std.mem.Allocator,
        values: *std.StringHashMapUnmanaged(*Tensor),
        node: onnx.Node,
        plan: Fusion,
    ) !void {
        const q = try self.input(arena, values, plan.operands[0]);
        const k = try self.input(arena, values, plan.operands[1]);
        const v = try self.input(arena, values, plan.operands[2]);

        // The blocked kernel reads its operands where they lie; the staged one
        // is what a driver without the 2D block loads falls back to. They take
        // the same arguments and produce the same tensor.
        if (self.env.gpu.attention_block orelse self.env.gpu.attention) |kernel| suited: {
            if (q.dtype != .f32 or k.dtype != .f32 or v.dtype != .f32) break :suited;
            const rank = q.dims.len;
            if (rank < 2 or k.dims.len != rank or v.dims.len != rank) break :suited;
            for (q.dims[0 .. rank - 2], k.dims[0 .. rank - 2], v.dims[0 .. rank - 2]) |a, b, c| {
                if (a != b or a != c) break :suited;
            }
            const queries = q.dims[rank - 2];
            const head = q.dims[rank - 1];
            const keys = k.dims[rank - 1];
            if (head != attention_head) break :suited;
            if (k.dims[rank - 2] != head or v.dims[rank - 2] != keys or v.dims[rank - 1] != head) break :suited;
            // The kernel stages a whole key step at a time and does not guard
            // the tail of one.
            if (@rem(keys, attention_key_step) != 0) break :suited;

            const batches = try elementCount(q.dims[0 .. rank - 2]);
            const dims = try arena.dupe(i64, q.dims);
            const storage = try self.newStorage(try elementCount(dims));
            errdefer self.releaseStorage(storage);
            try kernel.launch(.{
                .x = @intCast(@divFloor(queries + attention_query_tile - 1, attention_query_tile)),
                .y = @intCast(batches),
            }, .{ .x = 16, .y = attention_subgroups }, .{
                (try q.gpuBuffer()).ptr,
                (try k.gpuBuffer()).ptr,
                (try v.gpuBuffer()).ptr,
                storage.buffer.ptr,
                @as(u32, @intCast(queries)),
                @as(u32, @intCast(keys)),
                plan.scale,
            });
            return self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
        }

        // Not what the kernel was written for: run the two folded products --
        // the fusion pass has already established what they are -- along with
        // any scale it swallowed on the way in, and then this node, handing
        // their intermediates straight back.
        for (plan.swallowed[0..plan.swallowed_len]) |at| {
            try self.binary(arena, values, self.graph.nodes[at], .Mul);
        }
        const first = self.graph.nodes[plan.product];
        const middle = self.graph.nodes[plan.softmax];
        try self.matmul(arena, values, first);
        try self.softmax(arena, values, middle);
        try self.matmul(arena, values, node);
        for ([_][]const u8{ first.outputs[0], middle.outputs[0] }) |name| {
            if (values.get(name)) |tensor| self.release(tensor);
        }
        for (plan.swallowed[0..plan.swallowed_len]) |at| {
            if (values.get(self.graph.nodes[at].outputs[0])) |tensor| self.release(tensor);
        }
    }

    fn matmul(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node) !void {
        return self.matmulOf(arena, values, node.inputs[0], node.inputs[1], null, .none, node.outputs[0]);
    }

    /// The same over names given directly, so a product with an Add folded
    /// into it can name operands the node it stands for does not.
    fn matmulOf(
        self: *SessionState,
        arena: std.mem.Allocator,
        values: *std.StringHashMapUnmanaged(*Tensor),
        a_name: []const u8,
        b_name: []const u8,
        bias_name: ?[]const u8,
        activation: Activation,
        out_name: []const u8,
    ) !void {
        const a = try self.input(arena, values, a_name);
        const b = try self.input(arena, values, b_name);
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

        const shift = if (bias_name) |name| try (try self.input(arena, values, name)).gpuBuffer() else null;
        try self.product(.{
            .a = ab.ptr,
            .b = bb.ptr,
            .c = storage.buffer.ptr,
            .m = @intCast(rows),
            .n = @intCast(n),
            .k = @intCast(k),
            .a_batch = if (a_batches == 1) 0 else @intCast(m * k),
            .b_batch = if (b_batches == 1) 0 else @intCast(k * n),
            .c_batch = @intCast(m * n),
            .batches = @intCast(grid_batches),
            .bias = if (shift) |buffer| buffer.ptr else driver.null_ptr,
            .has_bias = shift != null,
            .activation = activation,
        });
        try self.put(arena, values, out_name, .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
    }

    fn unary(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node, op: Unary) !void {
        return self.unaryOf(arena, values, try self.input(arena, values, node.inputs[0]), node.outputs[0], op);
    }

    /// The same over a tensor and an output name given directly, which is how
    /// a fused activation reaches it: the node it stands for names neither.
    fn unaryOf(
        self: *SessionState,
        arena: std.mem.Allocator,
        values: *std.StringHashMapUnmanaged(*Tensor),
        x: *Tensor,
        name: []const u8,
        op: Unary,
    ) !void {
        if (x.dtype != .f32) return Error.UnsupportedDataType;
        const count = try x.count();
        const storage = try self.newStorage(count);
        errdefer self.releaseStorage(storage);
        const xb = try x.gpuBuffer();
        const block: u32 = 256;
        // Sign and IsNaN are the two the vector kernel leaves out; nothing in
        // these graphs reaches them often enough to be worth the select.
        scalar: {
            if (count % lane_step != 0 or op == .Sign or op == .IsNaN) break :scalar;
            const kernel = self.env.gpu.unary_vec orelse break :scalar;
            const groups = count / lane_step;
            try kernel.launch(.{ .x = @intCast((groups + block - 1) / block) }, .{ .x = block }, .{
                xb.ptr, storage.buffer.ptr, @as(u32, @intCast(groups)), @intFromEnum(op),
            });
            const dims = try arena.dupe(i64, x.dims);
            return self.put(arena, values, name, .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
        }
        try self.env.gpu.unary.launch(.{ .x = @intCast((count + block - 1) / block) }, .{ .x = block }, .{ xb.ptr, storage.buffer.ptr, @as(u32, @intCast(count)), @intFromEnum(op) });
        const dims = try arena.dupe(i64, x.dims);
        try self.put(arena, values, name, .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
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
        // A power of two, since the fallback tree reduction halves it, and
        // capped well below the hardware maximum: a row is reduced twice, and a
        // wide block spends more on the reduction than on the row, while a
        // narrow one just loops over more of the row.
        var block: u32 = 32;
        while (block < cols and block < 256) block *= 2;
        try self.env.gpu.softmax.launch(.{ .x = @intCast(rows) }, .{ .x = block }, .{ xb.ptr, storage.buffer.ptr, @as(u32, @intCast(cols)) });
        const dims = try arena.dupe(i64, x.dims);
        try self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
    }

    fn binary(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node, op: Binary) !void {
        const a = try self.input(arena, values, node.inputs[0]);
        const b = try self.input(arena, values, node.inputs[1]);
        if ((a.dtype == .i64 or a.dtype == .i32) and (b.dtype == .i64 or b.dtype == .i32)) {
            const dims = try broadcastShape(arena, a.dims, b.dims);
            if (dims.len > max_rank) return Error.RankTooLarge;
            var astrides: [max_rank]u32 = @splat(0);
            var bstrides: [max_rank]u32 = @splat(0);
            makeBroadcastStrides(a.dims, dims, &astrides);
            makeBroadcastStrides(b.dims, dims, &bstrides);
            const a_count = try a.count();
            const b_count = try b.count();
            const av = try arena.alloc(i64, a_count);
            for (0..a_count) |i| av[i] = if (a.dtype == .i64) (try a.i64s())[i] else @as([]const i32, @alignCast(std.mem.bytesAsSlice(i32, a.data.host)))[i];
            const bv = try arena.alloc(i64, b_count);
            for (0..b_count) |i| bv[i] = if (b.dtype == .i64) (try b.i64s())[i] else @as([]const i32, @alignCast(std.mem.bytesAsSlice(i32, b.data.host)))[i];

            const out = try arena.alloc(i64, try elementCount(dims));
            const modulo = std.mem.eql(u8, node.op_type, "Mod");
            for (out, 0..) |*value, i| {
                const x = av[hostOffset(i, dims, &astrides)];
                const y = bv[hostOffset(i, dims, &bstrides)];
                value.* = switch (op) {
                    .Add => x + y,
                    .Sub => x - y,
                    .Mul => x * y,
                    .Div => if (modulo) @mod(x, y) else @divFloor(x, y),
                    .Min => @min(x, y),
                    .Max => @max(x, y),
                    else => return Error.UnsupportedOperator,
                };
            }
            return self.put(arena, values, node.outputs[0], .{ .dtype = if (a.dtype == .i64 or b.dtype == .i64) .i64 else .i32, .dims = dims, .data = .{ .host = std.mem.sliceAsBytes(out) } });
        }
        if (a.dtype != .f32 or b.dtype != .f32) return Error.UnsupportedDataType;

        if (a.onHost() and b.onHost()) {
            const dims = try broadcastShape(arena, a.dims, b.dims);
            if (dims.len > max_rank) return Error.RankTooLarge;
            var astrides: [max_rank]u32 = @splat(0);
            var bstrides: [max_rank]u32 = @splat(0);
            makeBroadcastStrides(a.dims, dims, &astrides);
            makeBroadcastStrides(b.dims, dims, &bstrides);
            const av: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, a.data.host));
            const bv: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, b.data.host));
            const out = try arena.alloc(f32, try elementCount(dims));
            for (out, 0..) |*value, i| {
                const x = av[hostOffset(i, dims, &astrides)];
                const y = bv[hostOffset(i, dims, &bstrides)];
                value.* = switch (op) {
                    .Add => x + y,
                    .Sub => x - y,
                    .Mul => x * y,
                    .Div => x / y,
                    .Min => @min(x, y),
                    .Max => @max(x, y),
                    .Pow => std.math.pow(f32, x, y),
                };
            }
            return self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .host = std.mem.sliceAsBytes(out) } });
        }

        if (a.onHost()) {
            const count = try a.count();
            const storage = try self.newStorage(count);
            errdefer self.releaseStorage(storage);
            try uploadFloats(arena, storage.buffer, @alignCast(std.mem.bytesAsSlice(f32, a.data.host)));
            a.* = .{ .dtype = .f32, .dims = a.dims, .data = .{ .gpu = storage } };
        }
        if (b.onHost()) {
            const count = try b.count();
            const storage = try self.newStorage(count);
            errdefer self.releaseStorage(storage);
            try uploadFloats(arena, storage.buffer, @alignCast(std.mem.bytesAsSlice(f32, b.data.host)));
            b.* = .{ .dtype = .f32, .dims = b.dims, .data = .{ .gpu = storage } };
        }

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
        try self.env.uploadMetadata(metadata[0 .. 3 * rank]);
        // Neither side broadcast means the kernel can index straight through.
        const walk: u32 = if (std.mem.eql(i64, a.dims, dims) and std.mem.eql(i64, b.dims, dims)) 0 else @intCast(rank);
        const ab = try a.gpuBuffer();
        const bb = try b.gpuBuffer();
        const block: u32 = 256;
        const a_period = repeatingPeriod(a.dims, dims) orelse 0;
        const b_period = repeatingPeriod(b.dims, dims) orelse 0;
        // Pow is the one arithmetic operator the vector kernel leaves out: it
        // wants a scalar call per component, and these graphs spend too little
        // time in it for that to pay.
        const arithmetic = op != .Pow;
        if (arithmetic) blk: {
            const kernel = self.env.gpu.binary_vec orelse break :blk;
            const a_groups = groupPeriod(count, a_period) orelse break :blk;
            const b_groups = groupPeriod(count, b_period) orelse break :blk;
            const groups = count / lane_step;
            try kernel.launch(.{ .x = @intCast((groups + block - 1) / block) }, .{ .x = block }, .{
                ab.ptr,
                bb.ptr,
                storage.buffer.ptr,
                @as(u32, @intCast(groups)),
                @intFromEnum(op),
                a_groups,
                b_groups,
            });
            return self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
        }
        try self.env.gpu.binary.launch(.{ .x = @intCast((count + block - 1) / block) }, .{ .x = block }, .{
            ab.ptr,
            bb.ptr,
            storage.buffer.ptr,
            self.env.metadata.ptr,
            walk,
            @as(u32, @intCast(count)),
            @intFromEnum(op),
            @as(u32, @intCast(a_period)),
            @as(u32, @intCast(b_period)),
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
        // A power of two, since the fallback tree reduction halves it, and
        // capped well below the hardware maximum: a row is reduced twice, and a
        // wide block spends more on the reduction than on the row, while a
        // narrow one just loops over more of the row.
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
        try self.env.uploadMetadata(metadata[0 .. 2 * rank + 2 * reduced]);

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
        try self.product(.{
            .a = ab.ptr,
            .b = bb.ptr,
            .c = storage.buffer.ptr,
            .m = @intCast(m),
            .n = @intCast(n),
            .k = @intCast(k),
            .c_batch = @intCast(m * n),
            .b_transposed = transposed,
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
            try self.env.uploadMetadata(metadata[0..6]);
            // The sum lands back where the product is: each thread reads and
            // writes the one element it owns.
            const cb = try c.gpuBuffer();
            const block: u32 = 256;
            try self.env.gpu.binary.launch(.{ .x = @intCast((count + block - 1) / block) }, .{ .x = block }, .{
                storage.buffer.ptr,
                cb.ptr,
                storage.buffer.ptr,
                self.env.metadata.ptr,
                @as(u32, 2),
                @as(u32, @intCast(count)),
                @intFromEnum(Binary.Add),
                @as(u32, @intCast(count)),
                @as(u32, @intCast(repeatingPeriod(c.dims, dims) orelse 0)),
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
        try self.product(.{
            .a = ab.ptr,
            .b = bb.ptr,
            .c = storage.buffer.ptr,
            .m = @intCast(m),
            .n = @intCast(n),
            .k = @intCast(k),
            .a_batch = @intCast(m * @as(usize, @intCast(k))),
            .b_batch = @intCast(@as(usize, @intCast(k)) * n),
            .c_batch = @intCast(m * n),
            .batches = @intCast(batches),
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
        try self.env.uploadMetadata(metadata[0 .. 3 * rank]);

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

        const tensor_kernel = self.env.gpu.matmul_nbits_tensor;
        const tile_m: i64 = if (tensor_kernel != null) matmul_nbits_tensor_tile_m else matmul_tile_m;
        const tile_n: i64 = if (tensor_kernel != null) matmul_nbits_tensor_tile_n else matmul_tile_n;
        const subgroups: u32 = if (tensor_kernel != null) matmul_nbits_tensor_subgroups else 16;
        const kernel = tensor_kernel orelse self.env.gpu.matmul_nbits;
        try kernel.launch(.{
            .x = @intCast(@divFloor(width + tile_n - 1, tile_n)),
            .y = @intCast(@divFloor(@as(i64, @intCast(rows)) + tile_m - 1, tile_m)),
        }, .{ .x = 16, .y = subgroups }, .{
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
        try self.env.uploadMetadata(&meta);
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
                .Equal => x == y,
                .Greater => x > y,
                .GreaterOrEqual => x >= y,
                .Less => x < y,
                .LessOrEqual => x <= y,
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
        try self.env.uploadMetadata(metadata[0 .. 4 * rank]);

        const storage = try self.newStorage(count);
        errdefer self.releaseStorage(storage);
        const block: u32 = 256;
        try self.env.gpu.select.launch(.{ .x = @intCast((count + block - 1) / block) }, .{ .x = block }, .{
            cb.ptr,                ab.ptr,                   bb.ptr,                    storage.buffer.ptr,
            self.env.metadata.ptr, @as(u32, @intCast(rank)), @as(u32, @intCast(count)),
        });
        try self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
    }

    /// Uploads a host integer or boolean tensor as whatever the elementwise
    /// kernels work in. A half build cannot hold an integer past 2048 exactly,
    /// which is fine for the masks and small counts these graphs cast, and is
    /// why index arithmetic stays on the host as i64.
    fn deviceMask(self: *SessionState, arena: std.mem.Allocator, tensor: Tensor) !*Storage {
        const count = try tensor.count();
        const host = try arena.alloc(Element, count);
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
            const out = try arena.alloc(f32, count);
            for (out, 0..) |*value, i| {
                value.* = switch (x.dtype) {
                    .i64 => @floatFromInt((try x.i64s())[i]),
                    .i32 => @floatFromInt(@as([]const i32, @alignCast(std.mem.bytesAsSlice(i32, x.data.host)))[i]),
                    .u8 => @floatFromInt(x.data.host[i]),
                    .i8 => @floatFromInt(@as(i8, @bitCast(x.data.host[i]))),
                    .bool => if (x.data.host[i] != 0) 1.0 else 0.0,
                    .f32 => @as([]const f32, @alignCast(std.mem.bytesAsSlice(f32, x.data.host)))[i],
                    else => return Error.UnsupportedDataType,
                };
            }
            const storage = try self.newStorage(count);
            errdefer self.releaseStorage(storage);
            try uploadFloats(arena, storage.buffer, out);
            return self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
        }

        // Anything but f32 lives on the host already; f32 has to come back
        // down. The copy is stream-ordered, so it sees the finished tensor.
        const downloaded: ?[]const f32 = if (x.onHost()) blk: {
            if (x.dtype == .f32) {
                break :blk @as([]const f32, @alignCast(std.mem.bytesAsSlice(f32, x.data.host)));
            }
            break :blk null;
        } else blk: {
            const host = try arena.alloc(f32, count);
            try downloadFloats(arena, try x.gpuBuffer(), host);
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
            .i32 => {
                const out = try arena.alloc(i32, count);
                for (out, 0..) |*value, i| {
                    value.* = if (downloaded) |f| @intFromFloat(@trunc(f[i])) else switch (x.dtype) {
                        .i64 => @intCast((try x.i64s())[i]),
                        .i32 => @as([]const i32, @alignCast(std.mem.bytesAsSlice(i32, x.data.host)))[i],
                        .u8 => x.data.host[i],
                        .i8 => @as(i8, @bitCast(x.data.host[i])),
                        .bool => x.data.host[i],
                        else => @intCast(try x.element(i)),
                    };
                }
                try self.put(arena, values, node.outputs[0], .{ .dtype = .i32, .dims = dims, .data = .{ .host = std.mem.sliceAsBytes(out) } });
            },
            .u8 => {
                const out = try arena.alloc(u8, count);
                for (out, 0..) |*value, i| {
                    value.* = if (downloaded) |f| @intFromFloat(std.math.clamp(f[i], 0, 255)) else switch (x.dtype) {
                        .i64 => @intCast((try x.i64s())[i]),
                        .i32 => @intCast(@as([]const i32, @alignCast(std.mem.bytesAsSlice(i32, x.data.host)))[i]),
                        .u8 => x.data.host[i],
                        .i8 => x.data.host[i],
                        .bool => x.data.host[i],
                        else => @intCast(try x.element(i)),
                    };
                }
                try self.put(arena, values, node.outputs[0], .{ .dtype = .u8, .dims = dims, .data = .{ .host = out } });
            },
            .i8 => {
                const out = try arena.alloc(u8, count);
                for (out, 0..) |*value, i| {
                    value.* = if (downloaded) |f| @bitCast(@as(i8, @intFromFloat(std.math.clamp(f[i], -128, 127)))) else switch (x.dtype) {
                        .i64 => @bitCast(@as(i8, @intCast((try x.i64s())[i]))),
                        .i32 => @bitCast(@as(i8, @intCast(@as([]const i32, @alignCast(std.mem.bytesAsSlice(i32, x.data.host)))[i]))),
                        .u8 => x.data.host[i],
                        .i8 => x.data.host[i],
                        .bool => x.data.host[i],
                        else => @bitCast(@as(i8, @intCast(try x.element(i)))),
                    };
                }
                try self.put(arena, values, node.outputs[0], .{ .dtype = .i8, .dims = dims, .data = .{ .host = out } });
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

        const storage = try self.newStorage(count);
        errdefer self.releaseStorage(storage);
        const xb = try x.gpuBuffer();
        try self.strided(xb.ptr, storage.buffer.ptr, metadata[0 .. 2 * rank], count, 0, 0);
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
        try self.env.uploadMetadata(metadata[0 .. 3 * rank]);

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
        const inp0 = try self.input(arena, values, node.inputs[0]);
        const inp1 = try self.input(arena, values, node.inputs[1]);
        const inp2 = try self.input(arena, values, node.inputs[2]);

        if (inp0.dtype == .f32 or inp1.dtype == .f32 or inp2.dtype == .f32) {
            const start: f32 = if (inp0.onHost()) @as([]const f32, @alignCast(std.mem.bytesAsSlice(f32, inp0.data.host)))[0] else blk: {
                var f: [1]f32 = undefined;
                try downloadFloats(arena, try inp0.gpuBuffer(), &f);
                break :blk f[0];
            };
            const limit: f32 = if (inp1.onHost()) @as([]const f32, @alignCast(std.mem.bytesAsSlice(f32, inp1.data.host)))[0] else blk: {
                var f: [1]f32 = undefined;
                try downloadFloats(arena, try inp1.gpuBuffer(), &f);
                break :blk f[0];
            };
            const delta: f32 = if (inp2.onHost()) @as([]const f32, @alignCast(std.mem.bytesAsSlice(f32, inp2.data.host)))[0] else blk: {
                var f: [1]f32 = undefined;
                try downloadFloats(arena, try inp2.gpuBuffer(), &f);
                break :blk f[0];
            };
            if (delta == 0) return Error.InvalidShape;

            var count: usize = 0;
            if (delta > 0 and limit > start) count = @intFromFloat(@ceil((limit - start) / delta));
            if (delta < 0 and limit < start) count = @intFromFloat(@ceil((start - limit) / -delta));

            const out = try arena.alloc(f32, count);
            for (out, 0..) |*value, i| value.* = start + @as(f32, @floatFromInt(i)) * delta;
            const dims = try arena.dupe(i64, &.{@as(i64, @intCast(count))});
            try self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .host = std.mem.sliceAsBytes(out) } });
            return;
        }

        const start = try inp0.element(0);
        const limit = try inp1.element(0);
        const delta = try inp2.element(0);
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

        const storage = try self.newStorage(count);
        errdefer self.releaseStorage(storage);
        const db = try data.gpuBuffer();
        try self.strided(db.ptr, storage.buffer.ptr, metadata[0 .. 2 * rank], count, 0, 0);

        try self.ensureIndices(offsets.len);
        try self.env.indices.upload(offsets);
        const ub = try updates.gpuBuffer();
        const written = tuples * span;
        const block: u32 = 256;
        try self.env.gpu.scatter.launch(.{ .x = @intCast((written + block - 1) / block) }, .{ .x = block }, .{
            ub.ptr, self.env.indices.ptr, storage.buffer.ptr, @as(u32, @intCast(span)), @as(u32, @intCast(written)),
        });
        try self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
    }

    fn dynamicQuantizeLinear(
        self: *SessionState,
        arena: std.mem.Allocator,
        values: *std.StringHashMapUnmanaged(*Tensor),
        node: onnx.Node,
    ) !void {
        const x = try self.input(arena, values, node.inputs[0]);
        const count = try x.count();
        const dims = try arena.dupe(i64, x.dims);

        var x_f32: []const f32 = undefined;
        if (x.onHost()) {
            if (x.dtype == .f32) {
                x_f32 = @alignCast(std.mem.bytesAsSlice(f32, x.data.host));
            } else {
                return Error.UnsupportedDataType;
            }
        } else {
            const host = try arena.alloc(f32, count);
            try downloadFloats(arena, try x.gpuBuffer(), host);
            x_f32 = host;
        }

        var min_val: f32 = 0.0;
        var max_val: f32 = 0.0;
        if (x_f32.len > 0) {
            for (x_f32) |v| {
                if (v < min_val) min_val = v;
                if (v > max_val) max_val = v;
            }
        }

        const qmin: f32 = 0.0;
        const qmax: f32 = 255.0;
        const scale: f32 = if (max_val == min_val) 1.0 else (max_val - min_val) / (qmax - qmin);
        const zero_point: u8 = @intFromFloat(std.math.clamp(std.math.round(-min_val / scale), qmin, qmax));

        const y = try arena.alloc(u8, count);
        const inv_scale = 1.0 / scale;
        const zp_f32 = @as(f32, @floatFromInt(zero_point));
        for (x_f32, y) |v, *out| {
            const q = std.math.round(v * inv_scale) + zp_f32;
            out.* = @intFromFloat(std.math.clamp(q, qmin, qmax));
        }

        try self.put(arena, values, node.outputs[0], .{
            .dtype = .u8,
            .dims = dims,
            .data = .{ .host = y },
        });

        if (node.outputs.len > 1 and node.outputs[1].len > 0) {
            const scale_buf = try arena.alloc(f32, 1);
            scale_buf[0] = scale;
            const scale_dims = try arena.dupe(i64, &.{});
            try self.put(arena, values, node.outputs[1], .{
                .dtype = .f32,
                .dims = scale_dims,
                .data = .{ .host = std.mem.sliceAsBytes(scale_buf) },
            });
        }

        if (node.outputs.len > 2 and node.outputs[2].len > 0) {
            const zp_buf = try arena.alloc(u8, 1);
            zp_buf[0] = zero_point;
            const zp_dims = try arena.dupe(i64, &.{});
            try self.put(arena, values, node.outputs[2], .{
                .dtype = .u8,
                .dims = zp_dims,
                .data = .{ .host = zp_buf },
            });
        }
    }

    fn matmulInteger(
        self: *SessionState,
        arena: std.mem.Allocator,
        values: *std.StringHashMapUnmanaged(*Tensor),
        node: onnx.Node,
    ) !void {
        const a = try self.input(arena, values, node.inputs[0]);
        const b = try self.input(arena, values, node.inputs[1]);

        var a_zp: i32 = 0;
        if (node.inputs.len > 2 and node.inputs[2].len != 0) {
            const a_zp_tensor = try self.input(arena, values, node.inputs[2]);
            if (a_zp_tensor.data.host.len > 0) {
                a_zp = switch (a_zp_tensor.dtype) {
                    .u8 => a_zp_tensor.data.host[0],
                    .i8 => @as(i8, @bitCast(a_zp_tensor.data.host[0])),
                    .i32 => @as([]const i32, @alignCast(std.mem.bytesAsSlice(i32, a_zp_tensor.data.host)))[0],
                    else => 0,
                };
            }
        }

        var b_zp: i32 = 0;
        if (node.inputs.len > 3 and node.inputs[3].len != 0) {
            const b_zp_tensor = try self.input(arena, values, node.inputs[3]);
            if (b_zp_tensor.data.host.len > 0) {
                b_zp = switch (b_zp_tensor.dtype) {
                    .u8 => b_zp_tensor.data.host[0],
                    .i8 => @as(i8, @bitCast(b_zp_tensor.data.host[0])),
                    .i32 => @as([]const i32, @alignCast(std.mem.bytesAsSlice(i32, b_zp_tensor.data.host)))[0],
                    else => 0,
                };
            }
        }

        const rank_a = a.dims.len;
        const rank_b = b.dims.len;
        if (rank_a < 2 or rank_b < 2) return Error.UnsupportedOperator;

        const m = a.dims[rank_a - 2];
        const k = a.dims[rank_a - 1];
        const bk = b.dims[rank_b - 2];
        const n = b.dims[rank_b - 1];
        if (k != bk) return Error.InvalidShape;

        const batch_dims = try broadcastShape(arena, a.dims[0 .. rank_a - 2], b.dims[0 .. rank_b - 2]);
        const dims = try arena.alloc(i64, batch_dims.len + 2);
        @memcpy(dims[0..batch_dims.len], batch_dims);
        dims[dims.len - 2] = m;
        dims[dims.len - 1] = n;

        const batches = try elementCount(batch_dims);
        const a_batches = try elementCount(a.dims[0 .. rank_a - 2]);
        const b_batches = try elementCount(b.dims[0 .. rank_b - 2]);

        const out_count = try elementCount(dims);
        const out = try arena.alloc(i32, out_count);

        const a_bytes = a.data.host;
        const b_bytes = b.data.host;

        const m_usize: usize = @intCast(m);
        const n_usize: usize = @intCast(n);
        const k_usize: usize = @intCast(k);

        for (0..batches) |batch_idx| {
            const a_batch_idx = if (a_batches == 1) 0 else batch_idx;
            const b_batch_idx = if (b_batches == 1) 0 else batch_idx;

            const a_slice = a_bytes[a_batch_idx * m_usize * k_usize .. (a_batch_idx + 1) * m_usize * k_usize];
            const b_slice = b_bytes[b_batch_idx * k_usize * n_usize .. (b_batch_idx + 1) * k_usize * n_usize];
            const c_slice = out[batch_idx * m_usize * n_usize .. (batch_idx + 1) * m_usize * n_usize];

            for (0..m_usize) |row| {
                const a_row = a_slice[row * k_usize .. (row + 1) * k_usize];
                const c_row = c_slice[row * n_usize .. (row + 1) * n_usize];

                for (0..n_usize) |col| {
                    var sum: i32 = 0;
                    for (0..k_usize) |depth| {
                        const a_val: i32 = @as(i32, a_row[depth]) - a_zp;
                        const b_val: i32 = @as(i32, b_slice[depth * n_usize + col]) - b_zp;
                        sum += a_val * b_val;
                    }
                    c_row[col] = sum;
                }
            }
        }

        try self.put(arena, values, node.outputs[0], .{
            .dtype = .i32,
            .dims = dims,
            .data = .{ .host = std.mem.sliceAsBytes(out) },
        });
    }

    fn dequantizeLinear(
        self: *SessionState,
        arena: std.mem.Allocator,
        values: *std.StringHashMapUnmanaged(*Tensor),
        node: onnx.Node,
    ) !void {
        const x = try self.input(arena, values, node.inputs[0]);
        const scale_tensor = try self.input(arena, values, node.inputs[1]);
        const scale: f32 = @as([]const f32, @alignCast(std.mem.bytesAsSlice(f32, scale_tensor.data.host)))[0];

        var zp: f32 = 0.0;
        if (node.inputs.len > 2 and node.inputs[2].len != 0) {
            const zp_tensor = try self.input(arena, values, node.inputs[2]);
            if (zp_tensor.data.host.len > 0) {
                zp = switch (zp_tensor.dtype) {
                    .u8 => @floatFromInt(zp_tensor.data.host[0]),
                    .i8 => @floatFromInt(@as(i8, @bitCast(zp_tensor.data.host[0]))),
                    .i32 => @floatFromInt(@as([]const i32, @alignCast(std.mem.bytesAsSlice(i32, zp_tensor.data.host)))[0]),
                    .f32 => @as([]const f32, @alignCast(std.mem.bytesAsSlice(f32, zp_tensor.data.host)))[0],
                    else => 0.0,
                };
            }
        }

        const count = try x.count();
        const dims = try arena.dupe(i64, x.dims);
        const out = try arena.alloc(f32, count);

        for (out, 0..) |*val, i| {
            const x_val: f32 = switch (x.dtype) {
                .u8 => @floatFromInt(x.data.host[i]),
                .i8 => @floatFromInt(@as(i8, @bitCast(x.data.host[i]))),
                .i32 => @floatFromInt(@as([]const i32, @alignCast(std.mem.bytesAsSlice(i32, x.data.host)))[i]),
                .f32 => @as([]const f32, @alignCast(std.mem.bytesAsSlice(f32, x.data.host)))[i],
                else => return Error.UnsupportedDataType,
            };
            val.* = (x_val - zp) * scale;
        }

        const storage = try self.newStorage(count);
        errdefer self.releaseStorage(storage);
        try uploadFloats(arena, storage.buffer, out);
        try self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
    }

    fn quantizeLinear(
        self: *SessionState,
        arena: std.mem.Allocator,
        values: *std.StringHashMapUnmanaged(*Tensor),
        node: onnx.Node,
    ) !void {
        const x = try self.input(arena, values, node.inputs[0]);
        const scale_tensor = try self.input(arena, values, node.inputs[1]);
        const scale: f32 = @as([]const f32, @alignCast(std.mem.bytesAsSlice(f32, scale_tensor.data.host)))[0];

        var zp: f32 = 0.0;
        if (node.inputs.len > 2 and node.inputs[2].len != 0) {
            const zp_tensor = try self.input(arena, values, node.inputs[2]);
            if (zp_tensor.data.host.len > 0) {
                zp = switch (zp_tensor.dtype) {
                    .u8 => @floatFromInt(zp_tensor.data.host[0]),
                    .i8 => @floatFromInt(@as(i8, @bitCast(zp_tensor.data.host[0]))),
                    .i32 => @floatFromInt(@as([]const i32, @alignCast(std.mem.bytesAsSlice(i32, zp_tensor.data.host)))[0]),
                    .f32 => @as([]const f32, @alignCast(std.mem.bytesAsSlice(f32, zp_tensor.data.host)))[0],
                    else => 0.0,
                };
            }
        }

        const count = try x.count();
        const dims = try arena.dupe(i64, x.dims);

        var x_f32: []const f32 = undefined;
        if (x.onHost()) {
            if (x.dtype == .f32) {
                x_f32 = @alignCast(std.mem.bytesAsSlice(f32, x.data.host));
            } else {
                return Error.UnsupportedDataType;
            }
        } else {
            const host = try arena.alloc(f32, count);
            try downloadFloats(arena, try x.gpuBuffer(), host);
            x_f32 = host;
        }

        const y = try arena.alloc(u8, count);
        const inv_scale = 1.0 / scale;
        for (x_f32, y) |v, *out| {
            const q = std.math.round(v * inv_scale) + zp;
            out.* = @intFromFloat(std.math.clamp(q, 0.0, 255.0));
        }

        try self.put(arena, values, node.outputs[0], .{
            .dtype = .u8,
            .dims = dims,
            .data = .{ .host = y },
        });
    }

    fn reduceMean(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node) !void {
        const x = try self.input(arena, values, node.inputs[0]);
        if (x.dims.len > max_rank) return Error.RankTooLarge;
        const axes_raw = if (node.inputs.len > 1 and node.inputs[1].len != 0) blk: {
            const inp = try self.input(arena, values, node.inputs[1]);
            const inp_count = try inp.count();
            const ax = try arena.alloc(i64, inp_count);
            for (0..inp_count) |i| {
                ax[i] = if (inp.dtype == .i64) (try inp.i64s())[i] else @as([]const i32, @alignCast(std.mem.bytesAsSlice(i32, inp.data.host)))[i];
            }
            break :blk ax;
        } else node.ints("axes");

        const axes = if (axes_raw.len == 0) blk: {
            if (node.int("noop_with_empty_axes", 0) == 1) {
                break :blk @as([]const i64, &.{});
            } else {
                const all_axes = try arena.alloc(i64, x.dims.len);
                for (all_axes, 0..) |*a, i| a.* = @intCast(i);
                break :blk all_axes;
            }
        } else axes_raw;

        if (axes.len == 0) {
            const dims = try arena.dupe(i64, x.dims);
            return self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = x.data });
        }

        var strides: [max_rank]u32 = @splat(0);
        denseStrides(x.dims, &strides);

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

        const rank = x.dims.len;
        var metadata: [4 * max_rank]u32 = @splat(0);
        for (kept, 0..) |dim, i| metadata[i] = @intCast(dim);
        @memcpy(metadata[rank..][0..rank], strides[0..rank]);
        @memcpy(metadata[2 * rank ..][0..reduced], swept_dims[0..reduced]);
        @memcpy(metadata[2 * rank + reduced ..][0..reduced], swept_strides[0..reduced]);
        try self.env.uploadMetadata(metadata[0 .. 2 * rank + 2 * reduced]);

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

        const inv_swept: f32 = 1.0 / @as(f32, @floatFromInt(swept));
        const factor_storage = try self.newStorage(1);
        defer self.releaseStorage(factor_storage);
        try uploadFloats(arena, factor_storage.buffer, &.{inv_swept});

        var mul_meta: [3 * max_rank]u32 = @splat(0);
        for (kept, 0..) |dim, i| mul_meta[i] = @intCast(dim);
        var kept_strides: [max_rank]u32 = @splat(0);
        denseStrides(kept, &kept_strides);
        @memcpy(mul_meta[rank .. 2 * rank][0..rank], kept_strides[0..rank]);
        try self.env.uploadMetadata(mul_meta[0 .. 3 * rank]);

        try self.env.gpu.binary.launch(.{ .x = @intCast((count + block - 1) / block) }, .{ .x = block }, .{
            storage.buffer.ptr,
            factor_storage.buffer.ptr,
            storage.buffer.ptr,
            self.env.metadata.ptr,
            @as(u32, @intCast(rank)),
            @as(u32, @intCast(count)),
            @intFromEnum(Binary.Mul),
            @as(u32, @intCast(count)),
            @as(u32, 0),
        });

        var dims = kept;
        if (node.int("keepdims", 1) == 0) {
            var shrunk: std.ArrayList(i64) = .empty;
            for (kept, 0..) |dim, axis| if (!collapsed[axis]) try shrunk.append(arena, dim);
            dims = shrunk.items;
        }
        try self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
    }

    fn reduceMax(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node) !void {
        const x = try self.input(arena, values, node.inputs[0]);
        const axes_raw = if (node.inputs.len > 1 and node.inputs[1].len != 0) blk: {
            const inp = try self.input(arena, values, node.inputs[1]);
            const inp_count = try inp.count();
            const ax = try arena.alloc(i64, inp_count);
            for (0..inp_count) |i| {
                ax[i] = if (inp.dtype == .i64) (try inp.i64s())[i] else @as([]const i32, @alignCast(std.mem.bytesAsSlice(i32, inp.data.host)))[i];
            }
            break :blk ax;
        } else node.ints("axes");

        const axes = if (axes_raw.len == 0) blk: {
            if (node.int("noop_with_empty_axes", 0) == 1) {
                break :blk @as([]const i64, &.{});
            } else {
                const all_axes = try arena.alloc(i64, x.dims.len);
                for (all_axes, 0..) |*a, i| a.* = @intCast(i);
                break :blk all_axes;
            }
        } else axes_raw;

        if (axes.len == 0) {
            const dims = try arena.dupe(i64, x.dims);
            return self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = x.data });
        }

        const count = try x.count();
        const rank = x.dims.len;
        const kept = try arena.dupe(i64, x.dims);
        var collapsed: [max_rank]bool = @splat(false);
        for (axes) |raw| {
            const axis = normalizeAxis(raw, rank);
            collapsed[axis] = true;
            kept[axis] = 1;
        }

        var strides: [max_rank]u32 = @splat(0);
        denseStrides(x.dims, &strides);

        const out_count = try elementCount(kept);

        if (x.dtype == .f32) {
            const out = try arena.alloc(f32, out_count);
            @memset(out, -std.math.inf(f32));

            var x_f32: []const f32 = undefined;
            if (x.onHost()) {
                x_f32 = @alignCast(std.mem.bytesAsSlice(f32, x.data.host));
            } else {
                const host = try arena.alloc(f32, count);
                try downloadFloats(arena, try x.gpuBuffer(), host);
                x_f32 = host;
            }

            for (x_f32, 0..) |val, i| {
                var out_idx: usize = 0;
                var rem = i;
                for (0..rank) |d| {
                    const coord = rem / strides[d];
                    rem %= strides[d];
                    if (!collapsed[d]) {
                        out_idx = out_idx * @as(usize, @intCast(x.dims[d])) + coord;
                    }
                }
                if (val > out[out_idx]) out[out_idx] = val;
            }

            var dims = kept;
            if (node.int("keepdims", 1) == 0) {
                var shrunk: std.ArrayList(i64) = .empty;
                for (kept, 0..) |dim, axis| if (!collapsed[axis]) try shrunk.append(arena, dim);
                dims = shrunk.items;
            }

            const storage = try self.newStorage(out_count);
            errdefer self.releaseStorage(storage);
            try uploadFloats(arena, storage.buffer, out);
            try self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = dims, .data = .{ .gpu = storage } });
        } else if (x.dtype == .i64 or x.dtype == .i32) {
            const out = try arena.alloc(i64, out_count);
            @memset(out, std.math.minInt(i64));

            const av = try arena.alloc(i64, count);
            for (0..count) |i| av[i] = if (x.dtype == .i64) (try x.i64s())[i] else @as([]const i32, @alignCast(std.mem.bytesAsSlice(i32, x.data.host)))[i];

            for (av, 0..) |val, i| {
                var out_idx: usize = 0;
                var rem = i;
                for (0..rank) |d| {
                    const coord = rem / strides[d];
                    rem %= strides[d];
                    if (!collapsed[d]) {
                        out_idx = out_idx * @as(usize, @intCast(x.dims[d])) + coord;
                    }
                }
                if (val > out[out_idx]) out[out_idx] = val;
            }

            var dims = kept;
            if (node.int("keepdims", 1) == 0) {
                var shrunk: std.ArrayList(i64) = .empty;
                for (kept, 0..) |dim, axis| if (!collapsed[axis]) try shrunk.append(arena, dim);
                dims = shrunk.items;
            }

            if (x.dtype == .i64) {
                try self.put(arena, values, node.outputs[0], .{ .dtype = .i64, .dims = dims, .data = .{ .host = std.mem.sliceAsBytes(out) } });
            } else {
                const out_i32 = try arena.alloc(i32, out_count);
                for (out, out_i32) |src, *dst| dst.* = @intCast(src);
                try self.put(arena, values, node.outputs[0], .{ .dtype = .i32, .dims = dims, .data = .{ .host = std.mem.sliceAsBytes(out_i32) } });
            }
        } else if (x.dtype == .u8) {
            const out = try arena.alloc(u8, out_count);
            @memset(out, 0);

            for (x.data.host, 0..) |val, i| {
                var out_idx: usize = 0;
                var rem = i;
                for (0..rank) |d| {
                    const coord = rem / strides[d];
                    rem %= strides[d];
                    if (!collapsed[d]) {
                        out_idx = out_idx * @as(usize, @intCast(x.dims[d])) + coord;
                    }
                }
                if (val > out[out_idx]) out[out_idx] = val;
            }

            var dims = kept;
            if (node.int("keepdims", 1) == 0) {
                var shrunk: std.ArrayList(i64) = .empty;
                for (kept, 0..) |dim, axis| if (!collapsed[axis]) try shrunk.append(arena, dim);
                dims = shrunk.items;
            }

            try self.put(arena, values, node.outputs[0], .{ .dtype = .u8, .dims = dims, .data = .{ .host = out } });
        } else {
            return Error.UnsupportedDataType;
        }
    }

    fn lstm(self: *SessionState, arena: std.mem.Allocator, values: *std.StringHashMapUnmanaged(*Tensor), node: onnx.Node) !void {
        const x = try self.input(arena, values, node.inputs[0]);
        const w = try self.input(arena, values, node.inputs[1]);
        const r = try self.input(arena, values, node.inputs[2]);

        if (x.dims.len != 3 or w.dims.len != 3 or r.dims.len != 3) return Error.InvalidShape;

        const seq_len: usize = @intCast(x.dims[0]);
        const batch_size: usize = @intCast(x.dims[1]);
        const input_size: usize = @intCast(x.dims[2]);
        const num_directions: usize = @intCast(w.dims[0]);
        const hidden_size: usize = @intCast(r.dims[2]);

        const bias = if (node.inputs.len > 3 and node.inputs[3].len != 0)
            try self.input(arena, values, node.inputs[3])
        else
            null;

        const initial_h = if (node.inputs.len > 5 and node.inputs[5].len != 0)
            try self.input(arena, values, node.inputs[5])
        else
            null;

        const initial_c = if (node.inputs.len > 6 and node.inputs[6].len != 0)
            try self.input(arena, values, node.inputs[6])
        else
            null;

        const x_data = try arena.alloc(f32, try x.count());
        try downloadFloats(arena, try x.gpuBuffer(), x_data);

        const w_data = try arena.alloc(f32, try w.count());
        try downloadFloats(arena, try w.gpuBuffer(), w_data);

        const r_data = try arena.alloc(f32, try r.count());
        try downloadFloats(arena, try r.gpuBuffer(), r_data);

        const b_data = if (bias) |b| blk: {
            const buf = try arena.alloc(f32, try b.count());
            try downloadFloats(arena, try b.gpuBuffer(), buf);
            break :blk buf;
        } else null;

        const h_init_data = if (initial_h) |h0| blk: {
            const buf = try arena.alloc(f32, try h0.count());
            try downloadFloats(arena, try h0.gpuBuffer(), buf);
            break :blk buf;
        } else null;

        const c_init_data = if (initial_c) |c0| blk: {
            const buf = try arena.alloc(f32, try c0.count());
            try downloadFloats(arena, try c0.gpuBuffer(), buf);
            break :blk buf;
        } else null;

        const y_count = seq_len * num_directions * batch_size * hidden_size;
        const y_data = try arena.alloc(f32, y_count);

        const yh_count = num_directions * batch_size * hidden_size;
        const yh_data = try arena.alloc(f32, yh_count);

        const yc_count = num_directions * batch_size * hidden_size;
        const yc_data = try arena.alloc(f32, yc_count);

        const h_prev = try arena.alloc(f32, num_directions * batch_size * hidden_size);
        const c_prev = try arena.alloc(f32, num_directions * batch_size * hidden_size);

        if (h_init_data) |h0| {
            @memcpy(h_prev, h0);
        } else {
            @memset(h_prev, 0);
        }

        if (c_init_data) |c0| {
            @memcpy(c_prev, c0);
        } else {
            @memset(c_prev, 0);
        }

        const gates = try arena.alloc(f32, 4 * hidden_size);

        for (0..num_directions) |d| {
            const w_dir = w_data[d * 4 * hidden_size * input_size .. (d + 1) * 4 * hidden_size * input_size];
            const r_dir = r_data[d * 4 * hidden_size * hidden_size .. (d + 1) * 4 * hidden_size * hidden_size];
            const wb_dir = if (b_data) |b| b[d * 8 * hidden_size .. d * 8 * hidden_size + 4 * hidden_size] else null;
            const rb_dir = if (b_data) |b| b[d * 8 * hidden_size + 4 * hidden_size .. (d + 1) * 8 * hidden_size] else null;

            for (0..seq_len) |t_step| {
                const t = if (d == 0) t_step else seq_len - 1 - t_step;

                for (0..batch_size) |b_idx| {
                    const x_t = x_data[(t * batch_size + b_idx) * input_size .. (t * batch_size + b_idx + 1) * input_size];
                    const h_t_prev = h_prev[(d * batch_size + b_idx) * hidden_size .. (d * batch_size + b_idx + 1) * hidden_size];
                    const c_t_prev = c_prev[(d * batch_size + b_idx) * hidden_size .. (d * batch_size + b_idx + 1) * hidden_size];

                    for (0..4 * hidden_size) |g| {
                        var val: f32 = 0;
                        const w_row = w_dir[g * input_size .. (g + 1) * input_size];
                        for (x_t, w_row) |xi, wi| val += xi * wi;

                        const r_row = r_dir[g * hidden_size .. (g + 1) * hidden_size];
                        for (h_t_prev, r_row) |hi, ri| val += hi * ri;

                        if (wb_dir) |wb| val += wb[g];
                        if (rb_dir) |rb| val += rb[g];

                        gates[g] = val;
                    }

                    const y_out = y_data[((t * num_directions + d) * batch_size + b_idx) * hidden_size .. ((t * num_directions + d) * batch_size + b_idx + 1) * hidden_size];

                    for (0..hidden_size) |h| {
                        const i_gate = 1.0 / (1.0 + @exp(-gates[h]));
                        const o_gate = 1.0 / (1.0 + @exp(-gates[hidden_size + h]));
                        const f_gate = 1.0 / (1.0 + @exp(-gates[2 * hidden_size + h]));
                        const c_gate = std.math.tanh(gates[3 * hidden_size + h]);

                        const c_new = f_gate * c_t_prev[h] + i_gate * c_gate;
                        const h_new = o_gate * std.math.tanh(c_new);

                        c_t_prev[h] = c_new;
                        h_t_prev[h] = h_new;
                        y_out[h] = h_new;
                    }
                }
            }
        }

        @memcpy(yh_data, h_prev);
        @memcpy(yc_data, c_prev);

        if (node.outputs.len > 0 and node.outputs[0].len > 0) {
            const y_dims = try arena.dupe(i64, &.{ @intCast(seq_len), @intCast(num_directions), @intCast(batch_size), @intCast(hidden_size) });
            const y_storage = try self.newStorage(y_count);
            errdefer self.releaseStorage(y_storage);
            try uploadFloats(arena, y_storage.buffer, y_data);
            try self.put(arena, values, node.outputs[0], .{ .dtype = .f32, .dims = y_dims, .data = .{ .gpu = y_storage } });
        }

        if (node.outputs.len > 1 and node.outputs[1].len > 0) {
            const yh_dims = try arena.dupe(i64, &.{ @intCast(num_directions), @intCast(batch_size), @intCast(hidden_size) });
            const yh_storage = try self.newStorage(yh_count);
            errdefer self.releaseStorage(yh_storage);
            try uploadFloats(arena, yh_storage.buffer, yh_data);
            try self.put(arena, values, node.outputs[1], .{ .dtype = .f32, .dims = yh_dims, .data = .{ .gpu = yh_storage } });
        }

        if (node.outputs.len > 2 and node.outputs[2].len > 0) {
            const yc_dims = try arena.dupe(i64, &.{ @intCast(num_directions), @intCast(batch_size), @intCast(hidden_size) });
            const yc_storage = try self.newStorage(yc_count);
            errdefer self.releaseStorage(yc_storage);
            try uploadFloats(arena, yc_storage.buffer, yc_data);
            try self.put(arena, values, node.outputs[2], .{ .dtype = .f32, .dims = yc_dims, .data = .{ .gpu = yc_storage } });
        }
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
fn denseStrides(dims: []const i64, result: []u32) void {
    std.debug.assert(result.len >= dims.len);
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
        .f32 => {
            const source: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, tensor.data.host));
            const out = try arena.alloc(f32, sources.len);
            for (out, sources) |*value, index| value.* = source[index];
            return std.mem.sliceAsBytes(out);
        },
        .i32 => {
            const source: []const i32 = @alignCast(std.mem.bytesAsSlice(i32, tensor.data.host));
            const out = try arena.alloc(i32, sources.len);
            for (out, sources) |*value, index| value.* = source[index];
            return std.mem.sliceAsBytes(out);
        },
        .u8, .i8, .bool => {
            const source = tensor.data.host;
            const out = try arena.alloc(u8, sources.len);
            for (out, sources) |*value, index| value.* = source[index];
            return out;
        },
        else => return Error.UnsupportedDataType,
    }
}

/// Rewrites a copy's dims and strides -- laid out as `rank` of each, in one
/// array -- into the fewest axes that describe the same walk, and returns the
/// shorter slice. Every axis is a division and a modulo per element, and the
/// shapes these graphs transpose carry axes that earn neither: an extent of
/// one contributes nothing, and two adjacent axes that run contiguously
/// through the source are one axis with the extents multiplied. A window
/// partition's six drop to four this way.
fn collapseAxes(given: []u32) []u32 {
    const rank = given.len / 2;
    var dims: [max_rank]u32 = undefined;
    var strides: [max_rank]u32 = undefined;
    var kept: usize = 0;
    for (0..rank) |axis| {
        const dim = given[axis];
        const stride = given[rank + axis];
        if (dim == 1) continue;
        // Wrapping, because Slice spells a reversed axis as a negative stride
        // and the products of two of those still line up.
        if (kept != 0 and strides[kept - 1] == stride *% dim) {
            dims[kept - 1] *= dim;
            strides[kept - 1] = stride;
            continue;
        }
        dims[kept] = dim;
        strides[kept] = stride;
        kept += 1;
    }
    if (kept == 0) {
        dims[0] = 1;
        strides[0] = 0;
        kept = 1;
    }
    @memcpy(given[0..kept], dims[0..kept]);
    @memcpy(given[kept..][0..kept], strides[0..kept]);
    return given[0 .. 2 * kept];
}

/// Host floats on their way to the device, in whatever a tensor is stored as.
/// A float build hands the slice straight to the driver; a half build narrows
/// it through a scratch buffer first.
fn uploadFloats(allocator: std.mem.Allocator, buffer: driver.Buffer(Element), source: []const f32) !void {
    if (Element == f32) return buffer.upload(source);
    const scratch = try allocator.alloc(Element, source.len);
    defer allocator.free(scratch);
    for (scratch, source) |*narrow, value| narrow.* = @floatCast(value);
    try buffer.upload(scratch);
}

/// The same in reverse, for the graph outputs and the handful of operators
/// that finish their work on the host.
fn downloadFloats(allocator: std.mem.Allocator, buffer: driver.Buffer(Element), out: []f32) !void {
    if (Element == f32) return buffer.download(out);
    const scratch = try allocator.alloc(Element, out.len);
    defer allocator.free(scratch);
    try buffer.download(scratch);
    for (out, scratch) |*value, narrow| value.* = @floatCast(narrow);
}

/// `period` counted in groups of `lane_step`, for the vector kernels. Null
/// whenever a work item that owns one group of outputs would need values from
/// two turns of the repeat, or from a broadcast the kernel does not index. One
/// value repeated over everything is spelled zero, which a group cannot
/// express.
fn groupPeriod(count: usize, period: usize) ?u32 {
    if (period == 0 or count % lane_step != 0) return null;
    if (period == 1) return 0;
    if (period % lane_step != 0) return null;
    return @intCast(period / lane_step);
}

/// How many elements of `input` go by before it starts over, when it is
/// broadcast to `output` by simply repeating: a bias over the last axis, a
/// window every batch shares, a single value. Null for any other broadcast,
/// which then has to be walked stride by stride. Almost every broadcast in
/// these graphs is of this shape, and it is the difference between one wrap
/// and a division per axis for every element.
fn repeatingPeriod(input: []const i64, output: []const i64) ?usize {
    if (input.len > output.len) return null;
    const leading = output.len - input.len;
    var period: usize = 1;
    var repeating = true;
    for (input, 0..) |dim, axis| {
        if (dim < 0) return null;
        if (repeating and dim == 1 and output[leading + axis] != 1) continue;
        repeating = false;
        if (dim != output[leading + axis]) return null;
        period *= @intCast(dim);
    }
    return if (period == 0) null else period;
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
            if (self.owned_f32) |data| {
                allocator.free(data);
            } else if (self.owned_i64) |data| {
                allocator.free(data);
            } else if (self.bytes.len != 0) {
                allocator.free(self.bytes);
            }
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
                try downloadFloats(allocator, try tensor.gpuBuffer(), data);
                return .{ .dtype = .f32, .bytes = std.mem.sliceAsBytes(data), .dims = dims, .owned_allocator = allocator, .owned_f32 = data };
            },
            .i64 => {
                const source = try tensor.i64s();
                const data = try allocator.dupe(i64, source);
                return .{ .dtype = .i64, .bytes = std.mem.sliceAsBytes(data), .dims = dims, .owned_allocator = allocator, .owned_i64 = data };
            },
            .i32, .u8, .i8, .bool => {
                const data = try allocator.dupe(u8, tensor.data.host);
                return .{ .dtype = tensor.dtype, .bytes = data, .dims = dims, .owned_allocator = allocator };
            },
            else => return Error.UnsupportedDataType,
        }
    }
};

test {
    std.testing.refAllDecls(@This());
}

test "INT8 DynamicQuantizeLinear and MatMulInteger math" {
    const testing = std.testing;

    // Test DynamicQuantizeLinear calculation
    const x = [_]f32{ -1.0, 0.0, 1.0, 2.0 };
    var min_val: f32 = 0.0;
    var max_val: f32 = 0.0;
    for (x) |v| {
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }
    const scale = (max_val - min_val) / 255.0;
    const zero_point: u8 = @intFromFloat(std.math.clamp(std.math.round(-min_val / scale), 0.0, 255.0));

    try testing.expect(scale > 0.0);
    try testing.expectEqual(@as(u8, 85), zero_point);

    // Test 2x2 integer matrix product
    const a = [_]u8{ 1, 2, 3, 4 };
    const b = [_]u8{ 5, 6, 7, 8 };
    var c = [_]i32{ 0, 0, 0, 0 };

    for (0..2) |row| {
        for (0..2) |col| {
            var sum: i32 = 0;
            for (0..2) |k| {
                sum += @as(i32, a[row * 2 + k]) * @as(i32, b[k * 2 + col]);
            }
            c[row * 2 + col] = sum;
        }
    }

    try testing.expectEqual(@as(i32, 19), c[0]);
    try testing.expectEqual(@as(i32, 22), c[1]);
    try testing.expectEqual(@as(i32, 43), c[2]);
    try testing.expectEqual(@as(i32, 50), c[3]);
}
