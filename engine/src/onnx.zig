//! Reads an ONNX file -- the protobuf and its external weight blob -- without
//! any ONNX library. The format is only a container here: what comes back is a
//! plain list of nodes and tensors for the executor to walk.

const std = @import("std");

pub const DataType = enum(u32) {
    undefined = 0,
    f32 = 1,
    u8 = 2,
    i8 = 3,
    u16 = 4,
    i16 = 5,
    i32 = 6,
    i64 = 7,
    string = 8,
    bool = 9,
    f16 = 10,
    f64 = 11,
    u32 = 12,
    u64 = 13,
    _,

    pub fn size(self: DataType) usize {
        return switch (self) {
            .f32, .i32, .u32 => 4,
            .u8, .i8, .bool => 1,
            .u16, .i16, .f16 => 2,
            .i64, .f64, .u64 => 8,
            else => 0,
        };
    }
};

pub const Tensor = struct {
    name: []const u8 = "",
    dtype: DataType = .undefined,
    dims: []const i64 = &.{},
    /// Little-endian element data, either inline in the file or a window into
    /// the external weight blob.
    data: []const u8 = &.{},

    pub fn elementCount(self: Tensor) usize {
        var count: usize = 1;
        for (self.dims) |dim| count *= @intCast(@max(dim, 0));
        return count;
    }

    pub fn f32s(self: Tensor) []const f32 {
        return @alignCast(std.mem.bytesAsSlice(f32, self.data));
    }

    pub fn i64s(self: Tensor) []const i64 {
        return @alignCast(std.mem.bytesAsSlice(i64, self.data));
    }
};

pub const Attribute = struct {
    name: []const u8 = "",
    i: i64 = 0,
    f: f32 = 0,
    s: []const u8 = "",
    ints: []const i64 = &.{},
    floats: []const f32 = &.{},
    tensor: ?Tensor = null,
};

pub const Node = struct {
    op_type: []const u8 = "",
    name: []const u8 = "",
    inputs: []const []const u8 = &.{},
    outputs: []const []const u8 = &.{},
    attributes: []const Attribute = &.{},

    pub fn attribute(self: Node, name: []const u8) ?Attribute {
        for (self.attributes) |a| {
            if (std.mem.eql(u8, a.name, name)) return a;
        }
        return null;
    }

    pub fn int(self: Node, name: []const u8, default: i64) i64 {
        return if (self.attribute(name)) |a| a.i else default;
    }

    pub fn float(self: Node, name: []const u8, default: f32) f32 {
        return if (self.attribute(name)) |a| a.f else default;
    }

    pub fn ints(self: Node, name: []const u8) []const i64 {
        return if (self.attribute(name)) |a| a.ints else &.{};
    }
};

/// A graph input or output. A dimension the export left symbolic reads -1.
pub const ValueInfo = struct {
    name: []const u8 = "",
    dtype: DataType = .undefined,
    dims: []const i64 = &.{},
};

pub const Graph = struct {
    arena: std.heap.ArenaAllocator,
    file: []align(std.heap.page_size_min) u8,
    weights: []align(std.heap.page_size_min) u8,

    nodes: []Node,
    initializers: []Tensor,
    inputs: []ValueInfo,
    outputs: []ValueInfo,

    /// Maps every initializer name to its tensor.
    constants: std.StringHashMapUnmanaged(Tensor),

    pub fn open(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !Graph {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        errdefer arena.deinit();

        const file = try mapFile(io, path);
        errdefer std.posix.munmap(file);

        var parser: Parser = .{
            .arena = arena.allocator(),
            .io = io,
            .directory = std.fs.path.dirname(path) orelse ".",
        };
        var graph = try parser.parseModel(file);

        graph.arena = arena;
        graph.file = file;
        graph.weights = parser.weights;
        return graph;
    }

    pub fn deinit(self: *Graph) void {
        self.constants.deinit(self.arena.allocator());
        self.arena.deinit();
        if (self.weights.len != 0) std.posix.munmap(self.weights);
        std.posix.munmap(self.file);
        self.* = undefined;
    }

    pub fn constant(self: Graph, name: []const u8) ?Tensor {
        return self.constants.get(name);
    }
};

fn mapFile(io: std.Io, path: []const u8) ![]align(std.heap.page_size_min) u8 {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const size = (try file.stat(io)).size;
    if (size == 0) return error.EmptyFile;
    return std.posix.mmap(
        null,
        size,
        .{ .READ = true },
        .{ .TYPE = .PRIVATE },
        file.handle,
        0,
    );
}

/// Protobuf wire format: enough of it to read what an ONNX export contains.
const Reader = struct {
    bytes: []const u8,
    pos: usize = 0,

    const Field = struct { number: u32, wire: u3 };

    fn eof(self: Reader) bool {
        return self.pos >= self.bytes.len;
    }

    fn field(self: *Reader) !Field {
        const key = try self.varint();
        return .{ .number = @intCast(key >> 3), .wire = @intCast(key & 7) };
    }

    fn varint(self: *Reader) !u64 {
        var result: u64 = 0;
        var shift: u6 = 0;
        while (true) {
            if (self.pos >= self.bytes.len) return error.TruncatedProtobuf;
            const byte = self.bytes[self.pos];
            self.pos += 1;
            result |= @as(u64, byte & 0x7f) << shift;
            if (byte & 0x80 == 0) return result;
            if (shift >= 63) return error.MalformedVarint;
            shift += 7;
        }
    }

    fn fixed32(self: *Reader) !u32 {
        if (self.pos + 4 > self.bytes.len) return error.TruncatedProtobuf;
        defer self.pos += 4;
        return std.mem.readInt(u32, self.bytes[self.pos..][0..4], .little);
    }

    fn slice(self: *Reader) ![]const u8 {
        const len: usize = @intCast(try self.varint());
        if (self.pos + len > self.bytes.len) return error.TruncatedProtobuf;
        defer self.pos += len;
        return self.bytes[self.pos..][0..len];
    }

    fn sub(self: *Reader) !Reader {
        return .{ .bytes = try self.slice() };
    }

    fn skip(self: *Reader, wire: u3) !void {
        switch (wire) {
            0 => _ = try self.varint(),
            1 => self.pos += 8,
            2 => _ = try self.slice(),
            5 => self.pos += 4,
            else => return error.UnsupportedWireType,
        }
    }
};

const Parser = struct {
    arena: std.mem.Allocator,
    io: std.Io,
    directory: []const u8,
    weights: []align(std.heap.page_size_min) u8 = &.{},

    fn parseModel(self: *Parser, bytes: []const u8) !Graph {
        var model: Reader = .{ .bytes = bytes };
        while (!model.eof()) {
            const f = try model.field();
            if (f.number == 7 and f.wire == 2) return self.parseGraph(try model.sub());
            try model.skip(f.wire);
        }
        return error.NoGraph;
    }

    fn parseGraph(self: *Parser, reader: Reader) !Graph {
        var r = reader;
        var nodes: std.ArrayList(Node) = .empty;
        var initializers: std.ArrayList(Tensor) = .empty;
        var inputs: std.ArrayList(ValueInfo) = .empty;
        var outputs: std.ArrayList(ValueInfo) = .empty;

        while (!r.eof()) {
            const f = try r.field();
            switch (f.number) {
                1 => try nodes.append(self.arena, try self.parseNode(try r.sub())),
                5 => try initializers.append(self.arena, try self.parseTensor(try r.sub())),
                11 => try inputs.append(self.arena, try self.parseValueInfo(try r.sub())),
                12 => try outputs.append(self.arena, try self.parseValueInfo(try r.sub())),
                else => try r.skip(f.wire),
            }
        }

        var constants: std.StringHashMapUnmanaged(Tensor) = .empty;
        for (initializers.items) |tensor| try constants.put(self.arena, tensor.name, tensor);

        return .{
            .arena = undefined,
            .file = &.{},
            .weights = &.{},
            .nodes = nodes.items,
            .initializers = initializers.items,
            .inputs = inputs.items,
            .outputs = outputs.items,
            .constants = constants,
        };
    }

    fn parseNode(self: *Parser, reader: Reader) !Node {
        var r = reader;
        var node: Node = .{};
        var inputs: std.ArrayList([]const u8) = .empty;
        var outputs: std.ArrayList([]const u8) = .empty;
        var attributes: std.ArrayList(Attribute) = .empty;

        while (!r.eof()) {
            const f = try r.field();
            switch (f.number) {
                1 => try inputs.append(self.arena, try r.slice()),
                2 => try outputs.append(self.arena, try r.slice()),
                3 => node.name = try r.slice(),
                4 => node.op_type = try r.slice(),
                5 => try attributes.append(self.arena, try self.parseAttribute(try r.sub())),
                else => try r.skip(f.wire),
            }
        }

        node.inputs = inputs.items;
        node.outputs = outputs.items;
        node.attributes = attributes.items;
        return node;
    }

    fn parseAttribute(self: *Parser, reader: Reader) !Attribute {
        var r = reader;
        var attribute: Attribute = .{};
        var ints: std.ArrayList(i64) = .empty;
        var floats: std.ArrayList(f32) = .empty;

        while (!r.eof()) {
            const f = try r.field();
            switch (f.number) {
                1 => attribute.name = try r.slice(),
                2 => attribute.f = @bitCast(try r.fixed32()),
                3 => attribute.i = @bitCast(try r.varint()),
                4 => attribute.s = try r.slice(),
                5 => attribute.tensor = try self.parseTensor(try r.sub()),
                7 => if (f.wire == 2) {
                    var packed_floats = try r.sub();
                    while (!packed_floats.eof()) {
                        try floats.append(self.arena, @bitCast(try packed_floats.fixed32()));
                    }
                } else try floats.append(self.arena, @bitCast(try r.fixed32())),
                8 => if (f.wire == 2) {
                    var packed_ints = try r.sub();
                    while (!packed_ints.eof()) {
                        try ints.append(self.arena, @bitCast(try packed_ints.varint()));
                    }
                } else try ints.append(self.arena, @bitCast(try r.varint())),
                else => try r.skip(f.wire),
            }
        }

        attribute.ints = ints.items;
        attribute.floats = floats.items;
        return attribute;
    }

    fn parseTensor(self: *Parser, reader: Reader) !Tensor {
        var r = reader;
        var tensor: Tensor = .{};
        var dims: std.ArrayList(i64) = .empty;

        // Values small enough to live in the file appear in a typed list
        // rather than as raw bytes; they are rewritten into raw bytes below so
        // that everything downstream sees one representation.
        var typed_i64: std.ArrayList(i64) = .empty;
        var typed_f32: std.ArrayList(f32) = .empty;
        var external: []const u8 = &.{};
        var offset: u64 = 0;
        var length: u64 = 0;

        while (!r.eof()) {
            const f = try r.field();
            switch (f.number) {
                1 => if (f.wire == 2) {
                    var packed_dims = try r.sub();
                    while (!packed_dims.eof()) try dims.append(self.arena, @bitCast(try packed_dims.varint()));
                } else try dims.append(self.arena, @bitCast(try r.varint())),
                2 => tensor.dtype = @enumFromInt(try r.varint()),
                4 => if (f.wire == 2) {
                    var packed_floats = try r.sub();
                    while (!packed_floats.eof()) try typed_f32.append(self.arena, @bitCast(try packed_floats.fixed32()));
                } else try typed_f32.append(self.arena, @bitCast(try r.fixed32())),
                7 => if (f.wire == 2) {
                    var packed_ints = try r.sub();
                    while (!packed_ints.eof()) try typed_i64.append(self.arena, @bitCast(try packed_ints.varint()));
                } else try typed_i64.append(self.arena, @bitCast(try r.varint())),
                8 => tensor.name = try r.slice(),
                9 => tensor.data = try r.slice(),
                13 => {
                    var entry = try r.sub();
                    var key: []const u8 = "";
                    var value: []const u8 = "";
                    while (!entry.eof()) {
                        const ef = try entry.field();
                        switch (ef.number) {
                            1 => key = try entry.slice(),
                            2 => value = try entry.slice(),
                            else => try entry.skip(ef.wire),
                        }
                    }
                    if (std.mem.eql(u8, key, "location")) external = value;
                    if (std.mem.eql(u8, key, "offset")) offset = try std.fmt.parseInt(u64, value, 10);
                    if (std.mem.eql(u8, key, "length")) length = try std.fmt.parseInt(u64, value, 10);
                },
                else => try r.skip(f.wire),
            }
        }

        tensor.dims = dims.items;
        if (typed_i64.items.len != 0) tensor.data = std.mem.sliceAsBytes(typed_i64.items);
        if (typed_f32.items.len != 0) tensor.data = std.mem.sliceAsBytes(typed_f32.items);

        if (external.len != 0) {
            const blob = try self.externalWeights(external);
            if (offset + length > blob.len) return error.ExternalDataOutOfRange;
            tensor.data = blob[@intCast(offset)..][0..@intCast(length)];
        }
        if (tensor.data.len != 0 and @intFromPtr(tensor.data.ptr) % 8 != 0) {
            const aligned = try self.arena.alignedAlloc(u8, .@"8", tensor.data.len);
            @memcpy(aligned, tensor.data);
            tensor.data = aligned;
        }
        return tensor;
    }

    /// Every initializer in an export points at the same blob, so it is mapped
    /// once and shared. Mapping keeps a 1.8 GiB weight file out of the heap.
    fn externalWeights(self: *Parser, location: []const u8) ![]const u8 {
        if (self.weights.len != 0) return self.weights;
        const path = try std.fs.path.join(self.arena, &.{ self.directory, location });
        self.weights = try mapFile(self.io, path);
        return self.weights;
    }

    fn parseValueInfo(self: *Parser, reader: Reader) !ValueInfo {
        var r = reader;
        var info: ValueInfo = .{};
        var dims: std.ArrayList(i64) = .empty;

        while (!r.eof()) {
            const f = try r.field();
            switch (f.number) {
                1 => info.name = try r.slice(),
                2 => {
                    var type_proto = try r.sub();
                    while (!type_proto.eof()) {
                        const tf = try type_proto.field();
                        if (tf.number != 1) {
                            try type_proto.skip(tf.wire);
                            continue;
                        }
                        var tensor_type = try type_proto.sub();
                        while (!tensor_type.eof()) {
                            const ef = try tensor_type.field();
                            switch (ef.number) {
                                1 => info.dtype = @enumFromInt(try tensor_type.varint()),
                                2 => {
                                    var shape = try tensor_type.sub();
                                    while (!shape.eof()) {
                                        const sf = try shape.field();
                                        if (sf.number != 1) {
                                            try shape.skip(sf.wire);
                                            continue;
                                        }
                                        var dimension = try shape.sub();
                                        var value: i64 = -1;
                                        while (!dimension.eof()) {
                                            const df = try dimension.field();
                                            if (df.number == 1 and df.wire == 0) {
                                                value = @bitCast(try dimension.varint());
                                            } else try dimension.skip(df.wire);
                                        }
                                        try dims.append(self.arena, value);
                                    }
                                },
                                else => try tensor_type.skip(ef.wire),
                            }
                        }
                    }
                },
                else => try r.skip(f.wire),
            }
        }

        info.dims = dims.items;
        return info;
    }
};
