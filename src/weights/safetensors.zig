const std = @import("std");
const native_endian = @import("builtin").cpu.arch.endian();
const Tensor = @import("../tensor/tensor.zig").Tensor;

pub const DType = enum {
    F32,
    F16,
    BF16,
    I32,
    U8,

    pub fn fromString(str: []const u8) ?DType {
        if (std.mem.eql(u8, str, "F32")) return .F32;
        if (std.mem.eql(u8, str, "F16")) return .F16;
        if (std.mem.eql(u8, str, "BF16")) return .BF16;
        if (std.mem.eql(u8, str, "I32")) return .I32;
        if (std.mem.eql(u8, str, "U8")) return .U8;
        return null;
    }
};

pub inline fn f16ToF32(val: u16) f32 {
    const sign = (val >> 15) & 0x1;
    const exp = (val >> 10) & 0x1f;
    const frac = val & 0x3ff;

    if (exp == 0) {
        if (frac == 0) {
            return if (sign == 1) -0.0 else 0.0;
        }
        var f: f32 = @floatFromInt(frac);
        f /= 1024.0;
        f *= std.math.pow(f32, 2.0, -14.0);
        return if (sign == 1) -f else f;
    } else if (exp == 0x1f) {
        if (frac == 0) {
            return if (sign == 1) -std.math.inf(f32) else std.math.inf(f32);
        }
        return std.math.nan(f32);
    }

    const f32_sign = @as(u32, sign) << 31;
    const f32_exp = @as(u32, exp - 15 + 127) << 23;
    const f32_frac = @as(u32, frac) << 13;
    const u = f32_sign | f32_exp | f32_frac;
    return @bitCast(u);
}

pub inline fn bf16ToF32(val: u16) f32 {
    const u: u32 = @as(u32, val) << 16;
    return @bitCast(u);
}

pub const SafeTensorInfo = struct {
    dtype: DType,
    shape: []usize,
    data_offsets: [2]usize,
};

/// A single entry of a SafeTensors header, without any tensor data attached.
pub const TensorEntry = struct {
    name: []const u8,
    dtype: DType,
    shape: []usize,
    data_offsets: [2]usize,

    pub fn numElements(self: TensorEntry) usize {
        return Tensor.totalElements(self.shape);
    }

    pub fn byteLen(self: TensorEntry) usize {
        return self.data_offsets[1] - self.data_offsets[0];
    }
};

/// Header-only view of a SafeTensors file: every tensor name, dtype and shape,
/// parsed without reading (or allocating) any of the tensor payload. Lets a
/// multi-gigabyte checkpoint be inspected in milliseconds.
pub const Header = struct {
    allocator: std.mem.Allocator,
    entries: []TensorEntry,
    data_start: usize,
    file_size: usize,

    pub fn deinit(self: *Header) void {
        for (self.entries) |entry| {
            self.allocator.free(entry.name);
            self.allocator.free(entry.shape);
        }
        self.allocator.free(self.entries);
    }

    pub fn totalElements(self: Header) usize {
        var total: usize = 0;
        for (self.entries) |entry| total += entry.numElements();
        return total;
    }
};

/// Parses the JSON header block into owned entries, sorted by tensor name.
pub fn parseHeaderEntries(allocator: std.mem.Allocator, header_bytes: []const u8) ![]TensorEntry {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, header_bytes, .{});
    defer parsed.deinit();

    if (parsed.value != .object) return error.InvalidSafeTensorsHeader;

    var list: std.ArrayList(TensorEntry) = .empty;
    errdefer {
        for (list.items) |entry| {
            allocator.free(entry.name);
            allocator.free(entry.shape);
        }
        list.deinit(allocator);
    }

    var it = parsed.value.object.iterator();
    while (it.next()) |kv| {
        const name = kv.key_ptr.*;
        if (std.mem.eql(u8, name, "__metadata__")) continue;

        const t_obj = kv.value_ptr.*;
        if (t_obj != .object) continue;

        const dtype_val = t_obj.object.get("dtype") orelse continue;
        if (dtype_val != .string) continue;
        const dtype = DType.fromString(dtype_val.string) orelse return error.UnsupportedDType;

        const shape_val = t_obj.object.get("shape") orelse continue;
        if (shape_val != .array) continue;

        var shape_list: std.ArrayList(usize) = .empty;
        errdefer shape_list.deinit(allocator);
        for (shape_val.array.items) |item| {
            if (item == .integer) {
                try shape_list.append(allocator, @intCast(item.integer));
            }
        }

        const offsets_val = t_obj.object.get("data_offsets") orelse continue;
        if (offsets_val != .array or offsets_val.array.items.len != 2) continue;

        const name_dupe = try allocator.dupe(u8, name);
        errdefer allocator.free(name_dupe);

        try list.append(allocator, TensorEntry{
            .name = name_dupe,
            .dtype = dtype,
            .shape = try shape_list.toOwnedSlice(allocator),
            .data_offsets = .{
                @intCast(offsets_val.array.items[0].integer),
                @intCast(offsets_val.array.items[1].integer),
            },
        });
    }

    const entries = try list.toOwnedSlice(allocator);
    std.mem.sort(TensorEntry, entries, {}, struct {
        fn lessThan(_: void, a: TensorEntry, b: TensorEntry) bool {
            return std.mem.lessThan(u8, a.name, b.name);
        }
    }.lessThan);
    return entries;
}

/// Reads just the header of a SafeTensors file from disk.
/// Opens a file read-only through the raw syscall layer used across this
/// codebase, returning the descriptor and its size.
fn openReadOnly(file_path: []const u8) !struct { fd: i32, size: usize } {
    var path_z: [1024]u8 = undefined;
    if (file_path.len >= path_z.len) return error.PathTooLong;
    @memcpy(path_z[0..file_path.len], file_path);
    path_z[file_path.len] = 0;
    const p_slice: [:0]const u8 = path_z[0..file_path.len :0];

    const fd_res = std.os.linux.open(p_slice, .{ .ACCMODE = .RDONLY }, 0);
    const signed: isize = @bitCast(fd_res);
    if (signed < 0) return error.FileNotFound;
    const fd: i32 = @intCast(signed);
    errdefer _ = std.os.linux.close(fd);

    const size = std.os.linux.lseek(fd, 0, 2);
    _ = std.os.linux.lseek(fd, 0, 0);
    return .{ .fd = fd, .size = size };
}

/// Whether a checkpoint is readable at `file_path`, so callers can say
/// something useful about a missing one before starting any work.
pub fn exists(file_path: []const u8) bool {
    const file = openReadOnly(file_path) catch return false;
    _ = std.os.linux.close(file.fd);
    return true;
}

fn readExactly(fd: i32, buffer: []u8) !void {
    var total_read: usize = 0;
    while (total_read < buffer.len) {
        const n_res = std.os.linux.read(fd, buffer.ptr + total_read, buffer.len - total_read);
        const n_signed: isize = @bitCast(n_res);
        if (n_signed <= 0) break;
        total_read += @intCast(n_signed);
    }
    if (total_read < buffer.len) return error.UnexpectedEOF;
}

pub fn readHeader(allocator: std.mem.Allocator, file_path: []const u8) !Header {
    const file = try openReadOnly(file_path);
    defer _ = std.os.linux.close(file.fd);

    if (file.size < 8) return error.InvalidSafeTensorsFile;

    var len_bytes: [8]u8 = undefined;
    try readExactly(file.fd, &len_bytes);
    const header_len: usize = @intCast(std.mem.readInt(u64, &len_bytes, .little));
    if (8 + header_len > file.size) return error.InvalidSafeTensorsFile;

    const header_bytes = try allocator.alloc(u8, header_len);
    defer allocator.free(header_bytes);
    try readExactly(file.fd, header_bytes);

    return Header{
        .allocator = allocator,
        .entries = try parseHeaderEntries(allocator, header_bytes),
        .data_start = 8 + header_len,
        .file_size = file.size,
    };
}

pub const SafeTensors = struct {
    allocator: std.mem.Allocator,
    tensors: std.StringHashMap(Tensor),

    pub fn init(allocator: std.mem.Allocator) SafeTensors {
        return SafeTensors{
            .allocator = allocator,
            .tensors = std.StringHashMap(Tensor).init(allocator),
        };
    }

    pub fn deinit(self: *SafeTensors) void {
        var it = self.tensors.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
        }
        self.tensors.deinit();
    }

    pub fn loadFromSlice(allocator: std.mem.Allocator, bytes: []const u8) !SafeTensors {
        if (bytes.len < 8) return error.InvalidSafeTensorsFile;
        const header_len = std.mem.readInt(u64, bytes[0..8], .little);
        if (8 + header_len > bytes.len) return error.InvalidSafeTensorsFile;

        const header_bytes = bytes[8 .. 8 + header_len];
        const data_bytes = bytes[8 + header_len ..];

        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, header_bytes, .{});
        defer parsed.deinit();

        var res = SafeTensors.init(allocator);
        errdefer res.deinit();

        if (parsed.value != .object) return error.InvalidSafeTensorsHeader;

        var it = parsed.value.object.iterator();
        while (it.next()) |entry| {
            const name = entry.key_ptr.*;
            if (std.mem.eql(u8, name, "__metadata__")) continue;

            const t_obj = entry.value_ptr.*;
            if (t_obj != .object) continue;

            const dtype_val = t_obj.object.get("dtype") orelse continue;
            if (dtype_val != .string) continue;
            const dtype = DType.fromString(dtype_val.string) orelse continue;

            const shape_val = t_obj.object.get("shape") orelse continue;
            if (shape_val != .array) continue;

            var shape_list: std.ArrayList(usize) = .empty;
            defer shape_list.deinit(allocator);
            for (shape_val.array.items) |item| {
                if (item == .integer) {
                    try shape_list.append(allocator, @intCast(item.integer));
                }
            }

            const offsets_val = t_obj.object.get("data_offsets") orelse continue;
            if (offsets_val != .array or offsets_val.array.items.len != 2) continue;

            const start_off: usize = @intCast(offsets_val.array.items[0].integer);
            const end_off: usize = @intCast(offsets_val.array.items[1].integer);
            if (end_off > data_bytes.len or start_off > end_off) return error.InvalidDataOffsets;

            const raw_tensor_bytes = data_bytes[start_off..end_off];
            const num_elems = Tensor.totalElements(shape_list.items);

            var t = try Tensor.init(allocator, shape_list.items);
            errdefer t.deinit();

            switch (dtype) {
                .F32 => {
                    if (raw_tensor_bytes.len != num_elems * 4) return error.TensorSizeMismatch;
                    // Checkpoint payloads are 4-byte aligned in practice, which lets
                    // the 860M f32 weights of a full SAM 3 release land via memcpy
                    // instead of an element-at-a-time decode.
                    if (native_endian == .little and @intFromPtr(raw_tensor_bytes.ptr) % @alignOf(f32) == 0) {
                        const src: [*]const f32 = @ptrCast(@alignCast(raw_tensor_bytes.ptr));
                        @memcpy(t.data, src[0..num_elems]);
                    } else {
                        for (0..num_elems) |i| {
                            const val_u32 = std.mem.readInt(u32, raw_tensor_bytes[i * 4 ..][0..4], .little);
                            t.data[i] = @bitCast(val_u32);
                        }
                    }
                },
                .F16 => {
                    if (raw_tensor_bytes.len != num_elems * 2) return error.TensorSizeMismatch;
                    for (0..num_elems) |i| {
                        const val_u16 = std.mem.readInt(u16, raw_tensor_bytes[i * 2 ..][0..2], .little);
                        t.data[i] = f16ToF32(val_u16);
                    }
                },
                .BF16 => {
                    if (raw_tensor_bytes.len != num_elems * 2) return error.TensorSizeMismatch;
                    for (0..num_elems) |i| {
                        const val_u16 = std.mem.readInt(u16, raw_tensor_bytes[i * 2 ..][0..2], .little);
                        t.data[i] = bf16ToF32(val_u16);
                    }
                },
                .I32 => {
                    if (raw_tensor_bytes.len != num_elems * 4) return error.TensorSizeMismatch;
                    for (0..num_elems) |i| {
                        const val_i32 = std.mem.readInt(i32, raw_tensor_bytes[i * 4 ..][0..4], .little);
                        t.data[i] = @floatFromInt(val_i32);
                    }
                },
                .U8 => {
                    if (raw_tensor_bytes.len != num_elems) return error.TensorSizeMismatch;
                    for (0..num_elems) |i| {
                        t.data[i] = @floatFromInt(raw_tensor_bytes[i]);
                    }
                },
            }

            const name_dupe = try allocator.dupe(u8, name);
            errdefer allocator.free(name_dupe);
            try res.tensors.put(name_dupe, t);
        }

        return res;
    }

    /// Maps the checkpoint instead of buffering it, so peak memory is the size
    /// of the materialised f32 tensors rather than twice the file size. A 3.4 GB
    /// checkpoint like Meta's full SAM 3 release is otherwise unloadable on a
    /// machine with modest RAM.
    pub fn loadFromFile(allocator: std.mem.Allocator, file_path: []const u8) !SafeTensors {
        const file = try openReadOnly(file_path);
        defer _ = std.os.linux.close(file.fd);

        if (file.size < 8) return error.InvalidSafeTensorsFile;

        const mapped = try std.posix.mmap(
            null,
            file.size,
            .{ .READ = true },
            .{ .TYPE = .PRIVATE },
            file.fd,
            0,
        );
        defer std.posix.munmap(mapped);

        return loadFromSlice(allocator, mapped);
    }

    pub fn writeToSlice(self: SafeTensors, allocator: std.mem.Allocator) ![]u8 {
        var out_list: std.ArrayList(u8) = .empty;
        errdefer out_list.deinit(allocator);

        var json_buf: std.ArrayList(u8) = .empty;
        defer json_buf.deinit(allocator);
        var jwriter = json_buf.writer(allocator);

        try jwriter.writeAll("{");
        var curr_offset: usize = 0;
        var it = self.tensors.iterator();
        var first = true;

        while (it.next()) |entry| {
            if (!first) try jwriter.writeAll(",");
            first = false;

            const name = entry.key_ptr.*;
            const t = entry.value_ptr.*;
            const byte_len = t.numElements() * 4;

            try jwriter.print("\"{s}\":{{\"dtype\":\"F32\",\"shape\":[", .{name});
            for (t.shape, 0..) |s, idx| {
                if (idx > 0) try jwriter.writeAll(",");
                try jwriter.print("{d}", .{s});
            }
            try jwriter.print("],\"data_offsets\":[{d},{d}]}}", .{ curr_offset, curr_offset + byte_len });
            curr_offset += byte_len;
        }
        try jwriter.writeAll("}");

        const header_len: u64 = json_buf.items.len;
        var len_bytes: [8]u8 = undefined;
        std.mem.writeInt(u64, &len_bytes, header_len, .little);

        try out_list.appendSlice(allocator, &len_bytes);
        try out_list.appendSlice(allocator, json_buf.items);

        it = self.tensors.iterator();
        while (it.next()) |entry| {
            const t = entry.value_ptr.*;
            for (t.data) |val| {
                var val_bytes: [4]u8 = undefined;
                std.mem.writeInt(u32, &val_bytes, @bitCast(val), .little);
                try out_list.appendSlice(allocator, &val_bytes);
            }
        }

        return out_list.toOwnedSlice(allocator);
    }
};

test "header parsing reports tensor layout without materialising data" {
    const allocator = std.testing.allocator;

    var st = SafeTensors.init(allocator);
    defer st.deinit();

    const w_shape = [_]usize{ 2, 3 };
    try st.tensors.put(
        try allocator.dupe(u8, "encoder.weight"),
        try Tensor.initConstant(allocator, &w_shape, 1.5),
    );
    const b_shape = [_]usize{4};
    try st.tensors.put(
        try allocator.dupe(u8, "encoder.bias"),
        try Tensor.initZeros(allocator, &b_shape),
    );

    const bytes = try st.writeToSlice(allocator);
    defer allocator.free(bytes);

    const header_len: usize = @intCast(std.mem.readInt(u64, bytes[0..8], .little));
    const entries = try parseHeaderEntries(allocator, bytes[8 .. 8 + header_len]);
    defer {
        for (entries) |entry| {
            allocator.free(entry.name);
            allocator.free(entry.shape);
        }
        allocator.free(entries);
    }

    try std.testing.expectEqual(@as(usize, 2), entries.len);
    // Entries come back sorted by name.
    try std.testing.expectEqualStrings("encoder.bias", entries[0].name);
    try std.testing.expectEqualStrings("encoder.weight", entries[1].name);
    try std.testing.expectEqualSlices(usize, &b_shape, entries[0].shape);
    try std.testing.expectEqual(DType.F32, entries[1].dtype);
    try std.testing.expectEqual(@as(usize, 6), entries[1].numElements());
    try std.testing.expectEqual(@as(usize, 24), entries[1].byteLen());

    var loaded = try SafeTensors.loadFromSlice(allocator, bytes);
    defer loaded.deinit();
    const weight = loaded.tensors.get("encoder.weight").?;
    try std.testing.expectEqual(@as(f32, 1.5), weight.data[0]);
}
