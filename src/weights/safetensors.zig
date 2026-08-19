const std = @import("std");
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
                    for (0..num_elems) |i| {
                        const val_u32 = std.mem.readInt(u32, raw_tensor_bytes[i * 4 ..][0..4], .little);
                        t.data[i] = @bitCast(val_u32);
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

    pub fn loadFromFile(allocator: std.mem.Allocator, file_path: []const u8) !SafeTensors {
        var path_z: [1024]u8 = undefined;
        if (file_path.len >= path_z.len) return error.PathTooLong;
        @memcpy(path_z[0..file_path.len], file_path);
        path_z[file_path.len] = 0;
        const p_slice: [:0]const u8 = path_z[0..file_path.len :0];

        const fd_res = std.os.linux.open(p_slice, .{ .ACCMODE = .RDONLY }, 0);
        const signed: isize = @bitCast(fd_res);
        if (signed < 0) return error.FileNotFound;
        const fd: i32 = @intCast(signed);
        defer _ = std.os.linux.close(fd);

        const size = std.os.linux.lseek(fd, 0, 2);
        _ = std.os.linux.lseek(fd, 0, 0);

        const buffer = try allocator.alloc(u8, size);
        defer allocator.free(buffer);

        _ = std.os.linux.read(fd, buffer.ptr, buffer.len);
        return loadFromSlice(allocator, buffer);
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
