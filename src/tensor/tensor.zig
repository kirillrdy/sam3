const std = @import("std");

pub const Tensor = struct {
    data: []f32,
    shape: []usize,
    strides: []usize,
    allocator: ?std.mem.Allocator,

    pub fn computeStrides(shape: []const usize, strides: []usize) void {
        if (shape.len == 0) return;
        var s: usize = 1;
        var i: usize = shape.len;
        while (i > 0) {
            i -= 1;
            strides[i] = s;
            s *= shape[i];
        }
    }

    pub fn totalElements(shape: []const usize) usize {
        if (shape.len == 0) return 0;
        var count: usize = 1;
        for (shape) |d| {
            count *= d;
        }
        return count;
    }

    pub fn init(allocator: std.mem.Allocator, shape: []const usize) !Tensor {
        const num = totalElements(shape);
        const data = try allocator.alloc(f32, num);
        const shape_copy = try allocator.alloc(usize, shape.len);
        @memcpy(shape_copy, shape);
        const strides = try allocator.alloc(usize, shape.len);
        computeStrides(shape, strides);

        return Tensor{
            .data = data,
            .shape = shape_copy,
            .strides = strides,
            .allocator = allocator,
        };
    }

    pub fn initZeros(allocator: std.mem.Allocator, shape: []const usize) !Tensor {
        const t = try init(allocator, shape);
        @memset(t.data, 0.0);
        return t;
    }

    pub fn initOnes(allocator: std.mem.Allocator, shape: []const usize) !Tensor {
        const t = try init(allocator, shape);
        @memset(t.data, 1.0);
        return t;
    }

    pub fn initConstant(allocator: std.mem.Allocator, shape: []const usize, value: f32) !Tensor {
        const t = try init(allocator, shape);
        @memset(t.data, value);
        return t;
    }

    pub fn initRandom(allocator: std.mem.Allocator, shape: []const usize, seed: u64, min_val: f32, max_val: f32) !Tensor {
        const t = try init(allocator, shape);
        var prng = std.Random.DefaultPrng.init(seed);
        const rand = prng.random();
        const diff = max_val - min_val;
        for (t.data) |*v| {
            v.* = min_val + rand.float(f32) * diff;
        }
        return t;
    }

    pub fn fromSlice(allocator: std.mem.Allocator, shape: []const usize, src_slice: []const f32) !Tensor {
        const num = totalElements(shape);
        std.debug.assert(num == src_slice.len);
        const t = try init(allocator, shape);
        @memcpy(t.data, src_slice);
        return t;
    }

    pub fn deinit(self: *Tensor) void {
        if (self.allocator) |alloc| {
            alloc.free(self.data);
            alloc.free(self.shape);
            alloc.free(self.strides);
            self.allocator = null;
        }
    }

    pub fn clone(self: Tensor, allocator: std.mem.Allocator) !Tensor {
        return fromSlice(allocator, self.shape, self.data);
    }

    pub inline fn numElements(self: Tensor) usize {
        return self.data.len;
    }

    pub inline fn dim(self: Tensor) usize {
        return self.shape.len;
    }

    pub inline fn offset1(self: Tensor, i: usize) usize {
        return i * self.strides[0];
    }

    pub inline fn offset2(self: Tensor, i: usize, j: usize) usize {
        return i * self.strides[0] + j * self.strides[1];
    }

    pub inline fn offset3(self: Tensor, i: usize, j: usize, k: usize) usize {
        return i * self.strides[0] + j * self.strides[1] + k * self.strides[2];
    }

    pub inline fn offset4(self: Tensor, i: usize, j: usize, k: usize, l: usize) usize {
        return i * self.strides[0] + j * self.strides[1] + k * self.strides[2] + l * self.strides[3];
    }

    pub inline fn at1(self: Tensor, i: usize) f32 {
        return self.data[self.offset1(i)];
    }

    pub inline fn at2(self: Tensor, i: usize, j: usize) f32 {
        return self.data[self.offset2(i, j)];
    }

    pub inline fn at3(self: Tensor, i: usize, j: usize, k: usize) f32 {
        return self.data[self.offset3(i, j, k)];
    }

    pub inline fn at4(self: Tensor, i: usize, j: usize, k: usize, l: usize) f32 {
        return self.data[self.offset4(i, j, k, l)];
    }

    pub inline fn set1(self: *Tensor, i: usize, val: f32) void {
        self.data[self.offset1(i)] = val;
    }

    pub inline fn set2(self: *Tensor, i: usize, j: usize, val: f32) void {
        self.data[self.offset2(i, j)] = val;
    }

    pub inline fn set3(self: *Tensor, i: usize, j: usize, k: usize, val: f32) void {
        self.data[self.offset3(i, j, k)] = val;
    }

    pub inline fn set4(self: *Tensor, i: usize, j: usize, k: usize, l: usize, val: f32) void {
        self.data[self.offset4(i, j, k, l)] = val;
    }

    pub fn get(self: Tensor, indices: []const usize) f32 {
        var off: usize = 0;
        for (indices, 0..) |idx, d| {
            off += idx * self.strides[d];
        }
        return self.data[off];
    }

    pub fn set(self: *Tensor, indices: []const usize, val: f32) void {
        var off: usize = 0;
        for (indices, 0..) |idx, d| {
            off += idx * self.strides[d];
        }
        self.data[off] = val;
    }

    pub fn reshape(self: *Tensor, new_shape: []const usize) !void {
        const num = totalElements(new_shape);
        if (num != self.numElements()) {
            return error.ShapeMismatch;
        }
        if (self.allocator) |alloc| {
            alloc.free(self.shape);
            alloc.free(self.strides);
            self.shape = try alloc.alloc(usize, new_shape.len);
            @memcpy(self.shape, new_shape);
            self.strides = try alloc.alloc(usize, new_shape.len);
            computeStrides(new_shape, self.strides);
        } else {
            return error.NonOwningTensorCannotBeReshaped;
        }
    }

    pub fn copyFrom(self: *Tensor, other: Tensor) void {
        std.debug.assert(self.data.len == other.data.len);
        @memcpy(self.data, other.data);
    }

    pub fn fill(self: *Tensor, value: f32) void {
        @memset(self.data, value);
    }

    pub fn addInPlace(self: *Tensor, other: Tensor) void {
        std.debug.assert(self.data.len == other.data.len);
        for (self.data, other.data) |*s, o| {
            s.* += o;
        }
    }

    pub fn subInPlace(self: *Tensor, other: Tensor) void {
        std.debug.assert(self.data.len == other.data.len);
        for (self.data, other.data) |*s, o| {
            s.* -= o;
        }
    }

    pub fn mulInPlace(self: *Tensor, other: Tensor) void {
        std.debug.assert(self.data.len == other.data.len);
        for (self.data, other.data) |*s, o| {
            s.* *= o;
        }
    }

    pub fn scaleInPlace(self: *Tensor, factor: f32) void {
        for (self.data) |*s| {
            s.* *= factor;
        }
    }

    pub fn addScaledInPlace(self: *Tensor, other: Tensor, factor: f32) void {
        std.debug.assert(self.data.len == other.data.len);
        for (self.data, other.data) |*s, o| {
            s.* += o * factor;
        }
    }

    pub fn clampInPlace(self: *Tensor, min_val: f32, max_val: f32) void {
        for (self.data) |*s| {
            s.* = std.math.clamp(s.*, min_val, max_val);
        }
    }

    pub fn permute(self: Tensor, allocator: std.mem.Allocator, dims: []const usize) !Tensor {
        std.debug.assert(dims.len == self.shape.len);
        const new_shape = try allocator.alloc(usize, dims.len);
        defer allocator.free(new_shape);
        for (dims, 0..) |d, i| {
            new_shape[i] = self.shape[d];
        }

        var out = try Tensor.init(allocator, new_shape);

        const n = self.numElements();
        const coords = try allocator.alloc(usize, self.shape.len);
        defer allocator.free(coords);

        const out_coords = try allocator.alloc(usize, self.shape.len);
        defer allocator.free(out_coords);

        for (0..n) |i| {
            var rem = i;
            for (0..self.shape.len) |d| {
                coords[d] = rem / self.strides[d];
                rem = rem % self.strides[d];
            }
            for (dims, 0..) |d, target_d| {
                out_coords[target_d] = coords[d];
            }
            out.set(out_coords, self.data[i]);
        }

        return out;
    }

    pub fn slice(self: Tensor, allocator: std.mem.Allocator, axis: usize, start: usize, end: usize) !Tensor {
        std.debug.assert(axis < self.shape.len);
        std.debug.assert(start <= end and end <= self.shape[axis]);

        const new_shape = try allocator.alloc(usize, self.shape.len);
        defer allocator.free(new_shape);
        @memcpy(new_shape, self.shape);
        new_shape[axis] = end - start;

        var out = try Tensor.init(allocator, new_shape);

        const outer_size = blk: {
            var s: usize = 1;
            for (0..axis) |d| s *= self.shape[d];
            break :blk s;
        };
        const inner_size = blk: {
            var s: usize = 1;
            for (axis + 1..self.shape.len) |d| s *= self.shape[d];
            break :blk s;
        };

        const slice_len = end - start;
        const old_axis_size = self.shape[axis];

        for (0..outer_size) |out_idx| {
            for (0..slice_len) |s_idx| {
                const src_axis_idx = start + s_idx;
                const src_off = (out_idx * old_axis_size + src_axis_idx) * inner_size;
                const dst_off = (out_idx * slice_len + s_idx) * inner_size;
                @memcpy(out.data[dst_off .. dst_off + inner_size], self.data[src_off .. src_off + inner_size]);
            }
        }

        return out;
    }

    pub fn concat(allocator: std.mem.Allocator, tensors: []const Tensor, axis: usize) !Tensor {
        std.debug.assert(tensors.len > 0);
        const rank = tensors[0].shape.len;
        std.debug.assert(axis < rank);

        var total_axis_dim: usize = 0;
        for (tensors) |t| {
            std.debug.assert(t.shape.len == rank);
            for (0..rank) |d| {
                if (d != axis) {
                    std.debug.assert(t.shape[d] == tensors[0].shape[d]);
                }
            }
            total_axis_dim += t.shape[axis];
        }

        const new_shape = try allocator.alloc(usize, rank);
        defer allocator.free(new_shape);
        @memcpy(new_shape, tensors[0].shape);
        new_shape[axis] = total_axis_dim;

        var out = try Tensor.init(allocator, new_shape);

        const outer_size = blk: {
            var s: usize = 1;
            for (0..axis) |d| s *= out.shape[d];
            break :blk s;
        };
        const inner_size = blk: {
            var s: usize = 1;
            for (axis + 1..rank) |d| s *= out.shape[d];
            break :blk s;
        };

        for (0..outer_size) |out_idx| {
            var curr_axis_offset: usize = 0;
            for (tensors) |t| {
                const t_axis_len = t.shape[axis];
                for (0..t_axis_len) |s_idx| {
                    const src_off = (out_idx * t_axis_len + s_idx) * inner_size;
                    const dst_off = (out_idx * total_axis_dim + curr_axis_offset + s_idx) * inner_size;
                    @memcpy(out.data[dst_off .. dst_off + inner_size], t.data[src_off .. src_off + inner_size]);
                }
                curr_axis_offset += t_axis_len;
            }
        }

        return out;
    }
};

test "Tensor basic initialization and arithmetic" {
    const allocator = std.testing.allocator;
    const shape = [_]usize{ 2, 3, 4 };
    var t = try Tensor.initZeros(allocator, &shape);
    defer t.deinit();

    try std.testing.expectEqual(@as(usize, 24), t.numElements());
    try std.testing.expectEqual(@as(usize, 3), t.dim());
    try std.testing.expectEqual(@as(f32, 0.0), t.at3(1, 2, 3));

    t.set3(1, 2, 3, 42.0);
    try std.testing.expectEqual(@as(f32, 42.0), t.at3(1, 2, 3));

    var t2 = try Tensor.initConstant(allocator, &shape, 2.0);
    defer t2.deinit();

    t.addInPlace(t2);
    try std.testing.expectEqual(@as(f32, 44.0), t.at3(1, 2, 3));
    try std.testing.expectEqual(@as(f32, 2.0), t.at3(0, 0, 0));
}

test "Tensor permute, slice, concat" {
    const allocator = std.testing.allocator;
    const shape = [_]usize{ 2, 3 };
    var t = try Tensor.init(allocator, &shape);
    defer t.deinit();

    t.set2(0, 0, 1);
    t.set2(0, 1, 2);
    t.set2(0, 2, 3);
    t.set2(1, 0, 4);
    t.set2(1, 1, 5);
    t.set2(1, 2, 6);

    const dims = [_]usize{ 1, 0 };
    var perm = try t.permute(allocator, &dims);
    defer perm.deinit();

    try std.testing.expectEqual(@as(usize, 3), perm.shape[0]);
    try std.testing.expectEqual(@as(usize, 2), perm.shape[1]);
    try std.testing.expectEqual(@as(f32, 1), perm.at2(0, 0));
    try std.testing.expectEqual(@as(f32, 4), perm.at2(0, 1));
    try std.testing.expectEqual(@as(f32, 3), perm.at2(2, 0));
    try std.testing.expectEqual(@as(f32, 6), perm.at2(2, 1));

    var sl = try t.slice(allocator, 1, 1, 3);
    defer sl.deinit();
    try std.testing.expectEqual(@as(usize, 2), sl.shape[0]);
    try std.testing.expectEqual(@as(usize, 2), sl.shape[1]);
    try std.testing.expectEqual(@as(f32, 2), sl.at2(0, 0));
    try std.testing.expectEqual(@as(f32, 3), sl.at2(0, 1));

    const arr = [_]Tensor{ t, t };
    var cat = try Tensor.concat(allocator, &arr, 0);
    defer cat.deinit();
    try std.testing.expectEqual(@as(usize, 4), cat.shape[0]);
    try std.testing.expectEqual(@as(usize, 3), cat.shape[1]);
}
