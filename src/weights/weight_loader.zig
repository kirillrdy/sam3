const std = @import("std");
const Tensor = @import("../tensor/tensor.zig").Tensor;
const SafeTensors = @import("safetensors.zig").SafeTensors;

pub const InitType = enum {
    zeros,
    ones,
    xavier,
    kaiming,
    constant,
};

pub const WeightStore = struct {
    allocator: std.mem.Allocator,
    weights: std.StringHashMap(Tensor),

    pub fn init(allocator: std.mem.Allocator) WeightStore {
        return WeightStore{
            .allocator = allocator,
            .weights = std.StringHashMap(Tensor).init(allocator),
        };
    }

    pub fn deinit(self: *WeightStore) void {
        var it = self.weights.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
        }
        self.weights.deinit();
    }

    pub fn get(self: WeightStore, name: []const u8) ?Tensor {
        return self.weights.get(name);
    }

    pub fn put(self: *WeightStore, name: []const u8, tensor: Tensor) !void {
        const name_dupe = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(name_dupe);
        try self.weights.put(name_dupe, tensor);
    }

    pub fn getOrInit(
        self: *WeightStore,
        name: []const u8,
        shape: []const usize,
        init_type: InitType,
        seed: u64,
    ) !Tensor {
        if (self.get(name)) |existing| {
            return existing;
        }

        var t = try Tensor.init(self.allocator, shape);
        errdefer t.deinit();

        switch (init_type) {
            .zeros => @memset(t.data, 0.0),
            .ones => @memset(t.data, 1.0),
            .constant => @memset(t.data, 0.0),
            .xavier => {
                const fan_in: f32 = if (shape.len >= 2) @floatFromInt(shape[1]) else @floatFromInt(shape[0]);
                const fan_out: f32 = @floatFromInt(shape[0]);
                const limit = @sqrt(6.0 / (fan_in + fan_out));
                var prng = std.Random.DefaultPrng.init(seed);
                const rand = prng.random();
                for (t.data) |*v| {
                    v.* = (rand.float(f32) * 2.0 - 1.0) * limit;
                }
            },
            .kaiming => {
                const fan_in: f32 = if (shape.len >= 2) @floatFromInt(shape[1]) else @floatFromInt(shape[0]);
                const std_dev = @sqrt(2.0 / fan_in);
                var prng = std.Random.DefaultPrng.init(seed);
                const rand = prng.random();
                for (t.data) |*v| {
                    const r1 = @max(1e-7, rand.float(f32));
                    const r2 = rand.float(f32);
                    const z = @sqrt(-2.0 * @log(r1)) * @cos(2.0 * std.math.pi * r2);
                    v.* = z * std_dev;
                }
            },
        }

        const name_dupe = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(name_dupe);
        try self.weights.put(name_dupe, t);
        return t;
    }

    pub fn loadFromSafeTensors(self: *WeightStore, file_path: []const u8) !void {
        var st = try SafeTensors.loadFromFile(self.allocator, file_path);
        defer st.tensors.deinit();

        var it = st.tensors.iterator();
        while (it.next()) |entry| {
            try self.weights.put(entry.key_ptr.*, entry.value_ptr.*);
        }
    }
};
