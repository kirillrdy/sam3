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

/// How much of the model was actually served by a loaded checkpoint, as opposed
/// to being filled in with random initialisation.
pub const Coverage = struct {
    checkpoint_tensors: usize,
    resolved: usize,
    synthesised: usize,

    pub fn requested(self: Coverage) usize {
        return self.resolved + self.synthesised;
    }

    pub fn fraction(self: Coverage) f32 {
        if (self.requested() == 0) return 0.0;
        return @as(f32, @floatFromInt(self.resolved)) / @as(f32, @floatFromInt(self.requested()));
    }
};

pub const WeightStore = struct {
    allocator: std.mem.Allocator,
    weights: std.StringHashMap(Tensor),

    /// Tensors read from a checkpoint file.
    checkpoint_tensors: usize,
    /// Weights the model asked for and found in the checkpoint.
    resolved: usize,
    /// Weights the model asked for and had to randomly initialise instead.
    synthesised: usize,

    pub fn init(allocator: std.mem.Allocator) WeightStore {
        return WeightStore{
            .allocator = allocator,
            .weights = std.StringHashMap(Tensor).init(allocator),
            .checkpoint_tensors = 0,
            .resolved = 0,
            .synthesised = 0,
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

    /// Stores a tensor under a key this store takes ownership of, releasing any
    /// tensor already held under that name.
    fn putOwned(self: *WeightStore, name_owned: []const u8, tensor: Tensor) !void {
        const gop = try self.weights.getOrPut(name_owned);
        if (gop.found_existing) {
            self.allocator.free(name_owned);
            gop.value_ptr.*.deinit();
        }
        gop.value_ptr.* = tensor;
    }

    pub fn put(self: *WeightStore, name: []const u8, tensor: Tensor) !void {
        const name_dupe = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(name_dupe);
        try self.putOwned(name_dupe, tensor);
    }

    pub fn coverage(self: WeightStore) Coverage {
        return Coverage{
            .checkpoint_tensors = self.checkpoint_tensors,
            .resolved = self.resolved,
            .synthesised = self.synthesised,
        };
    }

    pub fn getOrInit(
        self: *WeightStore,
        name: []const u8,
        shape: []const usize,
        init_type: InitType,
        seed: u64,
    ) !Tensor {
        if (self.get(name)) |existing| {
            if (std.mem.eql(usize, existing.shape, shape)) {
                self.resolved += 1;
                return existing;
            }
        }
        self.synthesised += 1;

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
        try self.putOwned(name_dupe, t);
        return t;
    }

    pub fn loadFromSafeTensors(self: *WeightStore, file_path: []const u8) !void {
        var st = try SafeTensors.loadFromFile(self.allocator, file_path);
        defer st.tensors.deinit();

        var it = st.tensors.iterator();
        while (it.next()) |entry| {
            // Key and tensor ownership both move into this store.
            try self.putOwned(entry.key_ptr.*, entry.value_ptr.*);
            self.checkpoint_tensors += 1;
        }
    }
};
