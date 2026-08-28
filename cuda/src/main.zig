const std = @import("std");
const cuda = @import("cuda");

/// PTX built from src/kernels.zig by build.zig.
const ptx = @embedFile("kernels.ptx");

pub fn main() !void {
    run() catch |err| switch (err) {
        error.Cuda => {
            std.debug.print("CUDA: {s}\n", .{cuda.lastError()});
            return err;
        },
        else => return err,
    };
}

fn run() !void {
    try cuda.init();

    const ctx = try cuda.Context.init(0);
    defer ctx.deinit();

    var name_buf: [128]u8 = undefined;
    const capability = try ctx.computeCapability();
    std.debug.print("{s} (sm_{d}{d}), up to {d} threads per block\n", .{
        try ctx.name(&name_buf),
        capability[0],
        capability[1],
        try ctx.maxThreadsPerBlock(),
    });

    const module = try cuda.Module.load(ptx);
    defer module.unload();

    const n = 1 << 20;
    const block = 256;
    const grid: cuda.Dim = .{ .x = (n + block - 1) / block };

    var gpa_state: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa_state.deinit();
    const gpa = gpa_state.allocator();

    const a = try gpa.alloc(f32, n);
    defer gpa.free(a);
    const b = try gpa.alloc(f32, n);
    defer gpa.free(b);
    const out = try gpa.alloc(f32, n);
    defer gpa.free(out);
    for (a, b, 0..) |*x, *y, i| {
        x.* = @floatFromInt(i);
        y.* = @floatFromInt(2 * i);
    }

    const da = try cuda.Buffer(f32).alloc(n);
    defer da.free();
    const db = try cuda.Buffer(f32).alloc(n);
    defer db.free();
    const dout = try cuda.Buffer(f32).alloc(n);
    defer dout.free();

    try da.upload(a);
    try db.upload(b);

    const vec_add = try module.function("vecAdd");
    try vec_add.launch(grid, .{ .x = block }, .{ da.ptr, db.ptr, dout.ptr, @as(u32, n) });

    const scale = try module.function("scale");
    try scale.launch(grid, .{ .x = block }, .{ dout.ptr, @as(f32, 0.5), @as(u32, n) });

    try ctx.synchronize();
    try dout.download(out);

    for (out, 0..) |value, i| {
        const want: f32 = @as(f32, @floatFromInt(3 * i)) * 0.5;
        if (value != want) {
            std.debug.print("mismatch at {d}: {d} != {d}\n", .{ i, value, want });
            return error.WrongResult;
        }
    }
    std.debug.print("vecAdd + scale over {d} elements matched on the GPU\n", .{n});
}
