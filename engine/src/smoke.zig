//! Exercises the same native CUDA module the graph executor uses. This is a
//! small, deterministic check that driver loading, PTX JIT, memory transfers,
//! broadcasting, unary math and matrix multiplication all work on this GPU.

const std = @import("std");
const engine = @import("engine");

pub fn main() !void {
    var device = try engine.Device.init(0);
    defer device.deinit();

    var name_buffer: [128]u8 = undefined;
    const capability = try device.capability();
    std.debug.print("{s} (sm_{d}{d})\n", .{
        try device.name(&name_buffer),
        capability[0],
        capability[1],
    });

    try checkBinary(&device);
    try checkUnary(&device);
    try checkMatmul(&device);
    try device.synchronize();
    std.debug.print("native CUDA tensor kernels passed\n", .{});
}

fn checkBinary(device: *engine.Device) !void {
    const a_host = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const b_host = [_]f32{ 10, 20, 30 };
    var output: [6]f32 = undefined;
    // Output [2, 3], A [2, 3], B [1, 3]. A zero leading stride broadcasts B.
    const metadata = [_]u32{ 2, 3, 3, 1, 0, 1 };

    const a = try engine.Device.alloc(f32, a_host.len);
    defer a.free();
    const b = try engine.Device.alloc(f32, b_host.len);
    defer b.free();
    const out = try engine.Device.alloc(f32, output.len);
    defer out.free();
    const meta = try engine.Device.alloc(u32, metadata.len);
    defer meta.free();
    try a.upload(&a_host);
    try b.upload(&b_host);
    try meta.upload(&metadata);

    try device.binary.launch(.{ .x = 1 }, .{ .x = 32 }, .{
        a.ptr, b.ptr, out.ptr, meta.ptr, @as(u32, 2), @as(u32, output.len), @as(u32, 0),
    });
    try device.synchronize();
    try out.download(&output);
    try expectApprox(&output, &.{ 11, 22, 33, 14, 25, 36 });
}

fn checkUnary(device: *engine.Device) !void {
    const input = [_]f32{ -2, -1, 0, 1, 2 };
    var output: [input.len]f32 = undefined;
    const x = try engine.Device.alloc(f32, input.len);
    defer x.free();
    const out = try engine.Device.alloc(f32, output.len);
    defer out.free();
    try x.upload(&input);
    // Unary 7 is ReLU; enum values are part of the PTX host ABI.
    try device.unary.launch(.{ .x = 1 }, .{ .x = 32 }, .{
        x.ptr, out.ptr, @as(u32, output.len), @as(u32, 7),
    });
    try device.synchronize();
    try out.download(&output);
    try expectApprox(&output, &.{ 0, 0, 0, 1, 2 });
}

fn checkMatmul(device: *engine.Device) !void {
    const a_host = [_]f32{ 1, 2, 3, 4, 5, 6 }; // 2 x 3
    const b_host = [_]f32{ 7, 8, 9, 10, 11, 12 }; // 3 x 2
    var output: [4]f32 = undefined;
    const a = try engine.Device.alloc(f32, a_host.len);
    defer a.free();
    const b = try engine.Device.alloc(f32, b_host.len);
    defer b.free();
    const out = try engine.Device.alloc(f32, output.len);
    defer out.free();
    try a.upload(&a_host);
    try b.upload(&b_host);

    try device.matmul.launch(.{ .x = 1, .y = 1, .z = 1 }, .{ .x = 16, .y = 16 }, .{
        a.ptr,       b.ptr,       out.ptr,
        @as(u32, 2), @as(u32, 2), @as(u32, 3),
        @as(u32, 0), @as(u32, 0), @as(u32, 4),
        @as(u32, 0),
    });
    try device.synchronize();
    try out.download(&output);
    try expectApprox(&output, &.{ 58, 64, 139, 154 });
}

fn expectApprox(actual: []const f32, expected: []const f32) !void {
    if (actual.len != expected.len) return error.WrongLength;
    for (actual, expected, 0..) |got, want, i| {
        if (@abs(got - want) > 1e-4) {
            std.debug.print("mismatch at {d}: {d} != {d}\n", .{ i, got, want });
            return error.WrongResult;
        }
    }
}
