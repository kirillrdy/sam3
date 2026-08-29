//! Exercises the same native GPU module the graph executor uses. This is a
//! small, deterministic check that driver loading, kernel compilation, transfers,
//! broadcasting, unary math and matrix multiplication all work on this GPU.

const std = @import("std");
const engine = @import("engine");
const Element = engine.Element;

pub fn main() !void {
    var device = engine.Device.init(0) catch |err| {
        std.debug.print("GPU initialization failed: {t}: {s}\n", .{ err, engine.lastError() });
        return err;
    };
    defer device.deinit();

    var name_buffer: [128]u8 = undefined;
    const capability = try device.capability();
    if (@hasDecl(engine.driver, "is_metal")) {
        std.debug.print("{s} (Metal {d}.{d})\n", .{ try device.name(&name_buffer), capability[0], capability[1] });
    } else {
        std.debug.print("{s} (sm_{d}{d})\n", .{ try device.name(&name_buffer), capability[0], capability[1] });
    }

    try checkBinary(&device);
    try checkUnary(&device);
    try checkMatmul(&device);
    try checkMatmulSimd(&device);
    try checkMatmulNBits(&device);
    try device.synchronize();
    std.debug.print("native GPU tensor kernels passed\n", .{});
}

fn checkMatmulNBits(device: *engine.Device) !void {
    const m = 3;
    const n = 5;
    const k = 19;
    const block_size = 16;
    const blocks = 2;
    const blob = block_size / 2;
    var a_host: [m * k]f32 = undefined;
    for (&a_host, 0..) |*value, i| value.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 11)) - 5)) / 7.0;
    var weights: [n * blocks * blob]u8 = undefined;
    for (&weights, 0..) |*value, i| {
        const low: u8 = @intCast((i * 7 + 3) % 16);
        const high: u8 = @intCast((i * 11 + 9) % 16);
        value.* = low | (high << 4);
    }
    const scales_host = [_]f32{ 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25 };
    var expected: [m * n]f32 = @splat(0);
    for (0..m) |row| for (0..n) |col| for (0..k) |depth| {
        const block = depth / block_size;
        const byte = weights[col * blocks * blob + block * blob + (depth % block_size) / 2];
        const nibble = if (depth % 2 == 0) byte & 0xf else byte >> 4;
        expected[row * n + col] += a_host[row * k + depth] * (@as(f32, @floatFromInt(nibble)) - 8.0) * scales_host[col * blocks + block];
    };

    const a = try engine.Device.alloc(Element, a_host.len);
    defer a.free();
    const q = try engine.Device.alloc(u8, weights.len);
    defer q.free();
    const scales = try engine.Device.alloc(Element, scales_host.len);
    defer scales.free();
    const out = try engine.Device.alloc(Element, expected.len);
    defer out.free();
    try upload(a, &a_host);
    try q.upload(&weights);
    try upload(scales, &scales_host);
    try device.matmul_nbits.launch(.{ .x = 1, .y = 1 }, .{ .x = 16, .y = 16 }, .{
        a.ptr, q.ptr, scales.ptr, out.ptr,
        @as(u32, m), @as(u32, n), @as(u32, k), @as(u32, 4), @as(u32, blocks),
    });
    try device.synchronize();
    var actual: [expected.len]f32 = undefined;
    try download(out, &actual);
    try expectApprox(&actual, &expected);
}

fn checkBinary(device: *engine.Device) !void {
    const a_host = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const b_host = [_]f32{ 10, 20, 30 };
    var output: [6]f32 = undefined;
    // Output [2, 3], A [2, 3], B [1, 3]. A zero leading stride broadcasts B.
    const metadata = [_]u32{ 2, 3, 3, 1, 0, 1 };

    const a = try engine.Device.alloc(Element, a_host.len);
    defer a.free();
    const b = try engine.Device.alloc(Element, b_host.len);
    defer b.free();
    const out = try engine.Device.alloc(Element, output.len);
    defer out.free();
    const meta = try engine.Device.alloc(u32, metadata.len);
    defer meta.free();
    try upload(a, &a_host);
    try upload(b, &b_host);
    try meta.upload(&metadata);

    try device.binary.launch(.{ .x = 1 }, .{ .x = 32 }, .{
        a.ptr,                b.ptr,                out.ptr, meta.ptr, @as(u32, 2), @as(u32, output.len), @as(u32, 0),
        @as(u32, output.len), @as(u32, b_host.len),
    });
    try device.synchronize();
    try download(out, &output);
    try expectApprox(&output, &.{ 11, 22, 33, 14, 25, 36 });
}

fn checkUnary(device: *engine.Device) !void {
    const input = [_]f32{ -2, -1, 0, 1, 2 };
    var output: [input.len]f32 = undefined;
    const x = try engine.Device.alloc(Element, input.len);
    defer x.free();
    const out = try engine.Device.alloc(Element, output.len);
    defer out.free();
    try upload(x, &input);
    // Unary 7 is ReLU; enum values are part of the PTX host ABI.
    try device.unary.launch(.{ .x = 1 }, .{ .x = 32 }, .{
        x.ptr, out.ptr, @as(u32, output.len), @as(u32, 7),
    });
    try device.synchronize();
    try download(out, &output);
    try expectApprox(&output, &.{ 0, 0, 0, 1, 2 });
}

fn checkMatmul(device: *engine.Device) !void {
    const a_host = [_]f32{ 1, 2, 3, 4, 5, 6 }; // 2 x 3
    const b_host = [_]f32{ 7, 8, 9, 10, 11, 12 }; // 3 x 2
    var output: [4]f32 = undefined;
    const a = try engine.Device.alloc(Element, a_host.len);
    defer a.free();
    const b = try engine.Device.alloc(Element, b_host.len);
    defer b.free();
    const out = try engine.Device.alloc(Element, output.len);
    defer out.free();
    try upload(a, &a_host);
    try upload(b, &b_host);

    try device.matmul.launch(.{ .x = 1, .y = 1, .z = 1 }, .{ .x = 16, .y = 16 }, .{
        a.ptr,       b.ptr,                  out.ptr,
        @as(u32, 2), @as(u32, 2),            @as(u32, 3),
        @as(u32, 0), @as(u32, 0),            @as(u32, 4),
        @as(u32, 0), engine.driver.null_ptr, @as(u32, 0),
        @as(u32, 0), @as(u32, 1),            @as(u32, 1),
    });
    try device.synchronize();
    try download(out, &output);
    try expectApprox(&output, &.{ 58, 64, 139, 154 });
}

/// The matrix-unit product, where the driver has one. Big enough to cross
/// several of its tiles in every axis and to leave a ragged edge in each, since
/// that is where a tiled kernel goes wrong.
fn checkMatmulSimd(device: *engine.Device) !void {
    const kernel = device.matmul_simd orelse return;
    const m = 70;
    const n = 66;
    const k = 40;

    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const a_host = try allocator.alloc(f32, m * k);
    defer allocator.free(a_host);
    const b_host = try allocator.alloc(f32, k * n);
    defer allocator.free(b_host);
    const bias_host = try allocator.alloc(f32, n);
    defer allocator.free(bias_host);
    for (a_host, 0..) |*value, i| value.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) / 8.0;
    for (b_host, 0..) |*value, i| value.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) / 9.0;
    for (bias_host, 0..) |*value, i| value.* = @as(f32, @floatFromInt(i % 5)) / 4.0;

    const expected = try allocator.alloc(f32, m * n);
    defer allocator.free(expected);
    for (0..m) |row| for (0..n) |col| {
        var total: f32 = bias_host[col];
        for (0..k) |depth| total += a_host[row * k + depth] * b_host[depth * n + col];
        expected[row * n + col] = total;
    };

    const a = try engine.Device.alloc(Element, a_host.len);
    defer a.free();
    const b = try engine.Device.alloc(Element, b_host.len);
    defer b.free();
    const bias = try engine.Device.alloc(Element, bias_host.len);
    defer bias.free();
    const out = try engine.Device.alloc(Element, expected.len);
    defer out.free();
    try uploadAll(allocator, a, a_host);
    try uploadAll(allocator, b, b_host);
    try uploadAll(allocator, bias, bias_host);

    // One tile per work group: the block swizzle is what the executor varies,
    // not what this checks.
    const tiles_m = (m + 63) / 64;
    const tiles_n = (n + 63) / 64;
    try kernel.launch(.{ .x = tiles_m * tiles_n, .z = 1 }, .{ .x = 16, .y = 16 }, .{
        a.ptr,       b.ptr,        out.ptr,
        @as(u32, m), @as(u32, n),  @as(u32, k),
        @as(u32, 0), @as(u32, 0),  @as(u32, 0),
        @as(u32, 0), bias.ptr,     @as(u32, 1),
        @as(u32, 0), @as(u32, 1),  @as(u32, 1),
    });
    try device.synchronize();

    const actual = try allocator.alloc(f32, expected.len);
    defer allocator.free(actual);
    try downloadAll(allocator, out, actual);
    try expectApprox(actual, expected);
}

fn uploadAll(allocator: std.mem.Allocator, buffer: engine.driver.Buffer(Element), source: []const f32) !void {
    const values = try allocator.alloc(Element, source.len);
    defer allocator.free(values);
    for (values, source) |*dst, value| dst.* = @floatCast(value);
    try buffer.upload(values);
}

fn downloadAll(allocator: std.mem.Allocator, buffer: engine.driver.Buffer(Element), destination: []f32) !void {
    const values = try allocator.alloc(Element, destination.len);
    defer allocator.free(values);
    try buffer.download(values);
    for (destination, values) |*dst, value| dst.* = @floatCast(value);
}

fn upload(buffer: engine.driver.Buffer(Element), source: []const f32) !void {
    var values: [128]Element = undefined;
    std.debug.assert(source.len <= values.len);
    for (values[0..source.len], source) |*dst, value| dst.* = @floatCast(value);
    try buffer.upload(values[0..source.len]);
}

fn download(buffer: engine.driver.Buffer(Element), destination: []f32) !void {
    var values: [128]Element = undefined;
    std.debug.assert(destination.len <= values.len);
    try buffer.download(values[0..destination.len]);
    for (destination, values[0..destination.len]) |*dst, value| dst.* = @floatCast(value);
}

fn expectApprox(actual: []const f32, expected: []const f32) !void {
    if (actual.len != expected.len) return error.WrongLength;
    for (actual, expected, 0..) |got, want, i| {
        const delta = @abs(got - want);
        if (delta > 0.02 and delta / @max(@abs(want), 1.0) > 0.01) {
            std.debug.print("mismatch at {d}: {d} != {d}\n", .{ i, got, want });
            return error.WrongResult;
        }
    }
}
