const std = @import("std");
const sam3 = @import("sam3");

const input_names = [_][*:0]const u8{
    "fpn_hidden_state_0",      "fpn_hidden_state_1", "fpn_hidden_state_2",
    "fpn_position_encoding_2", "text_features",      "attention_mask",
};
const output_names = [_][*:0]const u8{
    "/detr_decoder/layers.0/text_cross_attn/Add_output_0",
    "/detr_decoder/layers.0/text_cross_attn/Softmax_output_0",
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();
    const model_arg = args.next() orelse return error.MissingModel;
    const vision_prefix = args.next() orelse return error.MissingVision;
    const text_path = args.next() orelse return error.MissingText;
    const output_prefix = args.next() orelse return error.MissingOutput;
    const model_path = try allocator.dupeZ(u8, model_arg);
    defer allocator.free(model_path);

    var data: [5][]align(@alignOf(f32)) u8 = undefined;
    const suffixes = [_][]const u8{ "0", "1", "2", "6" };
    for (suffixes, 0..) |suffix, i| {
        const path = try std.fmt.allocPrint(allocator, "{s}.{s}.bin", .{ vision_prefix, suffix });
        defer allocator.free(path);
        data[i] = try std.Io.Dir.cwd().readFileAllocOptions(io, path, allocator, .unlimited, .of(f32), null);
    }
    data[4] = try std.Io.Dir.cwd().readFileAllocOptions(io, text_path, allocator, .unlimited, .of(f32), null);
    defer for (data) |bytes| allocator.free(bytes);

    try sam3.onnx.init(allocator, io);
    const dims = [_][]const i64{
        &.{ 1, 256, 288, 288 }, &.{ 1, 256, 144, 144 }, &.{ 1, 256, 72, 72 },
        &.{ 1, 256, 72, 72 },   &.{ 1, 32, 256 },
    };
    var values: [6]sam3.onnx.Value = undefined;
    for (data, dims, values[0..5]) |bytes, shape, *value| value.* = try sam3.onnx.Value.borrowF32(std.mem.bytesAsSlice(f32, bytes), shape);
    defer for (values[0..5]) |value| value.deinit();
    var attention: [32]i64 = @splat(0);
    attention[0] = 1;
    attention[1] = 1;
    attention[2] = 1;
    values[5] = try sam3.onnx.Value.borrowI64(&attention, &.{ 1, 32 });
    defer values[5].deinit();

    const env = try sam3.onnx.Env.init(allocator, io, "decoder-trace");
    defer env.deinit();
    const session = sam3.onnx.Session.open(env, model_path, null, &.{}, null) catch |err| {
        std.debug.print("open failed: {s}\n", .{sam3.onnx.lastError()});
        return err;
    };
    defer session.deinit();
    var outputs: [output_names.len]sam3.onnx.Value = undefined;
    try session.run(&input_names, &values, &output_names, &outputs);
    defer for (outputs) |output| output.deinit();

    for (outputs, 0..) |output, index| {
        const floats = try output.dataF32();
        const path = try std.fmt.allocPrint(allocator, "{s}.{d}.bin", .{ output_prefix, index });
        defer allocator.free(path);
        var file = try std.Io.Dir.cwd().createFile(io, path, .{});
        defer file.close(io);
        try file.writeStreamingAll(io, std.mem.sliceAsBytes(floats));
        std.debug.print("{d}: {d} values", .{ index, floats.len });
        var shape_buffer: [8]i64 = undefined;
        std.debug.print(", shape={any}", .{try output.shape(&shape_buffer)});
        std.debug.print("\n", .{});
    }
}
