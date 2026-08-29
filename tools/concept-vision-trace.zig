const std = @import("std");
const sam3 = @import("sam3");

const names = [_][*:0]const u8{
    "fpn_hidden_state_0",      "fpn_hidden_state_1",      "fpn_hidden_state_2",      "fpn_hidden_state_3",
    "fpn_position_encoding_0", "fpn_position_encoding_1", "fpn_position_encoding_2", "fpn_position_encoding_3",
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();
    const model_arg = args.next() orelse return error.MissingModel;
    const image_path = args.next() orelse return error.MissingImage;
    const prefix = args.next() orelse return error.MissingOutput;
    const model_path = try allocator.dupeZ(u8, model_arg);
    defer allocator.free(model_path);

    const png = try std.Io.Dir.cwd().readFileAlloc(io, image_path, allocator, .limited(32 * 1024 * 1024));
    defer allocator.free(png);
    var image = try sam3.image.decode(allocator, png);
    defer image.deinit();
    const pixels = try preprocess(allocator, image);
    defer allocator.free(pixels);

    try sam3.onnx.init(allocator, io);
    const env = try sam3.onnx.Env.init(allocator, io, "concept-trace");
    defer env.deinit();
    const session = try sam3.onnx.Session.open(env, model_path, null, &.{}, null);
    defer session.deinit();
    const shape = [_]i64{ 1, 3, sam3.sam3.image_size, sam3.sam3.image_size };
    const input = try sam3.onnx.Value.borrowF32(pixels, &shape);
    defer input.deinit();
    var outputs: [names.len]sam3.onnx.Value = undefined;
    try session.run(&.{"pixel_values"}, &.{input}, &names, &outputs);
    defer for (outputs) |output| output.deinit();

    for (outputs, 0..) |output, index| {
        const values = try output.dataF32();
        const path = try std.fmt.allocPrint(allocator, "{s}.{d}.bin", .{ prefix, index });
        defer allocator.free(path);
        var file = try std.Io.Dir.cwd().createFile(io, path, .{});
        defer file.close(io);
        try file.writeStreamingAll(io, std.mem.sliceAsBytes(values));
        std.debug.print("{d}: {d} values\n", .{ index, values.len });
    }
}

fn preprocess(allocator: std.mem.Allocator, image: sam3.ImageRGB) ![]f32 {
    const size = sam3.sam3.image_size;
    const plane = size * size;
    const out = try allocator.alloc(f32, 3 * plane);
    errdefer allocator.free(out);
    const source = try allocator.alloc(f32, image.width * image.height);
    defer allocator.free(source);
    const mean = [_]f32{ 0.485, 0.456, 0.406 };
    const deviation = [_]f32{ 0.229, 0.224, 0.225 };
    for (0..3) |channel| {
        for (source, 0..) |*value, i| value.* = @as(f32, @floatFromInt(image.data[i * 3 + channel])) / 255.0;
        const resized = try sam3.resample.bilinear(allocator, source, image.width, image.height, size, size);
        defer allocator.free(resized);
        for (out[channel * plane ..][0..plane], resized) |*value, sample| value.* = (sample - mean[channel]) / deviation[channel];
    }
    return out;
}
