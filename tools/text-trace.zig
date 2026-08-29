const std = @import("std");
const sam3 = @import("sam3");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();
    const model_path = args.next() orelse return error.MissingModel;
    const tokenizer_path = args.next() orelse return error.MissingTokenizer;
    const output_path = args.next() orelse return error.MissingOutput;

    const tokenizer_json = try std.Io.Dir.cwd().readFileAlloc(io, tokenizer_path, allocator, .limited(8 * 1024 * 1024));
    defer allocator.free(tokenizer_json);
    var tokenizer = try sam3.tokenizer.Tokenizer.init(allocator, tokenizer_json);
    defer tokenizer.deinit();
    const encoding = try tokenizer.encode("cat");

    try sam3.onnx.init(allocator, io);
    const env = try sam3.onnx.Env.init(allocator, io, "text-trace");
    defer env.deinit();
    const model_path_z = try allocator.dupeZ(u8, model_path);
    defer allocator.free(model_path_z);
    const session = try sam3.onnx.Session.open(env, model_path_z, null, &.{}, null);
    defer session.deinit();

    const shape = [_]i64{ 1, sam3.tokenizer.max_tokens };
    const ids = try sam3.onnx.Value.borrowI64(&encoding.ids, &shape);
    defer ids.deinit();
    const attention = try sam3.onnx.Value.borrowI64(&encoding.attention, &shape);
    defer attention.deinit();
    var outputs: [1]sam3.onnx.Value = undefined;
    try session.run(&.{ "input_ids", "attention_mask" }, &.{ ids, attention }, &.{"text_features"}, &outputs);
    defer outputs[0].deinit();
    const values = try outputs[0].dataF32();

    var file = try std.Io.Dir.cwd().createFile(io, output_path, .{});
    defer file.close(io);
    try file.writeStreamingAll(io, std.mem.sliceAsBytes(values));

    var min: f32 = std.math.inf(f32);
    var max: f32 = -std.math.inf(f32);
    var sum: f64 = 0;
    for (values) |value| {
        min = @min(min, value);
        max = @max(max, value);
        sum += value;
    }
    std.debug.print("{d} values, min={d} max={d} mean={d}\n", .{ values.len, min, max, sum / @as(f64, @floatFromInt(values.len)) });
}
