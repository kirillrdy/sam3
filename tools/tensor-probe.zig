const std = @import("std");
const sam3 = @import("sam3");

pub fn main(init: std.process.Init) !void {
    var arena_state = std.heap.ArenaAllocator.init(init.gpa);
    defer arena_state.deinit();
    const allocator = arena_state.allocator();
    const io = init.io;

    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();
    const model_path = args.next() orelse return error.MissingModel;
    const tokenizer_path = args.next() orelse return error.MissingTokenizer;

    var names: std.ArrayList([*:0]const u8) = .empty;
    while (args.next()) |name| try names.append(allocator, try allocator.dupeZ(u8, name));

    const tokenizer_json = try std.Io.Dir.cwd().readFileAlloc(io, tokenizer_path, allocator, .limited(8 * 1024 * 1024));
    var tokenizer = try sam3.tokenizer.Tokenizer.init(allocator, tokenizer_json);
    defer tokenizer.deinit();
    const encoding = try tokenizer.encode("cat");
    std.debug.print("ids={any}\nmask={any}\n", .{ encoding.ids[0..6], encoding.attention[0..6] });

    try sam3.onnx.init(allocator, io);
    const env = try sam3.onnx.Env.init(allocator, io, "probe");
    defer env.deinit();
    const session = sam3.onnx.Session.open(env, try allocator.dupeZ(u8, model_path), null, &.{}, null) catch |err| {
        std.debug.print("open failed: {s}\n", .{sam3.onnx.lastError()});
        return err;
    };
    defer session.deinit();

    const shape = [_]i64{ 1, sam3.tokenizer.max_tokens };
    const ids = try sam3.onnx.Value.borrowI64(&encoding.ids, &shape);
    defer ids.deinit();
    const attention = try sam3.onnx.Value.borrowI64(&encoding.attention, &shape);
    defer attention.deinit();

    const outputs = try allocator.alloc(sam3.onnx.Value, names.items.len);
    session.run(&.{ "input_ids", "attention_mask" }, &.{ ids, attention }, names.items, outputs) catch |err| {
        std.debug.print("run failed: {s}\n", .{sam3.onnx.lastError()});
        return err;
    };
    defer for (outputs) |out| out.deinit();

    for (names.items, outputs) |name, output| {
        var dims: [8]i64 = undefined;
        const s = try output.shape(&dims);
        const values = try output.dataF32();
        std.debug.print("\n=== {s} shape={any} count={d}\n", .{ name, s, values.len });
        // Print the first few rows of the trailing 32-wide axis.
        const row: usize = if (s.len > 0 and s[s.len - 1] == 32) 32 else @min(values.len, 32);
        var r: usize = 0;
        while (r < @min(values.len / row, 5)) : (r += 1) {
            std.debug.print("  row {d}: ", .{r});
            for (values[r * row ..][0..@min(row, 8)]) |v| std.debug.print("{d:>12.4} ", .{v});
            std.debug.print(" ... ", .{});
            for (values[r * row ..][row - 3 .. row]) |v| std.debug.print("{d:>12.4} ", .{v});
            std.debug.print("\n", .{});
        }
    }
}

