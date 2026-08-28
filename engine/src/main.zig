//! Prints what an ONNX file contains: shapes, weights, and the operator mix.
//! Useful for deciding what the executor still has to implement.

const std = @import("std");
const onnx = @import("onnx.zig");

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;

    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, gpa);
    defer args.deinit();
    _ = args.next();
    const path = args.next() orelse {
        std.debug.print("usage: onnx-dump MODEL.onnx [NODE-COUNT]\n", .{});
        std.process.exit(2);
    };
    const node_limit = if (args.next()) |text| try std.fmt.parseInt(usize, text, 10) else 0;

    var graph = try onnx.Graph.open(gpa, init.io, path);
    defer graph.deinit();

    std.debug.print("\n{s}\n\n", .{path});
    for (graph.inputs) |input| {
        // Graph inputs include the initializers in some exports; those are
        // weights, not things a caller has to supply.
        if (graph.constant(input.name) != null) continue;
        std.debug.print("  in   {s}: {t} {any}\n", .{ input.name, input.dtype, input.dims });
    }
    for (graph.outputs) |output| {
        std.debug.print("  out  {s}: {t} {any}\n", .{ output.name, output.dtype, output.dims });
    }

    var weight_bytes: usize = 0;
    for (graph.initializers) |tensor| weight_bytes += tensor.data.len;

    std.debug.print("\n  {d} nodes, {d} initializers, {d:.1} MiB of weights\n\n", .{
        graph.nodes.len,
        graph.initializers.len,
        @as(f64, @floatFromInt(weight_bytes)) / (1 << 20),
    });

    var counts: std.StringHashMapUnmanaged(usize) = .empty;
    defer counts.deinit(gpa);
    for (graph.nodes) |node| {
        const entry = try counts.getOrPut(gpa, node.op_type);
        entry.value_ptr.* = if (entry.found_existing) entry.value_ptr.* + 1 else 1;
    }

    const Entry = struct { name: []const u8, count: usize };
    var entries: std.ArrayList(Entry) = .empty;
    defer entries.deinit(gpa);
    var it = counts.iterator();
    while (it.next()) |e| try entries.append(gpa, .{ .name = e.key_ptr.*, .count = e.value_ptr.* });
    std.mem.sort(Entry, entries.items, {}, struct {
        fn lessThan(_: void, a: Entry, b: Entry) bool {
            return a.count > b.count;
        }
    }.lessThan);

    std.debug.print("  operators ({d} kinds):\n", .{entries.items.len});
    for (entries.items) |e| std.debug.print("    {d:>6}  {s}\n", .{ e.count, e.name });
    std.debug.print("\n", .{});

    for (graph.nodes[0..@min(node_limit, graph.nodes.len)], 0..) |node, i| {
        std.debug.print("  [{d}] {s} {s}\n", .{ i, node.op_type, node.name });
        for (node.inputs) |name| {
            if (graph.constant(name)) |tensor| {
                std.debug.print("      in  {s}: {t} {any} (constant)\n", .{ name, tensor.dtype, tensor.dims });
                if (tensor.elementCount() <= 16) switch (tensor.dtype) {
                    .i64 => std.debug.print("          {any}\n", .{tensor.i64s()}),
                    .f32 => std.debug.print("          {any}\n", .{tensor.f32s()}),
                    else => {},
                };
            } else {
                std.debug.print("      in  {s}\n", .{name});
            }
        }
        for (node.outputs) |name| std.debug.print("      out {s}\n", .{name});
        for (node.attributes) |attribute| {
            std.debug.print("      @{s}: i={d} f={d} ints={any} s={s}\n", .{
                attribute.name,
                attribute.i,
                attribute.f,
                attribute.ints,
                attribute.s,
            });
        }
    }
}
