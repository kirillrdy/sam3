const std = @import("std");
const sam3 = @import("sam3");
const app_mod = @import("app.zig");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    std.debug.print("\n=== SAM 3 macOS App ===\n\n", .{});
    std.debug.print("  Model runtime: {s}\n", .{sam3.onnx.version()});

    var model = try sam3.assets.loadDefaultModel(allocator, init.io, init.environ_map, false);
    defer model.deinit();

    const cache_dir = try sam3.assets.cacheDir(allocator, init.environ_map);
    defer allocator.free(cache_dir);
    const example_path = try sam3.assets.assets.cat.get(allocator, init.io, cache_dir, false);
    defer allocator.free(example_path);

    std.debug.print("  Loaded segmentation and text lookup graphs\n", .{});
    std.debug.print("  Launching native macOS interface…\n\n", .{});

    var app = app_mod.App.init(allocator, init.io, &model, example_path);
    defer app.deinit();

    try app.start();
}
