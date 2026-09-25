const std = @import("std");
const sam3 = @import("sam3");
const app_mod = @import("app.zig");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    std.debug.print("\n=== SAM 3 macOS App ===\n\n", .{});
    std.debug.print("  Model runtime: {s}\n", .{sam3.onnx.version()});

    var loaded = try sam3.assets.loadDefaultModel(allocator, init.io, init.environ_map, false);
    defer loaded.deinit();

    std.debug.print("  Loaded segmentation and text lookup graphs\n", .{});
    std.debug.print("  Launching native macOS interface…\n\n", .{});

    var app = app_mod.App.init(allocator, init.io, &loaded.model, loaded.cached.paths[10]);
    defer app.deinit();

    try app.start();
}
