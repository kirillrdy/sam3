const std = @import("std");
const sam3 = @import("sam3");
const app_mod = @import("app.zig");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    std.debug.print("\n=== SAM 3 Wayland App ===\n\n", .{});
    std.debug.print("  Model runtime: {s}\n", .{sam3.onnx.version()});

    const cache_dir = try sam3.assets.cacheDir(allocator, init.environ_map);
    defer allocator.free(cache_dir);

    var model = try sam3.Model.open(allocator, init.io, cache_dir);
    defer model.deinit();

    const example_path = try sam3.assets.default_assets.cat.get(allocator, init.io, cache_dir);
    defer allocator.free(example_path);

    std.debug.print("  Loaded segmentation and text lookup graphs\n", .{});
    std.debug.print("  Connecting to Wayland display…\n\n", .{});

    var app = app_mod.App.init(allocator, init.io, &model, example_path, 1000, 720) catch |err| {
        std.debug.print("Failed to connect to Wayland display: {t}\n", .{err});
        std.debug.print("Hint: Make sure a Wayland compositor (e.g. Weston) is running and WAYLAND_DISPLAY is set.\n", .{});
        return err;
    };
    defer app.deinit();

    try app.run();
}
