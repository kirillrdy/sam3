const std = @import("std");
const sam3 = @import("sam3");
const web = @import("web/server.zig");
const build_options = @import("build_options");

const index_html = @embedFile("web/index.html");
const client_wasm = @embedFile("client_wasm");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    std.debug.print("\n=== SAM 3 Web UI ===\n\n", .{});
    std.debug.print("  Model runtime: {s}\n", .{sam3.onnx.version()});

    const cache_dir = try sam3.assets.cacheDir(allocator, init.environ_map);
    defer allocator.free(cache_dir);

    var model = try sam3.Model.open(allocator, init.io, cache_dir);
    defer model.deinit();

    const example_path = try sam3.assets.default_assets.cat.get(allocator, init.io, cache_dir);
    defer allocator.free(example_path);

    std.debug.print("  Loaded segmentation and text lookup graphs\n\n", .{});

    try web.run(
        allocator,
        init.io,
        &model,
        .{ .index_html = index_html, .client_wasm = client_wasm },
        .{
            .host = build_options.host,
            .port = build_options.port,
            .example_path = example_path,
        },
    );
}

