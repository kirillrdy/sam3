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

    var loaded = try sam3.assets.loadDefaultModel(allocator, init.io, init.environ_map, build_options.zig_http);
    defer loaded.deinit();

    std.debug.print("  Loaded segmentation and text lookup graphs\n\n", .{});

    try web.run(
        allocator,
        init.io,
        &loaded.model,
        .{ .index_html = index_html, .client_wasm = client_wasm },
        .{
            .host = build_options.host,
            .port = build_options.port,
            .example_path = loaded.cached.paths[10],
        },
    );
}

