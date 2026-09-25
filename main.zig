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

    var cached = try sam3.assets.cacheAssets(allocator, init.io, init.environ_map, build_options.zig_http);
    defer cached.deinit();

    const tokenizer_json = try std.Io.Dir.cwd().readFileAlloc(
        init.io,
        cached.paths[9],
        allocator,
        .limited(8 * 1024 * 1024),
    );
    defer allocator.free(tokenizer_json);

    var model = sam3.Model.open(allocator, init.io, .{
        .vision_encoder = cached.paths[0],
        .decoder = cached.paths[2],
        .concept_vision_encoder = cached.paths[4],
        .concept_text_encoder = cached.paths[6],
        .concept_decoder = cached.paths[8],
        .concept_tokenizer_json = tokenizer_json,
    }) catch |err| {
        const last = sam3.onnx.lastError();
        std.debug.print("Failed to initialize model: {t}{s}{s}\n", .{ err, if (last.len > 0) ": " else "", last });
        return err;
    };
    defer model.deinit();

    std.debug.print("  Loaded segmentation and text lookup graphs\n\n", .{});

    try web.run(
        allocator,
        init.io,
        &model,
        .{ .index_html = index_html, .client_wasm = client_wasm },
        .{
            .host = build_options.host,
            .port = build_options.port,
            .example_path = cached.paths[10],
        },
    );
}

