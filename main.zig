const std = @import("std");
const sam3 = @import("sam3");
const web = @import("src/web/server.zig");
const build_options = @import("build_options");

const index_html = @embedFile("src/web/index.html");
const client_wasm = @embedFile("client_wasm");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    std.debug.print("\n=== SAM 3 Web UI ===\n\n", .{});
    std.debug.print("  Model runtime: {s}\n", .{sam3.onnx.version()});

    const tokenizer_json = try std.Io.Dir.cwd().readFileAlloc(
        init.io,
        build_options.concept_tokenizer_path,
        allocator,
        .limited(8 * 1024 * 1024),
    );
    defer allocator.free(tokenizer_json);

    var model = sam3.Model.open(allocator, init.io, .{
        .vision_encoder = build_options.vision_encoder_path,
        .decoder = build_options.decoder_path,
        .concept_vision_encoder = build_options.concept_vision_path,
        .concept_text_encoder = build_options.concept_text_path,
        .concept_decoder = build_options.concept_decoder_path,
        .concept_tokenizer_json = tokenizer_json,
    }) catch |err| {
        const last = sam3.onnx.lastError();
        if (last.len > 0) {
            std.debug.print("Failed to initialize model: {t}: {s}\n", .{ err, last });
        } else {
            std.debug.print("Failed to initialize model: {t}\n", .{err});
        }
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
            .example_path = build_options.example_path,
        },
    );
}
