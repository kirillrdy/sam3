//! `zig build serve`: the same model the CLI runs, behind a browser.
//!
//! The executable carries the whole UI -- the page and the wasm module are
//! embedded -- so there is nothing to serve from a directory and nothing to
//! install. As with `sam3` itself, the build is what says where the model and
//! the sample image are; this takes no arguments.

const std = @import("std");
const sam3 = @import("sam3");
const build_options = @import("build_options");

const index_html = @embedFile("index.html");
const client_wasm = @embedFile("client_wasm");

const vision_encoder_path = terminate(build_options.vision_encoder_path);
const decoder_path = terminate(build_options.decoder_path);
const openvino_provider_path: ?[:0]const u8 = if (build_options.openvino_provider_path.len == 0)
    null
else
    terminate(build_options.openvino_provider_path);
const openvino_cache_path: ?[:0]const u8 = if (build_options.openvino_cache_path.len == 0)
    null
else
    terminate(build_options.openvino_cache_path);

/// Where to run, from `-Ddevice` and `-Duntested-npu`.
const target: sam3.Target = .{
    .device = std.meta.stringToEnum(sam3.DeviceKind, build_options.device).?,
    .untested_npu = build_options.untested_npu,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    for ([_][:0]const u8{ vision_encoder_path, decoder_path }) |path| {
        if (std.os.linux.access(path.ptr, std.os.linux.F_OK) != 0) {
            std.debug.print(
                \\Error: no model at '{s}'.
                \\
                \\Fetch it with `zig build fetch-weights`.
                \\
                \\
            , .{path});
            std.process.exit(1);
        }
    }

    std.debug.print("\n=== SAM 3 Web UI ===\n\n", .{});
    std.debug.print("  ONNX Runtime: {s}\n", .{sam3.onnx.version()});

    // Both graphs are loaded before the socket is opened, so the first click
    // waits on the encoder rather than on a 1.7 GiB file being read.
    var model = try sam3.Model.open(allocator, .{
        .vision_encoder = vision_encoder_path,
        .decoder = decoder_path,
        .openvino_provider = openvino_provider_path,
        .cache_dir = openvino_cache_path,
    }, target);
    defer model.deinit();

    std.debug.print("  Loaded both graphs\n", .{});
    std.debug.print("    vision encoder -> {t}\n", .{model.vision.device});
    std.debug.print("    mask decoder   -> {t}\n\n", .{model.decoder.device});

    try sam3.web.run(
        allocator,
        init.io,
        &model,
        .{ .index_html = index_html, .client_wasm = client_wasm },
        .{
            .host = build_options.host,
            .port = build_options.port,
            .example_path = if (build_options.image_path.len == 0) null else build_options.image_path,
        },
    );
}

/// `build_options` strings are plain slices; every path here crosses into the C
/// API, which wants a sentinel.
fn terminate(comptime path: []const u8) [:0]const u8 {
    return (path ++ "\x00")[0..path.len :0];
}
