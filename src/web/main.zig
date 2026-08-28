const std = @import("std");
const sam3 = @import("sam3");
const build_options = @import("build_options");

const index_html = @embedFile("index.html");
const client_wasm = @embedFile("client_wasm");

const vision_encoder_path = terminate(build_options.vision_encoder_path);
const decoder_path = terminate(build_options.decoder_path);
const concept_vision_path = terminate(build_options.concept_vision_path);
const concept_text_path = terminate(build_options.concept_text_path);
const concept_decoder_path = terminate(build_options.concept_decoder_path);
const concept_tokenizer_path = terminate(build_options.concept_tokenizer_path);
const openvino_provider_path = terminateOptional(build_options.openvino_provider_path);
const webgpu_provider_path = terminateOptional(build_options.webgpu_provider_path);
const provider_cache_path = terminateOptional(build_options.provider_cache_path);

const target: sam3.Target = .{
    .device = std.meta.stringToEnum(sam3.DeviceKind, build_options.device).?,
    .untested_npu = build_options.untested_npu,
    .cuda = build_options.cuda,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    for ([_][:0]const u8{
        vision_encoder_path,
        decoder_path,
        concept_vision_path,
        concept_text_path,
        concept_decoder_path,
        concept_tokenizer_path,
    }) |path| {
        std.Io.Dir.cwd().access(init.io, path, .{}) catch {
            std.debug.print(
                \\Error: no model at '{s}'.
                \\
                \\Fetch it with `zig build fetch-weights fetch-concept-weights`.
                \\
                \\
            , .{path});
            std.process.exit(1);
        };
    }

    std.debug.print("\n=== SAM 3 Web UI ===\n\n", .{});
    std.debug.print("  Model runtime: {s}\n", .{sam3.onnx.version()});

    const tokenizer_json = try std.Io.Dir.cwd().readFileAlloc(
        init.io,
        concept_tokenizer_path,
        allocator,
        .limited(8 * 1024 * 1024),
    );
    defer allocator.free(tokenizer_json);

    var model = try sam3.Model.open(allocator, init.io, .{
        .vision_encoder = vision_encoder_path,
        .decoder = decoder_path,
        .concept_vision_encoder = concept_vision_path,
        .concept_text_encoder = concept_text_path,
        .concept_decoder = concept_decoder_path,
        .concept_tokenizer_json = tokenizer_json,
        .openvino_provider = openvino_provider_path,
        .webgpu_provider = webgpu_provider_path,
        .cache_dir = provider_cache_path,
    }, target);
    defer model.deinit();

    std.debug.print("  Loaded both graphs\n", .{});
    std.debug.print("    vision encoder -> {t}\n", .{model.vision.device});
    std.debug.print("    mask decoder   -> {t}\n\n", .{model.decoder.device});
    std.debug.print("  Loaded text lookup graphs\n", .{});
    std.debug.print("    vision encoder -> {t}\n", .{model.concept_vision.device});
    std.debug.print("    text encoder   -> {t}\n", .{model.concept_text.device});
    std.debug.print("    object decoder -> {t}\n\n", .{model.concept_decoder.device});

    try sam3.web.run(
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

fn terminate(comptime path: []const u8) [:0]const u8 {
    return (path ++ "\x00")[0..path.len :0];
}

fn terminateOptional(comptime path: []const u8) ?[:0]const u8 {
    return if (path.len == 0) null else terminate(path);
}
