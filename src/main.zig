const std = @import("std");
const sam3 = @import("sam3");
const build_options = @import("build_options");

/// Everything this program does, spelled out. There is one released export, one
/// sample image worth defaulting to, and one sensible prompt, so there is
/// nothing left for a command line to decide: `zig build run` segments the cat.
const output_path = "cat_mask.bmp";

/// All baked in by `build.zig`, which is also what downloads the first two:
/// they live in the build cache, so nothing the build fetches lands in the
/// working tree. The provider is empty unless the build was given `-Dopenvino`.
const image_path = build_options.image_path;
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

/// A single foreground click on the cat's chest. Not the image centre: at
/// (0.5, 0.5) the cat's nose is under the cursor, and the model answers that
/// click literally - the highest-scoring of its three hypotheses is the nose,
/// 0.2% of the frame. The PyTorch reference does exactly the same thing there.
const points = [_]sam3.Point{
    .{ .x = 0.46, .y = 0.68, .label = 1 },
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    // A missing export is by far the likeliest failure here, and the fix is a
    // build step away, so say so rather than surfacing FileNotFound from
    // somewhere inside the runtime.
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

    try sam3.cli.runSegment(
        allocator,
        image_path,
        .{
            .vision_encoder = vision_encoder_path,
            .decoder = decoder_path,
            .openvino_provider = openvino_provider_path,
            .cache_dir = openvino_cache_path,
        },
        output_path,
        &points,
        target,
    );
}

/// `build_options` strings are plain slices; every path here crosses into the C
/// API, which wants a sentinel. Both ends are compile-time constants, so the
/// terminator costs a byte of rodata and nothing else.
fn terminate(comptime path: []const u8) [:0]const u8 {
    return (path ++ "\x00")[0..path.len :0];
}
