const std = @import("std");
const sam3 = @import("sam3");
const build_options = @import("build_options");

/// Everything this program does, spelled out. There is one released checkpoint,
/// one sample image worth defaulting to, and one sensible prompt, so there is
/// nothing left for a command line to decide: `zig build run` segments the cat.
const output_path = "cat_mask.bmp";

/// Both baked in by `build.zig`, which is also what downloads them: they live in
/// the build cache, so nothing the build fetches lands in the working tree.
const image_path = build_options.image_path;
const weights_path = build_options.weights_path;

/// The 1008x1008 the model was trained and exported at.
const image_size: usize = 1008;

/// A single foreground click on the cat's chest. Not the image centre: at
/// (0.5, 0.5) the cat's nose is under the cursor, and the model answers that
/// click literally - the highest-scoring of its three hypotheses is the nose,
/// 0.2% of the frame. The PyTorch reference does exactly the same thing there.
const points = [_]sam3.Point{
    .{ .x = 0.46, .y = 0.68, .label = 1 },
};

pub fn main(init: std.process.Init) !void {
    // Every large tensor in the pass - the MLP hidden state is 94 MiB - is
    // allocated and dropped once per layer, at a size that goes straight to
    // `mmap` rather than into any allocator's bins. Recycling those blocks
    // instead of returning them keeps the pass from spending its time faulting
    // the same pages back in thirty-two times over.
    var tensor_pool = sam3.pool.Pool.init(init.gpa);
    defer tensor_pool.deinit();
    const allocator = tensor_pool.allocator();

    // A missing checkpoint is by far the likeliest failure here, and the fix is
    // a build step away, so say so rather than surfacing FileNotFound.
    if (!sam3.safetensors.exists(weights_path)) {
        std.debug.print(
            \\Error: no checkpoint at '{s}'.
            \\
            \\Fetch it with `zig build fetch-weights`.
            \\
            \\
        , .{weights_path});
        std.process.exit(1);
    }

    try sam3.cli.runSegment(
        allocator,
        image_path,
        weights_path,
        output_path,
        &points,
        image_size,
        true,
    );
}
