//! SAM 3's tracker branch, run through ONNX Runtime.
//!
//! Meta's checkpoint is exported as two graphs and this drives both of them:
//! the vision encoder turns a frame into a three-level feature pyramid, and the
//! prompt encoder and mask decoder turn that pyramid plus a click into mask
//! logits. The pyramid crosses between them as the runtime's own tensors, so
//! the 100 MB of features are never copied back through here.
//!
//! Nothing in this file knows about devices beyond asking `onnx` for one. The
//! interesting consequence of the split is that the two graphs are sized very
//! differently -- 1.7 GiB of encoder against 21 MiB of decoder -- so they do not
//! always land on the same one.

const std = @import("std");
const onnx = @import("onnx.zig");
const resample = @import("resample.zig");
const ImageRGB = @import("io/image.zig").ImageRGB;

/// What the export was traced at. The vision encoder's position embeddings are
/// built for this grid, so it is not a resolution the caller gets to pick.
pub const image_size: usize = 1008;

/// A click, in coordinates normalised to the frame. `label` is 1 for a point
/// the mask should include and 0 for one it should exclude.
pub const Point = struct {
    x: f32,
    y: f32,
    label: i64 = 1,
};

/// Where the two exported graphs and the OpenVINO provider live on disk.
pub const Paths = struct {
    vision_encoder: [:0]const u8,
    decoder: [:0]const u8,
    /// `libonnxruntime_providers_openvino.so`, or null for a build without it,
    /// in which case only the CPU is available.
    openvino_provider: ?[:0]const u8 = null,
    /// Where the provider may keep graphs it has already compiled for the
    /// device. Null asks it not to, which costs the compile on every run.
    cache_dir: ?[:0]const u8 = null,
};

const vision_input = "pixel_values";
const embedding_names = [_][*:0]const u8{
    "image_embeddings.0",
    "image_embeddings.1",
    "image_embeddings.2",
};
const decoder_inputs = [_][*:0]const u8{
    "input_points",
    "input_labels",
    "input_boxes",
} ++ embedding_names;
const decoder_outputs = [_][*:0]const u8{
    "iou_scores",
    "pred_masks",
    "object_score_logits",
};

/// An Intel NPU generation, as the PCI id the part reports.
const Npu = struct { id: u32, name: []const u8 };

/// The NPU generations this is known to run on.
///
/// The vision encoder is a 1.7 GiB graph, and it has only ever been measured on
/// Lunar Lake's NPU 4. Meteor Lake and Arrow Lake carry NPU 3 -- around a third
/// of the throughput, and less room to compile a graph this size into -- and
/// nothing here has been near one, so they are left out rather than guessed
/// about. `Target.untested_npu` overrides that for anyone who wants to find out.
///
/// A list rather than a `>=` because the ids are not ordered by generation:
/// Meteor Lake is 0x7d1d and Lunar Lake, two years later, is 0x643e.
const known_npus = [_]Npu{
    .{ .id = 0x643e, .name = "Lunar Lake" },
    .{ .id = 0xb03e, .name = "Panther Lake" },
    .{ .id = 0xfd3e, .name = "Wildcat Lake" },
};

fn npuName(id: u32) ?[]const u8 {
    for (known_npus) |npu| if (npu.id == id) return npu.name;
    return null;
}

/// Where to run the model, and how insistent to be about it.
pub const Target = struct {
    device: onnx.DeviceKind = .cpu,
    /// Use an NPU whose generation is not in `known_npus`. Off by default,
    /// because finding out costs a quarter of an hour of compiling before it
    /// can even fail.
    untested_npu: bool = false,
};

pub const Model = struct {
    allocator: std.mem.Allocator,
    env: onnx.Env,
    vision: onnx.Session,
    decoder: onnx.Session,

    /// Loads both graphs, on `target` where that device will take them. A graph
    /// a device turns down falls back to the CPU on its own -- see
    /// `onnx.Session.open` -- so this succeeds as long as the files are
    /// readable.
    pub fn open(allocator: std.mem.Allocator, paths: Paths, target: Target) !Model {
        try onnx.init();

        // Never released, here or in `deinit`, and that is deliberate: see the
        // comment there.
        const env = try onnx.Env.init("sam3");

        // The provider is a library the runtime dlopens rather than links, so
        // until it is registered the NPU is not in the device list at all.
        if (paths.openvino_provider) |provider| {
            env.registerProvider("OpenVINO", provider) catch {
                std.debug.print("  ! OpenVINO provider at '{s}' would not load: {s}\n", .{
                    provider,
                    onnx.lastError(),
                });
            };
        }

        var device = if (target.device == .cpu) null else try env.find(target.device);
        // Set when the device was there and this turned it down, so the "no
        // device" advice below does not also fire and blame the library path.
        var declined = false;

        // An NPU is only worth the compile on a part this has been run on. The
        // runtime hands over the PCI id, which is what says which generation
        // turned up.
        if (device) |d| {
            if (d.kind == .npu) {
                if (npuName(d.id)) |name| {
                    std.debug.print("  NPU:          Intel {s} (0x{x:0>4})\n", .{ name, d.id });
                } else if (target.untested_npu) {
                    std.debug.print("  NPU:          unrecognised generation 0x{x:0>4}, trying it anyway\n", .{d.id});
                } else {
                    std.debug.print(
                        \\  ! the NPU here (0x{x:0>4}) is a generation this has not been run on,
                        \\    so this runs on the CPU. Only Lunar Lake and newer are in the list;
                        \\    Meteor Lake and Arrow Lake carry a third of the throughput and have
                        \\    never been tried. Build with -Duntested-npu to use it regardless.
                        \\
                        \\
                    , .{d.id});
                    device = null;
                    declined = true;
                }
            }
        }

        if (target.device != .cpu and device == null and !declined) {
            std.debug.print(
                \\  ! no {t} among the devices the runtime enumerated, so this runs on the CPU.
                \\    The provider has to be registered (a -Dopenvino build), and the device
                \\    reachable: an Intel NPU needs libze_loader.so.1 and libze_intel_npu.so.1
                \\    on the library path. On NixOS those are in two different directories:
                \\    LD_LIBRARY_PATH=/run/opengl-driver/lib:/run/current-system/sw/lib
                \\
                \\
            , .{target.device});
        }

        // Compiling a graph for an NPU takes minutes, and the result depends on
        // nothing but the graph and the device -- so the provider is given
        // somewhere to keep it, and every run after the first skips straight to
        // execution. The V2 entry point refuses the EP's own `cache_dir` and
        // wants this instead: OpenVINO's per-device property map, as JSON.
        var config_buf: [512]u8 = undefined;
        var option_buf: [1]onnx.Option = undefined;
        var options: []const onnx.Option = &.{};
        if (device != null) if (paths.cache_dir) |dir| {
            const config = try std.fmt.bufPrintZ(
                &config_buf,
                "{{\"{s}\":{{\"CACHE_DIR\":\"{s}\"}}}}",
                .{ target.device.openvinoName(), dir },
            );
            option_buf[0] = .{ .key = "load_config", .value = config.ptr };
            options = option_buf[0..1];
        };

        const vision = try onnx.Session.open(env, paths.vision_encoder, device, options, reportFallback);
        errdefer vision.deinit();

        // The mask decoder stays off the NPU. Intel's NPU compiler takes SIGFPE
        // compiling this graph -- inside its own one-time initialisation, on
        // its own thread, with nothing of ours on the stack -- and a crash
        // inside a prebuilt compiler is not something the fallback in
        // `Session.open` can catch. It costs little: the decoder is 21 MiB of
        // weights against the encoder's 1.7 GiB, and the encoder is where the
        // time goes. Every other device takes it.
        const decoder_device = if (target.device == .npu) null else device;
        const decoder = try onnx.Session.open(
            env,
            paths.decoder,
            decoder_device,
            if (decoder_device == null) &.{} else options,
            reportFallback,
        );

        return .{ .allocator = allocator, .env = env, .vision = vision, .decoder = decoder };
    }

    fn reportFallback(message: []const u8) void {
        std.debug.print("  ! falling back to the CPU: {s}\n", .{message});
    }

    /// Releases the sessions but deliberately not the environment. Releasing
    /// it unregisters the execution provider library, which dlcloses it and the
    /// OpenVINO stack under it -- and Intel's compiler libraries down there have
    /// by then registered `atexit` handlers. Unload them and `exit` jumps into
    /// unmapped memory. The environment lives as long as the process either
    /// way, so keeping it is the whole of the fix.
    pub fn deinit(self: *Model) void {
        self.decoder.deinit();
        self.vision.deinit();
    }

    /// Runs both graphs over one frame. The caller owns the result.
    pub fn segment(self: *Model, img: ImageRGB, points: []const Point) !Masks {
        const pixels = try preprocess(self.allocator, img);
        defer self.allocator.free(pixels);

        const pixel_shape = [_]i64{ 1, 3, image_size, image_size };
        const pixel_values = try onnx.Value.borrowF32(pixels, &pixel_shape);
        defer pixel_values.deinit();

        var embeddings: [embedding_names.len]onnx.Value = undefined;
        try self.vision.run(
            &.{vision_input},
            &.{pixel_values},
            &embedding_names,
            &embeddings,
        );
        defer for (embeddings) |e| e.deinit();

        // The decoder wants pixels of the encoder's own input, not of the frame.
        const coordinates = try self.allocator.alloc(f32, points.len * 2);
        defer self.allocator.free(coordinates);
        const labels = try self.allocator.alloc(i64, points.len);
        defer self.allocator.free(labels);

        const scale: f32 = @floatFromInt(image_size);
        for (points, 0..) |p, i| {
            coordinates[i * 2] = p.x * scale;
            coordinates[i * 2 + 1] = p.y * scale;
            labels[i] = p.label;
        }

        const point_count: i64 = @intCast(points.len);
        const point_shape = [_]i64{ 1, 1, point_count, 2 };
        const label_shape = [_]i64{ 1, 1, point_count };

        // The graph takes both prompt kinds and picks between them, so a
        // point-only call still has to pass boxes -- as an empty tensor, which
        // is what the export's own no-box path is written against.
        const no_boxes: [4]f32 = @splat(0.0);
        const box_shape = [_]i64{ 1, 0, 4 };

        const input_points = try onnx.Value.borrowF32(coordinates, &point_shape);
        defer input_points.deinit();
        const input_labels = try onnx.Value.borrowI64(labels, &label_shape);
        defer input_labels.deinit();
        const input_boxes = try onnx.Value.borrowF32(no_boxes[0..0], &box_shape);
        defer input_boxes.deinit();

        var inputs: [decoder_inputs.len]onnx.Value = undefined;
        inputs[0..3].* = .{ input_points, input_labels, input_boxes };
        inputs[3..].* = embeddings;

        var results: [decoder_outputs.len]onnx.Value = undefined;
        try self.decoder.run(
            &decoder_inputs,
            &inputs,
            &decoder_outputs,
            &results,
        );
        defer for (results) |r| r.deinit();

        return Masks.take(self.allocator, results[0], results[1], results[2]);
    }
};

/// The decoder's answer to one click: several competing mask hypotheses -- the
/// part, the subpart and the whole -- each with the IoU the model predicts for
/// it, at the decoder's own resolution.
pub const Masks = struct {
    allocator: std.mem.Allocator,
    /// `count` planes of `width` x `height` logits, one after another.
    logits: []f32,
    scores: []f32,
    count: usize,
    width: usize,
    height: usize,
    /// The model's confidence that the click landed on an object at all, as a
    /// logit: positive is present.
    object_score: f32,

    fn take(allocator: std.mem.Allocator, iou: onnx.Value, masks: onnx.Value, object: onnx.Value) !Masks {
        var dims: [8]i64 = undefined;
        const shape = try masks.shape(&dims);
        // [batch, prompt, hypothesis, height, width]
        std.debug.assert(shape.len == 5);

        const count: usize = @intCast(shape[2]);
        const height: usize = @intCast(shape[3]);
        const width: usize = @intCast(shape[4]);

        const logits = try allocator.dupe(f32, try masks.dataF32());
        errdefer allocator.free(logits);
        const scores = try allocator.dupe(f32, (try iou.dataF32())[0..count]);

        return .{
            .allocator = allocator,
            .logits = logits,
            .scores = scores,
            .count = count,
            .width = width,
            .height = height,
            .object_score = (try object.dataF32())[0],
        };
    }

    pub fn deinit(self: *Masks) void {
        self.allocator.free(self.logits);
        self.allocator.free(self.scores);
    }

    pub fn plane(self: Masks, index: usize) []const f32 {
        const stride = self.width * self.height;
        return self.logits[index * stride ..][0..stride];
    }

    /// The hypothesis the model rates highest, which is the one to show when
    /// only one mask is wanted.
    pub fn best(self: Masks) usize {
        var winner: usize = 0;
        for (self.scores, 0..) |score, i| {
            if (score > self.scores[winner]) winner = i;
        }
        return winner;
    }
};

/// Frame to `pixel_values`: RGB bytes to planar f32, resized to the encoder's
/// input and normalised the way `Sam3ImageProcessorFast` does -- rescale by
/// 1/255, then mean and standard deviation of 0.5, which is a plain map onto
/// [-1, 1].
fn preprocess(allocator: std.mem.Allocator, img: ImageRGB) ![]f32 {
    const plane_size = image_size * image_size;
    const out = try allocator.alloc(f32, 3 * plane_size);
    errdefer allocator.free(out);

    const source = try allocator.alloc(f32, img.width * img.height);
    defer allocator.free(source);

    for (0..3) |channel| {
        for (source, 0..) |*v, i| {
            v.* = @as(f32, @floatFromInt(img.data[i * 3 + channel])) / 255.0;
        }

        const resized = try resample.bilinear(
            allocator,
            source,
            img.width,
            img.height,
            image_size,
            image_size,
        );
        defer allocator.free(resized);

        const target = out[channel * plane_size ..][0..plane_size];
        for (target, resized) |*v, r| v.* = (r - 0.5) / 0.5;
    }
    return out;
}
