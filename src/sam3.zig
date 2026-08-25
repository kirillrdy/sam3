const std = @import("std");
const onnx = @import("onnx.zig");
const resample = @import("resample.zig");
const ImageRGB = @import("io/image.zig").ImageRGB;

pub const image_size: usize = 1008;

pub const Point = @import("point.zig").Point;

pub const Paths = struct {
    vision_encoder: [:0]const u8,
    decoder: [:0]const u8,

    openvino_provider: ?[:0]const u8 = null,

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

const Npu = struct { id: u32, name: []const u8 };

const known_npus = [_]Npu{
    .{ .id = 0x643e, .name = "Lunar Lake" },
    .{ .id = 0xb03e, .name = "Panther Lake" },
    .{ .id = 0xfd3e, .name = "Wildcat Lake" },
};

fn npuName(id: u32) ?[]const u8 {
    for (known_npus) |npu| if (npu.id == id) return npu.name;
    return null;
}

pub const Target = struct {
    device: onnx.DeviceKind = .cpu,

    untested_npu: bool = false,
};

pub const Model = struct {
    allocator: std.mem.Allocator,
    env: onnx.Env,
    vision: onnx.Session,
    decoder: onnx.Session,

    pub fn open(allocator: std.mem.Allocator, paths: Paths, target: Target) !Model {
        try onnx.init();

        const env = try onnx.Env.init("sam3");

        if (paths.openvino_provider) |provider| {
            env.registerProvider("OpenVINO", provider) catch {
                std.debug.print("  ! OpenVINO provider at '{s}' would not load: {s}\n", .{
                    provider,
                    onnx.lastError(),
                });
            };
        }

        var device = if (target.device == .cpu) null else try env.find(target.device);

        var declined = false;

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
                \\    The provider has to be registered (a -Dopenvino build), and whatever the
                \\    OpenVINO plugin dlopens to reach the device has to be on the library path:
                \\{s}
                \\
                \\
            , .{
                target.device,
                switch (target.device) {
                    .npu =>
                    \\    an NPU needs libze_loader.so.1 and libze_intel_npu.so.1, which on NixOS
                    \\    are in two different directories:
                    \\    LD_LIBRARY_PATH=/run/current-system/sw/lib:/run/opengl-driver/lib
                    ,
                    .gpu =>
                    \\    a GPU needs libOpenCL.so.1 -- the ICD loader, which the build compiles
                    \\    and installs beside the provider, so it is zig-out/lib that has to be on
                    \\    the path. The loader then needs a driver to open, which it takes from the
                    \\    ICD registry: NixOS has no /etc/OpenCL/vendors, so name the driver
                    \\    hardware.graphics installed instead --
                    \\    OCL_ICD_FILENAMES=/run/opengl-driver/lib/intel-opencl/libigdrcl.so
                    ,
                    .cpu => "",
                },
            });
        }

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

        const decoder = try onnx.Session.open(env, paths.decoder, null, &.{}, reportFallback);

        return .{ .allocator = allocator, .env = env, .vision = vision, .decoder = decoder };
    }

    fn reportFallback(message: []const u8) void {
        std.debug.print("  ! falling back to the CPU: {s}\n", .{message});
    }

    pub fn deinit(self: *Model) void {
        self.decoder.deinit();
        self.vision.deinit();
    }

    pub fn segment(self: *Model, img: ImageRGB, points: []const Point) !Masks {
        var embedding = try self.encode(img);
        defer embedding.deinit();
        return self.decode(embedding, points);
    }

    pub fn encode(self: *Model, img: ImageRGB) !Embedding {
        const pixels = try preprocess(self.allocator, img);
        defer self.allocator.free(pixels);

        const pixel_shape = [_]i64{ 1, 3, image_size, image_size };
        const pixel_values = try onnx.Value.borrowF32(pixels, &pixel_shape);
        defer pixel_values.deinit();

        var embedding: Embedding = undefined;
        try self.vision.run(
            &.{vision_input},
            &.{pixel_values},
            &embedding_names,
            &embedding.levels,
        );
        return embedding;
    }

    pub fn decode(self: *Model, embedding: Embedding, points: []const Point) !Masks {
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
        inputs[3..].* = embedding.levels;

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

pub const Embedding = struct {
    levels: [embedding_names.len]onnx.Value,

    pub fn deinit(self: *Embedding) void {
        for (self.levels) |level| level.deinit();
        self.* = undefined;
    }
};

pub const Masks = struct {
    allocator: std.mem.Allocator,

    logits: []f32,
    scores: []f32,
    count: usize,
    width: usize,
    height: usize,

    object_score: f32,

    fn take(allocator: std.mem.Allocator, iou: onnx.Value, masks: onnx.Value, object: onnx.Value) !Masks {
        var dims: [8]i64 = undefined;
        const shape = try masks.shape(&dims);

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

    pub fn best(self: Masks) usize {
        var winner: usize = 0;
        for (self.scores, 0..) |score, i| {
            if (score > self.scores[winner]) winner = i;
        }
        return winner;
    }
};

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
