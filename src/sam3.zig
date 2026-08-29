const std = @import("std");
const builtin = @import("builtin");
/// Selected by build.zig. The current implementation is ONNX Runtime; the
/// native CUDA engine implements this same session/value boundary.
pub const onnx = @import("runtime");
const resample = @import("resample.zig");
const tokenizer = @import("tokenizer.zig");
const ImageRGB = @import("io/image.zig").ImageRGB;
const gpu = @import("gpu");

pub const image_size: usize = 1008;

pub const Point = @import("point.zig").Point;

pub const Paths = struct {
    vision_encoder: [:0]const u8,
    decoder: [:0]const u8,
    concept_vision_encoder: [:0]const u8,
    concept_text_encoder: [:0]const u8,
    concept_decoder: [:0]const u8,
    concept_tokenizer_json: []const u8,

    openvino_provider: ?[:0]const u8 = null,
    webgpu_provider: ?[:0]const u8 = null,

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

const concept_embedding_names = [_][*:0]const u8{
    "fpn_hidden_state_0",
    "fpn_hidden_state_1",
    "fpn_hidden_state_2",
    "fpn_hidden_state_3",
    "fpn_position_encoding_0",
    "fpn_position_encoding_1",
    "fpn_position_encoding_2",
    "fpn_position_encoding_3",
};
const concept_decoder_inputs = [_][*:0]const u8{
    "fpn_hidden_state_0",
    "fpn_hidden_state_1",
    "fpn_hidden_state_2",
    "fpn_position_encoding_2",
    "text_features",
    "attention_mask",
};
const concept_decoder_outputs = [_][*:0]const u8{
    "pred_masks",
    "pred_boxes",
    "pred_logits",
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

    /// Preprocess images on a GPU rather than the CPU.
    cuda: bool = false,
    gpu: bool = false,
};

fn registerProvider(env: onnx.Env, name: [:0]const u8, path: ?[:0]const u8) void {
    const provider = path orelse return;
    env.registerProvider(name, provider) catch {
        std.debug.print("  ! {s} provider at '{s}' would not load: {s}\n", .{
            name,
            provider,
            onnx.lastError(),
        });
    };
}

pub const Model = struct {
    allocator: std.mem.Allocator,
    env: onnx.Env,
    vision: onnx.Session,
    decoder: onnx.Session,
    concept_vision: onnx.Session,
    concept_text: onnx.Session,
    concept_decoder: onnx.Session,
    concept_tokenizer: tokenizer.Tokenizer,

    /// Null when preprocessing runs on the CPU.
    preprocessor: ?gpu.Preprocessor,

    pub fn open(allocator: std.mem.Allocator, io: std.Io, paths: Paths, target: Target) !Model {
        try onnx.init(allocator, io);

        const env = try onnx.Env.init(allocator, io, "sam3");
        errdefer env.deinit();

        registerProvider(env, "OpenVINO", paths.openvino_provider);
        registerProvider(env, "webgpu", paths.webgpu_provider);

        var accelerator: ?onnx.Accelerator = if (target.device == .cpu)
            null
        else if (builtin.os.tag.isDarwin() and target.device != .webgpu)
            .{ .coreml = target.device }
        else if (try env.find(target.device)) |device|
            .{ .device = device }
        else if ((target.device == .gpu or target.device == .npu) and paths.openvino_provider != null)
            .{ .openvino = target.device }
        else
            null;

        var declined = false;

        if (accelerator) |selected| {
            if (selected == .device and selected.device.kind == .npu) {
                if (npuName(selected.device.id)) |name| {
                    std.debug.print("  NPU:          Intel {s} (0x{x:0>4})\n", .{ name, selected.device.id });
                } else if (target.untested_npu) {
                    std.debug.print("  NPU:          unrecognised generation 0x{x:0>4}, trying it anyway\n", .{selected.device.id});
                } else {
                    std.debug.print(
                        \\  ! the NPU here (0x{x:0>4}) is a generation this has not been run on,
                        \\    so this runs on the CPU. Only Lunar Lake and newer are in the list;
                        \\    Meteor Lake and Arrow Lake carry a third of the throughput and have
                        \\    never been tried. Build with -Duntested-npu to use it regardless.
                        \\
                        \\
                    , .{selected.device.id});
                    accelerator = null;
                    declined = true;
                }
            }
        }

        if (target.device != .cpu and accelerator == null and !declined) {
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
                    .webgpu =>
                    \\    WebGPU needs a working Vulkan driver. On Linux, Dawn selects a Vulkan
                    \\    adapter, so verify the GPU with `vulkaninfo --summary`.
                    ,
                    .cpu => "",
                },
            });
        }

        var config_buf: [512]u8 = undefined;
        var option_buf: [8]onnx.Option = undefined;
        var options: []const onnx.Option = &.{};
        if (accelerator) |selected| switch (selected) {
            .device => if (target.device != .webgpu) {
                if (paths.cache_dir) |dir| {
                    const config = try std.fmt.bufPrintZ(
                        &config_buf,
                        "{{\"{s}\":{{\"CACHE_DIR\":\"{s}\"}}}}",
                        .{ target.device.openvinoName(), dir },
                    );
                    option_buf[0] = .{ .key = "load_config", .value = config.ptr };
                    options = option_buf[0..1];
                }
            },
            .openvino => if (paths.cache_dir) |dir| {
                const config = try std.fmt.bufPrintZ(
                    &config_buf,
                    "{{\"{s}\":{{\"CACHE_DIR\":\"{s}\"}}}}",
                    .{ target.device.openvinoName(), dir },
                );
                option_buf[0] = .{ .key = "load_config", .value = config.ptr };
                options = option_buf[0..1];
            },
            .coreml => |device| {
                option_buf[0] = .{
                    .key = "MLComputeUnits",
                    .value = switch (device) {
                        .npu => "CPUAndNeuralEngine",
                        .gpu => "CPUAndGPU",
                        .webgpu => unreachable,
                        .cpu => unreachable,
                    },
                };
                option_buf[1] = .{ .key = "ModelFormat", .value = "MLProgram" };
                option_buf[2] = .{ .key = "RequireStaticInputShapes", .value = "1" };
                option_buf[3] = .{ .key = "AllowLowPrecisionAccumulationOnGPU", .value = "1" };
                var count: usize = 4;
                if (paths.cache_dir) |dir| {
                    const cache_dir = try std.fmt.bufPrintZ(&config_buf, "{s}", .{dir});
                    option_buf[count] = .{ .key = "ModelCacheDirectory", .value = cache_dir.ptr };
                    count += 1;
                }
                options = option_buf[0..count];
            },
        };

        const vision = try onnx.Session.open(env, paths.vision_encoder, accelerator, options, reportFallback);
        errdefer vision.deinit();

        const decoder = try onnx.Session.open(env, paths.decoder, null, &.{}, reportFallback);
        errdefer decoder.deinit();

        const concept_vision = try onnx.Session.open(env, paths.concept_vision_encoder, accelerator, options, reportFallback);
        errdefer concept_vision.deinit();
        // The text graph has a large weight set for a tiny input, so moving it
        // to the GPU costs more VRAM than it saves time. The smaller decoder
        // benefits from the selected accelerator.
        const concept_text = try onnx.Session.open(env, paths.concept_text_encoder, null, &.{}, reportFallback);
        errdefer concept_text.deinit();
        const concept_decoder = try onnx.Session.open(env, paths.concept_decoder, accelerator, options, reportFallback);
        errdefer concept_decoder.deinit();
        const concept_tokenizer = try tokenizer.Tokenizer.init(allocator, paths.concept_tokenizer_json);

        return .{
            .allocator = allocator,
            .env = env,
            .vision = vision,
            .decoder = decoder,
            .concept_vision = concept_vision,
            .concept_text = concept_text,
            .concept_decoder = concept_decoder,
            .concept_tokenizer = concept_tokenizer,
            .preprocessor = if (target.cuda or target.gpu) openPreprocessor() else null,
        };
    }

    /// The GPU only handles preprocessing, so failing to reach it costs a few
    /// milliseconds rather than the run: say so and stay on the CPU.
    fn openPreprocessor() ?gpu.Preprocessor {
        if (!gpu.available) {
            std.debug.print("  ! this build has no GPU preprocessing support, so images are preprocessed on the CPU.\n", .{});
            return null;
        }
        var preprocessor = gpu.Preprocessor.init(image_size) catch {
            std.debug.print("  ! no GPU device for preprocessing, so it runs on the CPU: {s}\n", .{gpu.lastError()});
            return null;
        };
        var name_buf: [128]u8 = undefined;
        std.debug.print("  GPU:          {s}\n", .{preprocessor.deviceName(&name_buf) catch "GPU"});
        return preprocessor;
    }

    fn reportFallback(message: []const u8) void {
        std.debug.print("  ! falling back to the CPU: {s}\n", .{message});
    }

    pub fn deinit(self: *Model) void {
        if (self.preprocessor) |*preprocessor| preprocessor.deinit();
        self.concept_tokenizer.deinit();
        self.concept_decoder.deinit();
        self.concept_text.deinit();
        self.concept_vision.deinit();
        self.decoder.deinit();
        self.vision.deinit();
        self.env.deinit();
    }

    pub fn segment(self: *Model, img: ImageRGB, points: []const Point) !Masks {
        var embedding = try self.encode(img);
        defer embedding.deinit();
        return self.decode(embedding, points);
    }

    pub fn encode(self: *Model, img: ImageRGB) !Embedding {
        const pixels = try self.preprocess(img);
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

    pub fn encodeConcept(self: *Model, img: ImageRGB) !ConceptEmbedding {
        const pixels = try self.preprocessConcept(img);
        defer self.allocator.free(pixels);

        const pixel_shape = [_]i64{ 1, 3, image_size, image_size };
        const pixel_values = try onnx.Value.borrowF32(pixels, &pixel_shape);
        defer pixel_values.deinit();

        var embedding: ConceptEmbedding = undefined;
        try self.concept_vision.run(
            &.{vision_input},
            &.{pixel_values},
            &concept_embedding_names,
            &embedding.levels,
        );
        return embedding;
    }

    pub fn lookup(self: *Model, embedding: ConceptEmbedding, phrase: []const u8, threshold: f32) !Masks {
        const encoding = try self.concept_tokenizer.encode(phrase);
        const token_shape = [_]i64{ 1, tokenizer.max_tokens };
        const ids = try onnx.Value.borrowI64(&encoding.ids, &token_shape);
        defer ids.deinit();
        const attention = try onnx.Value.borrowI64(&encoding.attention, &token_shape);
        defer attention.deinit();

        var text_features: [1]onnx.Value = undefined;
        try self.concept_text.run(
            &.{ "input_ids", "attention_mask" },
            &.{ ids, attention },
            &.{"text_features"},
            &text_features,
        );
        defer text_features[0].deinit();

        const inputs = [_]onnx.Value{
            embedding.levels[0],
            embedding.levels[1],
            embedding.levels[2],
            embedding.levels[6],
            text_features[0],
            attention,
        };
        var results: [concept_decoder_outputs.len]onnx.Value = undefined;
        try self.concept_decoder.run(
            &concept_decoder_inputs,
            &inputs,
            &concept_decoder_outputs,
            &results,
        );
        defer for (results) |result| result.deinit();
        return Masks.takeConcept(self.allocator, results[0], results[2], threshold);
    }

    /// Public so `zig build compare` can run the two preprocessing paths
    /// against each other; the pipeline calls it on its own.
    pub fn preprocess(self: *Model, img: ImageRGB) ![]f32 {
        return self.preprocessNormalized(img, @splat(0.5), @splat(0.5));
    }

    fn preprocessConcept(self: *Model, img: ImageRGB) ![]f32 {
        return self.preprocessNormalized(
            img,
            .{ 0.485, 0.456, 0.406 },
            .{ 0.229, 0.224, 0.225 },
        );
    }

    /// Resizes and normalizes an image into the tensor the encoder takes, on
    /// the GPU when one was opened for it. A GPU that fails part way through a
    /// session hands the work back to the CPU for the rest of it.
    fn preprocessNormalized(self: *Model, img: ImageRGB, mean: [3]f32, deviation: [3]f32) ![]f32 {
        if (self.preprocessor) |*preprocessor| {
            const out = try self.allocator.alloc(f32, 3 * image_size * image_size);
            if (preprocessor.run(img.data, img.width, img.height, mean, deviation, out)) {
                return out;
            } else |err| {
                self.allocator.free(out);
                std.debug.print("  ! CUDA preprocessing failed ({t}), moving it to the CPU: {s}\n", .{
                    err,
                    gpu.lastError(),
                });
                preprocessor.deinit();
                self.preprocessor = null;
            }
        }
        return preprocessCpu(self.allocator, img, mean, deviation);
    }
};

pub const Embedding = struct {
    levels: [embedding_names.len]onnx.Value,

    pub fn deinit(self: *Embedding) void {
        for (self.levels) |level| level.deinit();
        self.* = undefined;
    }
};

pub const ConceptEmbedding = struct {
    levels: [concept_embedding_names.len]onnx.Value,

    pub fn deinit(self: *ConceptEmbedding) void {
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

    fn takeConcept(allocator: std.mem.Allocator, masks: onnx.Value, scores_value: onnx.Value, threshold: f32) !Masks {
        var dims: [8]i64 = undefined;
        const shape = try masks.shape(&dims);
        if (shape.len != 4 or shape[0] != 1) return error.UnexpectedConceptMaskShape;

        const available: usize = @intCast(shape[1]);
        const height: usize = @intCast(shape[2]);
        const width: usize = @intCast(shape[3]);
        const planes = try masks.dataF32();
        const raw_scores = try scores_value.dataF32();
        if (raw_scores.len < available) return error.UnexpectedConceptScoreShape;

        var count: usize = 0;
        for (raw_scores[0..available]) |logit| {
            const score = sigmoid(logit);
            if (score >= threshold) count += 1;
        }

        const logits = try allocator.alloc(f32, count * width * height);
        errdefer allocator.free(logits);
        const scores = try allocator.alloc(f32, count);
        errdefer allocator.free(scores);

        const stride = width * height;
        var out: usize = 0;
        for (raw_scores[0..available], 0..) |logit, i| {
            const score = sigmoid(logit);
            if (score < threshold) continue;
            scores[out] = score;
            @memcpy(logits[out * stride ..][0..stride], planes[i * stride ..][0..stride]);
            out += 1;
        }

        return .{
            .allocator = allocator,
            .logits = logits,
            .scores = scores,
            .count = count,
            .width = width,
            .height = height,
            .object_score = if (count == 0) 0 else max(scores),
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
        if (self.count == 0) return 0;
        var winner: usize = 0;
        for (self.scores, 0..) |score, i| {
            if (score > self.scores[winner]) winner = i;
        }
        return winner;
    }
};

fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

fn max(values: []const f32) f32 {
    var result = values[0];
    for (values[1..]) |value| result = @max(result, value);
    return result;
}

/// The CPU path, and the reference the CUDA kernel is checked against.
fn preprocessCpu(allocator: std.mem.Allocator, img: ImageRGB, mean: [3]f32, deviation: [3]f32) ![]f32 {
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
        for (target, resized) |*v, r| v.* = (r - mean[channel]) / deviation[channel];
    }
    return out;
}
