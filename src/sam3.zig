const std = @import("std");
pub const onnx = @import("onnx");
const tokenizer = @import("tokenizer.zig");
const assets_mod = @import("assets.zig");
pub const assets = assets_mod.default_assets;
pub const render = @import("render.zig");
const zigimg = @import("zigimg");
pub const Image = zigimg.Image;
pub const Point = @import("point.zig").Point;
pub const PointLabel = @import("point.zig").Label;

/// Borrowed, tightly packed RGB pixels. The caller retains ownership.
pub const RgbImage = struct {
    pixels: []const u8,
    width: usize,
    height: usize,

    pub fn fromImage(img: Image) RgbImage {
        return .{ .pixels = img.rawBytes(), .width = img.width, .height = img.height };
    }
};

pub const FindOptions = struct {
    min_score: f32 = 0.5,
};

/// Decode file bytes for applications that use zigimg images.
pub fn decodeImage(allocator: std.mem.Allocator, bytes: []const u8) !Image {
    var decoded = try Image.fromMemory(allocator, bytes);
    errdefer decoded.deinit(allocator);
    try decoded.convert(allocator, .rgb24);
    return decoded;
}

const image_size: usize = 1008;

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
    "fpn_position_encoding_2",
};
const concept_decoder_inputs = concept_embedding_names ++ [_][*:0]const u8{
    "text_features",
    "attention_mask",
};
const concept_decoder_outputs = [_][*:0]const u8{
    "pred_masks",
    "pred_logits",
};

pub const Model = struct {
    allocator: std.mem.Allocator,
    env: onnx.Env,
    vision: ?onnx.Session,
    decoder: ?onnx.Session,
    concept_vision: onnx.Session,
    concept_text: onnx.Session,
    concept_decoder: onnx.Session,
    concept_tokenizer: tokenizer.Tokenizer,

    /// Downloads missing default graphs into the cache and opens them.
    pub fn open(allocator: std.mem.Allocator, io: std.Io) !Model {
        return openWithPoints(allocator, io, true);
    }

    /// Opens only text-prompt graphs, omitting the point encoder and decoder.
    pub fn openText(allocator: std.mem.Allocator, io: std.Io) !Model {
        return openWithPoints(allocator, io, false);
    }

    fn openWithPoints(allocator: std.mem.Allocator, io: std.Io, points: bool) !Model {
        const env = try onnx.Env.init(allocator, io);
        errdefer env.deinit();

        const vision: ?onnx.Session = if (points)
            try openSession(allocator, io, env, assets.vision_encoder, assets.vision_encoder_data)
        else
            null;
        errdefer if (vision) |session| session.deinit();

        const decoder: ?onnx.Session = if (points)
            try openSession(allocator, io, env, assets.decoder, assets.decoder_data)
        else
            null;
        errdefer if (decoder) |session| session.deinit();

        const concept_vision = try openSession(allocator, io, env, assets.concept_vision_encoder, assets.concept_vision_encoder_data);
        errdefer concept_vision.deinit();

        const concept_text = try openSession(allocator, io, env, assets.concept_text_encoder, assets.concept_text_encoder_data);
        errdefer concept_text.deinit();

        const concept_decoder = try openSession(allocator, io, env, assets.concept_decoder, null);
        errdefer concept_decoder.deinit();

        const tokenizer_json_path = try assets.concept_tokenizer_json.get(allocator, io);
        defer allocator.free(tokenizer_json_path);
        const tokenizer_json = try std.Io.Dir.cwd().readFileAlloc(
            io,
            tokenizer_json_path,
            allocator,
            .limited(8 * 1024 * 1024),
        );
        defer allocator.free(tokenizer_json);

        const concept_tokenizer = try tokenizer.Tokenizer.init(allocator, tokenizer_json);

        return .{
            .allocator = allocator,
            .env = env,
            .vision = vision,
            .decoder = decoder,
            .concept_vision = concept_vision,
            .concept_text = concept_text,
            .concept_decoder = concept_decoder,
            .concept_tokenizer = concept_tokenizer,
        };
    }

    pub fn deinit(self: *Model) void {
        self.concept_tokenizer.deinit();
        self.concept_decoder.deinit();
        self.concept_text.deinit();
        self.concept_vision.deinit();
        if (self.decoder) |session| session.deinit();
        if (self.vision) |session| session.deinit();
        self.env.deinit();
    }

    /// The returned embedding owns its memory and can serve multiple point prompts.
    pub fn encodePoints(self: *Model, img: RgbImage) !PointEmbedding {
        const vision = self.vision orelse return error.PointModelsNotLoaded;
        return .{ .levels = try encodeImage(self.allocator, vision, img, @splat(0.5), @splat(0.5), &embedding_names) };
    }

    pub fn segment(self: *Model, embedding: *const PointEmbedding, points: []const Point) !Masks {
        const coordinates = try self.allocator.alloc(f32, points.len * 2);
        defer self.allocator.free(coordinates);
        const labels = try self.allocator.alloc(i64, points.len);
        defer self.allocator.free(labels);

        const scale: f32 = @floatFromInt(image_size);
        for (points, 0..) |p, i| {
            coordinates[i * 2] = p.x * scale;
            coordinates[i * 2 + 1] = p.y * scale;
            labels[i] = @backingInt(p.label);
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
        const decoder = self.decoder orelse return error.PointModelsNotLoaded;
        try decoder.run(
            &decoder_inputs,
            &inputs,
            &decoder_outputs,
            &results,
        );
        defer for (results) |r| r.deinit();

        return Masks.take(self.allocator, results[0], results[1], results[2]);
    }

    /// The returned embedding owns its memory and can serve multiple text prompts.
    pub fn encodeForText(self: *Model, img: RgbImage) !TextImageEmbedding {
        return .{ .levels = try encodeImage(
            self.allocator,
            self.concept_vision,
            img,
            .{ 0.485, 0.456, 0.406 },
            .{ 0.229, 0.224, 0.225 },
            &concept_embedding_names,
        ) };
    }

    /// Caller owns the returned text encoder features and can cache them by phrase.
    pub fn encodeTextFeatures(self: *Model, phrase: []const u8) ![]f32 {
        const encoding = try self.concept_tokenizer.encode(phrase);
        return encodeConceptText(self, encoding);
    }

    pub fn find(self: *Model, embedding: *const TextImageEmbedding, phrase: []const u8, options: FindOptions) !Masks {
        return decodeConceptQuery(self, embedding, phrase, null, options.min_score);
    }

    /// Use text encoder features from `encodeTextFeatures`, possibly from a cache.
    pub fn findWithTextFeatures(self: *Model, embedding: *const TextImageEmbedding, phrase: []const u8, features: []const f32, options: FindOptions) !Masks {
        return decodeConceptQuery(self, embedding, phrase, features, options.min_score);
    }
};

fn DeviceEmbedding(comptime count: usize) type {
    return struct {
        levels: [count]onnx.Value,

        pub fn deinit(self: *@This()) void {
            for (self.levels) |level| level.deinit();
            self.* = undefined;
        }
    };
}

pub const PointEmbedding = DeviceEmbedding(embedding_names.len);
pub const TextImageEmbedding = DeviceEmbedding(concept_embedding_names.len);

fn openSession(
    allocator: std.mem.Allocator,
    io: std.Io,
    env: onnx.Env,
    model_asset: assets_mod.Asset,
    data_asset: ?assets_mod.Asset,
) !onnx.Session {
    if (data_asset) |asset| {
        const data = try asset.get(allocator, io);
        allocator.free(data);
    }
    const path = try model_asset.get(allocator, io);
    defer allocator.free(path);
    return onnx.Session.open(env, path);
}

fn encodeImage(
    allocator: std.mem.Allocator,
    session: onnx.Session,
    img: RgbImage,
    mean: [3]f32,
    deviation: [3]f32,
    comptime output_names: []const [*:0]const u8,
) ![output_names.len]onnx.Value {
    const pixels = try preprocessCpu(allocator, img, mean, deviation);
    defer allocator.free(pixels);

    const pixel_shape = [_]i64{ 1, 3, image_size, image_size };
    const pixel_values = try onnx.Value.borrowF32(pixels, &pixel_shape);
    defer pixel_values.deinit();

    var levels: [output_names.len]onnx.Value = undefined;
    try session.runDevice(&.{vision_input}, &.{pixel_values}, output_names, &levels);
    return levels;
}

fn encodeConceptText(model: *Model, encoding: tokenizer.Encoding) ![]f32 {
    const token_shape = [_]i64{ 1, tokenizer.max_tokens };
    const ids = try onnx.Value.borrowI64(&encoding.ids, &token_shape);
    defer ids.deinit();
    const attention = try onnx.Value.borrowI64(&encoding.attention, &token_shape);
    defer attention.deinit();

    var text_features: [1]onnx.Value = undefined;
    try model.concept_text.run(
        &.{ "input_ids", "attention_mask" },
        &.{ ids, attention },
        &.{"text_features"},
        &text_features,
    );
    defer text_features[0].deinit();

    return model.allocator.dupe(f32, try text_features[0].dataF32());
}

fn decodeConceptQuery(
    model: *Model,
    embedding: *const TextImageEmbedding,
    phrase: []const u8,
    features: ?[]const f32,
    threshold: f32,
) !Masks {
    const encoding = try model.concept_tokenizer.encode(phrase);
    const owned_features: ?[]f32 = if (features == null) try encodeConceptText(model, encoding) else null;
    defer if (owned_features) |data| model.allocator.free(data);

    const data = features orelse owned_features.?;
    if (data.len != tokenizer.max_tokens * 256) return error.InvalidTextFeatures;
    const text_value = try onnx.Value.borrowF32(data, &.{ 1, tokenizer.max_tokens, 256 });
    defer text_value.deinit();

    const token_shape = [_]i64{ 1, tokenizer.max_tokens };
    const attention = try onnx.Value.borrowI64(&encoding.attention, &token_shape);
    defer attention.deinit();

    const inputs = [_]onnx.Value{
        embedding.levels[0],
        embedding.levels[1],
        embedding.levels[2],
        embedding.levels[3],
        text_value,
        attention,
    };
    var results: [concept_decoder_outputs.len]onnx.Value = undefined;
    try model.concept_decoder.run(
        &concept_decoder_inputs,
        &inputs,
        &concept_decoder_outputs,
        &results,
    );
    defer for (results) |result| result.deinit();

    return Masks.takeConcept(model.allocator, results[0], results[1], threshold);
}

pub const Masks = struct {
    allocator: std.mem.Allocator,

    logits: []f32,
    scores: []f32,
    count: usize,
    width: usize,
    height: usize,

    object_score: f32,

    /// The borrowed logits for one mask, in row-major order at width x height.
    pub fn plane(self: *const Masks, index: usize) []const f32 {
        std.debug.assert(index < self.count);
        const stride = self.width * self.height;
        return self.logits[index * stride ..][0..stride];
    }

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
};

fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

fn max(values: []const f32) f32 {
    var result = values[0];
    for (values[1..]) |value| result = @max(result, value);
    return result;
}

/// Resizes and normalizes an RGB image into the planar float tensor expected by the vision encoders.
pub fn preprocessCpu(allocator: std.mem.Allocator, img: RgbImage, mean: [3]f32, deviation: [3]f32) ![]f32 {
    if (img.width == 0 or img.height == 0) return error.InvalidImage;
    const pixel_count = std.math.mul(usize, img.width, img.height) catch return error.InvalidImage;
    const byte_count = std.math.mul(usize, pixel_count, 3) catch return error.InvalidImage;
    if (img.pixels.len != byte_count) return error.InvalidImage;

    const pixels = img.pixels;
    const plane_size = image_size * image_size;
    const out = try allocator.alloc(f32, 3 * plane_size);
    errdefer allocator.free(out);

    const ratio_y = @as(f32, @floatFromInt(img.height)) / @as(f32, @floatFromInt(image_size));
    const ratio_x = @as(f32, @floatFromInt(img.width)) / @as(f32, @floatFromInt(image_size));
    const byte_scale: f32 = 1.0 / 255.0;

    // Resize all three interleaved byte channels together and write directly
    // into the model's planar normalized tensor.
    for (0..image_size) |y| {
        const sy = render.axisSample(y, ratio_y, img.height);
        const row0 = pixels[sy.lo * img.width * 3 ..][0 .. img.width * 3];
        const row1 = pixels[sy.hi * img.width * 3 ..][0 .. img.width * 3];

        for (0..image_size) |x| {
            const sx = render.axisSample(x, ratio_x, img.width);
            const p00 = row0[sx.lo * 3 ..][0..3];
            const p01 = row0[sx.hi * 3 ..][0..3];
            const p10 = row1[sx.lo * 3 ..][0..3];
            const p11 = row1[sx.hi * 3 ..][0..3];
            const index = y * image_size + x;

            inline for (0..3) |channel| {
                const resized = render.bilerp(
                    @floatFromInt(p00[channel]),
                    @floatFromInt(p01[channel]),
                    @floatFromInt(p10[channel]),
                    @floatFromInt(p11[channel]),
                    sx.w,
                    sy.w,
                ) * byte_scale;
                out[channel * plane_size + index] = (resized - mean[channel]) / deviation[channel];
            }
        }
    }
    return out;
}

test "RGB image view validates input and preserves channel order" {
    const allocator = std.testing.allocator;
    const rgb = [_]u8{ 255, 128, 0 };
    const pixels = try preprocessCpu(allocator, .{
        .pixels = &rgb,
        .width = 1,
        .height = 1,
    }, @splat(0), @splat(1));
    defer allocator.free(pixels);

    const plane_size = image_size * image_size;
    try std.testing.expectEqual(@as(f32, 1), pixels[0]);
    try std.testing.expectApproxEqAbs(@as(f32, 128.0 / 255.0), pixels[plane_size], 0.00001);
    try std.testing.expectEqual(@as(f32, 0), pixels[2 * plane_size]);
    try std.testing.expectError(error.InvalidImage, preprocessCpu(allocator, .{
        .pixels = rgb[0..2],
        .width = 1,
        .height = 1,
    }, @splat(0), @splat(1)));
}

test "mask planes borrow the corresponding logits" {
    var masks: Masks = .{
        .allocator = std.testing.allocator,
        .logits = try std.testing.allocator.dupe(f32, &.{ 1, 2, 3, 4 }),
        .scores = try std.testing.allocator.dupe(f32, &.{ 0.8, 0.9 }),
        .count = 2,
        .width = 2,
        .height = 1,
        .object_score = 0.9,
    };
    defer masks.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 3, 4 }, masks.plane(1));
}

test {
    _ = tokenizer;
    std.testing.refAllDecls(@This());
}
