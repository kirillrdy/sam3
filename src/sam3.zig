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
    const inv_deviation: [3]f32 = .{
        1.0 / deviation[0],
        1.0 / deviation[1],
        1.0 / deviation[2],
    };

    var x_samples: [image_size]render.AxisSample = undefined;
    for (&x_samples, 0..) |*sx, x| {
        sx.* = render.axisSample(x, ratio_x, img.width);
    }

    // Resize all three interleaved byte channels together and write directly
    // into the model's planar normalized tensor.
    for (0..image_size) |y| {
        const sy = render.axisSample(y, ratio_y, img.height);
        const row0 = pixels[sy.lo * img.width * 3 ..][0 .. img.width * 3];
        const row1 = pixels[sy.hi * img.width * 3 ..][0 .. img.width * 3];

        for (x_samples, 0..) |sx, x| {
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
                out[channel * plane_size + index] = (resized - mean[channel]) * inv_deviation[channel];
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

test "Masks.take extracts point decoder masks, scores, and object score" {
    const allocator = std.testing.allocator;
    const iou_data = [_]f32{ 0.42, 0.91 };
    const mask_data = [_]f32{
        -1.0, 2.0,
        3.0,  -4.0,
        5.0,  6.0,
        -7.0, 8.0,
    };
    const obj_data = [_]f32{0.85};

    const iou = try onnx.Value.borrowF32(&iou_data, &.{ 1, 1, 2 });
    defer iou.deinit();
    const mask_val = try onnx.Value.borrowF32(&mask_data, &.{ 1, 1, 2, 2, 2 });
    defer mask_val.deinit();
    const obj = try onnx.Value.borrowF32(&obj_data, &.{ 1, 1 });
    defer obj.deinit();

    var masks = try Masks.take(allocator, iou, mask_val, obj);
    defer masks.deinit();

    try std.testing.expectEqual(@as(usize, 2), masks.count);
    try std.testing.expectEqual(@as(usize, 2), masks.width);
    try std.testing.expectEqual(@as(usize, 2), masks.height);
    try std.testing.expectEqual(@as(f32, 0.85), masks.object_score);
    try std.testing.expectEqualSlices(f32, &iou_data, masks.scores);
    try std.testing.expectEqualSlices(f32, &.{ -1.0, 2.0, 3.0, -4.0 }, masks.plane(0));
    try std.testing.expectEqualSlices(f32, &.{ 5.0, 6.0, -7.0, 8.0 }, masks.plane(1));
}

test "Masks.takeConcept filters masks by sigmoid score threshold" {
    const allocator = std.testing.allocator;
    const mask_data = [_]f32{
        1.0,  2.0,
        3.0,  4.0,
        -1.0, -2.0,
        5.0,  6.0,
        7.0,  8.0,
        9.0,  10.0,
    };
    // sigmoid(2.0) ~= 0.8808, sigmoid(0.0) = 0.5, sigmoid(-2.0) ~= 0.1192
    const logits_data = [_]f32{ 2.0, -2.0, 0.0 };

    const mask_val = try onnx.Value.borrowF32(&mask_data, &.{ 1, 3, 2, 2 });
    defer mask_val.deinit();
    const score_val = try onnx.Value.borrowF32(&logits_data, &.{ 1, 3 });
    defer score_val.deinit();

    var kept = try Masks.takeConcept(allocator, mask_val, score_val, 0.5);
    defer kept.deinit();

    try std.testing.expectEqual(@as(usize, 2), kept.count);
    try std.testing.expectEqualSlices(f32, &.{ 1.0, 2.0, 3.0, 4.0 }, kept.plane(0));
    try std.testing.expectEqualSlices(f32, &.{ 7.0, 8.0, 9.0, 10.0 }, kept.plane(1));
    try std.testing.expectApproxEqAbs(sigmoid(2.0), kept.object_score, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), kept.scores[1], 1e-6);

    var empty = try Masks.takeConcept(allocator, mask_val, score_val, 0.99);
    defer empty.deinit();
    try std.testing.expectEqual(@as(usize, 0), empty.count);
    try std.testing.expectEqual(@as(f32, 0.0), empty.object_score);

    const bad_shape = try onnx.Value.borrowF32(&mask_data, &.{ 2, 3, 2, 1 });
    defer bad_shape.deinit();
    try std.testing.expectError(
        error.UnexpectedConceptMaskShape,
        Masks.takeConcept(allocator, bad_shape, score_val, 0.5),
    );
}

test "main sam3 image path: decode, preprocess, score, and composite" {
    const allocator = std.testing.allocator;

    // 2x2 RGB image encoded as PNG in memory, then decoded via decodeImage.
    var src = try Image.create(allocator, 2, 2, .rgb24);
    defer src.deinit(allocator);
    src.pixels.rgb24[0] = .{ .r = 255, .g = 0, .b = 0 };
    src.pixels.rgb24[1] = .{ .r = 0, .g = 255, .b = 0 };
    src.pixels.rgb24[2] = .{ .r = 0, .g = 0, .b = 255 };
    src.pixels.rgb24[3] = .{ .r = 255, .g = 255, .b = 255 };

    var png_buf: [1024]u8 = undefined;
    const png_bytes = try src.writeToMemory(allocator, &png_buf, .{ .png = .{} });

    var decoded = try decodeImage(allocator, png_bytes);
    defer decoded.deinit(allocator);

    const rgb = RgbImage.fromImage(decoded);
    const tensor = try preprocessCpu(allocator, rgb, .{ 0.5, 0.5, 0.5 }, .{ 0.5, 0.5, 0.5 });
    defer allocator.free(tensor);

    const plane_size = image_size * image_size;
    try std.testing.expectEqual(@as(usize, 3 * plane_size), tensor.len);
    // Top-left corner maps to pixel (0,0) = (255, 0, 0) -> normalized (1.0, -1.0, -1.0).
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), tensor[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), tensor[plane_size], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), tensor[2 * plane_size], 1e-5);

    const mask_logits = [_]f32{ 1.0, -1.0, -1.0, -1.0 };
    const scores = [_]f32{0.95};
    var coverages: [1]f32 = undefined;
    const best = render.scoreMasks(&mask_logits, &scores, 1, 2, 2, &coverages);
    try std.testing.expectEqual(@as(usize, 0), best);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), coverages[0], 1e-6);

    var rgba: [2 * 2 * 4]u8 = undefined;
    render.compositeRgba(decoded, &rgba, &mask_logits, 2, 2, &.{});
    // Pixel (0,0) is tinted with mask_color (0, 220, 100) at alpha 0.5 over (255, 0, 0).
    try std.testing.expectEqualSlices(u8, &.{ 127, 110, 50, 255 }, rgba[0..4]);
    // Pixel (1,0) is untinted (0, 255, 0).
    try std.testing.expectEqualSlices(u8, &.{ 0, 255, 0, 255 }, rgba[4..8]);
}

test {
    _ = tokenizer;
    std.testing.refAllDecls(@This());
}
