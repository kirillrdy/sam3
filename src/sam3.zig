const std = @import("std");
const builtin = @import("builtin");
pub const onnx = @import("runtime");
pub const resample = @import("resample.zig");
pub const tokenizer = @import("tokenizer.zig");
pub const zigimg = @import("zigimg");
pub const Image = zigimg.Image;

pub fn decode(allocator: std.mem.Allocator, bytes: []const u8) !Image {
    var decoded = try Image.fromMemory(allocator, bytes);
    errdefer decoded.deinit(allocator);
    try decoded.convert(allocator, .rgb24);
    return decoded;
}

pub const image_size: usize = 1008;

pub const Point = struct {
    x: f32,
    y: f32,
    label: i64 = 1,
};

pub const Paths = struct {
    vision_encoder: []const u8,
    decoder: []const u8,
    concept_vision_encoder: []const u8,
    concept_text_encoder: []const u8,
    concept_decoder: []const u8,
    concept_tokenizer_json: []const u8,
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

pub const Model = struct {
    allocator: std.mem.Allocator,
    env: onnx.Env,
    vision: onnx.Session,
    decoder: onnx.Session,
    concept_vision: onnx.Session,
    concept_text: onnx.Session,
    concept_decoder: onnx.Session,
    concept_tokenizer: tokenizer.Tokenizer,

    pub fn open(allocator: std.mem.Allocator, io: std.Io, paths: Paths) !Model {
        try onnx.init(allocator, io);

        const env = try onnx.Env.init(allocator, io);
        errdefer env.deinit();

        const vision = try onnx.Session.open(env, paths.vision_encoder);
        errdefer vision.deinit();

        const decoder = try onnx.Session.open(env, paths.decoder);
        errdefer decoder.deinit();

        const concept_vision = try onnx.Session.open(env, paths.concept_vision_encoder);
        errdefer concept_vision.deinit();

        const concept_text = try onnx.Session.open(env, paths.concept_text_encoder);
        errdefer concept_text.deinit();

        const concept_decoder = try onnx.Session.open(env, paths.concept_decoder);
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
        };
    }

    pub fn deinit(self: *Model) void {
        self.concept_tokenizer.deinit();
        self.concept_decoder.deinit();
        self.concept_text.deinit();
        self.concept_vision.deinit();
        self.decoder.deinit();
        self.vision.deinit();
        self.env.deinit();
    }

    pub fn encode(self: *Model, img: Image) !Embedding {
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

    pub fn encodeConcept(self: *Model, img: Image) !ConceptEmbedding {
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

    fn preprocess(self: *Model, img: Image) ![]f32 {
        return self.preprocessNormalized(img, @splat(0.5), @splat(0.5));
    }

    fn preprocessConcept(self: *Model, img: Image) ![]f32 {
        return self.preprocessNormalized(
            img,
            .{ 0.485, 0.456, 0.406 },
            .{ 0.229, 0.224, 0.225 },
        );
    }

    fn preprocessNormalized(self: *Model, img: Image, mean: [3]f32, deviation: [3]f32) ![]f32 {
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
fn preprocessCpu(allocator: std.mem.Allocator, img: Image, mean: [3]f32, deviation: [3]f32) ![]f32 {
    const pixels = img.pixels.rgb24;
    const plane_size = image_size * image_size;
    const out = try allocator.alloc(f32, 3 * plane_size);
    errdefer allocator.free(out);

    const source = try allocator.alloc(f32, img.width * img.height);
    defer allocator.free(source);

    for (0..3) |channel| {
        for (source, pixels) |*v, px| {
            const byte: u8 = switch (channel) {
                0 => px.r,
                1 => px.g,
                2 => px.b,
                else => unreachable,
            };
            v.* = @as(f32, @floatFromInt(byte)) / 255.0;
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

test {
    std.testing.refAllDecls(@This());
}
