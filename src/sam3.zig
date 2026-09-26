const std = @import("std");
pub const onnx = @import("onnx");
const tokenizer = @import("tokenizer.zig");
pub const assets = @import("assets.zig").default_assets;
pub const render = @import("render.zig");
const zigimg = @import("zigimg");
pub const Image = zigimg.Image;

pub fn decode(allocator: std.mem.Allocator, bytes: []const u8) !Image {
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

    pub fn open(allocator: std.mem.Allocator, io: std.Io) !Model {
        const env = try onnx.Env.init(allocator, io);
        errdefer env.deinit();

        const vision_data = try assets.vision_encoder_data.get(allocator, io);
        defer allocator.free(vision_data);
        const vision_path = try assets.vision_encoder.get(allocator, io);
        defer allocator.free(vision_path);
        const vision = try onnx.Session.open(env, vision_path);
        errdefer vision.deinit();

        const decoder_data = try assets.decoder_data.get(allocator, io);
        defer allocator.free(decoder_data);
        const decoder_path = try assets.decoder.get(allocator, io);
        defer allocator.free(decoder_path);
        const decoder = try onnx.Session.open(env, decoder_path);
        errdefer decoder.deinit();

        const concept_vision_data = try assets.concept_vision_encoder_data.get(allocator, io);
        defer allocator.free(concept_vision_data);
        const concept_vision_path = try assets.concept_vision_encoder.get(allocator, io);
        defer allocator.free(concept_vision_path);
        const concept_vision = try onnx.Session.open(env, concept_vision_path);
        errdefer concept_vision.deinit();

        const concept_text_data = try assets.concept_text_encoder_data.get(allocator, io);
        defer allocator.free(concept_text_data);
        const concept_text_path = try assets.concept_text_encoder.get(allocator, io);
        defer allocator.free(concept_text_path);
        const concept_text = try onnx.Session.open(env, concept_text_path);
        errdefer concept_text.deinit();

        const concept_decoder_path = try assets.concept_decoder.get(allocator, io);
        defer allocator.free(concept_decoder_path);
        const concept_decoder = try onnx.Session.open(env, concept_decoder_path);
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
        self.decoder.deinit();
        self.vision.deinit();
        self.env.deinit();
    }

    pub fn encode(self: *Model, img: Image) !Embedding {
        const data = try encodeVision(self, self.allocator, img);
        return Embedding.init(self.allocator, data);
    }

    pub fn decode(self: *Model, embedding: Embedding, points: []const render.Point) !Masks {
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
        const data = try encodeConceptVision(self, self.allocator, img);
        return ConceptEmbedding.init(self.allocator, data);
    }

    fn encodeText(self: *Model, encoding: tokenizer.Encoding) !TextEmbedding {
        const data = try encodeConceptText(self, self.allocator, encoding);
        return TextEmbedding.init(self.allocator, data);
    }

    pub fn lookup(self: *Model, embedding: ConceptEmbedding, phrase: []const u8, threshold: f32) !Masks {
        return decodeConceptQuery(self, self.allocator, embedding, phrase, threshold);
    }
};

pub const Embedding = struct {
    allocator: std.mem.Allocator,
    data: []f32,
    levels: [embedding_names.len]onnx.Value,

    fn init(allocator: std.mem.Allocator, data: []f32) !Embedding {
        return .{
            .allocator = allocator,
            .data = data,
            .levels = .{
                try onnx.Value.borrowF32(data[0..2654208], &.{ 1, 32, 288, 288 }),
                try onnx.Value.borrowF32(data[2654208..][0..1327104], &.{ 1, 64, 144, 144 }),
                try onnx.Value.borrowF32(data[3981312..][0..1327104], &.{ 1, 256, 72, 72 }),
            },
        };
    }

    pub fn deinit(self: *Embedding) void {
        self.allocator.free(self.data);
        self.* = undefined;
    }
};

pub const ConceptEmbedding = struct {
    allocator: std.mem.Allocator,
    data: []f32,
    levels: [concept_embedding_names.len]onnx.Value,

    fn init(allocator: std.mem.Allocator, data: []f32) !ConceptEmbedding {
        const offsets = [_]usize{
            0,
            21233664,
            26542080,
            27869184,
        };
        const lens = [_]usize{
            21233664,
            5308416,
            1327104,
            1327104,
        };
        const shapes = [_][4]i64{
            .{ 1, 256, 288, 288 },
            .{ 1, 256, 144, 144 },
            .{ 1, 256, 72, 72 },
            .{ 1, 256, 72, 72 },
        };
        var levels: [concept_embedding_names.len]onnx.Value = undefined;
        inline for (0..concept_embedding_names.len) |i| {
            levels[i] = try onnx.Value.borrowF32(data[offsets[i]..][0..lens[i]], &shapes[i]);
        }
        return .{
            .allocator = allocator,
            .data = data,
            .levels = levels,
        };
    }

    pub fn deinit(self: *ConceptEmbedding) void {
        self.allocator.free(self.data);
        self.* = undefined;
    }
};

const TextEmbedding = struct {
    allocator: std.mem.Allocator,
    data: []f32,
    value: onnx.Value,

    fn init(allocator: std.mem.Allocator, data: []f32) !TextEmbedding {
        return .{
            .allocator = allocator,
            .data = data,
            .value = try onnx.Value.borrowF32(data, &.{ 1, tokenizer.max_tokens, 256 }),
        };
    }

    fn deinit(self: *TextEmbedding) void {
        self.allocator.free(self.data);
        self.* = undefined;
    }
};

fn encodeVision(
    model: *Model,
    allocator: std.mem.Allocator,
    img: Image,
) ![]f32 {
    const pixels = try preprocessCpu(allocator, img, @splat(0.5), @splat(0.5));
    defer allocator.free(pixels);

    const pixel_shape = [_]i64{ 1, 3, image_size, image_size };
    const pixel_values = try onnx.Value.borrowF32(pixels, &pixel_shape);
    defer pixel_values.deinit();

    var levels: [embedding_names.len]onnx.Value = undefined;
    try model.vision.run(
        &.{vision_input},
        &.{pixel_values},
        &embedding_names,
        &levels,
    );
    defer for (levels) |level| level.deinit();

    const total_floats = 2654208 + 1327104 + 1327104;
    const out = try allocator.alloc(f32, total_floats);
    errdefer allocator.free(out);

    const l0 = try levels[0].dataF32();
    const l1 = try levels[1].dataF32();
    const l2 = try levels[2].dataF32();
    @memcpy(out[0..2654208], l0);
    @memcpy(out[2654208..][0..1327104], l1);
    @memcpy(out[3981312..][0..1327104], l2);

    return out;
}

fn encodeConceptVision(
    model: *Model,
    allocator: std.mem.Allocator,
    img: Image,
) ![]f32 {
    const pixels = try preprocessCpu(
        allocator,
        img,
        .{ 0.485, 0.456, 0.406 },
        .{ 0.229, 0.224, 0.225 },
    );
    defer allocator.free(pixels);

    const pixel_shape = [_]i64{ 1, 3, image_size, image_size };
    const pixel_values = try onnx.Value.borrowF32(pixels, &pixel_shape);
    defer pixel_values.deinit();

    var levels: [concept_embedding_names.len]onnx.Value = undefined;
    try model.concept_vision.run(
        &.{vision_input},
        &.{pixel_values},
        &concept_embedding_names,
        &levels,
    );
    defer for (levels) |level| level.deinit();

    const lens = [_]usize{ 21233664, 5308416, 1327104, 1327104 };
    const total_floats = 29196288;
    const out = try allocator.alloc(f32, total_floats);
    errdefer allocator.free(out);

    var offset: usize = 0;
    for (levels, lens) |level, len| {
        const data = try level.dataF32();
        @memcpy(out[offset..][0..len], data);
        offset += len;
    }

    return out;
}

fn encodeConceptText(
    model: *Model,
    allocator: std.mem.Allocator,
    encoding: tokenizer.Encoding,
) ![]f32 {
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

    const data = try text_features[0].dataF32();
    const out = try allocator.alloc(f32, data.len);
    @memcpy(out, data);
    return out;
}

fn decodeConceptQuery(
    model: *Model,
    allocator: std.mem.Allocator,
    embedding: ConceptEmbedding,
    phrase: []const u8,
    threshold: f32,
) !Masks {
    const encoding = try model.concept_tokenizer.encode(phrase);
    var text_embedding = try model.encodeText(encoding);
    defer text_embedding.deinit();

    const token_shape = [_]i64{ 1, tokenizer.max_tokens };
    const attention = try onnx.Value.borrowI64(&encoding.attention, &token_shape);
    defer attention.deinit();

    const inputs = [_]onnx.Value{
        embedding.levels[0],
        embedding.levels[1],
        embedding.levels[2],
        embedding.levels[3],
        text_embedding.value,
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

    return Masks.takeConcept(allocator, results[0], results[1], threshold);
}

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

    const ratio_y = @as(f32, @floatFromInt(img.height)) / @as(f32, @floatFromInt(image_size));
    const ratio_x = @as(f32, @floatFromInt(img.width)) / @as(f32, @floatFromInt(image_size));
    const byte_scale: f32 = 1.0 / 255.0;

    // Resize all three interleaved byte channels together and write directly
    // into the model's planar normalized tensor. The old path converted the
    // entire source image to float and resized it independently three times.
    for (0..image_size) |y| {
        const in_y = ratio_y * (@as(f32, @floatFromInt(y)) + 0.5) - 0.5;
        const y0 = clampPixelIndex(in_y, img.height);
        const y1 = @min(y0 + 1, img.height - 1);
        const wy = @max(0.0, in_y - @as(f32, @floatFromInt(y0)));
        const row0 = pixels[y0 * img.width ..][0..img.width];
        const row1 = pixels[y1 * img.width ..][0..img.width];

        for (0..image_size) |x| {
            const in_x = ratio_x * (@as(f32, @floatFromInt(x)) + 0.5) - 0.5;
            const x0 = clampPixelIndex(in_x, img.width);
            const x1 = @min(x0 + 1, img.width - 1);
            const wx = @max(0.0, in_x - @as(f32, @floatFromInt(x0)));
            const p00 = row0[x0];
            const p01 = row0[x1];
            const p10 = row1[x0];
            const p11 = row1[x1];
            const index = y * image_size + x;
            const a: [3]f32 = .{ @floatFromInt(p00.r), @floatFromInt(p00.g), @floatFromInt(p00.b) };
            const b: [3]f32 = .{ @floatFromInt(p01.r), @floatFromInt(p01.g), @floatFromInt(p01.b) };
            const c: [3]f32 = .{ @floatFromInt(p10.r), @floatFromInt(p10.g), @floatFromInt(p10.b) };
            const d: [3]f32 = .{ @floatFromInt(p11.r), @floatFromInt(p11.g), @floatFromInt(p11.b) };

            inline for (0..3) |channel| {
                const top = a[channel] + (b[channel] - a[channel]) * wx;
                const bottom = c[channel] + (d[channel] - c[channel]) * wx;
                const resized = (top + (bottom - top) * wy) * byte_scale;
                out[channel * plane_size + index] = (resized - mean[channel]) / deviation[channel];
            }
        }
    }
    return out;
}

fn clampPixelIndex(coordinate: f32, limit: usize) usize {
    if (coordinate <= 0.0) return 0;
    const floored: usize = @intFromFloat(@floor(coordinate));
    return @min(floored, limit - 1);
}

test {
    std.testing.refAllDecls(@This());
}
