const std = @import("std");
const builtin = @import("builtin");
pub const onnx = @import("runtime");
pub const tokenizer = @import("tokenizer.zig");
pub const tracking = @import("track.zig");
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

/// An axis-aligned box over the frame, which is the unit square whatever the
/// image's pixel size. Both the model's own boxes and the ones derived from a
/// mask are in these coordinates, so the two are directly comparable.
pub const Box = struct {
    x0: f32 = 0,
    y0: f32 = 0,
    x1: f32 = 0,
    y1: f32 = 0,

    pub fn width(self: Box) f32 {
        return @max(0.0, self.x1 - self.x0);
    }

    pub fn height(self: Box) f32 {
        return @max(0.0, self.y1 - self.y0);
    }

    pub fn area(self: Box) f32 {
        return self.width() * self.height();
    }

    pub fn isEmpty(self: Box) bool {
        return self.area() <= 0.0;
    }

    pub fn intersect(a: Box, b: Box) Box {
        return .{
            .x0 = @max(a.x0, b.x0),
            .y0 = @max(a.y0, b.y0),
            .x1 = @min(a.x1, b.x1),
            .y1 = @min(a.y1, b.y1),
        };
    }

    /// The ground two boxes agree on as a share of the ground they cover
    /// between them. Zero when either of them is empty.
    pub fn iou(a: Box, b: Box) f32 {
        const overlap = a.intersect(b).area();
        const combined = a.area() + b.area() - overlap;
        if (combined <= 0.0) return 0.0;
        return overlap / combined;
    }

    pub fn clamp(self: Box) Box {
        return .{
            .x0 = std.math.clamp(self.x0, 0.0, 1.0),
            .y0 = std.math.clamp(self.y0, 0.0, 1.0),
            .x1 = std.math.clamp(self.x1, 0.0, 1.0),
            .y1 = std.math.clamp(self.y1, 0.0, 1.0),
        };
    }
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
        const pixels = try preprocessCpu(self.allocator, img, @splat(0.5), @splat(0.5));
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

    /// Segments what `points` name, optionally hemmed in by `box`. Carrying a
    /// track from one frame to the next is the same call with the box the
    /// object left behind on the frame before.
    pub fn decode(self: *Model, embedding: Embedding, points: []const Point, box: ?Box) !Masks {
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

        // The decoder reads boxes in the same pixel space as points, and takes
        // none at all as a zero-length axis rather than as absent.
        var box_corners: [4]f32 = @splat(0.0);
        if (box) |b| box_corners = .{ b.x0 * scale, b.y0 * scale, b.x1 * scale, b.y1 * scale };
        const box_count: i64 = if (box == null) 0 else 1;
        const box_shape = [_]i64{ 1, box_count, 4 };

        const input_points = try onnx.Value.borrowF32(coordinates, &point_shape);
        defer input_points.deinit();
        const input_labels = try onnx.Value.borrowI64(labels, &label_shape);
        defer input_labels.deinit();
        const input_boxes = try onnx.Value.borrowF32(box_corners[0..@intCast(box_count * 4)], &box_shape);
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
        const pixels = try preprocessCpu(
            self.allocator,
            img,
            .{ 0.485, 0.456, 0.406 },
            .{ 0.229, 0.224, 0.225 },
        );
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
        return Masks.takeConcept(self.allocator, results[0], results[1], results[2], threshold);
    }
};

pub const Embedding = EmbeddingOf(embedding_names.len);
pub const ConceptEmbedding = EmbeddingOf(concept_embedding_names.len);

fn EmbeddingOf(comptime levels: usize) type {
    return struct {
        levels: [levels]onnx.Value,

        pub fn deinit(self: *@This()) void {
            for (self.levels) |level| level.deinit();
            self.* = undefined;
        }
    };
}

pub const Masks = struct {
    allocator: std.mem.Allocator,

    logits: []f32,
    scores: []f32,
    /// One box per hypothesis, over the frame. The text decoder predicts these
    /// itself; on the point path they are read back off the masks.
    boxes: []Box,
    count: usize,
    width: usize,
    height: usize,

    object_score: f32,

    /// The logits of one hypothesis, row-major over `width` by `height`.
    pub fn plane(self: Masks, index: usize) []const f32 {
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
        errdefer allocator.free(scores);

        const boxes = try allocator.alloc(Box, count);
        errdefer allocator.free(boxes);
        const stride = width * height;
        for (boxes, 0..) |*box, i| {
            box.* = maskBox(logits[i * stride ..][0..stride], width, height);
        }

        return .{
            .allocator = allocator,
            .logits = logits,
            .scores = scores,
            .boxes = boxes,
            .count = count,
            .width = width,
            .height = height,
            .object_score = (try object.dataF32())[0],
        };
    }

    fn takeConcept(
        allocator: std.mem.Allocator,
        masks: onnx.Value,
        boxes_value: onnx.Value,
        scores_value: onnx.Value,
        threshold: f32,
    ) !Masks {
        var dims: [8]i64 = undefined;
        const shape = try masks.shape(&dims);
        if (shape.len != 4 or shape[0] != 1) return error.UnexpectedConceptMaskShape;

        const available: usize = @intCast(shape[1]);
        const height: usize = @intCast(shape[2]);
        const width: usize = @intCast(shape[3]);
        const planes = try masks.dataF32();
        const raw_scores = try scores_value.dataF32();
        if (raw_scores.len < available) return error.UnexpectedConceptScoreShape;
        // Corners over the frame, one row of four per query.
        const raw_boxes = try boxes_value.dataF32();
        if (raw_boxes.len < available * 4) return error.UnexpectedConceptBoxShape;

        var count: usize = 0;
        for (raw_scores[0..available]) |logit| {
            const score = sigmoid(logit);
            if (score >= threshold) count += 1;
        }

        const logits = try allocator.alloc(f32, count * width * height);
        errdefer allocator.free(logits);
        const scores = try allocator.alloc(f32, count);
        errdefer allocator.free(scores);
        const boxes = try allocator.alloc(Box, count);
        errdefer allocator.free(boxes);

        const stride = width * height;
        var out: usize = 0;
        for (raw_scores[0..available], 0..) |logit, i| {
            const score = sigmoid(logit);
            if (score < threshold) continue;
            scores[out] = score;
            const corners = raw_boxes[i * 4 ..][0..4];
            boxes[out] = (Box{
                .x0 = corners[0],
                .y0 = corners[1],
                .x1 = corners[2],
                .y1 = corners[3],
            }).clamp();
            @memcpy(logits[out * stride ..][0..stride], planes[i * stride ..][0..stride]);
            out += 1;
        }

        return .{
            .allocator = allocator,
            .logits = logits,
            .scores = scores,
            .boxes = boxes,
            .count = count,
            .width = width,
            .height = height,
            .object_score = if (count == 0) 0 else max(scores),
        };
    }

    pub fn deinit(self: *Masks) void {
        self.allocator.free(self.logits);
        self.allocator.free(self.scores);
        self.allocator.free(self.boxes);
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

/// The box around everything the mask claims, or an empty box when it claims
/// nothing. Logits are positive inside the object, as the decoder emits them.
pub fn maskBox(plane: []const f32, width: usize, height: usize) Box {
    var x0: usize = width;
    var y0: usize = height;
    var x1: usize = 0;
    var y1: usize = 0;
    var any = false;

    for (0..height) |y| {
        const row = plane[y * width ..][0..width];
        for (row, 0..) |logit, x| {
            if (logit <= 0.0) continue;
            any = true;
            x0 = @min(x0, x);
            x1 = @max(x1, x);
            y0 = @min(y0, y);
            y1 = @max(y1, y);
        }
    }
    if (!any) return .{};

    // A pixel covers the half-open cell that starts at its index, so the far
    // edge is one past the last one that is set.
    return .{
        .x0 = @as(f32, @floatFromInt(x0)) / @as(f32, @floatFromInt(width)),
        .y0 = @as(f32, @floatFromInt(y0)) / @as(f32, @floatFromInt(height)),
        .x1 = @as(f32, @floatFromInt(x1 + 1)) / @as(f32, @floatFromInt(width)),
        .y1 = @as(f32, @floatFromInt(y1 + 1)) / @as(f32, @floatFromInt(height)),
    };
}

/// Intersection over union of two masks of the same size, counting the pixels
/// each of them claims.
pub fn maskIou(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var overlap: usize = 0;
    var combined: usize = 0;
    for (a, b) |left, right| {
        const in_left = left > 0.0;
        const in_right = right > 0.0;
        if (in_left and in_right) overlap += 1;
        if (in_left or in_right) combined += 1;
    }
    if (combined == 0) return 0.0;
    return @as(f32, @floatFromInt(overlap)) / @as(f32, @floatFromInt(combined));
}

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

test "overlap is measured against the ground two boxes cover between them" {
    const left: Box = .{ .x0 = 0, .y0 = 0, .x1 = 2, .y1 = 2 };
    const right: Box = .{ .x0 = 1, .y0 = 1, .x1 = 3, .y1 = 3 };

    try std.testing.expectEqual(@as(f32, 1.0), left.iou(left));
    try std.testing.expectEqual(@as(f32, 1.0 / 7.0), left.iou(right));
    try std.testing.expectEqual(@as(f32, 0.0), left.iou(.{ .x0 = 5, .y0 = 5, .x1 = 6, .y1 = 6 }));
    try std.testing.expectEqual(@as(f32, 0.0), left.iou(.{}));
}

test "a mask reports the box around what it claims" {
    // A 4x4 frame with one positive pixel at column 1, row 2.
    var plane: [16]f32 = @splat(-1.0);
    plane[2 * 4 + 1] = 3.0;

    const box = maskBox(&plane, 4, 4);
    try std.testing.expectEqual(@as(f32, 0.25), box.x0);
    try std.testing.expectEqual(@as(f32, 0.5), box.y0);
    try std.testing.expectEqual(@as(f32, 0.5), box.x1);
    try std.testing.expectEqual(@as(f32, 0.75), box.y1);

    const nothing: [16]f32 = @splat(-1.0);
    try std.testing.expect(maskBox(&nothing, 4, 4).isEmpty());
}

test "masks overlap by the pixels they both claim" {
    const a = [_]f32{ 1, 1, -1, -1 };
    const b = [_]f32{ -1, 1, 1, -1 };

    try std.testing.expectEqual(@as(f32, 1.0), maskIou(&a, &a));
    try std.testing.expectEqual(@as(f32, 1.0 / 3.0), maskIou(&a, &b));
    try std.testing.expectEqual(@as(f32, 0.0), maskIou(&a, &@as([4]f32, @splat(-1.0))));
}
