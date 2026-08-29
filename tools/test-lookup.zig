const std = @import("std");
const sam3 = @import("sam3");

const vision_path = ".zig-cache/sam3/concept/vision_encoder_int4.onnx";
const text_path = ".zig-cache/sam3/concept/text_encoder_int4.onnx";
const decoder_path = ".zig-cache/sam3/concept/decoder_int4.onnx";
const tokenizer_path = ".zig-cache/sam3/concept/tokenizer.json";
const image_path = ".zig-cache/sam3/examples/cat.png";

pub fn main(init: std.process.Init) !void {
    var arena_state = std.heap.ArenaAllocator.init(init.gpa);
    defer arena_state.deinit();
    const allocator = arena_state.allocator();
    const io = init.io;

    const png = try std.Io.Dir.cwd().readFileAlloc(io, image_path, allocator, .limited(32 * 1024 * 1024));
    defer allocator.free(png);
    var img = try sam3.image.decode(allocator, png);
    defer img.deinit();

    const tokenizer_json = try std.Io.Dir.cwd().readFileAlloc(io, tokenizer_path, allocator, .limited(8 * 1024 * 1024));
    defer allocator.free(tokenizer_json);

    var model = try sam3.Model.open(allocator, io, .{
        .vision_encoder = vision_path,
        .decoder = decoder_path,
        .concept_vision_encoder = vision_path,
        .concept_text_encoder = text_path,
        .concept_decoder = decoder_path,
        .concept_tokenizer_json = tokenizer_json,
    }, .{});
    defer model.deinit();

    std.debug.print("Model opened. Loading / encoding concept vision...\n", .{});
    // Let's create ConceptEmbedding from /tmp/vision-cpu if available or encode
    var embedding: sam3.ConceptEmbedding = undefined;
    var data: [8][]align(@alignOf(f32)) u8 = undefined;
    var loaded_from_cache = true;
    for (0..8) |i| {
        const path = try std.fmt.allocPrint(allocator, "/tmp/vision-cpu.{d}.bin", .{i});
        defer allocator.free(path);
        data[i] = std.Io.Dir.cwd().readFileAllocOptions(io, path, allocator, .unlimited, .of(f32), null) catch {
            loaded_from_cache = false;
            break;
        };
    }
    if (loaded_from_cache) {
        std.debug.print("Using cached vision features from /tmp/vision-cpu\n", .{});
        const dims = [_][]const i64{
            &.{ 1, 256, 288, 288 }, &.{ 1, 256, 144, 144 }, &.{ 1, 256, 72, 72 }, &.{ 1, 256, 36, 36 },
            &.{ 1, 256, 288, 288 }, &.{ 1, 256, 144, 144 }, &.{ 1, 256, 72, 72 }, &.{ 1, 256, 36, 36 },
        };
        for (data, dims, &embedding.levels) |bytes, shape, *level| {
            level.* = try sam3.onnx.Value.borrowF32(std.mem.bytesAsSlice(f32, bytes), shape);
        }
    } else {
        std.debug.print("Encoding image on CPU...\n", .{});
        embedding = try model.encodeConcept(img);
    }
    defer if (!loaded_from_cache) embedding.deinit() else for (embedding.levels, data) |level, bytes| {
        level.deinit();
        allocator.free(bytes);
    };

    const phrases = [_][]const u8{ "cat", "a cat", "the cat", "cat.", "dog", "tabby cat" };
    for (phrases) |phrase| {
        std.debug.print("\n--- Lookup phrase: '{s}' ---\n", .{phrase});
        const encoding = try model.concept_tokenizer.encode(phrase);
        std.debug.print("Token IDs: {any}\n", .{encoding.ids[0..8]});
        std.debug.print("Attention: {any}\n", .{encoding.attention[0..8]});

        // Let's run decoder directly to inspect raw pred_logits and pred_masks
        const token_shape = [_]i64{ 1, sam3.tokenizer.max_tokens };
        const ids = try sam3.onnx.Value.borrowI64(&encoding.ids, &token_shape);
        defer ids.deinit();
        const attention = try sam3.onnx.Value.borrowI64(&encoding.attention, &token_shape);
        defer attention.deinit();

        var text_features: [1]sam3.onnx.Value = undefined;
        try model.concept_text.run(
            &.{ "input_ids", "attention_mask" },
            &.{ ids, attention },
            &.{"text_features"},
            &text_features,
        );
        defer text_features[0].deinit();

        const tf_vals = try text_features[0].dataF32();
        var tf_min: f32 = std.math.inf(f32);
        var tf_max: f32 = -std.math.inf(f32);
        for (tf_vals) |v| {
            tf_min = @min(tf_min, v);
            tf_max = @max(tf_max, v);
        }
        std.debug.print("text_features: len={d} min={d:.4} max={d:.4}\n", .{ tf_vals.len, tf_min, tf_max });

        const inputs = [_]sam3.onnx.Value{
            embedding.levels[0],
            embedding.levels[1],
            embedding.levels[2],
            embedding.levels[6],
            text_features[0],
            attention,
        };
        const output_names = [_][*:0]const u8{ "pred_masks", "pred_boxes", "pred_logits" };
        var results: [3]sam3.onnx.Value = undefined;
        try model.concept_decoder.run(
            &.{ "fpn_hidden_state_0", "fpn_hidden_state_1", "fpn_hidden_state_2", "fpn_position_encoding_2", "text_features", "attention_mask" },
            &inputs,
            &output_names,
            &results,
        );
        defer for (results) |res| res.deinit();

        const raw_logits = try results[2].dataF32();
        var max_logit: f32 = -std.math.inf(f32);
        var max_idx: usize = 0;
        var top_scores: [5]f32 = @splat(-std.math.inf(f32));
        var top_indices: [5]usize = @splat(0);
        for (raw_logits, 0..) |logit, i| {
            if (logit > max_logit) {
                max_logit = logit;
                max_idx = i;
            }
            const sig = 1.0 / (1.0 + @exp(-logit));
            for (0..5) |k| {
                if (sig > top_scores[k]) {
                    var shift: usize = 4;
                    while (shift > k) : (shift -= 1) {
                        top_scores[shift] = top_scores[shift - 1];
                        top_indices[shift] = top_indices[shift - 1];
                    }
                    top_scores[k] = sig;
                    top_indices[k] = i;
                    break;
                }
            }
        }
        std.debug.print("pred_logits len={d}, max_logit={d:.4} (score={d:.4}) at idx {d}\n", .{ raw_logits.len, max_logit, 1.0 / (1.0 + @exp(-max_logit)), max_idx });
        std.debug.print("Top 5 scores: ", .{});
        for (top_scores, top_indices) |sc, idx| {
            std.debug.print("[idx {d}: {d:.4}] ", .{ idx, sc });
        }
        std.debug.print("\n", .{});
    }
}
