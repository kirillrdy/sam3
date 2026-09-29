const std = @import("std");
const sam3 = @import("sam3");

const Stats = struct {
    min_ms: f64,
    max_ms: f64,
    median_ms: f64,
    mean_ms: f64,
    samples: usize,

    pub fn compute(durations_ns: []const i96) Stats {
        std.debug.assert(durations_ns.len > 0);
        var sorted: [128]f64 = undefined;
        const n = @min(durations_ns.len, sorted.len);
        for (durations_ns[0..n], 0..) |d, i| {
            sorted[i] = @as(f64, @floatFromInt(d)) / 1_000_000.0;
        }
        std.mem.sort(f64, sorted[0..n], {}, std.sort.asc(f64));

        var sum: f64 = 0;
        for (sorted[0..n]) |val| sum += val;
        const mean = sum / @as(f64, @floatFromInt(n));
        const median = sorted[n / 2];

        return .{
            .min_ms = sorted[0],
            .max_ms = sorted[n - 1],
            .median_ms = median,
            .mean_ms = mean,
            .samples = n,
        };
    }
};

fn now(io: std.Io) std.Io.Timestamp {
    return std.Io.Timestamp.now(io, .awake);
}

fn elapsedNs(start: std.Io.Timestamp, end: std.Io.Timestamp) i96 {
    return end.nanoseconds - start.nanoseconds;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    std.debug.print("\n======================================================\n", .{});
    std.debug.print("  SAM 3 Text Querying Benchmark\n", .{});
    std.debug.print("======================================================\n\n", .{});
    std.debug.print("Model ONNX runtime: {s}\n", .{sam3.onnx.version()});

    // 1. Initialize model
    std.debug.print("Opening SAM 3 model graphs...\n", .{});
    const t_open_start = now(io);
    var model = try sam3.Model.open(allocator, io);
    defer model.deinit();
    const t_open_end = now(io);
    const open_ms = @as(f64, @floatFromInt(elapsedNs(t_open_start, t_open_end))) / 1_000_000.0;
    std.debug.print("Model opened in {d:.2} ms\n\n", .{open_ms});

    // 2. Load cat image
    std.debug.print("Loading cat image asset...\n", .{});
    const t_load_start = now(io);
    const image_path = try sam3.assets.cat.get(allocator, io);
    defer allocator.free(image_path);

    const file_bytes = try std.Io.Dir.cwd().readFileAlloc(
        io,
        image_path,
        allocator,
        .limited(32 * 1024 * 1024),
    );
    defer allocator.free(file_bytes);

    var img = try sam3.decodeImage(allocator, file_bytes);
    defer img.deinit(allocator);

    const rgb_image = sam3.RgbImage.fromImage(img);
    const t_load_end = now(io);
    const load_ms = @as(f64, @floatFromInt(elapsedNs(t_load_start, t_load_end))) / 1_000_000.0;
    std.debug.print("Loaded {s}: {d}x{d} ({d} bytes RGB24) in {d:.2} ms\n\n", .{
        image_path,
        rgb_image.width,
        rgb_image.height,
        rgb_image.pixels.len,
        load_ms,
    });

    // 3. Benchmark Image Preprocessing & Concept Vision Encoder (encodeForText)
    std.debug.print("--- Benchmarking Vision Concept Encoder Breakdown ---\n", .{});
    {
        var prep_durations: [5]i96 = undefined;
        for (0..5) |i| {
            const t0 = now(io);
            const prep = try sam3.preprocessCpu(
                allocator,
                rgb_image,
                .{ 0.485, 0.456, 0.406 },
                .{ 0.229, 0.224, 0.225 },
            );
            const t1 = now(io);
            allocator.free(prep);
            prep_durations[i] = elapsedNs(t0, t1);
        }
        const prep_stats = Stats.compute(&prep_durations);
        std.debug.print("preprocessCpu (5 runs): mean={d:.2} ms | median={d:.2} ms | min={d:.2} ms | max={d:.2} ms\n", .{
            prep_stats.mean_ms,
            prep_stats.median_ms,
            prep_stats.min_ms,
            prep_stats.max_ms,
        });
    }

    const t_enc_cold_start = now(io);
    var cold_embedding = try model.encodeForText(rgb_image);
    const t_enc_cold_end = now(io);
    cold_embedding.deinit();
    const enc_cold_ms = @as(f64, @floatFromInt(elapsedNs(t_enc_cold_start, t_enc_cold_end))) / 1_000_000.0;
    std.debug.print("Cold encodeForText: {d:.2} ms\n", .{enc_cold_ms});

    const vision_runs = 5;
    var vision_durations: [vision_runs]i96 = undefined;
    var final_embedding: ?sam3.TextImageEmbedding = null;
    defer if (final_embedding) |*emb| emb.deinit();

    for (0..vision_runs) |i| {
        const t0 = now(io);
        var emb = try model.encodeForText(rgb_image);
        const t1 = now(io);
        vision_durations[i] = elapsedNs(t0, t1);
        if (i + 1 == vision_runs) {
            final_embedding = emb;
        } else {
            emb.deinit();
        }
    }

    const vision_stats = Stats.compute(&vision_durations);
    std.debug.print("Warm encodeForText ({d} runs): mean={d:.2} ms | median={d:.2} ms | min={d:.2} ms | max={d:.2} ms\n\n", .{
        vision_stats.samples,
        vision_stats.mean_ms,
        vision_stats.median_ms,
        vision_stats.min_ms,
        vision_stats.max_ms,
    });

    const embedding = &(final_embedding.?);

    // Queries to test
    const phrases = [_][]const u8{
        "cat",
        "cat eye",
        "ear",
        "whiskers",
        "nose",
        "fur",
        "tail",
    };

    const query_runs = 10;

    std.debug.print("--- Benchmarking Text Feature Encoding (encodeTextFeatures) ---\n", .{});
    for (phrases) |phrase| {
        var text_durations: [query_runs]i96 = undefined;
        var text_features: ?[]f32 = null;
        for (0..query_runs) |i| {
            const t0 = now(io);
            const feat = try model.encodeTextFeatures(phrase);
            const t1 = now(io);
            text_durations[i] = elapsedNs(t0, t1);
            if (i == 0) {
                text_features = feat;
            } else {
                allocator.free(feat);
            }
        }
        defer if (text_features) |feat| allocator.free(feat);
        const text_stats = Stats.compute(&text_durations);
        std.debug.print("Text \"{s:<10}\" ({d} runs): mean={d:.2} ms | median={d:.2} ms | min={d:.2} ms | max={d:.2} ms (dim={d})\n", .{
            phrase,
            text_stats.samples,
            text_stats.mean_ms,
            text_stats.median_ms,
            text_stats.min_ms,
            text_stats.max_ms,
            text_features.?.len,
        });
    }
    std.debug.print("\n", .{});

    // 4. Benchmark Text Querying (find and findWithTextFeatures)
    std.debug.print("--- Benchmarking Text Querying / Concept Decoder ({d} runs each) ---\n", .{query_runs});
    std.debug.print("{s:<12} | {s:<10} | {s:<12} | {s:<12} | {s:<8} | {s:<10} | {s:<8}\n", .{
        "Query", "Masks", "Top Score", "find() Mean", "find() Med", "Cached Mean", "Speedup",
    });
    std.debug.print("{s:-<12}-+-{s:-<10}-+-{s:-<12}-+-{s:-<12}-+-{s:-<8}-+-{s:-<10}-+-{s:-<8}\n", .{
        "", "", "", "", "", "", "",
    });

    for (phrases) |phrase| {
        // Precompute text features for cached benchmark
        const text_features = try model.encodeTextFeatures(phrase);
        defer allocator.free(text_features);

        // Warm up find
        {
            var warmup_mask = try model.find(embedding, phrase, .{ .min_score = 0.5 });
            warmup_mask.deinit();
        }

        var find_durations: [query_runs]i96 = undefined;
        var last_mask_count: usize = 0;
        var last_top_score: f32 = 0;

        for (0..query_runs) |i| {
            const t0 = now(io);
            var masks = try model.find(embedding, phrase, .{ .min_score = 0.5 });
            const t1 = now(io);
            find_durations[i] = elapsedNs(t0, t1);
            last_mask_count = masks.count;
            last_top_score = masks.object_score;
            masks.deinit();
        }

        var cached_durations: [query_runs]i96 = undefined;
        for (0..query_runs) |i| {
            const t0 = now(io);
            var masks = try model.findWithTextFeatures(embedding, phrase, text_features, .{ .min_score = 0.5 });
            const t1 = now(io);
            cached_durations[i] = elapsedNs(t0, t1);
            masks.deinit();
        }

        const find_stats = Stats.compute(&find_durations);
        const cached_stats = Stats.compute(&cached_durations);
        const speedup = find_stats.mean_ms / cached_stats.mean_ms;

        std.debug.print("\"{s:<10}\" | {d:<10} | {d:<12.4} | {d:<9.2} ms | {d:<5.2} ms | {d:<7.2} ms | {d:<7.2}x\n", .{
            phrase,
            last_mask_count,
            last_top_score,
            find_stats.mean_ms,
            find_stats.median_ms,
            cached_stats.mean_ms,
            speedup,
        });
    }

    std.debug.print("\n======================================================\n", .{});
    std.debug.print("Benchmark completed successfully.\n\n", .{});
}
