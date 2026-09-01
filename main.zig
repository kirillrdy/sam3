const std = @import("std");
const sam3 = @import("sam3");
const web = @import("web/server.zig");
const build_options = @import("build_options");

const index_html = @embedFile("web/index.html");
const client_wasm = @embedFile("client_wasm");

const Asset = struct {
    name: []const u8,
    url: []const u8,
    sha256: []const u8,
};

const assets = [_]Asset{
    .{ .name = "vision_encoder.onnx", .url = "https://huggingface.co/onnx-community/sam3-tracker-ONNX/resolve/main/onnx/vision_encoder.onnx", .sha256 = "9f284aab8c3d8e81e9c79f7b566f9cea43b7bc9afdd920eee2390fb65b3db897" },
    .{ .name = "vision_encoder.onnx_data", .url = "https://huggingface.co/onnx-community/sam3-tracker-ONNX/resolve/main/onnx/vision_encoder.onnx_data", .sha256 = "838e1f0b2d0394ed3bd3b3499775dd6676524e1dfc5a7371948a76dcb69e4dd3" },
    .{ .name = "prompt_encoder_mask_decoder.onnx", .url = "https://huggingface.co/onnx-community/sam3-tracker-ONNX/resolve/main/onnx/prompt_encoder_mask_decoder.onnx", .sha256 = "4f9ac85291d634ae36a21ce940e3c09671cc05b6511966e5d3d96988b12b95f8" },
    .{ .name = "prompt_encoder_mask_decoder.onnx_data", .url = "https://huggingface.co/onnx-community/sam3-tracker-ONNX/resolve/main/onnx/prompt_encoder_mask_decoder.onnx_data", .sha256 = "2d870726d484cb496760fd139c21f115cf1b945c6b69583489faa2ac79f1d2ae" },
    .{ .name = "vision_encoder_int4.onnx", .url = "https://huggingface.co/danilobukvic/sam3-text-onnx/resolve/main/vision_encoder_int4.onnx", .sha256 = "88edb4602b7e7b2aa282543dea0b25a253bb13d5d7d5debbd19c2fb5e7941ae7" },
    .{ .name = "vision_encoder_int4.onnx.data", .url = "https://huggingface.co/danilobukvic/sam3-text-onnx/resolve/main/vision_encoder_int4.onnx.data", .sha256 = "b89c9156064e926761f29be3f87b160fd34f4c93f1de46593295d155621829a2" },
    .{ .name = "text_encoder_int4.onnx", .url = "https://huggingface.co/danilobukvic/sam3-text-onnx/resolve/main/text_encoder_int4.onnx", .sha256 = "92f824a1841b787dc8dafa8cb8e8dce0c874f8d2d629f6b1c8de88399ede3806" },
    .{ .name = "text_encoder_int4.onnx.data", .url = "https://huggingface.co/danilobukvic/sam3-text-onnx/resolve/main/text_encoder_int4.onnx.data", .sha256 = "fcf5adcd6ad7b5155409367efde4ee981a5482fd5700191499a666ba4b637db5" },
    .{ .name = "decoder_int4.onnx", .url = "https://huggingface.co/danilobukvic/sam3-text-onnx/resolve/main/decoder_int4.onnx", .sha256 = "2354b510382d025ab897fa158abe7da94d065c8f880d60aed35b01820361b06d" },
    .{ .name = "tokenizer.json", .url = "https://huggingface.co/danilobukvic/sam3-text-onnx/resolve/main/tokenizer.json", .sha256 = "6d9109cc838977f3ca94a379eec36aecc7c807e1785cd729660ca2fc0171fb35" },
    .{ .name = "cat.png", .url = "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=800&fm=png", .sha256 = "073adcb0112290e6928d6978789a6fa8266d2fa30a3d7c4330591d0b8c59d6a3" },
};

const CachedAssets = struct {
    allocator: std.mem.Allocator,
    paths: [assets.len][]u8,

    fn deinit(self: *CachedAssets) void {
        for (self.paths) |path| self.allocator.free(path);
        self.* = undefined;
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    std.debug.print("\n=== SAM 3 Web UI ===\n\n", .{});
    std.debug.print("  Model runtime: {s}\n", .{sam3.onnx.version()});

    var cached = try cacheAssets(allocator, init.io, init.environ_map);
    defer cached.deinit();

    const tokenizer_json = try std.Io.Dir.cwd().readFileAlloc(
        init.io,
        cached.paths[9],
        allocator,
        .limited(8 * 1024 * 1024),
    );
    defer allocator.free(tokenizer_json);

    var model = sam3.Model.open(allocator, init.io, .{
        .vision_encoder = cached.paths[0],
        .decoder = cached.paths[2],
        .concept_vision_encoder = cached.paths[4],
        .concept_text_encoder = cached.paths[6],
        .concept_decoder = cached.paths[8],
        .concept_tokenizer_json = tokenizer_json,
    }) catch |err| {
        const last = sam3.onnx.lastError();
        std.debug.print("Failed to initialize model: {t}{s}{s}\n", .{ err, if (last.len > 0) ": " else "", last });
        return err;
    };
    defer model.deinit();

    std.debug.print("  Loaded segmentation and text lookup graphs\n\n", .{});

    try web.run(
        allocator,
        init.io,
        &model,
        .{ .index_html = index_html, .client_wasm = client_wasm },
        .{
            .host = build_options.host,
            .port = build_options.port,
            .example_path = cached.paths[10],
        },
    );
}

fn cacheAssets(
    allocator: std.mem.Allocator,
    io: std.Io,
    environ: *const std.process.Environ.Map,
) !CachedAssets {
    const home = environ.get("HOME") orelse return error.HomeNotSet;
    const cache_dir = try std.fs.path.join(allocator, &.{ home, ".cache", "sam3-zig" });
    defer allocator.free(cache_dir);
    try std.Io.Dir.cwd().createDirPath(io, cache_dir);

    var cached: CachedAssets = .{ .allocator = allocator, .paths = undefined };
    var initialized: usize = 0;
    errdefer for (cached.paths[0..initialized]) |path| allocator.free(path);

    for (assets, 0..) |asset, i| {
        cached.paths[i] = try std.fs.path.join(allocator, &.{ cache_dir, asset.name });
        initialized += 1;
        try ensureAsset(allocator, io, asset, cached.paths[i]);
    }
    return cached;
}

fn ensureAsset(
    allocator: std.mem.Allocator,
    io: std.Io,
    asset: Asset,
    path: []const u8,
) !void {
    if (try hashFile(io, path)) |have| {
        if (std.ascii.eqlIgnoreCase(&have, asset.sha256)) return;
        std.debug.print("  {s}: present but checksum differs, re-downloading\n", .{asset.name});
    }

    const part_path = try std.fmt.allocPrint(allocator, "{s}.part", .{path});
    defer allocator.free(part_path);

    std.debug.print("  {s}: downloading\n", .{asset.name});
    try download(allocator, io, asset.url, part_path);

    const have = (try hashFile(io, part_path)) orelse return error.DownloadDisappeared;
    if (!std.ascii.eqlIgnoreCase(&have, asset.sha256)) {
        std.debug.print(
            \\  {s}: SHA-256 mismatch
            \\    expected {s}
            \\    actual   {s}
            \\
        , .{ asset.name, asset.sha256, &have });
        std.Io.Dir.cwd().deleteFile(io, part_path) catch {};
        return error.ChecksumMismatch;
    }
    std.debug.print("  {s}: verified against the published SHA-256\n", .{asset.name});

    const cwd = std.Io.Dir.cwd();
    try cwd.rename(part_path, cwd, path, io);
    std.debug.print("  {s}: cached in {s}\n", .{ asset.name, path });
}

fn download(
    allocator: std.mem.Allocator,
    io: std.Io,
    url: []const u8,
    dest_path: []const u8,
) !void {
    if (build_options.zig_http) {
        return downloadZig(allocator, io, url, dest_path);
    }
    return downloadWithCurl(allocator, io, url, dest_path);
}

fn downloadWithCurl(
    allocator: std.mem.Allocator,
    io: std.Io,
    url: []const u8,
    dest_path: []const u8,
) !void {
    const argv = &.{ "curl", "-fsSL", "--retry", "3", "-o", dest_path, url };

    const result = try std.process.run(allocator, io, .{ .argv = argv });
    defer {
        allocator.free(result.stdout);
        allocator.free(result.stderr);
    }

    return switch (result.term) {
        .exited => |code| if (code != 0) error.HttpRequestFailed,
        else => error.HttpRequestFailed,
    };
}

fn downloadZig(
    allocator: std.mem.Allocator,
    io: std.Io,
    url: []const u8,
    dest_path: []const u8,
) !void {
    var file = try std.Io.Dir.cwd().createFile(io, dest_path, .{});
    defer file.close(io);

    var write_buffer: [64 * 1024]u8 = undefined;
    var file_writer = file.writer(io, &write_buffer);

    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();

    const result = try client.fetch(.{
        .location = .{ .url = url },
        .method = .GET,
        .response_writer = &file_writer.interface,
    });
    try file_writer.interface.flush();

    if (result.status != .ok) {
        std.debug.print("  HTTP {d} for {s}\n", .{ @intFromEnum(result.status), url });
        return error.HttpRequestFailed;
    }
}

fn hashFile(io: std.Io, path: []const u8) !?[64]u8 {
    var file = std.Io.Dir.cwd().openFile(io, path, .{}) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };
    defer file.close(io);

    var read_buffer: [64 * 1024]u8 = undefined;
    var reader = file.reader(io, &read_buffer);
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    var chunk: [64 * 1024]u8 = undefined;
    while (true) {
        const count = try reader.interface.readSliceShort(&chunk);
        if (count == 0) break;
        hasher.update(chunk[0..count]);
    }

    var digest: [32]u8 = undefined;
    hasher.final(&digest);
    return std.fmt.bytesToHex(digest, .lower);
}
