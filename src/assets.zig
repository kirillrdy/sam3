const std = @import("std");
const sam3 = @import("sam3.zig");

pub const Asset = struct {
    name: []const u8,
    url: []const u8,
    sha256: []const u8,

    pub fn get(
        self: Asset,
        allocator: std.mem.Allocator,
        io: std.Io,
    ) ![]u8 {
        const home_c = std.c.getenv("HOME") orelse return error.HomeNotSet;
        const home = std.mem.span(home_c);
        const path = try std.fs.path.join(allocator, &.{ home, ".cache", "sam3-zig", self.name });
        errdefer allocator.free(path);

        if (try hashFile(io, path)) |have| {
            if (std.ascii.eqlIgnoreCase(&have, self.sha256)) return path;
            std.debug.print("  {s}: present but checksum differs, re-downloading\n", .{self.name});
        }

        const part_path = try std.fmt.allocPrint(allocator, "{s}.part", .{path});
        defer allocator.free(part_path);

        std.debug.print("  {s}: downloading\n", .{self.name});
        try download(allocator, io, self.url, part_path);

        const have = (try hashFile(io, part_path)) orelse return error.DownloadDisappeared;
        if (!std.ascii.eqlIgnoreCase(&have, self.sha256)) {
            std.debug.print(
                \\  {s}: SHA-256 mismatch
                \\    expected {s}
                \\    actual   {s}
                \\
            , .{ self.name, self.sha256, &have });
            std.Io.Dir.cwd().deleteFile(io, part_path) catch {};
            return error.ChecksumMismatch;
        }
        std.debug.print("  {s}: verified against the published SHA-256\n", .{self.name});

        const cwd = std.Io.Dir.cwd();
        try cwd.rename(part_path, cwd, path, io);
        std.debug.print("  {s}: cached in {s}\n", .{ self.name, path });

        return path;
    }
};

pub const Assets = struct {
    vision_encoder: Asset,
    vision_encoder_data: Asset,
    decoder: Asset,
    decoder_data: Asset,
    concept_vision_encoder: Asset,
    concept_vision_encoder_data: Asset,
    concept_text_encoder: Asset,
    concept_text_encoder_data: Asset,
    concept_decoder: Asset,
    concept_tokenizer_json: Asset,
    cat: Asset,
};

pub const default_assets: Assets = .{
    .vision_encoder = .{ .name = "vision_encoder.onnx", .url = "https://huggingface.co/onnx-community/sam3-tracker-ONNX/resolve/main/onnx/vision_encoder.onnx", .sha256 = "9f284aab8c3d8e81e9c79f7b566f9cea43b7bc9afdd920eee2390fb65b3db897" },
    .vision_encoder_data = .{ .name = "vision_encoder.onnx_data", .url = "https://huggingface.co/onnx-community/sam3-tracker-ONNX/resolve/main/onnx/vision_encoder.onnx_data", .sha256 = "838e1f0b2d0394ed3bd3b3499775dd6676524e1dfc5a7371948a76dcb69e4dd3" },
    .decoder = .{ .name = "prompt_encoder_mask_decoder.onnx", .url = "https://huggingface.co/onnx-community/sam3-tracker-ONNX/resolve/main/onnx/prompt_encoder_mask_decoder.onnx", .sha256 = "4f9ac85291d634ae36a21ce940e3c09671cc05b6511966e5d3d96988b12b95f8" },
    .decoder_data = .{ .name = "prompt_encoder_mask_decoder.onnx_data", .url = "https://huggingface.co/onnx-community/sam3-tracker-ONNX/resolve/main/onnx/prompt_encoder_mask_decoder.onnx_data", .sha256 = "2d870726d484cb496760fd139c21f115cf1b945c6b69583489faa2ac79f1d2ae" },
    .concept_vision_encoder = .{ .name = "vision_encoder_int4.onnx", .url = "https://huggingface.co/danilobukvic/sam3-text-onnx/resolve/main/vision_encoder_int4.onnx", .sha256 = "88edb4602b7e7b2aa282543dea0b25a253bb13d5d7d5debbd19c2fb5e7941ae7" },
    .concept_vision_encoder_data = .{ .name = "vision_encoder_int4.onnx.data", .url = "https://huggingface.co/danilobukvic/sam3-text-onnx/resolve/main/vision_encoder_int4.onnx.data", .sha256 = "b89c9156064e926761f29be3f87b160fd34f4c93f1de46593295d155621829a2" },
    .concept_text_encoder = .{ .name = "text_encoder_int4.onnx", .url = "https://huggingface.co/danilobukvic/sam3-text-onnx/resolve/main/text_encoder_int4.onnx", .sha256 = "92f824a1841b787dc8dafa8cb8e8dce0c874f8d2d629f6b1c8de88399ede3806" },
    .concept_text_encoder_data = .{ .name = "text_encoder_int4.onnx.data", .url = "https://huggingface.co/danilobukvic/sam3-text-onnx/resolve/main/text_encoder_int4.onnx.data", .sha256 = "fcf5adcd6ad7b5155409367efde4ee981a5482fd5700191499a666ba4b637db5" },
    .concept_decoder = .{ .name = "decoder_int4.onnx", .url = "https://huggingface.co/danilobukvic/sam3-text-onnx/resolve/main/decoder_int4.onnx", .sha256 = "2354b510382d025ab897fa158abe7da94d065c8f880d60aed35b01820361b06d" },
    .concept_tokenizer_json = .{ .name = "tokenizer.json", .url = "https://huggingface.co/danilobukvic/sam3-text-onnx/resolve/main/tokenizer.json", .sha256 = "6d9109cc838977f3ca94a379eec36aecc7c807e1785cd729660ca2fc0171fb35" },
    .cat = .{ .name = "cat.png", .url = "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=800&fm=png", .sha256 = "dc6a561fc58bf60caff7a62cdd7593f5b517e43e4a75e9b220a80c3f1229ba3c" },
};


fn download(
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
