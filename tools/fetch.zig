//! Downloader used by the `zig build fetch-weights` and `zig build fetch-examples`
//! steps, so pulling Meta's checkpoint and the playground sample images needs
//! nothing but Zig — no curl, no shell.
//!
//! Usage:
//!   fetch --url <url> --out <path> [--sha256 <hex>] [--token-env <VAR>] [--label <text>]
//!
//! With `--sha256` the download is verified and the file is only moved into
//! place once it matches; an existing file that already matches is left alone,
//! which makes the build steps idempotent and cheap to re-run.

const std = @import("std");

const Options = struct {
    url: []const u8,
    out: []const u8,
    sha256: ?[]const u8 = null,
    token_env: ?[]const u8 = null,
    label: ?[]const u8 = null,
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var args_it = try std.process.Args.Iterator.initAllocator(init.minimal.args, gpa);
    defer args_it.deinit();
    _ = args_it.skip();

    var url: ?[]const u8 = null;
    var out: ?[]const u8 = null;
    var sha256: ?[]const u8 = null;
    var token_env: ?[]const u8 = null;
    var label: ?[]const u8 = null;

    while (args_it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--url")) {
            url = args_it.next();
        } else if (std.mem.eql(u8, arg, "--out")) {
            out = args_it.next();
        } else if (std.mem.eql(u8, arg, "--sha256")) {
            sha256 = args_it.next();
        } else if (std.mem.eql(u8, arg, "--token-env")) {
            token_env = args_it.next();
        } else if (std.mem.eql(u8, arg, "--label")) {
            label = args_it.next();
        } else {
            std.debug.print("fetch: unknown argument '{s}'\n", .{arg});
            return error.InvalidArguments;
        }
    }

    if (url == null or out == null) {
        std.debug.print(
            \\usage: fetch --url <url> --out <path> [--sha256 <hex>] [--token-env <VAR>] [--label <text>]
            \\
        , .{});
        return error.InvalidArguments;
    }

    const opts = Options{
        .url = url.?,
        .out = out.?,
        .sha256 = sha256,
        .token_env = token_env,
        .label = label,
    };

    const name = opts.label orelse opts.out;

    if (opts.sha256) |want| {
        if (try hashFile(io, opts.out)) |have| {
            if (std.ascii.eqlIgnoreCase(&have, want)) {
                std.debug.print("  {s}: already present and verified\n", .{name});
                return;
            }
            std.debug.print("  {s}: present but checksum differs, re-downloading\n", .{name});
        }
    }

    try ensureParentDir(io, opts.out);

    var part_buf: [4096]u8 = undefined;
    const part_path = try std.fmt.bufPrint(&part_buf, "{s}.part", .{opts.out});

    std.debug.print("  {s}: downloading\n", .{name});
    const token = if (opts.token_env) |var_name| init.environ_map.get(var_name) else null;
    try download(gpa, io, opts, part_path, token);

    if (opts.sha256) |want| {
        const have = (try hashFile(io, part_path)) orelse return error.DownloadDisappeared;
        if (!std.ascii.eqlIgnoreCase(&have, want)) {
            std.debug.print(
                \\  {s}: SHA-256 mismatch
                \\    expected {s}
                \\    actual   {s}
                \\
            , .{ name, want, &have });
            std.Io.Dir.cwd().deleteFile(io, part_path) catch {};
            return error.ChecksumMismatch;
        }
        std.debug.print("  {s}: verified against the published SHA-256\n", .{name});
    }

    const cwd = std.Io.Dir.cwd();
    try cwd.rename(part_path, cwd, opts.out, io);
    std.debug.print("  {s}: -> {s}\n", .{ name, opts.out });
}

fn download(gpa: std.mem.Allocator, io: std.Io, opts: Options, dest_path: []const u8, token: ?[]const u8) !void {
    const cwd = std.Io.Dir.cwd();

    var file = try cwd.createFile(io, dest_path, .{});
    defer file.close(io);

    const write_buf = try gpa.alloc(u8, 1 << 20);
    defer gpa.free(write_buf);
    var file_writer = file.writer(io, write_buf);

    var client: std.http.Client = .{ .allocator = gpa, .io = io };
    defer client.deinit();

    // Hugging Face gates some repositories; an access token in the environment
    // is forwarded as a privileged header so it survives redirects to the CDN
    // only for the same host.
    var auth_buf: [4096]u8 = undefined;
    var extra: [1]std.http.Header = undefined;
    var extra_len: usize = 0;

    if (token) |value_raw| {
        const value = try std.fmt.bufPrint(&auth_buf, "Bearer {s}", .{value_raw});
        extra[0] = .{ .name = "authorization", .value = value };
        extra_len = 1;
    }

    const result = try client.fetch(.{
        .location = .{ .url = opts.url },
        .method = .GET,
        .response_writer = &file_writer.interface,
        .privileged_headers = extra[0..extra_len],
    });

    try file_writer.interface.flush();

    if (result.status != .ok) {
        std.debug.print("  HTTP {d} for {s}\n", .{ @intFromEnum(result.status), opts.url });
        return error.HttpRequestFailed;
    }
}

/// SHA-256 of a file as lowercase hex, or null when the file does not exist.
fn hashFile(io: std.Io, path: []const u8) !?[64]u8 {
    const cwd = std.Io.Dir.cwd();
    var file = cwd.openFile(io, path, .{}) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };
    defer file.close(io);

    var read_buf: [1 << 20]u8 = undefined;
    var file_reader = file.reader(io, &read_buf);

    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    var chunk: [1 << 16]u8 = undefined;
    while (true) {
        const n = file_reader.interface.readSliceShort(&chunk) catch |err| switch (err) {
            error.ReadFailed => return error.ReadFailed,
        };
        if (n == 0) break;
        hasher.update(chunk[0..n]);
    }

    var digest: [32]u8 = undefined;
    hasher.final(&digest);

    var hex: [64]u8 = undefined;
    _ = std.fmt.bufPrint(&hex, "{x}", .{&digest}) catch unreachable;
    return hex;
}

fn ensureParentDir(io: std.Io, path: []const u8) !void {
    const slash = std.mem.lastIndexOfScalar(u8, path, '/') orelse return;
    if (slash == 0) return;
    try std.Io.Dir.cwd().createDirPath(io, path[0..slash]);
}
