const std = @import("std");

const Options = struct {
    url: ?[]const u8 = null,
    out: ?[]const u8 = null,
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

    var opts: Options = .{};
    while (args_it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--url")) {
            opts.url = args_it.next();
        } else if (std.mem.eql(u8, arg, "--out")) {
            opts.out = args_it.next();
        } else if (std.mem.eql(u8, arg, "--sha256")) {
            opts.sha256 = args_it.next();
        } else if (std.mem.eql(u8, arg, "--token-env")) {
            opts.token_env = args_it.next();
        } else if (std.mem.eql(u8, arg, "--label")) {
            opts.label = args_it.next();
        } else {
            std.debug.print("fetch: unknown argument '{s}'\n", .{arg});
            return error.InvalidArguments;
        }
    }

    if (opts.url == null or opts.out == null) {
        std.debug.print(
            \\usage: fetch --url <url> --out <path> [--sha256 <hex>] [--token-env <VAR>] [--label <text>]
            \\
        , .{});
        return error.InvalidArguments;
    }

    const url = opts.url.?;
    const out = opts.out.?;
    const name = opts.label orelse out;

    if (opts.sha256) |want| {
        if (try hashFile(io, out)) |have| {
            if (std.ascii.eqlIgnoreCase(&have, want)) return;
            std.debug.print("  {s}: present but checksum differs, re-downloading\n", .{name});
        }
    } else {
        if (fileExists(io, out)) return;
    }

    try ensureParentDir(io, out);

    var part_buf: [4096]u8 = undefined;
    const part_path = try std.fmt.bufPrint(&part_buf, "{s}.part", .{out});

    std.debug.print("  {s}: downloading\n", .{name});
    const token = if (opts.token_env) |var_name| init.environ_map.get(var_name) else null;
    try download(gpa, io, url, part_path, token);

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
    try cwd.rename(part_path, cwd, out, io);
    std.debug.print("  {s}: -> {s}\n", .{ name, out });
}

fn fileExists(io: std.Io, path: []const u8) bool {
    std.Io.Dir.cwd().access(io, path, .{}) catch return false;
    return true;
}

fn download(gpa: std.mem.Allocator, io: std.Io, url: []const u8, dest_path: []const u8, token: ?[]const u8) !void {
    downloadWithCurl(gpa, io, url, dest_path, token) catch {
        return downloadZig(gpa, io, url, dest_path, token);
    };
}

fn downloadWithCurl(gpa: std.mem.Allocator, io: std.Io, url: []const u8, dest_path: []const u8, token: ?[]const u8) !void {
    var auth_buf: [4096]u8 = undefined;
    const argv: []const []const u8 = if (token) |tok|
        &.{ "curl", "-sSL", "--retry", "3", "-H", try std.fmt.bufPrint(&auth_buf, "Authorization: Bearer {s}", .{tok}), "-o", dest_path, url }
    else
        &.{ "curl", "-sSL", "--retry", "3", "-o", dest_path, url };

    const result = try std.process.run(gpa, io, .{ .argv = argv });
    defer {
        gpa.free(result.stdout);
        gpa.free(result.stderr);
    }

    return switch (result.term) {
        .exited => |code| if (code != 0) error.HttpRequestFailed,
        else => error.HttpRequestFailed,
    };
}

fn downloadZig(gpa: std.mem.Allocator, io: std.Io, url: []const u8, dest_path: []const u8, token: ?[]const u8) !void {
    var file = try std.Io.Dir.cwd().createFile(io, dest_path, .{});
    defer file.close(io);

    var write_buf: [64 * 1024]u8 = undefined;
    var file_writer = file.writer(io, &write_buf);

    var client: std.http.Client = .{ .allocator = gpa, .io = io };
    defer client.deinit();

    var auth_buf: [4096]u8 = undefined;
    const headers: []const std.http.Header = if (token) |tok|
        &.{.{ .name = "authorization", .value = try std.fmt.bufPrint(&auth_buf, "Bearer {s}", .{tok}) }}
    else
        &.{};

    const result = try client.fetch(.{
        .location = .{ .url = url },
        .method = .GET,
        .response_writer = &file_writer.interface,
        .privileged_headers = headers,
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

    var buf: [64 * 1024]u8 = undefined;
    var reader = file.reader(io, &buf);

    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    var chunk: [64 * 1024]u8 = undefined;
    while (true) {
        const n = try reader.interface.readSliceShort(&chunk);
        if (n == 0) break;
        hasher.update(chunk[0..n]);
    }

    var digest: [32]u8 = undefined;
    hasher.final(&digest);
    return std.fmt.bytesToHex(digest, .lower);
}

fn ensureParentDir(io: std.Io, path: []const u8) !void {
    if (std.fs.path.dirname(path)) |dir| {
        try std.Io.Dir.cwd().createDirPath(io, dir);
    }
}
