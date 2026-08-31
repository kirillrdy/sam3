const std = @import("std");
const sam3 = @import("sam3");
const protocol = @import("protocol.zig");
const Io = std.Io;

pub const Options = struct {
    host: []const u8 = "127.0.0.1",
    port: u16 = 3000,

    example_path: ?[]const u8 = null,
};

pub const Assets = struct {
    index_html: []const u8,
    client_wasm: []const u8,
};

const max_upload = 32 * 1024 * 1024;

const recv_buffer_size = 16 * 1024;
const send_buffer_size = 16 * 1024;
const body_buffer_size = 64 * 1024;
const connection_buffer_size = recv_buffer_size + send_buffer_size + body_buffer_size;

const max_points = 32;

pub fn run(
    gpa: std.mem.Allocator,
    io: Io,
    model: *sam3.Model,
    assets: Assets,
    options: Options,
) !void {
    var server: Server = .{
        .gpa = gpa,
        .io = io,
        .model = model,
        .assets = assets,
        .options = options,
    };
    defer dropCache(&server.cache);
    defer dropCache(&server.concept_cache);

    const address = try Io.net.IpAddress.parseIp4(options.host, options.port);
    var listener = try address.listen(io, .{ .reuse_address = true });
    defer listener.deinit(io);

    var connections: Io.Group = .init;
    defer connections.cancel(io);

    std.debug.print("  Serving http://{s}:{d}/\n\n", .{ options.host, options.port });

    while (true) {
        const stream = listener.accept(io) catch |err| {
            std.debug.print("  ! accept failed: {t}\n", .{err});
            continue;
        };
        connections.concurrent(io, Server.converse, .{ &server, stream }) catch |err| {
            std.debug.print("  ! could not serve a connection: {t}\n", .{err});
            stream.close(io);
        };
    }
}

const Server = struct {
    gpa: std.mem.Allocator,
    io: Io,
    model: *sam3.Model,
    assets: Assets,
    options: Options,

    mutex: Io.Mutex = .init,

    cache: ?Cache(sam3.Embedding) = null,
    concept_cache: ?Cache(sam3.ConceptEmbedding) = null,

    fn converse(self: *Server, stream: Io.net.Stream) void {
        defer stream.close(self.io);

        const buffers = self.gpa.alloc(u8, connection_buffer_size) catch return;
        defer self.gpa.free(buffers);

        var reader = stream.reader(self.io, buffers[0..recv_buffer_size]);
        var writer = stream.writer(self.io, buffers[recv_buffer_size..][0..send_buffer_size]);
        const body_buffer = buffers[recv_buffer_size + send_buffer_size ..];
        var http: std.http.Server = .init(&reader.interface, &writer.interface);

        while (true) {
            var request = http.receiveHead() catch return;
            self.respond(&request, body_buffer) catch |err| {
                std.debug.print("  ! {s}: {t}\n", .{ request.head.target, err });
                return;
            };
        }
    }

    fn respond(self: *Server, request: *std.http.Server.Request, body_buffer: []u8) !void {
        const target = request.head.target;
        const path = target[0 .. std.mem.indexOfScalar(u8, target, '?') orelse target.len];
        const query = target[@min(path.len + 1, target.len)..];

        if (request.head.method == .POST and std.mem.eql(u8, path, "/segment")) {
            return self.segment(request, query, body_buffer);
        }
        if (request.head.method == .POST and std.mem.eql(u8, path, "/lookup")) {
            return self.lookup(request, query, body_buffer);
        }
        if (request.head.method != .GET and request.head.method != .HEAD) {
            return request.respond("method not allowed\n", .{
                .status = .method_not_allowed,
                .keep_alive = false,
            });
        }

        if (std.mem.eql(u8, path, "/")) {
            return serveAsset(request, self.assets.index_html, "text/html; charset=utf-8");
        }
        if (std.mem.eql(u8, path, "/client.wasm")) {
            return serveAsset(request, self.assets.client_wasm, "application/wasm");
        }
        if (std.mem.eql(u8, path, "/example.png")) return self.serveExample(request);

        return request.respond("not found\n", .{ .status = .not_found });
    }

    fn serveExample(self: *Server, request: *std.http.Server.Request) !void {
        const path = self.options.example_path orelse
            return request.respond("no example image\n", .{ .status = .not_found });

        const bytes = Io.Dir.cwd().readFileAlloc(self.io, path, self.gpa, .limited(max_upload)) catch
            return request.respond("no example image\n", .{ .status = .not_found });
        defer self.gpa.free(bytes);

        return request.respond(bytes, .{
            .extra_headers = &.{.{ .name = "content-type", .value = "image/png" }},
        });
    }

    fn segment(
        self: *Server,
        request: *std.http.Server.Request,
        query: []const u8,
        body_buffer: []u8,
    ) !void {
        var points: [max_points]sam3.Point = undefined;
        const prompt = parsePoints(query, &points) catch
            return request.respond("bad prompt\n", .{ .status = .bad_request });
        if (prompt.len == 0) {
            return request.respond("no points in prompt\n", .{ .status = .bad_request });
        }

        const body_reader = try request.readerExpectContinue(body_buffer);
        const body = body_reader.allocRemaining(self.gpa, .limited(max_upload)) catch
            return request.respond("upload too large\n", .{
                .status = .payload_too_large,

                .keep_alive = false,
            });
        defer self.gpa.free(body);

        try self.mutex.lock(self.io);
        defer self.mutex.unlock(self.io);

        const embedding = self.encode(body, false) catch |err| {
            std.debug.print("  ! could not encode the frame: {t}: {s}\n", .{ err, sam3.onnx.lastError() });
            return request.respond("image encoder failed\n", .{ .status = .internal_server_error });
        };

        const started = Io.Timestamp.now(self.io, .awake);
        var masks = self.model.decode(embedding, prompt) catch |err| {
            std.debug.print("  ! decoder failed: {t}: {s}\n", .{ err, sam3.onnx.lastError() });
            return request.respond("segmentation failed\n", .{ .status = .internal_server_error });
        };
        defer masks.deinit();
        std.debug.print("  {d} point(s) -> {d} masks in {d:.2} s\n", .{
            prompt.len,
            masks.count,
            secondsSince(self.io, started),
        });

        return self.respondMasks(request, masks);
    }

    fn encode(self: *Server, body: []const u8, comptime concept: bool) !(if (concept) sam3.ConceptEmbedding else sam3.Embedding) {
        const cache = if (concept) &self.concept_cache else &self.cache;
        const hash = std.hash.Wyhash.hash(0, body);
        if (cache.*) |cached| if (cached.hash == hash) return cached.embedding;
        // Both encoder embeddings are large GPU allocations. Retain only the
        // prompting mode currently in use so switching modes cannot exhaust
        // device memory.
        dropCache(if (concept) &self.cache else &self.concept_cache);

        var img = try sam3.decode(self.gpa, body);
        defer img.deinit(self.gpa);

        const started = Io.Timestamp.now(self.io, .awake);
        const embedding = if (concept) try self.model.encodeConcept(img) else try self.model.encode(img);
        std.debug.print("  {s}encoded {d}x{d} in {d:.2} s\n", .{
            if (concept) "concept-" else "",
            img.width,
            img.height,
            secondsSince(self.io, started),
        });

        dropCache(cache);
        cache.* = .{ .hash = hash, .embedding = embedding };
        return embedding;
    }

    fn lookup(
        self: *Server,
        request: *std.http.Server.Request,
        query: []const u8,
        body_buffer: []u8,
    ) !void {
        var phrase_buffer: [256]u8 = undefined;
        const phrase = parseText(query, &phrase_buffer) catch
            return request.respond("bad text prompt\n", .{ .status = .bad_request });
        if (phrase.len == 0) {
            return request.respond("no text prompt\n", .{ .status = .bad_request });
        }

        const body_reader = try request.readerExpectContinue(body_buffer);
        const body = body_reader.allocRemaining(self.gpa, .limited(max_upload)) catch
            return request.respond("upload too large\n", .{
                .status = .payload_too_large,
                .keep_alive = false,
            });
        defer self.gpa.free(body);

        try self.mutex.lock(self.io);
        defer self.mutex.unlock(self.io);

        const embedding = self.encode(body, true) catch |err| {
            std.debug.print("  ! concept encoder failed: {t}: {s}\n", .{ err, sam3.onnx.lastError() });
            return request.respond("could not encode that image\n", .{ .status = .bad_request });
        };

        const started = Io.Timestamp.now(self.io, .awake);
        var masks = self.model.lookup(embedding, phrase, 0.5) catch |err| {
            std.debug.print("  ! text lookup failed: {t}: {s}\n", .{ err, sam3.onnx.lastError() });
            return request.respond("text lookup failed\n", .{ .status = .internal_server_error });
        };
        defer masks.deinit();
        std.debug.print("  \"{s}\" -> {d} object(s) in {d:.2} s\n", .{
            phrase,
            masks.count,
            secondsSince(self.io, started),
        });

        return self.respondMasks(request, masks);
    }

    fn respondMasks(self: *Server, request: *std.http.Server.Request, masks: sam3.Masks) !void {
        const payload = try serialize(self.gpa, masks);
        defer self.gpa.free(payload);
        return request.respond(payload, .{
            .extra_headers = &.{
                .{ .name = "content-type", .value = "application/octet-stream" },
                .{ .name = "cache-control", .value = "no-store" },
            },
        });
    }
};

fn Cache(comptime Embedding: type) type {
    return struct { hash: u64, embedding: Embedding };
}

fn dropCache(cache: anytype) void {
    if (cache.*) |*cached| cached.embedding.deinit();
    cache.* = null;
}

fn serveAsset(request: *std.http.Server.Request, contents: []const u8, content_type: []const u8) !void {
    return request.respond(contents, .{
        .extra_headers = &.{
            .{ .name = "content-type", .value = content_type },
            .{ .name = "cache-control", .value = "no-store" },
        },
    });
}

fn parseText(query: []const u8, out: []u8) ![]const u8 {
    var fields = std.mem.splitScalar(u8, query, '&');
    while (fields.next()) |field| {
        if (!std.mem.startsWith(u8, field, "text=")) continue;
        const encoded = field[5..];
        var written: usize = 0;
        var i: usize = 0;
        while (i < encoded.len) {
            if (written == out.len) return error.TextTooLong;
            if (encoded[i] == '%') {
                if (i + 2 >= encoded.len) return error.BadEscape;
                out[written] = try std.fmt.parseInt(u8, encoded[i + 1 .. i + 3], 16);
                i += 3;
            } else {
                out[written] = if (encoded[i] == '+') ' ' else encoded[i];
                i += 1;
            }
            written += 1;
        }
        return std.mem.trim(u8, out[0..written], " \t\r\n");
    }
    return "";
}

fn parsePoints(query: []const u8, out: []sam3.Point) ![]const sam3.Point {
    var count: usize = 0;
    var fields = std.mem.splitScalar(u8, query, '&');
    while (fields.next()) |field| {
        if (!std.mem.startsWith(u8, field, "p=")) continue;
        if (count == out.len) return error.TooManyPoints;

        var parts = std.mem.splitScalar(u8, field[2..], ',');
        const x = parts.next() orelse return error.MalformedPoint;
        const y = parts.next() orelse return error.MalformedPoint;
        const label = parts.next() orelse "1";
        if (parts.next() != null) return error.MalformedPoint;

        out[count] = .{
            .x = try std.fmt.parseFloat(f32, x),
            .y = try std.fmt.parseFloat(f32, y),
            .label = try std.fmt.parseInt(i64, label, 10),
        };
        count += 1;
    }
    return out[0..count];
}

fn serialize(gpa: std.mem.Allocator, masks: sam3.Masks) ![]u8 {
    const header: protocol.Header = .{
        .count = @intCast(masks.count),
        .width = @intCast(masks.width),
        .height = @intCast(masks.height),
        .object_score = masks.object_score,
    };

    const payload = try gpa.alloc(u8, header.responseSize());
    errdefer gpa.free(payload);

    header.write(payload[0..protocol.Header.size]);

    var offset: usize = protocol.Header.size;
    for (masks.scores[0..masks.count]) |score| {
        std.mem.writeInt(u32, payload[offset..][0..4], @bitCast(score), .little);
        offset += 4;
    }
    for (masks.logits) |logit| {
        std.mem.writeInt(u32, payload[offset..][0..4], @bitCast(logit), .little);
        offset += 4;
    }
    return payload;
}

fn secondsSince(io: Io, started: Io.Timestamp) f64 {
    const elapsed = started.durationTo(Io.Timestamp.now(io, .awake));
    return @as(f64, @floatFromInt(elapsed.nanoseconds)) / 1e9;
}

test "a prompt is read back as the points it names" {
    var buffer: [8]sam3.Point = undefined;

    const points = try parsePoints("p=0.25,0.5,1&p=0.75,0.5,0", &buffer);
    try std.testing.expectEqual(@as(usize, 2), points.len);
    try std.testing.expectEqual(@as(f32, 0.25), points[0].x);
    try std.testing.expectEqual(@as(i64, 1), points[0].label);
    try std.testing.expectEqual(@as(f32, 0.75), points[1].x);
    try std.testing.expectEqual(@as(i64, 0), points[1].label);

    const defaulted = try parsePoints("mode=x&p=0.5,0.5", &buffer);
    try std.testing.expectEqual(@as(usize, 1), defaulted.len);
    try std.testing.expectEqual(@as(i64, 1), defaulted[0].label);

    try std.testing.expectEqual(@as(usize, 0), (try parsePoints("", &buffer)).len);
}

test "a malformed point is refused rather than guessed at" {
    var buffer: [8]sam3.Point = undefined;
    try std.testing.expectError(error.MalformedPoint, parsePoints("p=0.5", &buffer));
    try std.testing.expectError(error.MalformedPoint, parsePoints("p=1,2,3,4", &buffer));
    try std.testing.expect(std.meta.isError(parsePoints("p=x,y,1", &buffer)));

    var small: [1]sam3.Point = undefined;
    try std.testing.expectError(error.TooManyPoints, parsePoints("p=0,0,1&p=1,1,1", &small));
}

test "a URL-encoded text prompt is decoded" {
    var buffer: [64]u8 = undefined;
    try std.testing.expectEqualStrings("red car", try parseText("text=red%20car", &buffer));
    try std.testing.expectEqualStrings("two cats", try parseText("x=1&text=two+cats", &buffer));
    try std.testing.expectError(error.BadEscape, parseText("text=bad%2", &buffer));
}
