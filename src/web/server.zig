//! The HTTP server behind the web UI.
//!
//! Three static files and one endpoint. `POST /segment` carries an image in its
//! body and the prompt in its query string, and answers with the decoder's mask
//! logits in the format `protocol.zig` describes -- the client does the
//! upsampling and the drawing, so the answer is the same quarter of a megabyte
//! whatever the image was.
//!
//! Connections are served concurrently and the model is not. A browser opens
//! several sockets to a page and leaves them idle, so a server that finished one
//! connection before accepting the next would be blocked by a socket nobody
//! intends to send anything on. What is behind the requests, on the other hand,
//! is a 1.7 GiB graph over one `onnx.Session`, no more parallel than the device
//! under it: `Server.mutex` is what holds the clicks to one at a time.
//!
//! The vision encoder is also where nearly all of the time goes -- seconds on a
//! CPU -- while the prompt it is independent of changes with every click. So the
//! last frame's pyramid is kept: click again on the same image and only the
//! 21 MiB decoder runs, which is the difference between a usable UI and one that
//! stalls on every click.

const std = @import("std");
const sam3 = @import("../sam3.zig");
const image_io = @import("../io/image.zig");
const protocol = @import("protocol.zig");
const Io = std.Io;

pub const Options = struct {
    /// Loopback by default: this serves an unauthenticated endpoint that will
    /// decode any image it is handed, which is not something to put on a
    /// network without meaning to.
    host: []const u8 = "127.0.0.1",
    port: u16 = 3000,
    /// An image to offer as a starting point, or null for none. Served as-is at
    /// `/example.png`.
    example_path: ?[]const u8 = null,
};

/// The page and the wasm module, embedded in the executable by `main.zig` so
/// that the server has no data directory to be run from.
pub const Assets = struct {
    index_html: []const u8,
    client_wasm: []const u8,
};

/// Bigger than a photograph, smaller than a denial of service.
const max_upload = 32 * 1024 * 1024;

/// One connection's working memory, on the heap rather than on the task's
/// stack, which is not this program's to size. Enough head room for the
/// request line and headers of anything a browser sends, and after that it is
/// only the size of the reads.
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
    defer server.dropCache();

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
    /// Held for as long as the model is in use, which includes the whole of a
    /// `/segment` request: the pyramid a decode reads belongs to `cache`, and
    /// nothing may replace it in the meantime.
    mutex: Io.Mutex = .init,
    /// The last frame's feature pyramid, and the hash of the bytes it came from.
    cache: ?Cache = null,

    const Cache = struct {
        hash: u64,
        embedding: sam3.Embedding,
    };

    /// Serves every request on one connection until the client goes away or
    /// says something this cannot parse. Nothing in here is fatal to the
    /// server: a failed connection is dropped and the others carry on.
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
        if (request.head.method != .GET and request.head.method != .HEAD) {
            // Closing rather than keeping the connection because the body of a
            // method this does not serve is never read: leaving it in the
            // stream would have the next request parsed out of the middle of it.
            return request.respond("method not allowed\n", .{
                .status = .method_not_allowed,
                .keep_alive = false,
            });
        }
        // Both of these are compiled into the executable, so a rebuild is what
        // changes them -- and a browser that had cached the previous page would
        // go on running it against a server that no longer matches.
        if (std.mem.eql(u8, path, "/")) {
            return request.respond(self.assets.index_html, .{
                .extra_headers = &.{
                    .{ .name = "content-type", .value = "text/html; charset=utf-8" },
                    .{ .name = "cache-control", .value = "no-store" },
                },
            });
        }
        if (std.mem.eql(u8, path, "/client.wasm")) {
            return request.respond(self.assets.client_wasm, .{
                .extra_headers = &.{
                    .{ .name = "content-type", .value = "application/wasm" },
                    .{ .name = "cache-control", .value = "no-store" },
                },
            });
        }
        if (std.mem.eql(u8, path, "/example.png")) return self.serveExample(request);

        return request.respond("not found\n", .{ .status = .not_found });
    }

    /// The sample image the build downloaded, read on demand rather than
    /// embedded: it lives in the build cache, and a build that never ran
    /// `fetch-examples` simply has nothing to offer here.
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

    // --- Segmentation ------------------------------------------------------

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
                // Whatever is left of the upload is still coming; the only way
                // not to read it as the next request is to hang up.
                .keep_alive = false,
            });
        defer self.gpa.free(body);

        // From here to the end of the decode, the model is this request's --
        // and so is the pyramid `encode` either found in the cache or put
        // there, which is why the lock covers both and not just the run.
        try self.mutex.lock(self.io);
        defer self.mutex.unlock(self.io);

        const embedding = self.encode(body) catch |err| {
            std.debug.print("  ! could not encode the frame: {t}\n", .{err});
            return request.respond("could not read that image\n", .{ .status = .bad_request });
        };

        const started = Io.Timestamp.now(self.io, .awake);
        var masks = self.model.decode(embedding, prompt) catch |err| {
            std.debug.print("  ! decoder failed: {t}\n", .{err});
            return request.respond("segmentation failed\n", .{ .status = .internal_server_error });
        };
        defer masks.deinit();
        std.debug.print("  {d} point(s) -> {d} masks in {d:.2} s\n", .{
            prompt.len,
            masks.count,
            secondsSince(self.io, started),
        });

        const payload = try serialize(self.gpa, masks);
        defer self.gpa.free(payload);

        return request.respond(payload, .{
            .extra_headers = &.{
                .{ .name = "content-type", .value = "application/octet-stream" },
                .{ .name = "cache-control", .value = "no-store" },
            },
        });
    }

    /// The frame's feature pyramid, from the cache when the same bytes came
    /// back. What arrives is the file the browser holds, byte for byte, so the
    /// hash of it identifies the frame exactly.
    fn encode(self: *Server, body: []const u8) !sam3.Embedding {
        const hash = std.hash.Wyhash.hash(0, body);
        if (self.cache) |cached| {
            if (cached.hash == hash) return cached.embedding;
        }

        var img = try image_io.decode(self.gpa, body);
        defer img.deinit();

        const started = Io.Timestamp.now(self.io, .awake);
        const embedding = try self.model.encode(img);
        std.debug.print("  encoded {d}x{d} in {d:.2} s\n", .{
            img.width,
            img.height,
            secondsSince(self.io, started),
        });

        self.dropCache();
        self.cache = .{ .hash = hash, .embedding = embedding };
        return embedding;
    }

    fn dropCache(self: *Server) void {
        if (self.cache) |*cached| cached.embedding.deinit();
        self.cache = null;
    }
};

/// `p=<x>,<y>,<label>`, repeated -- normalised coordinates, and 1 or 0 for a
/// point the mask should include or exclude. Anything else in the query string
/// is ignored; a malformed `p` is an error rather than a point somewhere
/// unintended.
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

/// The masks, as `protocol.zig` lays them out.
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

    // A label is optional, and anything else in the query is not a point.
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
