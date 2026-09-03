const std = @import("std");
const sam3 = @import("sam3");
const protocol = @import("protocol.zig");
const tracking = sam3.tracking;
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

/// How sure the text decoder has to be that a query found something.
const threshold = 0.5;

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
        .tracker = .init(gpa, Server.configFor(.idle)),
    };
    defer dropCache(&server.cache);
    defer dropCache(&server.concept_cache);
    defer server.tracker.deinit();

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

    /// What is being followed through the video the browser is feeding in,
    /// frame by frame. There is one model and one of these, because the
    /// mutex lets one frame through at a time anyway.
    tracker: tracking.Tracker,
    mode: Mode = .idle,
    /// The track the viewer is currently naming with clicks. Their next click
    /// sharpens this one rather than starting another object.
    building: ?u32 = null,

    /// How a video is being prompted. A phrase finds every object it names on
    /// each frame; a click names one object and it carries itself forward.
    const Mode = enum { idle, text, points };

    /// A phrase re-finds its objects on every frame, so a detection that
    /// flickers on one frame is worth waiting out. A clicked object cannot be
    /// re-found -- it only ever propagates from the frame before -- so it has
    /// to be followed from the first frame it is on.
    fn configFor(mode: Mode) tracking.Config {
        return switch (mode) {
            .idle, .points => .{ .min_hits = 1 },
            .text => .{ .min_hits = 2 },
        };
    }

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
        if (request.head.method == .POST and std.mem.eql(u8, path, "/track")) {
            return self.follow(request, query, body_buffer);
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

    fn readBody(
        self: *Server,
        request: *std.http.Server.Request,
        body_buffer: []u8,
    ) !?[]u8 {
        const body_reader = try request.readerExpectContinue(body_buffer);
        return body_reader.allocRemaining(self.gpa, .limited(max_upload)) catch {
            try request.respond("upload too large\n", .{
                .status = .payload_too_large,
                .keep_alive = false,
            });
            return null;
        };
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

        const body = try self.readBody(request, body_buffer) orelse return;
        defer self.gpa.free(body);

        try self.mutex.lock(self.io);
        defer self.mutex.unlock(self.io);

        const embedding = self.encode(body, false) catch |err| {
            std.debug.print("  ! could not encode the frame: {t}: {s}\n", .{ err, sam3.onnx.lastError() });
            return request.respond("image encoder failed\n", .{ .status = .internal_server_error });
        };

        const started = Io.Timestamp.now(self.io, .awake);
        var masks = self.model.decode(embedding, prompt, null) catch |err| {
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
        const body = try self.readBody(request, body_buffer) orelse return;
        defer self.gpa.free(body);

        try self.mutex.lock(self.io);
        defer self.mutex.unlock(self.io);

        const embedding = self.encode(body, true) catch |err| {
            std.debug.print("  ! concept encoder failed: {t}: {s}\n", .{ err, sam3.onnx.lastError() });
            return request.respond("could not encode that image\n", .{ .status = .bad_request });
        };

        const started = Io.Timestamp.now(self.io, .awake);
        var masks = self.model.lookup(embedding, phrase, threshold) catch |err| {
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

    /// One frame of a video, answered with everything currently being followed
    /// through it -- each object under the number it has had since it first
    /// appeared.
    ///
    /// The frame is prompted one of three ways, and which one it is says what
    /// the browser is doing:
    ///
    ///   * `text=` -- find these objects on this frame, and tie them to the
    ///     ones already being followed.
    ///   * `p=` -- the viewer pointed at an object on this frame. Segment it
    ///     and start following it.
    ///   * neither -- carry everything already being followed onto this frame.
    ///
    /// `reset=1` forgets the previous video first.
    fn follow(
        self: *Server,
        request: *std.http.Server.Request,
        query: []const u8,
        body_buffer: []u8,
    ) !void {
        // Every last thing the query says has to be read off it here, before
        // the body: it points into the buffer the request head arrived in, and
        // reading the body is what refills that buffer.
        var points: [max_points]sam3.Point = undefined;
        var phrase_buffer: [256]u8 = undefined;
        const parsed = parseFollow(query, &points, &phrase_buffer);

        // Then take the upload even if the answer is going to be a refusal.
        // Answering while the browser is still sending the frame reaches it as
        // a broken connection rather than as the reason it was refused.
        const body = try self.readBody(request, body_buffer) orelse return;
        defer self.gpa.free(body);

        const followed = parsed catch |err| return request.respond(switch (err) {
            error.AmbiguousPrompt => "a frame is prompted by a phrase or by points, not both\n",
            error.TextTooLong, error.BadEscape => "bad text prompt\n",
            else => "bad prompt\n",
        }, .{ .status = .bad_request });

        const prompt = followed.prompt;
        const phrase = followed.phrase;

        try self.mutex.lock(self.io);
        defer self.mutex.unlock(self.io);

        if (followed.reset) self.startOver(.idle);
        if (phrase.len != 0) self.startOver(.text);
        if (prompt.len != 0) self.startOver(.points);

        const started = Io.Timestamp.now(self.io, .awake);
        if (phrase.len != 0) {
            self.findOn(body, phrase) catch |err| {
                std.debug.print("  ! tracking by phrase failed: {t}: {s}\n", .{ err, sam3.onnx.lastError() });
                return request.respond("could not follow that frame\n", .{ .status = .internal_server_error });
            };
        } else if (prompt.len != 0) {
            self.pointAt(body, prompt, followed.names_another) catch |err| {
                std.debug.print("  ! could not take that prompt: {t}: {s}\n", .{ err, sam3.onnx.lastError() });
                return request.respond("could not segment what was pointed at\n", .{ .status = .internal_server_error });
            };
        } else if (self.mode == .points and !self.tracker.isEmpty()) {
            self.carryForward(body) catch |err| {
                std.debug.print("  ! could not carry the tracks forward: {t}: {s}\n", .{ err, sam3.onnx.lastError() });
                return request.respond("could not follow that frame\n", .{ .status = .internal_server_error });
            };
        } else {
            return request.respond("nothing is being followed yet\n", .{ .status = .bad_request });
        }

        std.debug.print("  frame -> {d} object(s) in {d:.2} s\n", .{
            self.visibleCount(),
            secondsSince(self.io, started),
        });
        return self.respondTracks(request);
    }

    /// Switches how the video is being prompted, forgetting what was being
    /// followed under the old prompt. Tracks from a phrase and tracks from a
    /// click cannot be continued into one another.
    fn startOver(self: *Server, mode: Mode) void {
        if (self.mode == mode and mode != .idle) return;
        self.mode = mode;
        self.building = null;
        self.tracker.reset();
        self.tracker.config = configFor(mode);
    }

    /// Finds every object the phrase names on this frame, and ties them to the
    /// ones already being followed.
    fn findOn(self: *Server, body: []const u8, phrase: []const u8) !void {
        const embedding = try self.encode(body, true);
        var masks = try self.model.lookup(embedding, phrase, threshold);
        defer masks.deinit();

        var detections: std.ArrayList(tracking.Detection) = .empty;
        defer detections.deinit(self.gpa);
        try detections.ensureTotalCapacity(self.gpa, masks.count);
        for (0..masks.count) |i| {
            detections.appendAssumeCapacity(.{
                .box = masks.boxes[i],
                .score = masks.scores[i],
                .plane = masks.plane(i),
            });
        }

        try self.tracker.update(detections.items, masks.width, masks.height);
    }

    /// Segments what the viewer pointed at and starts following it.
    ///
    /// A click that adds to the prompt they were already making replaces what
    /// that object is, however little the sharper reading has in common with
    /// the first one -- correcting a click that caught a whisker instead of a
    /// cat is the whole reason to click twice. Only a click that says it names
    /// something else starts another track.
    fn pointAt(
        self: *Server,
        body: []const u8,
        prompt: []const sam3.Point,
        names_another: bool,
    ) !void {
        const embedding = try self.encode(body, false);
        var masks = try self.model.decode(embedding, prompt, null);
        defer masks.deinit();
        if (masks.count == 0) return;

        const chosen = masks.best();
        const detection: tracking.Detection = .{
            .box = masks.boxes[chosen],
            .score = masks.scores[chosen],
            .plane = masks.plane(chosen),
        };

        if (!names_another) {
            if (self.building) |id| {
                if (self.tracker.refine(id, detection)) return;
            }
        }
        self.building = try self.tracker.seed(detection, masks.width, masks.height);
    }

    /// Carries every track onto this frame: the mask each one left on the frame
    /// before says where to prompt this one.
    fn carryForward(self: *Server, body: []const u8) !void {
        const embedding = try self.encode(body, false);

        const followed = self.tracker.tracks.items;

        // The masks have to outlive the carry, because the detections point
        // into them. Their logits are their own allocations, so growing this
        // list does not move what the detections refer to.
        var answers: std.ArrayList(sam3.Masks) = .empty;
        defer {
            for (answers.items) |*masks| masks.deinit();
            answers.deinit(self.gpa);
        }

        // One slot per track, in the order the tracker holds them.
        const carried = try self.gpa.alloc(?tracking.Detection, followed.len);
        defer self.gpa.free(carried);
        @memset(carried, null);

        for (followed, carried) |track, *slot| {
            const point = (try self.tracker.promptFrom(track)) orelse continue;

            var masks = try self.model.decode(embedding, &.{point}, null);
            // The decoder reads a point several ways -- the part, the object,
            // the whole -- and once the object is gone it reads it as whatever
            // is standing there instead. The tracker says which of those, if
            // any, is still the track.
            const detection = self.tracker.carriedFrom(track, masks) orelse {
                masks.deinit();
                continue;
            };
            // Two objects that have become one are one object: let the first
            // track keep it rather than following the same pixels twice.
            if (alreadyClaimed(carried, detection.plane)) {
                masks.deinit();
                continue;
            }

            // From here the masks belong to the list, and are freed with it.
            answers.append(self.gpa, masks) catch |err| {
                masks.deinit();
                return err;
            };
            slot.* = detection;
        }

        self.tracker.carry(carried);
    }

    fn visibleCount(self: *Server) usize {
        var count: usize = 0;
        for (self.tracker.tracks.items) |track| {
            if (track.isVisible(self.tracker.config)) count += 1;
        }
        return count;
    }

    fn respondTracks(self: *Server, request: *std.http.Server.Request) !void {
        const count = self.visibleCount();

        const instances = try self.gpa.alloc(protocol.Instance, count);
        defer self.gpa.free(instances);
        const planes = try self.gpa.alloc([]const f32, count);
        defer self.gpa.free(planes);

        var out: usize = 0;
        var strongest: f32 = 0;
        for (self.tracker.tracks.items) |track| {
            if (!track.isVisible(self.tracker.config)) continue;
            const box = track.box;
            instances[out] = .{
                .id = track.id,
                .score = track.score,
                .box = .{ box.x0, box.y0, box.x1, box.y1 },
            };
            planes[out] = track.mask;
            strongest = @max(strongest, track.score);
            out += 1;
        }

        const payload = try serialize(self.gpa, .{
            .width = self.tracker.width,
            .height = self.tracker.height,
            .object_score = strongest,
            .instances = instances,
            .planes = planes,
        });
        defer self.gpa.free(payload);
        return respondPayload(request, payload);
    }

    fn respondMasks(self: *Server, request: *std.http.Server.Request, masks: sam3.Masks) !void {
        const instances = try self.gpa.alloc(protocol.Instance, masks.count);
        defer self.gpa.free(instances);
        const planes = try self.gpa.alloc([]const f32, masks.count);
        defer self.gpa.free(planes);

        for (0..masks.count) |i| {
            const box = masks.boxes[i];
            // A still image has nothing to follow, so a mask is known by where
            // it sits in the reply.
            instances[i] = .{
                .id = @intCast(i),
                .score = masks.scores[i],
                .box = .{ box.x0, box.y0, box.x1, box.y1 },
            };
            planes[i] = masks.plane(i);
        }

        const payload = try serialize(self.gpa, .{
            .width = masks.width,
            .height = masks.height,
            .object_score = masks.object_score,
            .instances = instances,
            .planes = planes,
        });
        defer self.gpa.free(payload);
        return respondPayload(request, payload);
    }
};

/// True when a track already carried onto this frame covers all but a sliver
/// of the same pixels.
fn alreadyClaimed(carried: []const ?tracking.Detection, plane: []const f32) bool {
    for (carried) |slot| {
        const detection = slot orelse continue;
        if (sam3.maskIou(detection.plane, plane) > 0.9) return true;
    }
    return false;
}

fn respondPayload(request: *std.http.Server.Request, payload: []const u8) !void {
    return request.respond(payload, .{
        .extra_headers = &.{
            .{ .name = "content-type", .value = "application/octet-stream" },
            .{ .name = "cache-control", .value = "no-store" },
        },
    });
}

/// Everything a `/track` query says about the frame that follows it.
const Followed = struct {
    /// Where the viewer pointed, if they did.
    prompt: []const sam3.Point,
    /// What to look for, if a phrase was given.
    phrase: []const u8,
    /// Forget the video that was being followed before this frame.
    reset: bool,
    /// This click names an object, rather than sharpening the one the viewer
    /// has been clicking on.
    names_another: bool,
};

fn parseFollow(
    query: []const u8,
    points: []sam3.Point,
    phrase_buffer: []u8,
) !Followed {
    const prompt = try parsePoints(query, points);
    const phrase = try parseText(query, phrase_buffer);
    if (phrase.len != 0 and prompt.len != 0) return error.AmbiguousPrompt;
    return .{
        .prompt = prompt,
        .phrase = phrase,
        .reset = hasFlag(query, "reset"),
        .names_another = hasFlag(query, "new"),
    };
}

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

fn field(query: []const u8, name: []const u8) ?[]const u8 {
    var fields = std.mem.splitScalar(u8, query, '&');
    while (fields.next()) |candidate| {
        if (!std.mem.startsWith(u8, candidate, name)) continue;
        if (candidate.len <= name.len or candidate[name.len] != '=') continue;
        return candidate[name.len + 1 ..];
    }
    return null;
}

fn hasFlag(query: []const u8, name: []const u8) bool {
    const value = field(query, name) orelse return false;
    return !std.mem.eql(u8, value, "0");
}

fn parseText(query: []const u8, out: []u8) ![]const u8 {
    const encoded = field(query, "text") orelse return "";
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

fn parsePoints(query: []const u8, out: []sam3.Point) ![]const sam3.Point {
    var count: usize = 0;
    var fields = std.mem.splitScalar(u8, query, '&');
    while (fields.next()) |candidate| {
        if (!std.mem.startsWith(u8, candidate, "p=")) continue;
        if (count == out.len) return error.TooManyPoints;

        var parts = std.mem.splitScalar(u8, candidate[2..], ',');
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

/// One frame's worth of answer, whatever produced it.
const Frame = struct {
    width: usize,
    height: usize,
    object_score: f32,
    instances: []const protocol.Instance,
    planes: []const []const f32,
};

fn serialize(gpa: std.mem.Allocator, frame: Frame) ![]u8 {
    std.debug.assert(frame.instances.len == frame.planes.len);

    const header: protocol.Header = .{
        .count = @intCast(frame.instances.len),
        .width = @intCast(frame.width),
        .height = @intCast(frame.height),
        .object_score = frame.object_score,
    };

    const payload = try gpa.alloc(u8, header.responseSize());
    errdefer gpa.free(payload);

    header.write(payload[0..protocol.Header.size]);

    for (frame.instances, 0..) |instance, i| {
        instance.write(payload[protocol.Header.instanceOffset(i)..][0..protocol.Instance.size]);
    }

    const stride = frame.width * frame.height;
    for (frame.planes, 0..) |plane, i| {
        std.debug.assert(plane.len == stride);
        var offset = header.planeOffset(i);
        for (plane) |logit| {
            std.mem.writeInt(u32, payload[offset..][0..4], @bitCast(logit), .little);
            offset += 4;
        }
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
    try std.testing.expectEqualStrings("", try parseText("p=0.5,0.5,1", &buffer));
    try std.testing.expectError(error.BadEscape, parseText("text=bad%2", &buffer));
}

test "a frame is prompted by a phrase or by points, and never by both" {
    var points: [8]sam3.Point = undefined;
    var phrase: [64]u8 = undefined;

    const clicked = try parseFollow("p=0.5,0.5,1&new=1", &points, &phrase);
    try std.testing.expectEqual(@as(usize, 1), clicked.prompt.len);
    try std.testing.expectEqualStrings("", clicked.phrase);
    try std.testing.expect(clicked.names_another);
    try std.testing.expect(!clicked.reset);

    const spoken = try parseFollow("text=red+car&reset=1", &points, &phrase);
    try std.testing.expectEqual(@as(usize, 0), spoken.prompt.len);
    try std.testing.expectEqualStrings("red car", spoken.phrase);
    try std.testing.expect(spoken.reset);

    // A frame with no prompt at all carries on with what is already followed.
    const carried = try parseFollow("", &points, &phrase);
    try std.testing.expectEqual(@as(usize, 0), carried.prompt.len);
    try std.testing.expectEqualStrings("", carried.phrase);

    try std.testing.expectError(
        error.AmbiguousPrompt,
        parseFollow("text=cat&p=0.5,0.5,1", &points, &phrase),
    );
}

test "a field is only itself, not anything it is a prefix of" {
    try std.testing.expectEqualStrings("1", field("reset=1", "reset").?);
    try std.testing.expectEqualStrings("1", field("text=cat&reset=1", "reset").?);
    try std.testing.expectEqual(null, field("resetting=1", "reset"));
    try std.testing.expectEqual(null, field("reset", "reset"));

    try std.testing.expect(hasFlag("reset=1", "reset"));
    try std.testing.expect(!hasFlag("reset=0", "reset"));
    try std.testing.expect(!hasFlag("text=cat", "reset"));
}

test "a frame serializes to its header, its instances, and then its planes" {
    const gpa = std.testing.allocator;

    const first = [_]f32{ 1, -1, -1, 2 };
    const second = [_]f32{ -1, -1, 3, -1 };
    const payload = try serialize(gpa, .{
        .width = 2,
        .height = 2,
        .object_score = 0.5,
        .instances = &.{
            .{ .id = 4, .score = 0.9, .box = .{ 0, 0, 0.5, 1 } },
            .{ .id = 7, .score = 0.8, .box = .{ 0.5, 0.5, 1, 1 } },
        },
        .planes = &.{ &first, &second },
    });
    defer gpa.free(payload);

    const header = try protocol.Header.parse(payload);
    try std.testing.expectEqual(@as(u32, 2), header.count);
    try std.testing.expectEqual(@as(u32, 2), header.width);
    try std.testing.expectEqual(@as(f32, 0.5), header.object_score);
    try std.testing.expectEqual(payload.len, header.responseSize());

    const instance = protocol.Instance.parse(
        payload[protocol.Header.instanceOffset(1)..][0..protocol.Instance.size],
    );
    try std.testing.expectEqual(@as(u32, 7), instance.id);
    try std.testing.expectEqual(@as(f32, 0.8), instance.score);
    try std.testing.expectEqualSlices(f32, &.{ 0.5, 0.5, 1, 1 }, &instance.box);

    for ([_][]const f32{ &first, &second }, 0..) |plane, i| {
        var offset = header.planeOffset(i);
        for (plane) |logit| {
            const read: f32 = @bitCast(std.mem.readInt(u32, payload[offset..][0..4], .little));
            try std.testing.expectEqual(logit, read);
            offset += 4;
        }
    }
}
