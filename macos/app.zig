const std = @import("std");
const sam3 = @import("sam3");
const render = sam3.render;
const zigimg = @import("zigimg");

pub const SamCallbacks = extern struct {
    on_open_file: ?*const fn (path: [*:0]const u8) callconv(.c) void,
    on_sample_click: ?*const fn () callconv(.c) void,
    on_mode_change: ?*const fn (mode: c_int) callconv(.c) void,
    on_clear_points: ?*const fn () callconv(.c) void,
    on_find_text: ?*const fn (text: [*:0]const u8) callconv(.c) void,
    on_canvas_click: ?*const fn (norm_x: f32, norm_y: f32, is_positive: c_int) callconv(.c) void,
    on_select_mask: ?*const fn (mask_index: c_int) callconv(.c) void,
};

pub const SamMaskInfo = extern struct {
    score: f32,
    coverage: f32,
};

pub extern fn sam_macos_init(callbacks: *const SamCallbacks) c_int;
pub extern fn sam_macos_run() void;
pub extern fn sam_macos_set_status(text: [*:0]const u8) void;
pub extern fn sam_macos_set_image(rgba_pixels: ?[*]const u8, width: c_int, height: c_int) void;
pub extern fn sam_macos_set_masks(count: c_int, masks: ?[*]const SamMaskInfo, best_index: c_int, selected_index: c_int) void;
pub extern fn sam_macos_set_busy(is_busy: c_int) void;
pub extern fn sam_macos_dispatch_main(func: *const fn (?*anyopaque) callconv(.c) void, ctx: ?*anyopaque) void;

const max_points = 32;

fn Cache(comptime EmbeddingType: type) type {
    return struct { embedding: EmbeddingType };
}

pub const App = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    model: *sam3.Model,
    example_path: []const u8,

    mutex: std.Io.Mutex = .init,
    is_busy: bool = false,

    image: ?zigimg.Image = null,
    frame: []u8 = &.{},

    points: [max_points]sam3.Point = undefined,
    points_len: usize = 0,
    click_mode_add: bool = true,

    cache: ?Cache(sam3.Embedding) = null,
    concept_cache: ?Cache(sam3.ConceptEmbedding) = null,

    masks: ?sam3.Masks = null,
    coverages: []f32 = &.{},
    selected_mask: i32 = -1,
    best_mask_idx: i32 = -1,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, model: *sam3.Model, example_path: []const u8) App {
        return .{
            .allocator = allocator,
            .io = io,
            .model = model,
            .example_path = example_path,
        };
    }

    pub fn deinit(self: *App) void {
        self.mutex.lock(self.io) catch return;
        defer self.mutex.unlock(self.io);

        if (self.cache) |*c| c.embedding.deinit();
        if (self.concept_cache) |*c| c.embedding.deinit();
        if (self.masks) |*m| m.deinit();
        if (self.image) |*img| img.deinit(self.allocator);
        self.allocator.free(self.frame);
        self.allocator.free(self.coverages);
    }

    pub fn start(self: *App) !void {
        g_app = self;

        const callbacks: SamCallbacks = .{
            .on_open_file = &cOpenFile,
            .on_sample_click = &cSampleClick,
            .on_mode_change = &cModeChange,
            .on_clear_points = &cClearPoints,
            .on_find_text = &cFindText,
            .on_canvas_click = &cCanvasClick,
            .on_select_mask = &cSelectMask,
        };

        if (sam_macos_init(&callbacks) != 0) {
            return error.MacosUiInitFailed;
        }

        // Open sample image by default to give user an instant experience
        self.openImageFromPath(self.example_path);

        sam_macos_run();
    }

    pub fn openImageFromPath(self: *App, path: []const u8) void {
        const file_bytes = std.Io.Dir.cwd().readFileAlloc(
            self.io,
            path,
            self.allocator,
            .limited(64 * 1024 * 1024),
        ) catch |err| {
            std.debug.print("Failed to read image file {s}: {t}\n", .{ path, err });
            sam_macos_set_status("Could not open that file.");
            return;
        };
        defer self.allocator.free(file_bytes);

        self.openImageFromBytes(file_bytes);
    }

    pub fn openImageFromBytes(self: *App, bytes: []const u8) void {
        self.mutex.lock(self.io) catch return;
        defer self.mutex.unlock(self.io);

        var decoded = sam3.decode(self.allocator, bytes) catch |err| {
            std.debug.print("Failed to decode image: {t}\n", .{err});
            sam_macos_set_status("That file is not an image this can decode.");
            return;
        };

        if (self.image) |*old| old.deinit(self.allocator);
        self.allocator.free(self.frame);
        if (self.cache) |*c| c.embedding.deinit();
        self.cache = null;
        if (self.concept_cache) |*c| c.embedding.deinit();
        self.concept_cache = null;
        if (self.masks) |*m| m.deinit();
        self.masks = null;

        self.points_len = 0;
        self.selected_mask = -1;
        self.best_mask_idx = -1;

        self.image = decoded;
        self.frame = self.allocator.alloc(u8, decoded.width * decoded.height * 4) catch {
            decoded.deinit(self.allocator);
            self.image = null;
            self.frame = &.{};
            sam_macos_set_status("Out of memory for frame buffer.");
            return;
        };

        self.renderComposite(-1);
        sam_macos_set_image(self.frame.ptr, @intCast(decoded.width), @intCast(decoded.height));
        sam_macos_set_masks(0, null, 0, -1);

        var buf: [128]u8 = undefined;
        const msg = std.fmt.bufPrintZ(&buf, "{d} × {d} — click the object you want.", .{
            decoded.width,
            decoded.height,
        }) catch "Image loaded.";
        sam_macos_set_status(msg);
    }

    pub fn handleCanvasClick(self: *App, norm_x: f32, norm_y: f32, is_positive: c_int) void {
        if (self.is_busy or self.image == null or self.points_len >= max_points) return;

        self.mutex.lock(self.io) catch return;
        self.points[self.points_len] = .{
            .x = std.math.clamp(norm_x, 0.0, 1.0),
            .y = std.math.clamp(norm_y, 0.0, 1.0),
            .label = if (is_positive != 0) 1 else 0,
        };
        self.points_len += 1;

        self.renderComposite(-1);
        sam_macos_set_image(self.frame.ptr, @intCast(self.image.?.width), @intCast(self.image.?.height));
        self.mutex.unlock(self.io);

        self.is_busy = true;
        sam_macos_set_busy(1);
        sam_macos_set_status("Segmenting… the first click on an image also runs the vision encoder.");

        const thread = std.Thread.spawn(.{}, runSegmentWorker, .{self}) catch {
            self.is_busy = false;
            sam_macos_set_busy(0);
            return;
        };
        thread.detach();
    }

    fn runSegmentWorker(self: *App) void {
        const started = std.Io.Timestamp.now(self.io, .awake);

        const embedding = self.ensureEmbedding(false) catch |err| {
            std.debug.print("Vision encoder failed: {t}: {s}\n", .{ err, sam3.onnx.lastError() });
            sam_macos_set_status("Vision encoder failed");
            self.is_busy = false;
            sam_macos_set_busy(0);
            return;
        };

        var masks = self.model.decode(embedding, self.points[0..self.points_len]) catch |err| {
            std.debug.print("Decoder failed: {t}: {s}\n", .{ err, sam3.onnx.lastError() });
            sam_macos_set_status("Segmentation failed");
            self.is_busy = false;
            sam_macos_set_busy(0);
            return;
        };

        self.mutex.lock(self.io) catch return;
        defer self.mutex.unlock(self.io);

        if (self.masks) |*m| m.deinit();
        self.masks = masks;

        if (self.coverages.len != masks.count) {
            self.allocator.free(self.coverages);
            self.coverages = self.allocator.alloc(f32, masks.count) catch &.{};
        }

        const stride = masks.width * masks.height;
        var best_idx: usize = 0;
        for (0..masks.count) |i| {
            const plane = masks.logits[i * stride ..][0..stride];
            var covered: usize = 0;
            for (plane) |logit| {
                if (logit > 0.0) covered += 1;
            }
            if (self.coverages.len > i) {
                self.coverages[i] = @as(f32, @floatFromInt(covered)) / @as(f32, @floatFromInt(stride));
            }
            if (masks.scores[i] > masks.scores[best_idx]) {
                best_idx = i;
            }
        }
        self.best_mask_idx = @intCast(best_idx);
        self.selected_mask = self.best_mask_idx;

        self.renderComposite(self.selected_mask);
        sam_macos_set_image(self.frame.ptr, @intCast(self.image.?.width), @intCast(self.image.?.height));

        var mask_infos: [16]SamMaskInfo = undefined;
        const count = @min(masks.count, mask_infos.len);
        for (0..count) |i| {
            mask_infos[i] = .{
                .score = masks.scores[i],
                .coverage = if (self.coverages.len > i) self.coverages[i] else 0,
            };
        }
        sam_macos_set_masks(@intCast(count), &mask_infos, self.best_mask_idx, self.selected_mask);

        const elapsed = secondsSince(self.io, started);
        var status_buf: [128]u8 = undefined;
        const status = std.fmt.bufPrintZ(&status_buf, "{d} point(s) -> {d} masks in {d:.2} s", .{
            self.points_len,
            masks.count,
            elapsed,
        }) catch "Segmentation complete";
        sam_macos_set_status(status);

        self.is_busy = false;
        sam_macos_set_busy(0);
    }

    pub fn handleFindText(self: *App, text: [*:0]const u8) void {
        if (self.is_busy or self.image == null) return;
        const phrase = std.mem.span(text);
        if (phrase.len == 0) return;

        self.is_busy = true;
        sam_macos_set_busy(1);

        var status_buf: [256]u8 = undefined;
        const status = std.fmt.bufPrintZ(&status_buf, "Looking for “{s}”… the first lookup also runs vision encoder.", .{phrase}) catch "Searching…";
        sam_macos_set_status(status);

        const PhraseContext = struct {
            app: *App,
            phrase: []const u8,
        };
        const ctx = self.allocator.create(PhraseContext) catch {
            self.is_busy = false;
            sam_macos_set_busy(0);
            return;
        };
        const phrase_copy = self.allocator.dupe(u8, phrase) catch {
            self.allocator.destroy(ctx);
            self.is_busy = false;
            sam_macos_set_busy(0);
            return;
        };
        ctx.* = .{ .app = self, .phrase = phrase_copy };

        const thread = std.Thread.spawn(.{}, runLookupWorker, .{ctx}) catch {
            self.allocator.free(phrase_copy);
            self.allocator.destroy(ctx);
            self.is_busy = false;
            sam_macos_set_busy(0);
            return;
        };
        thread.detach();
    }

    fn runLookupWorker(ctx: anytype) void {
        defer {
            ctx.app.allocator.free(ctx.phrase);
            ctx.app.allocator.destroy(ctx);
        }
        const self = ctx.app;
        const phrase = ctx.phrase;
        const started = std.Io.Timestamp.now(self.io, .awake);

        const concept_embedding = self.ensureEmbedding(true) catch |err| {
            std.debug.print("Concept vision encoder failed: {t}: {s}\n", .{ err, sam3.onnx.lastError() });
            sam_macos_set_status("Concept vision encoder failed");
            self.is_busy = false;
            sam_macos_set_busy(0);
            return;
        };

        var masks = self.model.lookup(concept_embedding, phrase, 0.5) catch |err| {
            std.debug.print("Text lookup failed: {t}: {s}\n", .{ err, sam3.onnx.lastError() });
            sam_macos_set_status("Text lookup failed");
            self.is_busy = false;
            sam_macos_set_busy(0);
            return;
        };

        self.mutex.lock(self.io) catch return;
        defer self.mutex.unlock(self.io);

        if (self.masks) |*m| m.deinit();
        self.masks = masks;

        if (masks.count == 0) {
            self.selected_mask = -1;
            self.best_mask_idx = -1;
            self.renderComposite(-1);
            sam_macos_set_image(self.frame.ptr, @intCast(self.image.?.width), @intCast(self.image.?.height));
            sam_macos_set_masks(0, null, 0, -1);

            var msg_buf: [256]u8 = undefined;
            const msg = std.fmt.bufPrintZ(&msg_buf, "No objects matched “{s}”.", .{phrase}) catch "No objects matched.";
            sam_macos_set_status(msg);
        } else {
            if (self.coverages.len != masks.count) {
                self.allocator.free(self.coverages);
                self.coverages = self.allocator.alloc(f32, masks.count) catch &.{};
            }
            const stride = masks.width * masks.height;
            var best_idx: usize = 0;
            for (0..masks.count) |i| {
                const plane = masks.logits[i * stride ..][0..stride];
                var covered: usize = 0;
                for (plane) |logit| {
                    if (logit > 0.0) covered += 1;
                }
                if (self.coverages.len > i) {
                    self.coverages[i] = @as(f32, @floatFromInt(covered)) / @as(f32, @floatFromInt(stride));
                }
                if (masks.scores[i] > masks.scores[best_idx]) {
                    best_idx = i;
                }
            }
            self.best_mask_idx = @intCast(best_idx);
            self.selected_mask = self.best_mask_idx;

            self.renderComposite(self.selected_mask);
            sam_macos_set_image(self.frame.ptr, @intCast(self.image.?.width), @intCast(self.image.?.height));

            var mask_infos: [64]SamMaskInfo = undefined;
            const count = @min(masks.count, mask_infos.len);
            for (0..count) |i| {
                mask_infos[i] = .{
                    .score = masks.scores[i],
                    .coverage = if (self.coverages.len > i) self.coverages[i] else 0,
                };
            }
            sam_macos_set_masks(@intCast(count), &mask_infos, self.best_mask_idx, self.selected_mask);

            const elapsed = secondsSince(self.io, started);
            var status_buf: [256]u8 = undefined;
            const status = std.fmt.bufPrintZ(&status_buf, "{d} object(s) matched “{s}” in {d:.2} s", .{
                masks.count,
                phrase,
                elapsed,
            }) catch "Search complete";
            sam_macos_set_status(status);
        }

        self.is_busy = false;
        sam_macos_set_busy(0);
    }

    pub fn handleSelectMask(self: *App, mask_index: c_int) void {
        self.mutex.lock(self.io) catch return;
        defer self.mutex.unlock(self.io);

        self.selected_mask = mask_index;
        self.renderComposite(self.selected_mask);
        sam_macos_set_image(self.frame.ptr, @intCast(self.image.?.width), @intCast(self.image.?.height));
    }

    pub fn handleClearPoints(self: *App) void {
        self.mutex.lock(self.io) catch return;
        defer self.mutex.unlock(self.io);

        self.points_len = 0;
        if (self.masks) |*m| {
            m.deinit();
            self.masks = null;
        }
        self.selected_mask = -1;
        self.best_mask_idx = -1;
        self.renderComposite(-1);
        if (self.image) |img| {
            sam_macos_set_image(self.frame.ptr, @intCast(img.width), @intCast(img.height));
        }
        sam_macos_set_masks(0, null, 0, -1);
        sam_macos_set_status("Points cleared.");
    }

    fn ensureEmbedding(self: *App, comptime concept: bool) !if (concept) sam3.ConceptEmbedding else sam3.Embedding {
        const cache = if (concept) &self.concept_cache else &self.cache;
        if (cache.*) |cached| return cached.embedding;

        if (concept) {
            if (self.cache) |*c| c.embedding.deinit();
            self.cache = null;
        } else {
            if (self.concept_cache) |*c| c.embedding.deinit();
            self.concept_cache = null;
        }

        const img = self.image orelse return error.NoImageLoaded;
        const embedding = if (concept) try self.model.encodeConcept(img) else try self.model.encode(img);
        cache.* = .{ .embedding = embedding };
        return embedding;
    }

    fn renderComposite(self: *App, mask_index: i32) void {
        const img = self.image orelse return;
        var canvas = zigimg.Image.create(self.allocator, img.width, img.height, .rgb24) catch return;
        defer canvas.deinit(self.allocator);
        @memcpy(canvas.pixels.rgb24, img.pixels.rgb24);

        if (mask_index >= 0 and self.masks != null) {
            const masks = self.masks.?;
            const u_index: usize = @intCast(mask_index);
            if (u_index < masks.count) {
                const stride = masks.width * masks.height;
                const plane = masks.logits[u_index * stride ..][0..stride];
                if (render.bilinear(
                    self.allocator,
                    plane,
                    masks.width,
                    masks.height,
                    img.width,
                    img.height,
                )) |resampled| {
                    defer self.allocator.free(resampled);
                    render.overlayMask(&canvas, resampled, render.mask_color, render.mask_alpha);
                } else |_| {}
            }
        }

        for (self.points[0..self.points_len]) |p| {
            render.drawPointMarker(&canvas, p, render.marker_radius);
        }

        for (canvas.pixels.rgb24, 0..) |px, i| {
            self.frame[i * 4 + 0] = px.r;
            self.frame[i * 4 + 1] = px.g;
            self.frame[i * 4 + 2] = px.b;
            self.frame[i * 4 + 3] = 255;
        }
    }
};

fn secondsSince(io: std.Io, started: std.Io.Timestamp) f64 {
    const elapsed = started.durationTo(std.Io.Timestamp.now(io, .awake));
    return @as(f64, @floatFromInt(elapsed.nanoseconds)) / 1e9;
}

var g_app: ?*App = null;

fn cOpenFile(path: [*:0]const u8) callconv(.c) void {
    if (g_app) |app| {
        app.openImageFromPath(std.mem.span(path));
    }
}

fn cSampleClick() callconv(.c) void {
    if (g_app) |app| {
        app.openImageFromPath(app.example_path);
    }
}

fn cModeChange(mode: c_int) callconv(.c) void {
    if (g_app) |app| {
        app.click_mode_add = (mode == 1);
    }
}

fn cClearPoints() callconv(.c) void {
    if (g_app) |app| {
        app.handleClearPoints();
    }
}

fn cFindText(text: [*:0]const u8) callconv(.c) void {
    if (g_app) |app| {
        app.handleFindText(text);
    }
}

fn cCanvasClick(norm_x: f32, norm_y: f32, is_positive: c_int) callconv(.c) void {
    if (g_app) |app| {
        app.handleCanvasClick(norm_x, norm_y, is_positive);
    }
}

fn cSelectMask(mask_index: c_int) callconv(.c) void {
    if (g_app) |app| {
        app.handleSelectMask(mask_index);
    }
}
