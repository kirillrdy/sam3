const std = @import("std");
const sam3 = @import("sam3");
const zigimg = @import("zigimg");
const wayland = @import("wayland.zig");
const font = @import("font.zig");

const render = sam3.render;
const max_points = 32;

fn Cache(comptime EmbeddingType: type) type {
    return struct { embedding: EmbeddingType };
}

pub const App = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    model: *sam3.Model,
    example_path: []const u8,

    client: wayland.WaylandClient,

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

    status_text: [256]u8 = undefined,
    status_len: usize = 0,

    search_text: [160]u8 = undefined,
    search_len: usize = 0,

    // Layout geometry
    canvas_x: usize = 16,
    canvas_y: usize = 110,
    canvas_w: usize = 968,
    canvas_h: usize = 540,

    img_rect_x: usize = 0,
    img_rect_y: usize = 0,
    img_rect_w: usize = 0,
    img_rect_h: usize = 0,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        model: *sam3.Model,
        example_path: []const u8,
        width: u32,
        height: u32,
    ) !App {
        const client = try wayland.WaylandClient.connect(allocator, width, height);

        var app: App = .{
            .allocator = allocator,
            .io = io,
            .model = model,
            .example_path = example_path,
            .client = client,
        };

        app.setStatus("Initializing SAM 3…");
        return app;
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
        self.client.deinit();
    }

    pub fn setStatus(self: *App, text: []const u8) void {
        const len = @min(text.len, self.status_text.len);
        @memcpy(self.status_text[0..len], text[0..len]);
        self.status_len = len;
    }

    pub fn run(self: *App) !void {
        // Open sample image by default
        self.openImageFromPath(self.example_path);

        while (true) {
            self.redraw();
            try self.client.commitFrame();

            // Poll events with timeout
            const ev_opt = try self.client.pollEvent(16);
            if (ev_opt) |ev| {
                switch (ev) {
                    .close => break,
                    .configure => |cfg| {
                        if (cfg.width > 0 and cfg.height > 0 and (cfg.width != self.client.width or cfg.height != self.client.height)) {
                            try self.client.resizeShmBuffer(cfg.width, cfg.height);
                            self.canvas_w = if (self.client.width > 32) self.client.width - 32 else 100;
                            self.canvas_h = if (self.client.height > 180) self.client.height - 180 else 100;
                        }
                    },
                    .pointer_button => |btn| {
                        if (btn.state == 1) { // Pressed
                            self.handlePointerClick(btn.x, btn.y, btn.button);
                        }
                    },
                    .keyboard_key => |k| {
                        if (k.state == 1) { // Key down
                            self.handleKey(k.key);
                        }
                    },
                    .pointer_motion => {},
                }
            }
        }
    }

    pub fn openImageFromPath(self: *App, path: []const u8) void {
        const file_bytes = std.Io.Dir.cwd().readFileAlloc(
            self.io,
            path,
            self.allocator,
            .limited(64 * 1024 * 1024),
        ) catch |err| {
            std.debug.print("Failed to read image file {s}: {t}\n", .{ path, err });
            self.setStatus("Could not open image file.");
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
            self.setStatus("That file is not an image this can decode.");
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
            self.setStatus("Out of memory for frame buffer.");
            return;
        };

        self.renderComposite(-1);

        var buf: [128]u8 = undefined;
        const msg = std.fmt.bufPrint(&buf, "{d} × {d} — click the object you want.", .{
            decoded.width,
            decoded.height,
        }) catch "Image loaded.";
        self.setStatus(msg);
    }

    fn handlePointerClick(self: *App, px: f32, py: f32, button: u32) void {
        if (self.is_busy) return;
        const x: usize = @intFromFloat(@max(0, px));
        const y: usize = @intFromFloat(@max(0, py));

        // Top row buttons:
        // [Sample Image] (16, 12, 130, 28)
        if (x >= 16 and x <= 146 and y >= 12 and y <= 40) {
            self.openImageFromPath(self.example_path);
            return;
        }

        // [Mode toggle] (156, 12, 220, 28)
        if (x >= 156 and x <= 376 and y >= 12 and y <= 40) {
            self.click_mode_add = !self.click_mode_add;
            return;
        }

        // [Clear Points] (386, 12, 130, 28)
        if (x >= 386 and x <= 516 and y >= 12 and y <= 40) {
            self.handleClearPoints();
            return;
        }

        // Row 2:
        // [Find by Word button] (386, 48, 130, 28)
        if (x >= 386 and x <= 516 and y >= 48 and y <= 76) {
            self.triggerFind();
            return;
        }

        // Bottom row: mask candidate buttons
        if (self.masks) |m| {
            const btn_y = self.client.height - 44;
            if (y >= btn_y and y <= btn_y + 32) {
                var cur_btn_x: usize = 16;
                for (0..m.count) |i| {
                    if (x >= cur_btn_x and x <= cur_btn_x + 160) {
                        self.selected_mask = @intCast(i);
                        self.renderComposite(self.selected_mask);
                        return;
                    }
                    cur_btn_x += 168;
                }
            }
        }

        // Canvas click
        if (self.image != null and self.img_rect_w > 0 and self.img_rect_h > 0) {
            if (x >= self.img_rect_x and x < self.img_rect_x + self.img_rect_w and
                y >= self.img_rect_y and y < self.img_rect_y + self.img_rect_h)
            {
                const norm_x = @as(f32, @floatFromInt(x - self.img_rect_x)) / @as(f32, @floatFromInt(self.img_rect_w));
                const norm_y = @as(f32, @floatFromInt(y - self.img_rect_y)) / @as(f32, @floatFromInt(self.img_rect_h));

                // button 0x111 is right click -> cut point
                const is_pos: c_int = if (button == 0x111) 0 else if (self.click_mode_add) 1 else 0;
                self.handleCanvasClick(norm_x, norm_y, is_pos);
            }
        }
    }

    fn handleKey(self: *App, key: u32) void {
        // Evdev keys
        switch (key) {
            28 => self.triggerFind(), // KEY_ENTER
            14 => { // KEY_BACKSPACE
                if (self.search_len > 0) self.search_len -= 1;
            },
            57 => { // KEY_SPACE
                if (self.search_len < self.search_text.len) {
                    self.search_text[self.search_len] = ' ';
                    self.search_len += 1;
                }
            },
            else => {
                if (evdevToChar(key)) |ch| {
                    if (self.search_len < self.search_text.len) {
                        self.search_text[self.search_len] = ch;
                        self.search_len += 1;
                    }
                }
            },
        }
    }

    fn triggerFind(self: *App) void {
        if (self.search_len == 0 or self.is_busy or self.image == null) return;
        const phrase = self.search_text[0..self.search_len];
        self.handleFindText(phrase);
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
        self.mutex.unlock(self.io);

        self.is_busy = true;
        self.setStatus("Segmenting… the first click on an image also runs the vision encoder.");

        const thread = std.Thread.spawn(.{}, runSegmentWorker, .{self}) catch {
            self.is_busy = false;
            return;
        };
        thread.detach();
    }

    fn runSegmentWorker(self: *App) void {
        const started = std.Io.Timestamp.now(self.io, .awake);

        const embedding = self.ensureEmbedding(false) catch |err| {
            std.debug.print("Vision encoder failed: {t}\n", .{err});
            self.setStatus("Vision encoder failed");
            self.is_busy = false;
            return;
        };

        const decode_started = std.Io.Timestamp.now(self.io, .awake);
        const masks = self.model.decode(embedding, self.points[0..self.points_len]) catch |err| {
            std.debug.print("Decoder failed: {t}\n", .{err});
            self.setStatus("Segmentation failed");
            self.is_busy = false;
            return;
        };
        const decode_elapsed = secondsSince(self.io, decode_started);
        std.debug.print("  {d} point(s) -> {d} masks in {d:.2} s\n", .{
            self.points_len,
            masks.count,
            decode_elapsed,
        });

        self.mutex.lock(self.io) catch return;
        defer self.mutex.unlock(self.io);

        if (self.masks) |*m| m.deinit();
        self.masks = masks;

        if (self.coverages.len != masks.count) {
            self.allocator.free(self.coverages);
            self.coverages = self.allocator.alloc(f32, masks.count) catch &.{};
        }
        self.best_mask_idx = @intCast(render.scoreMasks(masks.logits, masks.scores, masks.count, masks.width, masks.height, self.coverages));
        self.selected_mask = self.best_mask_idx;

        self.renderComposite(self.selected_mask);

        const elapsed = secondsSince(self.io, started);
        var status_buf: [128]u8 = undefined;
        const status = std.fmt.bufPrint(&status_buf, "{d} point(s) -> {d} masks in {d:.2} s", .{
            self.points_len,
            masks.count,
            elapsed,
        }) catch "Segmentation complete";
        self.setStatus(status);

        self.is_busy = false;
    }

    pub fn handleFindText(self: *App, phrase: []const u8) void {
        self.is_busy = true;

        var status_buf: [256]u8 = undefined;
        const status = std.fmt.bufPrint(&status_buf, "Looking for “{s}”…", .{phrase}) catch "Searching…";
        self.setStatus(status);

        const PhraseContext = struct {
            app: *App,
            phrase: []const u8,
        };
        const ctx = self.allocator.create(PhraseContext) catch {
            self.is_busy = false;
            return;
        };
        const phrase_copy = self.allocator.dupe(u8, phrase) catch {
            self.allocator.destroy(ctx);
            self.is_busy = false;
            return;
        };
        ctx.* = .{ .app = self, .phrase = phrase_copy };

        const thread = std.Thread.spawn(.{}, runLookupWorker, .{ctx}) catch {
            self.allocator.free(phrase_copy);
            self.allocator.destroy(ctx);
            self.is_busy = false;
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
            std.debug.print("Concept encoder failed: {t}\n", .{err});
            self.setStatus("Concept encoder failed");
            self.is_busy = false;
            return;
        };

        const lookup_started = std.Io.Timestamp.now(self.io, .awake);
        const masks = self.model.lookup(concept_embedding, phrase, 0.5) catch |err| {
            std.debug.print("Lookup failed: {t}\n", .{err});
            self.setStatus("Lookup failed");
            self.is_busy = false;
            return;
        };
        const lookup_elapsed = secondsSince(self.io, lookup_started);
        std.debug.print("  \"{s}\" -> {d} object(s) in {d:.2} s\n", .{
            phrase,
            masks.count,
            lookup_elapsed,
        });

        self.mutex.lock(self.io) catch return;
        defer self.mutex.unlock(self.io);

        if (self.masks) |*m| m.deinit();
        self.masks = masks;

        if (masks.count == 0) {
            self.selected_mask = -1;
            self.best_mask_idx = -1;
            self.renderComposite(-1);

            var msg_buf: [256]u8 = undefined;
            const msg = std.fmt.bufPrint(&msg_buf, "No objects matched “{s}”.", .{phrase}) catch "No objects matched.";
            self.setStatus(msg);
        } else {
            if (self.coverages.len != masks.count) {
                self.allocator.free(self.coverages);
                self.coverages = self.allocator.alloc(f32, masks.count) catch &.{};
            }
            self.best_mask_idx = @intCast(render.scoreMasks(masks.logits, masks.scores, masks.count, masks.width, masks.height, self.coverages));
            self.selected_mask = self.best_mask_idx;
            self.renderComposite(self.selected_mask);

            const elapsed = secondsSince(self.io, started);
            var status_buf: [256]u8 = undefined;
            const msg = std.fmt.bufPrint(&status_buf, "{d} object(s) matched “{s}” in {d:.2} s", .{
                masks.count,
                phrase,
                elapsed,
            }) catch "Search complete";
            self.setStatus(msg);
        }

        self.is_busy = false;
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
        self.setStatus("Points cleared.");
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
        const started = std.Io.Timestamp.now(self.io, .awake);
        const embedding = if (concept) try self.model.encodeConcept(img) else try self.model.encode(img);
        std.debug.print("  {s}encoded {d}x{d} in {d:.2} s\n", .{
            if (concept) "concept-" else "",
            img.width,
            img.height,
            secondsSince(self.io, started),
        });
        cache.* = .{ .embedding = embedding };
        return embedding;
    }

    fn renderComposite(self: *App, mask_index: i32) void {
        const img = self.image orelse return;
        var plane: ?[]const f32 = null;
        var mw: usize = 0;
        var mh: usize = 0;
        if (mask_index >= 0 and self.masks != null) {
            const masks = self.masks.?;
            const u_index: usize = @intCast(mask_index);
            if (u_index < masks.count) {
                const stride = masks.width * masks.height;
                plane = masks.logits[u_index * stride ..][0..stride];
                mw = masks.width;
                mh = masks.height;
            }
        }
        render.compositeRgba(self.allocator, img, self.frame, plane, mw, mh, self.points[0..self.points_len]);
    }

    fn redraw(self: *App) void {
        const pixels = self.client.pixels;
        const stride = self.client.width;
        const h = self.client.height;

        // Background: #14161a
        @memset(pixels, 0x0014161a);

        // Row 1 Buttons:
        // [Sample Image]
        font.drawButton(pixels, stride, 16, 12, 130, 28, "Sample Image", false, false, 0x0000dc64);

        // [Clicks add / cut]
        const mode_text = if (self.click_mode_add) "Clicks add to mask" else "Clicks cut from mask";
        font.drawButton(pixels, stride, 156, 12, 220, 28, mode_text, false, !self.click_mode_add, 0x0000dc64);

        // [Clear Points]
        font.drawButton(pixels, stride, 386, 12, 130, 28, "Clear Points", false, false, 0x0000dc64);

        // Row 2: Search field + Find button
        font.fillRect(pixels, stride, 16, 48, 360, 28, 0x001c1f25);
        font.strokeRect(pixels, stride, 16, 48, 360, 28, 0x002c3038);
        if (self.search_len > 0) {
            font.drawText(pixels, stride, self.search_text[0..self.search_len], 24, 54, 0x00e6e8ec);
        } else {
            font.drawText(pixels, stride, "Find objects, e.g. cat or red car", 24, 54, 0x00969ba5);
        }

        font.drawButton(pixels, stride, 386, 48, 130, 28, "Find by Word", false, false, 0x0000dc64);

        // Status text
        font.drawText(pixels, stride, self.status_text[0..self.status_len], 16, 86, 0x00969ba5);

        // Canvas Area
        const cx = self.canvas_x;
        const cy = self.canvas_y;
        const cw = if (stride > 32) stride - 32 else 100;
        const ch = if (h > 170) h - 170 else 100;

        font.fillRect(pixels, stride, cx, cy, cw, ch, 0x001c1f25);
        font.strokeRect(pixels, stride, cx, cy, cw, ch, 0x002c3038);

        if (self.image) |img| {
            if (img.width > 0 and img.height > 0) {
                const scale_x = @as(f32, @floatFromInt(cw)) / @as(f32, @floatFromInt(img.width));
                const scale_y = @as(f32, @floatFromInt(ch)) / @as(f32, @floatFromInt(img.height));
                const scale = @min(scale_x, scale_y);

                const dw: usize = @intFromFloat(@as(f32, @floatFromInt(img.width)) * scale);
                const dh: usize = @intFromFloat(@as(f32, @floatFromInt(img.height)) * scale);
                const dx: usize = cx + (cw - dw) / 2;
                const dy: usize = cy + (ch - dh) / 2;

                self.img_rect_x = dx;
                self.img_rect_y = dy;
                self.img_rect_w = dw;
                self.img_rect_h = dh;

                // Blit frame buffer to window
                const frame_bytes = self.frame;
                for (0..dh) |py| {
                    const src_y = (py * img.height) / dh;
                    const row_out = (dy + py) * stride;
                    for (0..dw) |px| {
                        const src_x = (px * img.width) / dw;
                        const src_idx = (src_y * img.width + src_x) * 4;
                        const r = frame_bytes[src_idx + 0];
                        const g = frame_bytes[src_idx + 1];
                        const b = frame_bytes[src_idx + 2];
                        const color: u32 = (@as(u32, r) << 16) | (@as(u32, g) << 8) | b;
                        pixels[row_out + dx + px] = color;
                    }
                }
            }
        }

        // Bottom row: mask selection buttons
        if (self.masks) |m| {
            const btn_y = h - 44;
            var cur_btn_x: usize = 16;
            for (0..m.count) |i| {
                var title_buf: [64]u8 = undefined;
                const star = if (@as(i32, @intCast(i)) == self.best_mask_idx) "*" else "";
                const cov = if (self.coverages.len > i) self.coverages[i] * 100.0 else 0;
                const title = std.fmt.bufPrint(&title_buf, "Mask {d}{s} ({d:.2}%)", .{ i, star, cov }) catch "Mask";
                const is_active = (@as(i32, @intCast(i)) == self.selected_mask);

                font.drawButton(pixels, stride, cur_btn_x, btn_y, 160, 28, title, false, is_active, 0x0000dc64);
                cur_btn_x += 168;
            }
        }
    }
};

fn evdevToChar(key: u32) ?u8 {
    return switch (key) {
        16 => 'q',
        17 => 'w',
        18 => 'e',
        19 => 'r',
        20 => 't',
        21 => 'y',
        22 => 'u',
        23 => 'i',
        24 => 'o',
        25 => 'p',
        30 => 'a',
        31 => 's',
        32 => 'd',
        33 => 'f',
        34 => 'g',
        35 => 'h',
        36 => 'j',
        37 => 'k',
        38 => 'l',
        44 => 'z',
        45 => 'x',
        46 => 'c',
        47 => 'v',
        48 => 'b',
        49 => 'n',
        50 => 'm',
        2 => '1',
        3 => '2',
        4 => '3',
        5 => '4',
        6 => '5',
        7 => '6',
        8 => '7',
        9 => '8',
        10 => '9',
        11 => '0',
        else => null,
    };
}

fn secondsSince(io: std.Io, started: std.Io.Timestamp) f64 {
    const elapsed = started.durationTo(std.Io.Timestamp.now(io, .awake));
    return @as(f64, @floatFromInt(elapsed.nanoseconds)) / 1e9;
}
