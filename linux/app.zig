const std = @import("std");
const sam3 = @import("sam3");
const zigimg = @import("zigimg");
const wayland = @import("wayland.zig");
const font = @import("font.zig");

const render = sam3.render;
const max_points = 32;
const BrowserEntry = struct {
    name: []u8,
    is_dir: bool,
};

pub const App = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    model: *sam3.Model,
    example_path: []const u8,

    client: wayland.WaylandClient,

    mutex: std.Io.Mutex = .init,
    is_busy: bool = false,
    redraw_pending: std.atomic.Value(bool) = .init(true),

    image: ?zigimg.Image = null,
    frame: []u8 = &.{},

    points: [max_points]sam3.Point = undefined,
    points_len: usize = 0,
    click_mode_add: bool = true,

    masks: ?sam3.Masks = null,
    coverages: []f32 = &.{},
    selected_mask: i32 = -1,
    best_mask_idx: i32 = -1,

    status_text: [256]u8 = undefined,
    status_len: usize = 0,

    search_text: [160]u8 = undefined,
    search_len: usize = 0,
    search_focused: bool = true,
    search_caret: usize = 0,
    search_scroll: usize = 0,
    search_anchor: ?usize = null,
    browser_open: bool = false,
    browser_path: [4096]u8 = undefined,
    browser_path_len: usize = 0,
    browser_dir: []u8 = &.{},
    browser_entries: std.ArrayList(BrowserEntry) = .empty,
    browser_scroll: usize = 0,
    shift_down: bool = false,
    ctrl_down: bool = false,
    caps_lock: bool = false,

    // Window state
    is_maximized: bool = false,
    unmaximized_width: u32 = 1000,
    unmaximized_height: u32 = 720,
    pending_width: u32 = 1000,
    pending_height: u32 = 720,
    last_titlebar_click_time: ?std.Io.Timestamp = null,

    // Layout geometry
    canvas_x: usize = 16,
    canvas_y: usize = 140,
    canvas_w: usize = 968,
    canvas_h: usize = 510,

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
            .pending_width = width,
            .pending_height = height,
            .unmaximized_width = width,
            .unmaximized_height = height,
        };

        app.setStatus("Initializing SAM 3…");
        return app;
    }

    pub fn deinit(self: *App) void {
        self.mutex.lock(self.io) catch return;
        defer self.mutex.unlock(self.io);

        if (self.masks) |*m| m.deinit();
        if (self.image) |*img| img.deinit(self.allocator);
        self.allocator.free(self.frame);
        self.allocator.free(self.coverages);
        self.clearBrowserEntries();
        self.browser_entries.deinit(self.allocator);
        if (self.browser_dir.len > 0) self.allocator.free(self.browser_dir);
        self.client.deinit();
    }

    pub fn setStatus(self: *App, text: []const u8) void {
        const len = @min(text.len, self.status_text.len);
        @memcpy(self.status_text[0..len], text[0..len]);
        self.status_len = len;
        self.redraw_pending.store(true, .release);
    }

    pub fn run(self: *App) !void {
        // Open sample image by default
        _ = self.openImageFromPath(self.example_path);

        self.pending_width = self.client.width;
        self.pending_height = self.client.height;
        while (true) {
            if (self.pending_width != self.client.width or self.pending_height != self.client.height) {
                try self.client.resizeShmBuffer(self.pending_width, self.pending_height);
                self.redraw_pending.store(true, .release);
            }
            // Draw only when contents change, into a buffer released by the compositor.
            if (self.redraw_pending.load(.acquire) and try self.client.beginFrame()) {
                _ = self.redraw_pending.swap(false, .acq_rel);
                self.mutex.lock(self.io) catch return;
                self.redraw();
                self.mutex.unlock(self.io);
                try self.client.commitFrame();
            }

            // Poll events with timeout
            const ev_opt = try self.client.pollEvent(16);
            if (ev_opt) |ev| {
                switch (ev) {
                    .close => break,
                    .configure => |cfg| {
                        self.redraw_pending.store(true, .release);
                        self.is_maximized = cfg.maximized;
                        if (cfg.width > 0 and cfg.height > 0) {
                            if (!cfg.maximized) {
                                self.unmaximized_width = cfg.width;
                                self.unmaximized_height = cfg.height;
                            }
                            self.pending_width = cfg.width;
                            self.pending_height = cfg.height;
                        } else if (!cfg.maximized) {
                            self.pending_width = self.unmaximized_width;
                            self.pending_height = self.unmaximized_height;
                        }
                    },
                    .pointer_button => |btn| {
                        if (btn.state == 1) { // Pressed
                            if (try self.handlePointerClick(btn.x, btn.y, btn.button, btn.serial)) {
                                break;
                            }
                            self.redraw_pending.store(true, .release);
                        }
                    },
                    .keyboard_key => |k| {
                        self.handleKey(k.key, k.state);
                        if (k.state == 1) self.redraw_pending.store(true, .release);
                    },
                    .pointer_motion => |motion| try self.updateCursor(motion.x, motion.y),
                }
            }
        }
    }

    pub fn openImageFromPath(self: *App, path: []const u8) bool {
        const file_bytes = std.Io.Dir.cwd().readFileAlloc(
            self.io,
            path,
            self.allocator,
            .limited(64 * 1024 * 1024),
        ) catch |err| {
            std.debug.print("Failed to read image file {s}: {t}\n", .{ path, err });
            self.setStatus("Could not open image file.");
            return false;
        };
        defer self.allocator.free(file_bytes);

        return self.openImageFromBytes(file_bytes);
    }

    pub fn openImageFromBytes(self: *App, bytes: []const u8) bool {
        self.mutex.lock(self.io) catch return false;
        defer self.mutex.unlock(self.io);

        var decoded = sam3.decode(self.allocator, bytes) catch |err| {
            std.debug.print("Failed to decode image: {t}\n", .{err});
            self.setStatus("That file is not an image this can decode.");
            return false;
        };

        if (self.image) |*old| old.deinit(self.allocator);
        self.allocator.free(self.frame);
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
            return false;
        };

        self.renderComposite(-1);

        var buf: [128]u8 = undefined;
        const msg = std.fmt.bufPrint(&buf, "{d} × {d} — click the object you want.", .{
            decoded.width,
            decoded.height,
        }) catch "Image loaded.";
        self.setStatus(msg);
        return true;
    }

    fn clearBrowserEntries(self: *App) void {
        for (self.browser_entries.items) |entry| self.allocator.free(entry.name);
        self.browser_entries.clearRetainingCapacity();
    }

    fn browseDir(self: *App, path: []const u8) void {
        const dir = std.Io.Dir.cwd().openDir(self.io, path, .{ .iterate = true }) catch {
            self.setStatus("Cannot open that folder.");
            return;
        };
        defer dir.close(self.io);

        const owned = self.allocator.dupe(u8, path) catch return;
        if (self.browser_dir.len > 0) self.allocator.free(self.browser_dir);
        self.browser_dir = owned;
        self.clearBrowserEntries();
        self.browser_scroll = 0;
        self.setBrowserPath(owned);

        var it = dir.iterate();
        while (it.next(self.io) catch null) |entry| {
            if (entry.name.len == 0 or std.mem.eql(u8, entry.name, ".") or std.mem.eql(u8, entry.name, "..")) continue;
            const is_dir = entry.kind == .directory;
            const name = self.allocator.dupe(u8, entry.name) catch break;
            self.browser_entries.append(self.allocator, .{ .name = name, .is_dir = is_dir }) catch {
                self.allocator.free(name);
                break;
            };
        }
        std.mem.sort(BrowserEntry, self.browser_entries.items, {}, struct {
            fn lessThan(_: void, a: BrowserEntry, b: BrowserEntry) bool {
                if (a.is_dir != b.is_dir) return a.is_dir;
                return std.ascii.lessThanIgnoreCase(a.name, b.name);
            }
        }.lessThan);
    }

    fn setBrowserPath(self: *App, path: []const u8) void {
        self.browser_path_len = @min(path.len, self.browser_path.len);
        @memcpy(self.browser_path[0..self.browser_path_len], path[0..self.browser_path_len]);
    }

    fn openBrowser(self: *App) void {
        self.browser_open = true;
        const home = if (std.c.getenv("HOME")) |value| std.mem.span(value) else "/";
        self.browseDir(if (self.browser_dir.len > 0) self.browser_dir else home);
    }

    fn browserOpenPath(self: *App, path: []const u8) void {
        const dir = std.Io.Dir.cwd().openDir(self.io, path, .{ .iterate = true }) catch null;
        if (dir) |d| {
            d.close(self.io);
            self.browseDir(path);
        } else if (self.openImageFromPath(path)) {
            self.browser_open = false;
        }
    }

    fn browserActivate(self: *App, index: usize) void {
        if (index >= self.browser_entries.items.len) return;
        const entry = self.browser_entries.items[index];
        const path = std.fs.path.join(self.allocator, &.{ self.browser_dir, entry.name }) catch return;
        defer self.allocator.free(path);
        self.browserOpenPath(path);
    }

    fn getResizeEdge(self: *App, x: usize, y: usize) u32 {
        if (self.is_maximized) return 0;
        const margin: usize = 8;
        const w = self.client.width;
        const h = self.client.height;
        if (w >= 110 and x >= w - 105 and y < 32) return 0;
        var edges: u32 = 0;
        if (y < margin) edges |= 1;
        if (y + margin >= h) edges |= 2;
        if (x < margin) edges |= 4;
        if (x + margin >= w) edges |= 8;
        return edges;
    }

    fn getResizeCursor(edges: u32) ?wayland.Cursor {
        return switch (edges) {
            1, 2 => .resize_ns,
            4, 8 => .resize_ew,
            5, 10 => .resize_nwse,
            6, 9 => .resize_nesw,
            else => null,
        };
    }

    fn handlePointerClick(self: *App, px: f32, py: f32, button: u32, serial: u32) !bool {
        const x: usize = @intFromFloat(@max(0, px));
        const y: usize = @intFromFloat(@max(0, py));
        const stride = self.client.width;

        // Interactive window resize from borders
        if (button == 0x110) {
            const edges = self.getResizeEdge(x, y);
            if (edges != 0) {
                try self.client.startInteractiveResize(serial, edges);
                return false;
            }
        }

        // Header bar / Window controls
        if (y < 32 and button == 0x110) {
            if (stride >= 110) {
                // Close button [x]
                if (x >= stride - 36 and x < stride - 10 and y >= 5 and y < 27) {
                    return true;
                }
                // Maximize / restore button [+] / [=]
                if (x >= stride - 68 and x < stride - 42 and y >= 5 and y < 27) {
                    if (self.is_maximized) {
                        try self.client.unsetMaximized();
                        self.is_maximized = false;
                        self.pending_width = self.unmaximized_width;
                        self.pending_height = self.unmaximized_height;
                    } else {
                        self.unmaximized_width = self.client.width;
                        self.unmaximized_height = self.client.height;
                        try self.client.setMaximized();
                        self.is_maximized = true;
                    }
                    return false;
                }
                // Minimize button [-]
                if (x >= stride - 100 and x < stride - 74 and y >= 5 and y < 27) {
                    try self.client.setMinimized();
                    return false;
                }
            }

            // Drag title bar to move, or double-click to toggle maximize
            const now = std.Io.Timestamp.now(self.io, .awake);
            if (self.last_titlebar_click_time) |prev| {
                const dt = prev.durationTo(now).nanoseconds;
                if (dt < 400_000_000) {
                    self.last_titlebar_click_time = null;
                    if (self.is_maximized) {
                        try self.client.unsetMaximized();
                        self.is_maximized = false;
                        self.pending_width = self.unmaximized_width;
                        self.pending_height = self.unmaximized_height;
                    } else {
                        self.unmaximized_width = self.client.width;
                        self.unmaximized_height = self.client.height;
                        try self.client.setMaximized();
                        self.is_maximized = true;
                    }
                    return false;
                }
            }
            self.last_titlebar_click_time = now;
            try self.client.startInteractiveMove(serial);
            return false;
        }

        if (self.browser_open) {
            if (button != 0x110) return false;
            const bx: usize = 16;
            const by: usize = 140;
            const bw = @min(stride -| 32, 640);
            const bh = @min(self.client.height -| 190, 420);
            if (x >= bx + bw -| 92 and x < bx + bw -| 12 and y >= by + 8 and y < by + 36) {
                self.browser_open = false;
            } else if (x >= bx + 12 and x < bx + 92 and y >= by + 8 and y < by + 36) {
                const parent = std.fs.path.dirname(self.browser_dir) orelse "/";
                self.browseDir(parent);
            } else if (x >= bx + 12 and x < bx + bw -| 12 and y >= by + 46 and y < by + 74) {
                self.browser_path_len = 0;
            } else if (y >= by + 82 and y < by + bh -| 42 and x >= bx + 12 and x < bx + bw -| 12) {
                self.browserActivate(self.browser_scroll + (y - by - 82) / 24);
            } else if (x >= bx + 12 and x < bx + 92 and y >= by + bh -| 36 and y < by + bh -| 8) {
                if (self.browser_scroll > 0) self.browser_scroll -= 1;
            } else if (x >= bx + 100 and x < bx + 180 and y >= by + bh -| 36 and y < by + bh -| 8) {
                if (self.browser_scroll + browserVisibleRows(bh) < self.browser_entries.items.len) self.browser_scroll += 1;
            } else if (x >= bx + bw -| 100 and x < bx + bw -| 12 and y >= by + bh -| 36 and y < by + bh -| 8) {
                const path = self.browser_path[0..self.browser_path_len];
                self.browserOpenPath(path);
            }
            return false;
        }

        if (x >= 16 and x < 376 and y >= 78 and y < 106 and button == 0x110) {
            self.search_focused = true;
            const visible = (x -| 24) / font.font_width;
            self.search_caret = @min(self.search_len, self.search_scroll + visible);
            self.search_anchor = null;
            self.adjustSearchScroll();
            return false;
        }
        self.search_focused = false;
        self.search_anchor = null;
        if (self.is_busy) return false;

        // Top row buttons:
        if (x >= 526 and x < 656 and y >= 42 and y <= 70) {
            self.openBrowser();
            return false;
        }
        // [Sample Image] (16, 42, 130, 28)
        if (x >= 16 and x <= 146 and y >= 42 and y <= 70) {
            _ = self.openImageFromPath(self.example_path);
            return false;
        }

        // [Mode toggle] (156, 42, 220, 28)
        if (x >= 156 and x <= 376 and y >= 42 and y <= 70) {
            self.click_mode_add = !self.click_mode_add;
            return false;
        }

        // [Clear Points] (386, 42, 130, 28)
        if (x >= 386 and x <= 516 and y >= 42 and y <= 70) {
            self.handleClearPoints();
            return false;
        }

        // Row 2:
        // [Find by Word button] (386, 78, 130, 28)
        if (x >= 386 and x <= 516 and y >= 78 and y <= 106) {
            self.triggerFind();
            return false;
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
                        return false;
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
        return false;
    }

    fn updateCursor(self: *App, px: f32, py: f32) !void {
        const x: usize = @intFromFloat(@max(0, px));
        const y: usize = @intFromFloat(@max(0, py));
        const edges = self.getResizeEdge(x, y);
        const kind: wayland.Cursor = if (getResizeCursor(edges)) |c|
            c
        else if (x >= 16 and x < 376 and y >= 78 and y < 106)
            .text
        else if (self.image != null and x >= self.img_rect_x and x < self.img_rect_x + self.img_rect_w and
            y >= self.img_rect_y and y < self.img_rect_y + self.img_rect_h)
            .crosshair
        else
            .arrow;
        try self.client.setCursor(kind);
    }

    fn adjustSearchScroll(self: *App) void {
        const visible_chars: usize = 37;
        if (self.search_caret < self.search_scroll) self.search_scroll = self.search_caret;
        if (self.search_caret > self.search_scroll + visible_chars) {
            self.search_scroll = self.search_caret - visible_chars;
        }
    }

    fn selection(self: *App) ?struct { start: usize, end: usize } {
        const anchor = self.search_anchor orelse return null;
        if (anchor == self.search_caret) return null;
        return .{ .start = @min(anchor, self.search_caret), .end = @max(anchor, self.search_caret) };
    }

    fn deleteSelection(self: *App) bool {
        const selected = self.selection() orelse return false;
        std.mem.copyForwards(u8, self.search_text[selected.start..], self.search_text[selected.end..self.search_len]);
        self.search_len -= selected.end - selected.start;
        self.search_caret = selected.start;
        self.search_anchor = null;
        self.adjustSearchScroll();
        return true;
    }

    fn moveSearchCaret(self: *App, pos: usize) void {
        if (self.shift_down) {
            if (self.search_anchor == null) self.search_anchor = self.search_caret;
        } else {
            self.search_anchor = null;
        }
        self.search_caret = @min(pos, self.search_len);
        self.adjustSearchScroll();
    }

    fn handleKey(self: *App, key: u32, state: u32) void {
        if (key == 42 or key == 54) {
            self.shift_down = state == 1;
            return;
        }
        if (key == 29 or key == 97) {
            self.ctrl_down = state == 1;
            return;
        }
        if (state != 1) return;
        if (key == 58) {
            self.caps_lock = !self.caps_lock;
            return;
        }
        if (self.browser_open) {
            if (self.ctrl_down and key == 30) {
                self.browser_path_len = 0;
                return;
            }
            switch (key) {
                1 => self.browser_open = false, // Escape
                28 => self.browserOpenPath(self.browser_path[0..self.browser_path_len]), // Enter
                14 => self.browser_path_len -|= 1, // Backspace
                103 => { // Up
                    if (self.browser_scroll > 0) self.browser_scroll -= 1;
                },
                108 => { // Down
                    if (self.browser_scroll + browserVisibleRows(@min(self.client.height -| 190, 420)) < self.browser_entries.items.len) self.browser_scroll += 1;
                },
                else => if (!self.ctrl_down) {
                    if (evdevToChar(key, self.shift_down, self.caps_lock)) |ch| {
                        if (self.browser_path_len < self.browser_path.len) {
                            self.browser_path[self.browser_path_len] = ch;
                            self.browser_path_len += 1;
                        }
                    }
                },
            }
            return;
        }
        if (!self.search_focused) return;

        if (self.ctrl_down) {
            if (key == 30) { // Ctrl+A
                self.search_anchor = 0;
                self.search_caret = self.search_len;
                self.adjustSearchScroll();
            }
            return;
        }

        switch (key) {
            28 => self.triggerFind(), // Enter
            1 => self.search_focused = false, // Escape
            105 => self.moveSearchCaret(self.search_caret -| 1), // Left
            106 => self.moveSearchCaret(self.search_caret + 1), // Right
            102 => self.moveSearchCaret(0), // Home
            107 => self.moveSearchCaret(self.search_len), // End
            14 => { // Backspace
                if (!self.deleteSelection() and self.search_caret > 0) {
                    const pos = self.search_caret - 1;
                    std.mem.copyForwards(u8, self.search_text[pos..], self.search_text[self.search_caret..self.search_len]);
                    self.search_len -= 1;
                    self.search_caret = pos;
                }
            },
            111 => { // Delete
                if (!self.deleteSelection() and self.search_caret < self.search_len) {
                    std.mem.copyForwards(u8, self.search_text[self.search_caret..], self.search_text[self.search_caret + 1 .. self.search_len]);
                    self.search_len -= 1;
                }
            },
            else => {
                if (evdevToChar(key, self.shift_down, self.caps_lock)) |ch| {
                    _ = self.deleteSelection();
                    if (self.search_len < self.search_text.len) {
                        std.mem.copyBackwards(u8, self.search_text[self.search_caret + 1 .. self.search_len + 1], self.search_text[self.search_caret..self.search_len]);
                        self.search_text[self.search_caret] = ch;
                        self.search_len += 1;
                        self.search_caret += 1;
                    }
                }
            },
        }
        self.adjustSearchScroll();
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

        var embedding = self.ensureEmbedding(false) catch |err| {
            std.debug.print("Vision encoder failed: {t}\n", .{err});
            self.setStatus("Vision encoder failed");
            self.is_busy = false;
            return;
        };
        defer embedding.deinit();

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

        var concept_embedding = self.ensureEmbedding(true) catch |err| {
            std.debug.print("Concept encoder failed: {t}\n", .{err});
            self.setStatus("Concept encoder failed");
            self.is_busy = false;
            return;
        };
        defer concept_embedding.deinit();

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
        const img = self.image orelse return error.NoImageLoaded;
        const started = std.Io.Timestamp.now(self.io, .awake);
        const embedding = if (concept) try self.model.encodeConcept(img) else try self.model.encode(img);
        std.debug.print("  {s}encoded {d}x{d} in {d:.2} s\n", .{
            if (concept) "concept-" else "",
            img.width,
            img.height,
            secondsSince(self.io, started),
        });
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
        self.redraw_pending.store(true, .release);
    }

    fn redraw(self: *App) void {
        const pixels = self.client.pixels;
        const stride = self.client.width;
        const h = self.client.height;

        // Background: #14161a
        @memset(pixels, 0x0014161a);

        // Window border (1px) when not maximized
        if (!self.is_maximized and stride > 2 and h > 2) {
            font.strokeRect(pixels, stride, 0, 0, stride, h, 0x00323742);
        }

        // Header / Title bar (y: 0..32)
        font.fillRect(pixels, stride, 0, 0, stride, 32, 0x0017191e);
        font.fillRect(pixels, stride, 0, 31, stride, 1, 0x002c3038);
        font.drawText(pixels, stride, "SAM 3", 16, 7, 0x00d0d4dc);

        if (stride >= 110) {
            // Minimize [-]
            font.drawButton(pixels, stride, stride - 100, 5, 26, 22, "-", false, false, 0x0000dc64);
            // Maximize [+] / [=]
            font.drawButton(pixels, stride, stride - 68, 5, 26, 22, if (self.is_maximized) "=" else "+", false, false, 0x0000dc64);
            // Close [x]
            font.drawButton(pixels, stride, stride - 36, 5, 26, 22, "x", false, false, 0x00e05555);
        }

        // Row 1 Buttons:
        // [Sample Image]
        font.drawButton(pixels, stride, 16, 42, 130, 28, "Sample Image", false, false, 0x0000dc64);

        // [Clicks add / cut]
        const mode_text = if (self.click_mode_add) "Clicks add to mask" else "Clicks cut from mask";
        font.drawButton(pixels, stride, 156, 42, 220, 28, mode_text, false, !self.click_mode_add, 0x0000dc64);

        // [Clear Points]
        font.drawButton(pixels, stride, 386, 42, 130, 28, "Clear Points", false, false, 0x0000dc64);
        font.drawButton(pixels, stride, 526, 42, 130, 28, "Open Image", false, false, 0x0000dc64);

        // Row 2: Search field + Find button
        font.fillRect(pixels, stride, 16, 78, 360, 28, if (self.search_focused) 0x0023272e else 0x001c1f25);
        font.strokeRect(pixels, stride, 16, 78, 360, 28, if (self.search_focused) 0x0000dc64 else 0x00383d45);
        if (self.search_len > 0) {
            const end = @min(self.search_len, self.search_scroll + 37);
            if (self.selection()) |selected| {
                const start = @max(selected.start, self.search_scroll);
                const stop = @min(selected.end, end);
                if (start < stop) {
                    font.fillRect(pixels, stride, 24 + (start - self.search_scroll) * font.font_width, 81,
                        (stop - start) * font.font_width, 22, 0x00335d50);
                }
            }
            font.drawText(pixels, stride, self.search_text[self.search_scroll..end], 24, 83, 0x00f2f4f6);
        } else {
            font.drawText(pixels, stride, "Find objects, e.g. cat or red car", 24, 83, 0x008e949e);
        }
        if (self.search_focused) {
            const caret_x = 24 + (self.search_caret - self.search_scroll) * font.font_width;
            font.fillRect(pixels, stride, caret_x, 82, 1, 20, 0x0000dc64);
        }

        font.drawButton(pixels, stride, 386, 78, 130, 28, "Find by Word", false, false, 0x0000dc64);

        // Status text
        font.drawText(pixels, stride, self.status_text[0..self.status_len], 16, 116, 0x00969ba5);

        // Canvas Area
        const cx = self.canvas_x;
        const cy = self.canvas_y;
        const cw = if (stride > 32) stride - 32 else 100;
        const ch = if (h > 200) h - 200 else 100;
        self.canvas_w = cw;
        self.canvas_h = ch;

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
        if (self.browser_open) self.drawBrowser(pixels, stride, h);
    }

    fn drawBrowser(self: *App, pixels: []u32, stride: usize, height: usize) void {
        const x: usize = 16;
        const y: usize = 140;
        const w = @min(stride -| 32, 640);
        const h = @min(height -| 190, 420);
        if (w < 200 or h < 130) return;
        font.fillRect(pixels, stride, x, y, w, h, 0x0023272e);
        font.strokeRect(pixels, stride, x, y, w, h, 0x0000dc64);
        font.drawButton(pixels, stride, x + 12, y + 8, 80, 28, "Parent", false, false, 0x0000dc64);
        font.drawText(pixels, stride, "Choose image", x + 104, y + 14, 0x00f2f4f6);
        font.drawButton(pixels, stride, x + w - 92, y + 8, 80, 28, "Cancel", false, false, 0x0000dc64);
        font.fillRect(pixels, stride, x + 12, y + 46, w - 24, 28, 0x001c1f25);
        font.strokeRect(pixels, stride, x + 12, y + 46, w - 24, 28, 0x0000dc64);
        const path = self.browser_path[0..self.browser_path_len];
        const max_chars = (w - 42) / font.font_width;
        font.drawText(pixels, stride, if (path.len == 0) "Type an absolute path" else path[path.len -| max_chars ..], x + 20, y + 52, if (path.len == 0) 0x008e949e else 0x00f2f4f6);
        const rows = browserVisibleRows(h);
        for (0..rows) |row| {
            const index = self.browser_scroll + row;
            if (index >= self.browser_entries.items.len) break;
            const entry = self.browser_entries.items[index];
            const ry = y + 82 + row * 24;
            font.fillRect(pixels, stride, x + 12, ry, w - 24, 22, if (row % 2 == 0) 0x001c1f25 else 0x0023272e);
            const max_name = (w - 60) / font.font_width;
            font.drawText(pixels, stride, if (entry.is_dir) "/" else " ", x + 18, ry + 3, 0x0000dc64);
            font.drawText(pixels, stride, entry.name[0..@min(entry.name.len, max_name)], x + 30, ry + 3, 0x00e6e8ec);
        }
        font.drawButton(pixels, stride, x + 12, y + h - 36, 80, 28, "Up", false, false, 0x0000dc64);
        font.drawButton(pixels, stride, x + 100, y + h - 36, 80, 28, "Down", false, false, 0x0000dc64);
        font.drawButton(pixels, stride, x + w - 100, y + h - 36, 88, 28, "Open Path", false, false, 0x0000dc64);
    }
};

fn browserVisibleRows(height: usize) usize {
    return (height -| 124) / 24;
}

fn evdevToChar(key: u32, shift: bool, caps: bool) ?u8 {
    const ch: u8 = switch (key) {
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
        12 => '-',
        13 => '=',
        26 => '[',
        27 => ']',
        39 => ';',
        40 => '\'',
        41 => '`',
        43 => '\\',
        51 => ',',
        52 => '.',
        53 => '/',
        57 => ' ',
        else => return null,
    };
    if (ch >= 'a' and ch <= 'z') return if (shift != caps) ch - 32 else ch;
    if (!shift) return ch;
    return switch (ch) {
        '1' => '!', '2' => '@', '3' => '#', '4' => '$', '5' => '%',
        '6' => '^', '7' => '&', '8' => '*', '9' => '(', '0' => ')',
        '-' => '_', '=' => '+', '[' => '{', ']' => '}', ';' => ':',
        '\'' => '"', '`' => '~', '\\' => '|', ',' => '<', '.' => '>', '/' => '?',
        else => ch,
    };
}

fn secondsSince(io: std.Io, started: std.Io.Timestamp) f64 {
    const elapsed = started.durationTo(std.Io.Timestamp.now(io, .awake));
    return @as(f64, @floatFromInt(elapsed.nanoseconds)) / 1e9;
}
