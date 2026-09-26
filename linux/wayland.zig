const std = @import("std");

pub const WaylandEvent = union(enum) {
    pointer_motion: struct { x: f32, y: f32 },
    pointer_button: struct { button: u32, state: u32, x: f32, y: f32, serial: u32 },
    keyboard_key: struct { key: u32, state: u32 },
    configure: struct { width: u32, height: u32, maximized: bool = false },
    close,
};

const embedded_adwaita_cursors = @embedFile("adwaita_cursors.bin");

extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn read(fd: c_int, buf: [*]u8, count: usize) isize;

pub const Cursor = enum {
    arrow,
    text,
    crosshair,
    resize_ns,
    resize_ew,
    resize_nwse,
    resize_nesw,
};

pub const WaylandClient = struct {
    const ShmBuffer = struct {
        id: u32,
        pool_id: u32,
        fd: std.posix.fd_t,
        pixels: []u32,
        width: u32,
        height: u32,
        busy: bool = false,
    };
    const CursorData = struct {
        buffer: ShmBuffer,
        hotspot_x: i32,
        hotspot_y: i32,
    };

    allocator: std.mem.Allocator,
    socket_fd: std.posix.fd_t,

    next_id: u32 = 2,

    compositor_id: u32 = 0,
    shm_id: u32 = 0,
    xdg_wm_base_id: u32 = 0,
    seat_id: u32 = 0,

    surface_id: u32 = 0,
    xdg_surface_id: u32 = 0,
    xdg_toplevel_id: u32 = 0,
    pointer_id: u32 = 0,
    keyboard_id: u32 = 0,

    cursor_surface_id: u32 = 0,
    cursor_items: [std.meta.tags(Cursor).len]CursorData = undefined,
    cursor_count: usize = 0,
    pointer_enter_serial: ?u32 = null,
    last_pointer_serial: u32 = 0,
    current_cursor: ?Cursor = null,

    buffers: std.ArrayList(ShmBuffer) = .empty,
    frame_index: ?usize = null,
    pixels: []u32 = &.{},

    width: u32 = 1000,
    height: u32 = 720,

    pointer_x: f32 = 0,
    pointer_y: f32 = 0,

    recv_buf: [32768]u8 = undefined,
    recv_len: usize = 0,

    fn allocId(self: *WaylandClient) u32 {
        const id = self.next_id;
        self.next_id += 1;
        return id;
    }

    pub fn connect(allocator: std.mem.Allocator, initial_width: u32, initial_height: u32) !WaylandClient {
        const runtime_dir = if (std.c.getenv("XDG_RUNTIME_DIR")) |val| std.mem.span(val) else return error.NoXdgRuntimeDir;
        const display = if (std.c.getenv("WAYLAND_DISPLAY")) |val| std.mem.span(val) else "wayland-0";

        var path_buf: [std.posix.PATH_MAX]u8 = undefined;
        const socket_path = if (std.fs.path.isAbsolute(display))
            display
        else
            std.fmt.bufPrintZ(&path_buf, "{s}/{s}", .{ runtime_dir, display }) catch return error.PathTooLong;

        const fd = std.c.socket(std.posix.AF.UNIX, std.posix.SOCK.STREAM | std.posix.SOCK.CLOEXEC, 0);
        if (fd < 0) return error.SocketCreationFailed;
        errdefer _ = std.c.close(fd);

        var addr: std.os.linux.sockaddr.un = .{ .family = std.posix.AF.UNIX, .path = undefined };
        @memset(&addr.path, 0);
        if (socket_path.len >= addr.path.len) return error.PathTooLong;
        @memcpy(addr.path[0..socket_path.len], socket_path);

        const addr_len: std.posix.socklen_t = @intCast(@sizeOf(std.posix.sa_family_t) + socket_path.len + 1);
        if (std.c.connect(fd, @ptrCast(&addr), addr_len) != 0) {
            return error.ConnectionFailed;
        }

        var client: WaylandClient = .{
            .allocator = allocator,
            .socket_fd = fd,
            .width = initial_width,
            .height = initial_height,
        };

        // 1. Get registry
        const registry_id = client.allocId();
        try client.sendMsg(1, 1, .{registry_id});

        // 2. Sync callback to wait for registry events
        const sync_cb_id = client.allocId();
        try client.sendMsg(1, 0, .{sync_cb_id});

        // 3. Process events until sync callback fires
        var synced = false;
        while (!synced) {
            try client.readMessages();
            while (client.popEvent()) |ev| {
                if (ev == .sync and ev.sync == sync_cb_id) {
                    synced = true;
                    break;
                }
            }
        }

        if (client.compositor_id == 0 or client.shm_id == 0 or client.xdg_wm_base_id == 0) {
            return error.MissingRequiredWaylandGlobals;
        }

        // 4. Create surface & XDG toplevel
        client.surface_id = client.allocId();
        try client.sendMsg(client.compositor_id, 0, .{client.surface_id});

        client.xdg_surface_id = client.allocId();
        try client.sendMsg(client.xdg_wm_base_id, 2, .{ client.xdg_surface_id, client.surface_id });

        client.xdg_toplevel_id = client.allocId();
        try client.sendMsg(client.xdg_surface_id, 1, .{client.xdg_toplevel_id});

        try client.sendToplevelString(client.xdg_toplevel_id, 2, "SAM 3");
        try client.sendToplevelString(client.xdg_toplevel_id, 3, "sam3");
        try client.sendMsg(client.xdg_toplevel_id, 8, .{ @as(i32, 640), @as(i32, 480) });

        // 5. Setup pointer & keyboard if seat is available
        if (client.seat_id != 0) {
            client.pointer_id = client.allocId();
            try client.sendMsg(client.seat_id, 0, .{client.pointer_id});
            client.keyboard_id = client.allocId();
            try client.sendMsg(client.seat_id, 1, .{client.keyboard_id});
            try client.setupCursor();
        }

        // Initial commit to request configure
        try client.sendMsg(client.surface_id, 6, .{});

        // 6. Create initial SHM buffer
        try client.resizeShmBuffer(initial_width, initial_height);

        return client;
    }

    pub fn deinit(self: *WaylandClient) void {
        for (self.cursor_items[0..self.cursor_count]) |item| self.destroyBuffer(item.buffer);
        if (self.cursor_surface_id != 0) self.sendMsg(self.cursor_surface_id, 0, .{}) catch {};
        for (self.buffers.items) |buffer| self.destroyBuffer(buffer);
        self.buffers.deinit(self.allocator);
        _ = std.c.close(self.socket_fd);
    }

    pub fn resizeShmBuffer(self: *WaylandClient, w: u32, h: u32) !void {
        self.width = w;
        self.height = h;
        self.frame_index = null;
        self.pixels = &.{};
        self.discardUnusedBuffers();
        try self.createBuffer(w, h);
    }

    fn createBuffer(self: *WaylandClient, w: u32, h: u32) !void {
        const buffer = try self.createShmBuffer(w, h, 1); // XRGB8888
        try self.buffers.append(self.allocator, buffer);
    }

    fn createShmBuffer(self: *WaylandClient, w: u32, h: u32, format: u32) !ShmBuffer {
        const stride = w * 4;
        const size = stride * h;

        const name = "sam3-wl-shm";
        const res = std.os.linux.syscall2(.memfd_create, @intFromPtr(name), 0);
        const fd: std.posix.fd_t = @intCast(res);
        if (fd < 0) return error.MemfdFailed;
        errdefer _ = std.c.close(fd);

        if (std.c.ftruncate(fd, @intCast(size)) != 0) return error.FtruncateFailed;

        const mmap_ptr = try std.posix.mmap(null, size, .{ .READ = true, .WRITE = true }, .{ .TYPE = .SHARED }, fd, 0);
        const pixels: []u32 = @alignCast(std.mem.bytesAsSlice(u32, mmap_ptr));
        errdefer std.posix.munmap(@alignCast(std.mem.sliceAsBytes(pixels)));

        // Create pool & buffer
        const pool_id = self.allocId();
        try self.sendCreatePool(self.shm_id, pool_id, fd, @intCast(size));

        const buffer_id = self.allocId();
        try self.sendMsg(pool_id, 0, .{ buffer_id, @as(i32, 0), @as(i32, @intCast(w)), @as(i32, @intCast(h)), @as(i32, @intCast(stride)), format });
        return .{ .id = buffer_id, .pool_id = pool_id, .fd = fd, .pixels = pixels, .width = w, .height = h };
    }

    fn destroyBuffer(self: *WaylandClient, buffer: ShmBuffer) void {
        self.sendMsg(buffer.id, 0, .{}) catch {}; // wl_buffer.destroy
        self.sendMsg(buffer.pool_id, 1, .{}) catch {}; // wl_shm_pool.destroy
        std.posix.munmap(@alignCast(std.mem.sliceAsBytes(buffer.pixels)));
        _ = std.c.close(buffer.fd);
    }

    const ParsedCursor = struct {
        width: u32,
        height: u32,
        hotspot_x: i32,
        hotspot_y: i32,
        pixels: [64 * 64]u32,
    };

    fn parseXcursor(data: []const u8, target_size: u32) ?ParsedCursor {
        if (data.len < 16 or !std.mem.startsWith(u8, data, "Xcur")) return null;
        const ntoc = std.mem.readInt(u32, data[12..16], .little);
        if (data.len < 16 + ntoc * 12) return null;

        var best_pos: ?usize = null;
        var best_diff: u32 = std.math.maxInt(u32);

        for (0..ntoc) |i| {
            const offset = 16 + i * 12;
            const c_type = std.mem.readInt(u32, data[offset..][0..4], .little);
            const c_size = std.mem.readInt(u32, data[offset + 4 ..][0..4], .little);
            const c_pos = std.mem.readInt(u32, data[offset + 8 ..][0..4], .little);
            if (c_type == 0xfffd0002) {
                const diff = if (c_size >= target_size) c_size - target_size else target_size - c_size;
                if (diff < best_diff) {
                    best_diff = diff;
                    best_pos = c_pos;
                }
            }
        }

        const pos = best_pos orelse return null;
        if (data.len < pos + 36) return null;
        const chunk = data[pos..];
        const w = std.mem.readInt(u32, chunk[16..20], .little);
        const h = std.mem.readInt(u32, chunk[20..24], .little);
        const xhot = std.mem.readInt(u32, chunk[24..28], .little);
        const yhot = std.mem.readInt(u32, chunk[28..32], .little);
        if (w > 64 or h > 64 or w == 0 or h == 0) return null;
        const pixel_bytes_len = @as(usize, w) * h * 4;
        if (chunk.len < 36 + pixel_bytes_len) return null;

        var cur = ParsedCursor{
            .width = w,
            .height = h,
            .hotspot_x = @intCast(xhot),
            .hotspot_y = @intCast(yhot),
            .pixels = undefined,
        };
        const pixel_bytes = chunk[36 .. 36 + pixel_bytes_len];
        for (0..w * h) |i| {
            cur.pixels[i] = std.mem.readInt(u32, pixel_bytes[i * 4 ..][0..4], .little);
        }
        return cur;
    }

    fn loadEmbeddedCursor(kind: Cursor) ParsedCursor {
        const entry_size = 16 + 24 * 24 * 4;
        const offset = @as(usize, @intFromEnum(kind)) * entry_size;
        const chunk = embedded_adwaita_cursors[offset .. offset + entry_size];
        const w = std.mem.readInt(u32, chunk[0..4], .little);
        const h = std.mem.readInt(u32, chunk[4..8], .little);
        const xhot = std.mem.readInt(i32, chunk[8..12], .little);
        const yhot = std.mem.readInt(i32, chunk[12..16], .little);
        var cur = ParsedCursor{
            .width = w,
            .height = h,
            .hotspot_x = xhot,
            .hotspot_y = yhot,
            .pixels = undefined,
        };
        const pixel_bytes = chunk[16..];
        for (0..w * h) |i| {
            cur.pixels[i] = std.mem.readInt(u32, pixel_bytes[i * 4 ..][0..4], .little);
        }
        return cur;
    }

    fn readCursorFile(path: [*:0]const u8, target_size: u32) ?ParsedCursor {
        const fd = open(path, 0);
        if (fd < 0) return null;
        defer _ = std.c.close(fd);

        var buf: [128 * 1024]u8 = undefined;
        var total: usize = 0;
        while (total < buf.len) {
            const rc = read(fd, buf[total..].ptr, buf.len - total);
            if (rc <= 0) break;
            total += @intCast(rc);
        }
        return parseXcursor(buf[0..total], target_size);
    }

    fn loadCursorData(kind: Cursor) ParsedCursor {
        const target_size: u32 = blk: {
            if (std.c.getenv("XCURSOR_SIZE")) |val| {
                const span = std.mem.span(val);
                if (std.fmt.parseInt(u32, span, 10)) |s| break :blk s else |_| {}
            }
            break :blk 24;
        };

        const names: []const []const u8 = switch (kind) {
            .arrow => &.{ "default", "left_ptr" },
            .text => &.{ "text", "xterm", "ibeam" },
            .crosshair => &.{ "crosshair", "cross" },
            .resize_ns => &.{ "ns-resize", "n-resize", "s-resize", "row-resize" },
            .resize_ew => &.{ "ew-resize", "e-resize", "w-resize", "col-resize" },
            .resize_nwse => &.{ "nwse-resize", "nw-resize", "se-resize" },
            .resize_nesw => &.{ "nesw-resize", "ne-resize", "sw-resize" },
        };

        var themes_buf: [3][]const u8 = undefined;
        var theme_count: usize = 0;
        if (std.c.getenv("XCURSOR_THEME")) |val| {
            const span = std.mem.span(val);
            if (span.len > 0) {
                themes_buf[theme_count] = span;
                theme_count += 1;
            }
        }
        themes_buf[theme_count] = "Adwaita";
        theme_count += 1;
        themes_buf[theme_count] = "default";
        theme_count += 1;
        const themes = themes_buf[0..theme_count];

        var path_buf: [512]u8 = undefined;

        if (std.c.getenv("XCURSOR_PATH")) |env_path| {
            var it = std.mem.splitScalar(u8, std.mem.span(env_path), ':');
            while (it.next()) |dir| {
                if (dir.len == 0) continue;
                for (themes) |theme| {
                    for (names) |name| {
                        const path = std.fmt.bufPrintZ(&path_buf, "{s}/{s}/cursors/{s}", .{ dir, theme, name }) catch continue;
                        if (readCursorFile(path, target_size)) |cur| return cur;
                    }
                }
            }
        }

        const static_dirs = [_][]const u8{
            "/run/current-system/sw/share/icons",
            "/usr/share/icons",
            "/usr/local/share/icons",
        };

        for (themes) |theme| {
            if (std.c.getenv("HOME")) |home_ptr| {
                const home = std.mem.span(home_ptr);
                const user_dirs = [_][]const u8{ ".icons", ".local/share/icons" };
                for (user_dirs) |sub| {
                    for (names) |name| {
                        const path = std.fmt.bufPrintZ(&path_buf, "{s}/{s}/{s}/cursors/{s}", .{ home, sub, theme, name }) catch continue;
                        if (readCursorFile(path, target_size)) |cur| return cur;
                    }
                }
            }

            for (static_dirs) |dir| {
                for (names) |name| {
                    const path = std.fmt.bufPrintZ(&path_buf, "{s}/{s}/cursors/{s}", .{ dir, theme, name }) catch continue;
                    if (readCursorFile(path, target_size)) |cur| return cur;
                }
            }

            if (std.c.getenv("XDG_DATA_DIRS")) |xdg_dirs| {
                var it = std.mem.splitScalar(u8, std.mem.span(xdg_dirs), ':');
                while (it.next()) |data_dir| {
                    if (data_dir.len == 0) continue;
                    for (names) |name| {
                        const path = std.fmt.bufPrintZ(&path_buf, "{s}/icons/{s}/cursors/{s}", .{ data_dir, theme, name }) catch continue;
                        if (readCursorFile(path, target_size)) |cur| return cur;
                    }
                }
            }
        }

        return loadEmbeddedCursor(kind);
    }

    fn setupCursor(self: *WaylandClient) !void {
        self.cursor_surface_id = self.allocId();
        try self.sendMsg(self.compositor_id, 0, .{self.cursor_surface_id});
        inline for (std.meta.tags(Cursor)) |kind| {
            const parsed = loadCursorData(kind);
            const buffer = try self.createShmBuffer(parsed.width, parsed.height, 0); // ARGB8888
            const len = @as(usize, parsed.width) * parsed.height;
            @memcpy(buffer.pixels[0..len], parsed.pixels[0..len]);
            self.cursor_items[self.cursor_count] = .{
                .buffer = buffer,
                .hotspot_x = parsed.hotspot_x,
                .hotspot_y = parsed.hotspot_y,
            };
            self.cursor_count += 1;
        }
    }

    pub fn setCursor(self: *WaylandClient, kind: Cursor) !void {
        const serial = self.pointer_enter_serial orelse (if (self.last_pointer_serial != 0) self.last_pointer_serial else return);
        if (self.current_cursor == kind) return;
        const item = self.cursor_items[@intFromEnum(kind)];
        try self.sendMsg(self.cursor_surface_id, 1, .{ item.buffer.id, @as(i32, 0), @as(i32, 0) });
        try self.sendMsg(self.cursor_surface_id, 2, .{ @as(i32, 0), @as(i32, 0), @as(i32, @intCast(item.buffer.width)), @as(i32, @intCast(item.buffer.height)) });
        try self.sendMsg(self.cursor_surface_id, 6, .{});
        try self.sendMsg(self.pointer_id, 0, .{ serial, self.cursor_surface_id, item.hotspot_x, item.hotspot_y });
        self.current_cursor = kind;
    }

    pub fn startInteractiveMove(self: *WaylandClient, serial: u32) !void {
        if (self.seat_id == 0 or self.xdg_toplevel_id == 0) return;
        try self.sendMsg(self.xdg_toplevel_id, 5, .{ self.seat_id, serial });
    }

    pub fn startInteractiveResize(self: *WaylandClient, serial: u32, edges: u32) !void {
        if (self.seat_id == 0 or self.xdg_toplevel_id == 0) return;
        try self.sendMsg(self.xdg_toplevel_id, 6, .{ self.seat_id, serial, edges });
    }

    pub fn setMaximized(self: *WaylandClient) !void {
        if (self.xdg_toplevel_id == 0) return;
        try self.sendMsg(self.xdg_toplevel_id, 9, .{});
    }

    pub fn unsetMaximized(self: *WaylandClient) !void {
        if (self.xdg_toplevel_id == 0) return;
        try self.sendMsg(self.xdg_toplevel_id, 10, .{});
    }

    pub fn setMinimized(self: *WaylandClient) !void {
        if (self.xdg_toplevel_id == 0) return;
        try self.sendMsg(self.xdg_toplevel_id, 13, .{});
    }

    fn discardUnusedBuffers(self: *WaylandClient) void {
        var i: usize = 0;
        while (i < self.buffers.items.len) {
            const buffer = self.buffers.items[i];
            if (!buffer.busy and (buffer.width != self.width or buffer.height != self.height)) {
                self.destroyBuffer(buffer);
                _ = self.buffers.orderedRemove(i);
            } else {
                i += 1;
            }
        }
    }

    pub fn beginFrame(self: *WaylandClient) !bool {
        self.discardUnusedBuffers();
        var matching: usize = 0;
        for (self.buffers.items, 0..) |buffer, i| {
            if (buffer.width == self.width and buffer.height == self.height) {
                matching += 1;
                if (!buffer.busy) {
                    self.frame_index = i;
                    self.pixels = buffer.pixels;
                    return true;
                }
            }
        }
        if (matching >= 2) return false;
        try self.createBuffer(self.width, self.height);
        self.frame_index = self.buffers.items.len - 1;
        self.pixels = self.buffers.items[self.frame_index.?].pixels;
        return true;
    }

    pub fn commitFrame(self: *WaylandClient) !void {
        const index = self.frame_index orelse return error.NoFrame;
        try self.sendMsg(self.surface_id, 1, .{ self.buffers.items[index].id, @as(i32, 0), @as(i32, 0) });
        try self.sendMsg(self.surface_id, 2, .{ @as(i32, 0), @as(i32, 0), @as(i32, @intCast(self.width)), @as(i32, @intCast(self.height)) });
        try self.sendMsg(self.surface_id, 6, .{});
        self.buffers.items[index].busy = true;
        self.frame_index = null;
        self.pixels = &.{};
    }

    pub fn pollEvent(self: *WaylandClient, timeout_ms: i32) !?WaylandEvent {
        // If buffer has unprocessed data
        if (self.recv_len > 0) {
            if (self.parseNextEvent()) |ev| return ev;
        }

        var pfd = [_]std.posix.pollfd{.{
            .fd = self.socket_fd,
            .events = std.posix.POLL.IN,
            .revents = 0,
        }};
        const n = try std.posix.poll(&pfd, timeout_ms);
        if (n == 0) return null;

        try self.readMessages();
        return self.parseNextEvent();
    }

    fn readMessages(self: *WaylandClient) !void {
        if (self.recv_len >= self.recv_buf.len) return;
        const n = std.posix.system.recv(self.socket_fd, self.recv_buf[self.recv_len..].ptr, self.recv_buf.len - self.recv_len, 0);
        if (n > 0) {
            self.recv_len += @intCast(n);
        }
    }

    const InternalEvent = union(enum) {
        sync: u32,
        other,
    };

    fn popEvent(self: *WaylandClient) ?InternalEvent {
        if (self.recv_len < 8) return null;
        const bytes = self.recv_buf[0..self.recv_len];
        const id = std.mem.readInt(u32, bytes[0..4], .little);
        const header = std.mem.readInt(u32, bytes[4..8], .little);
        const opcode: u16 = @truncate(header);
        const size: u16 = @truncate(header >> 16);

        if (self.recv_len < size) return null;

        defer {
            const remaining = self.recv_len - size;
            if (remaining > 0) {
                std.mem.copyForwards(u8, self.recv_buf[0..remaining], self.recv_buf[size..self.recv_len]);
            }
            self.recv_len = remaining;
        }

        // Registry global event: id == 2, opcode == 0
        if (id == 2 and opcode == 0 and size >= 20) {
            const name = std.mem.readInt(u32, bytes[8..12], .little);
            const str_len = std.mem.readInt(u32, bytes[12..16], .little);
            if (16 + str_len <= size) {
                const iface_name = std.mem.sliceTo(bytes[16..][0..str_len], 0);
                self.handleGlobal(name, iface_name);
            }
            return .other;
        }

        // Callback done event: opcode == 0
        if (opcode == 0 and size == 12) {
            return .{ .sync = id };
        }

        return .other;
    }

    fn handleGlobal(self: *WaylandClient, name: u32, iface: []const u8) void {
        if (std.mem.eql(u8, iface, "wl_compositor")) {
            self.compositor_id = self.allocId();
            self.sendRegistryBind(name, "wl_compositor", 4, self.compositor_id) catch {};
        } else if (std.mem.eql(u8, iface, "wl_shm")) {
            self.shm_id = self.allocId();
            self.sendRegistryBind(name, "wl_shm", 1, self.shm_id) catch {};
        } else if (std.mem.eql(u8, iface, "xdg_wm_base")) {
            self.xdg_wm_base_id = self.allocId();
            self.sendRegistryBind(name, "xdg_wm_base", 1, self.xdg_wm_base_id) catch {};
        } else if (std.mem.eql(u8, iface, "wl_seat")) {
            self.seat_id = self.allocId();
            self.sendRegistryBind(name, "wl_seat", 5, self.seat_id) catch {};
        }
    }

    fn parseNextEvent(self: *WaylandClient) ?WaylandEvent {
        while (self.recv_len >= 8) {
            const bytes = self.recv_buf[0..self.recv_len];
            const id = std.mem.readInt(u32, bytes[0..4], .little);
            const header = std.mem.readInt(u32, bytes[4..8], .little);
            const opcode: u16 = @truncate(header);
            const size: u16 = @truncate(header >> 16);

            if (self.recv_len < size) return null;

            const msg_bytes = bytes[0..size];
            defer {
                const remaining = self.recv_len - size;
                if (remaining > 0) {
                    std.mem.copyForwards(u8, self.recv_buf[0..remaining], self.recv_buf[size..self.recv_len]);
                }
                self.recv_len = remaining;
            }

            // xdg_wm_base ping -> send pong
            if (id == self.xdg_wm_base_id and opcode == 0 and size >= 12) {
                const serial = std.mem.readInt(u32, msg_bytes[8..12], .little);
                self.sendMsg(self.xdg_wm_base_id, 3, .{serial}) catch {};
                continue;
            }

            // xdg_surface configure -> send ack_configure
            if (id == self.xdg_surface_id and opcode == 0 and size >= 12) {
                const serial = std.mem.readInt(u32, msg_bytes[8..12], .little);
                self.sendMsg(self.xdg_surface_id, 4, .{serial}) catch {};
                continue;
            }

            // xdg_toplevel configure
            if (id == self.xdg_toplevel_id and opcode == 0 and size >= 16) {
                const w = std.mem.readInt(i32, msg_bytes[8..12], .little);
                const h = std.mem.readInt(i32, msg_bytes[12..16], .little);
                var is_max = false;
                if (size >= 20) {
                    const states_len = std.mem.readInt(u32, msg_bytes[16..20], .little);
                    var offset: usize = 20;
                    while (offset + 4 <= size and offset - 20 < states_len) : (offset += 4) {
                        const state_val = std.mem.readInt(u32, msg_bytes[offset..][0..4], .little);
                        if (state_val == 1) { // XDG_TOPLEVEL_STATE_MAXIMIZED
                            is_max = true;
                        }
                    }
                }
                const width: u32 = if (w > 0) @intCast(w) else 0;
                const height: u32 = if (h > 0) @intCast(h) else 0;
                return .{ .configure = .{ .width = width, .height = height, .maximized = is_max } };
            }

            // xdg_toplevel close
            if (id == self.xdg_toplevel_id and opcode == 1) {
                return .close;
            }

            // wl_buffer.release: the compositor has finished reading these pixels.
            if (opcode == 0 and size == 8) {
                for (self.buffers.items) |*buffer| {
                    if (id == buffer.id) {
                        buffer.busy = false;
                        break;
                    }
                }
            }

            // Pointer enter supplies the coordinates of the first click, even if
            // there has not yet been a motion event.
            if (id == self.pointer_id and opcode == 0 and size >= 24) {
                self.pointer_enter_serial = std.mem.readInt(u32, msg_bytes[8..12], .little);
                self.current_cursor = null;
                const raw_x = std.mem.readInt(i32, msg_bytes[16..20], .little);
                const raw_y = std.mem.readInt(i32, msg_bytes[20..24], .little);
                self.pointer_x = @as(f32, @floatFromInt(raw_x)) / 256.0;
                self.pointer_y = @as(f32, @floatFromInt(raw_y)) / 256.0;
                return .{ .pointer_motion = .{ .x = self.pointer_x, .y = self.pointer_y } };
            }

            if (id == self.pointer_id and opcode == 1 and size >= 20) {
                self.pointer_enter_serial = null;
                self.current_cursor = null;
                continue;
            }

            // wl_pointer.motion is (time, surface_x, surface_y).
            if (id == self.pointer_id and opcode == 2 and size >= 20) {
                const raw_x = std.mem.readInt(i32, msg_bytes[12..16], .little);
                const raw_y = std.mem.readInt(i32, msg_bytes[16..20], .little);
                self.pointer_x = @as(f32, @floatFromInt(raw_x)) / 256.0;
                self.pointer_y = @as(f32, @floatFromInt(raw_y)) / 256.0;
                return .{ .pointer_motion = .{ .x = self.pointer_x, .y = self.pointer_y } };
            }

            // Pointer button
            if (id == self.pointer_id and opcode == 3 and size >= 24) {
                const serial = std.mem.readInt(u32, msg_bytes[8..12], .little);
                self.last_pointer_serial = serial;
                const btn = std.mem.readInt(u32, msg_bytes[16..20], .little);
                const state = std.mem.readInt(u32, msg_bytes[20..24], .little);
                return .{ .pointer_button = .{
                    .button = btn,
                    .state = state,
                    .x = self.pointer_x,
                    .y = self.pointer_y,
                    .serial = serial,
                } };
            }

            // Keyboard key
            if (id == self.keyboard_id and opcode == 3 and size >= 24) {
                const key = std.mem.readInt(u32, msg_bytes[16..20], .little);
                const state = std.mem.readInt(u32, msg_bytes[20..24], .little);
                return .{ .keyboard_key = .{ .key = key, .state = state } };
            }
        }
        return null;
    }

    // Wire sender helpers
    fn sendMsg(self: *WaylandClient, id: u32, opcode: u16, args: anytype) !void {
        const fields = @typeInfo(@TypeOf(args)).@"struct".fields;
        var words: [2 + fields.len]u32 = undefined;
        words[0] = id;
        words[1] = (@as(u32, words.len * 4) << 16) | opcode;
        inline for (fields, 0..) |f, i| {
            words[2 + i] = @as(u32, @bitCast(@field(args, f.name)));
        }
        _ = std.posix.system.send(self.socket_fd, @ptrCast(&words), words.len * 4, 0);
    }

    fn sendToplevelString(self: *WaylandClient, toplevel_id: u32, opcode: u16, str: []const u8) !void {
        const str_padded = ((str.len + 1 + 3) / 4) * 4;
        const total = 8 + 4 + str_padded;
        var buf: [64]u8 = undefined;
        std.mem.writeInt(u32, buf[0..4], toplevel_id, .little);
        std.mem.writeInt(u32, buf[4..8], (@as(u32, @intCast(total)) << 16) | opcode, .little);
        std.mem.writeInt(u32, buf[8..12], @intCast(str.len + 1), .little);
        @memset(buf[12 .. 12 + str_padded], 0);
        @memcpy(buf[12 .. 12 + str.len], str);
        _ = std.posix.system.send(self.socket_fd, buf[0..total].ptr, total, 0);
    }

    fn sendRegistryBind(self: *WaylandClient, name: u32, iface: []const u8, version: u32, new_id: u32) !void {
        const str_padded = ((iface.len + 1 + 3) / 4) * 4;
        const total_size = 8 + 4 + 4 + str_padded + 4 + 4;

        var buf: [64]u8 = undefined;
        std.mem.writeInt(u32, buf[0..4], 2, .little); // registry id
        std.mem.writeInt(u32, buf[4..8], (@as(u32, @intCast(total_size)) << 16) | 0, .little); // opcode=0 (bind)
        std.mem.writeInt(u32, buf[8..12], name, .little);
        std.mem.writeInt(u32, buf[12..16], @intCast(iface.len + 1), .little);
        @memset(buf[16 .. 16 + str_padded], 0);
        @memcpy(buf[16 .. 16 + iface.len], iface);

        var offset = 16 + str_padded;
        std.mem.writeInt(u32, buf[offset..][0..4], version, .little);
        offset += 4;
        std.mem.writeInt(u32, buf[offset..][0..4], new_id, .little);
        offset += 4;

        _ = std.posix.system.send(self.socket_fd, buf[0..total_size].ptr, total_size, 0);
    }

    fn sendCreatePool(self: *WaylandClient, shm_id: u32, pool_id: u32, fd: std.posix.fd_t, size: i32) !void {
        const Cmsg = extern struct {
            hdr: std.posix.system.cmsghdr,
            fd: i32,
        };
        var cmsg: Cmsg = .{
            .hdr = .{
                .len = @sizeOf(std.posix.system.cmsghdr) + @sizeOf(i32),
                .level = std.posix.SOL.SOCKET,
                .type = std.posix.SCM.RIGHTS,
            },
            .fd = fd,
        };

        var body: [16]u8 = undefined;
        std.mem.writeInt(u32, body[0..4], shm_id, .little);
        std.mem.writeInt(u32, body[4..8], (16 << 16) | 0, .little); // create_pool
        std.mem.writeInt(u32, body[8..12], pool_id, .little);
        std.mem.writeInt(i32, body[12..16], size, .little);

        const iov: [1]std.posix.iovec_const = .{.{ .base = &body, .len = body.len }};
        const msg: std.posix.system.msghdr_const = .{
            .name = null,
            .namelen = 0,
            .iov = &iov,
            .iovlen = 1,
            .control = &cmsg,
            .controllen = @sizeOf(Cmsg),
            .flags = 0,
        };
        const sent = std.posix.system.sendmsg(self.socket_fd, &msg, 0);
        if (sent < 0) return error.SendFailed;
    }
};

test "pointer coordinates use Wayland fixed-point fields and track enter before motion" {
    var client: WaylandClient = .{
        .allocator = std.testing.allocator,
        .socket_fd = -1,
        .pointer_id = 7,
    };
    try client.buffers.append(std.testing.allocator, .{
        .id = 8, .pool_id = 9, .fd = -1, .pixels = &.{}, .width = 1, .height = 1, .busy = true,
    });
    defer client.buffers.deinit(std.testing.allocator);

    // wl_pointer.enter(serial, surface, surface_x, surface_y)
    var enter: [24]u8 = @splat(0);
    std.mem.writeInt(u32, enter[0..4], 7, .little);
    std.mem.writeInt(u32, enter[4..8], 24 << 16, .little);
    std.mem.writeInt(i32, enter[16..20], 120 * 256, .little);
    std.mem.writeInt(i32, enter[20..24], 80 * 256, .little);
    @memcpy(client.recv_buf[0..enter.len], &enter);
    client.recv_len = enter.len;
    const enter_event = client.parseNextEvent().?;
    try std.testing.expectEqual(@as(f32, 120), enter_event.pointer_motion.x);
    try std.testing.expectEqual(@as(f32, 80), enter_event.pointer_motion.y);
    try std.testing.expectEqual(@as(f32, 120), client.pointer_x);
    try std.testing.expectEqual(@as(f32, 80), client.pointer_y);

    // wl_pointer.motion(time, surface_x, surface_y): time must not become X.
    var motion: [20]u8 = @splat(0);
    std.mem.writeInt(u32, motion[0..4], 7, .little);
    std.mem.writeInt(u32, motion[4..8], (20 << 16) | 2, .little);
    std.mem.writeInt(u32, motion[8..12], 999_999, .little);
    std.mem.writeInt(i32, motion[12..16], 42 * 256 + 128, .little);
    std.mem.writeInt(i32, motion[16..20], 31 * 256, .little);
    @memcpy(client.recv_buf[0..motion.len], &motion);
    client.recv_len = motion.len;
    const event = client.parseNextEvent().?;
    try std.testing.expectEqual(@as(f32, 42.5), event.pointer_motion.x);
    try std.testing.expectEqual(@as(f32, 31), event.pointer_motion.y);

    var release: [8]u8 = @splat(0);
    std.mem.writeInt(u32, release[0..4], 8, .little);
    std.mem.writeInt(u32, release[4..8], 8 << 16, .little);
    @memcpy(client.recv_buf[0..release.len], &release);
    client.recv_len = release.len;
    try std.testing.expect(client.parseNextEvent() == null);
    try std.testing.expect(!client.buffers.items[0].busy);
}

test "loadCursorData retrieves GNOME Adwaita cursor geometry and non-empty pixels" {
    inline for (std.meta.tags(Cursor)) |kind| {
        const cur = WaylandClient.loadCursorData(kind);
        try std.testing.expect(cur.width >= 24 and cur.height >= 24);
        var non_zero: usize = 0;
        for (cur.pixels[0 .. cur.width * cur.height]) |p| {
            if (p != 0) non_zero += 1;
        }
        try std.testing.expect(non_zero > 0);
        switch (kind) {
            .arrow => {
                try std.testing.expectEqual(@as(i32, 3), cur.hotspot_x);
                try std.testing.expectEqual(@as(i32, 1), cur.hotspot_y);
            },
            .text => {
                try std.testing.expectEqual(@as(i32, 11), cur.hotspot_x);
                try std.testing.expectEqual(@as(i32, 12), cur.hotspot_y);
            },
            .crosshair, .resize_ns, .resize_ew, .resize_nwse, .resize_nesw => {
                try std.testing.expect(cur.hotspot_x >= 0 and cur.hotspot_y >= 0);
            },
        }
    }
}
