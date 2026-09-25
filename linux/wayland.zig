const std = @import("std");

pub const WaylandEvent = union(enum) {
    pointer_motion: struct { x: f32, y: f32 },
    pointer_button: struct { button: u32, state: u32, x: f32, y: f32 },
    keyboard_key: struct { key: u32, state: u32 },
    configure: struct { width: u32, height: u32 },
    close,
};

pub const WaylandClient = struct {
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
    shm_pool_id: u32 = 0,
    buffer_id: u32 = 0,
    pointer_id: u32 = 0,
    keyboard_id: u32 = 0,

    shm_fd: std.posix.fd_t = -1,
    shm_size: usize = 0,
    pixels: []u32 = &.{},

    width: u32 = 1000,
    height: u32 = 720,

    pointer_x: f32 = 0,
    pointer_y: f32 = 0,

    recv_buf: [32768]u8 = undefined,
    recv_len: usize = 0,

    pub fn allocId(self: *WaylandClient) u32 {
        const id = self.next_id;
        self.next_id += 1;
        return id;
    }

    pub fn connect(allocator: std.mem.Allocator, initial_width: u32, initial_height: u32) !WaylandClient {
        const runtime_dir = if (std.c.getenv("XDG_RUNTIME_DIR")) |val| std.mem.span(val) else return error.NoXdgRuntimeDir;
        const display = if (std.c.getenv("WAYLAND_DISPLAY")) |val| std.mem.span(val) else "wayland-0";

        var path_buf: [std.posix.PATH_MAX]u8 = undefined;
        const socket_path = std.fmt.bufPrintZ(&path_buf, "{s}/{s}", .{ runtime_dir, display }) catch return error.PathTooLong;

        const fd = std.c.socket(std.posix.AF.UNIX, std.posix.SOCK.STREAM | std.posix.SOCK.CLOEXEC, 0);
        if (fd < 0) return error.SocketCreationFailed;
        errdefer _ = std.c.close(fd);

        var addr: std.os.linux.sockaddr.un = .{ .family = std.posix.AF.UNIX, .path = undefined };
        @memset(&addr.path, 0);
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
        try client.sendDisplayGetRegistry(registry_id);

        // 2. Sync callback to wait for registry events
        const sync_cb_id = client.allocId();
        try client.sendDisplaySync(sync_cb_id);

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
        try client.sendCompositorCreateSurface(client.compositor_id, client.surface_id);

        client.xdg_surface_id = client.allocId();
        try client.sendXdgWmBaseGetXdgSurface(client.xdg_wm_base_id, client.xdg_surface_id, client.surface_id);

        client.xdg_toplevel_id = client.allocId();
        try client.sendXdgSurfaceGetToplevel(client.xdg_surface_id, client.xdg_toplevel_id);

        try client.sendToplevelSetTitle(client.xdg_toplevel_id, "SAM 3");
        try client.sendToplevelSetAppId(client.xdg_toplevel_id, "sam3");

        // 5. Setup pointer & keyboard if seat is available
        if (client.seat_id != 0) {
            client.pointer_id = client.allocId();
            try client.sendSeatGetPointer(client.seat_id, client.pointer_id);
            client.keyboard_id = client.allocId();
            try client.sendSeatGetKeyboard(client.seat_id, client.keyboard_id);
        }

        // Initial commit to request configure
        try client.sendSurfaceCommit(client.surface_id);

        // 6. Create initial SHM buffer
        try client.resizeShmBuffer(initial_width, initial_height);

        return client;
    }

    pub fn deinit(self: *WaylandClient) void {
        if (self.pixels.len > 0) {
            std.posix.munmap(@alignCast(std.mem.sliceAsBytes(self.pixels)));
        }
        if (self.shm_fd >= 0) _ = std.c.close(self.shm_fd);
        _ = std.c.close(self.socket_fd);
    }

    pub fn resizeShmBuffer(self: *WaylandClient, w: u32, h: u32) !void {
        self.width = w;
        self.height = h;

        if (self.pixels.len > 0) {
            std.posix.munmap(@alignCast(std.mem.sliceAsBytes(self.pixels)));
            self.pixels = &.{};
        }
        if (self.shm_fd >= 0) {
            _ = std.c.close(self.shm_fd);
            self.shm_fd = -1;
        }

        const stride = w * 4;
        const size = stride * h;
        self.shm_size = size;

        // Create memfd
        const name = "sam3-wl-shm";
        const res = std.os.linux.syscall2(.memfd_create, @intFromPtr(name), 0);
        const fd: std.posix.fd_t = @intCast(res);
        if (fd < 0) return error.MemfdFailed;
        self.shm_fd = fd;

        if (std.c.ftruncate(fd, @intCast(size)) != 0) return error.FtruncateFailed;

        const mmap_ptr = try std.posix.mmap(null, size, .{ .READ = true, .WRITE = true }, .{ .TYPE = .SHARED }, fd, 0);
        self.pixels = @alignCast(std.mem.bytesAsSlice(u32, mmap_ptr));

        // Create pool & buffer
        self.shm_pool_id = self.allocId();
        try self.sendCreatePool(self.shm_id, self.shm_pool_id, fd, @intCast(size));

        self.buffer_id = self.allocId();
        // format 1 = WL_SHM_FORMAT_XRGB8888
        try self.sendCreateBuffer(self.shm_pool_id, self.buffer_id, 0, @intCast(w), @intCast(h), @intCast(stride), 1);
    }

    pub fn commitFrame(self: *WaylandClient) !void {
        try self.sendSurfaceAttach(self.surface_id, self.buffer_id, 0, 0);
        try self.sendSurfaceDamage(self.surface_id, 0, 0, @intCast(self.width), @intCast(self.height));
        try self.sendSurfaceCommit(self.surface_id);
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
                self.sendXdgWmBasePong(serial) catch {};
                continue;
            }

            // xdg_surface configure -> send ack_configure
            if (id == self.xdg_surface_id and opcode == 0 and size >= 12) {
                const serial = std.mem.readInt(u32, msg_bytes[8..12], .little);
                self.sendXdgSurfaceAckConfigure(serial) catch {};
                continue;
            }

            // xdg_toplevel configure
            if (id == self.xdg_toplevel_id and opcode == 0 and size >= 16) {
                const w = std.mem.readInt(i32, msg_bytes[8..12], .little);
                const h = std.mem.readInt(i32, msg_bytes[12..16], .little);
                if (w > 0 and h > 0) {
                    return .{ .configure = .{ .width = @intCast(w), .height = @intCast(h) } };
                }
                continue;
            }

            // xdg_toplevel close
            if (id == self.xdg_toplevel_id and opcode == 1) {
                return .close;
            }

            // Pointer motion
            if (id == self.pointer_id and opcode == 2 and size >= 16) {
                const raw_x = std.mem.readInt(i32, msg_bytes[8..12], .little);
                const raw_y = std.mem.readInt(i32, msg_bytes[12..16], .little);
                self.pointer_x = @as(f32, @floatFromInt(raw_x)) / 256.0;
                self.pointer_y = @as(f32, @floatFromInt(raw_y)) / 256.0;
                return .{ .pointer_motion = .{ .x = self.pointer_x, .y = self.pointer_y } };
            }

            // Pointer button
            if (id == self.pointer_id and opcode == 3 and size >= 24) {
                const btn = std.mem.readInt(u32, msg_bytes[16..20], .little);
                const state = std.mem.readInt(u32, msg_bytes[20..24], .little);
                return .{ .pointer_button = .{
                    .button = btn,
                    .state = state,
                    .x = self.pointer_x,
                    .y = self.pointer_y,
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
    fn sendDisplayGetRegistry(self: *WaylandClient, registry_id: u32) !void {
        var buf: [12]u8 = undefined;
        std.mem.writeInt(u32, buf[0..4], 1, .little); // wl_display id
        std.mem.writeInt(u32, buf[4..8], (12 << 16) | 1, .little); // size=12, opcode=1
        std.mem.writeInt(u32, buf[8..12], registry_id, .little);
        _ = std.posix.system.send(self.socket_fd, &buf, 12, 0);
    }

    fn sendDisplaySync(self: *WaylandClient, cb_id: u32) !void {
        var buf: [12]u8 = undefined;
        std.mem.writeInt(u32, buf[0..4], 1, .little);
        std.mem.writeInt(u32, buf[4..8], (12 << 16) | 0, .little); // opcode=0 (sync)
        std.mem.writeInt(u32, buf[8..12], cb_id, .little);
        _ = std.posix.system.send(self.socket_fd, &buf, 12, 0);
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

    fn sendCompositorCreateSurface(self: *WaylandClient, comp_id: u32, surf_id: u32) !void {
        var buf: [12]u8 = undefined;
        std.mem.writeInt(u32, buf[0..4], comp_id, .little);
        std.mem.writeInt(u32, buf[4..8], (12 << 16) | 0, .little); // create_surface
        std.mem.writeInt(u32, buf[8..12], surf_id, .little);
        _ = std.posix.system.send(self.socket_fd, &buf, 12, 0);
    }

    fn sendXdgWmBaseGetXdgSurface(self: *WaylandClient, wm_id: u32, xdg_surf_id: u32, surf_id: u32) !void {
        var buf: [16]u8 = undefined;
        std.mem.writeInt(u32, buf[0..4], wm_id, .little);
        std.mem.writeInt(u32, buf[4..8], (16 << 16) | 2, .little); // get_xdg_surface
        std.mem.writeInt(u32, buf[8..12], xdg_surf_id, .little);
        std.mem.writeInt(u32, buf[12..16], surf_id, .little);
        _ = std.posix.system.send(self.socket_fd, &buf, 16, 0);
    }

    fn sendXdgSurfaceGetToplevel(self: *WaylandClient, xdg_surf_id: u32, toplevel_id: u32) !void {
        var buf: [12]u8 = undefined;
        std.mem.writeInt(u32, buf[0..4], xdg_surf_id, .little);
        std.mem.writeInt(u32, buf[4..8], (12 << 16) | 1, .little); // get_toplevel
        std.mem.writeInt(u32, buf[8..12], toplevel_id, .little);
        _ = std.posix.system.send(self.socket_fd, &buf, 12, 0);
    }

    fn sendToplevelSetTitle(self: *WaylandClient, toplevel_id: u32, title: []const u8) !void {
        const str_padded = ((title.len + 1 + 3) / 4) * 4;
        const total = 8 + 4 + str_padded;
        var buf: [64]u8 = undefined;
        std.mem.writeInt(u32, buf[0..4], toplevel_id, .little);
        std.mem.writeInt(u32, buf[4..8], (@as(u32, @intCast(total)) << 16) | 2, .little); // set_title
        std.mem.writeInt(u32, buf[8..12], @intCast(title.len + 1), .little);
        @memset(buf[12 .. 12 + str_padded], 0);
        @memcpy(buf[12 .. 12 + title.len], title);
        _ = std.posix.system.send(self.socket_fd, buf[0..total].ptr, total, 0);
    }

    fn sendToplevelSetAppId(self: *WaylandClient, toplevel_id: u32, app_id: []const u8) !void {
        const str_padded = ((app_id.len + 1 + 3) / 4) * 4;
        const total = 8 + 4 + str_padded;
        var buf: [64]u8 = undefined;
        std.mem.writeInt(u32, buf[0..4], toplevel_id, .little);
        std.mem.writeInt(u32, buf[4..8], (@as(u32, @intCast(total)) << 16) | 3, .little); // set_app_id
        std.mem.writeInt(u32, buf[8..12], @intCast(app_id.len + 1), .little);
        @memset(buf[12 .. 12 + str_padded], 0);
        @memcpy(buf[12 .. 12 + app_id.len], app_id);
        _ = std.posix.system.send(self.socket_fd, buf[0..total].ptr, total, 0);
    }

    fn sendSeatGetPointer(self: *WaylandClient, seat_id: u32, pointer_id: u32) !void {
        var buf: [12]u8 = undefined;
        std.mem.writeInt(u32, buf[0..4], seat_id, .little);
        std.mem.writeInt(u32, buf[4..8], (12 << 16) | 0, .little); // get_pointer
        std.mem.writeInt(u32, buf[8..12], pointer_id, .little);
        _ = std.posix.system.send(self.socket_fd, &buf, 12, 0);
    }

    fn sendSeatGetKeyboard(self: *WaylandClient, seat_id: u32, keyboard_id: u32) !void {
        var buf: [12]u8 = undefined;
        std.mem.writeInt(u32, buf[0..4], seat_id, .little);
        std.mem.writeInt(u32, buf[4..8], (12 << 16) | 1, .little); // get_keyboard
        std.mem.writeInt(u32, buf[8..12], keyboard_id, .little);
        _ = std.posix.system.send(self.socket_fd, &buf, 12, 0);
    }

    fn sendCreatePool(self: *WaylandClient, shm_id: u32, pool_id: u32, fd: std.posix.fd_t, size: i32) !void {
        const FdCmsg = extern struct {
            len: usize = @sizeOf(@This()),
            level: i32 = 1, // SOL_SOCKET
            type: i32 = 1,  // SCM_RIGHTS
            fd: i32,
        };
        var cmsg: FdCmsg = .{ .fd = fd };

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
            .controllen = @sizeOf(FdCmsg),
            .flags = 0,
        };
        const sent = std.posix.system.sendmsg(self.socket_fd, &msg, 0);
        if (sent < 0) return error.SendFailed;
    }

    fn sendCreateBuffer(self: *WaylandClient, pool_id: u32, buf_id: u32, offset: i32, w: i32, h: i32, stride: i32, fmt: u32) !void {
        var buf: [32]u8 = undefined;
        std.mem.writeInt(u32, buf[0..4], pool_id, .little);
        std.mem.writeInt(u32, buf[4..8], (32 << 16) | 0, .little); // create_buffer
        std.mem.writeInt(u32, buf[8..12], buf_id, .little);
        std.mem.writeInt(i32, buf[12..16], offset, .little);
        std.mem.writeInt(i32, buf[16..20], w, .little);
        std.mem.writeInt(i32, buf[20..24], h, .little);
        std.mem.writeInt(i32, buf[24..28], stride, .little);
        std.mem.writeInt(u32, buf[28..32], fmt, .little);
        _ = std.posix.system.send(self.socket_fd, &buf, 32, 0);
    }

    fn sendSurfaceAttach(self: *WaylandClient, surf_id: u32, buf_id: u32, x: i32, y: i32) !void {
        var buf: [20]u8 = undefined;
        std.mem.writeInt(u32, buf[0..4], surf_id, .little);
        std.mem.writeInt(u32, buf[4..8], (20 << 16) | 1, .little); // attach
        std.mem.writeInt(u32, buf[8..12], buf_id, .little);
        std.mem.writeInt(i32, buf[12..16], x, .little);
        std.mem.writeInt(i32, buf[16..20], y, .little);
        _ = std.posix.system.send(self.socket_fd, &buf, 20, 0);
    }

    fn sendSurfaceDamage(self: *WaylandClient, surf_id: u32, x: i32, y: i32, w: i32, h: i32) !void {
        var buf: [24]u8 = undefined;
        std.mem.writeInt(u32, buf[0..4], surf_id, .little);
        std.mem.writeInt(u32, buf[4..8], (24 << 16) | 2, .little); // damage
        std.mem.writeInt(i32, buf[8..12], x, .little);
        std.mem.writeInt(i32, buf[12..16], y, .little);
        std.mem.writeInt(i32, buf[16..20], w, .little);
        std.mem.writeInt(i32, buf[20..24], h, .little);
        _ = std.posix.system.send(self.socket_fd, &buf, 24, 0);
    }

    fn sendSurfaceCommit(self: *WaylandClient, surf_id: u32) !void {
        var buf: [8]u8 = undefined;
        std.mem.writeInt(u32, buf[0..4], surf_id, .little);
        std.mem.writeInt(u32, buf[4..8], (8 << 16) | 6, .little); // commit
        _ = std.posix.system.send(self.socket_fd, &buf, 8, 0);
    }

    fn sendXdgWmBasePong(self: *WaylandClient, serial: u32) !void {
        var buf: [12]u8 = undefined;
        std.mem.writeInt(u32, buf[0..4], self.xdg_wm_base_id, .little);
        std.mem.writeInt(u32, buf[4..8], (12 << 16) | 3, .little); // pong
        std.mem.writeInt(u32, buf[8..12], serial, .little);
        _ = std.posix.system.send(self.socket_fd, &buf, 12, 0);
    }

    fn sendXdgSurfaceAckConfigure(self: *WaylandClient, serial: u32) !void {
        var buf: [12]u8 = undefined;
        std.mem.writeInt(u32, buf[0..4], self.xdg_surface_id, .little);
        std.mem.writeInt(u32, buf[4..8], (12 << 16) | 4, .little); // ack_configure
        std.mem.writeInt(u32, buf[8..12], serial, .little);
        _ = std.posix.system.send(self.socket_fd, &buf, 12, 0);
    }
};
