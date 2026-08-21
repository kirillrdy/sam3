//! An allocator that recycles large blocks instead of returning them to the OS.
//!
//! A forward pass allocates and frees the same handful of shapes over and over:
//! the MLP hidden state alone is 5184x4736 floats, 94 MiB, allocated and dropped
//! once per layer. Blocks that size bypass any general-purpose allocator's bins
//! and go straight to `mmap`, so each layer hands its pages back to the kernel
//! and faults an identical set in again a moment later - work that shows up as
//! system time rather than anything the model asked for.
//!
//! Recycling them turns those hundreds of map/unmap pairs into a handful, at the
//! cost of holding the high-water mark resident. Only blocks at or above
//! `min_block` are pooled; everything smaller is the backing allocator's job,
//! where it is already cheap.

const std = @import("std");
const Alignment = std.mem.Alignment;

/// Blocks below this are passed straight through: the backing allocator bins
/// them without a syscall, so pooling would only add bookkeeping.
const min_block = 1 << 20;

/// Ceiling on retained bytes. The pass reuses a small set of shapes, so the
/// working set settles well under this; the cap is here so an unusual sequence
/// of sizes cannot grow the pool without bound.
const max_retained = 1 << 31;

/// One recycled block. Size and alignment both have to match a request for it
/// to be reused - handing back a larger block would misreport its length to a
/// later `free`.
const Block = struct {
    ptr: [*]u8,
    len: usize,
    alignment: Alignment,
};

pub const Pool = struct {
    backing: std.mem.Allocator,
    lock: std.atomic.Mutex = .unlocked,
    free_list: std.ArrayList(Block) = .empty,
    retained: usize = 0,

    pub fn init(backing: std.mem.Allocator) Pool {
        return .{ .backing = backing };
    }

    pub fn deinit(self: *Pool) void {
        for (self.free_list.items) |blk| {
            self.backing.rawFree(blk.ptr[0..blk.len], blk.alignment, @returnAddress());
        }
        self.free_list.deinit(self.backing);
        self.retained = 0;
    }

    pub fn allocator(self: *Pool) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .remap = remap,
                .free = free,
            },
        };
    }

    /// Spinning here would be worse than missing the pool: a contended pool just
    /// means another thread is mid-allocation, and the backing allocator can
    /// serve this request concurrently.
    fn tryLock(self: *Pool) bool {
        return self.lock.tryLock();
    }

    fn alloc(ctx: *anyopaque, len: usize, alignment: Alignment, ret_addr: usize) ?[*]u8 {
        const self: *Pool = @ptrCast(@alignCast(ctx));

        if (len >= min_block and self.tryLock()) {
            defer self.lock.unlock();
            for (self.free_list.items, 0..) |blk, i| {
                if (blk.len == len and blk.alignment == alignment) {
                    _ = self.free_list.swapRemove(i);
                    self.retained -= blk.len;
                    return blk.ptr;
                }
            }
        }

        return self.backing.rawAlloc(len, alignment, ret_addr);
    }

    fn free(ctx: *anyopaque, memory: []u8, alignment: Alignment, ret_addr: usize) void {
        const self: *Pool = @ptrCast(@alignCast(ctx));

        if (memory.len >= min_block and
            self.retained + memory.len <= max_retained and
            self.tryLock())
        {
            defer self.lock.unlock();
            // Appending is the only step that can fail, and failing it just
            // means this block is released normally rather than kept.
            if (self.free_list.append(self.backing, .{
                .ptr = memory.ptr,
                .len = memory.len,
                .alignment = alignment,
            })) |_| {
                self.retained += memory.len;
                return;
            } else |_| {}
        }

        self.backing.rawFree(memory, alignment, ret_addr);
    }

    fn resize(ctx: *anyopaque, memory: []u8, alignment: Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *Pool = @ptrCast(@alignCast(ctx));
        return self.backing.rawResize(memory, alignment, new_len, ret_addr);
    }

    fn remap(ctx: *anyopaque, memory: []u8, alignment: Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const self: *Pool = @ptrCast(@alignCast(ctx));
        return self.backing.rawRemap(memory, alignment, new_len, ret_addr);
    }
};

test "pool hands back the same block for a repeated size" {
    var pool = Pool.init(std.testing.allocator);
    defer pool.deinit();
    const a = pool.allocator();

    const first = try a.alloc(u8, min_block);
    const addr = first.ptr;
    a.free(first);

    const second = try a.alloc(u8, min_block);
    defer a.free(second);
    try std.testing.expectEqual(addr, second.ptr);
}

test "pool passes small allocations straight through" {
    var pool = Pool.init(std.testing.allocator);
    defer pool.deinit();
    const a = pool.allocator();

    const small = try a.alloc(u8, 64);
    a.free(small);
    // Nothing retained: a leak-checking backing allocator would flag the block
    // if the pool had kept it without accounting.
    try std.testing.expectEqual(@as(usize, 0), pool.retained);
}

test "pool survives interleaved sizes and frees everything on deinit" {
    var pool = Pool.init(std.testing.allocator);
    defer pool.deinit();
    const a = pool.allocator();

    const x = try a.alloc(u8, min_block);
    const y = try a.alloc(u8, min_block * 2);
    a.free(x);
    a.free(y);

    const z = try a.alloc(u8, min_block * 2);
    try std.testing.expectEqual(y.ptr, z.ptr);
    a.free(z);
}
