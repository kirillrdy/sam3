const std = @import("std");

pub fn getWorkerCount() usize {
    const count = std.Thread.getCpuCount() catch 4;
    return @min(32, @max(1, count));
}

pub fn parallelFor(
    allocator: std.mem.Allocator,
    total_items: usize,
    context: anytype,
    comptime worker_fn: fn (@TypeOf(context), usize, usize) void,
) void {
    if (total_items == 0) return;
    const num_threads = getWorkerCount();
    if (num_threads <= 1 or total_items < num_threads) {
        worker_fn(context, 0, total_items);
        return;
    }

    const Worker = struct {
        ctx: @TypeOf(context),
        start: usize,
        end: usize,

        pub fn run(self: @This()) void {
            worker_fn(self.ctx, self.start, self.end);
        }
    };

    var threads = allocator.alloc(std.Thread, num_threads - 1) catch {
        worker_fn(context, 0, total_items);
        return;
    };
    defer allocator.free(threads);

    const chunk_size = (total_items + num_threads - 1) / num_threads;
    var spawned: usize = 0;

    for (0..num_threads - 1) |t| {
        const start = t * chunk_size;
        if (start >= total_items) break;
        const end = @min(total_items, start + chunk_size);
        const w = Worker{ .ctx = context, .start = start, .end = end };
        threads[t] = std.Thread.spawn(.{}, Worker.run, .{w}) catch break;
        spawned += 1;
    }

    // Run the remaining chunk on the calling thread
    const main_start = spawned * chunk_size;
    if (main_start < total_items) {
        const main_end = @min(total_items, main_start + chunk_size);
        worker_fn(context, main_start, main_end);
    }

    // Join spawned threads
    for (threads[0..spawned]) |t| {
        t.join();
    }
}
