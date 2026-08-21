//! A persistent worker pool behind a `parallelFor` that hands out work
//! dynamically.
//!
//! Two properties matter for the vision encoder, which dispatches tens of
//! thousands of parallel regions per forward pass:
//!
//!  * Threads are spawned once and parked between regions. Spawning a thread
//!    per region costs more than most of the regions themselves.
//!  * Chunks are claimed from a shared atomic cursor rather than handed out in
//!    equal slices up front. On a hybrid CPU the performance and efficiency
//!    cores retire a fixed slice at very different rates, so a static split
//!    leaves the fast cores idling at every join.
//!
//! Handoff is a generation counter rather than a condition variable: regions
//! are short and back to back, so the wakeup latency of a futex sleep would
//! show up directly in the forward pass. Waiters spin briefly, then yield, then
//! sleep, which keeps idle threads off the CPU during the serial stretches.

const std = @import("std");

const WorkerFn = *const fn (ctx: *anyopaque, start: usize, end: usize) void;

/// Set on every thread inside a parallel region, including the dispatching one.
/// Nested regions then run inline rather than waiting on a pool that is already
/// fully committed.
threadlocal var in_region: bool = false;

/// Stable index of the calling thread within the pool: 0 for the dispatcher,
/// 1..n for the workers. Lets a parallel region carve per-thread scratch out of
/// one allocation instead of allocating per task.
threadlocal var lane: usize = 0;

pub fn laneId() usize {
    return lane;
}

/// Upper bound on `laneId()`, i.e. how many scratch slots a region must reserve.
pub fn laneCount() usize {
    ensureStarted();
    return pool.threads.len + 1;
}

const Pool = struct {
    threads: []std.Thread = &.{},

    /// Bumped once per region. A worker compares it against the last generation
    /// it ran to distinguish a new region from a spurious wakeup.
    generation: std.atomic.Value(u64) = .init(0),
    /// Workers that have not yet finished the current region.
    pending: std.atomic.Value(usize) = .init(0),
    cursor: std.atomic.Value(usize) = .init(0),

    /// Written before the release store to `generation` that publishes them,
    /// and read after the matching acquire load.
    job_fn: WorkerFn = undefined,
    job_ctx: *anyopaque = undefined,
    job_total: usize = 0,
    job_chunk: usize = 1,
};

var pool: Pool = .{};
var start_lock: std.atomic.Value(u32) = .init(0);
var started: std.atomic.Value(bool) = .init(false);

pub fn getWorkerCount() usize {
    const count = std.Thread.getCpuCount() catch 4;
    return @min(32, @max(1, count));
}

/// Spin, then yield, then sleep. The early iterations cover the common case of
/// a region that is about to start; the later ones stop a parked worker from
/// stealing cycles from the serial code between regions.
fn backoff(spins: usize) void {
    if (spins < 256) {
        std.atomic.spinLoopHint();
    } else if (spins < 4096) {
        std.Thread.yield() catch std.atomic.spinLoopHint();
    } else if (@import("builtin").os.tag == .linux) {
        const ts = std.os.linux.timespec{ .sec = 0, .nsec = 50_000 };
        _ = std.os.linux.nanosleep(&ts, null);
    } else {
        std.Thread.yield() catch std.atomic.spinLoopHint();
    }
}

fn ensureStarted() void {
    if (started.load(.acquire)) return;

    while (start_lock.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) {
        std.atomic.spinLoopHint();
    }
    defer start_lock.store(0, .release);

    if (started.load(.monotonic)) return;

    const workers = getWorkerCount() - 1;
    if (workers > 0) {
        // The pool lives for the whole process, so it is deliberately not owned
        // by a caller's allocator - a test allocator would flag it as a leak.
        if (std.heap.page_allocator.alloc(std.Thread, workers)) |threads| {
            var spawned: usize = 0;
            for (0..workers) |i| {
                threads[i] = std.Thread.spawn(.{}, workerMain, .{i + 1}) catch break;
                spawned += 1;
            }
            pool.threads = threads[0..spawned];
        } else |_| {}
    }

    started.store(true, .release);
}

fn drainChunks() void {
    const total = pool.job_total;
    const chunk = pool.job_chunk;
    while (true) {
        const start = pool.cursor.fetchAdd(chunk, .acq_rel);
        if (start >= total) return;
        pool.job_fn(pool.job_ctx, start, @min(total, start + chunk));
    }
}

fn workerMain(lane_index: usize) void {
    lane = lane_index;
    in_region = true;
    var last_gen: u64 = 0;

    while (true) {
        var spins: usize = 0;
        while (pool.generation.load(.acquire) == last_gen) : (spins += 1) {
            backoff(spins);
        }
        last_gen = pool.generation.load(.acquire);

        drainChunks();
        _ = pool.pending.fetchSub(1, .release);
    }
}

fn dispatch(f: WorkerFn, ctx: *anyopaque, total: usize) void {
    ensureStarted();

    if (in_region or pool.threads.len == 0 or total <= 1) {
        f(ctx, 0, total);
        return;
    }

    const lanes = pool.threads.len + 1;
    // Several chunks per lane so one slow core cannot hold up the join, but not
    // so many that the cursor itself becomes the contended resource.
    const chunk = @max(1, total / (lanes * 8));

    pool.job_fn = f;
    pool.job_ctx = ctx;
    pool.job_total = total;
    pool.job_chunk = chunk;
    pool.cursor.store(0, .monotonic);
    pool.pending.store(pool.threads.len, .monotonic);
    // Publishes every field above to the workers.
    _ = pool.generation.fetchAdd(1, .release);

    in_region = true;
    drainChunks();
    in_region = false;

    var spins: usize = 0;
    while (pool.pending.load(.acquire) != 0) : (spins += 1) {
        backoff(spins);
    }
}

pub fn parallelFor(
    allocator: std.mem.Allocator,
    total_items: usize,
    context: anytype,
    comptime worker_fn: fn (@TypeOf(context), usize, usize) void,
) void {
    _ = allocator;
    if (total_items == 0) return;

    const Ctx = @TypeOf(context);
    const Trampoline = struct {
        fn call(ptr: *anyopaque, start: usize, end: usize) void {
            const typed: *const Ctx = @ptrCast(@alignCast(ptr));
            worker_fn(typed.*, start, end);
        }
    };

    var ctx_copy = context;
    dispatch(Trampoline.call, @ptrCast(&ctx_copy), total_items);
}

test "parallelFor covers every index exactly once" {
    const n = 10_000;
    const counts = try std.testing.allocator.alloc(u32, n);
    defer std.testing.allocator.free(counts);
    @memset(counts, 0);

    const Ctx = struct { counts: []u32 };
    parallelFor(std.testing.allocator, n, Ctx{ .counts = counts }, struct {
        fn worker(c: Ctx, start: usize, end: usize) void {
            for (start..end) |i| c.counts[i] += 1;
        }
    }.worker);

    for (counts) |c| try std.testing.expectEqual(@as(u32, 1), c);
}

test "nested parallelFor runs inline instead of deadlocking" {
    const Inner = struct { hits: *std.atomic.Value(usize) };
    var hits = std.atomic.Value(usize).init(0);

    parallelFor(std.testing.allocator, 16, Inner{ .hits = &hits }, struct {
        fn outer(c: Inner, start: usize, end: usize) void {
            for (start..end) |_| {
                parallelFor(std.testing.allocator, 4, c, struct {
                    fn inner(ic: Inner, s: usize, e: usize) void {
                        _ = ic.hits.fetchAdd(e - s, .monotonic);
                    }
                }.inner);
            }
        }
    }.outer);

    try std.testing.expectEqual(@as(usize, 64), hits.load(.monotonic));
}
