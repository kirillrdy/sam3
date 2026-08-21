//! Cache-blocked, register-tiled single-precision GEMM.
//!
//! Every projection in the backbone is `C[M,N] = A[M,K] . B[K,N] + bias`, and
//! at 1008x1008 the MLP alone is M=5184, K=1024, N=4736 - a 19 MiB weight
//! matrix. Computing it one output element at a time (a dot product per
//! element) re-reads that whole matrix for every token, which is roughly a
//! hundred gigabytes of traffic per layer and leaves the FMA units idle.
//!
//! The structure here is the standard one: pack both operands into panels laid
//! out exactly the way the micro-kernel walks them, then block the M dimension
//! so a packed A block stays resident in L2 while the whole of B streams past
//! it once. The micro-kernel keeps a `MR x NR` tile of C in vector registers
//! for the entire K loop, so each pass over the packed panels does `MR * NR`
//! multiply-adds per `MR + NR/VEC` loads.
//!
//! The blocking does not reorder anything: each output element is the same sum
//! of products in the same order as the naive loop. The micro-kernel does relax
//! the float mode, though, so the multiply-add contracts to a single FMA and
//! keeps the full-width product instead of rounding it first. That is one
//! rounding step fewer per term - closer to the exact result than the naive
//! loop, not further from it, but not bit-identical to it either.

const std = @import("std");
const parallel = @import("parallel.zig");

pub const VEC = 8;
pub const Vec = @Vector(VEC, f32);

/// Rows of C per register tile.
const MR = 6;
/// Vectors of C per register tile row. `MR * N_LANES` accumulators plus the
/// operands have to fit the sixteen vector registers AVX2 offers.
const N_LANES = 2;
/// Columns of C per register tile.
const NR = N_LANES * VEC;
/// Rows of A per cache block: the unit of work handed to a thread, and the
/// window of A that stays in L2 while the whole of B streams past it once.
///
/// Larger blocks cut how many times B is re-read, but this is a hybrid CPU and
/// the efficiency cores retire a block at roughly half the rate of the
/// performance cores, so what actually decides the wall clock is having enough
/// blocks left at the end for a slow core's last one not to hold up the join.
/// Measured at 1008x1008, 48 beats 96 and 192 despite the extra B traffic.
const MB_ROWS = 48;
const MB_PANELS = MB_ROWS / MR;

/// Below this the packing costs more than the blocking saves.
const PACK_THRESHOLD = 1 << 18;

/// `B` as the caller holds it. `.row_major` is `[K, N]`; `.transposed` is
/// `[N, K]`, which is how PyTorch stores `nn.Linear.weight`.
pub const BLayout = enum { row_major, transposed };

fn packA(a: []const f32, m: usize, k: usize, lda: usize, dst: []f32, panels: usize) void {
    const Ctx = struct {
        a: []const f32,
        dst: []f32,
        m: usize,
        k: usize,
        lda: usize,
    };
    const ctx = Ctx{ .a = a, .dst = dst, .m = m, .k = k, .lda = lda };

    parallel.parallelFor(std.heap.page_allocator, panels, ctx, struct {
        fn worker(c: Ctx, start: usize, end: usize) void {
            for (start..end) |p| {
                const m0 = p * MR;
                const panel = c.dst[p * MR * c.k ..][0 .. MR * c.k];
                const rows = @min(MR, c.m -| m0);

                for (0..rows) |i| {
                    const src = c.a[(m0 + i) * c.lda ..][0..c.k];
                    for (0..c.k) |kk| panel[kk * MR + i] = src[kk];
                }
                for (rows..MR) |i| {
                    for (0..c.k) |kk| panel[kk * MR + i] = 0.0;
                }
            }
        }
    }.worker);
}

fn packB(
    b: []const f32,
    n: usize,
    k: usize,
    ldb: usize,
    comptime layout: BLayout,
    dst: []f32,
    panels: usize,
) void {
    const Ctx = struct {
        b: []const f32,
        dst: []f32,
        n: usize,
        k: usize,
        ldb: usize,
    };
    const ctx = Ctx{ .b = b, .dst = dst, .n = n, .k = k, .ldb = ldb };

    parallel.parallelFor(std.heap.page_allocator, panels, ctx, struct {
        fn worker(c: Ctx, start: usize, end: usize) void {
            for (start..end) |q| {
                const n0 = q * NR;
                const panel = c.dst[q * NR * c.k ..][0 .. NR * c.k];
                const cols = @min(NR, c.n -| n0);

                switch (layout) {
                    .transposed => {
                        for (0..cols) |j| {
                            const src = c.b[(n0 + j) * c.ldb ..][0..c.k];
                            for (0..c.k) |kk| panel[kk * NR + j] = src[kk];
                        }
                        for (0..c.k) |kk| {
                            for (cols..NR) |j| panel[kk * NR + j] = 0.0;
                        }
                    },
                    .row_major => {
                        for (0..c.k) |kk| {
                            const src = c.b[kk * c.ldb + n0 ..];
                            for (0..cols) |j| panel[kk * NR + j] = src[j];
                            for (cols..NR) |j| panel[kk * NR + j] = 0.0;
                        }
                    },
                }
            }
        }
    }.worker);
}

/// Accumulates one `MR x NR` tile of C across the whole of K, holding the tile
/// in registers throughout.
inline fn microKernel(
    ap: []const f32,
    bp: []const f32,
    k: usize,
    bias: [N_LANES]Vec,
    out: *[MR][NR]f32,
) void {
    @setFloatMode(.optimized);
    var acc: [MR][N_LANES]Vec = undefined;
    inline for (0..MR) |i| acc[i] = bias;

    var kk: usize = 0;
    while (kk < k) : (kk += 1) {
        var bv: [N_LANES]Vec = undefined;
        inline for (0..N_LANES) |j| bv[j] = bp[kk * NR + j * VEC ..][0..VEC].*;

        const arow = ap[kk * MR ..][0..MR];
        inline for (0..MR) |i| {
            const av: Vec = @splat(arow[i]);
            inline for (0..N_LANES) |j| acc[i][j] += av * bv[j];
        }
    }

    inline for (0..MR) |i| {
        inline for (0..N_LANES) |j| {
            out[i][j * VEC ..][0..VEC].* = acc[i][j];
        }
    }
}

/// The packed-panel buffers, kept alive between calls.
///
/// A forward pass issues a couple of hundred large GEMMs whose panels are tens
/// of megabytes; allocating them each time returns the pages to the OS on every
/// call and pays the fault back in on the next one. Holding one pair and growing
/// it to the high-water mark turns that into a handful of allocations for the
/// whole run. The buffers are process-lifetime on purpose - they are handed back
/// to the OS when it exits, not owned by a caller's allocator.
const Scratch = struct {
    a: []f32,
    b: []f32,
    /// Whether the cache was free to lend these out, or they came from the
    /// caller's allocator because a GEMM further up the stack already holds it.
    cached: bool,

    var lock: std.atomic.Mutex = .unlocked;
    var cache_a: []f32 = &.{};
    var cache_b: []f32 = &.{};

    fn acquire(a_len: usize, b_len: usize, allocator: std.mem.Allocator) !Scratch {
        if (lock.tryLock()) {
            if (grow(&cache_a, a_len) and grow(&cache_b, b_len)) {
                return .{ .a = cache_a[0..a_len], .b = cache_b[0..b_len], .cached = true };
            }
            lock.unlock();
        }
        const a = try allocator.alloc(f32, a_len);
        errdefer allocator.free(a);
        return .{ .a = a, .b = try allocator.alloc(f32, b_len), .cached = false };
    }

    fn grow(buf: *[]f32, len: usize) bool {
        if (buf.len >= len) return true;
        const next = std.heap.page_allocator.alloc(f32, len) catch return false;
        std.heap.page_allocator.free(buf.*);
        buf.* = next;
        return true;
    }

    fn release(self: Scratch, allocator: std.mem.Allocator) void {
        if (self.cached) {
            lock.unlock();
        } else {
            allocator.free(self.a);
            allocator.free(self.b);
        }
    }
};

/// `C[M,N] = A[M,K] . B + bias`, with `B` interpreted per `layout`.
pub fn gemm(
    allocator: std.mem.Allocator,
    m: usize,
    n: usize,
    k: usize,
    a: []const f32,
    lda: usize,
    b: []const f32,
    ldb: usize,
    comptime layout: BLayout,
    bias: ?[]const f32,
    c: []f32,
) !void {
    if (m == 0 or n == 0) return;

    if (m * n * k < PACK_THRESHOLD) {
        gemmSmall(m, n, k, a, lda, b, ldb, layout, bias, c);
        return;
    }

    const m_panels = (m + MR - 1) / MR;
    const n_panels = (n + NR - 1) / NR;

    // The panel buffers are the same handful of sizes every layer and run to
    // tens of megabytes each, so allocating them per call means the pass spends
    // most of its kernel time faulting in fresh pages and handing them straight
    // back. `scratch` keeps one pair alive across calls; the allocator is the
    // fallback for the reentrant case, where the cache is already in use.
    const scratch = try Scratch.acquire(m_panels * MR * k, n_panels * NR * k, allocator);
    defer scratch.release(allocator);
    const a_pack = scratch.a;
    const b_pack = scratch.b;

    packA(a, m, k, lda, a_pack, m_panels);
    packB(b, n, k, ldb, layout, b_pack, n_panels);

    const Ctx = struct {
        a_pack: []const f32,
        b_pack: []const f32,
        bias: ?[]const f32,
        c: []f32,
        m: usize,
        n: usize,
        k: usize,
        m_panels: usize,
        n_panels: usize,
    };
    const ctx = Ctx{
        .a_pack = a_pack,
        .b_pack = b_pack,
        .bias = bias,
        .c = c,
        .m = m,
        .n = n,
        .k = k,
        .m_panels = m_panels,
        .n_panels = n_panels,
    };

    const blocks = (m_panels + MB_PANELS - 1) / MB_PANELS;

    parallel.parallelFor(allocator, blocks, ctx, struct {
        fn worker(cx: Ctx, start: usize, end: usize) void {
            var tile: [MR][NR]f32 = undefined;

            for (start..end) |block| {
                const p0 = block * MB_PANELS;
                const p1 = @min(cx.m_panels, p0 + MB_PANELS);

                // B panel outermost: it stays in L1 while the packed A block,
                // resident in L2, is swept underneath it.
                for (0..cx.n_panels) |q| {
                    const n0 = q * NR;
                    const cols = @min(NR, cx.n - n0);
                    const bp = cx.b_pack[q * NR * cx.k ..][0 .. NR * cx.k];

                    var bias_tile: [N_LANES]Vec = @splat(@as(Vec, @splat(0.0)));
                    if (cx.bias) |bs| {
                        var tmp: [NR]f32 = @splat(0.0);
                        @memcpy(tmp[0..cols], bs[n0..][0..cols]);
                        inline for (0..N_LANES) |j| bias_tile[j] = tmp[j * VEC ..][0..VEC].*;
                    }

                    for (p0..p1) |p| {
                        const m0 = p * MR;
                        const rows = @min(MR, cx.m - m0);
                        const ap = cx.a_pack[p * MR * cx.k ..][0 .. MR * cx.k];

                        microKernel(ap, bp, cx.k, bias_tile, &tile);

                        for (0..rows) |i| {
                            const dst = cx.c[(m0 + i) * cx.n + n0 ..][0..cols];
                            @memcpy(dst, tile[i][0..cols]);
                        }
                    }
                }
            }
        }
    }.worker);
}

/// Unblocked fallback for matrices too small to be worth packing.
fn gemmSmall(
    m: usize,
    n: usize,
    k: usize,
    a: []const f32,
    lda: usize,
    b: []const f32,
    ldb: usize,
    comptime layout: BLayout,
    bias: ?[]const f32,
    c: []f32,
) void {
    for (0..m) |i| {
        const a_row = a[i * lda ..][0..k];
        const c_row = c[i * n ..][0..n];

        switch (layout) {
            .transposed => {
                for (0..n) |j| {
                    var acc: Vec = @splat(0.0);
                    const b_row = b[j * ldb ..][0..k];
                    var kk: usize = 0;
                    while (kk + VEC <= k) : (kk += VEC) {
                        const av: Vec = a_row[kk..][0..VEC].*;
                        const bv: Vec = b_row[kk..][0..VEC].*;
                        acc += av * bv;
                    }
                    var sum = @reduce(.Add, acc);
                    while (kk < k) : (kk += 1) sum += a_row[kk] * b_row[kk];
                    c_row[j] = sum + if (bias) |bs| bs[j] else 0.0;
                }
            },
            .row_major => {
                if (bias) |bs| {
                    @memcpy(c_row, bs[0..n]);
                } else {
                    @memset(c_row, 0.0);
                }
                for (0..k) |kk| {
                    const scale = a_row[kk];
                    if (scale == 0.0) continue;
                    const sv: Vec = @splat(scale);
                    const b_row = b[kk * ldb ..][0..n];
                    var j: usize = 0;
                    while (j + VEC <= n) : (j += VEC) {
                        const bv: Vec = b_row[j..][0..VEC].*;
                        const cv: Vec = c_row[j..][0..VEC].*;
                        c_row[j..][0..VEC].* = cv + sv * bv;
                    }
                    while (j < n) : (j += 1) c_row[j] += scale * b_row[j];
                }
            },
        }
    }
}

fn referenceGemm(
    m: usize,
    n: usize,
    k: usize,
    a: []const f32,
    b: []const f32,
    comptime layout: BLayout,
    bias: ?[]const f32,
    c: []f32,
) void {
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = if (bias) |bs| bs[j] else 0.0;
            for (0..k) |kk| {
                const bv = switch (layout) {
                    .transposed => b[j * k + kk],
                    .row_major => b[kk * n + j],
                };
                sum += a[i * k + kk] * bv;
            }
            c[i * n + j] = sum;
        }
    }
}

test "blocked gemm matches the naive triple loop" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const rand = prng.random();

    // Sizes that straddle MR / NR / MB so every edge path is exercised.
    const cases = [_][3]usize{
        .{ 1, 1, 1 },
        .{ 7, 19, 13 },
        .{ 6, 16, 32 },
        .{ 200, 48, 64 },
        .{ 199, 47, 65 },
        .{ 260, 80, 96 },
    };

    inline for (.{ BLayout.transposed, BLayout.row_major }) |layout| {
        for (cases) |case| {
            const m, const n, const k = case;

            const a = try allocator.alloc(f32, m * k);
            defer allocator.free(a);
            const b = try allocator.alloc(f32, n * k);
            defer allocator.free(b);
            const bias = try allocator.alloc(f32, n);
            defer allocator.free(bias);
            const got = try allocator.alloc(f32, m * n);
            defer allocator.free(got);
            const want = try allocator.alloc(f32, m * n);
            defer allocator.free(want);

            for (a) |*v| v.* = rand.float(f32) - 0.5;
            for (b) |*v| v.* = rand.float(f32) - 0.5;
            for (bias) |*v| v.* = rand.float(f32) - 0.5;

            const ldb = if (layout == .transposed) k else n;
            try gemm(allocator, m, n, k, a, k, b, ldb, layout, bias, got);
            referenceGemm(m, n, k, a, b, layout, bias, want);

            for (want, got) |e, g| try std.testing.expectApproxEqAbs(e, g, 1e-4);
        }
    }
}

test "blocked gemm honours a null bias" {
    const allocator = std.testing.allocator;
    const m, const n, const k = .{ 64, 32, 48 };

    const a = try allocator.alloc(f32, m * k);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, n * k);
    defer allocator.free(b);
    const got = try allocator.alloc(f32, m * n);
    defer allocator.free(got);
    const want = try allocator.alloc(f32, m * n);
    defer allocator.free(want);

    for (a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 13)) * 0.1;
    for (b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 7)) * 0.2;

    try gemm(allocator, m, n, k, a, k, b, k, .transposed, null, got);
    referenceGemm(m, n, k, a, b, .transposed, null, want);

    for (want, got) |e, g| try std.testing.expectApproxEqAbs(e, g, 1e-4);
}
