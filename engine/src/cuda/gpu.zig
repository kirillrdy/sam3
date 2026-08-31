//! The device-side runtime: the pieces of a CUDA program that have no host
//! equivalent. Only device code may import this.
//!
//! Nothing here imports `std`. Touching it makes the compiler emit the
//! `@import("builtin")` target tables as global data, and ptxas rejects those
//! (they use odd bit widths like `.u2`), so device code stays std-free.

/// PTX exposes the launch geometry through special registers rather than
/// through arguments. LLVM's inline asm syntax spells a literal `%` as `%%`.
fn special(comptime name: []const u8) u32 {
    return asm volatile ("mov.u32 %[ret], %%" ++ name ++ ";"
        : [ret] "=r" (-> u32),
    );
}

/// Index of this thread within its block, `threadIdx.x` in CUDA C.
pub fn threadIndex() u32 {
    return special("tid.x");
}

/// Y coordinate of this thread within its block, `threadIdx.y`.
pub fn threadIndexY() u32 {
    return special("tid.y");
}

/// Index of this block within the grid, `blockIdx.x`.
pub fn blockIndex() u32 {
    return special("ctaid.x");
}

/// Y coordinate of this block within the grid, `blockIdx.y`.
pub fn blockIndexY() u32 {
    return special("ctaid.y");
}

/// Z coordinate of this block within the grid, commonly the tensor batch.
pub fn blockIndexZ() u32 {
    return special("ctaid.z");
}

/// Threads per block, `blockDim.x`.
pub fn blockSize() u32 {
    return special("ntid.x");
}

/// Index of this thread across the whole grid.
pub fn globalIndex() u32 {
    return blockIndex() * blockSize() + threadIndex();
}

/// Barrier across the block: no thread passes until all of them arrive, and
/// writes made before it are visible to the block afterwards.
pub fn syncThreads() void {
    asm volatile ("bar.sync 0;" ::: .{ .memory = true });
}

pub inline fn loadGlobalFloat4(ptr: [*]addrspace(.global) const f32) [4]f32 {
    var v0: f32 = undefined;
    var v1: f32 = undefined;
    var v2: f32 = undefined;
    var v3: f32 = undefined;
    asm volatile (
        "ld.global.nc.v4.f32 {%[v0], %[v1], %[v2], %[v3]}, [%[ptr]];"
        : [v0] "=f" (v0),
          [v1] "=f" (v1),
          [v2] "=f" (v2),
          [v3] "=f" (v3),
        : [ptr] "l" (ptr),
    );
    return .{ v0, v1, v2, v3 };
}

pub inline fn storeGlobalFloat4(ptr: [*]addrspace(.global) f32, v: [4]f32) void {
    asm volatile (
        "st.global.v4.f32 [%[ptr]], {%[v0], %[v1], %[v2], %[v3]};"
        :
        : [ptr] "l" (ptr),
          [v0] "f" (v[0]),
          [v1] "f" (v[1]),
          [v2] "f" (v[2]),
          [v3] "f" (v[3]),
        : .{ .memory = true }
    );
}

pub inline fn cpAsync16(shared_ptr: *addrspace(.shared) f32, global_ptr: [*]addrspace(.global) const f32) void {
    asm volatile (
        \\cp.async.ca.shared.global [%[sptr]], [%[gptr]], 16;
        :
        : [sptr] "r" (@as(u32, @truncate(@intFromPtr(shared_ptr)))),
          [gptr] "l" (global_ptr),
        : .{ .memory = true }
    );
}

pub inline fn cpAsyncCommit() void {
    asm volatile ("cp.async.commit_group;" ::: .{ .memory = true });
}

pub inline fn cpAsyncWait() void {
    asm volatile ("cp.async.wait_group 0;" ::: .{ .memory = true });
}

pub inline fn loadSharedFloat4(ptr: *addrspace(.shared) const f32) [4]f32 {
    var v0: f32 = undefined;
    var v1: f32 = undefined;
    var v2: f32 = undefined;
    var v3: f32 = undefined;
    asm volatile (
        "ld.shared.v4.f32 {%[v0], %[v1], %[v2], %[v3]}, [%[sptr]];"
        : [v0] "=f" (v0),
          [v1] "=f" (v1),
          [v2] "=f" (v2),
          [v3] "=f" (v3),
        : [sptr] "r" (@as(u32, @truncate(@intFromPtr(ptr)))),
    );
    return .{ v0, v1, v2, v3 };
}

pub inline fn mmaSyncM16N8K8(
    a: [4]f32,
    b: [2]f32,
    c: [4]f32,
) [4]f32 {
    var d0: f32 = undefined;
    var d1: f32 = undefined;
    var d2: f32 = undefined;
    var d3: f32 = undefined;
    asm volatile (
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 " ++
            "{%[d0], %[d1], %[d2], %[d3]}, " ++
            "{%[a0], %[a1], %[a2], %[a3]}, " ++
            "{%[b0], %[b1]}, " ++
            "{%[c0], %[c1], %[c2], %[c3]};"
        : [d0] "=f" (d0),
          [d1] "=f" (d1),
          [d2] "=f" (d2),
          [d3] "=f" (d3),
        : [a0] "r" (@as(u32, @bitCast(a[0]))),
          [a1] "r" (@as(u32, @bitCast(a[1]))),
          [a2] "r" (@as(u32, @bitCast(a[2]))),
          [a3] "r" (@as(u32, @bitCast(a[3]))),
          [b0] "r" (@as(u32, @bitCast(b[0]))),
          [b1] "r" (@as(u32, @bitCast(b[1]))),
          [c0] "f" (c[0]),
          [c1] "f" (c[1]),
          [c2] "f" (c[2]),
          [c3] "f" (c[3]),
    );
    return .{ d0, d1, d2, d3 };
}

/// A GPU thread has nowhere to print a panic message, so a failed safety check
/// becomes a `trap;`, which faults the launch and surfaces on the host at the
/// next synchronize. Same shape as `std.debug.no_panic`, spelled out to keep
/// `std` out of device code.
pub const panic = struct {
    pub fn call(_: []const u8, _: ?usize) noreturn {
        @trap();
    }
    pub fn sentinelMismatch(_: anytype, _: anytype) noreturn {
        @trap();
    }
    pub fn unwrapError(_: anyerror) noreturn {
        @trap();
    }
    pub fn outOfBounds(_: usize, _: usize) noreturn {
        @trap();
    }
    pub fn startGreaterThanEnd(_: usize, _: usize) noreturn {
        @trap();
    }
    pub fn inactiveUnionField(_: anytype, _: anytype) noreturn {
        @trap();
    }
    pub fn sliceCastLenRemainder(_: usize) noreturn {
        @trap();
    }
    pub fn reachedUnreachable() noreturn {
        @trap();
    }
    pub fn unwrapNull() noreturn {
        @trap();
    }
    pub fn castToNull() noreturn {
        @trap();
    }
    pub fn incorrectAlignment() noreturn {
        @trap();
    }
    pub fn invalidErrorCode() noreturn {
        @trap();
    }
    pub fn integerOutOfBounds() noreturn {
        @trap();
    }
    pub fn integerOverflow() noreturn {
        @trap();
    }
    pub fn shlOverflow() noreturn {
        @trap();
    }
    pub fn shrOverflow() noreturn {
        @trap();
    }
    pub fn divideByZero() noreturn {
        @trap();
    }
    pub fn exactDivisionRemainder() noreturn {
        @trap();
    }
    pub fn integerPartOutOfBounds() noreturn {
        @trap();
    }
    pub fn corruptSwitch() noreturn {
        @trap();
    }
    pub fn shiftRhsTooBig() noreturn {
        @trap();
    }
    pub fn invalidEnumValue() noreturn {
        @trap();
    }
    pub fn forLenMismatch() noreturn {
        @trap();
    }
    pub fn copyLenMismatch() noreturn {
        @trap();
    }
    pub fn memcpyAlias() noreturn {
        @trap();
    }
    pub fn noreturnReturned() noreturn {
        @trap();
    }
};
