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

/// Blocks in the grid, `gridDim.x`.
pub fn gridSize() u32 {
    return special("nctaid.x");
}

/// Index of this thread across the whole grid.
pub fn globalIndex() u32 {
    return blockIndex() * blockSize() + threadIndex();
}

/// Total number of threads launched, the stride for a grid-stride loop.
pub fn globalSize() u32 {
    return gridSize() * blockSize();
}

/// Barrier across the block: no thread passes until all of them arrive, and
/// writes made before it are visible to the block afterwards.
pub fn syncThreads() void {
    asm volatile ("bar.sync 0;" ::: .{ .memory = true });
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
