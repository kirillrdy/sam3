//! Device code. This file is compiled on its own for `nvptx64-cuda`, and the
//! PTX it produces is embedded in the host binary and JIT-compiled at load.
//!
//! Every function here runs on the GPU. Pointer parameters are
//! `addrspace(.global)`, the address space `cuda.Buffer` allocations live in,
//! and a kernel is any function with `callconv(.kernel)`.

const gpu = @import("gpu");

pub const panic = gpu.panic;

pub fn vecAdd(
    a: [*]addrspace(.global) const f32,
    b: [*]addrspace(.global) const f32,
    out: [*]addrspace(.global) f32,
    n: u32,
) callconv(.kernel) void {
    const i = gpu.globalIndex();
    if (i >= n) return;
    out[i] = a[i] + b[i];
}

pub fn scale(
    values: [*]addrspace(.global) f32,
    factor: f32,
    n: u32,
) callconv(.kernel) void {
    const i = gpu.globalIndex();
    if (i >= n) return;
    values[i] *= factor;
}

/// Zig only emits a function that something references, and marking a kernel
/// `export` turns it into an LLVM alias, which the NVPTX backend refuses to
/// lower. One exported plain function taking their addresses keeps them alive,
/// so every kernel above has to appear here.
export fn anchor() usize {
    return @intFromPtr(&vecAdd) ^ @intFromPtr(&scale);
}
