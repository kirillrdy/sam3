const std = @import("std");

/// Where the NVIDIA driver's libcuda tends to live. It ships with the driver,
/// not with the CUDA toolkit, so there is nothing else to install.
const cuda_library_dirs = [_][]const u8{
    "/run/opengl-driver/lib",
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib64",
    "/usr/lib",
};

/// Compute capability of an Ada RTX card.
pub const default_arch = "sm_89";

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const arch = b.option(
        []const u8,
        "sm",
        "Compute capability the kernels are compiled for (default: " ++ default_arch ++ ", an Ada RTX card)",
    ) orelse default_arch;

    const cuda_library_path = b.option(
        []const u8,
        "cuda-library-path",
        "Directory holding the driver's libcuda.so, for when this does not find it itself",
    ) orelse findLibrary(b) orelse {
        std.log.err("no libcuda.so found; pass -Dcuda-library-path=DIR", .{});
        std.process.exit(1);
    };

    // Host bindings, published so other packages can drive the GPU with them.
    const mod = b.addModule("cuda", .{
        .root_source_file = b.path("src/cuda.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    link(mod, cuda_library_path);

    const ptx = addPtx(b, b.path("src/gpu.zig"), .{
        .root_source_file = b.path("src/kernels.zig"),
        .arch = arch,
        .optimize = optimize,
    });

    const ptx_step = b.step("ptx", "Write the compiled kernels to zig-out/kernels.ptx");
    ptx_step.dependOn(&b.addInstallFile(ptx, "kernels.ptx").step);

    const demo = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "cuda", .module = mod }},
    });
    demo.addAnonymousImport("kernels.ptx", .{ .root_source_file = ptx });

    const exe = b.addExecutable(.{ .name = "cuda-demo", .root_module = demo });
    b.installArtifact(exe);

    const run = b.addRunArtifact(exe);
    run.step.dependOn(b.getInstallStep());
    if (b.args) |args| run.addArgs(args);
    b.step("run", "Run the demo on the GPU").dependOn(&run.step);
}

pub const PtxOptions = struct {
    /// Device code. It may `@import("gpu")` for the device-side runtime.
    root_source_file: std.Build.LazyPath,
    arch: []const u8 = default_arch,
    optimize: std.builtin.OptimizeMode = .ReleaseSafe,
};

/// Compiles device code to PTX for a package that depends on this one.
pub fn addPtxFor(b: *std.Build, dep: *std.Build.Dependency, options: PtxOptions) std.Build.LazyPath {
    return addPtx(b, dep.path("src/gpu.zig"), options);
}

/// Points a module at the driver library. Every compile that imports the
/// `cuda` module needs this, and modules pass it on to whatever imports them.
pub fn link(mod: *std.Build.Module, cuda_library_path: []const u8) void {
    mod.addLibraryPath(.{ .cwd_relative = cuda_library_path });
    mod.addRPath(.{ .cwd_relative = cuda_library_path });
    mod.linkSystemLibrary("cuda", .{});
}

/// Directory holding libcuda.so, or null on a machine without the driver.
pub fn findLibrary(b: *std.Build) ?[]const u8 {
    for (cuda_library_dirs) |dir| {
        const path = b.pathJoin(&.{ dir, "libcuda.so" });
        std.Io.Dir.accessAbsolute(b.graph.io, path, .{}) catch continue;
        return dir;
    }
    return null;
}

fn addPtx(b: *std.Build, gpu_source: std.Build.LazyPath, options: PtxOptions) std.Build.LazyPath {
    // ptxas rejects the `@import("builtin")` target tables that a Debug build
    // leaves in the module -- they use bit widths like `.u2` that PTX has no
    // syntax for -- so device code is built at ReleaseSafe or better. Safety
    // checks survive that; they just trap the launch instead of printing.
    const optimize = if (options.optimize == .Debug) .ReleaseSafe else options.optimize;

    // Device code is a separate compilation: a different target, no libc, no
    // std. The NVPTX backend has no object writer, so the artifact is the
    // assembly it prints -- which for NVPTX is PTX, what the driver JITs.
    const cmd = b.addSystemCommand(&.{ b.graph.zig_exe, "build-obj" });
    cmd.addArgs(&.{
        "-target",                           "nvptx64-cuda",
        b.fmt("-mcpu={s}", .{options.arch}), "-O",
        @tagName(optimize),                  "-fno-emit-bin",
        "-fstrip",                           "--dep",
        "gpu",
    });
    cmd.addPrefixedFileArg("-Mroot=", options.root_source_file);
    cmd.addPrefixedFileArg("-Mgpu=", gpu_source);
    if (b.cache_root.path) |path| cmd.addArgs(&.{ "--cache-dir", path });
    if (b.graph.global_cache_root.path) |path| cmd.addArgs(&.{ "--global-cache-dir", path });
    return cmd.addPrefixedOutputFileArg("-femit-asm=", "kernels.ptx");
}
