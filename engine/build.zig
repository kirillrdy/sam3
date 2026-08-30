const std = @import("std");

pub const Backend = enum { cuda, opencl, metal };

/// Compute capability of an Ada RTX card.
pub const default_cuda_arch = "sm_89";

/// Where the NVIDIA driver's libcuda tends to live. It ships with the driver,
/// not with the CUDA toolkit, so there is nothing else to install.
const cuda_library_dirs = [_][]const u8{
    "/run/opengl-driver/lib",
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib64",
    "/usr/lib",
};

fn findCudaLibrary(b: *std.Build) ?[]const u8 {
    for (cuda_library_dirs) |dir| {
        const path = b.pathJoin(&.{ dir, "libcuda.so" });
        std.Io.Dir.accessAbsolute(b.graph.io, path, .{}) catch continue;
        return dir;
    }
    return null;
}

fn linkCuda(mod: *std.Build.Module, cuda_library_path: []const u8) void {
    mod.addLibraryPath(.{ .cwd_relative = cuda_library_path });
    mod.addRPath(.{ .cwd_relative = cuda_library_path });
    mod.linkSystemLibrary("cuda", .{});
}

pub const PtxOptions = struct {
    /// Device code. It may `@import("gpu")` for the device-side runtime.
    root_source_file: std.Build.LazyPath,
    arch: []const u8 = default_cuda_arch,
    optimize: std.builtin.OptimizeMode = .Debug,
};

pub fn addPtx(b: *std.Build, gpu_source: std.Build.LazyPath, options: PtxOptions) std.Build.LazyPath {
    const optimize = options.optimize;

    // Device code is a separate compilation: a different target, no libc, no
    // std. The NVPTX backend has no object writer, so the artifact is the
    // assembly it prints -- which for NVPTX is PTX, what the driver JITs.
    const mcpu = if (std.mem.indexOfScalar(u8, options.arch, '+') == null)
        b.fmt("{s}+ptx70", .{options.arch})
    else
        options.arch;

    const cmd = b.addSystemCommand(&.{ b.graph.zig_exe, "build-obj" });
    cmd.addArgs(&.{
        "-target",          "nvptx64-cuda",
        b.fmt("-mcpu={s}", .{mcpu}),
        "-O",               @tagName(optimize),
        "-fno-emit-bin",    "-fstrip",
        "--dep",            "gpu",
    });
    cmd.addPrefixedFileArg("-Mroot=", options.root_source_file);
    cmd.addPrefixedFileArg("-Mgpu=", gpu_source);
    if (b.cache_root.path) |path| cmd.addArgs(&.{ "--cache-dir", path });
    if (b.graph.global_cache_root.path) |path| cmd.addArgs(&.{ "--global-cache-dir", path });
    return cmd.addPrefixedOutputFileArg("-femit-asm=", "kernels.ptx");
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const backend = b.option(
        Backend,
        "backend",
        "GPU driver backend (cuda, opencl, or metal; default: opencl)",
    ) orelse .opencl;

    const cuda_arch = b.option(
        []const u8,
        "sm",
        "Compute capability the CUDA kernels are built for (default: " ++ default_cuda_arch ++ ")",
    ) orelse default_cuda_arch;

    const cuda_library_path_opt = b.option(
        []const u8,
        "cuda-library-path",
        "Directory holding the driver's libcuda.so, for when this does not find it itself",
    );

    // Every operator but the matrix product is bound by how many bytes it
    // moves, so storing tensors as halves is most of what these graphs cost.
    // It is also the precision GPU execution providers commonly use. The
    // OpenCL and Metal kernels support it; the CUDA ones stay float.
    const half = b.option(
        bool,
        "half",
        "Store float tensors on the device as halves (default: true on opencl and metal)",
    ) orelse (backend != .cuda);

    var driver_mod: *std.Build.Module = undefined;
    var kernels: std.Build.LazyPath = undefined;

    switch (backend) {
        .cuda => {
            const cuda_library_path = cuda_library_path_opt orelse findCudaLibrary(b) orelse {
                std.log.err("no libcuda.so found; pass -Dcuda-library-path=DIR", .{});
                std.process.exit(1);
            };
            driver_mod = b.createModule(.{
                .root_source_file = b.path("src/cuda/cuda.zig"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            });
            linkCuda(driver_mod, cuda_library_path);
            kernels = addPtx(b, b.path("src/cuda/gpu.zig"), .{
                .root_source_file = b.path("src/kernels.zig"),
                .arch = cuda_arch,
                .optimize = optimize,
            });
        },
        .opencl => {
            driver_mod = b.createModule(.{
                .root_source_file = b.path("src/opencl/opencl.zig"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            });
            kernels = b.path("src/kernels.cl");
        },
        .metal => {
            driver_mod = b.createModule(.{
                .root_source_file = b.path("src/metal/metal.zig"),
                .target = target,
                .optimize = optimize,
            });
            driver_mod.addIncludePath(b.path("src/metal"));
            driver_mod.addCSourceFile(.{ .file = b.path("src/metal/bridge.m"), .flags = &.{"-fobjc-arc"} });
            driver_mod.link_libc = true;
            driver_mod.linkSystemLibrary("objc", .{});
            driver_mod.linkFramework("Foundation", .{});
            driver_mod.linkFramework("Metal", .{});
            kernels = b.path("src/kernels.metal");
        },
    }

    const mod = b.addModule("engine", .{
        .root_source_file = b.path("src/runtime.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "gpu_driver", .module = driver_mod }},
    });
    mod.addAnonymousImport("kernels", .{ .root_source_file = kernels });

    const options = b.addOptions();
    options.addOption(bool, "half", half and backend != .cuda);
    mod.addOptions("build_options", options);

    const dump = b.addExecutable(.{
        .name = "onnx-dump",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(dump);

    const smoke = b.addExecutable(.{
        .name = "engine-smoke",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/smoke.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "engine", .module = mod }},
        }),
    });
    b.installArtifact(smoke);

    const run_smoke = b.addRunArtifact(smoke);
    run_smoke.step.dependOn(b.getInstallStep());
    b.step("smoke", "Run native tensor kernels on the GPU").dependOn(&run_smoke.step);

    const run = b.addRunArtifact(dump);
    if (b.args) |args| run.addArgs(args);
    b.step("dump", "Print what an ONNX file contains").dependOn(&run.step);

    const tests = b.addTest(.{ .root_module = mod });
    b.step("test", "Run tests").dependOn(&b.addRunArtifact(tests).step);
}
