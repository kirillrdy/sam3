const std = @import("std");

pub const Backend = enum { cuda, opencl, metal };

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
        "Compute capability the CUDA kernels are built for",
    ) orelse "sm_80";

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
            const cuda_build = @import("cuda");
            const cuda_dep = b.dependency("cuda", .{ .target = target, .optimize = optimize });
            kernels = cuda_build.addPtxFor(b, cuda_dep, .{
                .root_source_file = b.path("src/kernels.zig"),
                .arch = cuda_arch,
                .optimize = optimize,
            });
            driver_mod = cuda_dep.module("cuda");
        },
        .opencl => {
            const opencl_dep = b.dependency("opencl", .{ .target = target, .optimize = optimize });
            kernels = b.path("src/kernels.cl");
            driver_mod = opencl_dep.module("opencl");
        },
        .metal => {
            const metal_dep = b.dependency("metal", .{ .target = target, .optimize = optimize });
            kernels = b.path("src/kernels.metal");
            driver_mod = metal_dep.module("metal");
        },
    }

    const mod = b.addModule("engine", .{
        .root_source_file = b.path("src/root.zig"),
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
