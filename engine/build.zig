const std = @import("std");
const cuda_build = @import("cuda");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const cuda_arch = b.option(
        []const u8,
        "sm",
        "Compute capability the CUDA kernels are built for (default: " ++ cuda_build.default_arch ++ ")",
    ) orelse cuda_build.default_arch;

    const cuda_dep = b.dependency("cuda", .{ .target = target, .optimize = optimize });
    const kernels = cuda_build.addPtxFor(b, cuda_dep, .{
        .root_source_file = b.path("src/kernels.zig"),
        .arch = cuda_arch,
        .optimize = optimize,
    });

    const mod = b.addModule("engine", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "cuda", .module = cuda_dep.module("cuda") }},
    });
    mod.addAnonymousImport("kernels.ptx", .{ .root_source_file = kernels });

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
    b.step("smoke", "Run native tensor kernels on the NVIDIA GPU").dependOn(&run_smoke.step);

    const run = b.addRunArtifact(dump);
    if (b.args) |args| run.addArgs(args);
    b.step("dump", "Print what an ONNX file contains").dependOn(&run.step);

    const tests = b.addTest(.{ .root_module = mod });
    b.step("test", "Run tests").dependOn(&b.addRunArtifact(tests).step);
}
