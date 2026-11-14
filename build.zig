const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const zigimg = b.dependency("zigimg", .{
        .target = target,
        .optimize = optimize,
    });

    const mod = b.addModule("sam3", .{
        .root_source_file = b.path("src/sam3.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zigimg", .module = zigimg.module("zigimg") },
        },
    });

    const onnx = b.dependency("onnx", .{
        .target = target,
        .optimize = optimize,
    });
    mod.addImport("onnx", onnx.module("onnx"));

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = mod })).step);

    const bench_mod = b.createModule(.{
        .root_source_file = b.path("src/bench.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "sam3", .module = mod },
        },
    });
    const bench_exe = b.addExecutable(.{
        .name = "benchmark",
        .root_module = bench_mod,
    });
    const run_bench = b.addRunArtifact(bench_exe);
    b.installArtifact(bench_exe);
    run_bench.addPassthruArgs();
    const bench_step = b.step("bench", "Run text querying benchmark on cat image");
    bench_step.dependOn(&run_bench.step);
}
