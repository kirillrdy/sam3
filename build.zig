const std = @import("std");
const onnx_build = @import("onnx");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const default_backend: onnx_build.Backend = if (target.result.os.tag.isDarwin()) .metal else .opencl;
    const backend = b.option(
        onnx_build.Backend,
        "backend",
        "ONNX execution backend: native OpenCL, Metal, or CUDA",
    ) orelse default_backend;

    if (backend == .metal and !target.result.os.tag.isDarwin()) {
        std.log.err("the Metal backend requires an Apple target", .{});
        std.process.exit(1);
    }

    const cuda_arch = b.option(
        []const u8,
        "sm",
        "Compute capability the CUDA kernels are built for (default: " ++ onnx_build.default_cuda_arch ++ ")",
    ) orelse onnx_build.default_cuda_arch;

    // What the runtime stores a float tensor as. Half is the default
    // on OpenCL and Metal, where nearly every operator is bound by how many
    // bytes it moves, and is the precision GPU execution providers use anyway.
    const half = b.option(
        bool,
        "half",
        "Store float tensors on the device as halves (default: true with -Dbackend=opencl or metal)",
    ) orelse (backend != .cuda);

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
        .backend = backend,
        .sm = cuda_arch,
        .half = half,
    });
    mod.addImport("onnx", onnx.module("onnx"));

    const test_step = b.step("test", "Run tests");
    addTest(b, test_step, mod);
    addTest(b, test_step, b.createModule(.{
        .root_source_file = b.path("src/render.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zigimg", .module = zigimg.module("zigimg") },
        },
    }));

    const bench_mod = b.createModule(.{
        .root_source_file = b.path("src/bench.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "sam3", .module = mod },
            .{ .name = "zigimg", .module = zigimg.module("zigimg") },
        },
    });
    const bench_exe = b.addExecutable(.{
        .name = "benchmark",
        .root_module = bench_mod,
    });
    const run_bench = b.addRunArtifact(bench_exe);
    if (b.args) |args| {
        run_bench.addArgs(args);
    }
    const bench_step = b.step("bench", "Run text querying benchmark on cat image");
    bench_step.dependOn(&run_bench.step);
}

fn addTest(b: *std.Build, step: *std.Build.Step, module: *std.Build.Module) void {
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = module })).step);
}
