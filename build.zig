const std = @import("std");
const engine_build = @import("engine");

/// Where the model runs on the native graph runtime.
const Device = enum { cuda, opencl, metal };

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const default_device: Device = if (target.result.os.tag.isDarwin()) .metal else .opencl;
    const device = b.option(
        Device,
        "device",
        "Which device to run the model on: native Intel opencl, Apple metal, or nvidia cuda",
    ) orelse default_device;

    if (device == .metal and !target.result.os.tag.isDarwin()) {
        std.log.err("the Metal backend requires an Apple target", .{});
        std.process.exit(1);
    }

    const cuda_arch = b.option(
        []const u8,
        "sm",
        "Compute capability the CUDA kernels are built for (default: " ++ engine_build.default_cuda_arch ++ ")",
    ) orelse engine_build.default_cuda_arch;

    // What the in-tree runtime stores a float tensor as. Half is the default
    // on OpenCL and Metal, where nearly every operator is bound by how many
    // bytes it moves, and is the precision GPU execution providers use anyway.
    const half = b.option(
        bool,
        "half",
        "Store float tensors on the device as halves (default: true with -Ddevice=opencl or metal)",
    ) orelse (device != .cuda);

    const host = b.option(
        []const u8,
        "host",
        "Address the web UI binds to (default: 127.0.0.1)",
    ) orelse "127.0.0.1";

    const port = b.option(
        u16,
        "port",
        "Port the web UI listens on (default: 3000)",
    ) orelse 3000;

    const zig_http = b.option(
        bool,
        "zig-http",
        "Use Zig's built-in HTTP client for model downloads instead of curl (default: false)",
    ) orelse false;

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

    const engine = b.dependency("engine", .{
        .target = target,
        .optimize = optimize,
        .backend = device,
        .sm = cuda_arch,
        .half = half,
    });
    mod.addImport("runtime", engine.module("engine"));

    const server_options = b.addOptions();
    server_options.addOption([]const u8, "host", host);
    server_options.addOption(u16, "port", port);
    server_options.addOption(bool, "zig_http", zig_http);

    const wasm_target = b.resolveTargetQuery(.{
        .cpu_arch = .wasm32,
        .os_tag = .freestanding,
    });

    const client = b.addExecutable(.{
        .name = "client",
        .root_module = b.createModule(.{
            .root_source_file = b.path("web/client.zig"),
            .target = wasm_target,
            .optimize = optimize,
            .strip = if (optimize == .ReleaseFast) true else null,
            .imports = &.{
                .{ .name = "zigimg", .module = b.dependency("zigimg", .{
                    .target = wasm_target,
                    .optimize = optimize,
                }).module("zigimg") },
            },
        }),
    });

    client.entry = .disabled;
    client.rdynamic = true;

    const web_exe = b.addExecutable(.{
        .name = "sam3-web",
        .root_module = b.createModule(.{
            .root_source_file = b.path("main.zig"),
            .target = target,
            .optimize = optimize,
            .strip = if (optimize == .ReleaseFast) true else null,
            .imports = &.{
                .{ .name = "sam3", .module = mod },
            },
        }),
    });

    web_exe.root_module.addAnonymousImport("client_wasm", .{
        .root_source_file = client.getEmittedBin(),
    });
    web_exe.root_module.addOptions("build_options", server_options);
    b.installArtifact(web_exe);

    const run_step = b.step("run", b.fmt("Run the web UI on http://{s}:{d}/", .{ host, port }));
    const run_cmd = b.addRunArtifact(web_exe);
    run_step.dependOn(&run_cmd.step);

    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.setCwd(b.path("."));

    const test_step = b.step("test", "Run tests");
    addTest(b, test_step, mod);
    addTest(b, test_step, b.createModule(.{
        .root_source_file = b.path("web/server.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "sam3", .module = mod },
        },
    }));
    addTest(b, test_step, b.createModule(.{
        .root_source_file = b.path("web/client.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zigimg", .module = zigimg.module("zigimg") },
        },
    }));
}

fn addTest(b: *std.Build, step: *std.Build.Step, module: *std.Build.Module) void {
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = module })).step);
}
