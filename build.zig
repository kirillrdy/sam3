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

    const zimo = b.dependency("zimo", .{
        .target = target,
        .optimize = optimize,
    });
    mod.addImport("zimo", zimo.module("zimo"));

    const onnx = b.dependency("onnx", .{
        .target = target,
        .optimize = optimize,
        .backend = backend,
        .sm = cuda_arch,
        .half = half,
    });
    mod.addImport("onnx", onnx.module("onnx"));

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
    run_cmd.setCwd(b.path("."));

    if (target.result.os.tag.isDarwin()) {
        const macos_mod = b.createModule(.{
            .root_source_file = b.path("macos/main.zig"),
            .target = target,
            .optimize = optimize,
            .strip = if (optimize == .ReleaseFast) true else null,
            .imports = &.{
                .{ .name = "sam3", .module = mod },
                .{ .name = "zigimg", .module = zigimg.module("zigimg") },
            },
        });
        macos_mod.addIncludePath(b.path("macos"));
        macos_mod.addCSourceFile(.{
            .file = b.path("macos/bridge.m"),
            .flags = &.{"-fobjc-arc"},
        });
        macos_mod.link_libc = true;
        macos_mod.linkSystemLibrary("objc", .{});
        macos_mod.linkFramework("Foundation", .{});
        macos_mod.linkFramework("AppKit", .{});
        macos_mod.linkFramework("QuartzCore", .{});
        macos_mod.linkFramework("UniformTypeIdentifiers", .{});

        const macos_exe = b.addExecutable(.{
            .name = "sam3-macos",
            .root_module = macos_mod,
        });
        b.installArtifact(macos_exe);

        const run_macos_step = b.step("run-macos", "Run the native macOS UI");
        const run_macos_cmd = b.addRunArtifact(macos_exe);
        run_macos_step.dependOn(&run_macos_cmd.step);
        run_macos_cmd.setCwd(b.path("."));
    }

    if (target.result.os.tag == .linux or target.result.os.tag.isDarwin()) {
        const wayland_mod = b.createModule(.{
            .root_source_file = b.path("linux/main.zig"),
            .target = target,
            .optimize = optimize,
            .strip = if (optimize == .ReleaseFast) true else null,
            .imports = &.{
                .{ .name = "sam3", .module = mod },
                .{ .name = "zigimg", .module = zigimg.module("zigimg") },
            },
        });
        wayland_mod.link_libc = true;

        const wayland_exe_name = if (target.result.os.tag == .linux) "sam3-linux" else "sam3-wayland";
        const wayland_exe = b.addExecutable(.{
            .name = wayland_exe_name,
            .root_module = wayland_mod,
        });
        b.installArtifact(wayland_exe);

        const run_wayland_step = b.step("run-wayland", "Run the native Wayland UI");
        const run_wayland_cmd = b.addRunArtifact(wayland_exe);
        run_wayland_step.dependOn(&run_wayland_cmd.step);
        run_wayland_cmd.setCwd(b.path("."));

        if (target.result.os.tag == .linux) {
            const run_linux_step = b.step("run-linux", "Run the native Linux Wayland UI");
            run_linux_step.dependOn(&run_wayland_cmd.step);
        }
    }

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
    addTest(b, test_step, b.createModule(.{
        .root_source_file = b.path("src/render.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zigimg", .module = zigimg.module("zigimg") },
        },
    }));
    addTest(b, test_step, b.createModule(.{
        .root_source_file = b.path("linux/font.zig"),
        .target = target,
        .optimize = optimize,
    }));
}

fn addTest(b: *std.Build, step: *std.Build.Step, module: *std.Build.Module) void {
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = module })).step);
}
