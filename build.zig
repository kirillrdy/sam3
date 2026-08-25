const std = @import("std");
const onnxruntime = @import("onnxruntime");

const Device = onnxruntime.OpenVinoDevice;

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const device = b.option(
        Device,
        "device",
        "Which device to run the model on (default: cpu). Asking for anything but the CPU builds the OpenVINO execution provider",
    ) orelse .cpu;

    const untested_npu = b.option(
        bool,
        "untested-npu",
        "Use the NPU even on a generation this has not been run on (default: false)",
    ) orelse false;

    const device_library_path = b.option(
        []const []const u8,
        "device-library-path",
        "Directory holding the device's own libraries -- the Level Zero loader, the NPU driver, the GPU's OpenCL driver -- for when `run` does not find them itself",
    ) orelse &.{};

    const opencl_driver_path = b.option(
        []const u8,
        "opencl-driver",
        "Path to the Intel GPU's OpenCL driver (libigdrcl.so), for when there is no ICD registry naming it and `run` does not find it itself",
    );

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

    const with_openvino = device != .cpu;
    const ort = b.dependency("onnxruntime", .{
        .target = target,
        .optimize = optimize,
        .openvino = with_openvino,
    });

    var runtime_paths: std.ArrayList([]const u8) = .empty;
    if (with_openvino) {
        runtime_paths.appendSlice(b.allocator, onnxruntime.openvinoRuntimeLibraryPaths(b)) catch @panic("OOM");
    }

    const zigimg = b.dependency("zigimg", .{
        .target = target,
        .optimize = optimize,
    });

    const mod = b.addModule("sam3", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zigimg", .module = zigimg.module("zigimg") },
        },
    });

    mod.linkLibrary(ort.artifact("onnxruntime"));

    var provider_path: []const u8 = "";
    var cache_path: []const u8 = "";
    if (with_openvino) {
        cache_path = b.cache_root.join(b.allocator, &.{ "sam3", "openvino-cache" }) catch @panic("OOM");

        onnxruntime.linkStdCxx(b, mod);

        b.getInstallStep().dependOn(&b.addInstallArtifact(
            ort.artifact("onnxruntime_providers_shared"),
            .{ .dest_dir = .{ .override = .bin } },
        ).step);

        const provider = b.addInstallArtifact(ort.artifact("onnxruntime_providers_openvino"), .{});
        b.getInstallStep().dependOn(&provider.step);
        provider_path = b.getInstallPath(.lib, "libonnxruntime_providers_openvino.so");
    }

    if (device == .gpu) {
        b.installArtifact(ort.artifact("OpenCL"));
        runtime_paths.append(b.allocator, b.getInstallPath(.lib, "")) catch @panic("OOM");
    }

    const fetch_exe = b.addExecutable(.{
        .name = "fetch",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/fetch.zig"),
            .target = b.graph.host,
            .optimize = optimize,
            .strip = if (optimize == .ReleaseFast) true else null,
        }),
    });

    const hf_repo = b.option(
        []const u8,
        "hf-repo",
        "Hugging Face repo to pull the SAM 3 ONNX export from",
    ) orelse "onnx-community/sam3-tracker-ONNX";

    const weights_step = b.step("fetch-weights", "Download the SAM 3 tracker ONNX export");

    var vision_path: []const u8 = "";
    var decoder_path: []const u8 = "";

    for ([_][3][]const u8{
        .{
            "vision_encoder.onnx",
            "9f284aab8c3d8e81e9c79f7b566f9cea43b7bc9afdd920eee2390fb65b3db897",
            "vision_encoder.onnx",
        },
        .{
            "vision_encoder.onnx_data",
            "838e1f0b2d0394ed3bd3b3499775dd6676524e1dfc5a7371948a76dcb69e4dd3",
            "vision_encoder.onnx_data (1.7 GiB)",
        },
        .{
            "prompt_encoder_mask_decoder.onnx",
            "4f9ac85291d634ae36a21ce940e3c09671cc05b6511966e5d3d96988b12b95f8",
            "prompt_encoder_mask_decoder.onnx",
        },
        .{
            "prompt_encoder_mask_decoder.onnx_data",
            "2d870726d484cb496760fd139c21f115cf1b945c6b69583489faa2ac79f1d2ae",
            "prompt_encoder_mask_decoder.onnx_data (21 MiB)",
        },
    }) |asset| {
        const out = b.cache_root.join(b.allocator, &.{ "sam3", "onnx", asset[0] }) catch @panic("OOM");
        if (std.mem.eql(u8, asset[0], "vision_encoder.onnx")) vision_path = out;
        if (std.mem.eql(u8, asset[0], "prompt_encoder_mask_decoder.onnx")) decoder_path = out;

        const run = b.addRunArtifact(fetch_exe);
        run.has_side_effects = true;
        run.setCwd(b.path("."));
        run.addArgs(&.{
            "--url",
            b.fmt("https://huggingface.co/{s}/resolve/main/onnx/{s}", .{ hf_repo, asset[0] }),
            "--out",
            out,
            "--sha256",
            asset[1],
            "--token-env",
            "HF_TOKEN",
            "--label",
            asset[2],
        });
        weights_step.dependOn(&run.step);
    }

    const examples_step = b.step("fetch-examples", "Download the playground sample image");
    const example_path = b.cache_root.join(b.allocator, &.{ "sam3", "examples", "cat.png" }) catch @panic("OOM");
    const fetch_example = b.addRunArtifact(fetch_exe);
    fetch_example.has_side_effects = true;
    fetch_example.setCwd(b.path("."));
    fetch_example.addArgs(&.{
        "--url",
        "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=800&fm=png",
        "--out",
        example_path,
        "--label",
        "cat.png",
    });
    examples_step.dependOn(&fetch_example.step);

    const server_options = b.addOptions();
    server_options.addOption([]const u8, "vision_encoder_path", vision_path);
    server_options.addOption([]const u8, "decoder_path", decoder_path);
    server_options.addOption([]const u8, "openvino_provider_path", provider_path);
    server_options.addOption([]const u8, "openvino_cache_path", cache_path);
    server_options.addOption([]const u8, "example_path", example_path);
    server_options.addOption([]const u8, "device", @tagName(device));
    server_options.addOption(bool, "untested_npu", untested_npu);
    server_options.addOption([]const u8, "host", host);
    server_options.addOption(u16, "port", port);

    const wasm_target = b.resolveTargetQuery(.{
        .cpu_arch = .wasm32,
        .os_tag = .freestanding,
    });

    const client = b.addExecutable(.{
        .name = "client",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/client.zig"),
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
            .root_source_file = b.path("src/web/main.zig"),
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
    if (with_openvino) onnxruntime.linkStdCxx(b, web_exe.root_module);
    b.installArtifact(web_exe);

    const run_step = b.step("run", b.fmt("Run the web UI on http://{s}:{d}/", .{ host, port }));
    const run_cmd = b.addRunArtifact(web_exe);
    run_step.dependOn(&run_cmd.step);

    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.step.dependOn(weights_step);
    run_cmd.step.dependOn(examples_step);
    run_cmd.setCwd(b.path("."));
    if (with_openvino) onnxruntime.addOpenVinoRuntimeEnvironment(
        b,
        run_cmd,
        device,
        device_library_path,
        runtime_paths.items,
        opencl_driver_path,
    );

    const mod_tests = b.addTest(.{ .root_module = mod });
    const run_mod_tests = b.addRunArtifact(mod_tests);

    const web_tests = b.addTest(.{ .root_module = web_exe.root_module });
    const run_web_tests = b.addRunArtifact(web_tests);

    const client_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/client.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zigimg", .module = zigimg.module("zigimg") },
            },
        }),
    });
    const run_client_tests = b.addRunArtifact(client_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_web_tests.step);
    test_step.dependOn(&run_client_tests.step);
}
