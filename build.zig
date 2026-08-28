const std = @import("std");
const onnxruntime = @import("onnxruntime");
const cuda_build = @import("cuda");

/// Where the model runs. Every one but `cuda` is an ONNX Runtime execution
/// provider; `cuda` swaps the executor itself for the in-tree graph runtime,
/// which talks to the NVIDIA driver directly.
const Device = enum { cpu, npu, gpu, webgpu, cuda };

const webgpu_version = "0.2.1";
const webgpu_url = "https://files.pythonhosted.org/packages/f4/c4/f7de789c43f8a25468c0e5d4a69c28cfb3c29c84c2556c1e6e7dd4c4cee4/onnxruntime_ep_webgpu-0.2.1-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
const webgpu_sha256 = "3ba8e49e09bac60501e71e66a5d8376f5e5fa71b94d71fe90330c4487eb0bd82";

const ModelAsset = struct {
    name: []const u8,
    sha256: []const u8,
    label: []const u8,
};

const ModelFetch = struct {
    path: []const u8,
    run: *std.Build.Step.Run,
};

fn addModelFetch(
    b: *std.Build,
    fetch_exe: *std.Build.Step.Compile,
    hf_repo: []const u8,
    asset: ModelAsset,
) ModelFetch {
    return addHfFetch(b, fetch_exe, hf_repo, "onnx", b.fmt("onnx/{s}", .{asset.name}), asset);
}

fn addHfFetch(
    b: *std.Build,
    fetch_exe: *std.Build.Step.Compile,
    hf_repo: []const u8,
    cache_group: []const u8,
    remote_path: []const u8,
    asset: ModelAsset,
) ModelFetch {
    const path = b.cache_root.join(b.allocator, &.{ "sam3", cache_group, asset.name }) catch @panic("OOM");
    const run = b.addRunArtifact(fetch_exe);
    run.has_side_effects = true;
    run.setCwd(b.path("."));
    run.addArgs(&.{
        "--url",
        b.fmt("https://huggingface.co/{s}/resolve/main/{s}", .{ hf_repo, remote_path }),
        "--out",
        path,
        "--sha256",
        asset.sha256,
        "--token-env",
        "HF_TOKEN",
        "--label",
        asset.label,
    });
    return .{ .path = path, .run = run };
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const device = b.option(
        Device,
        "device",
        "Which device to run the model on: cpu, Intel npu/gpu, cross-vendor webgpu, or nvidia cuda (default: cpu)",
    ) orelse .cpu;

    // The CUDA build is the one that leaves ONNX Runtime behind entirely, so
    // it decides which executor the `runtime` module below resolves to.
    const native_runtime = device == .cuda;

    const with_cuda = b.option(
        bool,
        "cuda",
        "Preprocess images on an NVIDIA GPU through the CUDA driver API, on a build that is otherwise ONNX Runtime's (default: false; implied by -Ddevice=cuda)",
    ) orelse false;

    const cuda_arch = b.option(
        []const u8,
        "sm",
        "Compute capability the CUDA kernels are built for (default: " ++ cuda_build.default_arch ++ ")",
    ) orelse cuda_build.default_arch;

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

    const with_openvino = (device == .npu or device == .gpu) and target.result.os.tag == .linux;
    const with_coreml = !native_runtime and device != .cpu and target.result.os.tag.isDarwin();
    const with_webgpu = device == .webgpu;
    if (with_webgpu and (target.result.os.tag != .linux or target.result.cpu.arch != .x86_64)) {
        std.log.err("the pinned WebGPU provider currently supports x86_64-linux only", .{});
        std.process.exit(1);
    }
    const ort = if (native_runtime) null else b.dependency("onnxruntime", .{
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

    // Keep the model layer independent of the executor: it imports `runtime`
    // and never learns which one it got. ONNX Runtime stays the default
    // because it runs anywhere. The CUDA engine covers every operator both
    // prompting paths reach, but only on an NVIDIA GPU.
    if (native_runtime) {
        const engine = b.dependency("engine", .{
            .target = target,
            .optimize = optimize,
            .sm = cuda_arch,
        });
        mod.addImport("runtime", engine.module("engine"));
    } else {
        const ort_runtime = b.createModule(.{
            .root_source_file = b.path("src/onnx.zig"),
            .target = target,
            .optimize = optimize,
        });
        ort_runtime.linkLibrary(ort.?.artifact("onnxruntime"));
        mod.addImport("runtime", ort_runtime);
    }

    // The GPU preprocessing path is a swappable module, so nothing else in
    // sam3 needs to know whether this build can talk to a GPU at all.
    if (with_cuda or native_runtime) {
        const cuda_dep = b.dependency("cuda", .{ .target = target, .optimize = optimize });
        const kernels = cuda_build.addPtxFor(b, cuda_dep, .{
            .root_source_file = b.path("src/gpu/kernels.zig"),
            .arch = cuda_arch,
            .optimize = optimize,
        });
        const backend = b.createModule(.{
            .root_source_file = b.path("src/gpu/cuda.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "cuda", .module = cuda_dep.module("cuda") }},
        });
        backend.addAnonymousImport("kernels.ptx", .{ .root_source_file = kernels });
        mod.addImport("gpu", backend);
    } else {
        mod.addImport("gpu", b.createModule(.{
            .root_source_file = b.path("src/gpu/disabled.zig"),
            .target = target,
            .optimize = optimize,
        }));
    }

    var openvino_provider_path: []const u8 = "";
    var provider_cache_path: []const u8 = "";
    if (with_openvino) {
        provider_cache_path = b.cache_root.join(b.allocator, &.{ "sam3", "openvino-cache" }) catch @panic("OOM");

        onnxruntime.linkStdCxx(b, mod);

        b.getInstallStep().dependOn(&b.addInstallArtifact(
            ort.?.artifact("onnxruntime_providers_shared"),
            .{ .dest_dir = .{ .override = .bin } },
        ).step);

        const provider = b.addInstallArtifact(ort.?.artifact("onnxruntime_providers_openvino"), .{});
        b.getInstallStep().dependOn(&provider.step);
        b.getInstallStep().dependOn(&b.addInstallArtifact(
            ort.artifact("onnxruntime_providers_openvino"),
            .{ .dest_dir = .{ .override = .bin } },
        ).step);
        openvino_provider_path = b.getInstallPath(.lib, "libonnxruntime_providers_openvino.so");
    }
    if (with_coreml) {
        provider_cache_path = b.cache_root.join(b.allocator, &.{ "sam3", "coreml-cache" }) catch @panic("OOM");
    }

    if (device == .gpu and with_openvino) {
        b.installArtifact(ort.?.artifact("OpenCL"));
        runtime_paths.append(b.allocator, b.getInstallPath(.lib, "")) catch @panic("OOM");
    }
    if (with_webgpu) {
        for ([_][]const u8{
            "/run/current-system/sw/lib",
            "/run/current-system/sw/share/google/chrome",
            "/run/opengl-driver/lib",
        }) |dir| {
            std.Io.Dir.accessAbsolute(b.graph.io, dir, .{}) catch continue;
            runtime_paths.append(b.allocator, dir) catch @panic("OOM");
        }
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

    var webgpu_provider_path: []const u8 = "";
    if (with_webgpu) {
        const unzip_exe = b.addExecutable(.{
            .name = "unzip",
            .root_module = b.createModule(.{
                .root_source_file = b.path("tools/unzip.zig"),
                .target = b.graph.host,
                .optimize = optimize,
                .strip = if (optimize == .ReleaseFast) true else null,
            }),
        });

        const wheel = b.cache_root.join(b.allocator, &.{
            "sam3",
            "webgpu",
            "onnxruntime_ep_webgpu-" ++ webgpu_version ++ ".whl",
        }) catch @panic("OOM");
        const fetch_webgpu = b.addRunArtifact(fetch_exe);
        fetch_webgpu.has_side_effects = true;
        fetch_webgpu.addArgs(&.{
            "--url",    webgpu_url,
            "--out",    wheel,
            "--sha256", webgpu_sha256,
            "--label",  "ONNX Runtime WebGPU provider (4.2 MiB)",
        });

        const unzip_webgpu = b.addRunArtifact(unzip_exe);
        unzip_webgpu.addFileArg(.{ .cwd_relative = wheel });
        const unpacked = unzip_webgpu.addOutputDirectoryArg("onnxruntime-webgpu-" ++ webgpu_version);
        unzip_webgpu.step.dependOn(&fetch_webgpu.step);
        const provider = unpacked.path(b, "onnxruntime_ep_webgpu/libonnxruntime_providers_webgpu.so");

        const install_provider = b.addInstallFile(provider, "lib/libonnxruntime_providers_webgpu.so");
        b.getInstallStep().dependOn(&install_provider.step);
        webgpu_provider_path = b.getInstallPath(.lib, "libonnxruntime_providers_webgpu.so");
    }

    const hf_repo = b.option(
        []const u8,
        "hf-repo",
        "Hugging Face repo to pull the SAM 3 ONNX export from",
    ) orelse "onnx-community/sam3-tracker-ONNX";

    const weights_step = b.step("fetch-weights", "Download the SAM 3 tracker ONNX export");

    const fp16_model = addModelFetch(b, fetch_exe, hf_repo, .{
        .name = "vision_encoder_fp16.onnx",
        .sha256 = "f8c52be6de99124bb17f25792054406abc482ce3502166f948d5c5849cd14d02",
        .label = "vision_encoder_fp16.onnx",
    });
    const fp16_data = addModelFetch(b, fetch_exe, hf_repo, .{
        .name = "vision_encoder_fp16.onnx_data",
        .sha256 = "4b021a4d3068c9e9153f2f2fcc9e6d280156228671c00ab6c2ddfa737da7f151",
        .label = "vision_encoder_fp16.onnx_data (892 MiB)",
    });

    const fp16_step = b.step("fp16-model", "Build the checksum-verified FP16 vision encoder in zig-out/models");
    const install_fp16_model = b.addInstallFile(
        .{ .cwd_relative = fp16_model.path },
        "models/vision_encoder_fp16.onnx",
    );
    install_fp16_model.step.dependOn(&fp16_model.run.step);
    fp16_step.dependOn(&install_fp16_model.step);
    const install_fp16_data = b.addInstallFile(
        .{ .cwd_relative = fp16_data.path },
        "models/vision_encoder_fp16.onnx_data",
    );
    install_fp16_data.step.dependOn(&fp16_data.run.step);
    fp16_step.dependOn(&install_fp16_data.step);

    const fp32_model = addModelFetch(b, fetch_exe, hf_repo, .{
        .name = "vision_encoder.onnx",
        .sha256 = "9f284aab8c3d8e81e9c79f7b566f9cea43b7bc9afdd920eee2390fb65b3db897",
        .label = "vision_encoder.onnx",
    });
    const fp32_data = addModelFetch(b, fetch_exe, hf_repo, .{
        .name = "vision_encoder.onnx_data",
        .sha256 = "838e1f0b2d0394ed3bd3b3499775dd6676524e1dfc5a7371948a76dcb69e4dd3",
        .label = "vision_encoder.onnx_data (1.7 GiB)",
    });
    const q4_model = addModelFetch(b, fetch_exe, hf_repo, .{
        .name = "vision_encoder_q4.onnx",
        .sha256 = "b80cb1cb6ab80efe646ad49ffa11d398af76e8912eb966971932ef2d8e7fe11b",
        .label = "vision_encoder_q4.onnx",
    });
    const q4_data = addModelFetch(b, fetch_exe, hf_repo, .{
        .name = "vision_encoder_q4.onnx_data",
        .sha256 = "be632cacb4a82ef8be28285e49821b9dd7f96e3231cb90df631890f367110555",
        .label = "vision_encoder_q4.onnx_data (352 MiB)",
    });
    const decoder_model = addModelFetch(b, fetch_exe, hf_repo, .{
        .name = "prompt_encoder_mask_decoder.onnx",
        .sha256 = "4f9ac85291d634ae36a21ce940e3c09671cc05b6511966e5d3d96988b12b95f8",
        .label = "prompt_encoder_mask_decoder.onnx",
    });
    const decoder_data = addModelFetch(b, fetch_exe, hf_repo, .{
        .name = "prompt_encoder_mask_decoder.onnx_data",
        .sha256 = "2d870726d484cb496760fd139c21f115cf1b945c6b69583489faa2ac79f1d2ae",
        .label = "prompt_encoder_mask_decoder.onnx_data (21 MiB)",
    });

    if (with_coreml) {
        weights_step.dependOn(&fp16_model.run.step);
        weights_step.dependOn(&fp16_data.run.step);
    } else if (with_webgpu) {
        weights_step.dependOn(&q4_model.run.step);
        weights_step.dependOn(&q4_data.run.step);
    } else {
        weights_step.dependOn(&fp32_model.run.step);
        weights_step.dependOn(&fp32_data.run.step);
    }
    weights_step.dependOn(&decoder_model.run.step);
    weights_step.dependOn(&decoder_data.run.step);

    const concept_repo = b.option(
        []const u8,
        "concept-hf-repo",
        "Hugging Face repo containing the SAM 3 text-prompt ONNX export",
    ) orelse "danilobukvic/sam3-text-onnx";
    const concept_step = b.step("fetch-concept-weights", "Download the quantized SAM 3 text-prompt export");
    const concept_vision = addHfFetch(b, fetch_exe, concept_repo, "concept", "vision_encoder_int4.onnx", .{
        .name = "vision_encoder_int4.onnx",
        .sha256 = "88edb4602b7e7b2aa282543dea0b25a253bb13d5d7d5debbd19c2fb5e7941ae7",
        .label = "concept vision encoder (5.6 MiB)",
    });
    const concept_vision_data = addHfFetch(b, fetch_exe, concept_repo, "concept", "vision_encoder_int4.onnx.data", .{
        .name = "vision_encoder_int4.onnx.data",
        .sha256 = "b89c9156064e926761f29be3f87b160fd34f4c93f1de46593295d155621829a2",
        .label = "concept vision weights (279 MiB)",
    });
    const concept_text = addHfFetch(b, fetch_exe, concept_repo, "concept", "text_encoder_int4.onnx", .{
        .name = "text_encoder_int4.onnx",
        .sha256 = "92f824a1841b787dc8dafa8cb8e8dce0c874f8d2d629f6b1c8de88399ede3806",
        .label = "concept text encoder (2.7 MiB)",
    });
    const concept_text_data = addHfFetch(b, fetch_exe, concept_repo, "concept", "text_encoder_int4.onnx.data", .{
        .name = "text_encoder_int4.onnx.data",
        .sha256 = "fcf5adcd6ad7b5155409367efde4ee981a5482fd5700191499a666ba4b637db5",
        .label = "concept text weights (347 MiB)",
    });
    const concept_decoder = addHfFetch(b, fetch_exe, concept_repo, "concept", "decoder_int4.onnx", .{
        .name = "decoder_int4.onnx",
        .sha256 = "2354b510382d025ab897fa158abe7da94d065c8f880d60aed35b01820361b06d",
        .label = "concept decoder (19 MiB)",
    });
    const concept_tokenizer = addHfFetch(b, fetch_exe, concept_repo, "concept", "tokenizer.json", .{
        .name = "tokenizer.json",
        .sha256 = "6d9109cc838977f3ca94a379eec36aecc7c807e1785cd729660ca2fc0171fb35",
        .label = "concept tokenizer (3.5 MiB)",
    });
    for ([_]*std.Build.Step.Run{
        concept_vision.run,
        concept_vision_data.run,
        concept_text.run,
        concept_text_data.run,
        concept_decoder.run,
        concept_tokenizer.run,
    }) |fetch| concept_step.dependOn(&fetch.step);

    const default_vision_path = if (with_coreml)
        fp16_model.path
    else if (with_webgpu)
        q4_model.path
    else
        fp32_model.path;
    const vision_path = b.option(
        []const u8,
        "vision-encoder-path",
        "Use a local vision encoder ONNX model instead of the downloaded default",
    ) orelse default_vision_path;
    const decoder_path = decoder_model.path;

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
    server_options.addOption([]const u8, "concept_vision_path", concept_vision.path);
    server_options.addOption([]const u8, "concept_text_path", concept_text.path);
    server_options.addOption([]const u8, "concept_decoder_path", concept_decoder.path);
    server_options.addOption([]const u8, "concept_tokenizer_path", concept_tokenizer.path);
    server_options.addOption([]const u8, "openvino_provider_path", openvino_provider_path);
    server_options.addOption([]const u8, "webgpu_provider_path", webgpu_provider_path);
    server_options.addOption([]const u8, "provider_cache_path", provider_cache_path);
    server_options.addOption([]const u8, "example_path", example_path);
    // The application names devices the way an execution provider does, and
    // the in-tree runtime enumerates exactly one of them: an NVIDIA GPU.
    server_options.addOption([]const u8, "device", if (native_runtime) "gpu" else @tagName(device));
    server_options.addOption(bool, "untested_npu", untested_npu);
    server_options.addOption(bool, "cuda", with_cuda or native_runtime);
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
    if (with_openvino or with_webgpu) onnxruntime.linkStdCxx(b, web_exe.root_module);
    b.installArtifact(web_exe);

    const compare_exe = b.addExecutable(.{
        .name = "sam3-compare",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/compare/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "sam3", .module = mod }},
        }),
    });
    compare_exe.root_module.addOptions("build_options", server_options);
    if (with_openvino or with_webgpu) onnxruntime.linkStdCxx(b, compare_exe.root_module);

    const compare_step = b.step(
        "compare",
        "Segment one image twice, preprocessing on the CPU and on the GPU, and diff the results",
    );
    const compare_cmd = b.addRunArtifact(compare_exe);
    compare_cmd.step.dependOn(weights_step);
    compare_cmd.step.dependOn(examples_step);
    compare_cmd.setCwd(b.path("."));
    if (b.args) |args| compare_cmd.addArgs(args);
    compare_step.dependOn(&compare_cmd.step);

    const run_step = b.step("run", b.fmt("Run the web UI on http://{s}:{d}/", .{ host, port }));
    const run_cmd = b.addRunArtifact(web_exe);
    run_step.dependOn(&run_cmd.step);

    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.step.dependOn(weights_step);
    run_cmd.step.dependOn(concept_step);
    run_cmd.step.dependOn(examples_step);
    run_cmd.setCwd(b.path("."));
    if (with_openvino or with_webgpu) {
        onnxruntime.addOpenVinoRuntimeEnvironment(
            b,
            run_cmd,
            if (with_openvino)
                std.meta.stringToEnum(onnxruntime.OpenVinoDevice, @tagName(device)).?
            else
                .cpu,
            device_library_path,
            runtime_paths.items,
            if (with_openvino) opencl_driver_path else null,
        );
    }

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
