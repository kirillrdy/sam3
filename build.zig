const std = @import("std");
const engine_build = @import("engine");

/// Where the model runs on the native graph runtime.
const Device = enum { cuda, opencl, metal };

const ModelAsset = struct {
    name: []const u8,
    sha256: []const u8,
    label: []const u8,
};

fn modelAsset(name: []const u8, sha256: []const u8, label: []const u8) ModelAsset {
    return .{ .name = name, .sha256 = sha256, .label = label };
}

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

fn dependOnFetches(step: *std.Build.Step, fetches: []const ModelFetch) void {
    for (fetches) |fetch| step.dependOn(&fetch.run.step);
}

fn installFetch(b: *std.Build, step: *std.Build.Step, fetch: ModelFetch, destination: []const u8) void {
    const install = b.addInstallFile(.{ .cwd_relative = fetch.path }, destination);
    install.step.dependOn(&fetch.run.step);
    step.dependOn(&install.step);
}

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

    const tracker_assets = [_]ModelAsset{
        modelAsset("vision_encoder_fp16.onnx", "f8c52be6de99124bb17f25792054406abc482ce3502166f948d5c5849cd14d02", "vision_encoder_fp16.onnx"),
        modelAsset("vision_encoder_fp16.onnx_data", "4b021a4d3068c9e9153f2f2fcc9e6d280156228671c00ab6c2ddfa737da7f151", "vision_encoder_fp16.onnx_data (892 MiB)"),
        modelAsset("vision_encoder.onnx", "9f284aab8c3d8e81e9c79f7b566f9cea43b7bc9afdd920eee2390fb65b3db897", "vision_encoder.onnx"),
        modelAsset("vision_encoder.onnx_data", "838e1f0b2d0394ed3bd3b3499775dd6676524e1dfc5a7371948a76dcb69e4dd3", "vision_encoder.onnx_data (1.7 GiB)"),
        modelAsset("prompt_encoder_mask_decoder.onnx", "4f9ac85291d634ae36a21ce940e3c09671cc05b6511966e5d3d96988b12b95f8", "prompt_encoder_mask_decoder.onnx"),
        modelAsset("prompt_encoder_mask_decoder.onnx_data", "2d870726d484cb496760fd139c21f115cf1b945c6b69583489faa2ac79f1d2ae", "prompt_encoder_mask_decoder.onnx_data (21 MiB)"),
    };
    var tracker_fetches: [tracker_assets.len]ModelFetch = undefined;
    for (tracker_assets, &tracker_fetches) |model_asset, *fetch| fetch.* = addModelFetch(b, fetch_exe, hf_repo, model_asset);
    const fp16_model = tracker_fetches[0];
    const fp16_data = tracker_fetches[1];
    const fp32_model = tracker_fetches[2];
    const decoder_model = tracker_fetches[4];

    const fp16_step = b.step("fp16-model", "Build the checksum-verified FP16 vision encoder in zig-out/models");
    installFetch(b, fp16_step, fp16_model, "models/vision_encoder_fp16.onnx");
    installFetch(b, fp16_step, fp16_data, "models/vision_encoder_fp16.onnx_data");

    dependOnFetches(weights_step, tracker_fetches[2..]);

    const concept_repo = b.option(
        []const u8,
        "concept-hf-repo",
        "Hugging Face repo containing the SAM 3 text-prompt ONNX export",
    ) orelse "danilobukvic/sam3-text-onnx";
    const concept_step = b.step("fetch-concept-weights", "Download the quantized SAM 3 text-prompt export");
    const concept_assets = [_]ModelAsset{
        modelAsset("vision_encoder_int4.onnx", "88edb4602b7e7b2aa282543dea0b25a253bb13d5d7d5debbd19c2fb5e7941ae7", "concept vision encoder (5.6 MiB)"),
        modelAsset("vision_encoder_int4.onnx.data", "b89c9156064e926761f29be3f87b160fd34f4c93f1de46593295d155621829a2", "concept vision weights (279 MiB)"),
        modelAsset("text_encoder_int4.onnx", "92f824a1841b787dc8dafa8cb8e8dce0c874f8d2d629f6b1c8de88399ede3806", "concept text encoder (2.7 MiB)"),
        modelAsset("text_encoder_int4.onnx.data", "fcf5adcd6ad7b5155409367efde4ee981a5482fd5700191499a666ba4b637db5", "concept text weights (347 MiB)"),
        modelAsset("decoder_int4.onnx", "2354b510382d025ab897fa158abe7da94d065c8f880d60aed35b01820361b06d", "concept decoder (19 MiB)"),
        modelAsset("tokenizer.json", "6d9109cc838977f3ca94a379eec36aecc7c807e1785cd729660ca2fc0171fb35", "concept tokenizer (3.5 MiB)"),
    };
    var concept_fetches: [concept_assets.len]ModelFetch = undefined;
    for (concept_assets, &concept_fetches) |model_asset, *fetch| {
        fetch.* = addHfFetch(b, fetch_exe, concept_repo, "concept", model_asset.name, model_asset);
    }
    dependOnFetches(concept_step, &concept_fetches);
    const concept_vision = concept_fetches[0];
    const concept_text = concept_fetches[2];
    const concept_decoder = concept_fetches[4];
    const concept_tokenizer = concept_fetches[5];

    const default_vision_path = fp32_model.path;
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
    server_options.addOption([]const u8, "example_path", example_path);
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
    run_cmd.step.dependOn(weights_step);
    run_cmd.step.dependOn(concept_step);
    run_cmd.step.dependOn(examples_step);
    run_cmd.setCwd(b.path("."));

    const mod_tests = b.addTest(.{ .root_module = mod });
    const run_mod_tests = b.addRunArtifact(mod_tests);

    const web_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/web/server.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "sam3", .module = mod },
            },
        }),
    });
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
