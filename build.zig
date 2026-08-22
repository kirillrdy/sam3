//! SAM 3 on ONNX Runtime.
//!
//! The runtime is built from source by the `onnxruntime` package next door, so
//! there is nothing to install for the default build. `-Dopenvino=<prefix>`
//! additionally builds that package's OpenVINO execution provider, which is how
//! the runtime reaches an Intel NPU:
//!
//!     zig build run -Dopenvino=/path/to/openvino
//!
//! That build links a system OpenVINO and the host's libstdc++ rather than
//! Zig's libc++ -- the provider and the runtime hand each other C++ objects
//! across a dlopen boundary, so both have to match the ABI of the prebuilt
//! `libopenvino.so`. See the `onnxruntime` package for what that costs.
//!
//! At run time the NPU plugin reaches the device through the Level Zero loader,
//! which it dlopens by name, and the loader in turn dlopens the NPU driver. Both
//! `libze_loader.so.1` and `libze_intel_npu.so.1` have to be on the library path.
//! On NixOS they are in two different places -- the loader is an ordinary
//! package, the driver is a hardware driver -- and neither is anywhere the
//! dynamic loader looks by itself:
//!
//!     LD_LIBRARY_PATH=/run/opengl-driver/lib:/run/current-system/sw/lib \\
//!         zig build run -Dopenvino=...

const std = @import("std");
const onnxruntime = @import("onnxruntime");

/// Where a session is asked to run. Not a promise: a graph a device turns down
/// -- too large for the NPU, an operator the provider does not implement --
/// falls back to the CPU at run time and says so.
const Device = enum { npu, gpu, cpu };

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    // ReleaseFast by default: preprocessing a frame and resampling masks back
    // up to it are the only arithmetic left in this program, but a Debug build
    // of them is slow enough to look like the model is at fault.
    const optimize = b.option(
        std.builtin.OptimizeMode,
        "optimize",
        "Prioritize performance, safety, or binary size (default: ReleaseFast)",
    ) orelse .ReleaseFast;

    // --- ONNX Runtime -------------------------------------------------------
    //
    // Without `-Dopenvino` this is one static library and nothing else: the CPU
    // execution provider, compiled from source, depending on no system library
    // beyond libc. With it there are three artifacts and a system OpenVINO
    // behind them, and the wiring below is what the runtime needs to find them
    // again at run time.

    const openvino = b.option(
        []const u8,
        "openvino",
        "Prefix of a system OpenVINO (2026.0+) to build the execution provider against, for Intel NPU and GPU support",
    );
    const openvino_include = b.option(
        []const u8,
        "openvino-include",
        "Include directory of the OpenVINO headers, if not <prefix>/include",
    );
    const device = b.option(
        Device,
        "device",
        "Which device to run the model on (default: npu with -Dopenvino, otherwise cpu)",
    ) orelse if (openvino != null) Device.npu else Device.cpu;

    const untested_npu = b.option(
        bool,
        "untested-npu",
        "Use the NPU even on a generation this has not been run on (default: false)",
    ) orelse false;

    if (openvino == null and device != .cpu) {
        std.log.err("-Ddevice={t} needs -Dopenvino: only the CPU provider is built without it", .{device});
        std.process.exit(1);
    }

    const ort = if (openvino) |prefix| b.dependency("onnxruntime", .{
        .target = target,
        .optimize = optimize,
        .openvino = prefix,
        .@"openvino-include" = openvino_include orelse b.pathJoin(&.{ prefix, "include" }),
    }) else b.dependency("onnxruntime", .{
        .target = target,
        .optimize = optimize,
    });

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
    // Linking the artifact carries its headers along, so `@cInclude`ing
    // "onnxruntime_c_api.h" needs no include path of our own.
    mod.linkLibrary(ort.artifact("onnxruntime"));

    const exe = b.addExecutable(.{
        .name = "sam3",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "sam3", .module = mod },
            },
        }),
    });
    b.installArtifact(exe);

    var provider_path: []const u8 = "";
    var cache_path: []const u8 = "";
    if (openvino != null) {
        // Compiling the vision encoder for an NPU takes minutes. The provider
        // keeps the result here, so only the first run pays for it -- and it is
        // derived data, so it belongs beside everything else the build cached.
        cache_path = b.cache_root.join(b.allocator, &.{ "sam3", "openvino-cache" }) catch @panic("OOM");

        // An OpenVINO build runs on the host's libstdc++, and Zig does not
        // carry a static library's link objects across to whoever links it.
        onnxruntime.linkStdCxx(b, mod);
        onnxruntime.linkStdCxx(b, exe.root_module);

        // `libonnxruntime_providers_shared.so` is one global pointer and
        // nothing else. The runtime dlopens it by name, out of the directory
        // the running binary sits in -- hence `.bin` rather than the usual
        // `lib` -- and leaves a vtable of its internals there for the provider
        // to pick back up.
        b.getInstallStep().dependOn(&b.addInstallArtifact(
            ort.artifact("onnxruntime_providers_shared"),
            .{ .dest_dir = .{ .override = .bin } },
        ).step);

        // The provider itself the runtime does not link at all: it dlopens it
        // by path when asked, so all the executable needs is to know where it
        // was installed.
        const provider = b.addInstallArtifact(ort.artifact("onnxruntime_providers_openvino"), .{});
        b.getInstallStep().dependOn(&provider.step);
        provider_path = b.getInstallPath(.lib, "libonnxruntime_providers_openvino.so");
    }

    // --- Asset fetching -----------------------------------------------------
    //
    // `zig build fetch-weights` and `zig build fetch-examples` download what the
    // model needs through a small Zig downloader, so a checkout needs no curl,
    // no shell and no image tooling. Both steps are idempotent: a file that is
    // already present and matches its published SHA-256 is left alone.
    //
    // The exports are not `build.zig.zon` dependencies on purpose. Together
    // they are 1.8 GiB, and the package manager would have to hash them into
    // the global cache before the build could look at them.

    const fetch_exe = b.addExecutable(.{
        .name = "fetch",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/fetch.zig"),
            .target = b.graph.host,
            .optimize = .ReleaseSafe,
        }),
    });

    const hf_repo = b.option(
        []const u8,
        "hf-repo",
        "Hugging Face repo to pull the SAM 3 ONNX export from",
    ) orelse "onnx-community/sam3-tracker-ONNX";

    const weights_step = b.step("fetch-weights", "Download the SAM 3 tracker ONNX export");

    // Everything the build downloads is derived data, not source, so it goes to
    // the build cache rather than the working tree, and the executable is told
    // where it landed through `build_options` -- the program itself takes no
    // arguments. Stable names (rather than content-hashed step outputs) are
    // what make re-running a fetch cheap.
    //
    // Each graph is a pair: a small `.onnx` holding the structure, and an
    // `.onnx_data` holding the weights, which the first names by a path
    // relative to its own directory. They have to land side by side.
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

    // The four sample images the SAM 3 playground offers, as PNG (the CLI
    // decodes PNG and PPM natively). No checksums: Unsplash re-encodes on
    // demand, so pinning bytes would be a false promise.
    const examples_step = b.step("fetch-examples", "Download the playground sample images");

    var cat_path: []const u8 = "";
    for ([_][2][]const u8{
        .{ "dog", "photo-1543466835-00a7907e9de1" },
        .{ "cat", "photo-1514888286974-6c03e2ca1dba" },
        .{ "person", "photo-1507003211169-0a1dd7228f2d" },
        .{ "car", "photo-1459603677915-a62079ffd002" },
    }) |example| {
        const out = b.cache_root.join(b.allocator, &.{
            "sam3",
            "examples",
            b.fmt("{s}.png", .{example[0]}),
        }) catch @panic("OOM");
        if (std.mem.eql(u8, example[0], "cat")) cat_path = out;

        const run = b.addRunArtifact(fetch_exe);
        run.has_side_effects = true;
        run.setCwd(b.path("."));
        run.addArgs(&.{
            "--url",
            b.fmt("https://images.unsplash.com/{s}?w=800&fm=png", .{example[1]}),
            "--out",
            out,
            "--label",
            b.fmt("{s}.png", .{example[0]}),
        });
        examples_step.dependOn(&run.step);
    }

    // Where those two downloads ended up, and which device to ask for, as
    // compile-time constants in the executable. This is the only channel:
    // `sam3` parses no arguments, so the build steps that fetch the model and
    // the image are also what say where they are.
    const exe_options = b.addOptions();
    exe_options.addOption([]const u8, "vision_encoder_path", vision_path);
    exe_options.addOption([]const u8, "decoder_path", decoder_path);
    exe_options.addOption([]const u8, "openvino_provider_path", provider_path);
    exe_options.addOption([]const u8, "openvino_cache_path", cache_path);
    exe_options.addOption([]const u8, "image_path", cat_path);
    exe_options.addOption([]const u8, "device", @tagName(device));
    exe_options.addOption(bool, "untested_npu", untested_npu);
    exe.root_module.addOptions("build_options", exe_options);

    const run_step = b.step("run", "Segment the sample cat image with the real checkpoint");
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);

    // By making the run step depend on the default step, it will be run from the
    // installation directory rather than directly from within the cache
    // directory -- which is also where the runtime looks for
    // `libonnxruntime_providers_shared.so`.
    run_cmd.step.dependOn(b.getInstallStep());
    // The app takes no arguments: it reads the model and the sample image from
    // fixed paths under the build cache, so both fetch steps have to have run
    // first, and the working directory has to be the project root whatever
    // directory `zig build` itself was invoked from.
    run_cmd.step.dependOn(weights_step);
    run_cmd.step.dependOn(examples_step);
    run_cmd.setCwd(b.path("."));

    const mod_tests = b.addTest(.{ .root_module = mod });
    const run_mod_tests = b.addRunArtifact(mod_tests);

    const exe_tests = b.addTest(.{ .root_module = exe.root_module });
    const run_exe_tests = b.addRunArtifact(exe_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);
}
