//! SAM 3 on ONNX Runtime.
//!
//! Two programs over one model: `zig build run` segments the sample image from
//! the command line, and `zig build serve` puts the same thing behind a
//! browser, where clicking the picture is the prompt.
//!
//! The runtime is built from source by the `onnxruntime` package next door, so
//! there is nothing to install for the default build. Asking for a device the
//! CPU provider cannot serve additionally builds that package's OpenVINO
//! execution provider, which is how the runtime reaches an Intel NPU:
//!
//!     zig build run -Ddevice=npu
//!
//! The OpenVINO behind that is Intel's own release, which the `onnxruntime`
//! package downloads and unpacks on its own -- where it puts it, and why it
//! cannot be an ordinary package dependency, are questions for that package.
//! Nothing here knows more than which of the two to ask for.
//!
//! `-Dopenvino=<prefix>` builds against an installed one instead, and implies
//! `-Ddevice=npu`, so naming one is the whole of it. `-Dopenvino-lib` and
//! `-Dopenvino-include` say where inside that prefix to look when its layout is
//! not the usual one; both are passed straight through.
//!
//! Either way the build links that OpenVINO and the host's libstdc++ rather than
//! Zig's libc++ -- the provider and the runtime hand each other C++ objects
//! across a dlopen boundary, so both have to match the ABI of the prebuilt
//! `libopenvino.so`. See the `onnxruntime` package for what that costs.
//!
//! At run time the NPU plugin reaches the device through the Level Zero loader,
//! which it dlopens by name, and the loader in turn dlopens the NPU driver. Both
//! `libze_loader.so.1` and `libze_intel_npu.so.1` have to be on the library path,
//! and on NixOS neither is anywhere the dynamic loader looks by itself. `run` and
//! `serve` go and find them -- see `npu_library_dirs` -- so neither step needs an
//! `LD_LIBRARY_PATH` of its own; `-Dnpu-library-path=<dir>` says where when the
//! search comes up short. A binary run straight out of `zig-out/bin` is on its
//! own, and does still need the variable set.

const std = @import("std");
const onnxruntime = @import("onnxruntime");

/// Where a session is asked to run. Not a promise: a graph a device turns down
/// -- too large for the NPU, an operator the provider does not implement --
/// falls back to the CPU at run time and says so.
const Device = enum { npu, gpu, cpu };

/// What the NPU stack is dlopened by name, and so has to be found by name.
///
/// Neither of these is linked by anything the build produces: the NPU plugin
/// dlopens the Level Zero loader, and the loader dlopens the driver behind it.
/// Nothing in that chain consults an rpath of ours, which leaves
/// `LD_LIBRARY_PATH` as the only channel -- so `run` and `serve` set it.
const npu_libraries = [_][]const u8{ "libze_loader.so.1", "libze_intel_npu.so.1" };

/// Where to look for them, in order, after anything `-Dnpu-library-path` gave.
///
/// The two are in different trees on NixOS -- the loader is an ordinary
/// package, the driver is a hardware driver -- and neither tree is anywhere the
/// dynamic loader looks by itself. Distributions that put both somewhere on the
/// default path need none of this, and searching for them there costs a stat.
const npu_library_dirs = [_][]const u8{
    "/run/opengl-driver/lib",
    "/run/current-system/sw/lib",
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib64",
    "/usr/lib",
};

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

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
    const openvino_lib = b.option(
        []const u8,
        "openvino-lib",
        "Directory holding libopenvino.so and the plugins, if not <prefix>/lib",
    );
    const device = b.option(
        Device,
        "device",
        "Which device to run the model on (default: cpu, or npu when -Dopenvino names one). Asking for anything but the CPU is what builds the OpenVINO execution provider",
    ) orelse if (openvino != null) Device.npu else Device.cpu;

    const untested_npu = b.option(
        bool,
        "untested-npu",
        "Use the NPU even on a generation this has not been run on (default: false)",
    ) orelse false;

    const npu_library_path = b.option(
        []const []const u8,
        "npu-library-path",
        "Directory holding the Level Zero loader or the NPU driver, for when `run` and `serve` do not find them themselves",
    ) orelse &.{};

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

    // Whether the provider is built at all, and against what. Naming an
    // installed OpenVINO is always honoured; otherwise asking for a device the
    // CPU provider cannot serve is what calls for the one the package next door
    // downloads. Where that lands, and what it costs to reach, are its business
    // rather than ours -- all this says is which of the two it is.
    const with_openvino = openvino != null or device != .cpu;

    const ort = if (!with_openvino) b.dependency("onnxruntime", .{
        .target = target,
        .optimize = optimize,
    }) else if (openvino) |prefix| b.dependency("onnxruntime", .{
        .target = target,
        .optimize = optimize,
        .openvino = prefix,
        .@"openvino-include" = openvino_include orelse b.pathJoin(&.{ prefix, "include" }),
        .@"openvino-lib" = openvino_lib orelse b.pathJoin(&.{ prefix, "lib" }),
    }) else b.dependency("onnxruntime", .{
        .target = target,
        .optimize = optimize,
        .@"openvino-fetch" = true,
    });

    // A downloaded OpenVINO sits in the build cache with its own oneTBB beside
    // it, and none of that is anywhere the dynamic loader looks, so `run` and
    // `serve` are told where it went. An installed one is the system's problem.
    const openvino_runtime_paths: []const []const u8 =
        if (with_openvino and openvino == null) onnxruntime.openvinoRuntimeLibraryPaths(b) else &.{};

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
            .strip = if (optimize == .ReleaseFast) true else null,
            .imports = &.{
                .{ .name = "sam3", .module = mod },
            },
        }),
    });
    b.installArtifact(exe);

    var provider_path: []const u8 = "";
    var cache_path: []const u8 = "";
    if (with_openvino) {
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
    // neither executable parses arguments, so the build steps that fetch the
    // model and the image are also what say where they are, and the two share
    // the table -- each reads the entries it has a use for.
    const exe_options = b.addOptions();
    exe_options.addOption([]const u8, "vision_encoder_path", vision_path);
    exe_options.addOption([]const u8, "decoder_path", decoder_path);
    exe_options.addOption([]const u8, "openvino_provider_path", provider_path);
    exe_options.addOption([]const u8, "openvino_cache_path", cache_path);
    exe_options.addOption([]const u8, "image_path", cat_path);
    exe_options.addOption([]const u8, "device", @tagName(device));
    exe_options.addOption(bool, "untested_npu", untested_npu);
    exe_options.addOption([]const u8, "host", host);
    exe_options.addOption(u16, "port", port);
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
    if (with_openvino) addNpuLibraryPath(b, run_cmd, npu_library_path, openvino_runtime_paths);

    // --- Web UI -------------------------------------------------------------
    //
    // `zig build serve` puts the same two graphs behind a browser. The client
    // half is Zig as well: `src/web/client.zig` is compiled to wasm, and it and
    // the page that loads it are embedded in the server, so the executable is
    // the whole of the UI -- there is no directory to serve and nothing to
    // install beside it.

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
    // A module the page instantiates rather than a program: there is no `main`
    // to enter through, and the exports are the only thing keeping the code
    // they reach alive through the linker's garbage collection.
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
    // What `@embedFile("client_wasm")` picks up, which is also what makes the
    // server wait for the wasm build.
    web_exe.root_module.addAnonymousImport("client_wasm", .{
        .root_source_file = client.getEmittedBin(),
    });
    web_exe.root_module.addOptions("build_options", exe_options);
    if (with_openvino) onnxruntime.linkStdCxx(b, web_exe.root_module);
    b.installArtifact(web_exe);

    const serve_step = b.step("serve", b.fmt("Serve the web UI on http://{s}:{d}/", .{ host, port }));
    const serve_cmd = b.addRunArtifact(web_exe);
    serve_step.dependOn(&serve_cmd.step);

    // Same three reasons as `run`: the installed copy is what sits next to
    // `libonnxruntime_providers_shared.so`, the model has to have been fetched,
    // and the sample image is read from the build cache at a path relative to
    // the project root.
    serve_cmd.step.dependOn(b.getInstallStep());
    serve_cmd.step.dependOn(weights_step);
    serve_cmd.step.dependOn(examples_step);
    serve_cmd.setCwd(b.path("."));
    if (with_openvino) addNpuLibraryPath(b, serve_cmd, npu_library_path, openvino_runtime_paths);

    const mod_tests = b.addTest(.{ .root_module = mod });
    const run_mod_tests = b.addRunArtifact(mod_tests);

    const exe_tests = b.addTest(.{ .root_module = exe.root_module });
    const run_exe_tests = b.addRunArtifact(exe_tests);

    // The wasm client, on the host. Its exports are ordinary functions and the
    // allocator behind them is `page_allocator` either way, so the sequence the
    // page drives can be run here -- which is the only way this file is tested,
    // since a wasm module needs a browser to be run in.
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
    test_step.dependOn(&run_exe_tests.step);
    test_step.dependOn(&run_client_tests.step);
}

/// Point a run step's `LD_LIBRARY_PATH` at the NPU stack.
///
/// Each of `npu_libraries` is looked for in `extra` and then in
/// `npu_library_dirs`, and the directory that has it goes on the path. Only
/// directories that turned something up are added, so on a machine where the
/// dynamic loader finds them by itself this adds nothing.
///
/// Searching the build machine is sound in a way that baking a path into the
/// executable would not be: this only ever applies to the build's own `run` and
/// `serve` steps, which are the same machine by definition.
fn addNpuLibraryPath(
    b: *std.Build,
    run: *std.Build.Step.Run,
    extra: []const []const u8,
    openvino: []const []const u8,
) void {
    var dirs: std.ArrayList([]const u8) = .empty;
    defer dirs.deinit(b.allocator);

    // Whatever a downloaded OpenVINO needs goes on the path unconditionally:
    // the package next door named those directories rather than guessing at
    // them, so there is nothing to search for and nothing to leave out.
    dirs.appendSlice(b.allocator, openvino) catch @panic("OOM");

    for (npu_libraries) |library| {
        var found = false;
        for (extra) |dir| found = found or hasLibrary(b, dir, library);
        if (found) continue;
        for (npu_library_dirs) |dir| {
            if (!hasLibrary(b, dir, library)) continue;
            for (dirs.items) |seen| {
                if (std.mem.eql(u8, seen, dir)) break;
            } else dirs.append(b.allocator, dir) catch @panic("OOM");
            break;
        }
    }

    var path: std.ArrayList(u8) = .empty;
    defer path.deinit(b.allocator);

    // Whatever the build was started with comes first: a caller who set
    // `LD_LIBRARY_PATH` themselves meant it, and the run step inherits it.
    if (run.getEnvMap().get("LD_LIBRARY_PATH")) |inherited| {
        if (inherited.len != 0) path.appendSlice(b.allocator, inherited) catch @panic("OOM");
    }
    // Then `-Dnpu-library-path`, which was given by hand and so outranks the
    // search, and then whatever the search turned up.
    for ([_][]const []const u8{ extra, dirs.items }) |list| {
        for (list) |dir| {
            if (path.items.len != 0) path.append(b.allocator, ':') catch @panic("OOM");
            path.appendSlice(b.allocator, dir) catch @panic("OOM");
        }
    }
    if (path.items.len == 0) return;

    run.setEnvironmentVariable("LD_LIBRARY_PATH", path.items);
}

/// Whether `dir` holds `library`. A directory that is not there at all is the
/// ordinary case on any machine that is not the one this was written on, so it
/// is not worth distinguishing from one that is there without the library.
fn hasLibrary(b: *std.Build, dir: []const u8, library: []const u8) bool {
    const path = b.pathJoin(&.{ dir, library });
    std.Io.Dir.accessAbsolute(b.graph.io, path, .{}) catch return false;
    return true;
}
