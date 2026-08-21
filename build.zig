const std = @import("std");

// Although this function looks imperative, it does not perform the build
// directly and instead it mutates the build graph (`b`) that will be then
// executed by an external runner. The functions in `std.Build` implement a DSL
// for defining build steps and express dependencies between them, allowing the
// build runner to parallelize the build automatically (and the cache system to
// know when a step doesn't need to be re-run).
pub fn build(b: *std.Build) void {
    // Standard target options allow the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});
    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    // ReleaseFast by default: this runs an 859M-parameter model, and a Debug
    // build of the same pass is over ten times slower - slow enough that the
    // default build would look broken rather than merely unoptimised.
    // `-Doptimize=Debug` still works for actually debugging. This is spelled out
    // rather than using `preferred_optimize_mode`, which only takes effect under
    // `-Drelease` and drops the `-Doptimize` flag entirely.
    const optimize = b.option(
        std.builtin.OptimizeMode,
        "optimize",
        "Prioritize performance, safety, or binary size (default: ReleaseFast)",
    ) orelse .ReleaseFast;
    // It's also possible to define more custom flags to toggle optional features
    // of this build script using `b.option()`. All defined flags (including
    // target and optimize options) will be listed when running `zig build --help`
    // in this directory.

    // This creates a module, which represents a collection of source files alongside
    // some compilation options, such as optimization mode and linked system libraries.
    // Zig modules are the preferred way of making Zig code available to consumers.
    // addModule defines a module that we intend to make available for importing
    // to our consumers. We must give it a name because a Zig package can expose
    // multiple modules and consumers will need to be able to specify which
    // module they want to access.
    const zigimg = b.dependency("zigimg", .{
        .target = target,
        .optimize = optimize,
    });

    const mod = b.addModule("sam3", .{
        // The root source file is the "entry point" of this module. Users of
        // this module will only be able to access public declarations contained
        // in this file, which means that if you have declarations that you
        // intend to expose to consumers that were defined in other files part
        // of this module, you will have to make sure to re-export them from
        // the root file.
        .root_source_file = b.path("src/root.zig"),
        // Later on we'll use this module as the root module of a test executable
        // which requires us to specify a target.
        .target = target,
        .imports = &.{
            .{ .name = "zigimg", .module = zigimg.module("zigimg") },
        },
    });

    // Here we define an executable. An executable needs to have a root module
    // which needs to expose a `main` function. While we could add a main function
    // to the module defined above, it's sometimes preferable to split business
    // logic and the CLI into two separate modules.
    //
    // If your goal is to create a Zig library for others to use, consider if
    // it might benefit from also exposing a CLI tool. A parser library for a
    // data serialization format could also bundle a CLI syntax checker, for example.
    //
    // If instead your goal is to create an executable, consider if users might
    // be interested in also being able to embed the core functionality of your
    // program in their own executable in order to avoid the overhead involved in
    // subprocessing your CLI tool.
    //
    // If neither case applies to you, feel free to delete the declaration you
    // don't need and to put everything under a single module.
    const exe = b.addExecutable(.{
        .name = "sam3",
        .root_module = b.createModule(.{
            // b.createModule defines a new module just like b.addModule but,
            // unlike b.addModule, it does not expose the module to consumers of
            // this package, which is why in this case we don't have to give it a name.
            .root_source_file = b.path("src/main.zig"),
            // Target and optimization levels must be explicitly wired in when
            // defining an executable or library (in the root module), and you
            // can also hardcode a specific target for an executable or library
            // definition if desireable (e.g. firmware for embedded devices).
            .target = target,
            .optimize = optimize,
            // List of modules available for import in source files part of the
            // root module.
            .imports = &.{
                // Here "sam3" is the name you will use in your source code to
                // import this module (e.g. `@import("sam3")`). The name is
                // repeated because you are allowed to rename your imports, which
                // can be extremely useful in case of collisions (which can happen
                // importing modules from different packages).
                .{ .name = "sam3", .module = mod },
            },
        }),
    });

    // This declares intent for the executable to be installed into the
    // install prefix when running `zig build` (i.e. when executing the default
    // step). By default the install prefix is `zig-out/` but can be overridden
    // by passing `--prefix` or `-p`.
    b.installArtifact(exe);

    // --- Asset fetching -----------------------------------------------------
    //
    // `zig build fetch-weights` and `zig build fetch-examples` download what the
    // model needs through a small Zig downloader, so a checkout needs no curl,
    // no shell and no image tooling. Both steps are idempotent: a file that is
    // already present and matches its published SHA-256 is left alone.
    //
    // The checkpoint is not a `build.zig.zon` dependency on purpose. It is
    // 3.2 GiB, `facebook/sam3` is gated behind a manual approval form, and the
    // package manager cannot send the `Authorization` header that unlocks it —
    // so the download is verified against Meta's published SHA-256 instead of a
    // package hash.

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
        "Hugging Face repo to pull SAM 3 from (default: a public mirror; set HF_TOKEN and pass facebook/sam3 for the official one)",
    ) orelse "jetjodh/sam3";

    const weights_step = b.step("fetch-weights", "Download Meta's SAM 3 checkpoint and tokenizer assets");

    // Everything the build downloads is derived data, not source, so it goes to
    // the build cache rather than the working tree, and the executable is told
    // where it landed through `build_options` — the program itself takes no
    // arguments. Stable names (rather than content-hashed step outputs) are what
    // make re-running a fetch cheap: a file that is already there and matches
    // its published SHA-256 is left alone.
    const checkpoint_path = b.cache_root.join(b.allocator, &.{ "sam3", "sam3.safetensors" }) catch @panic("OOM");

    const checkpoint = b.addRunArtifact(fetch_exe);
    checkpoint.has_side_effects = true;
    checkpoint.setCwd(b.path("."));
    checkpoint.addArgs(&.{
        "--url",
        b.fmt("https://huggingface.co/{s}/resolve/main/model.safetensors", .{hf_repo}),
        "--out",
        checkpoint_path,
        // Meta's published checksum for the 859.9M-parameter F32 checkpoint.
        "--sha256",
        "6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a",
        "--token-env",
        "HF_TOKEN",
        "--label",
        "sam3.safetensors (3.2 GiB)",
    });
    weights_step.dependOn(&checkpoint.step);

    for ([_][]const u8{
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "LICENSE",
    }) |asset| {
        const run = b.addRunArtifact(fetch_exe);
        run.has_side_effects = true;
        run.setCwd(b.path("."));
        run.addArgs(&.{
            "--url",
            b.fmt("https://huggingface.co/{s}/resolve/main/{s}", .{ hf_repo, asset }),
            "--out",
            b.cache_root.join(b.allocator, &.{ "sam3", asset }) catch @panic("OOM"),
            "--token-env",
            "HF_TOKEN",
            "--label",
            asset,
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

    // Where those two downloads ended up, as compile-time constants in the
    // executable. This is the only channel: `sam3` parses no arguments, so the
    // build steps that fetch the checkpoint and the image are also what say
    // where they are.
    const exe_options = b.addOptions();
    exe_options.addOption([]const u8, "weights_path", checkpoint_path);
    exe_options.addOption([]const u8, "image_path", cat_path);
    exe.root_module.addOptions("build_options", exe_options);

    // This creates a top level step. Top level steps have a name and can be
    // invoked by name when running `zig build` (e.g. `zig build run`).
    // This will evaluate the `run` step rather than the default step.
    // For a top level step to actually do something, it must depend on other
    // steps (e.g. a Run step, as we will see in a moment).
    const run_step = b.step("run", "Segment the sample cat image with the real checkpoint");

    // This creates a RunArtifact step in the build graph. A RunArtifact step
    // invokes an executable compiled by Zig. Steps will only be executed by the
    // runner if invoked directly by the user (in the case of top level steps)
    // or if another step depends on it, so it's up to you to define when and
    // how this Run step will be executed. In our case we want to run it when
    // the user runs `zig build run`, so we create a dependency link.
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);

    // By making the run step depend on the default step, it will be run from the
    // installation directory rather than directly from within the cache directory.
    run_cmd.step.dependOn(b.getInstallStep());
    // The app takes no arguments: it reads the checkpoint and the sample image
    // from fixed paths relative to the project root, so both fetch steps have to
    // have run first, and the working directory has to be that root whatever
    // directory `zig build` itself was invoked from.
    run_cmd.step.dependOn(weights_step);
    run_cmd.step.dependOn(examples_step);
    run_cmd.setCwd(b.path("."));

    // Creates an executable that will run `test` blocks from the provided module.
    const mod_tests = b.addTest(.{
        .root_module = mod,
    });

    // A run step that will run the test executable.
    const run_mod_tests = b.addRunArtifact(mod_tests);

    // Creates an executable that will run `test` blocks from the executable's
    // root module.
    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
    });

    // A run step that will run the second test executable.
    const run_exe_tests = b.addRunArtifact(exe_tests);

    // A top level step for running all tests.
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);

    // Just like flags, top level steps are also listed in the `--help` menu.
    //
    // The Zig build system is entirely implemented in userland, which means
    // that it cannot hook into private compiler APIs. All compilation work
    // orchestrated by the build system will result in other Zig compiler
    // subcommands being invoked with the right flags defined. You can observe
    // these invocations when one fails (or you pass a flag to increase
    // verbosity) to validate assumptions and diagnose problems.
    //
    // Lastly, the Zig build system is relatively simple and self-contained,
    // and reading its source code will allow you to master it.
}
