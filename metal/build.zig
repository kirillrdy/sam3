const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mod = b.addModule("metal", .{
        .root_source_file = b.path("src/metal.zig"),
        .target = target,
        .optimize = optimize,
    });
    mod.addIncludePath(b.path("src"));
    mod.addCSourceFile(.{ .file = b.path("src/bridge.m"), .flags = &.{"-fobjc-arc"} });
    mod.link_libc = true;
    mod.linkSystemLibrary("objc", .{});
    mod.linkFramework("Foundation", .{});
    mod.linkFramework("Metal", .{});
}
