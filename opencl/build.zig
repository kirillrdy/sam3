const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mod = b.addModule("opencl", .{
        .root_source_file = b.path("src/opencl.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    _ = mod;
}
