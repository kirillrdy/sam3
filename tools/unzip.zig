const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, gpa);
    defer args.deinit();
    _ = args.skip();

    const archive_path = args.next() orelse return usage();
    const destination_path = args.next() orelse return usage();
    if (args.next() != null) return usage();

    var archive = try std.Io.Dir.cwd().openFile(io, archive_path, .{});
    defer archive.close(io);
    var archive_buffer: [64 * 1024]u8 = undefined;
    var archive_reader = archive.reader(io, &archive_buffer);

    var destination = try std.Io.Dir.cwd().createDirPathOpen(io, destination_path, .{});
    defer destination.close(io);

    try std.zip.extract(destination, &archive_reader, .{});
}

fn usage() error{InvalidArguments} {
    std.debug.print("usage: unzip <archive.zip> <destination>\n", .{});
    return error.InvalidArguments;
}
