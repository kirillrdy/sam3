const std = @import("std");
const ImageRGB = @import("image.zig").ImageRGB;
const RGB = @import("image.zig").RGB;
const Tensor = @import("../tensor/tensor.zig").Tensor;

pub const VideoFrame = struct {
    frame_idx: usize,
    image: ImageRGB,

    pub fn deinit(self: *VideoFrame) void {
        self.image.deinit();
    }
};

pub const VideoSequence = struct {
    frames: []VideoFrame,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *VideoSequence) void {
        for (self.frames) |*f| {
            f.deinit();
        }
        self.allocator.free(self.frames);
    }

    pub fn generateSyntheticVideo(
        allocator: std.mem.Allocator,
        num_frames: usize,
        width: usize,
        height: usize,
    ) !VideoSequence {
        var frame_list: std.ArrayList(VideoFrame) = .empty;
        errdefer {
            for (frame_list.items) |*f| f.deinit();
            frame_list.deinit(allocator);
        }

        for (0..num_frames) |i| {
            var img = try ImageRGB.init(allocator, width, height);

            // Dark background gradient
            for (0..height) |y| {
                for (0..width) |x| {
                    const bg_val = @as(u8, @intCast(20 + (x * 30 / width)));
                    img.setPixel(x, y, RGB{ .r = bg_val, .g = bg_val, .b = bg_val + 10 });
                }
            }

            // Object 1: Moving bright yellow circle
            const cx1: isize = @intCast(30 + i * (width - 60) / num_frames);
            const cy1: isize = @intCast(height / 3 + @as(usize, @intFromFloat(15.0 * @sin(@as(f32, @floatFromInt(i)) * 0.5))));
            const radius: isize = 16;

            var dy = -radius;
            while (dy <= radius) : (dy += 1) {
                var dx = -radius;
                while (dx <= radius) : (dx += 1) {
                    if (dx * dx + dy * dy <= radius * radius) {
                        const px = cx1 + dx;
                        const py = cy1 + dy;
                        if (px >= 0 and py >= 0 and px < @as(isize, @intCast(width)) and py < @as(isize, @intCast(height))) {
                            img.setPixel(@intCast(px), @intCast(py), RGB{ .r = 255, .g = 215, .b = 0 });
                        }
                    }
                }
            }

            // Object 2: Moving cyan square
            const cx2: usize = @intCast(width - 40 - i * (width - 80) / num_frames);
            const cy2: usize = height * 2 / 3;
            const size: usize = 24;

            for (0..size) |sy| {
                for (0..size) |sx| {
                    const px = cx2 + sx - size / 2;
                    const py = cy2 + sy - size / 2;
                    if (px < width and py < height) {
                        img.setPixel(px, py, RGB{ .r = 0, .g = 200, .b = 255 });
                    }
                }
            }

            try frame_list.append(allocator, VideoFrame{
                .frame_idx = i,
                .image = img,
            });
        }

        return VideoSequence{
            .frames = try frame_list.toOwnedSlice(allocator),
            .allocator = allocator,
        };
    }
};
