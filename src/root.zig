//! SAM 3, running on ONNX Runtime.
//!
//! `onnx` is the runtime binding and knows nothing about this model; `sam3`
//! drives Meta's two exported graphs through it; the rest is image IO and the
//! command the executable runs.

const std = @import("std");

pub const onnx = @import("onnx.zig");
pub const sam3 = @import("sam3.zig");
pub const resample = @import("resample.zig");
pub const image = @import("io/image.zig");
pub const visualization = @import("io/visualization.zig");
pub const cli = @import("cli/cli.zig");

pub const Model = sam3.Model;
pub const Masks = sam3.Masks;
pub const Paths = sam3.Paths;
pub const Target = sam3.Target;
pub const Point = sam3.Point;
pub const DeviceKind = onnx.DeviceKind;
pub const ImageRGB = image.ImageRGB;
pub const RGB = image.RGB;

test {
    std.testing.refAllDecls(@This());
}
