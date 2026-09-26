pub const Label = enum(i64) {
    negative = 0,
    positive = 1,
};

/// Coordinates are normalized to the image, from 0 to 1.
pub const Point = struct {
    x: f32,
    y: f32,
    label: Label = .positive,
};
