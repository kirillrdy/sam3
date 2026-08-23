//! A prompt click, on its own so that both halves of the program can name one.
//!
//! The model driver needs it because it is what the decoder is asked about, and
//! the drawing code needs it because it is what the marker is drawn at -- and
//! the drawing code also runs in the browser, where `sam3.zig` cannot follow:
//! that file reaches ONNX Runtime through C headers there is no wasm build of.

/// A click, in coordinates normalised to the frame. `label` is 1 for a point
/// the mask should include and 0 for one it should exclude.
pub const Point = struct {
    x: f32,
    y: f32,
    label: i64 = 1,
};
