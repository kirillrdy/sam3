# SAM 3 in Zig

Native SAM 3 image segmentation and text lookup with CUDA, Intel ARC, OpenCL, and Metal
backends. Model graphs execute through the repository's Zig ONNX runtime.
The SAM 3 library runs inference directly; applications can add their own caching, including zimo, around calls to the library.

- support CUDA, Intel Arc, OpenCL, Metal
- zero dependencies ( except for metal backend )
- native Zig PTX output without cuda toolchain
- fast compilation
- small binaries
- target older GPUs  `-Dsm=sm_61`

## Use as a Zig library

Add this package as a Zig dependency and import its `sam3` module:

```zig
const sam3_dep = b.dependency("sam3", .{ .target = target, .optimize = optimize });
exe.root_module.addImport("sam3", sam3_dep.module("sam3"));
```

`Model.open` downloads missing default model files into the cache and opens them.

```zig
var model = try sam3.Model.open(allocator, io);
defer model.deinit();

// rgb is borrowed, tightly packed RGB24 data: width * height * 3 bytes.
const image: sam3.RgbImage = .{ .pixels = rgb, .width = width, .height = height };
var embedding = try model.encodePoints(image);
defer embedding.deinit();
var masks = try model.segment(&embedding, &.{.{ .x = 0.5, .y = 0.5, .label = .positive }});
defer masks.deinit();
const first_mask = masks.plane(0); // Borrowed logits at masks.width x masks.height.
```

For text lookup, call `encodeForText(image)` once, then `find(&embedding, phrase,
.{ .min_score = 0.5 })` for each phrase. Embeddings and masks own their memory;
call `deinit` on each. The image only needs to remain valid during its encode
call. Serialize inference calls when sharing a model between threads.

## Run the web UI

```sh
zig build run --release=fast -Dbackend=cuda
```

Then open <http://127.0.0.1:3000/>.

## Run the native macOS UI

On macOS, you can run the native Cocoa/AppKit UI directly:

```sh
zig build run-macos --release=fast
```

The native macOS app provides the same capabilities as the web UI without needing a browser:
- Native AppKit window with dark mode theme
- Open image dialog and drag-and-drop file loading
- Interactive point-based segmentation (clicks add to or cut from the mask)
- Concept text search ("Find by word")
- Candidate mask selection with score and frame coverage statistics
- In-process Metal GPU inference with asynchronous background execution

## Run the native Linux UI (Wayland)

On Linux under Wayland, run the native desktop UI directly:

```sh
zig build run-linux --release=fast -Dbackend=cuda
```

Features:
- Pure Zig implementation of the Wayland wire protocol (`wl_shm`, `xdg_wm_base`, `wl_seat`, `wl_pointer`, `wl_keyboard`) over UNIX domain sockets with zero external C library dependencies (no `libwayland-client`).
- Embedded bitmap font and software rasterizer for fast, lightweight rendering.
- Interactive point segmentation: left click to add positive points, right click to cut (negative points).
- Concept text search ("Find by word") with direct keyboard typing.
- Candidate mask selection with score and coverage statistics.
- Asynchronous background inference with thread-safe UI updates.
- Open Image browser for loading images from disk. Select a file in the browser, or click the path field and type an absolute path. Use Parent, Up, and Down to navigate; Escape closes the browser.


Model downloads use `curl`.

The CUDA backend runs pure native Zig + PTX kernels (including double-buffered
TF32 Tensor Core MMA and fused bias/GELU epilogues on Ampere+, with synchronous staging fallbacks for earlier architectures) directly on the CUDA driver API without linking or requiring cuBLAS.
