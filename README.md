# SAM 3 in Zig

Native SAM 3 image segmentation and text lookup with CUDA, Intel ARC, OpenCL, and Metal
backends. Model graphs execute through the repository's Zig ONNX runtime.
The SAM 3 library runs inference directly. Applications can cache the owned
`encodeTextFeatures` result and pass it to `findWithTextFeatures`.

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

## Native apps

The macOS and Linux desktop apps live in `../playground/sam3-apps` and share one
`build.zig`. See that project's README for build and run commands.

Model downloads use `curl`.

The CUDA backend runs pure native Zig + PTX kernels (including double-buffered
TF32 Tensor Core MMA and fused bias/GELU epilogues on Ampere+, with synchronous staging fallbacks for earlier architectures) directly on the CUDA driver API without linking or requiring cuBLAS.
