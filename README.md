# SAM 3 in Zig

Native SAM 3 image segmentation and text lookup with CUDA, Intel ARC, OpenCL, and Metal
backends. Model graphs execute through the repository's Zig ONNX runtime.

- support CUDA, Intel Arc, OpenCL, Metal
- zero dependencies ( except for metal backend )
- native Zig PTX output without cuda toolchain
- fast compilation
- small binaries
- target older GPUs  `-Dsm=sm_61`

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

Model downloads use `curl` by default. If `curl` is not available, build with
`-Dzig-http=true` to use Zig's built-in HTTP client instead.

The CUDA backend runs pure native Zig + PTX kernels (including double-buffered
TF32 Tensor Core MMA and fused bias/GELU epilogues on Ampere+, with synchronous staging fallbacks for earlier architectures) directly on the CUDA driver API without linking or requiring cuBLAS.

