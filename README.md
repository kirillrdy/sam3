# SAM 3 in Zig

Native SAM 3 image segmentation and text lookup with CUDA, Intel ARC, OpenCL, and Metal
backends. Model graphs execute through the repository's Zig ONNX runtime.

## Run the web UI

On NVIDIA GPUs (defaults to Ada `sm_89`):

```sh
zig build run --release=fast -Ddevice=cuda
```

For older GPUs (e.g. GTX 1080 / Pascal `sm_61`), set the compute capability with `-Dsm`:

```sh
zig build run --release=fast -Ddevice=cuda -Dsm=sm_61
```

Then open <http://127.0.0.1:3000/>.

Model downloads use `curl` by default. If `curl` is not available, build with
`-Dzig-http=true` to use Zig's built-in HTTP client instead.

The CUDA backend runs pure native Zig + PTX kernels (including double-buffered
TF32 Tensor Core MMA and fused bias/GELU epilogues on Ampere+, with synchronous staging fallbacks for earlier architectures) directly on the CUDA driver API without linking or requiring cuBLAS.
