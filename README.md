# SAM 3 in Zig

Native SAM 3 image segmentation, text lookup, and video object tracking with CUDA,
Intel ARC, OpenCL, and Metal backends. Model graphs execute through the
repository's Zig ONNX runtime.

- support CUDA, Intel Arc, OpenCL, Metal
- zero dependencies ( except for metal backend )
- native Zig PTX output without cuda toolchain
- fast compilation
- small binaries
- target older GPUs  `-Dsm=sm_61`

## Run the web UI

```sh
zig build run --release=fast -Ddevice=cuda
```

Then open <http://127.0.0.1:3000/>.

Model downloads use `curl` by default. If `curl` is not available, build with
`-Dzig-http=true` to use Zig's built-in HTTP client instead.

## Tracking objects through a video

Open a video and either click an object or type what to look for, then press
**Follow through the video**. Each object is drawn in its own colour and keeps
its number for as long as it is on screen.

* **A click** names one object. The mask it leaves on each frame says where to
  prompt the next one, so the object carries itself forward. Click again to
  sharpen the same object -- a first click often catches a whisker rather than
  the cat -- or press **Follow another object** and click to add a second one.
* **A phrase** re-finds every object it names on every frame, and the frames are
  tied together by matching this frame's masks to last frame's tracks. Objects
  that come and go are picked up and dropped as they do.

Every frame is followed. The page reads the video's own frame rate -- by
watching how far presentation time moves between the frames the browser hands
over, falling back to 30 fps where it will not say -- and steps the video by
exactly that, sending one frame at a time to `POST /track`.

Every frame also costs a pass of the vision encoder: roughly 0.9 s by click and
1.9 s by phrase on an M-series Mac. A run over a whole clip therefore takes
considerably longer than the clip does, so it reports how much of it is left as
it goes and stops when told to. Following a passage of a clip rather than the
whole of it is what the scrubber is for: it parks the video anywhere before a
run starts, which is also how a clip that opens on a title card or a fade gets
prompted on a frame that has something on it.

Frame by frame is also what carries a clicked object most reliably. It is found
again each frame by prompting where it was on the frame before, so what matters
is how far it moves between the frames it is followed on, measured against its
own size: around a tenth of its width is followed comfortably, a fifth starts to
lose it, because the prompt lands on the background and the decoder answers,
quite correctly, with the background. At a video's own frame rate ordinary
motion is nowhere near that.

### What this is not

SAM 3 follows a video with a memory bank -- a memory encoder folds each frame's
mask back into features, and memory attention lets the next frame read them. The
ONNX exports published for the tracker are the vision encoder and the prompt
encoder / mask decoder only; neither memory graph is among them. So the temporal
part is not the model's, and lives in `src/track.zig` instead: per-frame
segmentation, stitched into tracks that keep their identity, survive the frames
they are missed on, and are forgotten once they are gone.

What that costs is the memory bank's recall. An object that leaves the frame and
comes back is a new object here, and appearance is only ever carried one frame at
a time. If the exports ever grow the two memory graphs, this is where they go.

The CUDA backend runs pure native Zig + PTX kernels (including double-buffered
TF32 Tensor Core MMA and fused bias/GELU epilogues on Ampere+, with synchronous staging fallbacks for earlier architectures) directly on the CUDA driver API without linking or requiring cuBLAS.
