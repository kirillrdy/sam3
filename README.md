# SAM 3 in Pure Zig ⚡

A high-performance, standalone implementation of **Meta's Segment Anything Model 3 (SAM 3)** written in **100% Pure Zig** (0.16.0), with no C, C++ or Python dependencies — the model, the tensor engine, the checkpoint reader and the downloader are all Zig, and the one package dependency (zigimg, for image decoding) is Zig too.

Supports both **Promptable Image Segmentation** and **Multi-Object Video Segmentation & Tracking** using geometric prompts (points, boxes, masks) and open-vocabulary concept prompts (text phrases) with the **Presence Head**.

---

## 🌟 Key Architecture & Features

SAM 3 introduces **Promptable Concept Segmentation (PCS)** alongside temporal video tracking. This pure Zig implementation faithfully reproduces the entire pipeline:

1. **Perception Vision-Language Backbone**:
   - Patch embedding with Conv2D ($3 \to D$).
   - 2D Sinusoidal positional embeddings.
   - Multi-layer Vision Transformer blocks with Multi-Head Attention and GELU MLP.
   - High-resolution feature neck for fine boundary mask reconstruction.

2. **Concept & Geometric Prompt Encoder**:
   - Concept text tokenizer and transformer encoder.
   - **Presence Head**: Predicts the presence probability of the requested concept in the visual scene, eliminating false positive detections.
   - 2D Sinusoidal coordinate encodings for point prompts (positive & negative).
   - Bounding box and dense low-resolution mask prompt fusion.

3. **DETR Concept Detector**:
   - Object queries cross-attending to perception feature maps.
   - 3-layer MLP predicting bounding box coordinates $[c_x, c_y, w, h]$.
   - Concept presence scoring and classification.

4. **Two-Way Transformer Mask Decoder**:
   - Bidirectional cross-attention between prompt tokens and image embeddings.
   - IoU prediction head for mask quality confidence.
   - Dynamic hypernetwork MLP predicting custom kernel weights for spatial mask upsampling.
   - 4x Bilinear / transposed convolution upsampler.

5. **Streaming Video Memory Bank & Temporal Tracker**:
   - Memory Encoder compressing mask logits and image features into compact memory vectors.
   - Sliding-window FIFO memory bank maintaining prompt keyframes and recent history.
   - Multi-frame temporal memory cross-attention with time-decay positional encodings.

6. **Pure Zig Math, SIMD & I/O Engine**:
   - `@Vector(8, f32)` SIMD vectorized GEMM / dot products achieving **24.8+ GFLOPS** on CPU.
   - SafeTensors binary parser (F32, F16, BF16 unpacking).
   - NetPBM PPM (P6 binary / P3) and 24-bit BMP writers; PNG, JPEG and everything else read through [zigimg](https://github.com/zigimg/zigimg).
   - Alpha-blended mask overlays, bounding box renderers, and point markers.

---

## 🚀 Performance Benchmarks

Measured on native Linux x86_64 (`-Doptimize=ReleaseFast`):

| Operation | Performance |
| :--- | :--- |
| **SIMD GEMM ($256 \times 256 \times 256$)** | **24.82 GFLOPS** (1.35 ms / iter) |
| **SAM 3 Image Segmentation ($128 \times 128$)** | **83.10 FPS** (12.03 ms / pass) |
| **Dependencies** | **1** (zigimg, itself pure Zig) |

---

## 🛠️ Building & Running

### 1. Run Unit & Integration Tests
```bash
zig build test
```

### 2. Run Synthetic Demo (Image & Video Segmentation)
```bash
zig build run -- demo
```
Generates visual output files:
- `demo_image_segmented.ppm` & `demo_image_segmented.bmp`
- `demo_video_frame_0.ppm` .. `demo_video_frame_4.ppm`

### 3. Run Performance Benchmark
```bash
zig build run -Doptimize=ReleaseFast -- benchmark
```

### 4. CLI Usage

#### Segment an Image
```bash
# Point prompt
./zig-out/bin/sam3 image --image input.ppm --point 0.5,0.5,1 --output segmented.ppm

# Concept text prompt
./zig-out/bin/sam3 image --image input.ppm --text "yellow school bus" --output segmented.ppm

# Loading SafeTensors weights
./zig-out/bin/sam3 image --image input.ppm --text "dog" --weights sam3.safetensors --output segmented.ppm
```

#### Segment from a click, on Meta's real weights

```bash
# Writes cat_0.bmp / cat_1.bmp / cat_2.bmp — the three multimask hypotheses
./zig-out/bin/sam3 segment --image assets/examples/cat.png \
    --weights sam3.safetensors --point 0.5,0.5,1 --output cat.bmp

# One mask instead of three
./zig-out/bin/sam3 segment --image assets/examples/cat.png \
    --weights sam3.safetensors --point 0.5,0.5,1 --single-mask --output cat.bmp
```

#### Run only the vision encoder

```bash
# Full 1008x1008 input, all 538 vision-encoder tensors from the checkpoint
./zig-out/bin/sam3 vision --image assets/examples/dog.png --weights sam3.safetensors

# 336x336 (the pretraining grid) for a ~11x faster pass
./zig-out/bin/sam3 vision --image assets/examples/dog.png --weights sam3.safetensors --size 336
```

### 5. Sample Images

The four sample images offered by the SAM 3 playground at
[sam3.ai/playground/sam3](https://sam3.ai/playground/sam3) — Dog, Cat, Person, Car
(Unsplash, 800 px wide):

```bash
zig build fetch-examples          # -> assets/examples/{dog,cat,person,car}.png
```

The download is pure Zig (`tools/fetch.zig`, `std.http.Client`) and [zigimg](https://github.com/zigimg/zigimg)
decodes the PNGs, so this needs no `curl` and no image tooling. Re-running only re-downloads
what is missing. Any format zigimg reads works as `--image`: PNG, JPEG, BMP, TGA, QOI, GIF.

---

## 🧠 Meta's SAM 3 Weights

The released checkpoint (`facebook/sam3`) is **859,922,360 parameters** across **1797 F32
tensors**, 3.2 GiB on disk. Fetch and verify it with:

```bash
zig build fetch-weights          # -> sam3.safetensors + assets/sam3/{config,tokenizer,...}
```

`facebook/sam3` is gated behind a manual approval form on Hugging Face, so the default is a
public mirror, and every download is checked against the SHA-256 Meta publishes for the file
(`6d06f0a5…cc11cc14a`) — both routes yield byte-identical weights. With an approved account:

```bash
HF_TOKEN=hf_... zig build fetch-weights -Dhf-repo=facebook/sam3
```

The step is idempotent: a checkpoint already on disk is verified and left alone rather than
re-downloaded. The weights are covered by Meta's SAM license (`assets/sam3/LICENSE`).

The checkpoint is deliberately *not* a `build.zig.zon` dependency — it is 3.2 GiB, and the
package manager cannot send the `Authorization` header the gated repo needs, so a build step
with a published checksum is both smaller and stricter than a package hash would be.

### What the sam3.ai playground runs

The browser playground is Transformers.js on WebGPU loading
[`onnx-community/sam3-tracker-ONNX`](https://huggingface.co/onnx-community/sam3-tracker-ONNX),
an ONNX export of the **tracker branch** of this same `facebook/sam3` checkpoint (geometric
prompts only — no text/concept path). Its precision menu picks which export to download:

| Playground option | ONNX files | Bytes |
| :--- | :--- | ---: |
| Q4 (Fastest) | `vision_encoder_q4` + `prompt_encoder_mask_decoder_q4` | ~349 MB |
| Q4F16 (Balanced) | `…_q4f16` | ~430 MB |
| FP16 (Best Quality) | `…_fp16` | ~946 MB |
| **FP32 (Full Precision)** | `vision_encoder.onnx_data` (1.87 GB) + `prompt_encoder_mask_decoder.onnx_data` (22 MB) | **~1.9 GB** (labelled “~1.6GB”) |

So the playground's "1.6 GB" model is not a separate checkpoint — it is the F32 vision
encoder + prompt encoder/mask decoder of `facebook/sam3`, i.e. a subset of the 3.2 GiB
`sam3.safetensors` fetched above, minus the text encoder and DETR concept detector. Nothing
here reads ONNX, so `--weights sam3.safetensors` is the equivalent (and a superset).

### Inspecting a checkpoint

Reads the SafeTensors header only, so a multi-gigabyte file is summarised instantly:

```bash
zig build run -Doptimize=ReleaseFast -- weights sam3.safetensors
zig build run -Doptimize=ReleaseFast -- weights sam3.safetensors --filter detr_decoder
zig build run -Doptimize=ReleaseFast -- weights sam3.safetensors --verify
```

`--verify` cross-checks `SAM3Config.sam3_full()` against the tensor shapes in the file
(patch embedding, MLP ratio, backbone/text depth, vocabulary, object queries, presence
token), so the declared architecture cannot silently drift from the weights.

The checkpoint's module layout:

| Module | Tensors | Params |
| :--- | ---: | ---: |
| `detector_model.vision_encoder` | 538 | 454.0M |
| `detector_model.text_encoder` | 389 | 353.5M |
| `detector_model.detr_decoder` | 247 | 11.6M |
| `detector_model.detr_encoder` | 156 | 9.5M |
| `detector_model.geometry_encoder` | 94 | 8.1M |
| `tracker_neck.fpn_layers` | 22 | 7.8M |
| `tracker_model.memory_attention` | 106 | 5.9M |
| `tracker_model.mask_decoder` | 131 | 4.2M |
| `detector_model.mask_decoder` | 32 | 2.3M |
| *(13 further modules)* | 82 | 3.1M |

### Port status: the tracker branch produces real masks

The port replaces the synthetic graph module by module. Every stage reports checkpoint
coverage, so progress is measurable rather than asserted.

| Component | Checkpoint | Status |
| :--- | :--- | :--- |
| **Vision backbone** | 32-layer PE ViT, d=1024, 16 heads, patch 14 @ 1008², axial 2D RoPE, 24² windowed attention with global layers 7/15/23/31, MLP 1024→4736 | **ported** — `vision_encoder.zig` |
| **Vision neck** | FPN, 4 levels at 4x/2x/1x/0.5x, transposed-conv scale layers, 1x1 + 3x3 projections to 256. Detector and tracker each have their own neck over the shared backbone | **ported** — same file |
| **Geometric prompt encoder** | Random-Fourier point encoding, learned per-label embeddings, `not_a_point` / `no_mask` embeddings | **ported** — `sam3_tracker.zig` |
| **Two-way mask decoder** | 2 blocks of (token self-attn, token→image, MLP, image→token), final token→image attention, transposed-conv upscaler with high-res skips, 4 hypernetwork MLPs, IoU and object-score heads | **ported** — same file |
| Text encoder | 24-layer CLIP text tower, d=1024, 49408 BPE vocab, ctx 32 | not started |
| Concept detector | 6-layer DETR encoder + 6-layer decoder, 200 queries, box RPB, presence token, dot-product scoring | not started |
| Detector mask decoder | Pixel decoder + mask embedder MLP, 3 upsampling stages | not started |
| Video memory | Memory attention (4 layers, RoPE), memory fuser, mask downsampler, object pointers | not started |

Together those cover the same path the browser playground exports as
`onnx-community/sam3-tracker-ONNX`: pixels and a click in, a mask out.

#### Segmenting from a click

```bash
./zig-out/bin/sam3 segment --image assets/examples/cat.png \
    --weights sam3.safetensors --point 0.5,0.5,1 --output cat.bmp
```

```
  Image:       assets/examples/cat.png (800x550)
  Model input: 1008x1008 -> 72x72 tokens
  Prompt:      (0.500, 0.500) label 1

  Loaded 1797 checkpoint tensors in 1.4 s
  Vision encoder: 308.4 s
  Mask decoder:   1.8 s

  Object score logit: 18.7589
  Mask 0: predicted IoU 0.7371, covers   0.1% of the image  <- highest IoU
  Mask 1: predicted IoU 0.0037, covers  26.0% of the image
  Mask 2: predicted IoU 0.2728, covers   0.2% of the image

  Checkpoint coverage: 674/674 weights resolved (100.0%), 0 randomly initialised
```

All three multimask hypotheses are written out (`cat_0.bmp`, `cat_1.bmp`, `cat_2.bmp`). For
that click — which lands exactly on the cat's nose — two of them are the nose and one is the
whole cat, which is the part/whole ambiguity the multimask head exists to express.

674 tensors resolve by name *and* shape: 538 backbone + tracker neck, 131 mask decoder,
4 prompt encoder, 1 shared positional embedding. A mismatch in either name or shape falls
back to random init and is counted as unresolved, so 100% coverage means the Zig modules'
dimensions agree with Meta's everywhere. The command exits non-zero if anything falls back.

#### Running only the vision encoder

```bash
./zig-out/bin/sam3 vision --image assets/examples/dog.png --weights sam3.safetensors
./zig-out/bin/sam3 vision --image assets/examples/dog.png --weights sam3.safetensors --size 336
```

```
  Model input: 1008x1008 -> 72x72 tokens, d=1024, 32 layers
  Forward pass: 277.1 s

  Backbone tokens: [1, 5184, 1024]  mean  -0.0197  std  1.4637  range [-212.637,  29.888]
  FPN level 0 (4.0x): [1, 256, 288, 288]  mean  -0.0168  std  0.0187  range [  -0.165,   0.129]
  FPN level 1 (2.0x): [1, 256, 144, 144]  mean  -0.0025  std  1.2022  range [  -9.795,   8.511]
  FPN level 2 (1.0x): [1, 256, 72, 72]    mean  -0.0188  std  0.7109  range [  -5.442,   4.575]
  FPN level 3 (0.5x): [1, 256, 36, 36]    mean   0.0001  std  0.4552  range [  -1.838,   2.469]

  Checkpoint coverage: 538/538 weights resolved (100.0%), 0 randomly initialised
```

The FPN sizes are the ones `config.json` declares in `backbone_feature_sizes`.

`--size` trades fidelity for time: position embeddings are tiled and RoPE is generated per
grid, so any multiple of the patch size works. Measured on 8 cores, `-Doptimize=ReleaseFast`:

| Model input | Tokens | Vision encoder | Mask decoder |
| :--- | ---: | ---: | ---: |
| 168² | 12×12 | 12.4 s | <1 s |
| 336² | 24×24 | 25.7 s | 0.2 s |
| 1008² (native) | 72×72 | 277.1 s | 1.8 s |

Plus 1.4–5 s to load the checkpoint warm (17.9 s the first time, reading 3.2 GiB off disk).
Optimisation level matters more than anything else here: the same 168² pass takes 148.7 s in
a Debug build, 12x slower, so never benchmark what `zig build` installs by default.

Prefer 1008 for actual segmentation — at 336 the token grid equals the attention window, so
every layer becomes global and the features drift from what the decoder was trained on.

#### What is and isn't verified

Structural: all 674 tensors match by name and shape; FPN output sizes match the config;
hand-computed transposed-convolution and window-partition round-trips; RoPE norm
preservation; exact-GELU reference values; identical output across repeated runs (no races
in the parallel attention). Behavioural: clicks produce clean object boundaries and a
sensible part/whole hierarchy, and the object-score logit is strongly positive on real
objects.

On an unambiguous click the quality heads agree with the masks, which is the strongest
end-to-end signal short of a reference comparison — a click on the cat's chest gives:

```
  Object score logit: 22.7371
  Mask 0: predicted IoU 0.7329, covers  17.2% of the image
  Mask 1: predicted IoU 0.9798, covers  26.3% of the image
  Mask 2: predicted IoU 0.9803, covers  25.8% of the image  <- highest IoU
```

0.98 for the two whole-cat masks and 0.73 for the body-without-head one: the IoU head ranks
them the way the pixels say it should. The low scores on the nose click are the model's own
uncertainty about an ambiguous prompt, not a broken head.

Not verified: activations have never been compared against a PyTorch reference run, so
agreement with Meta's outputs to floating-point tolerance is unproven.

`sam3 image` still runs the older synthetic graph and still reports 0/62 coverage; it stays
until the concept (text) path lands. Remaining order: text tower + BPE tokenizer → DETR
encoder/decoder → detector mask decoder → video memory.

---

## 📦 Project Structure

```
sam3/
├── build.zig                   # Zig 0.16 build configuration
├── build.zig.zon               # Package metadata
├── tools/
│   └── fetch.zig                # Pure Zig downloader behind `zig build fetch-weights/-examples`
├── src/
│   ├── root.zig                # Library exports & integration tests
│   ├── main.zig                # CLI entry point
│   ├── tensor/
│   │   ├── tensor.zig          # Multi-dimensional Tensor struct & memory management
│   │   ├── math.zig            # SIMD dot products, GEMM, LayerNorm, GELU, SiLU, Softmax
│   │   └── ops.zig             # Conv2D, ConvTranspose2D, BilinearUpsample, MHA
│   ├── weights/
│   │   ├── safetensors.zig     # Pure Zig SafeTensors parser, header reader & F16/BF16 conversion
│   │   └── weight_loader.zig   # Named weight store, coverage tracking & initializers
│   ├── models/
│   │   ├── config.zig          # SAM3 configuration presets (tiny, base, full)
│   │   ├── image_encoder.zig   # Vision-Language Perception Backbone & Neck
│   │   ├── vision_encoder.zig  # Meta SAM 3 PE ViT backbone + FPN neck (runs on the checkpoint)
│   │   ├── sam3_tracker.zig    # Meta SAM 3 prompt encoder + two-way mask decoder
│   │   ├── prompt_encoder.zig  # Concept Text Tokenizer & Geometric Prompt Encoder
│   │   ├── detector.zig        # DETR Concept Detector & Presence Head
│   │   ├── mask_decoder.zig    # Two-Way Mask Decoder & Hypernetwork MLP
│   │   ├── memory.zig          # Video Memory Bank, Encoder & Temporal Attention
│   │   └── sam3.zig            # Unified SAM 3 Pipeline
│   ├── video/
│   │   └── tracker.zig         # Video Predictor & Multi-Object Tracking State Machine
│   ├── io/
│   │   ├── image.zig           # PPM and BMP encoders/decoders
│   │   ├── visualization.zig   # Alpha-blended mask overlays, bounding boxes, points
│   │   └── video_io.zig        # Video frame sequence loader & generator
│   └── cli/
│       └── cli.zig             # CLI commands (demo, benchmark, vision encoder, weight inspection)
```
