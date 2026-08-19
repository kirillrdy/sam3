# SAM 3 in Pure Zig ⚡

A high-performance, standalone implementation of **Meta's Segment Anything Model 3 (SAM 3)** written in **100% Pure Zig** (0.16.0) with zero external C++ or Python dependencies.

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
   - NetPBM PPM (P6 binary / P3) and 24-bit uncompressed BMP image codecs.
   - Alpha-blended mask overlays, bounding box renderers, and point markers.

---

## 🚀 Performance Benchmarks

Measured on native Linux x86_64 (`-Doptimize=ReleaseFast`):

| Operation | Performance |
| :--- | :--- |
| **SIMD GEMM ($256 \times 256 \times 256$)** | **24.82 GFLOPS** (1.35 ms / iter) |
| **SAM 3 Image Segmentation ($128 \times 128$)** | **83.10 FPS** (12.03 ms / pass) |
| **Dependencies** | **0** (Pure Zig Standard Library) |

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
./zig-out/bin/sam3 image --image input.ppm --text "dog" --weights sam3_base.safetensors --output segmented.ppm
```

---

## 📦 Project Structure

```
sam3/
├── build.zig                   # Zig 0.16 build configuration
├── build.zig.zon               # Package metadata
├── src/
│   ├── root.zig                # Library exports & integration tests
│   ├── main.zig                # CLI entry point
│   ├── tensor/
│   │   ├── tensor.zig          # Multi-dimensional Tensor struct & memory management
│   │   ├── math.zig            # SIMD dot products, GEMM, LayerNorm, GELU, SiLU, Softmax
│   │   └── ops.zig             # Conv2D, ConvTranspose2D, BilinearUpsample, MHA
│   ├── weights/
│   │   ├── safetensors.zig     # Pure Zig SafeTensors parser & F16/BF16 conversion
│   │   └── weight_loader.zig   # Named weight store & Xavier/Kaiming initializers
│   ├── models/
│   │   ├── config.zig          # SAM3 configuration presets (tiny, base)
│   │   ├── image_encoder.zig   # Vision-Language Perception Backbone & Neck
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
│       └── cli.zig             # CLI commands (demo, benchmark, runners)
```
