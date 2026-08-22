# PyTorch reference

Runs Meta's SAM 3 checkpoint through HF Transformers so the port can be checked against it —
same weights, same image, same prompt, stage for stage. Nothing here is part of the Zig build;
it exists only to be disagreed with.

The port now runs the ONNX export of the same checkpoint through ONNX Runtime, so what this
mainly checks is the code on either side of the graphs: the preprocessing that produces
`pixel_values`, and the postprocessing that turns mask logits back into a mask.

## Setup

```bash
cd reference
nix develop            # torch 2.12 (CPU), transformers 5.15, python 3.14
```

`transformers` 5.15 ships the SAM 3 modelling code, so no `facebookresearch/sam3` checkout is
needed. It does need the PyTorch checkpoint in Hugging Face's directory layout, which the build
no longer downloads — `zig build fetch-weights` fetches the ONNX export instead — so pull it
here:

```bash
hf download jetjodh/sam3 --local-dir sam3-hf
```

`facebook/sam3` is the official repo, but it is gated behind a manual approval form.
`jetjodh/sam3` is the public mirror the build used to default to and holds the same weights.

## Running

```bash
# Vision encoder only, with the detector neck
python reference_run.py --image ../.zig-cache/sam3/examples/dog.png --neck detector --vision-only

# Full point-prompted segmentation, matching what `zig build run` does
python reference_run.py --image ../.zig-cache/sam3/examples/cat.png --point 0.46,0.68,1
```

The output is laid out like the Zig CLI's so the two can be read side by side. `--json`
writes the same numbers as a machine-readable record.

Two things to know when comparing:

* **Preprocessing** is done by hand (rescale, bilinear resize, normalise) rather than through
  `Sam3ImageProcessorFast`, so that it matches `preprocess` in `src/sam3.zig` exactly rather
  than approximately. The processor's own settings — mean/std 0.5, size 1008, bilinear — are
  the same, and `processor_config.json` in the checkpoint confirms them.
* **The mask decoder is run twice**, with and without the learned `no_memory_embedding` that
  the reference adds to the coarsest FPN level on a frame with no memory bank behind it. The
  ONNX export has it, so it is the `[reference]` block that should match; the other block is
  kept as a control.

Both sides are hardwired to 1008x1008: the vision encoder's RoPE is built from the configured
grid on the PyTorch side, and the ONNX export was traced at that resolution.
