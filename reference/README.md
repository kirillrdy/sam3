# PyTorch reference

Runs Meta's SAM 3 checkpoint through HF Transformers so the pure Zig port can be checked
against it — same weights, same image, same prompt, stage for stage. Nothing here is part of
the Zig build; it exists only to be disagreed with.

## Setup

```bash
cd reference
nix develop            # torch 2.12 (CPU), transformers 5.15, python 3.14
```

`transformers` 5.15 ships the SAM 3 modelling code, so no `facebookresearch/sam3` checkout is
needed. It does need the checkpoint in Hugging Face's directory layout, which is one symlink
away from what `zig build fetch-weights` already produced:

```bash
mkdir -p sam3-hf
ln -s ../../.zig-cache/sam3/sam3.safetensors sam3-hf/model.safetensors
for f in config.json tokenizer.json tokenizer_config.json vocab.json merges.txt \
         special_tokens_map.json; do ln -s ../../.zig-cache/sam3/$f sam3-hf/$f; done

# Not part of `zig build fetch-weights`, and `facebook/sam3` is gated, so take it
# from the same public mirror the fetch step defaults to.
curl -sSL -o sam3-hf/processor_config.json \
  https://huggingface.co/jetjodh/sam3/resolve/main/processor_config.json
```

## Running

```bash
# Vision encoder only, matching `sam3 vision` (which uses the detector neck)
python reference_run.py --image ../.zig-cache/sam3/examples/dog.png --neck detector --vision-only

# Full point-prompted segmentation, matching `sam3 segment`
python reference_run.py --image ../.zig-cache/sam3/examples/dog.png --point 0.5,0.5,1
```

The output is laid out like the Zig CLI's so the two can be read side by side. `--json`
writes the same numbers as a machine-readable record.

Two things to know when comparing:

* **Preprocessing** is done by hand (rescale, bilinear resize, normalise) rather than through
  `Sam3ImageProcessorFast`, so that it matches `VisionEncoder.preprocess` exactly rather than
  approximately. The processor's own settings — mean/std 0.5, size 1008, bilinear — are the
  same, and `processor_config.json` in the checkpoint confirms them.
* **The mask decoder is run twice**, with and without the learned `no_memory_embedding` that
  the reference adds to the coarsest FPN level on a frame with no memory bank behind it. The
  port adds it, so it is the `[reference]` block that should match; the other block is what
  the port produced before that was fixed, kept as a control.

The vision encoder is hardwired to 1008x1008 on the PyTorch side (RoPE is built from the
configured grid), so the Zig `--size` sweep has no counterpart here without editing the
config.
