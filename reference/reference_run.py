"""PyTorch/transformers reference run for the pure-Zig SAM 3 port.

Mirrors what `sam3 vision` and `sam3 segment` do, stage for stage, so the
timings and the tensor statistics line up with what the Zig CLI prints.
"""

import argparse, json, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import Sam3VideoModel

torch.set_grad_enabled(False)


def stats(t):
    t = t.float()
    return "mean %8.4f  std %8.4f  range [%8.3f, %8.3f]" % (
        t.mean().item(), t.std().item(), t.min().item(), t.max().item())


def preprocess(path, size):
    """Same as VisionEncoder.preprocess in Zig: /255, bilinear resize, (x-0.5)/0.5."""
    img = Image.open(path).convert("RGB")
    raw = torch.from_numpy(np.asarray(img)).permute(2, 0, 1)[None].float() / 255.0
    px = F.interpolate(raw, size=(size, size), mode="bilinear", align_corners=False, antialias=False)
    return (px - 0.5) / 0.5, img.size  # (w, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./sam3-hf")
    ap.add_argument("--image", required=True)
    ap.add_argument("--size", type=int, default=1008)
    ap.add_argument("--point", default="0.5,0.5,1")
    ap.add_argument("--vision-only", action="store_true")
    ap.add_argument("--neck", default="tracker", choices=["tracker", "detector"])
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    rec = {"size": args.size, "image": args.image, "threads": torch.get_num_threads()}
    print("\n=== SAM 3 PyTorch reference (transformers, CPU, %d threads) ===\n" % torch.get_num_threads())

    t0 = time.perf_counter()
    model = Sam3VideoModel.from_pretrained(args.model, dtype=torch.float32).eval()
    rec["load_s"] = time.perf_counter() - t0
    print("  Loaded checkpoint in %.1f s" % rec["load_s"])

    pixel_values, (w, h) = preprocess(args.image, args.size)
    print("  Image:       %s (%dx%d)" % (args.image, w, h))
    grid = args.size // 14
    print("  Model input: %dx%d -> %dx%d tokens\n" % (args.size, args.size, grid, grid))

    vision = model.detector_model.vision_encoder
    neck = model.tracker_neck if args.neck == "tracker" else model.detector_model.vision_encoder.neck

    # --- stage 1: vision backbone ---
    backbone_times = []
    for _ in range(args.repeat):
        t = time.perf_counter()
        vout = vision(pixel_values)
        backbone_times.append(time.perf_counter() - t)
    hs = vout.last_hidden_state
    rec["backbone_s"] = min(backbone_times)

    # --- stage 2: tracker FPN neck ---
    spatial = hs.view(1, grid, grid, -1).permute(0, 3, 1, 2)
    t = time.perf_counter()
    fpn, fpn_pos = neck(spatial)
    rec["neck_s"] = time.perf_counter() - t
    rec["vision_s"] = rec["backbone_s"] + rec["neck_s"]

    print("  Forward pass: %.1f s  (backbone %.1f s + %s neck %.1f s)\n"
          % (rec["vision_s"], rec["backbone_s"], args.neck, rec["neck_s"]))
    print("  Backbone tokens: [%d, %d, %d]  %s" % (hs.shape[0], hs.shape[1], hs.shape[2], stats(hs)))
    rec["backbone_stats"] = [hs.mean().item(), hs.std().item(), hs.min().item(), hs.max().item()]
    rec["fpn_stats"] = []
    for i, f in enumerate(fpn):
        print("  FPN level %d: [%d, %d, %d, %d]  %s"
              % (i, f.shape[0], f.shape[1], f.shape[2], f.shape[3], stats(f)))
        rec["fpn_stats"].append([list(f.shape), f.mean().item(), f.std().item(), f.min().item(), f.max().item()])

    if args.vision_only:
        if args.json:
            json.dump(rec, open(args.json, "w"), indent=2)
        return


    # --- stage 3: prompt encoder + two-way mask decoder ---
    tracker = model.tracker_model
    base = list(fpn[:-1])  # levels 0,1,2 (288, 144, 72); level 3 is detector-only
    base[0] = tracker.mask_decoder.conv_s0(base[0])
    base[1] = tracker.mask_decoder.conv_s1(base[1])

    px, py, plab = (float(v) for v in args.point.split(","))
    pts = torch.tensor([[[[px * args.size, py * args.size]]]])
    labels = torch.tensor([[[int(plab)]]], dtype=torch.int32)
    print("\n  Prompt:      (%.3f, %.3f) label %d" % (px, py, int(plab)))

    image_pe = tracker.get_image_wide_positional_embeddings()
    rec["variants"] = {}

    # The reference adds a learned "no memory" embedding to the coarsest level on
    # a frame with no memory bank; the Zig port does not, so both are reported.
    for name, add_no_mem in (("reference", True), ("zig-equivalent (no no_memory_embedding)", False)):
        feats = list(base)
        if add_no_mem:
            feats[-1] = feats[-1] + tracker.no_memory_embedding.view(1, -1, 1, 1)

        t = time.perf_counter()
        sparse, dense = tracker.prompt_encoder(
            input_points=pts, input_labels=labels, input_boxes=None, input_masks=None)
        low_res, iou, _, obj_logits = tracker.mask_decoder(
            image_embeddings=feats[-1],
            image_positional_embeddings=image_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=True,
            high_resolution_features=feats[:-1],
        )
        dt = time.perf_counter() - t

        obj = obj_logits.flatten()[0].item()
        iou = iou.flatten()
        masks = low_res.reshape(-1, low_res.shape[-2], low_res.shape[-1])
        print("\n  [%s]  mask decoder %.2f s" % (name, dt))
        print("  Object score logit: %.4f" % obj)
        entry = {"decoder_s": dt, "object_score": obj, "masks": []}
        best = int(iou.argmax())
        for i in range(masks.shape[0]):
            full = F.interpolate(masks[i][None, None].float(), size=(h, w),
                                 mode="bilinear", align_corners=False)
            cov = 100.0 * (full > 0).float().mean().item()
            print("  Mask %d: predicted IoU %.4f, covers %5.1f%% of the image%s"
                  % (i, iou[i].item(), cov, "  <- highest IoU" if i == best else ""))
            entry["masks"].append({"iou": iou[i].item(), "coverage_pct": cov})
        rec["variants"][name] = entry
        rec["decoder_s"] = dt

    if args.json:
        json.dump(rec, open(args.json, "w"), indent=2)


if __name__ == "__main__":
    main()
