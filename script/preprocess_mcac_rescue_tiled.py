"""Build MCAC 4x4 rescue caches for no-GT frontend failure triggers.

The trigger matches the audited FSC147 12.67 policy: an image is rescued only
when its full-image fast cache has zero candidates. GT counts are never used to
select images. The output directory is an overlay: only triggered image IDs are
written, and eval_mcac.py falls back to the base cache for every other image.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.candidates.crops import build_three_crops
from code.encoders.dinov2_encoder import DINOv2RegionEncoder
from script.preprocess_fsc147_tiled import (
    bbox_iou_xywh,
    compute_tiles,
    merge_masks_by_bbox,
)
from script.preprocess_mcac import (
    build_sam2_amg,
    dot_matching_multiclass,
    encode_masks_rle,
    load_mcac_gt,
)


def collect_tiled_masks(image, amg, tiles, overlap):
    h, w = image.shape[:2]
    masks, bboxes = [], []
    for y1, x1, y2, x2 in compute_tiles(h, w, tiles, overlap):
        tile = image[y1:y2, x1:x2]
        th, tw = tile.shape[:2]
        for raw in amg.generate(tile):
            mask = np.asarray(raw["segmentation"]).astype(np.uint8)
            area = float(mask.sum())
            area_ratio = area / max(th * tw, 1)
            if area == 0 or area_ratio < 1e-4 or area_ratio > 0.95:
                continue
            ys, xs = np.where(mask)
            if xs.size == 0:
                continue
            lx1, ly1 = int(xs.min()), int(ys.min())
            lx2, ly2 = int(xs.max()) + 1, int(ys.max()) + 1
            if lx2 - lx1 < 4 or ly2 - ly1 < 4:
                continue
            global_mask = np.zeros((h, w), dtype=np.uint8)
            global_mask[y1:y2, x1:x2] = mask
            masks.append(global_mask)
            bboxes.append([
                float(x1 + lx1),
                float(y1 + ly1),
                float(lx2 - lx1),
                float(ly2 - ly1),
            ])

    if not masks:
        return [], []

    keep = merge_masks_by_bbox(bboxes, iou_thresh=0.7)
    masks = [masks[i] for i in keep]
    bboxes = [bboxes[i] for i in keep]

    # Match the existing rescue implementation's second near-duplicate pass.
    order = sorted(range(len(masks)), key=lambda i: -float(masks[i].sum()))
    kept = []
    for i in order:
        if all(bbox_iou_xywh(bboxes[i], bboxes[j]) <= 0.9 for j in kept):
            kept.append(i)
    kept.sort()
    return [masks[i] for i in kept], [bboxes[i] for i in kept]


def zero_candidate_result(im_id, counts, dots, h, w, frontend):
    dot_class = np.concatenate([
        np.full(len(d), ci, dtype=np.int64) for ci, d in enumerate(dots)
    ]) if dots else np.zeros((0,), dtype=np.int64)
    dots_flat = np.concatenate(dots, axis=0) if dots else np.zeros((0, 2))
    return {
        "img_id": im_id,
        "file_name": f"{im_id}/img.png",
        "gt_count": int(sum(counts)),
        "gt_class_counts": [int(c) for c in counts],
        "n_gt_classes": len(counts),
        "gt_dots": torch.from_numpy(dots_flat).float(),
        "gt_dot_class": torch.from_numpy(dot_class).long(),
        "z": torch.zeros(0, 1152),
        "bbox": torch.zeros(0, 4),
        "matched_class": torch.zeros(0, dtype=torch.long),
        "matched_instance_id": torch.zeros(0, dtype=torch.long),
        "iou": torch.zeros(0),
        "purity": torch.zeros(0),
        "coverage": torch.zeros(0),
        "valid": torch.zeros(0),
        "is_part": torch.zeros(0),
        "is_countable": torch.zeros(0),
        "masks_rle": [],
        "height": int(h),
        "width": int(w),
        "frontend": frontend,
    }


def process_image(image, im_id, dots, counts, amg, encoder, tiles, overlap, pts):
    h, w = image.shape[:2]
    frontend = {
        "policy": "fast n_candidates == 0 -> 4x4 tiled rescue",
        "trigger_uses_gt": False,
        "tiles": int(tiles),
        "overlap": float(overlap),
        "points_per_side": int(pts),
        "merge": "bbox IoU 0.7, then near-duplicate bbox IoU 0.9",
    }
    masks, bboxes = collect_tiled_masks(image, amg, tiles, overlap)
    if not masks:
        return zero_candidate_result(im_id, counts, dots, h, w, frontend)

    match = dot_matching_multiclass(masks, dots, h, w)
    masked_crops, box_crops, context_crops = [], [], []
    for mask, (x, y, bw, bh) in zip(masks, bboxes):
        mc, bc, cc = build_three_crops(image, mask, (x, y, x + bw, y + bh))
        masked_crops.append(mc)
        box_crops.append(bc)
        context_crops.append(cc)
    z = encoder.encode_views(masked_crops, box_crops, context_crops, batch_size=64)

    dots_flat = np.concatenate(dots, axis=0) if dots else np.zeros((0, 2))
    dot_class = np.concatenate([
        np.full(len(d), ci, dtype=np.int64) for ci, d in enumerate(dots)
    ]) if dots else np.zeros((0,), dtype=np.int64)
    n = len(masks)
    return {
        "img_id": im_id,
        "file_name": f"{im_id}/img.png",
        "gt_count": int(sum(counts)),
        "gt_class_counts": [int(c) for c in counts],
        "n_gt_classes": len(counts),
        "gt_dots": torch.from_numpy(dots_flat).float(),
        "gt_dot_class": torch.from_numpy(dot_class).long(),
        "z": z.float(),
        "bbox": torch.tensor(bboxes, dtype=torch.float32),
        "matched_class": torch.from_numpy(match["matched_class"]).long(),
        "matched_instance_id": torch.from_numpy(match["matched_instance_id"]).long(),
        "iou": torch.from_numpy(match["coverage"]).float(),
        "purity": torch.from_numpy(match["purity"]).float(),
        "coverage": torch.from_numpy(match["coverage"]).float(),
        "valid": torch.from_numpy(match["valid"]).float(),
        "is_part": torch.zeros(n),
        "is_countable": torch.ones(n),
        "masks_rle": encode_masks_rle(masks),
        "height": int(h),
        "width": int(w),
        "frontend": frontend,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mcac-dir", default="/home/czp/ljs/dataset/MCAC")
    ap.add_argument("--split", default="test")
    ap.add_argument("--fast-cache-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pts-per-side", type=int, default=32)
    ap.add_argument("--tiles", type=int, default=4)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--crop-size", type=int, default=672)
    ap.add_argument("--occ-limit", type=float, default=70.0)
    ap.add_argument("--axis-mode", default="xy_flip")
    ap.add_argument("--image-ids", default="",
                    help="Comma-separated override; otherwise infer zero-candidate fast caches.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    fast_dir = Path(args.fast_cache_dir)
    fast_files = sorted(fast_dir.glob("*.pt"))
    if args.image_ids:
        trigger_ids = [x.strip() for x in args.image_ids.split(",") if x.strip()]
    else:
        trigger_ids = []
        for path in fast_files:
            d = torch.load(path, map_location="cpu", weights_only=False)
            if int(d["z"].shape[0]) == 0:
                trigger_ids.append(path.stem)
    if not trigger_ids:
        print("[done] no fast n_candidates==0 triggers")
        return

    print(f"[trigger] {len(trigger_ids)} images: {trigger_ids}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    amg = build_sam2_amg(args.device, args.pts_per_side)
    encoder = DINOv2RegionEncoder(device=args.device)
    split_dir = Path(args.mcac_dir) / args.split

    for idx, im_id in enumerate(trigger_ids, 1):
        out_path = out_dir / f"{im_id}.pt"
        if out_path.exists() and not args.force:
            print(f"[{idx}/{len(trigger_ids)}] {im_id}: cached")
            continue
        image = np.array(Image.open(split_dir / im_id / "img.png").convert("RGB"))
        h, w = image.shape[:2]
        cs = args.crop_size
        if cs > 0 and (h > cs or w > cs):
            y0, x0 = (h - cs) // 2, (w - cs) // 2
            image = image[y0:y0 + cs, x0:x0 + cs]
        dots, counts = load_mcac_gt(
            split_dir / im_id / "info_with_occ_bbox.json",
            args.crop_size,
            args.occ_limit,
            args.axis_mode,
        )
        t0 = time.time()
        result = process_image(
            image, im_id, dots, counts, amg, encoder,
            args.tiles, args.overlap, args.pts_per_side,
        )
        torch.save(result, out_path)
        print(f"[{idx}/{len(trigger_ids)}] {im_id}: candidates={result['z'].shape[0]} "
              f"valid={int((result['valid'] > 0).sum())} elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
