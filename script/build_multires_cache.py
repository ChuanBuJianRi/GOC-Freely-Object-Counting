"""Build multi-resolution cache: merge pts=16 (fast) + pts=32 (tiled) candidates.

P2: Multi-Resolution Counting Fusion.
Combines coarse (pts=16, ~120 candidates, better classification) with
fine (pts=32 2x2 tiled, ~285 candidates, better recall) candidates.
Bbox-IoU dedup removes redundant overlapping candidates.

Usage:
    python script/build_multires_cache.py \
        --cache-16 /home/czp/ws_yiyang/ovcud_cache/fsc147_test_fast \
        --cache-32 /home/czp/ws_yiyang/ovcud_cache/fsc147_test_tiled \
        --out-dir /home/czp/ws_yiyang/ovcud_cache/fsc147_test_multires
"""

import argparse, os, sys, time
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mask_utils


def bbox_iou_matrix(bboxes_xywh):
    """Vectorized pairwise bbox IoU."""
    n = len(bboxes_xywh)
    if n <= 1:
        return np.zeros((n, n))
    x1 = bboxes_xywh[:, 0]
    y1 = bboxes_xywh[:, 1]
    x2 = bboxes_xywh[:, 0] + bboxes_xywh[:, 2]
    y2 = bboxes_xywh[:, 1] + bboxes_xywh[:, 3]
    areas = bboxes_xywh[:, 2] * bboxes_xywh[:, 3]

    ix1 = np.maximum(x1[:, None], x1[None, :])
    iy1 = np.maximum(y1[:, None], y1[None, :])
    ix2 = np.minimum(x2[:, None], x2[None, :])
    iy2 = np.minimum(y2[:, None], y2[None, :])
    iw = np.maximum(0, ix2 - ix1)
    ih = np.maximum(0, iy2 - iy1)
    inter = iw * ih
    union = areas[:, None] + areas[None, :] - inter
    return inter / np.maximum(union, 1.0)


def greedy_nms(ious, areas, thresh=0.7):
    order = sorted(range(len(areas)), key=lambda i: areas[i], reverse=True)
    kept = []
    for i in order:
        dup = False
        for j in kept:
            if ious[i, j] > thresh:
                dup = True
                break
        if not dup:
            kept.append(i)
    return sorted(kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-16", default="/home/czp/ws_yiyang/ovcud_cache/fsc147_test_fast")
    ap.add_argument("--cache-32", default="/home/czp/ws_yiyang/ovcud_cache/fsc147_test_tiled")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--iou-thresh", type=float, default=0.5,
                    help="Bbox IoU threshold for merging (lower = more aggressive)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cache16 = {f.replace(".pt", ""): os.path.join(args.cache_16, f)
               for f in os.listdir(args.cache_16) if f.endswith(".pt")}
    cache32 = {f.replace(".pt", ""): os.path.join(args.cache_32, f)
               for f in os.listdir(args.cache_32) if f.endswith(".pt")}

    common = sorted(set(cache16.keys()) & set(cache32.keys()))
    print(f"Common images: {len(common)}")

    n_ok = n_error = 0
    total_16 = total_32 = total_merged = 0
    t_start = time.time()

    for i, img_id in enumerate(common):
        out_path = os.path.join(args.out_dir, f"{img_id}.pt")
        if os.path.exists(out_path):
            continue

        try:
            d16 = torch.load(cache16[img_id], map_location="cpu", weights_only=False)
            d32 = torch.load(cache32[img_id], map_location="cpu", weights_only=False)
        except Exception:
            n_error += 1
            continue

        # Combine
        h, w = int(d16["height"]), int(d16["width"])
        z16, bbox16 = d16["z"].float(), d16["bbox"].float().numpy()
        z32, bbox32 = d32["z"].float(), d32["bbox"].float().numpy()
        n16, n32 = z16.shape[0], z32.shape[0]

        z_all = torch.cat([z16, z32], dim=0)
        bbox_all = np.concatenate([bbox16, bbox32], axis=0)

        # Bbox-IoU dedup: remove redundant candidates
        areas = bbox_all[:, 2] * bbox_all[:, 3]
        if len(areas) > 1:
            ious = bbox_iou_matrix(bbox_all)
            keep = greedy_nms(ious, areas, thresh=args.iou_thresh)
        else:
            keep = [0]

        z_all = z_all[keep]
        bbox_all = bbox_all[keep]

        # Build result
        n_merged = len(keep)
        # Combine match arrays from both sources
        mc16 = d16.get("matched_class", torch.zeros(n16, dtype=torch.long))
        mc32 = d32.get("matched_class", torch.zeros(n32, dtype=torch.long))
        mc_all = torch.cat([mc16, mc32], dim=0)[keep]
        vi16 = d16.get("valid", torch.ones(n16))
        vi32 = d32.get("valid", torch.ones(n32))
        vi_all = torch.cat([vi16, vi32], dim=0)[keep]

        result = {
            "img_id": d16.get("img_id", img_id),
            "file_name": d16.get("file_name", f"{img_id}.jpg"),
            "class_name": d16.get("class_name", "unknown"),
            "gt_count": int(d16.get("gt_count", 0)),
            "z": z_all,
            "bbox": torch.from_numpy(bbox_all).float(),
            "matched_class": mc_all.long(),
            "matched_instance_id": torch.arange(n_merged).long(),
            "iou": torch.zeros(n_merged),
            "purity": torch.zeros(n_merged),
            "coverage": torch.zeros(n_merged),
            "valid": vi_all.float(),
            "is_part": torch.zeros(n_merged),
            "is_countable": torch.ones(n_merged),
            "masks_rle": [{"size": [h, w], "counts": "_"}],
            "height": h, "width": w,
        }
        torch.save(result, out_path)
        n_ok += 1
        total_16 += n16
        total_32 += n32
        total_merged += n_merged

        if (i + 1) % 200 == 0:
            dt = time.time() - t_start
            print(f"  [{i+1}/{len(common)}] avg 16={total_16/n_ok:.0f} "
                  f"32={total_32/n_ok:.0f} merged={total_merged/n_ok:.0f} "
                  f"rate={n_ok/dt:.1f}/s")

    dt = time.time() - t_start
    print(f"\nDone: {n_ok} files, {n_error} errors, {dt:.1f}s")
    print(f"Avg: pts16={total_16/max(n_ok,1):.0f}, pts32={total_32/max(n_ok,1):.0f}, "
          f"merged={total_merged/max(n_ok,1):.0f}")
    print(f"Saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
