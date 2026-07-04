"""Adaptive two-pass tiling for FSC147 dense images.

Unlike the standard tiling approach which REPLACES full-image candidates
with tiled candidates, adaptive tiling AUGMENTS full-image candidates with
targeted tile candidates only in under-segmented regions.

Algorithm:
  1. Pass 1: SAM2 on full image (pts=32) → base candidates
  2. Density Analysis: count candidates per 2×2 cell
  3. Decision: cells with < 40% of avg density → TILE
  4. Pass 2: SAM2 on flagged cells → tile candidates
  5. Merge: base + tile candidates via IoU NMS
  6. DINOv2 3-view encoding → .pt cache

This preserves good candidates from the global view while adding
fine-grained candidates only where SAM2 may have missed objects
due to density. Sparse images (< 50 global candidates) skip tiling entirely.

Inspired by S-DCNet (Xiong et al., ICCV 2019) — spatial divide-and-conquer.

Usage:
    python script/preprocess_fsc147_adaptive.py \
        --ann /home/czp/official_code/dataset/FSC147/annotation_FSC147_384.json \
        --img-dir /home/czp/official_code/dataset/FSC147/images_384_VarV2 \
        --images-file result/logs/fsc147_test_100plus.json \
        --out-dir /home/czp/ws_yiyang/ovcud_cache/fsc147_test_tiled_adaptive \
        --tiles 2 --overlap 0.25 --pts-per-side 32
"""

from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.encoders.dinov2_encoder import DINOv2RegionEncoder
from code.candidates.crops import build_three_crops


# ---------------------------------------------------------------------------
# SAM2 AMG
# ---------------------------------------------------------------------------
def build_sam2_amg(device: str, pts_per_side: int = 32):
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    config = "configs/sam2.1/sam2.1_hiera_s.yaml"
    ckpt = "/home/czp/ws_yiyang/FreeCounting/ws_yiyang/OCCAM/checkpoints/sam2.1_hiera_small.pt"
    model = build_sam2(config, ckpt, device=device)
    return SAM2AutomaticMaskGenerator(
        model=model, points_per_side=pts_per_side, points_per_batch=64,
        pred_iou_thresh=0.7, stability_score_thresh=0.8,
        stability_score_offset=1.0, box_nms_thresh=0.7,
        crop_n_layers=0, crop_nms_thresh=0.7,
        use_m2m=False, multimask_output=True,
    )


def encode_masks_rle(masks: List[np.ndarray]) -> List[dict]:
    rles = []
    for m in masks:
        mm = np.asfortranarray(np.asarray(m).astype(np.uint8))
        rle = mask_utils.encode(mm)
        counts = rle["counts"]
        if isinstance(counts, bytes):
            counts = counts.decode("ascii")
        rles.append({"size": [int(rle["size"][0]), int(rle["size"][1])], "counts": counts})
    return rles


# ---------------------------------------------------------------------------
# SAM2 candidate extraction (shared by full-image and tile passes)
# ---------------------------------------------------------------------------
def extract_candidates(raw_masks, tile_h, tile_w):
    """Filter and convert raw SAM2 output to (mask, bbox_xywh) pairs."""
    masks, bboxes = [], []
    for r in raw_masks:
        m = np.asarray(r["segmentation"]).astype(np.uint8)
        area = float(m.sum())
        if area == 0:
            continue
        ar = area / (tile_h * tile_w)
        if ar < 1e-4 or ar > 0.95:
            continue
        ys, xs = np.where(m)
        if xs.size == 0:
            continue
        lx1, ly1 = int(xs.min()), int(ys.min())
        lx2, ly2 = int(xs.max()) + 1, int(ys.max()) + 1
        if (lx2 - lx1) < 4 or (ly2 - ly1) < 4:
            continue
        masks.append(m)
        bboxes.append([float(lx1), float(ly1),
                       float(lx2 - lx1), float(ly2 - ly1)])
    return masks, bboxes


# ---------------------------------------------------------------------------
# Tiling geometry
# ---------------------------------------------------------------------------
def compute_tiles(h, w, n_tiles, overlap):
    tile_h = int(h / n_tiles)
    tile_w = int(w / n_tiles)
    stride_h = int(tile_h * (1 - overlap))
    stride_w = int(tile_w * (1 - overlap))
    tiles = []
    for ty in range(n_tiles):
        for tx in range(n_tiles):
            y1 = max(0, ty * stride_h)
            x1 = max(0, tx * stride_w)
            y2 = min(h, y1 + tile_h)
            x2 = min(w, x1 + tile_w)
            if ty == n_tiles - 1: y2 = h
            if tx == n_tiles - 1: x2 = w
            tiles.append((y1, x1, y2, x2))
    return tiles


def merge_masks_by_iou(masks, bboxes, iou_thresh=0.7):
    n = len(masks)
    if n <= 1:
        return list(range(n))
    iou_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            inter = float((masks[i] & masks[j]).sum())
            union = float((masks[i] | masks[j]).sum())
            iou_mat[i, j] = iou_mat[j, i] = inter / max(union, 1.0)
    areas = [m.sum() for m in masks]
    order = sorted(range(n), key=lambda i: areas[i], reverse=True)
    kept = []
    for i in order:
        dup = False
        for j in kept:
            if iou_mat[i, j] > iou_thresh:
                dup = True
                break
        if not dup:
            kept.append(i)
    kept.sort()
    return kept


# ---------------------------------------------------------------------------
# Density Analysis (the "Density Decider")
# ---------------------------------------------------------------------------
def analyze_cell_density(bboxes_xywh, h, w, grid=2):
    """Count candidates per grid cell and compute per-cell statistics.

    Returns:
        cell_counts: np.array [grid*grid] — candidate count per cell
        cell_small_ratio: np.array [grid*grid] — fraction of small candidates per cell
        avg_per_cell: global average candidates per cell
    """
    tiles = compute_tiles(h, w, grid, overlap=0.0)  # non-overlapping for analysis
    n_cells = len(tiles)
    cell_counts = np.zeros(n_cells, dtype=int)
    cell_small = np.zeros(n_cells, dtype=int)

    # Compute bbox centers
    cx = bboxes_xywh[:, 0] + bboxes_xywh[:, 2] / 2
    cy = bboxes_xywh[:, 1] + bboxes_xywh[:, 3] / 2
    areas = bboxes_xywh[:, 2] * bboxes_xywh[:, 3]
    image_area = h * w
    small_thresh = 0.002 * image_area  # 0.2% of image area = "small" object

    for ci, (y1, x1, y2, x2) in enumerate(tiles):
        in_cell = (cx >= x1) & (cx < x2) & (cy >= y1) & (cy < y2)
        cell_counts[ci] = int(in_cell.sum())
        cell_small[ci] = int((in_cell & (areas < small_thresh)).sum())

    avg_per_cell = max(cell_counts.mean(), 1.0)
    cell_small_ratio = cell_small / np.maximum(cell_counts, 1)

    return cell_counts, cell_small_ratio, avg_per_cell, tiles


def decide_tiling_plan(cell_counts, cell_small_ratio, avg_per_cell, n_total):
    """Decide which cells to tile.

    Heuristic:
      - If global n_cand < MIN_GLOBAL: image is too sparse → skip all tiling
      - If cell_count < 0.4 * avg AND the cell has any small candidates → TILE
      - Otherwise → SKIP

    The 0.4 threshold means: if a cell has less than 40% of the average
    per-cell candidate count, it's likely under-segmented.
    """
    MIN_GLOBAL = 50  # If total candidates < 50, skip tiling entirely

    if n_total < MIN_GLOBAL:
        return ["SKIP"] * len(cell_counts)

    plan = []
    for ci in range(len(cell_counts)):
        if cell_counts[ci] < 0.4 * avg_per_cell:
            plan.append("TILE")
        else:
            plan.append("SKIP")

    return plan


# ---------------------------------------------------------------------------
# Dot-based matching (FSC147 specific)
# ---------------------------------------------------------------------------
def dot_based_matching(
    masks: List[np.ndarray],
    points: List[Tuple[float, float]],
    class_idx: int,
    h: int, w: int,
) -> dict:
    n_cand = len(masks)
    n_dots = len(points)
    pts_int = []
    for x, y in points:
        xi, yi = int(round(float(x))), int(round(float(y)))
        if 0 <= xi < w and 0 <= yi < h:
            pts_int.append((xi, yi))
    n_dots_valid = len(pts_int)

    purity = np.zeros(n_cand, dtype=np.float32)
    coverage = np.zeros(n_cand, dtype=np.float32)
    valid = np.zeros(n_cand, dtype=np.float32)

    for i, m in enumerate(masks):
        area = float(m.sum())
        if area == 0:
            continue
        dots_covered = 0
        for xi, yi in pts_int:
            if m[yi, xi]:
                dots_covered += 1
        purity[i] = dots_covered / max(area, 1.0)
        coverage[i] = dots_covered / max(n_dots_valid, 1)
        area_ratio = area / (h * w)
        if dots_covered >= 1 and 1e-4 < area_ratio < 0.95:
            valid[i] = 1.0
    return {
        "purity": purity, "coverage": coverage,
        "valid": valid,
        "matched_class": np.full(n_cand, class_idx, dtype=np.int64),
        "matched_instance_id": np.arange(n_cand, dtype=np.int64),
    }


# ---------------------------------------------------------------------------
# Main adaptive two-pass pipeline
# ---------------------------------------------------------------------------
def process_image_adaptive(image, file_name, ann_entry, class_idx, class_name,
                           amg, encoder, n_tiles, overlap):
    """Two-pass adaptive tiling: full-image SAM2 + targeted tile SAM2."""
    h, w = image.shape[:2]

    # --- Pass 1: SAM2 on full image ---
    raw_full = amg.generate(image)
    base_masks, base_bboxes = extract_candidates(raw_full, h, w)
    n_base = len(base_masks)

    if n_base == 0:
        return None

    # --- Density Analysis ---
    bboxes_arr = np.array(base_bboxes) if base_bboxes else np.zeros((0, 4))
    if bboxes_arr.shape[0] > 0:
        cell_counts, cell_small_ratio, avg_per_cell, analysis_tiles = \
            analyze_cell_density(bboxes_arr, h, w, grid=n_tiles)
        tile_plan = decide_tiling_plan(cell_counts, cell_small_ratio,
                                       avg_per_cell, n_base)
    else:
        tile_plan = ["SKIP"] * (n_tiles * n_tiles)

    n_tiled_cells = sum(1 for p in tile_plan if p == "TILE")

    # --- Pass 2: SAM2 on under-segmented cells ---
    tile_geometry = compute_tiles(h, w, n_tiles, overlap)
    all_masks = list(base_masks)
    all_bboxes = list(base_bboxes)

    for ci, (y1, x1, y2, x2) in enumerate(tile_geometry):
        if tile_plan[ci] != "TILE":
            continue

        tile_img = image[y1:y2, x1:x2]
        th, tw = y2 - y1, x2 - x1
        raw_tile = amg.generate(tile_img)
        tile_masks, tile_bboxes = extract_candidates(raw_tile, th, tw)

        # Map tile-local to global coordinates
        for m, (lx, ly, lw, lh) in zip(tile_masks, tile_bboxes):
            global_mask = np.zeros((h, w), dtype=np.uint8)
            # Place tile mask at the right location
            m_region = np.zeros((th, tw), dtype=np.uint8)
            ly_i, lx_i = int(ly), int(lx)
            lh_i, lw_i = int(lh), int(lw)
            m_region[ly_i:ly_i+lh_i, lx_i:lx_i+lw_i] = m[ly_i:ly_i+lh_i, lx_i:lx_i+lw_i]
            global_mask[y1:y2, x1:x2] = m_region
            global_x1 = x1 + lx
            global_y1 = y1 + ly
            all_masks.append(global_mask)
            all_bboxes.append([float(global_x1), float(global_y1),
                              float(lw), float(lh)])

    if len(all_masks) == 0:
        return None

    # --- Merge base + tiled candidates ---
    keep_idx = merge_masks_by_iou(all_masks, all_bboxes, iou_thresh=0.7)
    masks = [all_masks[i] for i in keep_idx]
    bboxes = [all_bboxes[i] for i in keep_idx]
    n_merged = len(masks)

    if n_merged == 0:
        return None

    # --- Dot-based matching ---
    points = ann_entry.get("points", [])
    match = dot_based_matching(masks, points, class_idx, h, w)

    # --- Near-duplicate dedup ---
    order = sorted(range(n_merged), key=lambda i: -float(masks[i].sum()))
    kept_idx = []
    for i in order:
        dup = False
        mi = masks[i].astype(bool)
        for j in kept_idx:
            mj = masks[j].astype(bool)
            inter = float(np.logical_and(mi, mj).sum())
            union = float(np.logical_or(mi, mj).sum())
            if union > 0 and inter / union > 0.9:
                dup = True
                break
        if not dup:
            kept_idx.append(i)
    kept_idx = sorted(kept_idx)
    masks = [masks[i] for i in kept_idx]
    bboxes_sorted = [bboxes[i] for i in kept_idx]
    for k in match:
        match[k] = match[k][kept_idx]
    n_final = len(masks)

    if n_final == 0:
        return None

    # --- DINOv2 3-view encoding ---
    masked_crops, box_crops, ctx_crops = [], [], []
    for i in range(n_final):
        bb = (bboxes_sorted[i][0], bboxes_sorted[i][1],
              bboxes_sorted[i][0] + bboxes_sorted[i][2],
              bboxes_sorted[i][1] + bboxes_sorted[i][3])
        mc, bc, cc = build_three_crops(image, masks[i], bb)
        masked_crops.append(mc)
        box_crops.append(bc)
        ctx_crops.append(cc)

    z = encoder.encode_views(masked_crops, box_crops, ctx_crops, batch_size=64)

    return {
        "img_id": os.path.splitext(file_name)[0],
        "file_name": file_name,
        "class_name": class_name,
        "gt_count": len(points),
        "z": z.float(),
        "bbox": torch.tensor(bboxes_sorted, dtype=torch.float32),
        "matched_class": torch.from_numpy(match["matched_class"]).long(),
        "matched_instance_id": torch.from_numpy(match["matched_instance_id"]).long(),
        "iou": torch.from_numpy(match["coverage"]).float(),
        "purity": torch.from_numpy(match["purity"]).float(),
        "coverage": torch.from_numpy(match["coverage"]).float(),
        "valid": torch.from_numpy(match["valid"]).float(),
        "is_part": torch.zeros(n_final),
        "is_countable": torch.ones(n_final),
        "masks_rle": encode_masks_rle(masks),
        "height": int(h),
        "width": int(w),
        # Debug info
        "n_base": n_base,
        "n_tiled_cells": n_tiled_cells,
        "tile_plan": tile_plan,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default="/home/czp/official_code/dataset/FSC147/annotation_FSC147_384.json")
    ap.add_argument("--img-dir", default="/home/czp/official_code/dataset/FSC147/images_384_VarV2")
    ap.add_argument("--images-file", required=True, help="JSON list of image filenames to process")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--class-map", default="/home/czp/official_code/dataset/FSC147/ImageClasses_FSC147.txt")
    ap.add_argument("--categories-json", default="result/checkpoints/text_prototypes_fsc147_categories.json")
    ap.add_argument("--tiles", type=int, default=2)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--pts-per-side", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device

    # Load annotations and class mappings
    ann = json.load(open(args.ann))
    cats = json.load(open(args.categories_json))["categories"]
    name_to_idx = {c["name"]: c["contiguous_id"] for c in cats}

    img_to_class = {}
    with open(args.class_map) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split("\t")
            if len(parts) >= 2:
                img_to_class[parts[0]] = parts[1]

    # Load image list
    with open(args.images_file) as f:
        img_files = json.load(f)
    print(f"[init] {len(img_files)} images to process (adaptive two-pass tiling)")

    # Build SAM2
    print(f"[init] Building SAM2 AMG (pts={args.pts_per_side}, {args.tiles}×{args.tiles} grid)...")
    t0 = time.time()
    amg = build_sam2_amg(device, args.pts_per_side)
    print(f"[init] SAM2 ready in {time.time() - t0:.0f}s")
    encoder = DINOv2RegionEncoder(device=device)

    n_ok = n_skip = n_error = 0
    n_total_tiled = 0
    t_start = time.time()

    for i, fn in enumerate(img_files):
        out_path = os.path.join(args.out_dir, f"{os.path.splitext(fn)[0]}.pt")
        if os.path.exists(out_path):
            n_skip += 1
            continue

        # Find class
        class_name = img_to_class.get(fn, "")
        if not class_name:
            print(f"  [warn] No class for {fn}")
            n_error += 1
            continue
        class_idx = name_to_idx.get(class_name, -1)
        if class_idx < 0:
            print(f"  [warn] Unknown class: {class_name}")
            n_error += 1
            continue

        entry = ann.get(fn)
        if entry is None:
            n_error += 1
            continue

        img_path = entry.get("img_path", "")
        if not img_path or not os.path.exists(img_path):
            alt = os.path.join(args.img_dir, fn)
            if os.path.exists(alt):
                img_path = alt
            else:
                print(f"  [warn] No image for {fn}")
                n_error += 1
                continue

        try:
            image = np.array(Image.open(img_path).convert("RGB"))
        except Exception as e:
            print(f"  [warn] Cannot load {fn}: {e}")
            n_error += 1
            continue

        t_img = time.time()
        result = process_image_adaptive(
            image, fn, entry, class_idx, class_name,
            amg, encoder, args.tiles, args.overlap,
        )
        if result is None:
            print(f"  [warn] No candidates for {fn}")
            n_error += 1
            continue

        torch.save(result, out_path)
        n_ok += 1
        n_total_tiled += result["n_tiled_cells"]
        dt = time.time() - t_img

        gt = result["gt_count"]
        nc = result["z"].shape[0]
        tp = result["tile_plan"]
        tp_str = "".join("T" if p == "TILE" else "." for p in tp)
        print(f"  [{i+1}/{len(img_files)}] {fn}: GT={gt}, base={result['n_base']}, "
              f"final={nc}, plan=[{tp_str}], dt={dt:.1f}s  (ok={n_ok} skip={n_skip})")

    elapsed = time.time() - t_start
    print(f"\nDone: {n_ok} processed, {n_skip} skipped, {n_error} errors")
    print(f"Total time: {elapsed/60:.1f}min → {args.out_dir}")
    print(f"Avg tiled cells per image: {n_total_tiled/max(n_ok,1):.1f}")

    # Quick SAM2 recall stats
    files = sorted(Path(args.out_dir).glob("*.pt"))
    if files:
        gts, valids = [], []
        for f in files:
            d = torch.load(f, map_location="cpu", weights_only=False)
            gts.append(int(d["gt_count"]))
            valids.append(int((d["valid"] > 0).sum()))
        gts = np.array(gts); valids = np.array(valids)
        recall = valids.sum() / max(gts.sum(), 1)
        mae = np.mean(np.abs(valids - gts))
        print(f"Oracle stats ({len(files)} files):")
        print(f"  Mean GT={gts.mean():.1f}, SAM2 recall={recall:.1%}, Oracle MAE={mae:.2f}")


if __name__ == "__main__":
    main()
