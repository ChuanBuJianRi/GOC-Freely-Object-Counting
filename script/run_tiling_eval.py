"""P2-4: Multi-scale Tiling for Dense Scene Counting.

Splits images into overlapping tiles, runs SAM2 AMG on each tile, merges
overlapping candidates, and evaluates against the non-tiled baseline.

This addresses the SAM2 recall bottleneck identified in P2-1 (PUCPR+) where
dense parking lots at pts=32 only achieve ~84% candidate recall.

Usage:
    # Preprocess with tiling
    python script/run_tiling_eval.py \
        --img-dir datasets/PUCPR+_devkit/data/Images \
        --ann-dir datasets/PUCPR+_devkit/data/Annotations \
        --out-dir /home/czp/ws_yiyang/ovcud_cache/pucpr_tiled \
        --image-set datasets/PUCPR+_devkit/data/ImageSets/test.txt \
        --tiles 2 --overlap 0.25 --pts-per-side 32

    # Evaluate (same as P1 ablation but with tiled cache)
    python script/run_p1_ablations.py \
        --cache-dir /home/czp/ws_yiyang/ovcud_cache/pucpr_tiled \
        --cat-ckpt result/checkpoints/category_cosine_pts32.pt \
        --rel-ckpt result/checkpoints/fsc147_relation_pts32_exp5c.pt \
        --text-prototypes result/checkpoints/text_prototypes_fsc147.pt \
        --dataset carpk --out result/logs/p2_tiled_eval.json
"""

from __future__ import annotations

import argparse, os, sys, time
from pathlib import Path
from typing import List

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
# Tiling
# ---------------------------------------------------------------------------
def compute_tiles(h, w, n_tiles, overlap):
    """Compute tile bounding boxes [y1, x1, y2, x2] for an image.

    Args:
        h, w: image dimensions
        n_tiles: number of tiles per dimension (2 → 2×2=4 tiles)
        overlap: overlap fraction (0.25 → 25% overlap between adjacent tiles)
    """
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
            # Adjust last row/col to cover edge
            if ty == n_tiles - 1:
                y2 = h
            if tx == n_tiles - 1:
                x2 = w
            tiles.append((y1, x1, y2, x2))
    return tiles


def merge_masks_by_iou(masks, bboxes, iou_thresh=0.7):
    """Merge masks that have high IoU overlap (keep the larger one)."""
    n = len(masks)
    if n <= 1:
        return list(range(n))

    # Compute IoU matrix
    iou_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            inter = float((masks[i] & masks[j]).sum())
            union = float((masks[i] | masks[j]).sum())
            iou = inter / max(union, 1.0)
            iou_mat[i, j] = iou_mat[j, i] = iou

    # Greedy merge: sort by area descending, keep if no high IoU with already kept
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
# Annotation Loading
# ---------------------------------------------------------------------------
def load_pucpr_annotations(ann_dir, img_name, h, w):
    """Load PUCPR+ bbox annotations, return dot (center) coordinates."""
    base = os.path.splitext(img_name)[0]
    ann_path = os.path.join(ann_dir, f"{base}.txt")
    if not os.path.exists(ann_path):
        return []

    points = []
    with open(ann_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 4:
                continue
            try:
                x1, y1, x2, y2 = map(float, parts[:4])
            except ValueError:
                continue
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            if 0 <= cx < w and 0 <= cy < h:
                points.append((cx, cy))
    return points


def dot_matching(masks, points, class_idx, h, w):
    """Match candidate masks to GT dots."""
    pts_int = [(int(round(float(x))), int(round(float(y)))) for x, y in points
               if 0 <= int(round(float(x))) < w and 0 <= int(round(float(y))) < h]
    n_dots = len(pts_int)
    n_cand = len(masks)

    purity = np.zeros(n_cand, dtype=np.float32)
    coverage = np.zeros(n_cand, dtype=np.float32)
    valid = np.zeros(n_cand, dtype=np.float32)
    matched_instance_id = np.full(n_cand, -1, dtype=np.int64)

    for i, m in enumerate(masks):
        area = float(m.sum())
        if area == 0:
            continue
        covered_dots = []
        for di, (xi, yi) in enumerate(pts_int):
            if m[yi, xi]:
                covered_dots.append(di)
        dc = len(covered_dots)
        purity[i] = dc / max(area, 1.0)
        coverage[i] = dc / max(n_dots, 1)
        ar = area / (h * w)
        if dc >= 1 and 1e-4 < ar < 0.95:
            valid[i] = 1.0
            matched_instance_id[i] = covered_dots[0]
    return {
        "purity": purity, "coverage": coverage, "valid": valid,
        "matched_class": np.full(n_cand, class_idx, dtype=np.int64),
        "matched_instance_id": matched_instance_id,
    }


# ---------------------------------------------------------------------------
# Process with tiling
# ---------------------------------------------------------------------------
def process_image_tiled(image, file_name, ann_points, class_idx, class_name,
                        amg, encoder, n_tiles, overlap):
    h, w = image.shape[:2]
    tiles = compute_tiles(h, w, n_tiles, overlap)
    print(f"  [tiling] {len(tiles)} tiles ({n_tiles}x{n_tiles}), overlap={overlap:.0%}")

    all_masks = []
    all_bboxes = []

    for ti, (y1, x1, y2, x2) in enumerate(tiles):
        tile_img = image[y1:y2, x1:x2]
        tile_h, tile_w = y2 - y1, x2 - x1

        # Run SAM2 on tile
        raw = amg.generate(tile_img)

        for r in raw:
            m = np.asarray(r["segmentation"]).astype(np.uint8)
            area = float(m.sum())
            if area == 0 or area / (tile_h * tile_w) < 1e-4 or area / (tile_h * tile_w) > 0.95:
                continue
            ys, xs = np.where(m)
            if xs.size == 0:
                continue
            lx1, ly1 = int(xs.min()), int(ys.min())
            lx2, ly2 = int(xs.max()) + 1, int(ys.max()) + 1
            if (lx2 - lx1) < 4 or (ly2 - ly1) < 4:
                continue

            # Map tile-local coordinates to image-global
            global_mask = np.zeros((h, w), dtype=np.uint8)
            global_mask[y1:y2, x1:x2] = m
            global_x1 = x1 + lx1
            global_y1 = y1 + ly1
            global_w = lx2 - lx1
            global_h = ly2 - ly1

            all_masks.append(global_mask)
            all_bboxes.append([float(global_x1), float(global_y1),
                              float(global_w), float(global_h)])

        if ti == 0:
            print(f"    tile[{ti}]: {len(raw)} raw → {len(all_masks)} kept")

    n_raw = len(all_masks)
    print(f"  [tiling] total raw candidates: {n_raw}")

    if n_raw == 0:
        return None

    # Merge overlapping candidates from different tiles
    keep_idx = merge_masks_by_iou(all_masks, all_bboxes, iou_thresh=0.7)
    masks = [all_masks[i] for i in keep_idx]
    bboxes = [all_bboxes[i] for i in keep_idx]
    n_merged = len(masks)
    print(f"  [tiling] after merge: {n_merged} candidates (removed {n_raw - n_merged})")

    if n_merged == 0:
        return None

    # Matching to GT
    match = dot_matching(masks, ann_points, class_idx, h, w)
    for k in match:
        match[k] = match[k][:n_merged]

    # DINOv2 3-view encoding
    masked_crops, box_crops, ctx_crops = [], [], []
    for i in range(n_merged):
        bb = (bboxes[i][0], bboxes[i][1],
              bboxes[i][0] + bboxes[i][2], bboxes[i][1] + bboxes[i][3])
        mc, bc, cc = build_three_crops(image, masks[i], bb)
        masked_crops.append(mc)
        box_crops.append(bc)
        ctx_crops.append(cc)

    z = encoder.encode_views(masked_crops, box_crops, ctx_crops, batch_size=64)

    return {
        "img_id": os.path.splitext(file_name)[0],
        "file_name": file_name,
        "class_name": class_name,
        "gt_count": len(ann_points),
        "z": z.float(),
        "bbox": torch.tensor(bboxes, dtype=torch.float32),
        "matched_class": torch.from_numpy(match["matched_class"]).long(),
        "matched_instance_id": torch.from_numpy(match["matched_instance_id"]).long(),
        "iou": torch.from_numpy(match["coverage"]).float(),
        "purity": torch.from_numpy(match["purity"]).float(),
        "coverage": torch.from_numpy(match["coverage"]).float(),
        "valid": torch.from_numpy(match["valid"]).float(),
        "is_part": torch.zeros(n_merged),
        "is_countable": torch.ones(n_merged),
        "masks_rle": encode_masks_rle(masks),
        "height": int(h),
        "width": int(w),
    }


def main():
    ap = argparse.ArgumentParser(description="P2-4: Multi-scale tiling preprocessing + eval")
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--ann-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--image-set", default="", help="ImageSet file")
    ap.add_argument("--tiles", type=int, default=2, help="Tiles per dimension (2→4 tiles)")
    ap.add_argument("--overlap", type=float, default=0.25, help="Tile overlap fraction")
    ap.add_argument("--pts-per-side", type=int, default=32)
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device
    CLASS_IDX = 29  # cars
    CLASS_NAME = "cars"

    # Collect images
    img_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    if args.image_set and os.path.exists(args.image_set):
        with open(args.image_set) as f:
            wanted_ids = {line.strip() for line in f if line.strip()}
        all_imgs = {os.path.splitext(fn)[0]: fn for fn in os.listdir(args.img_dir)
                    if os.path.splitext(fn)[1] in img_exts}
        img_files = []
        for wid in wanted_ids:
            if wid in all_imgs:
                img_files.append(all_imgs[wid])
        img_files.sort()
        print(f"[init] {len(img_files)} images from image_set")
    else:
        img_files = sorted([fn for fn in os.listdir(args.img_dir)
                           if os.path.splitext(fn)[1] in img_exts])

    if args.limit > 0:
        img_files = img_files[:args.limit]

    print(f"[init] Processing {len(img_files)} images")
    print(f"[init] Tiling: {args.tiles}x{args.tiles} = {args.tiles**2} tiles, "
          f"overlap={args.overlap:.0%}, pts={args.pts_per_side}")

    # Load models
    print("[init] Building SAM2 AMG...")
    t0 = time.time()
    amg = build_sam2_amg(device, args.pts_per_side)
    print(f"[init] SAM2 ready in {time.time() - t0:.0f}s")
    encoder = DINOv2RegionEncoder(device=device)
    print(f"[init] DINOv2 ready")

    n_ok = n_skip = 0
    t_start = time.time()

    for i, fn in enumerate(img_files):
        out_path = os.path.join(args.out_dir, f"{os.path.splitext(fn)[0]}.pt")
        if os.path.exists(out_path):
            n_skip += 1
            continue

        img_path = os.path.join(args.img_dir, fn)
        try:
            image = np.array(Image.open(img_path).convert("RGB"))
        except Exception as e:
            print(f"  [warn] Cannot load {fn}: {e}")
            continue

        h, w = image.shape[:2]
        ann_points = load_pucpr_annotations(args.ann_dir, fn, h, w)
        if not ann_points:
            print(f"  [warn] No annotations for {fn}, skipping")
            continue

        t_img = time.time()
        result = process_image_tiled(image, fn, ann_points, CLASS_IDX, CLASS_NAME,
                                     amg, encoder, args.tiles, args.overlap)
        if result is None:
            continue

        torch.save(result, out_path)
        n_ok += 1
        dt = time.time() - t_img

        print(f"  [{i+1}/{len(img_files)}] {fn}: gt={result['gt_count']}, "
              f"cands={result['z'].shape[0]}, dt={dt:.1f}s")

    elapsed = time.time() - t_start
    print(f"\nDone: {n_ok} processed, {n_skip} skipped")
    print(f"Total time: {elapsed / 60:.1f}min → {args.out_dir}")

    # ---- Quick oracle comparison ----
    print(f"\n{'='*60}")
    print("Tiling vs Non-Tiled Oracle Comparison")
    print(f"{'='*60}")

    for label, cache_dir in [("Tiled", args.out_dir)]:
        files = sorted(Path(cache_dir).glob("*.pt"))
        gts, valids = [], []
        for f in files:
            d = torch.load(f, map_location="cpu")
            gts.append(int(d["gt_count"]))
            valids.append(int((d["valid"] > 0).sum()))
        gts = np.array(gts); valids = np.array(valids)
        mae = np.mean(np.abs(valids - gts))
        recall = valids.sum() / max(gts.sum(), 1)
        print(f"  {label}: n={len(gts)}, mean_GT={gts.mean():.1f}, "
              f"SAM2_recall={recall:.1%}, oracle_MAE={mae:.2f}")


if __name__ == "__main__":
    main()
