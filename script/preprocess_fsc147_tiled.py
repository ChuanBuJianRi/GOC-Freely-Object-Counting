"""Preprocess FSC147 100+ images with 2×2 multi-scale tiling.

For each high-density image (GT > 100), splits it into 2×2 overlapping tiles,
runs SAM2 AMG on each tile, merges overlapping candidates via IoU, then
encodes with DINOv2 3-view. This improves SAM2 candidate recall for dense scenes.

When --upscale is set, each tile is resized to the original image resolution
before SAM2 AMG, then masks are scaled back. This makes small objects larger
in each tile view, further improving recall for dense-small-object scenes.

Usage:
    # Standard tiling
    python script/preprocess_fsc147_tiled.py \
        --ann /home/czp/official_code/dataset/FSC147/annotation_FSC147_384.json \
        --img-dir /home/czp/official_code/dataset/FSC147/images_384_VarV2 \
        --images-file result/logs/fsc147_test_100plus.json \
        --out-dir /home/czp/ws_yiyang/ovcud_cache/fsc147_test_tiled \
        --tiles 2 --overlap 0.25 --pts-per-side 32

    # Upscaled tiling (tile → upscale → SAM2 → downscale masks)
    python script/preprocess_fsc147_tiled.py \
        --ann ... --img-dir ... --images-file ... \
        --out-dir /home/czp/ws_yiyang/ovcud_cache/fsc147_test_tiled_2x2_upscale \
        --tiles 2 --overlap 0.25 --pts-per-side 32 --upscale
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
# Tiling
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


def bbox_iou_xywh(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    return inter / max(aw * ah + bw * bh - inter, 1.0)


def merge_masks_by_bbox(bboxes, iou_thresh=0.7):
    n = len(bboxes)
    if n <= 1:
        return list(range(n))
    areas = [b[2] * b[3] for b in bboxes]
    order = sorted(range(n), key=lambda i: areas[i], reverse=True)
    kept = []
    for i in order:
        dup = False
        for j in kept:
            if bbox_iou_xywh(bboxes[i], bboxes[j]) > iou_thresh:
                dup = True
                break
        if not dup:
            kept.append(i)
    kept.sort()
    return kept


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
    matched_instance_id = np.full(n_cand, -1, dtype=np.int64)

    for i, m in enumerate(masks):
        area = float(m.sum())
        if area == 0:
            continue
        covered_dots = []
        for dot_index, (xi, yi) in enumerate(pts_int):
            if m[yi, xi]:
                covered_dots.append(dot_index)
        dots_covered = len(covered_dots)
        purity[i] = dots_covered / max(area, 1.0)
        coverage[i] = dots_covered / max(n_dots_valid, 1)
        area_ratio = area / (h * w)
        if dots_covered >= 1 and 1e-4 < area_ratio < 0.95:
            valid[i] = 1.0
            matched_instance_id[i] = covered_dots[0]
    return {
        "purity": purity, "coverage": coverage,
        "valid": valid,
        "matched_class": np.full(n_cand, class_idx, dtype=np.int64),
        "matched_instance_id": matched_instance_id,
    }


# ---------------------------------------------------------------------------
# Process single image with tiling
# ---------------------------------------------------------------------------
def process_image_tiled(image, file_name, ann_entry, class_idx, class_name,
                         amg, encoder, n_tiles, overlap, upscale=False,
                         merge_mode="mask"):
    h, w = image.shape[:2]
    tiles = compute_tiles(h, w, n_tiles, overlap)

    all_masks = []
    all_bboxes = []

    for ti, (y1, x1, y2, x2) in enumerate(tiles):
        tile_img = image[y1:y2, x1:x2]
        th, tw = y2 - y1, x2 - x1

        if upscale:
            # Upscale tile to original image resolution so SAM2 sees
            # larger objects in dense scenes → better recall.
            tile_img_pil = Image.fromarray(tile_img)
            tile_img_pil = tile_img_pil.resize((w, h), Image.BILINEAR)
            tile_img = np.array(tile_img_pil)
            # Scale factors for mapping back
            sy, sx = th / h, tw / w

        raw = amg.generate(tile_img)
        for r in raw:
            m = np.asarray(r["segmentation"]).astype(np.uint8)
            area = float(m.sum())
            if area == 0:
                continue

            if upscale:
                # Scale mask back to tile coordinates
                m_pil = Image.fromarray(m)
                m_pil = m_pil.resize((tw, th), Image.NEAREST)
                m = np.array(m_pil).astype(np.uint8)
                # Recompute area after downscaling
                area = float(m.sum())
                if area == 0:
                    continue

            ar = area / (th * tw)
            if ar < 1e-4 or ar > 0.95:
                continue
            ys, xs = np.where(m)
            if xs.size == 0:
                continue
            lx1, ly1 = int(xs.min()), int(ys.min())
            lx2, ly2 = int(xs.max()) + 1, int(ys.max()) + 1
            if (lx2 - lx1) < 4 or (ly2 - ly1) < 4:
                continue
            # Map tile-local to global
            global_mask = np.zeros((h, w), dtype=np.uint8)
            global_mask[y1:y2, x1:x2] = m
            global_x1 = x1 + lx1; global_y1 = y1 + ly1
            global_w = lx2 - lx1; global_h = ly2 - ly1
            all_masks.append(global_mask)
            all_bboxes.append([float(global_x1), float(global_y1),
                              float(global_w), float(global_h)])

    if len(all_masks) == 0:
        return None

    # Merge overlapping candidates from different tiles. Mask IoU is the
    # original recipe; bbox mode is used for extreme rescue sweeps where
    # dense tiling can make full mask-pair IoU prohibitively slow.
    if merge_mode == "bbox":
        keep_idx = merge_masks_by_bbox(all_bboxes, iou_thresh=0.7)
    elif merge_mode == "mask":
        keep_idx = merge_masks_by_iou(all_masks, all_bboxes, iou_thresh=0.7)
    else:
        raise ValueError(f"unknown merge_mode: {merge_mode}")
    masks = [all_masks[i] for i in keep_idx]
    bboxes = [all_bboxes[i] for i in keep_idx]
    n_cand = len(masks)

    if n_cand == 0:
        return None

    # Dot-based matching
    points = ann_entry.get("points", [])
    match = dot_based_matching(masks, points, class_idx, h, w)

    # Near-duplicate dedup (high IoU masks within same tile)
    order = sorted(range(n_cand), key=lambda i: -float(masks[i].sum()))
    kept_idx = []
    for i in order:
        dup = False
        for j in kept_idx:
            if merge_mode == "bbox":
                if bbox_iou_xywh(bboxes[i], bboxes[j]) > 0.9:
                    dup = True
                    break
            else:
                mi = masks[i].astype(bool)
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
    bboxes = [bboxes[i] for i in kept_idx]
    for k in match:
        match[k] = match[k][kept_idx]
    n_cand = len(masks)

    if n_cand == 0:
        return None

    # DINOv2 3-view encoding
    masked_crops, box_crops, ctx_crops = [], [], []
    for i in range(n_cand):
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
        "gt_count": len(points),
        "z": z.float(),
        "bbox": torch.tensor(bboxes, dtype=torch.float32),
        "matched_class": torch.from_numpy(match["matched_class"]).long(),
        "matched_instance_id": torch.from_numpy(match["matched_instance_id"]).long(),
        "iou": torch.from_numpy(match["coverage"]).float(),
        "purity": torch.from_numpy(match["purity"]).float(),
        "coverage": torch.from_numpy(match["coverage"]).float(),
        "valid": torch.from_numpy(match["valid"]).float(),
        "is_part": torch.zeros(n_cand),
        "is_countable": torch.ones(n_cand),
        "masks_rle": encode_masks_rle(masks),
        "height": int(h),
        "width": int(w),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default="/home/czp/official_code/dataset/FSC147/annotation_FSC147_384.json")
    ap.add_argument("--img-dir", default="/home/czp/official_code/dataset/FSC147/images_384_VarV2")
    ap.add_argument("--images-file", default=None, help="JSON list of image filenames to process")
    ap.add_argument("--split-file", default=None, help="official split JSON used when --images-file is omitted")
    ap.add_argument("--split-key", default="val", help="split selected from --split-file")
    ap.add_argument("--min-count", type=int, default=0, help="minimum GT count for split-file filtering")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--class-map", default="/home/czp/official_code/dataset/FSC147/ImageClasses_FSC147.txt")
    ap.add_argument("--categories-json", default="result/checkpoints/text_prototypes_fsc147_categories.json")
    ap.add_argument("--tiles", type=int, default=2)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--pts-per-side", type=int, default=32)
    ap.add_argument("--upscale", action="store_true",
                    help="Upscale each tile to original image resolution before SAM2")
    ap.add_argument("--merge-mode", choices=["mask", "bbox"], default="mask")
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

    # Load image list. Split-derived lists avoid hand-maintained validation files.
    if args.images_file:
        with open(args.images_file) as f:
            img_files = json.load(f)
    elif args.split_file:
        split = json.load(open(args.split_file))
        if args.split_key not in split:
            raise KeyError(f"split key {args.split_key!r} not found in {args.split_file}")
        img_files = [
            fn for fn in split[args.split_key]
            if len(ann.get(fn, {}).get("points", [])) >= args.min_count
        ]
    else:
        ap.error("one of --images-file or --split-file is required")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        ap.error("--num-shards must be positive and --shard-index must be in range")
    total_images = len(img_files)
    img_files = img_files[args.shard_index::args.num_shards]
    print(
        f"[init] {len(img_files)}/{total_images} images to process "
        f"(shard {args.shard_index}/{args.num_shards})"
    )

    # Build SAM2
    print(f"[init] Building SAM2 AMG (pts={args.pts_per_side}, {args.tiles}×{args.tiles} tiles, "
          f"upscale={args.upscale})...")
    t0 = time.time()
    amg = build_sam2_amg(device, args.pts_per_side)
    print(f"[init] SAM2 ready in {time.time() - t0:.0f}s")
    encoder = DINOv2RegionEncoder(device=device)

    n_ok = n_skip = n_error = 0
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
        result = process_image_tiled(
            image, fn, entry, class_idx, class_name,
            amg, encoder, args.tiles, args.overlap,
            upscale=args.upscale,
            merge_mode=args.merge_mode,
        )
        if result is None:
            print(f"  [warn] No candidates for {fn}")
            n_error += 1
            continue

        torch.save(result, out_path)
        n_ok += 1
        dt = time.time() - t_img

        gt = result["gt_count"]
        nc = result["z"].shape[0]
        print(f"  [{i+1}/{len(img_files)}] {fn}: GT={gt}, cands={nc}, "
              f"dt={dt:.1f}s  (ok={n_ok} skip={n_skip})")

    elapsed = time.time() - t_start
    print(f"\nDone: {n_ok} processed, {n_skip} skipped, {n_error} errors")
    print(f"Total time: {elapsed/60:.1f}min → {args.out_dir}")

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
