"""Preprocess OmniCount-191 test images for OV-CUD evaluation.

Reads COCO-format annotations from OmniCount-191, converts bbox centers to
dot annotations, runs SAM2 AMG + DINOv2 3-view encoding.

OmniCount-191 has MULTI-LABEL images: each image can contain multiple object
categories. We store both per-class counts (for class-aware evaluation) and
total count (for class-agnostic evaluation).

Usage:
    # Preprocess all OmniCount test images
    python script/preprocess_omnicount.py \
        --omnicount-dir /home/czp/official_code/dataset/omnicount/OmniCount-191 \
        --out-dir /home/czp/ws_yiyang/ovcud_cache/omnicount_test \
        --pts-per-side 32
"""

from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
# COCO Annotation Loading
# ---------------------------------------------------------------------------
def load_omnicount_coco(omnicount_dir: str) -> List[dict]:
    """Load all OmniCount test images with their annotations.

    Returns list of dicts with:
        img_path, file_name, width, height, annotations: [{class_name, cx, cy, bbox}]
    """
    categories = ["Birds", "Fruits", "Pets", "Satellite", "Supermarket", "Urban", "Wild"]
    entries = []

    for cat in categories:
        test_dir = os.path.join(omnicount_dir, cat, "test")
        coco_file = os.path.join(test_dir, "_annotations.coco.json")
        if not os.path.exists(coco_file):
            print(f"  [warn] Missing: {coco_file}")
            continue

        with open(coco_file) as f:
            ann = json.load(f)

        cat_map = {c["id"]: c["name"] for c in ann["categories"]}
        img_map = {img["id"]: img for img in ann["images"]}

        # Group annotations by image_id
        img_anns: dict = {}
        for a in ann["annotations"]:
            img_id = a["image_id"]
            if img_id not in img_anns:
                img_anns[img_id] = []
            cls_name = cat_map[a["category_id"]]
            bbox = a["bbox"]  # [x, y, w, h]
            cx = bbox[0] + bbox[2] / 2.0
            cy = bbox[1] + bbox[3] / 2.0
            img_anns[img_id].append({
                "class_name": cls_name,
                "cx": float(cx),
                "cy": float(cy),
                "bbox": [float(x) for x in bbox],
            })

        for img_id, img_info in img_map.items():
            img_path = os.path.join(test_dir, img_info["file_name"])
            if not os.path.exists(img_path):
                continue
            anns = img_anns.get(img_id, [])
            if not anns:
                continue

            # Compute per-class counts
            class_counts: dict = {}
            for a in anns:
                cname = a["class_name"]
                class_counts[cname] = class_counts.get(cname, 0) + 1
            total_count = sum(class_counts.values())

            # Build global class index mapping (0..C-1)
            unique_classes = sorted(class_counts.keys())

            entries.append({
                "img_path": img_path,
                "file_name": img_info["file_name"],
                "category": cat,
                "width": img_info["width"],
                "height": img_info["height"],
                "annotations": anns,
                "class_counts": class_counts,
                "unique_classes": unique_classes,
                "total_count": total_count,
            })

    return entries


# ---------------------------------------------------------------------------
# Dot-based matching (class-agnostic: all dots belong to class 0)
# ---------------------------------------------------------------------------
def dot_based_matching(
    masks: List[np.ndarray],
    points: List[Tuple[float, float]],
    h: int, w: int,
) -> Dict[str, np.ndarray]:
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
        "purity": purity,
        "coverage": coverage,
        "valid": valid,
    }


# ---------------------------------------------------------------------------
# Single image processing
# ---------------------------------------------------------------------------
def process_image(
    image: np.ndarray,
    entry: dict,
    amg,
    encoder: DINOv2RegionEncoder,
) -> Optional[dict]:
    h, w = image.shape[:2]

    # SAM2 candidate generation
    raw = amg.generate(image)
    masks = []
    bboxes = []
    for r in raw:
        m = np.asarray(r["segmentation"]).astype(np.uint8)
        area = float(m.sum())
        if area == 0:
            continue
        ar = area / (h * w)
        if ar < 1e-4 or ar > 0.95:
            continue
        ys, xs = np.where(m)
        if xs.size == 0:
            continue
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
        if (x2 - x1) < 4 or (y2 - y1) < 4:
            continue
        masks.append(m)
        bboxes.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])

    n_cand = len(masks)
    if n_cand == 0:
        return None

    # Dot-based matching (class-agnostic: all dots belong to class 0)
    all_points = [(a["cx"], a["cy"]) for a in entry["annotations"]]
    match = dot_based_matching(masks, all_points, h, w)

    # Near-duplicate dedup (high IoU masks)
    order = sorted(range(n_cand), key=lambda i: -float(masks[i].sum()))
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
        "img_id": os.path.splitext(entry["file_name"])[0],
        "file_name": entry["file_name"],
        "category": entry["category"],
        "class_counts": entry["class_counts"],
        "unique_classes": entry["unique_classes"],
        "gt_count": entry["total_count"],
        "z": z.float(),
        "bbox": torch.tensor(bboxes, dtype=torch.float32),
        "matched_class": torch.zeros(n_cand, dtype=torch.long),  # class-agnostic
        "matched_instance_id": torch.arange(n_cand, dtype=torch.long),
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
    ap = argparse.ArgumentParser(description="Preprocess OmniCount-191 test images")
    ap.add_argument("--omnicount-dir", default="/home/czp/official_code/dataset/omnicount/OmniCount-191")
    ap.add_argument("--out-dir", default="/home/czp/ws_yiyang/ovcud_cache/omnicount_test")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pts-per-side", type=int, default=32)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device

    # Load OmniCount data
    print("[init] Loading OmniCount annotations...")
    entries = load_omnicount_coco(args.omnicount_dir)
    print(f"[init] {len(entries)} test images loaded")

    if args.limit > 0:
        entries = entries[:args.limit]
        print(f"[init] Limited to {len(entries)} images")

    # Per-category stats
    cat_counts = {}
    for e in entries:
        cat_counts[e["category"]] = cat_counts.get(e["category"], 0) + 1
    for cat, n in sorted(cat_counts.items()):
        print(f"  {cat}: {n} images")

    # Total objects
    total_obj = sum(e["total_count"] for e in entries)
    unique_classes = set()
    for e in entries:
        unique_classes.update(e["unique_classes"])
    print(f"[init] {total_obj} total objects, {len(unique_classes)} unique classes")
    print(f"[init] Mean GT per image: {total_obj/len(entries):.1f}")

    # Load models
    print(f"[init] Building SAM2 AMG (pts={args.pts_per_side})...")
    t0 = time.time()
    amg = build_sam2_amg(device, args.pts_per_side)
    print(f"[init] SAM2 ready in {time.time() - t0:.0f}s")
    encoder = DINOv2RegionEncoder(device=device)
    print(f"[init] DINOv2 ready")

    n_ok = n_skip = n_error = 0
    t_start = time.time()

    for i, entry in enumerate(entries):
        file_name = entry["file_name"]
        out_path = os.path.join(args.out_dir, f"{os.path.splitext(file_name)[0]}.pt")

        if args.skip_existing and os.path.exists(out_path):
            n_skip += 1
            continue

        try:
            image = np.array(Image.open(entry["img_path"]).convert("RGB"))
        except Exception as e:
            print(f"  [warn] Cannot load {file_name}: {e}")
            n_error += 1
            continue

        t_img = time.time()
        result = process_image(image, entry, amg, encoder)
        if result is None:
            n_error += 1
            print(f"  [warn] No candidates for {file_name}")
            continue

        torch.save(result, out_path)
        n_ok += 1
        dt = time.time() - t_img

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            rate = (n_ok) / max(elapsed, 0.01)
            eta = (len(entries) - i - 1) / max(rate, 0.01)
            print(f"  [{i+1}/{len(entries)}] ok={n_ok} skip={n_skip} "
                  f"dt={dt:.1f}s  rate={rate:.2f}/s  ETA={eta/60:.1f}min")

    elapsed = time.time() - t_start
    print(f"\nDone: {n_ok} processed, {n_skip} skipped, {n_error} errors")
    print(f"Total time: {elapsed/60:.1f}min → {args.out_dir}")

    # Quick stats
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
        print(f"\nOracle stats ({len(files)} files):")
        print(f"  Mean GT={gts.mean():.1f}, SAM2 recall={recall:.1%}, "
              f"Oracle MAE={mae:.2f}")


if __name__ == "__main__":
    main()
