"""快速重处理：复用已有 SAM2 mask，重算三路 DINOv2 特征 + 统一 dot-based matching。

不重新跑 SAM2（省 ~1.5h），只做：
    1. 三路 DINOv2 编码 (384 → 1152 dim)
    2. Dot-based matching（对齐测试缓存的 valid/purity 标准）

预计时间 (RTX 4090, 3657 图): ~20 min
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from code.candidates.crops import build_three_crops
from code.encoders.dinov2_encoder import DINOv2RegionEncoder


def dot_based_matching(
    masks: List[np.ndarray],
    points: List,
    class_idx: int,
    h: int, w: int,
    tau_purity: float = 0.0,  # 只要有 dot 覆盖就 valid
) -> dict:
    """Dot-based candidate-GT matching（对齐测试缓存标准）。"""
    n_cand = len(masks)
    pts_int = []
    for x, y in points:
        xi, yi = int(round(float(x))), int(round(float(y)))
        if 0 <= xi < w and 0 <= yi < h:
            pts_int.append((xi, yi))
    n_dots = len(pts_int)

    purity = np.zeros(n_cand, dtype=np.float32)
    coverage = np.zeros(n_cand, dtype=np.float32)
    iou_arr = np.zeros(n_cand, dtype=np.float32)
    matched_class = np.full(n_cand, class_idx, dtype=np.int64)
    matched_instance_id = np.full(n_cand, -1, dtype=np.int64)
    valid = np.zeros(n_cand, dtype=np.float32)

    for i, m in enumerate(masks):
        area = float(m.sum())
        if area == 0:
            continue
        covered_dots = [di for di, (xi, yi) in enumerate(pts_int) if m[yi, xi]]
        dots_covered = len(covered_dots)
        purity[i] = dots_covered / max(area, 1.0)  # dot density
        coverage[i] = dots_covered / max(n_dots, 1)
        iou_arr[i] = min(coverage[i], 1.0)
        area_ratio = area / (h * w)
        if dots_covered >= 1 and 1e-4 < area_ratio < 0.95 and purity[i] >= tau_purity:
            valid[i] = 1.0
            matched_instance_id[i] = covered_dots[0]

    return {
        "purity": purity,
        "coverage": coverage,
        "iou": iou_arr,
        "matched_class": matched_class,
        "matched_instance_id": matched_instance_id,
        "valid": valid,
    }


def process_image(
    cache_path: str,
    image_dir: str,
    out_dir: str,
    ann: dict,
    img_to_class: dict,
    name_to_idx: dict,
    encoder: DINOv2RegionEncoder,
) -> Optional[str]:
    """处理单张图，返回输出路径。"""
    d = torch.load(cache_path, map_location="cpu")
    file_name = d.get("file_name") or f"{d.get('img_id')}.jpg"
    img_path = os.path.join(image_dir, file_name)
    if not os.path.exists(img_path):
        return None

    image = np.array(Image.open(img_path).convert("RGB"))
    h, w = image.shape[:2]

    # 解码 mask
    masks_rle = d["masks_rle"]
    masks = []
    for r in masks_rle:
        m = mask_utils.decode(r).astype(bool)
        if m.shape == (h, w):
            masks.append(m)
    n_cand = len(masks)
    if n_cand == 0:
        return None

    # 获取类别
    class_name = img_to_class.get(file_name, d.get("class_name", "unknown"))
    class_idx = name_to_idx.get(class_name, -1)
    if class_idx < 0:
        return None

    # 三路 crop + DINOv2
    bbox = d["bbox"].numpy()  # [N,4] XYWH
    masked_crops, box_crops, ctx_crops = [], [], []
    for i in range(n_cand):
        x, y, bw, bh = bbox[i]
        bb = (x, y, x + bw, y + bh)
        mc, bc, cc = build_three_crops(image, masks[i].astype(np.uint8), bb)
        masked_crops.append(mc)
        box_crops.append(bc)
        ctx_crops.append(cc)

    z_new = encoder.encode_views(masked_crops, box_crops, ctx_crops, batch_size=64)

    # Dot-based matching
    entry = ann.get(file_name, {})
    points = entry.get("points", [])
    match = dot_based_matching(masks, points, class_idx, h, w)

    # 输出
    masks_rle_out = []
    for m in masks:
        mm = np.asfortranarray(np.asarray(m).astype(np.uint8))
        rle = mask_utils.encode(mm)
        counts = rle["counts"]
        if isinstance(counts, bytes):
            counts = counts.decode("ascii")
        masks_rle_out.append({"size": [int(rle["size"][0]), int(rle["size"][1])], "counts": counts})

    gt_count = len(points)

    out = {
        "img_id": os.path.splitext(file_name)[0],
        "file_name": file_name,
        "class_name": class_name,
        "gt_count": gt_count,
        "z": z_new.float(),
        "bbox": torch.tensor(bbox, dtype=torch.float32),
        "matched_class": torch.from_numpy(match["matched_class"]).long(),
        "matched_instance_id": torch.from_numpy(match["matched_instance_id"]).long(),
        "iou": torch.from_numpy(match["iou"]).float(),
        "purity": torch.from_numpy(match["purity"]).float(),
        "coverage": torch.from_numpy(match["coverage"]).float(),
        "valid": torch.from_numpy(match["valid"]).float(),
        "is_part": torch.zeros(n_cand),
        "is_countable": torch.ones(n_cand),
        "masks_rle": masks_rle_out,
        "height": int(h),
        "width": int(w),
    }

    out_path = os.path.join(out_dir, os.path.basename(cache_path))
    torch.save(out, out_path)
    return file_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True, help="原始训练缓存（只读）")
    ap.add_argument("--out-dir", required=True, help="输出目录（新缓存）")
    ap.add_argument("--img-dir", default="/home/czp/official_code/dataset/FSC147/images_384_VarV2")
    ap.add_argument("--ann", default="/home/czp/official_code/dataset/FSC147/annotation_FSC147_384.json")
    ap.add_argument("--class-map", default="/home/czp/official_code/dataset/FSC147/ImageClasses_FSC147.txt")
    ap.add_argument("--categories-json", default="result/checkpoints/text_prototypes_fsc147_categories.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()

    import json
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
    print(f"[init] {len(img_to_class)} class mappings, {len(name_to_idx)} categories")

    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(args.cache_dir) if f.endswith(".pt"))
    if args.limit > 0:
        files = files[:args.limit]
    print(f"[init] {len(files)} cache files to process")

    encoder = DINOv2RegionEncoder(device=args.device)
    print(f"[init] DINOv2 encoder on {args.device}")

    t0 = time.time()
    n_ok = n_skip = 0
    for i, fn in enumerate(files):
        cache_path = os.path.join(args.cache_dir, fn)
        result = process_image(cache_path, args.img_dir, args.out_dir, ann, img_to_class, name_to_idx, encoder)
        if result:
            n_ok += 1
        else:
            n_skip += 1

        if (i + 1) % 200 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(files) - (i + 1)) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(files)}] ok={n_ok} skip={n_skip} "
                  f"rate={rate:.1f}/s ETA={eta/60:.0f}min")

    elapsed = time.time() - t0
    print(f"\nDone: {n_ok} processed, {n_skip} skipped in {elapsed/60:.1f}min")


if __name__ == "__main__":
    main()
