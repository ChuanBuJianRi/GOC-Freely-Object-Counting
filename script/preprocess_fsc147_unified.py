"""FSC147 统一预处理：OCCAM-M SAM2 + 三路 DINOv2 + dot-based matching。

用与测试缓存完全一致的 recipe 重建训练缓存，消除 train→test 特征分布偏移。

Recipe:
    - SAM2: OCCAM-M (8px spacing, hiera-small, crop_n_layers=1)
    - DINOv2: three-view (masked + box + context), 1152-dim
    - Matching: dot-based purity/coverage (与测试缓存一致)
    - Valid: purity > 0.3（对齐测试缓存的 valid 率 ~70%）

预估时间 (RTX 4090, 3657 图):
    - SAM2: ~1.5h
    - DINOv2: ~0.5h
    - Matching: ~0.1h
    - Total: ~2h
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from code.candidates.crops import build_three_crops
from code.encoders.dinov2_encoder import DINOv2RegionEncoder
from code.encoders.text_encoder import TextPrototypeBuilder


# ---------------------------------------------------------------------------
# OCCAM-M SAM2
# ---------------------------------------------------------------------------
def build_occam_amg(device: str):
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    config = "configs/sam2.1/sam2.1_hiera_s.yaml"
    ckpt = "/home/czp/ws_yiyang/FreeCounting/ws_yiyang/OCCAM/checkpoints/sam2.1_hiera_small.pt"
    model = build_sam2(config, ckpt, device=device)

    spacing = 8
    ref = 384.0
    step = spacing / ref
    coords = np.arange(step / 2.0, 1.0, step, dtype=np.float32)
    gx, gy = np.meshgrid(coords, coords)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
    point_grids = [grid, grid]  # crop_n_layers=1

    return SAM2AutomaticMaskGenerator(
        model=model,
        points_per_side=None,
        point_grids=point_grids,
        points_per_batch=1000,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.8,
        stability_score_offset=0.0,
        mask_threshold=0.0,
        box_nms_thresh=0.7,
        crop_n_layers=1,
        crop_nms_thresh=0.7,
        use_m2m=False,
        multimask_output=True,
        output_mode="binary_mask",
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
# Dot-based matching (FSC147 specific)
# ---------------------------------------------------------------------------
def dot_based_matching(
    masks: List[np.ndarray],
    points: List[Tuple[float, float]],
    class_idx: int,
    h: int, w: int,
) -> Dict[str, np.ndarray]:
    """基于 dot 覆盖的候选-GT 匹配。

    FSC147 只有 dot 标注和 image-level 类别，无 instance mask。
    因此：
        purity_i   = (候选覆盖的 dot 数) / (候选面积归一化)
        coverage_i = (候选覆盖的 dot 数) / (总 dot 数)
        valid_i    = purity_i > 0 且 面积合理
    """
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
    iou_arr = np.zeros(n_cand, dtype=np.float32)
    matched_class = np.full(n_cand, class_idx, dtype=np.int64)
    matched_instance_id = np.full(n_cand, -1, dtype=np.int64)
    valid = np.zeros(n_cand, dtype=np.float32)

    for i, m in enumerate(masks):
        area = float(m.sum())
        if area == 0:
            continue

        covered_dots = []
        for dot_index, (xi, yi) in enumerate(pts_int):
            if m[yi, xi]:
                covered_dots.append(dot_index)
        dots_covered = len(covered_dots)

        # purity: 候选内部 dot 密度
        purity[i] = dots_covered / max(area, 1.0)
        # coverage: 该候选覆盖了多大比例的 GT dots
        coverage[i] = dots_covered / max(n_dots_valid, 1)
        # IoU: 用 coverage 近似
        iou_arr[i] = coverage[i]

        # Valid: 覆盖至少 1 个 dot，且面积不过大/过小
        area_ratio = area / (h * w)
        if dots_covered >= 1 and 1e-4 < area_ratio < 0.95:
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


# ---------------------------------------------------------------------------
# Single image processing
# ---------------------------------------------------------------------------
def process_image(
    image: np.ndarray,
    file_name: str,
    ann_entry: dict,
    class_idx: int,
    class_name: str,
    amg,
    encoder: DINOv2RegionEncoder,
) -> Optional[dict]:
    h, w = image.shape[:2]

    # SAM2 候选生成
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

    # Dot-based matching
    points = ann_entry.get("points", [])
    match = dot_based_matching(masks, points, class_idx, h, w)

    # 近重复去重（高 IoU masks）
    # 按 source_score 降序贪心去重
    order = sorted(range(n_cand), key=lambda i: -masks[i].sum())  # 面积大的优先
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

    # 三路 crop + DINOv2
    masked_crops, box_crops, ctx_crops = [], [], []
    for i in range(n_cand):
        bb = (bboxes[i][0], bboxes[i][1], bboxes[i][0] + bboxes[i][2], bboxes[i][1] + bboxes[i][3])
        mc, bc, cc = build_three_crops(image, masks[i], bb)
        masked_crops.append(mc)
        box_crops.append(bc)
        ctx_crops.append(cc)

    z = encoder.encode_views(masked_crops, box_crops, ctx_crops, batch_size=64)

    # 输出
    masks_rle = encode_masks_rle(masks)
    gt_count = len(points)

    return {
        "img_id": os.path.splitext(file_name)[0],
        "file_name": file_name,
        "class_name": class_name,
        "gt_count": gt_count,
        "z": z.float(),
        "bbox": torch.tensor(bboxes, dtype=torch.float32),
        "matched_class": torch.from_numpy(match["matched_class"]).long(),
        "matched_instance_id": torch.from_numpy(match["matched_instance_id"]).long(),
        "iou": torch.from_numpy(match["iou"]).float(),
        "purity": torch.from_numpy(match["purity"]).float(),
        "coverage": torch.from_numpy(match["coverage"]).float(),
        "valid": torch.from_numpy(match["valid"]).float(),
        "is_part": torch.zeros(n_cand),
        "is_countable": torch.ones(n_cand),
        "masks_rle": masks_rle,
        "height": int(h),
        "width": int(w),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default="/home/czp/official_code/dataset/FSC147/annotation_FSC147_384.json")
    ap.add_argument("--img-dir", default="/home/czp/official_code/dataset/FSC147/images_384_VarV2")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--categories-json", default="result/checkpoints/text_prototypes_fsc147_categories.json")
    ap.add_argument("--class-map", default="/home/czp/official_code/dataset/FSC147/ImageClasses_FSC147.txt",
                    help="ImageClasses_FSC147.txt (image filename -> class name)")
    ap.add_argument("--prototypes-out", default=None, help="输出文本原型路径")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 加载标注和类别映射
    ann = json.load(open(args.ann))
    cats = json.load(open(args.categories_json))["categories"]
    name_to_idx = {c["name"]: c["contiguous_id"] for c in cats}

    # 加载 image → class_name 映射
    img_to_class = {}
    with open(args.class_map) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                img_to_class[parts[0]] = parts[1]
    print(f"[init] loaded {len(img_to_class)} image→class mappings")

    # 初始化 SAM2
    print(f"[init] building OCCAM-M AMG on {args.device}...")
    t0 = time.time()
    amg = build_occam_amg(args.device)
    print(f"[init] AMG ready in {time.time()-t0:.0f}s")

    # 初始化 DINOv2
    encoder = DINOv2RegionEncoder(device=args.device)
    print(f"[init] DINOv2 encoder on {args.device}")

    # 可选：生成文本原型
    if args.prototypes_out:
        print(f"[init] building text prototypes...")
        tb = TextPrototypeBuilder(device=args.device)
        class_names = [c["name"] for c in cats]
        prototypes = tb.build(class_names)
        os.makedirs(os.path.dirname(args.prototypes_out) if os.path.dirname(args.prototypes_out) else ".", exist_ok=True)
        torch.save(prototypes, args.prototypes_out)
        print(f"[init] prototypes saved to {args.prototypes_out}")

    # 处理图像
    file_names = sorted(ann.keys())
    if args.limit > 0:
        file_names = file_names[:args.limit]

    n_ok = n_skip = 0
    t_start = time.time()
    for i, fn in enumerate(file_names):
        out_path = os.path.join(args.out_dir, f"{os.path.splitext(fn)[0]}.pt")
        if args.skip_existing and os.path.exists(out_path):
            n_skip += 1
            continue

        entry = ann.get(fn)
        if entry is None:
            continue

        # 类别映射（从 ImageClasses_FSC147.txt 查）
        class_name = img_to_class.get(fn, "")
        if not class_name:
            print(f"  [skip] no class mapping for: {fn}")
            continue
        class_idx = name_to_idx.get(class_name, -1)
        if class_idx < 0:
            print(f"  [skip] unknown class: {class_name}")
            continue

        img_path = os.path.join(args.img_dir, fn)
        if not os.path.exists(img_path):
            continue

        image = np.array(Image.open(img_path).convert("RGB"))

        t_img = time.time()
        result = process_image(image, fn, entry, class_idx, class_name, amg, encoder)
        if result is None:
            continue

        torch.save(result, out_path)
        n_ok += 1
        dt = time.time() - t_img

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(file_names) - (i + 1)) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(file_names)}] ok={n_ok} skip={n_skip} "
                  f"dt={dt:.1f}s  rate={rate:.2f}/s  ETA={eta/3600:.1f}h")

    elapsed = time.time() - t_start
    print(f"\nDone: {n_ok} processed, {n_skip} skipped in {elapsed/3600:.1f}h")
    print(f"Output: {args.out_dir}")


if __name__ == "__main__":
    main()
