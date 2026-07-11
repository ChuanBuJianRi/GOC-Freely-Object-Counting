"""CARPK 数据集预处理：SAM2 AMG + 三路 DINOv2 + GT Bbox Matching。

用于 OV-CUD 在 CARPK 上的交叉数据集验证（zero-shot transfer from FSC147）。

用法:
    # 先下载 CARPK 数据集，解压到指定目录
    # 下载地址: https://lafi.github.io/LPN/  (需签署 EULA)
    # 预期目录结构:
    #   CARPK_devkit/
    #     data/
    #       Images/       # 所有图片
    #       Annotations/  # 每张图片一个 .txt，每行: x1 y1 x2 y2

    python script/preprocess_carpk.py \
        --img-dir /path/to/CARPK_devkit/data/Images \
        --ann-dir /path/to/CARPK_devkit/data/Annotations \
        --out-dir /home/czp/ws_yiyang/ovcud_cache/carpk_test \
        --pts-per-side 32 --device cuda

预估时间 (RTX 4090, ~459 张测试图, pts=32):
    SAM2 AMG: ~1.5s/img
    DINOv2: ~0.1s/img
    总计: ~12min
"""

from __future__ import annotations

import argparse, os, sys, time
from pathlib import Path
from typing import List, Optional

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
# CARPK Annotation Loading
# ---------------------------------------------------------------------------
def load_carpk_annotations(ann_dir: str, img_name: str, h: int, w: int) -> List[tuple]:
    """加载 CARPK bbox 标注，返回 dot 坐标列表 [(x, y), ...]。

    CARPK 标注格式: 每张图片一个 .txt，每行: x1 y1 x2 y2
    x1,y1 = 左上角, x2,y2 = 右下角 (像素坐标)
    """
    base = os.path.splitext(img_name)[0]
    points = []

    # 尝试多种文件命名
    candidates = [
        os.path.join(ann_dir, f"{base}.txt"),
        os.path.join(ann_dir, f"{base}_ann.txt"),
        os.path.join(ann_dir, f"{img_name}.txt"),
    ]

    ann_path = None
    for c in candidates:
        if os.path.exists(c):
            ann_path = c
            break

    if ann_path is None:
        return []

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
            # 转换为 bbox 中心点
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            if 0 <= cx < w and 0 <= cy < h:
                points.append((cx, cy))

    return points


# ---------------------------------------------------------------------------
# Dot Matching (adapted for bbox-center dots)
# ---------------------------------------------------------------------------
def dot_matching(masks, points, class_idx, h, w) -> dict:
    """将候选 mask 匹配到 GT dot。"""
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

        # 统计 mask 覆盖的 dot 数量
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
            matched_instance_id[i] = covered_dots[0]  # 用第一个 dot 作为 instance_id

    return {
        "purity": purity,
        "coverage": coverage,
        "valid": valid,
        "matched_class": np.full(n_cand, class_idx, dtype=np.int64),
        "matched_instance_id": matched_instance_id,
    }


# ---------------------------------------------------------------------------
# Process Single Image
# ---------------------------------------------------------------------------
def process_image(image, file_name, ann_points, class_idx, class_name, amg, encoder):
    h, w = image.shape[:2]

    # SAM2 AMG
    raw = amg.generate(image)
    masks, bboxes = [], []
    for r in raw:
        m = np.asarray(r["segmentation"]).astype(np.uint8)
        area = float(m.sum())
        if area == 0 or area / (h * w) < 1e-4 or area / (h * w) > 0.95:
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

    # Bbox-IoU NMS (fast dedup)
    def _box_iou(a, b):
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
        return inter / (aw * ah + bw * bh - inter)

    order = sorted(range(n_cand), key=lambda i: masks[i].sum(), reverse=True)
    kept = []
    for i in order:
        dup = False
        for j in kept:
            if _box_iou(bboxes[i], bboxes[j]) > 0.9:
                dup = True
                break
        if not dup:
            kept.append(i)
    kept.sort()
    masks = [masks[i] for i in kept]
    bboxes = [bboxes[i] for i in kept]
    n_cand = len(masks)
    if n_cand == 0:
        return None

    # Matching to GT
    match = dot_matching(masks, ann_points, class_idx, h, w)
    for k in match:
        match[k] = match[k][:n_cand]

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
        "gt_count": len(ann_points),
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="CARPK preprocessing for OV-CUD")
    ap.add_argument("--img-dir", required=True, help="CARPK Images 目录")
    ap.add_argument("--ann-dir", required=True, help="CARPK Annotations 目录")
    ap.add_argument("--out-dir", required=True, help="输出缓存目录")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=-1, help="限制处理图像数 (-1=全部)")
    ap.add_argument("--skip-existing", action="store_true", default=True,
                    help="跳过已存在的缓存文件")
    ap.add_argument("--pts-per-side", type=int, default=32,
                    help="SAM2 pts_per_side (CARPK 密集场景建议 32)")
    ap.add_argument("--image-set", default="", help="ImageSet 文件路径 (如 test.txt)，只处理指定图片")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 收集图片文件
    img_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

    # 如果指定了 image_set，只处理其中的图片
    if args.image_set and os.path.exists(args.image_set):
        with open(args.image_set) as f:
            wanted_ids = {line.strip() for line in f if line.strip()}
        # 尝试匹配：base_name 可能是 "20161225_TPZ_00071" 而文件是 "20161225_TPZ_00071.png"
        all_imgs = {os.path.splitext(f)[0]: f for f in os.listdir(args.img_dir)
                    if os.path.splitext(f)[1] in img_exts}
        img_files = []
        missing = []
        for wid in wanted_ids:
            if wid in all_imgs:
                img_files.append(all_imgs[wid])
            else:
                missing.append(wid)
        if missing:
            print(f"[warn] {len(missing)} IDs in image_set not found in images dir")
            if len(missing) <= 10:
                print(f"  Missing: {missing}")
        img_files.sort()
        print(f"[init] {len(img_files)} images from image_set (out of {len(wanted_ids)} IDs)")
    else:
        img_files = sorted([
            f for f in os.listdir(args.img_dir)
            if os.path.splitext(f)[1] in img_exts
        ])
        print(f"[init] {len(img_files)} images in {args.img_dir}")

    if args.limit > 0:
        img_files = img_files[:args.limit]

    print(f"[init] Annotations from {args.ann_dir}")

    # 固定参数：CARPK 只有 cars 一类
    CLASS_IDX = 29  # FSC147 词表中 "cars" 的 contiguous_id
    CLASS_NAME = "cars"

    # 加载模型
    print("[init] Building SAM2 AMG...")
    t0 = time.time()
    amg = build_sam2_amg(args.device, args.pts_per_side)
    print(f"[init] SAM2 ready in {time.time() - t0:.0f}s")
    encoder = DINOv2RegionEncoder(device=args.device)
    print(f"[init] DINOv2 ready")

    n_ok = n_skip = 0
    n_no_ann = 0
    t_start = time.time()

    for i, fn in enumerate(img_files):
        out_path = os.path.join(args.out_dir, f"{os.path.splitext(fn)[0]}.pt")
        if args.skip_existing and os.path.exists(out_path):
            n_skip += 1
            continue

        img_path = os.path.join(args.img_dir, fn)

        try:
            image = np.array(Image.open(img_path).convert("RGB"))
        except Exception as e:
            print(f"  [warn] Cannot load {fn}: {e}")
            continue

        h, w = image.shape[:2]

        # 加载标注
        ann_points = load_carpk_annotations(args.ann_dir, fn, h, w)
        if not ann_points:
            n_no_ann += 1
            if n_no_ann <= 5:
                print(f"  [warn] No annotations for {fn}, skipping")
            continue

        t_img = time.time()
        result = process_image(image, fn, ann_points, CLASS_IDX, CLASS_NAME, amg, encoder)
        if result is None:
            continue

        torch.save(result, out_path)
        n_ok += 1
        dt = time.time() - t_img

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t_start
            processed = n_ok + n_skip
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = len(img_files) - (i + 1)
            eta = remaining / rate if rate > 0 else 0
            print(f"  [{i + 1}/{len(img_files)}] ok={n_ok} skip={n_skip} "
                  f"dt={dt:.1f}s rate={rate:.1f}/s ETA={eta / 60:.0f}min")

    elapsed = time.time() - t_start
    total = n_ok + n_skip
    print(f"\nDone: {n_ok} processed, {n_skip} skipped, {n_no_ann} no-annotation")
    print(f"Total time: {elapsed / 60:.1f}min → {args.out_dir}")


if __name__ == "__main__":
    main()
