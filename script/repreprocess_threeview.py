"""三路特征重预处理：复用已有 SAM2 masks，重算 DINOv2 三路 (masked+box+context) 特征。

输入：已有缓存（含 masks_rle + bbox + 元数据）
输出：新缓存，z 从 384 维升级为 1152 维

预计时间：
    - DINOv2 encoding: ~3 min（RTX 4090 batch=64，561K crops）
    - Image loading + crop building: ~2h（3657 张图，每图 ~200 候选）
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from code.candidates.crops import build_three_crops
from code.encoders.dinov2_encoder import DINOv2RegionEncoder


def process_image(
    cache_path: str,
    image_dir: str,
    encoder: DINOv2RegionEncoder,
    out_dir: str,
) -> bool:
    """处理单张图：加载旧缓存 + 原图 → 重算三路特征 → 写出新缓存。"""
    d = torch.load(cache_path, map_location="cpu", weights_only=False)

    file_name = d.get("file_name") or f"{d.get('img_id')}.jpg"
    img_path = os.path.join(image_dir, file_name)
    if not os.path.exists(img_path):
        print(f"  [skip] image not found: {file_name}")
        return False

    image = np.array(Image.open(img_path).convert("RGB"))
    h, w = image.shape[:2]

    # 解码 masks
    masks_rle = d["masks_rle"]
    masks = []
    for r in masks_rle:
        m = mask_utils.decode(r).astype(bool)
        if m.shape == (h, w):
            masks.append(m)
    n_cand = len(masks)

    if n_cand == 0:
        return False

    # 获取 bbox
    bbox = d["bbox"]  # [N, 4] XYWH
    if bbox.shape[0] != n_cand:
        print(f"  [warn] bbox count {bbox.shape[0]} != mask count {n_cand}")
        return False

    # 构建三路 crop
    masked_crops, box_crops, ctx_crops = [], [], []
    for i in range(n_cand):
        # Convert XYWH bbox to [x1, y1, x2, y2]
        x, y, bw, bh = bbox[i].tolist()
        bb = (x, y, x + bw, y + bh)
        mc, bc, cc = build_three_crops(image, masks[i].astype(np.uint8), bb)
        masked_crops.append(mc)
        box_crops.append(bc)
        ctx_crops.append(cc)

    # DINOv2 三路编码
    z_new = encoder.encode_views(masked_crops, box_crops, ctx_crops, batch_size=64)

    # 写出新缓存（保留所有原有字段，只替换 z）
    out = dict(d)  # 浅拷贝
    out["z"] = z_new.float()
    out["_z_dim_original"] = d["z"].shape[1]  # 记录原始维度

    out_path = os.path.join(out_dir, os.path.basename(cache_path))
    torch.save(out, out_path)
    return True


def main():
    ap = argparse.ArgumentParser(description="三路特征重预处理（384→1152）")
    ap.add_argument("--cache-dir", required=True, help="原始缓存目录（384-dim）")
    ap.add_argument("--image-dir", default="/home/czp/official_code/dataset/FSC147/images_384_VarV2")
    ap.add_argument("--out-dir", required=True, help="输出目录（1152-dim）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=-1, help="最多处理 N 张（调试）")
    ap.add_argument("--skip-existing", action="store_true", default=True, help="跳过已存在的输出")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(args.cache_dir) if f.endswith(".pt"))
    if args.limit > 0:
        files = files[:args.limit]

    encoder = DINOv2RegionEncoder(device=args.device)
    print(f"[init] DINOv2 encoder on {args.device}, {len(files)} images to process")

    t0 = time.time()
    n_ok = n_skip = 0
    for i, fn in enumerate(files):
        if args.skip_existing and os.path.exists(os.path.join(args.out_dir, fn)):
            n_skip += 1
            continue

        cache_path = os.path.join(args.cache_dir, fn)
        ok = process_image(cache_path, args.image_dir, encoder, args.out_dir)
        if ok:
            n_ok += 1

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(files) - (i + 1)) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(files)}] ok={n_ok} skip={n_skip} "
                  f"rate={rate:.1f} img/s  ETA={eta/60:.0f}min")

    elapsed = time.time() - t0
    print(f"\nDone: {n_ok} processed, {n_skip} skipped in {elapsed/60:.1f} min")
    print(f"Output: {args.out_dir}")


if __name__ == "__main__":
    main()
