"""COCO 实例分割预训练缓存：为关系头提供精确的 same-instance / part-whole 标签。

与 FSC147 dot-based 弱标签不同，COCO 提供：
    - same-instance: GT instance mask IoU matching → 精确二值标签
    - same-category: GT category id → 精确二值标签
    - part-whole: mask containment → 精确软标签

输出与 FSC147 1152-dim 缓存格式兼容（z, bbox, matched_class, matched_instance_id, valid 等）。

用法:
    python script/preprocess_coco_relation.py \
        --ann /home/czp/official_code/dataset/coco/annotations/instances_val2017.json \
        --img-dir /home/czp/official_code/dataset/coco/images/val2017 \
        --out-dir /home/czp/ws_yiyang/ovcud_cache/coco_val_3view \
        --limit 2000 --device cuda
"""

from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from code.encoders.dinov2_encoder import DINOv2RegionEncoder
from code.candidates.crops import build_three_crops
from code.candidates.matching import match_candidates, stack_match_labels


def build_sam2_amg(device: str):
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    config = "configs/sam2.1/sam2.1_hiera_s.yaml"
    ckpt = "/home/czp/ws_yiyang/FreeCounting/ws_yiyang/OCCAM/checkpoints/sam2.1_hiera_small.pt"
    model = build_sam2(config, ckpt, device=device)
    return SAM2AutomaticMaskGenerator(
        model=model, points_per_side=16, points_per_batch=64,
        pred_iou_thresh=0.7, stability_score_thresh=0.8,
        box_nms_thresh=0.7, crop_n_layers=0, use_m2m=False, multimask_output=True,
    )


def _box_iou(a, b):
    ax1, ay1, aw, ah = a; bx1, by1, bw, bh = b
    ax2, ay2 = ax1+aw, ay1+ah; bx2, by2 = bx1+bw, by1+bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw * ih
    return inter / (aw*ah + bw*bh - inter) if (aw*ah + bw*bh - inter) > 0 else 0.0


def process_image(img_id, coco, img_dir, amg, encoder, cat_id_to_idx):
    """处理单张 COCO 图。"""
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(img_dir, img_info["file_name"])
    if not os.path.exists(img_path): return None

    image = np.array(Image.open(img_path).convert("RGB"))
    h, w = image.shape[:2]

    # SAM2 候选生成
    raw = amg.generate(image)
    masks, bboxes = [], []
    for r in raw:
        m = np.asarray(r["segmentation"]).astype(np.uint8)
        if m.sum() == 0 or m.sum()/(h*w) > 0.95: continue
        ys, xs = np.where(m)
        if xs.size == 0: continue
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max())+1, int(ys.max())+1
        if (x2-x1)<4 or (y2-y1)<4: continue
        masks.append(m); bboxes.append([float(x1),float(y1),float(x2-x1),float(y2-y1)])

    n_cand = len(masks)
    if n_cand == 0: return None

    # 快速去重
    order = sorted(range(n_cand), key=lambda i: -masks[i].sum())
    kept = []
    for i in order:
        dup = False
        for j in kept:
            if _box_iou(bboxes[i], bboxes[j]) > 0.9: dup = True; break
        if not dup: kept.append(i)
    kept.sort()
    masks = [masks[i] for i in kept]; bboxes = [bboxes[i] for i in kept]
    n_cand = len(masks)

    # GT 实例（用于精确匹配）
    ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
    anns = coco.loadAnns(ann_ids)
    gt_instances = []
    for a in anns:
        m = coco.annToMask(a)
        if m.shape != (h, w) or m.sum() == 0: continue
        cat_idx = cat_id_to_idx.get(a["category_id"], -1)
        if cat_idx < 0: continue
        gt_instances.append({
            "mask": m.astype(np.uint8),
            "class_idx": cat_idx,
            "instance_id": a["id"],
        })

    # 候选-GT 匹配（精确 IoU/purity/coverage + instance_id）
    results = match_candidates(masks, gt_instances)
    labels = stack_match_labels(results)

    # 检查：有足够正样本吗？
    valid = labels["valid"] > 0
    if valid.sum() < 2: return None

    # DINOv2 三路
    masked_crops, box_crops, ctx_crops = [], [], []
    for i in range(n_cand):
        bb = (bboxes[i][0], bboxes[i][1], bboxes[i][0]+bboxes[i][2], bboxes[i][1]+bboxes[i][3])
        mc, bc, cc = build_three_crops(image, masks[i], bb)
        masked_crops.append(mc); box_crops.append(bc); ctx_crops.append(cc)
    z = encoder.encode_views(masked_crops, box_crops, ctx_crops, batch_size=64)

    return {
        "img_id": str(img_id), "file_name": img_info["file_name"],
        "class_name": f"coco_{img_id}", "gt_count": len(gt_instances),
        "z": z.float(), "bbox": torch.tensor(bboxes, dtype=torch.float32),
        "matched_class": torch.from_numpy(labels["matched_class"]).long(),
        "matched_instance_id": torch.from_numpy(labels["matched_instance_id"]).long(),
        "iou": torch.from_numpy(labels["iou"]).float(),
        "purity": torch.from_numpy(labels["purity"]).float(),
        "coverage": torch.from_numpy(labels["coverage"]).float(),
        "valid": torch.from_numpy(labels["valid"]).long(),
        "is_part": torch.zeros(n_cand), "is_countable": torch.ones(n_cand),
        "masks_rle": [], "height": int(h), "width": int(w),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default="/home/czp/official_code/dataset/coco/annotations/instances_val2017.json")
    ap.add_argument("--img-dir", default="/home/czp/official_code/dataset/coco/images/val2017")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--prototypes-out", default=None, help="输出 COCO 文本原型")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    coco = COCO(args.ann)
    cat_ids = sorted(coco.getCatIds())
    cats = coco.loadCats(cat_ids)
    cat_id_to_idx = {cid: i for i, cid in enumerate(cat_ids)}
    class_names = [c["name"] for c in cats]
    print(f"[init] COCO: {len(cat_ids)} classes, {len(class_names)} categories")

    # 可选：生成文本原型
    if args.prototypes_out:
        from code.encoders.text_encoder import TextPrototypeBuilder
        tb = TextPrototypeBuilder(device=args.device)
        protos = tb.build(class_names)
        os.makedirs(os.path.dirname(args.prototypes_out) if os.path.dirname(args.prototypes_out) else ".", exist_ok=True)
        torch.save(protos, args.prototypes_out)
        print(f"[init] prototypes -> {args.prototypes_out}")

    # SAM2 + DINOv2
    print("[init] building SAM2 AMG...")
    amg = build_sam2_amg(args.device)
    encoder = DINOv2RegionEncoder(device=args.device)
    print("[init] models ready")

    img_ids = sorted(coco.getImgIds())
    img_ids = [i for i in img_ids if len(coco.getAnnIds(imgIds=i, iscrowd=False)) > 0]
    if args.limit > 0: img_ids = img_ids[:args.limit]
    print(f"[init] {len(img_ids)} images to process")

    t0 = time.time(); n_ok = n_skip = 0
    for idx, img_id in enumerate(img_ids):
        out_path = os.path.join(args.out_dir, f"{img_id:012d}.pt")
        if args.skip_existing and os.path.exists(out_path):
            n_skip += 1; continue

        result = process_image(img_id, coco, args.img_dir, amg, encoder, cat_id_to_idx)
        if result is None: continue

        torch.save(result, out_path); n_ok += 1

        if (idx+1) % 100 == 0 or idx == 0:
            elapsed = time.time() - t0
            rate = (idx+1)/elapsed if elapsed > 0 else 0
            eta = (len(img_ids)-(idx+1))/rate if rate > 0 else 0
            print(f"  [{idx+1}/{len(img_ids)}] ok={n_ok} skip={n_skip} rate={rate:.1f}/s ETA={eta/60:.0f}min")

    print(f"\nDone: {n_ok} processed in {(time.time()-t0)/60:.1f}min -> {args.out_dir}")


if __name__ == "__main__":
    main()
