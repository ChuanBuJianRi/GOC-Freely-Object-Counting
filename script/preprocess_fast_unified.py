"""快速统一预处理：标准 SAM2 (pts_per_side=16) + 三路 DINOv2 + dot matching。

用一致的 SAM2 配方处理训练和测试集，消除 recipe mismatch。

速度 (RTX 4090):
    SAM2: ~0.3s/img
    DINOv2 3-view: ~0.1s/img
    总计: ~0.5s/img
    3657 训练图: ~30 min
    1190 测试图: ~10 min
"""

from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.encoders.dinov2_encoder import DINOv2RegionEncoder
from code.candidates.crops import build_three_crops


def build_sam2_amg(device: str, pts_per_side: int = 16):
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
        if isinstance(counts, bytes): counts = counts.decode("ascii")
        rles.append({"size": [int(rle["size"][0]), int(rle["size"][1])], "counts": counts})
    return rles


def dot_matching(masks, points, class_idx, h, w) -> dict:
    pts_int = [(int(round(float(x))), int(round(float(y)))) for x, y in points
               if 0 <= int(round(float(x))) < w and 0 <= int(round(float(y))) < h]
    n_dots = len(pts_int); n_cand = len(masks)
    purity = np.zeros(n_cand, dtype=np.float32); coverage = np.zeros(n_cand, dtype=np.float32)
    valid = np.zeros(n_cand, dtype=np.float32)
    matched_instance_id = np.full(n_cand, -1, dtype=np.int64)
    for i, m in enumerate(masks):
        area = float(m.sum())
        if area == 0: continue
        covered_dots = [di for di, (xi, yi) in enumerate(pts_int) if m[yi, xi]]
        dc = len(covered_dots)
        purity[i] = dc / max(area, 1.0)
        coverage[i] = dc / max(n_dots, 1)
        ar = area / (h * w)
        if dc >= 1 and 1e-4 < ar < 0.95:
            valid[i] = 1.0
            matched_instance_id[i] = covered_dots[0]
    return {"purity": purity, "coverage": coverage, "valid": valid,
            "matched_class": np.full(n_cand, class_idx, dtype=np.int64),
            "matched_instance_id": matched_instance_id}


def process_image(image, file_name, ann_entry, class_idx, class_name, amg, encoder):
    h, w = image.shape[:2]
    raw = amg.generate(image)
    masks, bboxes = [], []
    for r in raw:
        m = np.asarray(r["segmentation"]).astype(np.uint8)
        area = float(m.sum())
        if area == 0 or area/(h*w) < 1e-4 or area/(h*w) > 0.95: continue
        ys, xs = np.where(m)
        if xs.size == 0: continue
        x1, y1 = int(xs.min()), int(ys.min()); x2, y2 = int(xs.max())+1, int(ys.max())+1
        if (x2-x1) < 4 or (y2-y1) < 4: continue
        masks.append(m); bboxes.append([float(x1),float(y1),float(x2-x1),float(y2-y1)])
    n_cand = len(masks)
    if n_cand == 0: return None

    # 快速去重（bbox IoU，比 mask IoU 快 100×）
    def _box_iou(a, b):
        ax1, ay1, aw, ah = a; bx1, by1, bw, bh = b
        ax2, ay2 = ax1+aw, ay1+ah; bx2, by2 = bx1+bw, by1+bh
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
        inter = iw * ih
        if inter == 0: return 0.0
        return inter / (aw*ah + bw*bh - inter)
    order = sorted(range(n_cand), key=lambda i: masks[i].sum(), reverse=True)
    kept = []
    for i in order:
        dup = False
        for j in kept:
            if _box_iou(bboxes[i], bboxes[j]) > 0.9:
                dup = True; break
        if not dup: kept.append(i)
    kept.sort()
    masks = [masks[i] for i in kept]; bboxes = [bboxes[i] for i in kept]
    n_cand = len(masks)
    if n_cand == 0: return None

    # Matching
    points = ann_entry.get("points", [])
    match = dot_matching(masks, points, class_idx, h, w)
    for k in match: match[k] = match[k][:n_cand]

    # DINOv2
    masked_crops, box_crops, ctx_crops = [], [], []
    for i in range(n_cand):
        bb = (bboxes[i][0], bboxes[i][1], bboxes[i][0]+bboxes[i][2], bboxes[i][1]+bboxes[i][3])
        mc, bc, cc = build_three_crops(image, masks[i], bb)
        masked_crops.append(mc); box_crops.append(bc); ctx_crops.append(cc)
    z = encoder.encode_views(masked_crops, box_crops, ctx_crops, batch_size=64)

    return {
        "img_id": os.path.splitext(file_name)[0], "file_name": file_name,
        "class_name": class_name, "gt_count": len(points),
        "z": z.float(), "bbox": torch.tensor(bboxes, dtype=torch.float32),
        "matched_class": torch.from_numpy(match["matched_class"]).long(),
        "matched_instance_id": torch.from_numpy(match["matched_instance_id"]).long(),
        "iou": torch.from_numpy(match["coverage"]).float(),
        "purity": torch.from_numpy(match["purity"]).float(),
        "coverage": torch.from_numpy(match["coverage"]).float(),
        "valid": torch.from_numpy(match["valid"]).float(),
        "is_part": torch.zeros(n_cand), "is_countable": torch.ones(n_cand),
        "masks_rle": encode_masks_rle(masks),
        "height": int(h), "width": int(w),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default="/home/czp/official_code/dataset/FSC147/annotation_FSC147_384.json")
    ap.add_argument("--img-dir", default="/home/czp/official_code/dataset/FSC147/images_384_VarV2")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--class-map", default="/home/czp/official_code/dataset/FSC147/ImageClasses_FSC147.txt")
    ap.add_argument("--categories-json", default="result/checkpoints/text_prototypes_fsc147_categories.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--pts-per-side", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load mappings
    ann = json.load(open(args.ann))
    cats = json.load(open(args.categories_json))["categories"]
    name_to_idx = {c["name"]: c["contiguous_id"] for c in cats}
    img_to_class = {}
    with open(args.class_map) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2: img_to_class[parts[0]] = parts[1]
    print(f"[init] {len(img_to_class)} class mappings, {len(name_to_idx)} categories")

    file_names = sorted(ann.keys())
    if args.limit > 0: file_names = file_names[:args.limit]
    print(f"[init] {len(file_names)} images to process")

    # Models
    print("[init] building SAM2 AMG...")
    t0 = time.time()
    amg = build_sam2_amg(args.device, args.pts_per_side)
    print(f"[init] SAM2 ready in {time.time()-t0:.0f}s")
    encoder = DINOv2RegionEncoder(device=args.device)
    print(f"[init] DINOv2 ready")

    n_ok = n_skip = 0
    t_start = time.time()
    for i, fn in enumerate(file_names):
        out_path = os.path.join(args.out_dir, f"{os.path.splitext(fn)[0]}.pt")
        if args.skip_existing and os.path.exists(out_path):
            n_skip += 1; continue

        entry = ann.get(fn)
        if not entry: continue
        class_name = img_to_class.get(fn, "")
        class_idx = name_to_idx.get(class_name, -1)
        if class_idx < 0: continue

        img_path = os.path.join(args.img_dir, fn)
        if not os.path.exists(img_path): continue

        try:
            image = np.array(Image.open(img_path).convert("RGB"))
        except Exception as e:
            print(f"  [warn] cannot load {fn}: {e}")
            continue
        t_img = time.time()
        result = process_image(image, fn, entry, class_idx, class_name, amg, encoder)
        if result is None: continue

        torch.save(result, out_path)
        n_ok += 1
        dt = time.time() - t_img

        if (i+1) % 200 == 0 or i == 0:
            elapsed = time.time() - t_start
            rate = (n_ok+n_skip)/elapsed if elapsed > 0 else 0
            eta = (len(file_names)-(i+1))/rate if rate > 0 else 0
            print(f"  [{i+1}/{len(file_names)}] ok={n_ok} dt={dt:.1f}s rate={rate:.1f}/s ETA={eta/60:.0f}min")

    elapsed = time.time() - t_start
    print(f"\nDone: {n_ok} processed, {n_skip} skipped in {elapsed/60:.1f}min -> {args.out_dir}")


if __name__ == "__main__":
    main()
