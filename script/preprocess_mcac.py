"""MCAC test 预处理：SAM2 AMG + 三路 DINOv2 + per-class GT dots。

MCAC (ABC123, arXiv:2309.04820) 多类别 class-agnostic counting 合成基准。
目录结构: MCAC/{train,val,test}/<im_id>/{img.png, info_with_occ_bbox.json, ...}

评测协议对齐 ABC123 官方 test 配置 (configs/ABC123test.yml):
  - center-crop 672x672 (MCAC_crop_size=672)
  - occlusion 过滤: occ < 70 (MCAC_occ_limit=70)
  - per-class GT = 过滤后各 countable 的 center dots

坐标约定 (由 script/inspect_mcac.py 实测确定):
  centers_crop672 为 [0,1] 归一化, x = c0 * S, y = (S-1) - c1 * S  (y 翻转)
  若实测 votes 显示其他约定, 用 --axis-mode 覆盖。

用法:
    cd /home/czp/ljs/Freely-Object-Counting && python3 script/preprocess_mcac.py \
        --mcac-dir /home/czp/ljs/dataset/MCAC --split test \
        --out-dir /home/czp/ws_yiyang/ovcud_cache/mcac_test_pts32 \
        --pts-per-side 32 --device cuda
"""

from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.encoders.dinov2_encoder import DINOv2RegionEncoder
from code.candidates.crops import build_three_crops


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
# MCAC GT loading
# ---------------------------------------------------------------------------
def load_mcac_gt(json_path: str, crop_size: int, occ_limit: float, axis_mode: str
                 ) -> Tuple[List[np.ndarray], List[int]]:
    """返回 (per_class_dots, per_class_counts)。

    per_class_dots: list of [Ni,2] arrays, (x, y) 像素坐标 (crop 内)。
    只保留 occlusion < occ_limit 且过滤后 count >= 1 的类。
    """
    with open(json_path) as f:
        info = json.load(f)

    suffix = f"_crop{crop_size}" if crop_size > 0 else ""
    S = float(crop_size)

    per_class_dots, per_class_counts = [], []
    for c in info["countables"]:
        centers = np.array(c[f"centers{suffix}"], dtype=float)
        if centers.size == 0:
            continue
        centers = centers[:, :2]
        occ = np.array(c[f"occlusions{suffix}"], dtype=float).reshape(-1)
        keep = occ < occ_limit
        centers = centers[keep]
        if len(centers) == 0:
            continue
        if axis_mode == "xy_flip":
            xs = centers[:, 0] * S
            ys = (S - 1) - centers[:, 1] * S
        elif axis_mode == "xy_noflip":
            xs = centers[:, 0] * S
            ys = centers[:, 1] * S
        elif axis_mode == "yx_flip":
            ys = centers[:, 0] * S
            xs = (S - 1) - centers[:, 1] * S
        elif axis_mode == "yx_noflip":
            ys = centers[:, 0] * S
            xs = centers[:, 1] * S
        else:
            raise ValueError(axis_mode)
        xs = np.clip(xs, 0, S - 1)
        ys = np.clip(ys, 0, S - 1)
        dots = np.stack([xs, ys], axis=1)
        per_class_dots.append(dots)
        per_class_counts.append(len(dots))

    return per_class_dots, per_class_counts


# ---------------------------------------------------------------------------
# Dot matching: per-class dots -> candidate labels
# ---------------------------------------------------------------------------
def dot_matching_multiclass(masks, per_class_dots, h, w) -> Dict[str, np.ndarray]:
    n_cand = len(masks)
    purity = np.zeros(n_cand, dtype=np.float32)
    coverage = np.zeros(n_cand, dtype=np.float32)
    valid = np.zeros(n_cand, dtype=np.float32)
    matched_class = np.full(n_cand, -1, dtype=np.int64)
    matched_instance_id = np.full(n_cand, -1, dtype=np.int64)

    # flatten dots with class idx and global instance id
    all_pts, all_cls, all_iid = [], [], []
    gid = 0
    for ci, dots in enumerate(per_class_dots):
        for (x, y) in dots:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < w and 0 <= yi < h:
                all_pts.append((xi, yi))
                all_cls.append(ci)
                all_iid.append(gid)
            gid += 1
    n_dots = len(all_pts)

    for i, m in enumerate(masks):
        area = float(m.sum())
        if area == 0:
            continue
        covered = [k for k, (xi, yi) in enumerate(all_pts) if m[yi, xi]]
        dc = len(covered)
        purity[i] = dc / max(area, 1.0)
        coverage[i] = dc / max(n_dots, 1)
        ar = area / (h * w)
        if dc >= 1 and 1e-4 < ar < 0.95:
            valid[i] = 1.0
            # majority class among covered dots
            cls_votes: Dict[int, int] = {}
            for k in covered:
                cls_votes[all_cls[k]] = cls_votes.get(all_cls[k], 0) + 1
            best_cls = max(cls_votes, key=cls_votes.get)
            matched_class[i] = best_cls
            first = [k for k in covered if all_cls[k] == best_cls][0]
            matched_instance_id[i] = all_iid[first]

    return {
        "purity": purity, "coverage": coverage, "valid": valid,
        "matched_class": matched_class, "matched_instance_id": matched_instance_id,
    }


def process_image(image, im_id, per_class_dots, per_class_counts, amg, encoder):
    h, w = image.shape[:2]

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

    # box-IoU 0.9 near-duplicate removal (同 preprocess_carpk.py)
    def _box_iou(a, b):
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
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

    match = dot_matching_multiclass(masks, per_class_dots, h, w)

    masked_crops, box_crops, ctx_crops = [], [], []
    for i in range(n_cand):
        bb = (bboxes[i][0], bboxes[i][1],
              bboxes[i][0] + bboxes[i][2], bboxes[i][1] + bboxes[i][3])
        mc, bc, cc = build_three_crops(image, masks[i], bb)
        masked_crops.append(mc)
        box_crops.append(bc)
        ctx_crops.append(cc)

    z = encoder.encode_views(masked_crops, box_crops, ctx_crops, batch_size=64)

    gt_dots_flat = (np.concatenate(per_class_dots, axis=0)
                    if per_class_dots else np.zeros((0, 2)))
    gt_dot_class = (np.concatenate([np.full(len(d), ci, dtype=np.int64)
                                    for ci, d in enumerate(per_class_dots)])
                    if per_class_dots else np.zeros((0,), dtype=np.int64))

    return {
        "img_id": im_id,
        "file_name": f"{im_id}/img.png",
        "gt_count": int(sum(per_class_counts)),
        "gt_class_counts": [int(c) for c in per_class_counts],
        "n_gt_classes": len(per_class_counts),
        "gt_dots": torch.from_numpy(gt_dots_flat).float(),
        "gt_dot_class": torch.from_numpy(gt_dot_class).long(),
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
    ap = argparse.ArgumentParser(description="MCAC preprocessing for OV-CUD")
    ap.add_argument("--mcac-dir", default="/home/czp/ljs/dataset/MCAC")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pts-per-side", type=int, default=32)
    ap.add_argument("--crop-size", type=int, default=672)
    ap.add_argument("--occ-limit", type=float, default=70.0)
    ap.add_argument("--axis-mode", default="xy_flip",
                    choices=["xy_flip", "xy_noflip", "yx_flip", "yx_noflip"])
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sdir = os.path.join(args.mcac_dir, args.split)
    im_ids = sorted([f for f in os.listdir(sdir)
                     if os.path.isdir(os.path.join(sdir, f))])
    if args.limit > 0:
        im_ids = im_ids[: args.limit]
    print(f"[init] {len(im_ids)} image dirs in {sdir}")

    print("[init] Building SAM2 AMG...")
    t0 = time.time()
    amg = build_sam2_amg(args.device, args.pts_per_side)
    print(f"[init] SAM2 ready in {time.time() - t0:.0f}s")
    encoder = DINOv2RegionEncoder(device=args.device)
    print("[init] DINOv2 ready")

    n_ok = n_skip = n_fail = 0
    t_start = time.time()
    for i, im_id in enumerate(im_ids):
        out_path = os.path.join(args.out_dir, f"{im_id}.pt")
        if args.skip_existing and os.path.exists(out_path):
            n_skip += 1
            continue

        d = os.path.join(sdir, im_id)
        try:
            img = Image.open(os.path.join(d, "img.png"))
            img.load()
            if img.mode != "RGB":
                img = img.convert("RGB")
            image = np.array(img)
        except Exception as e:
            print(f"  [warn] cannot load {im_id}: {e}")
            n_fail += 1
            continue

        H, W = image.shape[:2]
        cs = args.crop_size
        if cs > 0 and (H > cs or W > cs):
            cb0 = (H - cs) // 2
            cb1 = (W - cs) // 2
            image = image[cb0:cb0 + cs, cb1:cb1 + cs]

        try:
            per_class_dots, per_class_counts = load_mcac_gt(
                os.path.join(d, "info_with_occ_bbox.json"),
                args.crop_size, args.occ_limit, args.axis_mode,
            )
        except Exception as e:
            print(f"  [warn] GT load failed {im_id}: {e}")
            n_fail += 1
            continue

        if not per_class_counts:
            # The official MCAC test split contains one image for which every
            # object is removed by the occ<70 protocol. Keep it as a valid
            # zero-target image so full-split diagnostics cover all 2,115
            # images; it contributes no (image, class) pair to per-class MAE.
            print(f"  [info] {im_id}: zero countable classes after occ filter; "
                  "keeping as a zero-target image")

        result = process_image(image, im_id, per_class_dots, per_class_counts,
                               amg, encoder)
        if result is None:
            # keep a zero-candidate stub so eval counts this image as pred=0
            result = {
                "img_id": im_id, "file_name": f"{im_id}/img.png",
                "gt_count": int(sum(per_class_counts)),
                "gt_class_counts": [int(c) for c in per_class_counts],
                "n_gt_classes": len(per_class_counts),
                "gt_dots": torch.zeros(0, 2), "gt_dot_class": torch.zeros(0, dtype=torch.long),
                "z": torch.zeros(0, 1152), "bbox": torch.zeros(0, 4),
                "matched_class": torch.zeros(0, dtype=torch.long),
                "matched_instance_id": torch.zeros(0, dtype=torch.long),
                "iou": torch.zeros(0), "purity": torch.zeros(0),
                "coverage": torch.zeros(0), "valid": torch.zeros(0),
                "is_part": torch.zeros(0), "is_countable": torch.zeros(0),
                "masks_rle": [], "height": int(image.shape[0]), "width": int(image.shape[1]),
            }
        torch.save(result, out_path)
        n_ok += 1

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t_start
            done = n_ok + n_fail
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(im_ids) - (i + 1) - n_skip) / rate if rate > 0 else 0
            print(f"  [{i + 1}/{len(im_ids)}] ok={n_ok} skip={n_skip} fail={n_fail} "
                  f"rate={rate:.2f}/s ETA={eta / 60:.0f}min", flush=True)

    print(f"\nDone: {n_ok} processed, {n_skip} skipped, {n_fail} failed "
          f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
