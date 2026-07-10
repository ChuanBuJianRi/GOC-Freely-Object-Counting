"""OCCAM (local training-free reimplementation) on MCAC test — per-class counting eval.

对应论文 `tab:mcac` OCCAM 行（docs/experiment_plan_ablation_mcac_20260709(1).md §2.2/§2.4）。

方法配置（OCCAM-M，与 script/diag_occam_dot_recall.py 的 OCCAM 论文配方一致）:
  - SAM2.1 hiera-small AMG, 8px(@384 参考边) 归一化 point grid, pred_iou=0.7,
    stability=0.8, offset=0.0, mask_threshold=0.0, box_nms=0.7, crop_n_layers=1,
    multimask=True, use_m2m=False
  - mask 过滤: OCCAM repo OccamConfig(mode="multi") 默认 p0 area window
    [0.0005, 0.5] + IoU 0.5 去重
  - 特征: ImageNet ResNet-50 2048-d, 500x500 aspect-pad crop
  - 聚类: thresholded-FINCH, thresholds (5.0, 4.0, 3.0)

数据协议（与 ABC123 官方 test 配置 /tmp/ABC123/configs/ABC123test.yml 同口径）:
  - split=test, 中心裁剪 672x672 (MCAC_crop_size=672)
  - GT: info_with_occ_bbox.json -> countables[*].centers_crop672 /
    occlusions_crop672, 仅保留 occlusion < 70 的物体 (MCAC_occ_limit=70)
  - dot 像素坐标: x = cx*672, y = (672-1) - cy*672 (对齐 ABC123 data.py 的 y 翻转)

评测协议（计划 §2.4）:
  - 预测组(cluster) 与 GT 类别做最优匹配（Hungarian 语义）:
    cost = -(cluster 的 mask 并集覆盖的该类 GT dot 数)，只允许 coverage>0 的配对；
    C<=4，用 bitmask DP 求全局最优（等价 Hungarian 最大化）。
  - 每个 (image, GT class) 一个样本: 匹配到 cluster -> |pred-gt|; 未匹配 -> gt。
    多余 cluster 忽略（对齐 ABC123 的 5-head 匹配后忽略多余 head 的做法）。
  - MAE / RMSE 在所有 (image, class) 对上聚合。
  - 附加诊断: count-optimal matching（按 |pred-gt| 最优）与 total-count 误差。

运行（GPU 重，需 flock 排队）:
  flock -x -w 86400 /tmp/foc_gpu.lock -c \
    '/home/czp/ws_yiyang/FreeCounting/venv/bin/python script/eval_mcac_occam.py'
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

OCCAM_ROOT = "/home/czp/ws_yiyang/FreeCounting/ws_yiyang/OCCAM"
if OCCAM_ROOT not in sys.path:
    sys.path.insert(0, OCCAM_ROOT)

SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
SAM2_CKPT = os.path.join(OCCAM_ROOT, "checkpoints/sam2.1_hiera_small.pt")

CROP_SIZE = 672
OCC_LIMIT = 70.0


# ---------------------------------------------------------------- data ------

def center_crop(image: np.ndarray, crop: int) -> np.ndarray:
    """复刻 ABC123 data.py 的中心裁剪口径（int 向下取整边界，[c0:-c0] 切片）。"""
    h, w = image.shape[:2]
    c0 = int((h - crop) / 2)
    c1 = int((w - crop) / 2)
    out = image
    if c0 > 0:
        out = out[c0:-c0, :]
    if c1 > 0:
        out = out[:, c1:-c1]
    return out


def load_gt(info_path: Path, crop: int, occ_limit: float):
    """返回 per-class GT: list of (count:int, dots:np.ndarray[N,2] (x,y) int)."""
    with open(info_path, "r") as f:
        info = json.load(f)
    classes = []
    for c in info["countables"]:
        centers = np.asarray(c[f"centers_crop{crop}"], dtype=np.float64)
        occ = np.asarray(c[f"occlusions_crop{crop}"], dtype=np.float64)
        if centers.ndim == 1:
            centers = centers.reshape(0, 2)
        centers = centers[:, :2] if centers.size else centers.reshape(0, 2)
        keep = occ < occ_limit
        centers = centers[keep]
        xs = np.clip((centers[:, 0] * crop).astype(int), 0, crop - 1)
        ys = np.clip(((crop - 1) - centers[:, 1] * crop).astype(int), 0, crop - 1)
        dots = np.stack([xs, ys], axis=1) if len(xs) else np.zeros((0, 2), dtype=int)
        count = int(keep.sum())
        if count == 0:
            continue
        classes.append({"count": count, "dots": dots})
    return classes


# ------------------------------------------------------------- matching -----

def coverage_matrix(cluster_masks: list[np.ndarray], gt_classes) -> np.ndarray:
    """cov[g, c] = cluster g 的 mask 并集覆盖的 class c GT dot 数。

    精确像素查找（mask[y, x]），与 script/eval_mcac.py evaluate_image 同口径。
    """
    G, C = len(cluster_masks), len(gt_classes)
    cov = np.zeros((G, C), dtype=np.int64)
    for g, um in enumerate(cluster_masks):
        for c, cls in enumerate(gt_classes):
            dots = cls["dots"]
            if len(dots) == 0:
                continue
            cov[g, c] = int(um[dots[:, 1], dots[:, 0]].sum())
    return cov


def optimal_assignment_max(score: np.ndarray) -> dict[int, int]:
    """最大化总 score 的一对一分配（class 维 <=4，bitmask DP，等价 Hungarian）。

    只允许 score>0 的配对。返回 {class_idx: cluster_idx}。
    """
    G, C = score.shape
    if G == 0 or C == 0:
        return {}
    NEG = -1
    # dp over clusters; parent[g+1][S] = (prevS, c or -1)
    parent: list[dict[int, tuple[int, int]]] = [dict() for _ in range(G + 1)]
    dp_prev = {0: 0}
    for g in range(G):
        dp_next: dict[int, int] = dict(dp_prev)
        par: dict[int, tuple[int, int]] = {S: (S, -1) for S in dp_prev}
        for S, v in dp_prev.items():
            for c in range(C):
                if S & (1 << c):
                    continue
                s = int(score[g, c])
                if s <= 0:
                    continue
                nS = S | (1 << c)
                nv = v + s
                if nv > dp_next.get(nS, NEG):
                    dp_next[nS] = nv
                    par[nS] = (S, c)
        parent[g + 1] = par
        dp_prev = dp_next
    # best final state
    bestS = max(dp_prev, key=lambda S: dp_prev[S])
    if dp_prev[bestS] <= 0:
        return {}
    # backtrack
    assign: dict[int, int] = {}
    S = bestS
    for g in range(G, 0, -1):
        par = parent[g]
        if S not in par:
            continue
        prevS, c = par[S]
        if c >= 0 and prevS != S:
            assign[c] = g - 1
            S = prevS
    return assign


def optimal_assignment_min_err(pred_counts: list[int], gt_counts: list[int]) -> float:
    """诊断口径: 允许任意 cluster->class 一对一，最小化 sum|pred-gt|（未匹配类计 gt）。"""
    C = len(gt_counts)
    G = len(pred_counts)
    full = 1 << C
    INF = float("inf")
    dp = {0: 0.0}
    for g in range(G):
        nxt = dict(dp)
        for S, v in dp.items():
            for c in range(C):
                if S & (1 << c):
                    continue
                nS = S | (1 << c)
                nv = v + abs(pred_counts[g] - gt_counts[c])
                if nv < nxt.get(nS, INF):
                    nxt[nS] = nv
        dp = nxt
    best = INF
    for S, v in dp.items():
        for c in range(C):
            if not (S & (1 << c)):
                v += gt_counts[c]
        best = min(best, v)
    return best


# ----------------------------------------------------------------- main -----

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mcac-root", default="/home/czp/ljs/dataset/MCAC/MCAC")
    p.add_argument("--split", default="test")
    p.add_argument("--out", default="/home/czp/ljs/Freely-Object-Counting/result/logs/mcac_occam.json")
    p.add_argument("--progress", default="/home/czp/ljs/Freely-Object-Counting/result/logs/mcac_occam_progress.jsonl")
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--points-per-batch", type=int, default=768)
    p.add_argument("--spacing", type=int, default=8, help="seed spacing px @384 参考边")
    p.add_argument("--crop-n-layers", type=int, default=1)
    p.add_argument("--sanity", action="store_true", help="只打印 GT 解析 sanity check，不跑模型")
    p.add_argument("--from-cache", default="",
                   help="复用 SAM 候选 cache（如 mcac_test_pts32）：跳过 AMG，"
                        "对 cache 候选做 OCCAM p0 过滤+IoU0.5 去重后走 ResNet+FINCH。"
                        "GT 取 cache 内字段（与 eval_mcac.py 完全同口径）。")
    return p.parse_args()


def build_amg(device: str, spacing: int, points_per_batch: int, crop_n_layers: int):
    """OCCAM-M AMG 配方，与 diag_occam_dot_recall.py 完全一致。"""
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    model = build_sam2(SAM2_CONFIG, SAM2_CKPT, device=device)
    ref = 384.0
    step = spacing / ref
    coords = np.arange(step / 2.0, 1.0, step, dtype=np.float32)
    gx, gy = np.meshgrid(coords, coords)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
    point_grids = [grid]
    for _ in range(1, crop_n_layers + 1):
        point_grids.append(grid)
    return SAM2AutomaticMaskGenerator(
        model=model,
        points_per_side=None,
        point_grids=point_grids,
        points_per_batch=points_per_batch,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.8,
        stability_score_offset=0.0,
        mask_threshold=0.0,
        box_nms_thresh=0.7,
        crop_n_layers=crop_n_layers,
        crop_nms_thresh=0.7,
        use_m2m=False,
        multimask_output=True,
        output_mode="binary_mask",
    )


def main():
    args = parse_args()
    from PIL import Image

    split_dir = Path(args.mcac_root) / args.split
    im_ids = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
    if args.limit > 0:
        im_ids = im_ids[: args.limit]
    print(f"[data] {split_dir} images={len(im_ids)}", flush=True)

    if args.sanity:
        for im_id in im_ids[:5]:
            img = np.array(Image.open(split_dir / im_id / "img.png").convert("RGB"))
            crop = center_crop(img, CROP_SIZE)
            gt = load_gt(split_dir / im_id / "info_with_occ_bbox.json", CROP_SIZE, OCC_LIMIT)
            with open(split_dir / im_id / "info_with_occ_bbox.json") as f:
                info = json.load(f)
            print(f"--- {im_id} orig={img.shape} crop={crop.shape}")
            for c_i, (cls, raw) in enumerate(zip(gt, info["countables"])):
                bb = np.asarray(raw.get(f"bboxes_crop{CROP_SIZE}", []), dtype=np.float64)
                occ = np.asarray(raw[f"occlusions_crop{CROP_SIZE}"], dtype=np.float64)
                inbox = -1.0
                if bb.size and len(cls["dots"]):
                    keep = occ < OCC_LIMIT
                    bb = bb[keep]
                    n_in = 0
                    # bbox 格式实测为 [[y0,y1],[x0,x1]]（密度图+bbox 100% 交叉验证）
                    for (x, y), box in zip(cls["dots"], bb):
                        ys_, xs_ = box[0], box[1]
                        if xs_.min() - 2 <= x <= xs_.max() + 2 and ys_.min() - 2 <= y <= ys_.max() + 2:
                            n_in += 1
                    inbox = n_in / max(1, len(cls["dots"]))
                print(f"    class{c_i}: total={len(raw['inds'])} occ<70={cls['count']} "
                      f"occ_range=[{occ.min():.1f},{occ.max():.1f}] dots_in_bbox={inbox:.2f}")
        return

    import torch
    from occam import OccamConfig, OccamCounter
    from occam.features import ResNetFeatureExtractor
    from occam.clustering import thresholded_finch
    from occam.masks import CandidateMask, deduplicate_masks

    config = OccamConfig.for_mode("multi", device=args.device)
    if args.from_cache:
        from pycocotools import mask as mask_utils
        cache_dir = Path(args.from_cache)
        counter = None
        _fe = ResNetFeatureExtractor(device=args.device, crop_size=config.crop_size)
        print(f"[model] OCCAM-M counting stage on cached candidates ({cache_dir.name}); "
              f"p0 area window [{config.min_mask_area_ratio},{config.max_mask_area_ratio}] "
              f"+ mask-IoU {config.duplicate_iou_threshold} dedup; "
              f"finch={config.finch_thresholds} crop_size={config.crop_size}", flush=True)
    else:
        amg = build_amg(args.device, args.spacing, args.points_per_batch, args.crop_n_layers)
        counter = OccamCounter(config, amg=amg)

    # OCCAM repo 的 ResNetFeatureExtractor.extract 会把全部 crop 一次性堆 batch，
    # MCAC 密集图（可达 >1000 候选 × 500x500）会 OOM。这里做分块包装，
    # 数值结果与原实现完全一致（不改动 OCCAM 仓库文件）。
    if counter is not None:
        _fe = counter.feature_extractor
    _CHUNK = 96

    def _chunked_extract(image, masks):
        if not masks:
            return np.empty((0, 2048), dtype=np.float32)
        outs = []
        for i in range(0, len(masks), _CHUNK):
            chunk = masks[i:i + _CHUNK]
            crops = [_fe._crop_object(image, cand) for cand in chunk]
            batch = torch.stack([_fe.transforms(cr) for cr in crops]).to(_fe.device)
            with torch.inference_mode():
                feats = _fe.model(batch).flatten(1)
            outs.append(feats.cpu().numpy().astype(np.float32))
        return np.concatenate(outs, axis=0)

    if counter is not None:
        counter.feature_extractor = type("ChunkedFE", (), {"extract": staticmethod(_chunked_extract)})()
        print(f"[model] OCCAM-M sam2.1-hiera-small spacing={args.spacing}px@384 "
              f"crop_n_layers={args.crop_n_layers} finch={config.finch_thresholds} "
              f"crop_size={config.crop_size}", flush=True)

    def count_from_cache(pt_path, crop_img):
        """cache 候选 -> OCCAM p0 过滤 + IoU0.5 去重 -> ResNet 特征 -> FINCH。

        返回 (clusters, cand_masks, gt_classes)。GT 直接取 cache 字段，
        与 script/eval_mcac.py 的评测输入完全一致。
        """
        d = torch.load(pt_path, map_location="cpu", weights_only=False)
        h, w = int(d["height"]), int(d["width"])
        gt_dots_all = np.asarray(d["gt_dots"], dtype=np.float64).reshape(-1, 2)
        gt_dot_cls = np.asarray(d["gt_dot_class"]).reshape(-1)
        gt_classes = []
        for ci, cnt in enumerate(d["gt_class_counts"]):
            dsel = gt_dots_all[gt_dot_cls == ci]
            xs = np.clip(np.round(dsel[:, 0]).astype(int), 0, w - 1)
            ys = np.clip(np.round(dsel[:, 1]).astype(int), 0, h - 1)
            gt_classes.append({"count": int(cnt),
                               "dots": np.stack([xs, ys], 1) if len(xs) else np.zeros((0, 2), int)})
        cands = []
        area_lo = config.min_mask_area_ratio * h * w
        area_hi = config.max_mask_area_ratio * h * w
        for rle, bb in zip(d["masks_rle"], d["bbox"].tolist()):
            r = dict(rle)
            if isinstance(r["counts"], str):
                r["counts"] = r["counts"].encode("ascii")
            m = mask_utils.decode(r).astype(bool)
            area = int(m.sum())
            if area < area_lo or area > area_hi:
                continue
            x0, y0, bw, bh = bb
            cands.append(CandidateMask(mask=m, bbox=(int(x0), int(y0),
                                                     int(x0 + bw), int(y0 + bh)), score=1.0))
        cands = deduplicate_masks(cands, iou_threshold=config.duplicate_iou_threshold)
        feats = _chunked_extract(crop_img, cands)
        clusters = thresholded_finch(feats, thresholds=config.finch_thresholds,
                                     steady_threshold=config.steady_threshold)
        return clusters, cands, gt_classes

    # resume
    done: dict[str, dict] = {}
    prog_path = Path(args.progress)
    if prog_path.exists():
        with open(prog_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    done[r["image"]] = r
        print(f"[resume] {len(done)} images already done", flush=True)

    prog_f = open(prog_path, "a")

    for idx, im_id in enumerate(im_ids):
        if im_id in done:
            continue
        t0 = time.time()
        img = np.array(Image.open(split_dir / im_id / "img.png").convert("RGB"))
        crop = center_crop(img, CROP_SIZE)

        if args.from_cache:
            pt_path = cache_dir / f"{im_id}.pt"
            if not pt_path.exists():
                print(f"[skip] no cache for {im_id}", flush=True)
                continue
            clusters, cand_masks, gt = count_from_cache(pt_path, crop)
            # 交叉校验：cache 内 GT 应与 json 解析一致（口径 sanity）
            if idx < 20:
                gt_json = load_gt(split_dir / im_id / "info_with_occ_bbox.json",
                                  CROP_SIZE, OCC_LIMIT)
                assert [g["count"] for g in gt] == [g["count"] for g in gt_json], \
                    f"GT count mismatch cache vs json for {im_id}"
        else:
            gt = load_gt(split_dir / im_id / "info_with_occ_bbox.json", CROP_SIZE, OCC_LIMIT)
            result = counter.count(crop)
            clusters, cand_masks = result.clusters, result.masks

        pred_counts = [len(cl.indices) for cl in clusters]

        h, w = crop.shape[:2]
        union_masks = []
        for cl in clusters:
            um = np.zeros((h, w), dtype=bool)
            for mi in cl.indices:
                um |= cand_masks[mi].mask
            union_masks.append(um)

        cov = coverage_matrix(union_masks, gt)
        assign = optimal_assignment_max(cov)

        per_class = []
        abs_err_sum = 0.0
        for c, cls in enumerate(gt):
            if c in assign:
                pc = pred_counts[assign[c]]
                cvg = int(cov[assign[c], c])
            else:
                pc, cvg = 0, 0
            per_class.append({"gt": cls["count"], "pred": pc, "dot_cov": cvg,
                              "cluster": assign.get(c, -1)})
            abs_err_sum += abs(pc - cls["count"])

        gt_counts = [cls["count"] for cls in gt]
        diag_min_err = optimal_assignment_min_err(pred_counts, gt_counts)

        rec = {
            "image": im_id,
            "gt_counts": gt_counts,
            "pred_cluster_counts": pred_counts,
            "num_clusters": len(pred_counts),
            "per_class": per_class,
            "abs_err_spatial": abs_err_sum,
            "abs_err_countopt": diag_min_err,
            "gt_total": int(sum(gt_counts)),
            "pred_total": int(sum(pred_counts)),
            "elapsed_sec": round(time.time() - t0, 2),
        }
        done[im_id] = rec
        prog_f.write(json.dumps(rec) + "\n")
        prog_f.flush()
        print(f"[{idx+1}/{len(im_ids)}] {im_id} gt={gt_counts} "
              f"pred={[p['pred'] for p in per_class]} clusters={len(pred_counts)} "
              f"{rec['elapsed_sec']:.1f}s", flush=True)

    prog_f.close()

    # aggregate
    rows = [done[i] for i in im_ids if i in done]
    errs, errs_opt, tot_errs = [], [], []
    for r in rows:
        for pc in r["per_class"]:
            errs.append(abs(pc["pred"] - pc["gt"]))
        # count-opt per-class errors: only sum available; distribute for RMSE not possible,
        # keep sum-based MAE only
        errs_opt.append((r["abs_err_countopt"], len(r["per_class"])))
        tot_errs.append(abs(r["pred_total"] - r["gt_total"]))
    errs = np.asarray(errs, dtype=np.float64)
    n_pairs = len(errs)
    mae = float(errs.mean())
    rmse = float(np.sqrt((errs ** 2).mean()))
    mae_countopt = float(sum(e for e, _ in errs_opt) / max(1, sum(n for _, n in errs_opt)))
    tot = np.asarray(tot_errs, dtype=np.float64)

    summary = {
        "method": "OCCAM-M (local training-free reimplementation)",
        "dataset": f"MCAC {args.split}",
        "protocol": {
            "crop_size": CROP_SIZE,
            "occ_limit": OCC_LIMIT,
            "matching": "Hungarian(max GT-dot coverage by cluster union mask, cov>0 only); "
                        "unmatched GT class -> pred 0; extra clusters ignored",
            "metric": "per-(image,class) MAE/RMSE, ABC123-style",
        },
        "config": ({
            "candidates": f"shared SAM cache {Path(args.from_cache).name} "
                          "(sam2.1-hiera-small pts32, pred_iou 0.7, stability 0.8, "
                          "crop_n_layers 0; same candidate pool as our-method row), "
                          "then OCCAM p0 area window [0.0005,0.5] + mask-IoU 0.5 dedup",
            "mode": "multi", "finch_thresholds": [5.0, 4.0, 3.0],
            "feature": "resnet50-imagenet-2048d, 500px aspect-pad crop",
        } if args.from_cache else {
            "sam2": "sam2.1-hiera-small", "spacing_px_at_384": args.spacing,
            "crop_n_layers": args.crop_n_layers, "pred_iou": 0.7, "stability": 0.8,
            "mode": "multi", "finch_thresholds": [5.0, 4.0, 3.0],
            "feature": "resnet50-imagenet-2048d, 500px aspect-pad crop",
        }),
        "num_images": len(rows),
        "num_image_class_pairs": n_pairs,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAE_count_optimal_matching_diag": round(mae_countopt, 4),
        "total_count_MAE": round(float(tot.mean()), 4),
        "total_count_RMSE": round(float(np.sqrt((tot ** 2).mean())), 4),
        "per_image": rows,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] images={len(rows)} pairs={n_pairs} MAE={mae:.4f} RMSE={rmse:.4f} "
          f"(count-opt diag MAE={mae_countopt:.4f}) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
