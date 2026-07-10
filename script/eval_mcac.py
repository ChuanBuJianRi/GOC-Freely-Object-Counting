"""OV-CUD 在 MCAC test 上的多类别评测 (实验计划 §2.4/§2.5)。

协议 (与 ABC123 MCAC test 表同口径, per-class counts 的 MAE/RMSE):
  1. pipeline 输出 class-agnostic 实例 (representative 候选), 每个实例带
     top-1 predicted class (FSC147 文本原型只作 anonymous 分组特征, 无语义对齐)。
  2. 预测实例按 top-1 class 聚成 anonymous predicted groups
     (等价于 ABC123 的匿名 per-class 输出)。
  3. 与 GT per-class dots 做 Hungarian 最优匹配, 匹配代价 = -(该 group 的
     representative masks 覆盖到的该类 GT dots 数)。
  4. 每个 GT 类: 匹配到 group 则 pred=group count, 否则 pred=0;
     覆盖分数为 0 的匹配视为未匹配。多余 predicted groups 不计入
     (与 ABC123 5-head 中未匹配 head 不计一致)。
  5. MAE/RMSE 在所有 (image, GT class) 对上聚合。

变体 (--variant):
  m6: 完整模型 (relation head dedup + adaptive dedup + conf filter 0.2)
  m1: 去重换 class-bucket box-IoU NMS@0.5 (无 relation head)
  m2: 关闭 ADF (conf_threshold=0, 不过滤; MCAC 协议本身无 density routing)
  m5: 关闭 4x4 rescue tiling (不传 --overlay-cache-dir)
  m4: pts16 头 (通过 --cache-dir/--category-ckpt/--relation-ckpt 切换, 逻辑同 m6)

用法:
    cd /home/czp/ljs/Freely-Object-Counting && python3 script/eval_mcac.py \
        --cache-dir /home/czp/ws_yiyang/ovcud_cache/mcac_test_pts32 \
        --category-ckpt result/checkpoints/category_cosine_pts32.pt \
        --relation-ckpt result/checkpoints/fsc147_relation_pts32_best.pt \
        --text-prototypes result/checkpoints/text_prototypes_fsc147.pt \
        --variant m6 --tau-inst 0.99 --tau-affinity 0.1 --conf-threshold 0.2 \
        --out result/logs/mcac_m6.json --device cuda
"""

from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.clustering.first_neighbor import category_aware_clustering_with_spatial
from code.counting.deduplicate import (
    build_same_instance_components,
    build_same_instance_components_adaptive,
)
from code.counting.representative import select_representatives
from code.matrix.pairwise_features import build_pairwise_features, box_geometry
from script.eval_carpk import load_category_head, load_relation_head, get_category_probs


# ---------------------------------------------------------------------------
# A_inst via relation head (同 eval_carpk)
# ---------------------------------------------------------------------------
def relation_matrix(d, z, probs, top_conf, relation_head, device):
    n_cand = z.shape[0]
    p_t = torch.from_numpy(probs).float()
    box_t = d["bbox"].float()
    z_t = z
    max_cand = min(n_cand, 200)
    if n_cand > max_cand:
        keep = np.argsort(-top_conf)[:max_cand]
        z_t = z_t[keep]
        p_t = p_t[keep]
        box_t = box_t[keep]
    else:
        keep = None

    m = max_cand if n_cand > max_cand else n_cand
    ii, jj = np.triu_indices(m, k=1)
    A_inst = np.zeros((n_cand, n_cand), dtype=np.float32)
    if len(ii) == 0:
        return A_inst

    bi, bj = box_t[ii], box_t[jj]
    phi_ij = build_pairwise_features(
        z_t[ii], z_t[jj], p_t[ii], p_t[jj], bi, bj, geom=box_geometry(bi, bj))
    phi_ji = build_pairwise_features(
        z_t[jj], z_t[ii], p_t[jj], p_t[ii], bj, bi, geom=box_geometry(bj, bi))
    with torch.no_grad():
        out = relation_head(torch.cat([phi_ij.to(device), phi_ji.to(device)], dim=0))
        p = len(ii)
        logits = 0.5 * (out["inst"][:p] + out["inst"][p:]).cpu().numpy()
        logits = np.clip(logits, -20, 20)
    for k in range(len(ii)):
        i, j = (keep[ii[k]], keep[jj[k]]) if keep is not None else (ii[k], jj[k])
        A_inst[i, j] = A_inst[j, i] = logits[k]
    return A_inst


def iou_nms_instances(idxs, bbox_np, top_conf, top_class, iou_thr=0.5):
    """M1: class-bucket box-IoU NMS@0.5, 返回保留的实例索引列表。"""
    kept_all = []
    for cls in np.unique(top_class[idxs]):
        cand = [i for i in idxs if top_class[i] == cls]
        cand.sort(key=lambda i: -top_conf[i])
        kept = []
        for i in cand:
            xi, yi, wi, hi = bbox_np[i]
            ok = True
            for j in kept:
                xj, yj, wj, hj = bbox_np[j]
                x1, y1 = max(xi, xj), max(yi, yj)
                x2, y2 = min(xi + wi, xj + wj), min(yi + hi, yj + hj)
                inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                union = wi * hi + wj * hj - inter
                if union > 0 and inter / union >= iou_thr:
                    ok = False
                    break
            if ok:
                kept.append(i)
        kept_all.extend(kept)
    return kept_all


def predict_instances(d, z, probs, top_conf, top_class, bbox_np, relation_head,
                      device, variant, tau_inst, tau_affinity, conf_threshold,
                      max_group_size=30, candidate_filter="gt_dot_valid"):
    """执行 counting pipeline, 返回预测实例 (representative) 全局索引列表。"""
    h, w = int(d["height"]), int(d["width"])
    n_cand = z.shape[0]
    if candidate_filter == "gt_dot_valid":
        valid_orig = np.asarray(d["valid"]) > 0
    elif candidate_filter == "all":
        valid_orig = np.ones(n_cand, dtype=bool)
    else:
        raise ValueError(f"unknown candidate_filter: {candidate_filter}")

    if variant == "m2":
        effective_valid = valid_orig  # ADF off: 不做置信度过滤
    else:
        effective_valid = valid_orig & (top_conf >= conf_threshold)

    vidx = [i for i in range(n_cand) if effective_valid[i]]
    if len(vidx) == 0:
        return []
    if len(vidx) == 1:
        return vidx

    if variant == "m1":
        return iou_nms_instances(vidx, bbox_np, top_conf, top_class, iou_thr=0.5)

    # relation head dedup 路线 (m6/m2/m5/m4)
    A_sem = (probs @ probs.T).astype(np.float32) * 10.0
    np.fill_diagonal(A_sem, 1.0)
    A_inst = relation_matrix(d, z, probs, top_conf, relation_head, device)

    image_area = float(h * w)
    groups = category_aware_clustering_with_spatial(
        probs[vidx], A_sem[np.ix_(vidx, vidx)], bbox_np[vidx], image_area,
        tau_affinity=tau_affinity, max_group_size=max_group_size,
        use_bucketing=True,
    )

    reps_all = []
    for group in groups:
        group_global = [vidx[i] for i in group]
        if len(group_global) == 0:
            continue
        if len(group_global) == 1:
            reps_all.append(group_global[0])
            continue
        sub_A_inst = A_inst[np.ix_(group_global, group_global)]
        if len(group_global) > 20:
            components = build_same_instance_components_adaptive(
                list(range(len(group_global))), sub_A_inst,
                base_tau=tau_inst, use_greedy=True)
        else:
            components = build_same_instance_components(
                list(range(len(group_global))), sub_A_inst, tau_inst=tau_inst)
        reps = select_representatives(
            [[group_global[i] for i in comp] for comp in components],
            np.zeros((n_cand, n_cand)), probs, bbox_np, image_area,
            min_category_conf=0.05)
        reps_all.extend(reps)
    return reps_all


# ---------------------------------------------------------------------------
# Per-class Hungarian matching evaluation
# ---------------------------------------------------------------------------
def evaluate_image(d, rep_idxs, top_class):
    """返回 per-class 记录列表 + 诊断信息。"""
    gt_class_counts = d["gt_class_counts"]
    n_gt_cls = len(gt_class_counts)
    gt_dots = d["gt_dots"].numpy() if torch.is_tensor(d["gt_dots"]) else np.asarray(d["gt_dots"])
    gt_dot_cls = (d["gt_dot_class"].numpy() if torch.is_tensor(d["gt_dot_class"])
                  else np.asarray(d["gt_dot_class"]))
    h, w = int(d["height"]), int(d["width"])

    # predicted anonymous groups: bucket reps by top-1 predicted class
    buckets: Dict[int, List[int]] = {}
    for r in rep_idxs:
        buckets.setdefault(int(top_class[r]), []).append(r)
    bucket_keys = sorted(buckets.keys())
    n_buckets = len(bucket_keys)

    # decode rep masks per bucket, build coverage score matrix [n_buckets, n_gt_cls]
    score = np.zeros((max(n_buckets, 1), n_gt_cls), dtype=float)
    if n_buckets > 0 and len(gt_dots) > 0:
        rles = d["masks_rle"]
        xi = np.clip(np.round(gt_dots[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(gt_dots[:, 1]).astype(int), 0, h - 1)
        for bi, bk in enumerate(bucket_keys):
            covered = np.zeros(len(gt_dots), dtype=bool)
            for r in buckets[bk]:
                rle = dict(rles[r])
                if isinstance(rle["counts"], str):
                    rle["counts"] = rle["counts"].encode("ascii")
                m = mask_utils.decode(rle)
                covered |= m[yi, xi] > 0
            for ci in range(n_gt_cls):
                score[bi, ci] = float(np.sum(covered & (gt_dot_cls == ci)))

    # Hungarian max matching (cost = -score)
    pred_per_class = [0] * n_gt_cls
    matched_bucket = [-1] * n_gt_cls
    if n_buckets > 0:
        rows, cols = linear_sum_assignment(-score[:n_buckets])
        for r, c in zip(rows, cols):
            if score[r, c] > 0:  # 零覆盖视为未匹配
                pred_per_class[c] = len(buckets[bucket_keys[r]])
                matched_bucket[c] = bucket_keys[r]

    per_class_records = []
    for ci in range(n_gt_cls):
        per_class_records.append({
            "gt": int(gt_class_counts[ci]),
            "pred": int(pred_per_class[ci]),
            "matched_bucket": int(matched_bucket[ci]),
        })

    diag = {
        "n_pred_instances": len(rep_idxs),
        "n_pred_buckets": n_buckets,
        "bucket_counts": {int(k): len(v) for k, v in buckets.items()},
        "total_gt": int(sum(gt_class_counts)),
        "total_pred": len(rep_idxs),
    }
    return per_class_records, diag


def metrics(preds, gts):
    err = np.asarray(preds, dtype=float) - np.asarray(gts, dtype=float)
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)),
        "n": int(len(err)),
    }


def main():
    ap = argparse.ArgumentParser(description="OV-CUD MCAC multi-class evaluation")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--category-ckpt", required=True)
    ap.add_argument("--relation-ckpt", required=True)
    ap.add_argument("--text-prototypes", required=True)
    ap.add_argument("--variant", default="m6",
                    choices=["m6", "m1", "m2", "m3", "m4", "m5"])
    ap.add_argument(
        "--overlay-cache-dir",
        default="",
        help="Optional cache overlay. A same-named .pt here replaces the base "
             "cache entry, e.g. the no-GT-triggered 4x4 rescue candidates.",
    )
    ap.add_argument(
        "--expected-images",
        type=int,
        default=2115,
        help="Fail if a full run does not contain this many cache entries.",
    )
    ap.add_argument("--tau-inst", type=float, default=0.99)
    ap.add_argument("--tau-affinity", type=float, default=0.1)
    ap.add_argument("--conf-threshold", type=float, default=0.2)
    ap.add_argument(
        "--candidate-filter",
        choices=["gt_dot_valid", "all"],
        default="gt_dot_valid",
        help="gt_dot_valid reproduces the existing cache/oracle-filtered protocol; "
             "all is the strict no-GT inference sanity check.",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()

    base_cache_files = sorted(Path(args.cache_dir).glob("*.pt"))
    overlay_dir = Path(args.overlay_cache_dir) if args.overlay_cache_dir else None
    cache_files = []
    n_overlay = 0
    for base_path in base_cache_files:
        overlay_path = overlay_dir / base_path.name if overlay_dir else None
        if overlay_path is not None and overlay_path.exists():
            cache_files.append(overlay_path)
            n_overlay += 1
        else:
            cache_files.append(base_path)
    if args.limit > 0:
        cache_files = cache_files[: args.limit]
    elif args.expected_images > 0 and len(cache_files) != args.expected_images:
        raise RuntimeError(
            f"expected {args.expected_images} cache files, found {len(cache_files)} "
            f"in {args.cache_dir}"
        )
    print(f"[eval] variant={args.variant}, {len(cache_files)} cache files, "
          f"overlay={n_overlay}")

    category_head = load_category_head(args.category_ckpt, args.device)
    relation_head = (None if args.variant == "m1"
                     else load_relation_head(args.relation_ckpt, args.device))
    tp = torch.nn.functional.normalize(
        torch.load(args.text_prototypes, map_location="cpu", weights_only=False).float(),
        dim=-1).to(args.device)
    print("[eval] models loaded")

    all_class_records = []
    per_image = []
    t0 = time.time()
    for i, cf in enumerate(cache_files):
        d = torch.load(cf, map_location="cpu", weights_only=False)
        z = d["z"].float()
        n_cand = z.shape[0]

        if n_cand == 0:
            recs = [{"gt": int(c), "pred": 0, "matched_bucket": -1}
                    for c in d["gt_class_counts"]]
            diag = {"n_pred_instances": 0, "n_pred_buckets": 0,
                    "bucket_counts": {}, "total_gt": int(d["gt_count"]),
                    "total_pred": 0}
        else:
            probs = get_category_probs(category_head, z, tp, args.device)
            top_conf = probs.max(axis=1)
            top_class = probs.argmax(axis=1)
            bbox_np = d["bbox"].float().numpy()
            reps = predict_instances(
                d, z, probs, top_conf, top_class, bbox_np, relation_head,
                args.device, args.variant, args.tau_inst, args.tau_affinity,
                args.conf_threshold, candidate_filter=args.candidate_filter)
            recs, diag = evaluate_image(d, reps, top_class)

        for r in recs:
            r["img_id"] = d["img_id"]
        all_class_records.extend(recs)
        per_image.append({"img_id": d["img_id"],
                          "cache_source": ("overlay" if overlay_dir and cf.parent == overlay_dir
                                           else "base"),
                          **diag,
                          "per_class": [{k: r[k] for k in ("gt", "pred", "matched_bucket")}
                                        for r in recs]})

        if (i + 1) % 100 == 0:
            el = time.time() - t0
            rate = (i + 1) / el
            print(f"  [{i + 1}/{len(cache_files)}] rate={rate:.2f}/s "
                  f"ETA={(len(cache_files) - i - 1) / rate / 60:.0f}min", flush=True)

    m = metrics([r["pred"] for r in all_class_records],
                [r["gt"] for r in all_class_records])
    tot = metrics([p["total_pred"] for p in per_image],
                  [p["total_gt"] for p in per_image])

    print(f"\n{'=' * 60}")
    print(f"MCAC test, variant={args.variant}")
    print(f"per-class ({m['n']} image-class pairs over {len(per_image)} images):")
    print(f"  MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  bias={m['bias']:+.2f}")
    print(f"total-count (diagnostic): MAE={tot['MAE']:.2f} RMSE={tot['RMSE']:.2f} "
          f"bias={tot['bias']:+.2f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({
        "config": {
            "variant": args.variant,
            "cache_dir": args.cache_dir,
            "overlay_cache_dir": args.overlay_cache_dir or None,
            "n_overlay_images": n_overlay,
            "category_ckpt": args.category_ckpt,
            "relation_ckpt": args.relation_ckpt,
            "tau_inst": args.tau_inst,
            "tau_affinity": args.tau_affinity,
            "conf_threshold": (0.0 if args.variant == "m2" else args.conf_threshold),
            "candidate_filter": args.candidate_filter,
            "uses_gt_candidate_filter": args.candidate_filter == "gt_dot_valid",
            "protocol": "ABC123-style: center-crop672, occ<70, per-class Hungarian "
                        "matching on predicted top1-class buckets vs GT per-class dots, "
                        "cost=-dot-coverage, zero-coverage=unmatched, unmatched GT class pred=0",
            "note": ("M5 disables the no-GT-triggered 4x4 rescue overlay while "
                     "keeping the M6 heads and thresholds"
                     if args.variant == "m5" else ""),
        },
        "num_images": len(per_image),
        "num_image_class_pairs": len(all_class_records),
        "per_class_metrics": m,
        "total_count_metrics": tot,
        "results": per_image,
    }, open(args.out, "w"), indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
