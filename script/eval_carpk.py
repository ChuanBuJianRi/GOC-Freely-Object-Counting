"""OV-CUD 在 CARPK 上的 Zero-Shot Cross-Dataset 评估。

直接使用 FSC147 训练的模型 (分类头 + 关系头)，不做任何微调。

用法:
    python script/eval_carpk.py \
        --cache-dir /home/czp/ws_yiyang/ovcud_cache/carpk_test \
        --category-ckpt result/checkpoints/category_cosine_pts32.pt \
        --relation-ckpt result/checkpoints/fsc147_relation_pts32.pt \
        --text-prototypes result/checkpoints/text_prototypes_fsc147.pt \
        --out result/logs/carpk_eval.json \
        --device cuda
"""

from __future__ import annotations

import argparse, json, os, sys, time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.clustering.first_neighbor import (
    category_aware_clustering_with_spatial,
)
from code.counting.deduplicate import (
    build_same_instance_components,
    build_same_instance_components_adaptive,
)
from code.counting.representative import select_representatives

BINS = [
    ("0-10", 0, 10), ("11-20", 11, 20), ("21-50", 21, 50),
    ("51-100", 51, 100), ("100+", 101, 10 ** 9),
]


def bin_of(c):
    for lab, lo, hi in BINS:
        if lo <= c <= hi:
            return lab
    return "100+"


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------
def load_category_head(ckpt_path, device):
    from script.train_category_v2 import CosineCategoryHead
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    head = CosineCategoryHead(
        in_dim=ck["in_dim"], proj_dim=ck["proj_dim"], dropout=0.3, num_layers=2
    )
    head.load_state_dict(ck["head"])
    head.to(device).eval()
    return head


def load_relation_head(ckpt_path, device):
    from code.heads.relation_head import PairwiseRelationHead, RelationHeadConfig
    from code.matrix.pairwise_features import pairwise_feature_dim
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    feat_dim = pairwise_feature_dim(ck["z_dim"])
    cfg = RelationHeadConfig(
        feat_dim=feat_dim, hidden_dim=ck.get("hidden_dim", 512),
        num_layers=ck.get("num_layers", 3), dropout=0.1,
    )
    head = PairwiseRelationHead(cfg)
    head.load_state_dict(ck["relation_head"])
    head.to(device).eval()
    return head


def get_category_probs(head, z, tp, device):
    with torch.no_grad():
        logits = head(z.to(device), tp.to(device))
    return torch.softmax(logits, dim=-1).cpu().numpy()


# ---------------------------------------------------------------------------
# Single Image Counting
# ---------------------------------------------------------------------------
def count_image_carpk(
    d, category_head, relation_head, text_prototypes, device,
    tau_inst=0.4, tau_affinity=0.1, conf_threshold=0.1,
    max_group_size=30,
) -> dict:
    """对单张 CARPK 图像执行 counting pipeline。"""
    h, w = int(d["height"]), int(d["width"])
    file_name = d.get("file_name", "unknown")
    gt_count = int(d.get("gt_count", 0))
    z = d["z"].float()
    bbox_np = d["bbox"].float().numpy()
    n_cand = z.shape[0]

    if n_cand == 0:
        return {
            "pred_count": 0, "n_candidates": 0, "n_groups": 0,
            "gt_count": gt_count, "file_name": file_name,
        }

    # Category probs
    category_probs = get_category_probs(category_head, z, text_prototypes, device)
    top_conf = category_probs.max(axis=1)
    top_class = category_probs.argmax(axis=1)

    # 置信度过滤
    conf_valid = top_conf >= conf_threshold
    valid_orig = np.asarray(d["valid"]) > 0
    effective_valid = valid_orig & conf_valid

    n_conf_filtered = int((valid_orig & ~conf_valid).sum())

    if effective_valid.sum() == 0:
        return {
            "pred_count": 0, "n_candidates": n_cand, "n_groups": 0,
            "gt_count": gt_count, "file_name": file_name,
            "n_conf_filtered": n_conf_filtered,
        }

    # A_sem proxy (category compatibility)
    A_sem = np.eye(n_cand, dtype=np.float32)
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            sem_score = float((category_probs[i] * category_probs[j]).sum())
            A_sem[i, j] = A_sem[j, i] = sem_score * 10.0

    # A_inst: 使用 relation head
    if relation_head is not None:
        from code.matrix.pairwise_features import (
            build_pairwise_features, box_geometry,
        )
        p_t = torch.from_numpy(category_probs).float()
        box_t = d["bbox"].float()
        z_t = z
        max_cand = min(n_cand, 200)
        if n_cand > max_cand:
            keep = np.argsort(-top_conf)[:max_cand]
            z_t = z_t[keep]; p_t = p_t[keep]; box_t = box_t[keep]
            keep_set = set(keep)
        else:
            keep_set = set(range(n_cand))

        ii, jj = np.triu_indices(
            len(keep_set) if n_cand > max_cand else n_cand, k=1,
        )
        if len(ii) > 0:
            zi, zj = z_t[ii], z_t[jj]
            pi, pj = p_t[ii], p_t[jj]
            bi, bj = box_t[ii], box_t[jj]
            geom_ij = box_geometry(bi, bj)
            geom_ji = box_geometry(bj, bi)
            phi_ij = build_pairwise_features(zi, zj, pi, pj, bi, bj, geom=geom_ij)
            phi_ji = build_pairwise_features(zj, zi, pj, pi, bj, bi, geom=geom_ji)
            with torch.no_grad():
                out = relation_head(torch.cat([phi_ij.to(device), phi_ji.to(device)], dim=0))
                P = len(ii)
                inst_logits = 0.5 * (out["inst"][:P] + out["inst"][P:]).cpu().numpy()
            A_inst = np.zeros((n_cand, n_cand), dtype=np.float32)
            for k in range(P):
                i, j = (
                    (keep[ii[k]], keep[jj[k]])
                    if n_cand > max_cand
                    else (ii[k], jj[k])
                )
                A_inst[i, j] = A_inst[j, i] = inst_logits[k]
        else:
            A_inst = np.zeros((n_cand, n_cand), dtype=np.float32)
    else:
        # Heuristic fallback: bbox IoU
        A_inst = np.zeros((n_cand, n_cand), dtype=np.float32)
        for i in range(n_cand):
            for j in range(i + 1, n_cand):
                bi, bj = bbox_np[i], bbox_np[j]
                x1 = max(bi[0], bj[0]); y1 = max(bi[1], bj[1])
                x2 = min(bi[0] + bi[2], bj[0] + bj[2])
                y2 = min(bi[1] + bi[3], bj[1] + bj[3])
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                ai = bi[2] * bi[3]; aj = bj[2] * bj[3]
                union = ai + aj - inter
                iou = inter / union if union > 0 else 0
                A_inst[i, j] = A_inst[j, i] = iou * 10.0

    # Clustering (只对有效候选)
    valid_indices = [i for i in range(n_cand) if effective_valid[i]]
    if len(valid_indices) <= 1:
        return {
            "pred_count": len(valid_indices), "n_candidates": n_cand,
            "n_groups": len(valid_indices), "gt_count": gt_count,
            "file_name": file_name, "n_conf_filtered": n_conf_filtered,
        }

    sub_probs = category_probs[valid_indices]
    sub_A_sem = A_sem[valid_indices][:, valid_indices]
    image_area = float(h * w)

    groups = category_aware_clustering_with_spatial(
        sub_probs, sub_A_sem, bbox_np[valid_indices], image_area,
        tau_affinity=tau_affinity, max_group_size=max_group_size,
        use_bucketing=True,
    )

    # Dedup + Representative Selection
    total_reps = 0
    for group in groups:
        group_global = [valid_indices[i] for i in group]
        if len(group_global) == 0:
            continue
        if len(group_global) == 1:
            total_reps += 1
            continue

        sub_A_inst = A_inst[group_global][:, group_global]
        if len(group_global) > 20:
            components = build_same_instance_components_adaptive(
                list(range(len(group_global))), sub_A_inst,
                base_tau=tau_inst, use_greedy=True,
            )
        else:
            components = build_same_instance_components(
                list(range(len(group_global))), sub_A_inst, tau_inst=tau_inst,
            )

        reps = select_representatives(
            [[group_global[i] for i in comp] for comp in components],
            np.zeros((n_cand, n_cand)),
            category_probs, bbox_np, image_area, min_category_conf=0.05,
        )
        total_reps += len(reps)

    return {
        "pred_count": total_reps, "n_candidates": n_cand,
        "n_groups": len(groups), "gt_count": gt_count,
        "file_name": file_name, "n_conf_filtered": n_conf_filtered,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="OV-CUD Zero-Shot Evaluation on CARPK")
    ap.add_argument("--cache-dir", required=True, help="预处理后的缓存目录")
    ap.add_argument("--category-ckpt", required=True, help="FSC147 分类头 checkpoint")
    ap.add_argument("--relation-ckpt", required=True, help="FSC147 关系头 checkpoint")
    ap.add_argument("--text-prototypes", required=True, help="Text prototypes .pt")
    ap.add_argument("--tau-inst", type=float, default=0.4)
    ap.add_argument("--tau-affinity", type=float, default=0.1)
    ap.add_argument("--conf-threshold", type=float, default=0.1)
    ap.add_argument("--out", default="result/logs/carpk_eval.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=-1, help="限制评估图像数")
    args = ap.parse_args()

    # 加载缓存文件
    cache_files = sorted(Path(args.cache_dir).glob("*.pt"))
    if args.limit > 0:
        cache_files = cache_files[:args.limit]
    print(f"[eval] {len(cache_files)} cache files found")

    if not cache_files:
        print("[eval] No cache files found! Run preprocess_carpk.py first.")
        return

    # 加载模型
    print("[eval] Loading models...")
    category_head = load_category_head(args.category_ckpt, args.device)
    relation_head = load_relation_head(args.relation_ckpt, args.device)
    tp = torch.nn.functional.normalize(
        torch.load(args.text_prototypes, map_location="cpu", weights_only=False).float(),
        dim=-1,
    ).to(args.device)
    print("[eval] Models loaded")

    # 评估
    results = []
    t_start = time.time()
    for i, cf in enumerate(cache_files):
        d = torch.load(cf, map_location="cpu", weights_only=False)
        r = count_image_carpk(
            d, category_head, relation_head, tp, args.device,
            tau_inst=args.tau_inst, tau_affinity=args.tau_affinity,
            conf_threshold=args.conf_threshold,
        )
        results.append(r)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(cache_files) - (i + 1)) / rate if rate > 0 else 0
            print(f"  [{i + 1}/{len(cache_files)}] rate={rate:.1f}/s ETA={eta:.0f}s")

    elapsed = time.time() - t_start
    print(f"\n[eval] Done in {elapsed:.1f}s ({elapsed / len(results):.2f}s/img)")

    if not results:
        print("No results!")
        return

    # 聚合指标
    def agg(rs, key):
        gts = np.array([r["gt_count"] for r in rs], dtype=float)
        preds = np.array([r[key] for r in rs], dtype=float)
        err = preds - gts
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        bias = float(np.mean(err))
        return {"MAE": mae, "RMSE": rmse, "bias": bias}

    m = agg(results, "pred_count")
    total_cand = sum(r["n_candidates"] for r in results)
    total_filtered = sum(r.get("n_conf_filtered", 0) for r in results)

    print(f"\n{'=' * 60}")
    print(f"OV-CUD Zero-Shot on CARPK")
    print(f"{'=' * 60}")
    print(f"Images evaluated: {len(results)}")
    print(f"Total candidates: {total_cand}")
    print(f"Confidence filtered: {total_filtered}")
    print(f"tau_inst={args.tau_inst}, tau_affinity={args.tau_affinity}, "
          f"conf_threshold={args.conf_threshold}")
    print(f"\nOverall:")
    print(f"  MAE  = {m['MAE']:.2f}")
    print(f"  RMSE = {m['RMSE']:.2f}")
    print(f"  bias = {m['bias']:+.2f}")

    # 分区间
    print(f"\n{'Interval':<10} {'#Imgs':>6} {'MAE':>9} {'RMSE':>9} {'bias':>9}")
    print(f"{'-' * 43}")
    by_bin = defaultdict(list)
    for r in results:
        by_bin[bin_of(r["gt_count"])].append(r)
    for lab, _, _ in BINS:
        rs = by_bin.get(lab)
        if not rs:
            continue
        ma = agg(rs, "pred_count")
        print(f"{lab:<10} {len(rs):>6} {ma['MAE']:>9.2f} {ma['RMSE']:>9.2f} "
              f"{ma['bias']:>+9.2f}")

    # 保存结果
    if args.out:
        os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
        json.dump({
            "config": {
                "tau_inst": args.tau_inst, "tau_affinity": args.tau_affinity,
                "conf_threshold": args.conf_threshold,
                "category_ckpt": args.category_ckpt,
                "relation_ckpt": args.relation_ckpt,
            },
            "overall": m,
            "results": results,
        }, open(args.out, "w"), indent=2)
        print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
