"""Evaluate OV-CUD on OmniCount-191 test set.

Two modes:
1. Class-agnostic (prompt-free): Count all objects per image, no class labels needed
2. Oracle (upper bound): Use GT matched_class for perfect classification

Metrics:
- Overall MAE, RMSE, bias (per-image total count)
- Per-category MAE (Birds, Fruits, Pets, Satellite, Supermarket, Urban, Wild)
- Per-bin MAE

For comparison, also reports Vanilla SAM2 baseline (count = number of masks).

Usage:
    # Class-agnostic (prompt-free)
    python script/eval_omnicount.py \
        --cache-dir /home/czp/ws_yiyang/ovcud_cache/omnicount_test \
        --out result/logs/omnicount_class_agnostic.json

    # Oracle (upper bound)
    python script/eval_omnicount.py \
        --cache-dir /home/czp/ws_yiyang/ovcud_cache/omnicount_test \
        --oracle \
        --out result/logs/omnicount_oracle.json
"""

from __future__ import annotations

import argparse, json, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.clustering.first_neighbor import category_aware_clustering_with_spatial
from code.counting.deduplicate import (
    build_same_instance_components, build_same_instance_components_adaptive,
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
# Image-level counting with heuristic dedup (no category/relation heads)
# ---------------------------------------------------------------------------
def count_image_class_agnostic(d, tau_inst=0.5):
    """Count all objects in an image using heuristic bbox IoU dedup."""
    h, w = int(d["height"]), int(d["width"])
    n_cand = d["z"].shape[0]
    gt_count = int(d.get("gt_count", 0))

    if n_cand == 0:
        return {"pred_count": 0, "n_candidates": 0, "gt_count": gt_count}

    bbox_np = d["bbox"].float().numpy()

    # Build A_sem and A_inst from bbox IoU
    n = n_cand
    A_sem = np.eye(n, dtype=np.float32)
    A_inst = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            bi, bj = bbox_np[i], bbox_np[j]
            x1 = max(bi[0], bj[0]); y1 = max(bi[1], bj[1])
            x2 = min(bi[0] + bi[2], bj[0] + bj[2]); y2 = min(bi[1] + bi[3], bj[1] + bj[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area_i = bi[2] * bi[3]; area_j = bj[2] * bj[3]
            union = area_i + area_j - inter
            iou = inter / union if union > 0 else 0.0
            # Scale IoU to logit range
            A_inst[i, j] = iou * 10.0
            A_inst[j, i] = iou * 10.0
            A_sem[i, j] = iou * 10.0
            A_sem[j, i] = A_sem[i, j]
    A_part = np.zeros((n, n), dtype=np.float32)

    # Category probabilities: all "unknown" (single group for class-agnostic)
    category_probs = np.zeros((n, 2), dtype=np.float32)
    category_probs[:, 0] = 1.0

    # Clustering with spatial sub-clustering
    image_area = float(h * w)
    groups = category_aware_clustering_with_spatial(
        category_probs, A_sem, bbox_np, image_area,
        tau_affinity=0.3, max_group_size=30, use_bucketing=True,
    )

    # Dedup + counting
    total_reps = 0
    for group in groups:
        if len(group) == 0:
            continue
        if len(group) == 1:
            total_reps += 1
            continue

        if len(group) > 20:
            components = build_same_instance_components_adaptive(
                group, A_inst, base_tau=tau_inst, use_greedy=True, max_comp_size=5)
        else:
            components = build_same_instance_components(group, A_inst, tau_inst=tau_inst)
        reps = select_representatives(
            components, A_part, category_probs, bbox_np, image_area,
            min_category_conf=0.05,
        )
        total_reps += len(reps)

    return {
        "pred_count": total_reps,
        "n_candidates": n_cand,
        "n_groups": len(groups),
        "gt_count": gt_count,
    }


# ---------------------------------------------------------------------------
# Vanilla SAM2 baseline: just count all masks (filter by size)
# ---------------------------------------------------------------------------
def count_image_sam2_only(d):
    """Count = number of SAM2 candidates (a simple baseline)."""
    n_cand = d["z"].shape[0]
    gt_count = int(d.get("gt_count", 0))
    return {"pred_count": n_cand, "n_candidates": n_cand, "gt_count": gt_count}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(results):
    preds = np.array([r["pred_count"] for r in results], float)
    gts = np.array([r["gt_count"] for r in results], float)
    err = preds - gts
    abs_err = np.abs(err)

    m = {
        "n_images": len(results),
        "MAE": float(np.mean(abs_err)),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)),
        "mean_GT": float(np.mean(gts)),
        "mean_Pred": float(np.mean(preds)),
        "nMAE": float(np.mean(abs_err) / max(np.mean(gts), 1)),
    }

    # Per-bin
    by_bin = defaultdict(list)
    for r in results:
        by_bin[bin_of(r["gt_count"])].append(r)
    per_bin = {}
    for lab, _, _ in BINS:
        rs = by_bin.get(lab, [])
        if not rs:
            continue
        bp = np.array([r["pred_count"] for r in rs], float)
        bg = np.array([r["gt_count"] for r in rs], float)
        be = bp - bg
        per_bin[lab] = {
            "n": len(rs),
            "MAE": float(np.mean(np.abs(be))),
            "RMSE": float(np.sqrt(np.mean(be ** 2))),
            "bias": float(np.mean(be)),
        }
    m["per_bin"] = per_bin
    return m


def compute_per_category(results_by_category):
    """Compute metrics per OmniCount category."""
    per_cat = {}
    for cat, results in sorted(results_by_category.items()):
        if not results:
            continue
        m = compute_metrics(results)
        per_cat[cat] = {
            "n": m["n_images"],
            "MAE": m["MAE"], "RMSE": m["RMSE"], "bias": m["bias"],
            "mean_GT": m["mean_GT"],
        }
    return per_cat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--tau-inst", type=float, default=0.5)
    ap.add_argument("--oracle", action="store_true",
                    help="Use oracle classification (GT matched_class) for upper bound")
    ap.add_argument("--sam2-only", action="store_true",
                    help="Vanilla SAM2 baseline (count = n_candidates)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()

    files = sorted(Path(args.cache_dir).glob("*.pt"))
    if args.limit > 0:
        files = files[:args.limit]

    print(f"Evaluating {len(files)} images from {args.cache_dir}")
    mode = "SAM2-only" if args.sam2_only else ("Oracle" if args.oracle else "Class-agnostic (prompt-free)")
    print(f"Mode: {mode}")

    results = []
    results_by_category = defaultdict(list)

    t_start = time.time()
    for i, fp in enumerate(files):
        d = torch.load(fp, map_location="cpu", weights_only=False)

        file_name = d.get("file_name", fp.name)
        category = d.get("category", "unknown")

        if args.sam2_only:
            r = count_image_sam2_only(d)
        elif args.oracle:
            r = count_image_oracle(d)
        else:
            r = count_image_class_agnostic(d, tau_inst=args.tau_inst)

        r["file_name"] = file_name
        r["category"] = category
        results.append(r)
        results_by_category[category].append(r)

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / max(elapsed, 0.01)
            m = compute_metrics(results)
            print(f"  [{i+1}/{len(files)}] MAE={m['MAE']:.2f} RMSE={m['RMSE']:.2f} "
                  f"rate={rate:.1f}/s ETA={((len(files)-i-1)/max(rate,0.01)/60):.1f}min")

    elapsed = time.time() - t_start

    # Overall metrics
    metrics = compute_metrics(results)
    metrics["runtime_s"] = float(elapsed)
    metrics["runtime_per_image_ms"] = float(elapsed / max(len(results), 1) * 1000)

    print(f"\n{'='*70}")
    print(f"OmniCount-191 Results: {mode}")
    print(f"{'='*70}")
    print(f"Images: {metrics['n_images']}")
    print(f"MAE: {metrics['MAE']:.2f}  RMSE: {metrics['RMSE']:.2f}  bias: {metrics['bias']:+.2f}")
    print(f"Mean GT: {metrics['mean_GT']:.1f}  Mean Pred: {metrics['mean_Pred']:.1f}")
    print(f"nMAE: {metrics['nMAE']:.3f}")
    print(f"Runtime: {elapsed:.1f}s ({elapsed/len(results)*1000:.1f}ms/img)")

    print(f"\nPer-bin:")
    print(f"  {'Bin':<10} {'#':>4} {'MAE':>8} {'RMSE':>8} {'bias':>8}")
    for lab, _, _ in BINS:
        if lab in metrics["per_bin"]:
            b = metrics["per_bin"][lab]
            print(f"  {lab:<10} {b['n']:>4} {b['MAE']:>8.2f} {b['RMSE']:>8.2f} {b['bias']:>+8.2f}")

    # Per-category
    per_cat = compute_per_category(results_by_category)
    print(f"\nPer-category:")
    print(f"  {'Category':<20} {'#':>4} {'MAE':>8} {'RMSE':>8} {'bias':>8} {'meanGT':>8}")
    for cat, m in sorted(per_cat.items()):
        print(f"  {cat:<20} {m['n']:>4} {m['MAE']:>8.2f} {m['RMSE']:>8.2f} {m['bias']:>+8.2f} {m['mean_GT']:>8.1f}")

    # Save
    output = {
        "method": mode,
        "overall": metrics,
        "per_category": per_cat,
        "per_bin": metrics.pop("per_bin"),
        "results": results,
    }
    # Put per_bin back
    metrics["per_bin"] = output["per_bin"]

    if args.out:
        os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.out}")


# ---------------------------------------------------------------------------
# Oracle mode (use GT matched_class for perfect classification)
# ---------------------------------------------------------------------------
def count_image_oracle(d, tau_inst=0.5):
    """Upper bound: perfect classification, heuristic dedup."""
    h, w = int(d["height"]), int(d["width"])
    n_cand = d["z"].shape[0]
    gt_count = int(d.get("gt_count", 0))

    if n_cand == 0:
        return {"pred_count": 0, "n_candidates": 0, "gt_count": gt_count}

    bbox_np = d["bbox"].float().numpy()
    valid = np.asarray(d["valid"]) > 0
    matched_class = np.asarray(d["matched_class"])

    # Build A_sem and A_inst
    n = n_cand
    A_sem = np.eye(n, dtype=np.float32)
    A_inst = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            bi, bj = bbox_np[i], bbox_np[j]
            x1 = max(bi[0], bj[0]); y1 = max(bi[1], bj[1])
            x2 = min(bi[0] + bi[2], bj[0] + bj[2]); y2 = min(bi[1] + bi[3], bj[1] + bj[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area_i = bi[2] * bi[3]; area_j = bj[2] * bj[3]
            union = area_i + area_j - inter
            iou = inter / union if union > 0 else 0.0
            A_inst[i, j] = iou * 10.0
            A_inst[j, i] = iou * 10.0
            A_sem[i, j] = iou * 10.0
            A_sem[j, i] = A_sem[i, j]
    A_part = np.zeros((n, n), dtype=np.float32)

    # Use GT matched_class for grouping
    class_to_indices = defaultdict(list)
    for i in range(n_cand):
        if valid[i] and matched_class[i] >= 0:
            class_to_indices[int(matched_class[i])].append(i)

    # Count per-class with dedup
    total_reps = 0
    for cls_idx, indices in class_to_indices.items():
        if len(indices) == 0:
            continue
        if len(indices) == 1:
            total_reps += 1
            continue

        # Dedup within class
        if len(indices) > 20:
            components = build_same_instance_components_adaptive(
                indices, A_inst, base_tau=tau_inst, use_greedy=True, max_comp_size=5)
        else:
            components = build_same_instance_components(indices, A_inst, tau_inst=tau_inst)

        # Category-aware representative selection (all same class)
        category_probs = np.zeros((n_cand, 1), dtype=np.float32)
        category_probs[:, 0] = 1.0
        reps = select_representatives(
            components, A_part, category_probs, bbox_np, float(h * w),
            min_category_conf=0.0,
        )
        total_reps += len(reps)

    return {
        "pred_count": total_reps,
        "n_candidates": n_cand,
        "n_groups": len(class_to_indices),
        "gt_count": gt_count,
    }


if __name__ == "__main__":
    main()
