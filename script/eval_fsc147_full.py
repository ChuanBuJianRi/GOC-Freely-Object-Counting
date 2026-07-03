"""Evaluate OV-CUD on full FSC147 test set (1190 images) with best config.

Uses the same approach as run_p1_ablations.py but focused on a single best config:
- category_cosine_pts32.pt for classification
- fsc147_relation_pts32_exp5c.pt for relation head (inst logits only)
- category dot-product for A_sem (NOT relation head sem logits)
- tau_inst = 0.99 (best from P0-5)

Also supports class-agnostic mode (no category head) and heuristic-only mode.
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
from code.matrix.pairwise_features import build_pairwise_features, box_geometry
from code.heads.relation_head import PairwiseRelationHead, RelationHeadConfig
from code.matrix.pairwise_features import pairwise_feature_dim

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
# Model loading
# ---------------------------------------------------------------------------
def load_category_head(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    from script.train_category_v2 import CosineCategoryHead
    head = CosineCategoryHead(
        in_dim=ck["in_dim"], proj_dim=ck["proj_dim"],
        dropout=0.3, num_layers=2,
    )
    head.load_state_dict(ck["head"])
    head.to(device).eval()
    print(f"[load] Category head: cosine in_dim={ck['in_dim']}")
    return head


def load_relation_head(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    feat_dim = pairwise_feature_dim(ck["z_dim"])
    cfg = RelationHeadConfig(
        feat_dim=feat_dim, hidden_dim=ck["hidden_dim"],
        num_layers=ck["num_layers"], dropout=0.1,
    )
    head = PairwiseRelationHead(cfg)
    head.load_state_dict(ck["relation_head"])
    head.to(device).eval()
    print(f"[load] Relation head: feat_dim={feat_dim} hidden={ck['hidden_dim']}")
    return head


# ---------------------------------------------------------------------------
# Category probs
# ---------------------------------------------------------------------------
@torch.no_grad()
def get_category_probs(head, z, tp, device):
    from code.training.train_category import forward_logits
    logits, _ = forward_logits(head, z.float().to(device), tp.to(device), None)
    return torch.softmax(logits, dim=-1).cpu().numpy()


# ---------------------------------------------------------------------------
# A_inst from relation head (inst logits only, limited to top-200 by confidence)
# ---------------------------------------------------------------------------
def compute_A_inst(head, z, category_probs, bbox, top_conf, n_cand, device):
    p_t = torch.from_numpy(category_probs).float()
    box_t = bbox.float()
    z_t = z.float()
    max_cand = min(n_cand, 200)
    if n_cand > max_cand:
        keep = np.argsort(-top_conf)[:max_cand]
        z_t = z_t[keep]; p_t = p_t[keep]; box_t = box_t[keep]
        n_sub = max_cand
    else:
        keep = np.arange(n_cand)
        n_sub = n_cand

    ii, jj = np.triu_indices(n_sub, k=1)
    A_inst = np.zeros((n_cand, n_cand), dtype=np.float32)
    if len(ii) == 0:
        return A_inst

    zi, zj = z_t[ii], z_t[jj]; pi, pj = p_t[ii], p_t[jj]
    bi, bj = box_t[ii], box_t[jj]
    geom_ij = box_geometry(bi, bj); geom_ji = box_geometry(bj, bi)
    phi_ij = build_pairwise_features(zi, zj, pi, pj, bi, bj, geom=geom_ij)
    phi_ji = build_pairwise_features(zj, zi, pj, pi, bj, bi, geom=geom_ji)

    with torch.no_grad():
        out = head(torch.cat([phi_ij.to(device), phi_ji.to(device)], dim=0))
        P = len(ii)
        inst_logits = 0.5 * (out["inst"][:P] + out["inst"][P:]).cpu().numpy()

    for k in range(P):
        i, j = (keep[ii[k]], keep[jj[k]]) if n_cand > max_cand else (ii[k], jj[k])
        # Clip inst_logits to avoid overflow: sigmoid(-20) ≈ 0, sigmoid(20) ≈ 1
        clipped = float(np.clip(inst_logits[k], -20, 20))
        A_inst[i, j] = A_inst[j, i] = clipped
    return A_inst


# ---------------------------------------------------------------------------
# A_sem from category probabilities (dot product, like P1 ablations)
# ---------------------------------------------------------------------------
def compute_A_sem(category_probs):
    n = category_probs.shape[0]
    A_sem = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            score = float((category_probs[i] * category_probs[j]).sum())
            A_sem[i, j] = A_sem[j, i] = score * 10.0
    return A_sem


# ---------------------------------------------------------------------------
# A_inst fallback: bbox IoU (for candidates not in top-200)
# ---------------------------------------------------------------------------
def compute_A_inst_bbox(d, A_inst_from_head):
    """Fill gaps in A_inst (where value is 0) with bbox IoU."""
    bbox_np = d["bbox"].float().numpy()
    n = bbox_np.shape[0]
    result = A_inst_from_head.copy()
    for i in range(n):
        for j in range(i + 1, n):
            if abs(result[i, j]) > 1e-6:
                continue  # Already filled by relation head
            bi, bj = bbox_np[i], bbox_np[j]
            x1 = max(bi[0], bj[0]); y1 = max(bi[1], bj[1])
            x2 = min(bi[0] + bi[2], bj[0] + bj[2]); y2 = min(bi[1] + bi[3], bj[1] + bj[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area_i = bi[2] * bi[3]; area_j = bj[2] * bj[3]
            union = area_i + area_j - inter
            iou = inter / union if union > 0 else 0.0
            # IoU of 0.8 → logit of log(0.8/0.2) ≈ 1.39 → bbox IoU * 1.5 gives ~0-1.5 range
            result[i, j] = result[j, i] = iou * 1.5 - 0.75  # center at 0
    return result


# ---------------------------------------------------------------------------
# Single image counting
# ---------------------------------------------------------------------------
def count_image(d, category_head, relation_head, tp, device,
                tau_affinity=0.3, tau_inst=0.99):
    h, w = int(d["height"]), int(d["width"])
    gt = int(d.get("gt_count", 0))
    z = d["z"]
    bbox_np = d["bbox"].float().numpy()
    n_cand = z.shape[0]

    if n_cand == 0:
        return {"pred_count": 0, "gt_count": gt, "n_candidates": 0,
                "n_groups": 0, "n_valid": 0}

    # Category probabilities
    cat_probs = get_category_probs(category_head, z, tp, device)
    top_conf = cat_probs.max(axis=1)

    # Valid candidates
    valid_orig = np.asarray(d["valid"]) > 0
    conf_valid = top_conf >= 0.05  # min confidence
    effective_valid = valid_orig & conf_valid
    valid_indices = [i for i in range(n_cand) if effective_valid[i]]
    N_v = len(valid_indices)

    if N_v <= 1:
        return {"pred_count": N_v, "gt_count": gt, "n_candidates": n_cand,
                "n_groups": N_v, "n_valid": N_v}

    # A_sem from category dot-product
    A_sem = compute_A_sem(cat_probs)

    # A_inst from relation head (or heuristic if disabled)
    if relation_head is not None:
        A_inst_head = compute_A_inst(relation_head, z, cat_probs, d["bbox"],
                                      top_conf, n_cand, device)
    else:
        A_inst_head = np.zeros((n_cand, n_cand), dtype=np.float32)
    A_inst = compute_A_inst_bbox(d, A_inst_head)

    image_area = float(h * w)

    # Clustering
    sub_probs = cat_probs[valid_indices]
    sub_A_sem = A_sem[valid_indices][:, valid_indices]
    groups = category_aware_clustering_with_spatial(
        sub_probs, sub_A_sem, bbox_np[valid_indices], image_area,
        tau_affinity=tau_affinity, max_group_size=30, use_bucketing=True,
    )

    # Dedup + Representative selection per group
    total_reps = 0
    total_components = 0
    for group in groups:
        group_global = [valid_indices[i] for i in group]
        if len(group_global) == 0:
            continue
        if len(group_global) == 1:
            total_reps += 1
            total_components += 1
            continue

        sub_A = A_inst[group_global][:, group_global]
        if len(group_global) > 20:
            components = build_same_instance_components_adaptive(
                list(range(len(group_global))), sub_A, base_tau=tau_inst,
                use_greedy=True, max_comp_size=5)
        else:
            components = build_same_instance_components(
                list(range(len(group_global))), sub_A, tau_inst=tau_inst)

        # Representative selection
        A_part = np.zeros((len(group_global), len(group_global)), dtype=np.float32)
        reps = select_representatives(
            components, A_part, cat_probs[group_global],
            bbox_np[group_global], image_area, min_category_conf=0.05,
        )
        total_reps += len(reps)
        total_components += len(components)

    return {
        "pred_count": total_reps, "gt_count": gt,
        "n_candidates": n_cand, "n_groups": len(groups),
        "n_valid": N_v, "n_components": total_components,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(results):
    if not results:
        return {"MAE": 0, "RMSE": 0, "bias": 0, "n_images": 0}
    preds = np.array([r["pred_count"] for r in results], float)
    gts = np.array([r["gt_count"] for r in results], float)
    err = preds - gts
    m = {
        "n_images": len(results),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)),
        "mean_GT": float(np.mean(gts)),
        "mean_Pred": float(np.mean(preds)),
        "nMAE": float(np.mean(np.abs(err)) / max(np.mean(gts), 1)),
    }
    # Per-bin
    per_bin = {}
    for lab, lo, hi in BINS:
        bin_results = [r for r in results if lo <= r["gt_count"] <= hi]
        if not bin_results:
            continue
        bp = np.array([r["pred_count"] for r in bin_results], float)
        bg = np.array([r["gt_count"] for r in bin_results], float)
        be = bp - bg
        per_bin[lab] = {
            "n": len(bin_results), "MAE": float(np.mean(np.abs(be))),
            "RMSE": float(np.sqrt(np.mean(be ** 2))), "bias": float(np.mean(be)),
        }
    m["per_bin"] = per_bin
    return m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/home/czp/ws_yiyang/ovcud_cache/fsc147_test_3view")
    ap.add_argument("--category-ckpt", default="result/checkpoints/category_cosine_pts32.pt")
    ap.add_argument("--relation-ckpt", default="result/checkpoints/fsc147_relation_pts32_exp5c.pt")
    ap.add_argument("--text-prototypes", default="result/checkpoints/text_prototypes_fsc147.pt")
    ap.add_argument("--images-file", default="result/logs/fsc147_test_1190_images.json")
    ap.add_argument("--ann", default="/home/czp/official_code/dataset/FSC147/annotation_FSC147_384.json")
    ap.add_argument("--out", default="result/logs/fsc147_full_1190.json")
    ap.add_argument("--tau-affinity", type=float, default=0.3)
    ap.add_argument("--tau-inst", type=float, default=0.99)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--no-relation", action="store_true",
                    help="Use heuristic bbox IoU for A_inst instead of relation head")
    args = ap.parse_args()

    device = args.device
    print("=" * 70)
    print("OV-CUD Full FSC147 Evaluation (1190 images)")
    print("=" * 70)
    print(f"tau_inst={args.tau_inst}, tau_affinity={args.tau_affinity}")
    print(f"Relation head: {'disabled (heuristic IoU)' if args.no_relation else 'enabled'}")

    # Load models
    cat_head = load_category_head(args.category_ckpt, device)
    tp = torch.load(args.text_prototypes, map_location="cpu", weights_only=False)
    tp = torch.nn.functional.normalize(tp.float(), dim=-1)
    rel_head = None if args.no_relation else load_relation_head(args.relation_ckpt, device)

    # Load image list
    wanted = None
    if args.images_file and os.path.exists(args.images_file):
        with open(args.images_file) as f:
            img_list = json.load(f)
        if isinstance(img_list, list):
            # Convert filenames to cache base names
            wanted = {os.path.splitext(fn)[0] for fn in img_list}
            print(f"[init] {len(wanted)} images in test set")

    # Load caches
    files = sorted(Path(args.cache_dir).glob("*.pt"))
    if args.limit > 0:
        files = files[:args.limit]

    results = []
    n_skipped = 0
    t_start = time.time()

    for i, fp in enumerate(files):
        base = fp.stem
        if wanted is not None and base not in wanted:
            n_skipped += 1
            continue

        d = torch.load(fp, map_location="cpu", weights_only=False)
        file_name = d.get("file_name", fp.name)
        gt = int(d.get("gt_count", 0))

        if d["z"].shape[0] == 0:
            results.append({"pred_count": 0, "gt_count": gt, "n_candidates": 0,
                           "n_groups": 0, "n_valid": 0, "file_name": file_name})
            continue

        r = count_image(d, cat_head, rel_head, tp, device,
                       tau_affinity=args.tau_affinity, tau_inst=args.tau_inst)
        r["file_name"] = file_name
        results.append(r)

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t_start
            n_eval = len(results)
            rate = n_eval / max(elapsed, 0.01)
            m = compute_metrics(results)
            print(f"  [{i+1}/{len(files)}] eval={n_eval} skip={n_skipped} "
                  f"MAE={m['MAE']:.2f}  rate={rate:.1f}/s  "
                  f"ETA={((len(files)-i-1)/max(rate,0.01)/60):.1f}min")

    elapsed = time.time() - t_start
    m = compute_metrics(results)

    print(f"\n{'='*70}")
    print(f"FSC147 Full Test Results (n={m['n_images']})")
    print(f"{'='*70}")
    print(f"MAE: {m['MAE']:.2f}  RMSE: {m['RMSE']:.2f}  bias: {m['bias']:+.2f}")
    print(f"Mean GT: {m['mean_GT']:.1f}  Mean Pred: {m['mean_Pred']:.1f}")
    print(f"nMAE: {m['nMAE']:.3f}")
    print(f"Runtime: {elapsed:.1f}s ({elapsed/max(m['n_images'],1)*1000:.1f}ms/img)")

    print(f"\nPer-bin:")
    print(f"  {'Bin':<10} {'#':>5} {'MAE':>9} {'RMSE':>9} {'bias':>9}")
    for lab, _, _ in BINS:
        if lab in m["per_bin"]:
            b = m["per_bin"][lab]
            print(f"  {lab:<10} {b['n']:>5} {b['MAE']:>9.2f} {b['RMSE']:>9.2f} {b['bias']:>+9.2f}")

    # Save
    if args.out:
        os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
        output = {
            "config": {
                "cache_dir": args.cache_dir,
                "category_ckpt": args.category_ckpt,
                "relation_ckpt": args.relation_ckpt,
                "tau_inst": args.tau_inst,
                "tau_affinity": args.tau_affinity,
                "no_relation": args.no_relation,
            },
            "overall": {k: v for k, v in m.items() if k != "per_bin"},
            "per_bin": m["per_bin"],
            "results": [{k: v for k, v in r.items()}
                       for r in sorted(results, key=lambda x: x.get("file_name", ""))],
        }
        with open(args.out, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
