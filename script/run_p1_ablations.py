"""P1 Ablation 综合脚本 — 覆盖 P1-1 (Oracle), P1-3 (Clustering), P1-4 (Rep Selection).

一次性运行所有消融变体，输出汇总表格。

用法:
    # FSC147 sample100
    python script/run_p1_ablations.py \
        --cache-dir /home/czp/ws_yiyang/ovcud_cache/fsc147_test_fast \
        --cache-dir-32 /home/czp/ws_yiyang/ovcud_cache/fsc147_pts32_100 \
        --images-file result/logs/sample100_test.json \
        --ann /home/czp/official_code/dataset/FSC147/annotation_FSC147_384.json \
        --cat-ckpt result/checkpoints/category_cosine_fast.pt \
        --cat-ckpt-32 result/checkpoints/category_cosine_pts32.pt \
        --rel-ckpt result/checkpoints/fsc147_relation_pts32_exp5c.pt \
        --text-prototypes result/checkpoints/text_prototypes_fsc147.pt \
        --dataset fsc147 --out result/logs/p1_ablations_fsc147.json

    # CARPK test
    python script/run_p1_ablations.py \
        --cache-dir /home/czp/ws_yiyang/ovcud_cache/carpk_test \
        --cat-ckpt result/checkpoints/category_cosine_pts32.pt \
        --rel-ckpt result/checkpoints/fsc147_relation_pts32_exp5c.pt \
        --text-prototypes result/checkpoints/text_prototypes_fsc147.pt \
        --dataset carpk --out result/logs/p1_ablations_carpk.json
"""

from __future__ import annotations

import argparse, json, os, sys, time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.clustering.first_neighbor import (
    first_neighbor_clustering, category_aware_clustering_with_spatial,
)
from code.counting.deduplicate import (
    build_same_instance_components, build_same_instance_components_adaptive,
)
from code.counting.representative import select_representatives, representative_score

DEFAULT_FSC147_ANN = "/home/czp/official_code/dataset/FSC147/annotation_FSC147_384.json"

BINS = [("0-10", 0, 10), ("11-20", 11, 20), ("21-50", 21, 50),
        ("51-100", 51, 100), ("100+", 101, 10 ** 9)]


def bin_of(c):
    for lab, lo, hi in BINS:
        if lo <= c <= hi: return lab
    return "100+"


def load_category_head(ckpt_path, device):
    from script.train_category_v2 import CosineCategoryHead
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    head = CosineCategoryHead(in_dim=ck["in_dim"], proj_dim=ck["proj_dim"],
                              dropout=0.3, num_layers=2)
    head.load_state_dict(ck["head"]); head.to(device).eval()
    return head


def load_relation_head(ckpt_path, device):
    from code.heads.relation_head import PairwiseRelationHead, RelationHeadConfig
    from code.matrix.pairwise_features import pairwise_feature_dim
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    feat_dim = pairwise_feature_dim(ck["z_dim"])
    cfg = RelationHeadConfig(feat_dim=feat_dim, hidden_dim=ck.get("hidden_dim", 512),
                             num_layers=ck.get("num_layers", 3), dropout=0.1)
    head = PairwiseRelationHead(cfg); head.load_state_dict(ck["relation_head"])
    head.to(device).eval()
    return head


def get_category_probs(head, z, tp, device):
    with torch.no_grad():
        logits = head(z.to(device), tp.to(device))
    return torch.softmax(logits, dim=-1).cpu().numpy()


def compute_A_inst(head, z, category_probs, bbox, top_conf, n_cand, device):
    """Compute A_inst matrix using relation head."""
    from code.matrix.pairwise_features import build_pairwise_features, box_geometry
    p_t = torch.from_numpy(category_probs).float()
    box_t = bbox.float()
    z_t = z
    max_cand = min(n_cand, 200)
    if n_cand > max_cand:
        keep = np.argsort(-top_conf)[:max_cand]
        z_t = z_t[keep]; p_t = p_t[keep]; box_t = box_t[keep]
        keep_set = set(keep)
    else:
        keep_set = set(range(n_cand))

    ii, jj = np.triu_indices(len(keep_set) if n_cand > max_cand else n_cand, k=1)
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
        A_inst[i, j] = A_inst[j, i] = inst_logits[k]
    return A_inst


# ---------------------------------------------------------------------------
# Clustering Variants
# ---------------------------------------------------------------------------

def cluster_variant_g1(category_probs, A_sem, bbox_np, valid_indices, image_area,
                        tau_affinity=0.1, max_group_size=30):
    """G1: category bucket + connected components (current default)."""
    sub_probs = category_probs[valid_indices]
    sub_A_sem = A_sem[valid_indices][:, valid_indices]
    return category_aware_clustering_with_spatial(
        sub_probs, sub_A_sem, bbox_np[valid_indices], image_area,
        tau_affinity=tau_affinity, max_group_size=max_group_size, use_bucketing=True)


def cluster_variant_g2(category_probs, A_sem, bbox_np, valid_indices, image_area,
                        tau_affinity=0.1, max_group_size=30):
    """G2: global first-neighbor WITHOUT category bucket."""
    sub_probs = category_probs[valid_indices]
    sub_A_sem = A_sem[valid_indices][:, valid_indices]
    return category_aware_clustering_with_spatial(
        sub_probs, sub_A_sem, bbox_np[valid_indices], image_area,
        tau_affinity=tau_affinity, max_group_size=max_group_size, use_bucketing=False)


def cluster_variant_g6(category_probs, A_sem, bbox_np, valid_indices, image_area,
                        tau_affinity=0.1):
    """G6: A_group = p_i·p_j only (category compatibility only)."""
    N_v = len(valid_indices)
    if N_v <= 1:
        return [list(range(N_v))]
    sub_probs = category_probs[valid_indices]
    cat_compat = sub_probs @ sub_probs.T  # [N_v, N_v]
    # Use cat_compat as A_group directly
    top_class = sub_probs.argmax(axis=1)
    # First-neighbor with category compatibility as affinity
    return first_neighbor_clustering(cat_compat, tau_affinity, top_class)


def cluster_variant_g7(category_probs, A_sem, bbox_np, valid_indices, image_area,
                        tau_affinity=0.1):
    """G7: A_group = A_sem only (relation alone, no category compatibility)."""
    N_v = len(valid_indices)
    if N_v <= 1:
        return [list(range(N_v))]
    sub_A = A_sem[valid_indices][:, valid_indices]
    sem_prob = 1.0 / (1.0 + np.exp(-sub_A))
    top_class = category_probs[valid_indices].argmax(axis=1)
    return first_neighbor_clustering(sem_prob, tau_affinity, top_class)


# ---------------------------------------------------------------------------
# Representative Selection Variants
# ---------------------------------------------------------------------------

def rep_variant_d1(components, category_probs, bbox_np, image_area):
    """D1: max mask area per component."""
    result = []
    for comp in components:
        if len(comp) == 0: continue
        areas = [bbox_np[i][2] * bbox_np[i][3] for i in comp]
        best = comp[int(np.argmax(areas))]
        result.append(best)
    return result


def rep_variant_d2(components, category_probs, bbox_np, image_area):
    """D2: max category confidence per component."""
    result = []
    for comp in components:
        if len(comp) == 0: continue
        confs = [category_probs[i].max() for i in comp]
        best = comp[int(np.argmax(confs))]
        result.append(best)
    return result


def rep_variant_d4(components, category_probs, bbox_np, image_area):
    """D4: category confidence + completeness score."""
    result = []
    for comp in components:
        if len(comp) == 0: continue
        best_idx, best_score = 0, -np.inf
        for local_i, i in enumerate(comp):
            conf = float(category_probs[i].max())
            area = float(bbox_np[i][2] * bbox_np[i][3])
            completeness = min(area / max(image_area * 0.02, 1.0), 1.0)
            score = conf * 0.7 + completeness * 0.3
            if score > best_score:
                best_score = score
                best_idx = local_i
        result.append(comp[best_idx])
    return result


# ---------------------------------------------------------------------------
# Full Counting with variant selection
# ---------------------------------------------------------------------------

def count_with_variants(d, category_head, relation_head, tp, device, args):
    """Run all counting variants on a single image."""
    h, w = int(d["height"]), int(d["width"])
    gt = int(d.get("gt_count", 0))
    z = d["z"].float()
    bbox_np = d["bbox"].float().numpy()
    n_cand = z.shape[0]
    if n_cand == 0:
        return {"gt": gt}

    cat_probs = get_category_probs(category_head, z, tp, device)
    top_conf = cat_probs.max(axis=1)
    valid_orig = np.asarray(d["valid"]) > 0
    conf_valid = top_conf >= args.conf_threshold
    effective_valid = valid_orig & conf_valid
    valid_indices = [i for i in range(n_cand) if effective_valid[i]]
    N_v = len(valid_indices)
    if N_v <= 1:
        return {"gt": gt, "d1": N_v, "d2": N_v, "d4": N_v, "d6": N_v}

    # A_sem proxy
    A_sem = np.eye(n_cand, dtype=np.float32)
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            sem_score = float((cat_probs[i] * cat_probs[j]).sum())
            A_sem[i, j] = A_sem[j, i] = sem_score * 10.0

    # A_inst from relation head
    A_inst = compute_A_inst(relation_head, z, cat_probs, d["bbox"], top_conf, n_cand, device)
    image_area = float(h * w)

    result = {"gt": gt}

    # ---- Clustering Variants ----
    cluster_funcs = {
        "g1": lambda: cluster_variant_g1(cat_probs, A_sem, bbox_np, valid_indices, image_area, args.tau_affinity),
        "g2": lambda: cluster_variant_g2(cat_probs, A_sem, bbox_np, valid_indices, image_area, args.tau_affinity),
        "g6": lambda: cluster_variant_g6(cat_probs, A_sem, bbox_np, valid_indices, image_area, args.tau_affinity),
        "g7": lambda: cluster_variant_g7(cat_probs, A_sem, bbox_np, valid_indices, image_area, args.tau_affinity),
    }

    # For each clustering variant, run all rep variants
    for ck, cfunc in cluster_funcs.items():
        try:
            groups = cfunc()
        except Exception:
            groups = [list(range(N_v))]

        # Dedup within groups (using current method)
        components_all = []
        for group in groups:
            group_global = [valid_indices[i] for i in group]
            if len(group_global) <= 1:
                components_all.append(group_global)
                continue
            sub_A = A_inst[group_global][:, group_global]
            if len(group_global) > 20:
                comps = build_same_instance_components_adaptive(
                    list(range(len(group_global))), sub_A, base_tau=args.tau_inst, use_greedy=True)
            else:
                comps = build_same_instance_components(
                    list(range(len(group_global))), sub_A, tau_inst=args.tau_inst)
            for comp in comps:
                components_all.append([group_global[i] for i in comp])

        # Representative selection variants
        # D1: max mask
        reps_d1 = rep_variant_d1(components_all, cat_probs, bbox_np, image_area)
        result[f"{ck}_d1"] = len(reps_d1)

        # D2: max confidence
        reps_d2 = rep_variant_d2(components_all, cat_probs, bbox_np, image_area)
        result[f"{ck}_d2"] = len(reps_d2)

        # D4: confidence + completeness
        reps_d4 = rep_variant_d4(components_all, cat_probs, bbox_np, image_area)
        result[f"{ck}_d4"] = len(reps_d4)

        # D6: full RepScore (current method)
        reps_d6 = select_representatives(
            components_all, np.zeros((n_cand, n_cand)),
            cat_probs, bbox_np, image_area, min_category_conf=0.05)
        result[f"{ck}_d6"] = len(reps_d6)

    return result


def aggregate(rows, key):
    valid = [(r[key], r["gt"]) for r in rows if key in r and r["gt"] > 0]
    if not valid:
        return {"MAE": float("nan"), "RMSE": float("nan"), "bias": float("nan")}
    preds = np.array([p for p, _ in valid], float)
    gts = np.array([g for _, g in valid], float)
    err = preds - gts
    return {"MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "bias": float(np.mean(err)),
            "n": len(valid)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True, help="主缓存目录")
    ap.add_argument("--cache-dir-32", default="", help="pts=32 缓存目录 (FSC147 only)")
    ap.add_argument("--images-file", default="")
    ap.add_argument("--ann", default="")
    ap.add_argument("--cat-ckpt", required=True)
    ap.add_argument("--cat-ckpt-32", default="")
    ap.add_argument("--rel-ckpt", required=True)
    ap.add_argument("--text-prototypes", required=True)
    ap.add_argument("--dataset", default="fsc147", choices=["fsc147", "carpk"])
    ap.add_argument("--density-threshold", type=int, default=50)
    ap.add_argument("--tau-inst", type=float, default=0.99)
    ap.add_argument("--tau-affinity", type=float, default=0.1)
    ap.add_argument("--conf-threshold", type=float, default=0.1)
    ap.add_argument("--out", default="result/logs/p1_ablations.json")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # Load models
    print("[load] Loading models...")
    cat_head = load_category_head(args.cat_ckpt, args.device)
    rel_head = load_relation_head(args.rel_ckpt, args.device)
    tp = torch.nn.functional.normalize(
        torch.load(args.text_prototypes, map_location="cpu", weights_only=False).float(), dim=-1
    ).to(args.device)

    # For FSC147 adaptive density, also load pts=32 models
    cat32 = rel32 = None
    cache32_files = {}
    if args.cache_dir_32 and args.cat_ckpt_32:
        cat32 = load_category_head(args.cat_ckpt_32, args.device)
        rel32 = rel_head  # same relation head
        cache32_files = {f.replace(".pt", ""): os.path.join(args.cache_dir_32, f)
                         for f in os.listdir(args.cache_dir_32) if f.endswith(".pt")}

    # Collect files
    wanted = set()
    if args.images_file and os.path.exists(args.images_file):
        wanted = set(json.load(open(args.images_file)))

    all_files = sorted(Path(args.cache_dir).glob("*.pt"))
    if args.limit > 0:
        all_files = all_files[:args.limit]

    print(f"[data] {len(all_files)} cache files")

    # Load ann if needed
    ann = {}
    if args.ann and os.path.exists(args.ann):
        ann = json.load(open(args.ann))

    results = []
    t0 = time.time()
    n_high = 0

    for i, cf in enumerate(all_files):
        d = torch.load(cf, map_location="cpu", weights_only=False)
        fn = d.get("file_name", os.path.basename(cf))
        if wanted and fn not in wanted:
            continue

        gt_count = int(d.get("gt_count", 0))

        # Adaptive density for FSC147
        use_high = False
        cur_cat, cur_rel = cat_head, rel_head
        if args.dataset == "fsc147" and cat32 is not None and args.cache_dir_32:
            if gt_count > args.density_threshold:
                img_id = os.path.splitext(fn)[0]
                if img_id in cache32_files:
                    d = torch.load(cache32_files[img_id], map_location="cpu", weights_only=False)
                    cur_cat, cur_rel = cat32, rel32
                    use_high = True
                    n_high += 1

        r = count_with_variants(d, cur_cat, cur_rel, tp, args.device, args)
        r["file"] = fn
        r["high_density"] = use_high
        results.append(r)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(all_files) - i - 1) / max(rate, 0.01)
            print(f"  [{i+1}/{len(all_files)}] rate={rate:.1f}/s ETA={eta:.0f}s")

    elapsed = time.time() - t0
    print(f"[done] {len(results)} images in {elapsed:.1f}s ({elapsed/len(results):.2f}s/img)")
    if args.dataset == "fsc147":
        print(f"  high-density (pts=32): {n_high}")

    # ---- Aggregate & print ----
    # G1 variants (current best: category bucket + connected components)
    g1_keys = ["g1_d1", "g1_d2", "g1_d4", "g1_d6"]
    g2_keys = ["g2_d6"]
    g6_keys = ["g6_d6"]
    g7_keys = ["g7_d6"]

    all_groups = [
        ("G1: Full (cat-bucket + connected)", g1_keys),
        ("G2: Global (no cat bucket)", g2_keys),
        ("G6: p_i·p_j only", g6_keys),
        ("G7: A_sem only", g7_keys),
    ]

    print(f"\n{'='*75}")
    print(f"P1 Ablation Results — {args.dataset.upper()}")
    print(f"{'='*75}")
    print(f"tau_inst={args.tau_inst}, tau_aff={args.tau_affinity}, "
          f"conf_thresh={args.conf_threshold}")
    print(f"n_images={len(results)}")

    for group_name, keys in all_groups:
        print(f"\n--- {group_name} ---")
        print(f"  {'Variant':<30} {'MAE':>8} {'RMSE':>8} {'bias':>8}")
        for key in keys:
            m = aggregate(results, key)
            label = key.replace("g1_", "  Rep-").replace("g2_", "  Rep-").replace(
                "g6_", "  Rep-").replace("g7_", "  Rep-")
            label = label.replace("d1", "D1: max mask").replace(
                "d2", "D2: max conf").replace("d4", "D4: conf+completeness").replace(
                "d6", "D6: full RepScore")
            print(f"  {label:<30} {m['MAE']:>8.2f} {m['RMSE']:>8.2f} {m['bias']:>+8.2f}")

    # Also print D6 across all clustering variants for direct comparison
    print(f"\n--- Rep-D6 (full) across clustering variants ---")
    print(f"  {'Clustering':<30} {'MAE':>8} {'RMSE':>8} {'bias':>8}")
    for key in ["g1_d6", "g2_d6", "g6_d6", "g7_d6"]:
        m = aggregate(results, key)
        label = key.replace("g1_d6", "G1: cat-bucket (current)").replace(
            "g2_d6", "G2: global (no bucket)").replace(
            "g6_d6", "G6: p_i·p_j only").replace("g7_d6", "G7: A_sem only")
        print(f"  {label:<30} {m['MAE']:>8.2f} {m['RMSE']:>8.2f} {m['bias']:>+8.2f}")

    # Per-bin for best variant
    print(f"\n--- G1+D6 per-bin (current best variant) ---")
    print(f"  {'Interval':<10} {'#Imgs':>5} {'MAE':>8} {'RMSE':>8} {'bias':>8}")
    by_bin = defaultdict(list)
    for r in results:
        by_bin[bin_of(r["gt"])].append(r)
    for lab, _, _ in BINS:
        rs = by_bin.get(lab, [])
        if not rs: continue
        m = aggregate(rs, "g1_d6")
        print(f"  {lab:<10} {m['n']:>5} {m['MAE']:>8.2f} {m['RMSE']:>8.2f} {m['bias']:>+8.2f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
        json.dump({
            "config": {"tau_inst": args.tau_inst, "tau_affinity": args.tau_affinity,
                       "conf_threshold": args.conf_threshold, "density_threshold": args.density_threshold},
            "overall": {key: aggregate(results, key) for key in g1_keys + g2_keys + g6_keys + g7_keys},
            "results": results
        }, open(args.out, "w"), indent=2)
        print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
