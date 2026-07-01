"""P2 Experiments: Fully Prompt-Free Protocol + COCO Multi-Category Counting.

P2-5: Fully prompt-free single-count protocol on FSC147
  - Runs the full counting pipeline
  - Compares class-aware protocol (GT class matching) vs fully prompt-free auto-selection
  - Auto-selection heuristics: largest group, highest confidence, highest quality, most countable

P2-2: COCO-count subset multi-category evaluation
  - Runs counting pipeline on COCO val images
  - Per-class MAE/RMSE breakdown
  - Multi-category class-aware result

Usage:
  # P2-5: FSC147 fully prompt-free
  python script/run_p2_experiments.py \
      --experiment p25 \
      --fsc147-cache /home/czp/ws_yiyang/ovcud_cache/fsc147_test_pts32_sample \
      --fsc147-ann /home/czp/official_code/dataset/FSC147/annotation_FSC147_384.json \
      --output result/logs/p2_prompt_free.json

  # P2-2: COCO multi-category
  python script/run_p2_experiments.py \
      --experiment p22 \
      --coco-cache /home/czp/ws_yiyang/ovcud_cache/coco_val_3view \
      --output result/logs/p2_coco_count.json

  # Both
  python script/run_p2_experiments.py --experiment all --output result/logs/p2_all.json
"""

from __future__ import annotations

import argparse, json, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from code.clustering.first_neighbor import category_aware_clustering_with_spatial
from code.counting.deduplicate import build_same_instance_components, build_same_instance_components_adaptive
from code.counting.representative import select_representatives
from code.matrix.pairwise_features import build_pairwise_features, box_geometry

BINS = [("0-10", 0, 10), ("11-20", 11, 20), ("21-50", 21, 50),
        ("51-100", 51, 100), ("100+", 101, 10**9)]


def bin_of(c):
    for lab, lo, hi in BINS:
        if lo <= c <= hi: return lab
    return "100+"


# =========================================================================== #
# Model Loading
# =========================================================================== #

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


# =========================================================================== #
# Core Pipeline (returns per-group info for prompt-free analysis)
# =========================================================================== #

@torch.no_grad()
def count_image_with_groups(d, category_head, relation_head, text_prototypes, device,
                             tau_inst=0.99, tau_aff=0.1, conf_threshold=0.2,
                             class_names=None, gt_class_name=None):
    """Run counting pipeline and return per-group details for prompt-free analysis.

    Returns:
        groups_info: list of dicts with keys:
            class_idx, class_name, pred_count, n_candidates, mean_conf, group_quality
        pred_count: total reps across all groups (for class-aware comparison)
    """
    h, w = int(d["height"]), int(d["width"])
    z = d["z"].float(); bbox_np = d["bbox"].float().numpy()
    n_cand = z.shape[0]
    gt_count = int(d.get("gt_count", 0))

    if n_cand == 0:
        return [], 0, gt_count

    # 1. Category probs
    with torch.no_grad():
        logits = category_head(z.to(device), text_prototypes.to(device))
    cat_probs = torch.softmax(logits, dim=-1).cpu().numpy()
    top_conf = cat_probs.max(axis=1)
    top_class = cat_probs.argmax(axis=1)

    # Filter
    valid_orig = np.asarray(d.get("valid", np.ones(n_cand))) > 0
    conf_valid = top_conf >= conf_threshold
    effective = valid_orig & conf_valid
    valid_idx = np.where(effective)[0]
    if len(valid_idx) <= 1:
        return [], len(valid_idx), gt_count

    # 2. A_sem
    A_sem = np.eye(n_cand, dtype=np.float32)
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            A_sem[i, j] = A_sem[j, i] = float((cat_probs[i] * cat_probs[j]).sum()) * 10.0

    # 3. A_inst
    max_cand = min(n_cand, 200)
    if n_cand > max_cand:
        keep = np.argsort(-top_conf)[:max_cand]
        z_sub = z[keep]; p_sub = torch.from_numpy(cat_probs[keep]).float()
        b_sub = d["bbox"].float()[keep]
        keep_set = set(keep)
    else:
        z_sub = z; p_sub = torch.from_numpy(cat_probs).float()
        b_sub = d["bbox"].float(); keep_set = set(range(n_cand))

    ii, jj = np.triu_indices(len(p_sub), k=1)
    A_inst = np.zeros((n_cand, n_cand), dtype=np.float32)
    if len(ii) > 0:
        zi = z_sub[ii].to(device); zj = z_sub[jj].to(device)
        pi = p_sub[ii].to(device); pj = p_sub[jj].to(device)
        bi = b_sub[ii].to(device); bj = b_sub[jj].to(device)
        geom_ij = box_geometry(bi, bj).to(device); geom_ji = box_geometry(bj, bi).to(device)
        phi_ij = build_pairwise_features(zi, zj, pi, pj, bi, bj, geom=geom_ij).to(device)
        phi_ji = build_pairwise_features(zj, zi, pj, pi, bj, bi, geom=geom_ji).to(device)
        out = relation_head(torch.cat([phi_ij, phi_ji], dim=0))
        P = len(ii)
        inst_logits = 0.5 * (out["inst"][:P] + out["inst"][P:]).cpu().numpy()
        for k in range(P):
            i, j = (keep[ii[k]], keep[jj[k]]) if n_cand > max_cand else (ii[k], jj[k])
            A_inst[i, j] = A_inst[j, i] = float(inst_logits[k])

    # 4. Clustering
    sub_probs = cat_probs[valid_idx]
    sub_A_sem = A_sem[np.ix_(valid_idx, valid_idx)]
    image_area = float(h * w)

    groups = category_aware_clustering_with_spatial(
        sub_probs, sub_A_sem, bbox_np[valid_idx], image_area,
        tau_affinity=tau_aff, use_bucketing=True,
    )

    # 5. Dedup + Rep per group — collect detailed info
    groups_info = []
    for group in groups:
        group_global = [valid_idx[i] for i in group]
        if len(group_global) == 0:
            continue
        sub_A = A_inst[np.ix_(group_global, group_global)] if len(group_global) > 1 else None

        # Dedup
        if len(group_global) == 1:
            components = [group_global]
        elif len(group_global) > 20:
            comps_local = build_same_instance_components_adaptive(
                list(range(len(group_global))), sub_A, base_tau=tau_inst, use_greedy=True)
            components = [[group_global[i] for i in comp] for comp in comps_local]
        else:
            comps_local = build_same_instance_components(
                list(range(len(group_global))), sub_A, tau_inst=tau_inst)
            components = [[group_global[i] for i in comp] for comp in comps_local]

        # Representative selection
        reps = select_representatives(
            [[c for c in comp] for comp in components],
            np.zeros((n_cand, n_cand)),
            cat_probs, bbox_np, image_area, min_category_conf=0.05,
        )

        # Determine dominant class for this group
        group_class_idx = int(np.bincount(top_class[group_global]).argmax())
        group_mean_conf = float(top_conf[group_global].mean())

        # Group quality: n_candidates * mean_confidence * (1 - duplicate_rate)
        dup_rate = 1.0 - len(components) / max(len(group_global), 1)
        group_quality = len(group_global) * group_mean_conf * (1.0 - dup_rate)

        group_info = {
            "class_idx": group_class_idx,
            "class_name": class_names[group_class_idx] if class_names else str(group_class_idx),
            "pred_count": len(reps),
            "n_candidates": len(group_global),
            "n_components": len(components),
            "mean_conf": group_mean_conf,
            "group_quality": group_quality,
            "dup_rate": dup_rate,
        }
        groups_info.append(group_info)

    total_pred = sum(g["pred_count"] for g in groups_info)
    return groups_info, total_pred, gt_count


# =========================================================================== #
# P2-5: Fully Prompt-Free Protocol
# =========================================================================== #

def run_p25(args, cat_head, rel_head, tp, device, class_names):
    """Run P2-5: Compare class-aware vs fully prompt-free protocols on FSC147."""
    from pycocotools import mask as mask_utils

    # Load FSC147 annotations for GT class matching
    fsc147_ann = json.load(open(args.fsc147_ann)) if args.fsc147_ann and os.path.exists(args.fsc147_ann) else {}

    cache_files = sorted(Path(args.fsc147_cache).glob("*.pt"))
    if args.limit > 0:
        cache_files = cache_files[:args.limit]
    print(f"[P2-5] {len(cache_files)} FSC147 images")

    # Results collectors
    class_aware_errors = []      # Match GT class
    prompt_free_errors = {       # Auto-select heuristics
        "largest": [],
        "highest_conf": [],
        "highest_quality": [],
        "most_countable": [],
        "all_sum": [],
    }

    t0 = time.time()
    for fi, cf in enumerate(cache_files):
        d = torch.load(cf, map_location="cpu", weights_only=False)
        fn = d.get("file_name", os.path.basename(cf))
        gt_class = d.get("class_name", "")
        gt_count = int(d.get("gt_count", 0))

        if gt_count <= 0:
            continue

        # Run pipeline
        groups_info, total_pred, _ = count_image_with_groups(
            d, cat_head, rel_head, tp, device,
            tau_inst=args.tau_inst, tau_aff=args.tau_aff,
            conf_threshold=args.conf_threshold,
            class_names=class_names, gt_class_name=gt_class,
        )

        if not groups_info:
            # No groups detected → pred=0 for all protocols
            class_aware_errors.append(abs(0 - gt_count))
            for key in prompt_free_errors:
                prompt_free_errors[key].append(abs(0 - gt_count))
            continue

        # ---- Class-Aware Protocol ----
        # Find groups matching GT class
        matching_groups = [g for g in groups_info if g["class_name"] == gt_class]
        if matching_groups:
            class_aware_count = sum(g["pred_count"] for g in matching_groups)
        else:
            # Fallback: use dominant class group
            class_aware_count = max(groups_info, key=lambda g: g["n_candidates"])["pred_count"]
        class_aware_errors.append(abs(class_aware_count - gt_count))

        # ---- Fully Prompt-Free Protocols ----
        # Heuristic 1: Largest group (most candidates before dedup)
        largest = max(groups_info, key=lambda g: g["n_candidates"])
        prompt_free_errors["largest"].append(abs(largest["pred_count"] - gt_count))

        # Heuristic 2: Highest confidence group
        highest_conf = max(groups_info, key=lambda g: g["mean_conf"])
        prompt_free_errors["highest_conf"].append(abs(highest_conf["pred_count"] - gt_count))

        # Heuristic 3: Highest quality group
        highest_quality = max(groups_info, key=lambda g: g["group_quality"])
        prompt_free_errors["highest_quality"].append(abs(highest_quality["pred_count"] - gt_count))

        # Heuristic 4: Group with most reps (most countable)
        most_countable = max(groups_info, key=lambda g: g["pred_count"])
        prompt_free_errors["most_countable"].append(abs(most_countable["pred_count"] - gt_count))

        # Heuristic 5: Sum of ALL groups (current default behavior)
        prompt_free_errors["all_sum"].append(abs(total_pred - gt_count))

        if (fi + 1) % 50 == 0:
            mae_ca = np.mean(class_aware_errors)
            mae_pf = np.mean(prompt_free_errors["largest"])
            elapsed = time.time() - t0
            print(f"  [{fi+1}/{len(cache_files)}] class-aware MAE={mae_ca:.2f}, "
                  f"prompt-free(largest) MAE={mae_pf:.2f}  rate={fi/elapsed:.1f}/s")

    # Aggregate
    results = {}
    results["class_aware"] = {
        "mae": float(np.mean(class_aware_errors)),
        "rmse": float(np.sqrt(np.mean(np.array(class_aware_errors)**2))),
        "n_images": len(class_aware_errors),
    }
    for key, errors in prompt_free_errors.items():
        if errors:
            e = np.array(errors)
            results[f"prompt_free_{key}"] = {
                "mae": float(np.mean(e)),
                "rmse": float(np.sqrt(np.mean(e**2))),
                "n_images": len(e),
            }

    return results


# =========================================================================== #
# P2-2: COCO Multi-Category Counting
# =========================================================================== #

def run_p22(args, cat_head, rel_head, tp, device, class_names):
    """Run P2-2: Multi-category class-agnostic counting on COCO val.

    Uses COCO-trained head + COCO text prototypes.
    Evaluates total-count MAE (class-agnostic) since COCO images are multi-class.
    """
    from pycocotools import mask as mask_utils

    cache_files = sorted(Path(args.coco_cache).glob("*.pt"))
    if args.limit > 0:
        cache_files = cache_files[:args.limit]
    print(f"[P2-2] {len(cache_files)} COCO val images")

    # Determine dominant GT class from matched_class (valid candidates only)
    # Use this for class-aware matching
    all_errors = []
    per_dominant_class = defaultdict(list)
    per_bin = defaultdict(list)

    # For COCO, use class_agnostic approach (count all instances)
    class_agnostic_errors = []
    per_image_details = []

    t0 = time.time()
    for fi, cf in enumerate(cache_files):
        d = torch.load(cf, map_location="cpu", weights_only=False)
        gt_count = int(d.get("gt_count", 0))
        if gt_count <= 0:
            continue

        # Determine dominant GT class from matched_class
        valid_mask = d["valid"].numpy() > 0
        matched = d["matched_class"].numpy()
        valid_classes = matched[valid_mask]
        valid_classes = valid_classes[valid_classes >= 0]
        if len(valid_classes) > 0:
            dom_class_idx = int(np.bincount(valid_classes).argmax())
            dom_class_name = class_names[dom_class_idx] if dom_class_idx < len(class_names) else f"cls_{dom_class_idx}"
        else:
            dom_class_idx = -1
            dom_class_name = "unknown"

        # Run pipeline
        groups_info, total_pred, _ = count_image_with_groups(
            d, cat_head, rel_head, tp, device,
            tau_inst=args.tau_inst, tau_aff=args.tau_aff,
            conf_threshold=args.conf_threshold,
            class_names=class_names, gt_class_name=dom_class_name,
        )

        # Class-agnostic: sum all groups
        pred_count = total_pred
        err = abs(pred_count - gt_count)
        all_errors.append(err)
        class_agnostic_errors.append(err)

        # Per-dominant-class stats
        per_dominant_class[dom_class_name].append(err)

        # Per-bin stats
        per_bin[bin_of(gt_count)].append(err)

        per_image_details.append({
            "file": d.get("file_name", os.path.basename(cf)),
            "gt_count": gt_count, "pred_count": pred_count,
            "dominant_class": dom_class_name,
            "n_groups": len(groups_info),
            "error": err,
        })

        if (fi + 1) % 50 == 0:
            maes = np.mean(all_errors)
            elapsed = time.time() - t0
            print(f"  [{fi+1}/{len(cache_files)}] MAE={maes:.2f}  rate={fi/elapsed:.1f}/s")

    # Aggregate
    results = {
        "overall": {
            "mae": float(np.mean(all_errors)),
            "rmse": float(np.sqrt(np.mean(np.array(all_errors)**2))),
            "n_images": len(all_errors),
            "mean_gt": float(np.mean([d["gt_count"] for d in per_image_details])),
        },
        "per_bin": {},
        "per_dominant_class": {},
        "per_image": per_image_details[:20],  # first 20 for inspection
    }

    for bin_name, errors in sorted(per_bin.items()):
        e = np.array(errors)
        results["per_bin"][bin_name] = {
            "mae": float(np.mean(e)), "n_images": len(e),
        }

    # Only show classes with >2 images
    for cls_name, errors in sorted(per_dominant_class.items(), key=lambda x: -len(x[1])):
        if len(errors) >= 2:
            e = np.array(errors)
            results["per_dominant_class"][cls_name] = {
                "mae": float(np.mean(e)), "n_images": len(e),
            }

    return results


# =========================================================================== #
# Main
# =========================================================================== #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="all", choices=["p25", "p22", "all"])
    ap.add_argument("--fsc147-cache", default="/home/czp/ws_yiyang/ovcud_cache/fsc147_test_pts32_sample")
    ap.add_argument("--fsc147-ann", default="/home/czp/official_code/dataset/FSC147/annotation_FSC147_384.json")
    ap.add_argument("--coco-cache", default="/home/czp/ws_yiyang/ovcud_cache/coco_val_3view")
    ap.add_argument("--output", default="result/logs/p2_experiments.json")
    ap.add_argument("--category-ckpt", default="result/checkpoints/category_cosine_pts32.pt")
    ap.add_argument("--relation-ckpt", default="result/checkpoints/fsc147_relation_pts32_exp5c.pt")
    ap.add_argument("--text-prototypes", default="result/checkpoints/text_prototypes_fsc147.pt")
    ap.add_argument("--coco-category-ckpt", default="result/checkpoints/category_coco80_cosine.pt")
    ap.add_argument("--coco-text-prototypes", default="result/checkpoints/text_prototypes_coco80.pt")
    ap.add_argument("--tau-inst", type=float, default=0.99)
    ap.add_argument("--tau-aff", type=float, default=0.1)
    ap.add_argument("--conf-threshold", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    print("=" * 70)
    print(f"P2 Experiments: {args.experiment}")
    print(f"tau_inst={args.tau_inst}, tau_aff={args.tau_aff}, conf_thresh={args.conf_threshold}")
    print("=" * 70)

    # Load models
    print("\n[load] Loading models...")
    cat_head = load_category_head(args.category_ckpt, device)
    rel_head = load_relation_head(args.relation_ckpt, device) if os.path.exists(args.relation_ckpt) else None
    tp = F.normalize(torch.load(args.text_prototypes, map_location="cpu", weights_only=False).float(), dim=-1)

    # Load class names
    import json as _json
    proto_dir = os.path.dirname(args.text_prototypes)
    cat_file = os.path.join(proto_dir, "text_prototypes_fsc147_categories.json")
    class_names = []
    if os.path.exists(cat_file):
        with open(cat_file) as f:
            cats = _json.load(f)
        class_names = [c["name"] for c in cats.get("categories", [])]
    print(f"[load] {len(class_names)} class names loaded")

    all_results = {}

    # ---- P2-5: Fully Prompt-Free ----
    if args.experiment in ("p25", "all"):
        print(f"\n{'='*70}")
        print("P2-5: Fully Prompt-Free Single-Count Protocol")
        print("=" * 70)
        p25_results = run_p25(args, cat_head, rel_head, tp, device, class_names)
        all_results["p25_prompt_free"] = p25_results

        print("\n--- P2-5 Results ---")
        print(f"{'Protocol':<30} {'MAE':>8} {'RMSE':>8} {'n':>6}")
        print("-" * 55)
        for key, res in p25_results.items():
            label = key.replace("prompt_free_", "prompt-free: ").replace("_", " ")
            print(f"{label:<30} {res['mae']:>8.2f} {res['rmse']:>8.2f} {res['n_images']:>6}")

    # ---- P2-2: COCO Multi-Category ----
    if args.experiment in ("p22", "all"):
        print(f"\n{'='*70}")
        print("P2-2: COCO-count Multi-Category Evaluation")
        print("=" * 70)

        # Use COCO-trained head and COCO text prototypes for COCO evaluation
        coco_cat_ckpt = args.coco_category_ckpt or args.category_ckpt
        coco_tp_path = args.coco_text_prototypes or args.text_prototypes

        if os.path.exists(coco_cat_ckpt) and os.path.exists(coco_tp_path):
            print(f"[load] COCO head: {coco_cat_ckpt}")
            print(f"[load] COCO prototypes: {coco_tp_path}")
            coco_cat_head = load_category_head(coco_cat_ckpt, device)
            coco_tp = F.normalize(torch.load(coco_tp_path, map_location="cpu", weights_only=False).float(), dim=-1)

            # Load COCO class names (standard 80 COCO categories)
            coco_class_names = [
                'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
                'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog',
                'horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella',
                'handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite',
                'baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle',
                'wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich',
                'orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
                'potted plant','bed','dining table','toilet','tv','laptop','mouse','remote',
                'keyboard','cell phone','microwave','oven','toaster','sink','refrigerator',
                'book','clock','vase','scissors','teddy bear','hair drier','toothbrush',
            ]
            print(f"[load] Using {len(coco_class_names)} COCO class names")

            p22_results = run_p22(args, coco_cat_head, rel_head, coco_tp, device, coco_class_names)
            all_results["p22_coco_count"] = p22_results

            print("\n--- P2-2 Overall ---")
            print(f"MAE={p22_results['overall']['mae']:.2f}  "
                  f"RMSE={p22_results['overall']['rmse']:.2f}  "
                  f"n={p22_results['overall']['n_images']}  "
                  f"mean_GT={p22_results['overall']['mean_gt']:.1f}")

            print("\n--- P2-2 Per-Bin ---")
            for bin_name, res in sorted(p22_results["per_bin"].items()):
                print(f"  {bin_name}: MAE={res['mae']:.2f} (n={res['n_images']})")

            print("\n--- P2-2 Per Dominant Class (n>=2) ---")
            print(f"{'Class':<25} {'MAE':>8} {'n':>6}")
            print("-" * 42)
            for cls_name, res in sorted(p22_results["per_dominant_class"].items(),
                                         key=lambda x: -x[1]["n_images"])[:15]:
                print(f"{cls_name:<25} {res['mae']:>8.2f} {res['n_images']:>6}")
        else:
            print("[P2-2] COCO head/prototypes not found. Using FSC147 head (expect poor results).")
            p22_results = run_p22(args, cat_head, rel_head, tp, device, class_names)
            all_results["p22_coco_count"] = p22_results

    # Save
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
