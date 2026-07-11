"""P1-2: Classification Head Ablation — Minimal Counting Pipeline.

Runs the counting pipeline with different classification heads on CARPK and FSC147.
Reuses the core logic from run_p1_ablations.py with head-type dispatch.
"""

from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from code.clustering.first_neighbor import (
    first_neighbor_clustering, category_aware_clustering_with_spatial,
)
from code.counting.deduplicate import (
    build_same_instance_components, build_same_instance_components_adaptive,
)


def load_cosine_head(ckpt_path, device):
    from script.train_category_v2 import CosineCategoryHead
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    head = CosineCategoryHead(in_dim=ck["in_dim"], proj_dim=ck["proj_dim"],
                              dropout=0.3, num_layers=2)
    head.load_state_dict(ck["head"])
    head.to(device).eval()
    return head

def load_linear_head(ckpt_path, device):
    from script.train_category_linear import LinearPrototypeHead
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    head = LinearPrototypeHead(in_dim=ck["in_dim"], num_classes=ck["num_classes"],
                               proj_dim=ck["proj_dim"], dropout=0.3, num_layers=2)
    head.load_state_dict(ck["head"])
    head.to(device).eval()
    return head

def load_relation_head(ckpt_path, device):
    from code.heads.relation_head import PairwiseRelationHead, RelationHeadConfig
    from code.matrix.pairwise_features import pairwise_feature_dim
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    feat_dim = pairwise_feature_dim(ck["z_dim"])
    cfg = RelationHeadConfig(feat_dim=feat_dim, hidden_dim=ck.get("hidden_dim", 512),
                             num_layers=ck.get("num_layers", 3), dropout=0.1)
    head = PairwiseRelationHead(cfg)
    head.load_state_dict(ck["relation_head"])
    head.to(device).eval()
    return head


def get_category_probs(head, z, tp, device, head_type):
    """Dispatch category probability computation by head type."""
    with torch.no_grad():
        if head_type == "cosine":
            logits = head(z.to(device), tp.to(device))
        elif head_type == "linear":
            logits = head(z.to(device))  # Linear head ignores text proto
        elif head_type == "raw_cosine":
            z_proj = F.normalize(z[:, :tp.shape[1]].to(device), dim=-1)
            logits = z_proj @ tp.to(device).t() / 0.07
        else:
            raise ValueError(f"Unknown head_type: {head_type}")
    return torch.softmax(logits, dim=-1).cpu().numpy()


def count_one_image(d, head, tp, rel_head, device, head_type, args):
    """Run counting pipeline on a single image. Returns predicted count."""
    z = d["z"].float()
    bbox = d["bbox"].float()
    n_cand = z.shape[0]
    if n_cand == 0:
        return 0

    # 1. Category probabilities
    cat_probs = get_category_probs(head, z, tp, device, head_type)
    top_conf = cat_probs.max(axis=1)

    # Filter by confidence + validity
    valid_orig = np.asarray(d.get("valid", np.ones(n_cand))) > 0
    conf_valid = top_conf >= args.conf_threshold
    effective = valid_orig & conf_valid
    valid_idx = np.where(effective)[0]
    N_v = len(valid_idx)
    if N_v <= 1:
        return N_v

    # 2. Build A_sem from category compatibility
    A_sem = np.eye(n_cand, dtype=np.float32)
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            dot = float((cat_probs[i] * cat_probs[j]).sum())
            A_sem[i, j] = A_sem[j, i] = dot * 10.0

    # 3. Compute A_inst from relation head
    from code.matrix.pairwise_features import build_pairwise_features, box_geometry
    max_cand = min(n_cand, 200)
    if n_cand > max_cand:
        keep = np.argsort(-top_conf)[:max_cand]
        z_sub = z[keep]; p_sub = torch.from_numpy(cat_probs[keep]).float()
        b_sub = bbox[keep]
    else:
        z_sub = z; p_sub = torch.from_numpy(cat_probs).float()
        b_sub = bbox

    ii, jj = np.triu_indices(len(p_sub), k=1)
    A_inst = np.zeros((n_cand, n_cand), dtype=np.float32)
    if len(ii) > 0:
        zi = z_sub[ii].to(device); zj = z_sub[jj].to(device)
        pi = p_sub[ii].to(device); pj = p_sub[jj].to(device)
        bi = b_sub[ii].to(device); bj = b_sub[jj].to(device)
        geom_ij = box_geometry(bi, bj).to(device)
        geom_ji = box_geometry(bj, bi).to(device)
        phi_ij = build_pairwise_features(zi, zj, pi, pj, bi, bj, geom=geom_ij).to(device)
        phi_ji = build_pairwise_features(zj, zi, pj, pi, bj, bi, geom=geom_ji).to(device)

        with torch.no_grad():
            A_inst_ij = rel_head(phi_ij)["inst"].cpu().numpy().squeeze()
            A_inst_ji = rel_head(phi_ji)["inst"].cpu().numpy().squeeze()

        if n_cand > max_cand:
            for idx, (i, j) in enumerate(zip(ii, jj)):
                A_inst[i, j] = A_inst_ij[idx]
                A_inst[j, i] = A_inst_ji[idx]
        else:
            A_inst[ii, jj] = A_inst_ij
            A_inst[jj, ii] = A_inst_ji

    # 4. Clustering (G1: cat-bucket + connected)
    image_area = float(d.get("height", 640) * d.get("width", 640))
    sub_probs = cat_probs[valid_idx]
    A_sem_sub = A_sem[np.ix_(valid_idx, valid_idx)]
    bbox_np = bbox.numpy()

    groups = category_aware_clustering_with_spatial(
        sub_probs, A_sem_sub, bbox_np[valid_idx],
        image_area=image_area,
        tau_affinity=args.tau_aff,
        use_bucketing=True,
    )

    # 5. Dedup per group + count
    pred_count = 0
    for group_indices in groups:
        if len(group_indices) == 0:
            continue
        if len(group_indices) == 1:
            pred_count += 1
            continue
        A_sub = A_inst[np.ix_(group_indices, group_indices)]
        if len(group_indices) > 20:
            comps = build_same_instance_components_adaptive(
                list(range(len(group_indices))), A_sub, base_tau=args.tau_inst, use_greedy=True)
        else:
            comps = build_same_instance_components(
                list(range(len(group_indices))), A_sub, tau_inst=args.tau_inst)
        pred_count += len(comps)

    return pred_count


def run_dataset(cache_dir, head, tp, rel_head, device, head_type, args, max_imgs=None):
    """Run counting pipeline over all images in a cache directory."""
    files = sorted(Path(cache_dir).glob("*.pt"))
    if max_imgs:
        files = files[:max_imgs]

    errors = []
    preds = []
    gts = []
    t0 = time.time()

    for fi, f in enumerate(files):
        d = torch.load(f, map_location="cpu", weights_only=False)
        gt = int(d.get("gt_count", 0))
        if gt <= 0:
            continue

        pred = count_one_image(d, head, tp, rel_head, device, head_type, args)
        errors.append(abs(pred - gt))
        preds.append(pred)
        gts.append(gt)

        if (fi + 1) % 50 == 0:
            mae = np.mean(errors)
            elapsed = time.time() - t0
            rate = (fi + 1) / max(elapsed, 0.01)
            print(f"  [{fi+1}/{len(files)}] MAE={mae:.2f}  rate={rate:.1f}/s")

    errors = np.array(errors); preds = np.array(preds); gts = np.array(gts)
    return {
        "mae": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "bias": float(np.mean(preds - gts)),
        "n_images": len(errors),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fsc147-cache", default="/home/czp/ws_yiyang/ovcud_cache/fsc147_test_fast")
    ap.add_argument("--carpk-cache", default="/home/czp/ws_yiyang/ovcud_cache/carpk_test")
    ap.add_argument("--output", default="result/logs/p1_classification_ablation.json")
    ap.add_argument("--fsc147-ckpt", default="result/checkpoints/category_cosine_pts32.pt")
    ap.add_argument("--coco-ckpt", default="result/checkpoints/category_coco80_cosine.pt")
    ap.add_argument("--linear-ckpt", default="result/checkpoints/category_fsc147_linear.pt")
    ap.add_argument("--fsc147-protos", default="result/checkpoints/text_prototypes_fsc147.pt")
    ap.add_argument("--coco-protos", default="result/checkpoints/text_prototypes_coco80.pt")
    ap.add_argument("--relation-ckpt", default="result/checkpoints/fsc147_relation_pts32_exp5c.pt")
    ap.add_argument("--tau-inst", type=float, default=0.99)
    ap.add_argument("--tau-aff", type=float, default=0.1)
    ap.add_argument("--conf-threshold", type=float, default=0.2)
    ap.add_argument("--n-fsc147", type=int, default=100, help="Max FSC147 images")
    ap.add_argument("--n-carpk", type=int, default=0, help="Max CARPK images (0=all)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    print("=" * 70)
    print("P1-2: Classification Head Ablation — Counting Pipeline")
    print("=" * 70)
    print(f"tau_inst={args.tau_inst}, tau_aff={args.tau_aff}, conf_thresh={args.conf_threshold}")

    # Load shared resources
    print("\n[load] Loading models...")
    rel_head = load_relation_head(args.relation_ckpt, device)
    tp_fsc147 = F.normalize(torch.load(args.fsc147_protos, map_location="cpu", weights_only=False).float(), dim=-1)

    # Define head variants to test
    variants = {}

    # C4: FSC147 Cosine + FSC147 prototypes (baseline)
    head_fsc147 = load_cosine_head(args.fsc147_ckpt, device)
    variants["C4_fsc147_cosine+fsc147_proto"] = (head_fsc147, tp_fsc147, "cosine")

    # C6: FSC147 Linear (closed-set)
    head_linear = load_linear_head(args.linear_ckpt, device)
    variants["C6_fsc147_linear"] = (head_linear, None, "linear")

    # C1: COCO Cosine + FSC147 prototypes (general-domain proj)
    head_coco = load_cosine_head(args.coco_ckpt, device)
    variants["C1_coco_cosine+fsc147_proto"] = (head_coco, tp_fsc147, "cosine")

    # C1b: COCO Cosine + COCO prototypes (only for CARPK, where "car" exists in COCO)
    tp_coco = F.normalize(torch.load(args.coco_protos, map_location="cpu", weights_only=False).float(), dim=-1)
    variants["C1b_coco_cosine+coco_proto"] = (head_coco, tp_coco, "cosine")

    results = {}

    # ---- FSC147 Evaluation ----
    fsc147_results = {}
    if args.n_fsc147 > 0:
        print(f"\n{'='*70}")
        print(f"FSC147 Test (max {args.n_fsc147} images)")
        print("=" * 70)
        for vname, (head, tp, htype) in variants.items():
            # Skip C1b for FSC147 (COCO prototypes don't cover FSC147 classes)
            if vname == "C1b_coco_cosine+coco_proto":
                continue
            # Skip C1 for FSC147 (known to fail - 0.1% accuracy)
            if vname == "C1_coco_cosine+fsc147_proto":
                print(f"\n{vname}: SKIPPED (classification accuracy 0.1%, will not work)")
                fsc147_results[vname] = {"mae": float("nan"), "rmse": float("nan"), "bias": float("nan"),
                                          "n_images": 0, "note": "classification fails (0.1% top1)"}
                continue

            print(f"\n{vname}...")
            res = run_dataset(args.fsc147_cache, head, tp, rel_head, device, htype, args,
                            max_imgs=args.n_fsc147)
            fsc147_results[vname] = res
            print(f"  MAE={res['mae']:.2f}  RMSE={res['rmse']:.2f}  bias={res['bias']:+.2f}  n={res['n_images']}")
    results["fsc147"] = fsc147_results

    # ---- CARPK Evaluation ----
    carpk_n = args.n_carpk if args.n_carpk > 0 else None
    print(f"\n{'='*70}")
    print(f"CARPK Test ({carpk_n or 'all'} images)")
    print("=" * 70)
    carpk_results = {}
    carpk_conf = 0.1  # Lower threshold for CARPK

    for vname, (head, tp, htype) in variants.items():
        if vname == "C1_coco_cosine+fsc147_proto":
            print(f"\n{vname}: SKIPPED (classification fails, 0 car hits)")
            carpk_results[vname] = {"mae": float("nan"), "rmse": float("nan"), "bias": float("nan"),
                                     "n_images": 0, "note": "classification fails (0 car hits)"}
            continue

        print(f"\n{vname}...")
        # Temporarily override conf_threshold for CARPK
        save_conf = args.conf_threshold
        args.conf_threshold = carpk_conf
        res = run_dataset(args.carpk_cache, head, tp, rel_head, device, htype, args,
                        max_imgs=carpk_n)
        args.conf_threshold = save_conf
        carpk_results[vname] = res
        print(f"  MAE={res['mae']:.2f}  RMSE={res['rmse']:.2f}  bias={res['bias']:+.2f}  n={res['n_images']}")
    results["carpk"] = carpk_results

    # ---- Summary ----
    print(f"\n{'='*70}")
    print("P1-2 Results Summary")
    print("=" * 70)

    for dataset, ds_results in [("FSC147", results.get("fsc147", {})), ("CARPK", results.get("carpk", {}))]:
        if not ds_results:
            continue
        print(f"\n{dataset}:")
        print(f"{'Variant':<40} {'MAE':>8} {'RMSE':>8} {'bias':>8} {'n':>6}")
        print("-" * 70)
        for vname, res in ds_results.items():
            mae_str = f"{res['mae']:.2f}" if not np.isnan(res['mae']) else "N/A"
            rmse_str = f"{res['rmse']:.2f}" if not np.isnan(res['rmse']) else "N/A"
            bias_str = f"{res['bias']:8.2f}" if not np.isnan(res['bias']) else "N/A"
            note = res.get("note", "")
            print(f"{vname:<40} {mae_str:>8} {rmse_str:>8} {bias_str:>8} {res['n_images']:>6}  {note}")

    # Save
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
