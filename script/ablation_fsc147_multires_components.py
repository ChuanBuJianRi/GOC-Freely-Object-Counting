"""FSC147 full-test component ablations for the 12.74 multi-resolution result.

This script evaluates the final P2 Extended cache protocol used by
result/logs/fsc147_multires_extended.json and keeps the learning weights fixed:

  - fast/base cache head: result/checkpoints/category_cosine_fast.pt
  - fast/base relation: result/checkpoints/fsc147_relation_best.pt
  - multires cache head: result/checkpoints/category_cosine_pts32.pt
  - multires relation: result/checkpoints/fsc147_relation_pts32_best.pt
  - text prototypes: result/checkpoints/text_prototypes_fsc147.pt

Variants:
  A1: category confidence filter only, no dedup
  A2: class-bucket IoU NMS@0.5, no relation head
  A3: relation dedup with one global group, no semantic grouping
  A4: semantic/category grouping + relation dedup, no spatial refinement
  A5: semantic/category grouping + spatial refinement, fixed relation dedup
  A8: final full pipeline, adaptive large-group relation dedup

High-density frontend ablations:
  A6: remove 51-100 multi-resolution cache, keep 100+ multi-resolution cache
  A7: remove 100+ multi-resolution cache, keep 51-100 multi-resolution cache

Oracle diagnostics on the final cache:
  O1: oracle category for candidates covering at least one GT dot
  O2: predicted grouping + oracle dot-sharing dedup
  O3: proposal cover upper bound, count unique GT dots covered by any candidate

The multi-resolution cache itself stores placeholder masks, so O2/O3 reconstruct
candidate masks from the source fast/tiled caches used to build each final file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mask_utils

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from code.clustering.first_neighbor import (  # noqa: E402
    category_aware_clustering_with_spatial,
    first_neighbor_clustering,
)
from code.counting.deduplicate import (  # noqa: E402
    build_same_instance_components,
    build_same_instance_components_adaptive,
)
from code.counting.representative import select_representatives  # noqa: E402
from code.matrix.pairwise_features import build_pairwise_features, box_geometry  # noqa: E402
from script.eval_carpk import load_category_head, load_relation_head, get_category_probs  # noqa: E402
from script.build_multires_cache import bbox_iou_matrix, greedy_nms  # noqa: E402


FSC_DIR = Path("/home/czp/official_code/dataset/FSC147")
CACHE_ROOT = Path("/home/czp/ws_yiyang/ovcud_cache")
DEFAULT_ANN = FSC_DIR / "annotation_FSC147_384.json"
DEFAULT_SPLIT = FSC_DIR / "Train_Test_Val_FSC_147.json"

BINS = [
    ("0-10", 0, 10),
    ("11-20", 11, 20),
    ("21-50", 21, 50),
    ("51-100", 51, 100),
    ("100+", 101, 10**9),
]

TAU_INST = 0.99
TAU_AFF = 0.1
CONF = 0.2
MAX_GROUP_SIZE = 30


def metrics(preds, gts):
    err = np.asarray(preds, dtype=float) - np.asarray(gts, dtype=float)
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
    }


def perbin(rows, key):
    out = []
    for lab, lo, hi in BINS:
        rs = [r for r in rows if lo <= r["gt_count"] <= hi]
        if not rs:
            continue
        out.append({
            "bin": lab,
            "n_images": len(rs),
            **metrics([r[key] for r in rs], [r["gt_count"] for r in rs]),
        })
    return out


def load_test_names(split_file: Path) -> list[str]:
    split = json.loads(split_file.read_text())
    return sorted(split["test"])


def relation_matrices(d, z, probs, top_conf, relation_head, device):
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
        z_t[ii], z_t[jj], p_t[ii], p_t[jj], bi, bj, geom=box_geometry(bi, bj)
    )
    phi_ji = build_pairwise_features(
        z_t[jj], z_t[ii], p_t[jj], p_t[ii], bj, bi, geom=box_geometry(bj, bi)
    )
    with torch.no_grad():
        out = relation_head(torch.cat([phi_ij.to(device), phi_ji.to(device)], dim=0))
        p = len(ii)
        logits = 0.5 * (out["inst"][:p] + out["inst"][p:]).cpu().numpy()
        logits = np.clip(logits, -20, 20)

    for k in range(len(ii)):
        i, j = (keep[ii[k]], keep[jj[k]]) if keep is not None else (ii[k], jj[k])
        A_inst[i, j] = A_inst[j, i] = logits[k]
    return A_inst


def semantic_affinity(probs):
    A_sem = (probs @ probs.T).astype(np.float32) * 10.0
    np.fill_diagonal(A_sem, 1.0)
    return A_sem


def valid_indices(d, top_conf, conf=CONF):
    valid_orig = np.asarray(d["valid"]) > 0
    return [i for i in range(len(top_conf)) if valid_orig[i] and top_conf[i] >= conf]


def nms_count(idxs, bbox_np, top_conf, top_class, iou_thr=0.5):
    kept_total = 0
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
        kept_total += len(kept)
    return kept_total


def cluster_no_refine(probs, A_sem, vidx):
    if len(vidx) == 0:
        return []
    sub_probs = probs[vidx]
    sub_A = A_sem[vidx][:, vidx]
    cat_compat = sub_probs @ sub_probs.T
    sem_prob = 1.0 / (1.0 + np.exp(-sub_A))
    A_group = cat_compat * sem_prob
    return first_neighbor_clustering(A_group, TAU_AFF, sub_probs.argmax(axis=1))


def cluster_full(probs, A_sem, bbox_np, vidx, image_area):
    if len(vidx) == 0:
        return []
    return category_aware_clustering_with_spatial(
        probs[vidx],
        A_sem[vidx][:, vidx],
        bbox_np[vidx],
        image_area,
        tau_affinity=TAU_AFF,
        max_group_size=MAX_GROUP_SIZE,
        use_bucketing=True,
    )


def dedup_count(groups_global, A_inst, probs, bbox_np, image_area, n_cand, adaptive=True):
    total = 0
    zeros_part = np.zeros((n_cand, n_cand), dtype=np.float32)
    for gg in groups_global:
        if len(gg) == 0:
            continue
        if len(gg) == 1:
            total += 1
            continue
        sub_A = A_inst[gg][:, gg]
        if adaptive and len(gg) > 20:
            comps = build_same_instance_components_adaptive(
                list(range(len(gg))), sub_A, base_tau=TAU_INST, use_greedy=True
            )
        else:
            comps = build_same_instance_components(list(range(len(gg))), sub_A, tau_inst=TAU_INST)
        reps = select_representatives(
            [[gg[i] for i in comp] for comp in comps],
            zeros_part,
            probs,
            bbox_np,
            image_area,
            min_category_conf=0.05,
        )
        total += len(reps)
    return total


def oracle_components(gg, cover_sets):
    parent = {g: g for g in gg}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    dot_owner = {}
    for g in gg:
        for dot in cover_sets[g]:
            if dot in dot_owner:
                ra, rb = find(dot_owner[dot]), find(g)
                if ra != rb:
                    parent[rb] = ra
            else:
                dot_owner[dot] = g

    comps = {}
    for g in gg:
        comps.setdefault(find(g), []).append(g)
    return list(comps.values())


def oracle_dedup_count(groups_global, cover_sets, probs, bbox_np, image_area, n_cand):
    total = 0
    zeros_part = np.zeros((n_cand, n_cand), dtype=np.float32)
    for gg in groups_global:
        if not gg:
            continue
        comps = oracle_components(gg, cover_sets)
        reps = select_representatives(
            comps,
            zeros_part,
            probs,
            bbox_np,
            image_area,
            min_category_conf=0.05,
        )
        total += len(reps)
    return total


def decode_cover_sets_from_masks(d, points):
    h, w = int(d["height"]), int(d["width"])
    pts = []
    for x, y in points:
        xi, yi = int(round(float(x))), int(round(float(y)))
        if 0 <= xi < w and 0 <= yi < h:
            pts.append((xi, yi))

    cover = []
    for rle in d.get("masks_rle", []):
        try:
            m = mask_utils.decode(rle)
        except Exception:
            cover.append(set())
            continue
        s = set()
        for k, (x, y) in enumerate(pts):
            if m[y, x]:
                s.add(k)
        cover.append(s)
    return cover


def source_cover_sets(stem, final_d, ann, dirs, cache_source=""):
    """Reconstruct final-cache candidate dot coverage from source caches."""
    points = ann.get(f"{stem}.jpg", {}).get("points", [])
    if str(cache_source).startswith("tiled"):
        return decode_cover_sets_from_masks(final_d, points)

    fast_path = dirs["fast"] / f"{stem}.pt"
    mr100_path = dirs["mr100"] / f"{stem}.pt"
    mr51_path = dirs["mr51"] / f"{stem}.pt"

    if mr100_path.exists():
        source_b = dirs["tiled100"] / f"{stem}.pt"
    elif mr51_path.exists():
        source_b = dirs["tiled51"] / f"{stem}.pt"
    else:
        d = torch.load(fast_path, map_location="cpu", weights_only=False)
        cover = decode_cover_sets_from_masks(d, points)
        return cover[: final_d["z"].shape[0]]

    if not fast_path.exists() or not source_b.exists():
        return [set() for _ in range(final_d["z"].shape[0])]

    d16 = torch.load(fast_path, map_location="cpu", weights_only=False)
    d32 = torch.load(source_b, map_location="cpu", weights_only=False)
    bbox16 = d16["bbox"].float().numpy()
    bbox32 = d32["bbox"].float().numpy()
    bbox_all = np.concatenate([bbox16, bbox32], axis=0)
    areas = bbox_all[:, 2] * bbox_all[:, 3]
    if len(areas) > 1:
        keep = greedy_nms(bbox_iou_matrix(bbox_all), areas, thresh=0.5)
    else:
        keep = [0]

    cover_all = decode_cover_sets_from_masks(d16, points) + decode_cover_sets_from_masks(d32, points)
    cover = [cover_all[i] for i in keep]

    if len(cover) != final_d["z"].shape[0]:
        return match_cover_by_bbox(final_d, bbox_all, cover_all)
    return cover


def match_cover_by_bbox(final_d, source_bbox, source_cover):
    final_bbox = final_d["bbox"].float().numpy()
    cover = []
    used = set()
    for fb in final_bbox:
        diffs = np.abs(source_bbox - fb[None, :]).sum(axis=1)
        order = np.argsort(diffs)
        chosen = None
        for j in order[:5]:
            if j not in used and diffs[j] < 1e-4:
                chosen = int(j)
                break
        if chosen is None:
            chosen = int(order[0])
        used.add(chosen)
        cover.append(source_cover[chosen])
    return cover


def gt_count_from_ann(stem, ann):
    entry = ann.get(f"{stem}.jpg", {})
    if "gt_count" in entry:
        return int(entry["gt_count"])
    return len(entry.get("points", []))


def empty_prediction_row(stem, ann):
    gt_count = gt_count_from_ann(stem, ann)
    return {
        "gt_count": gt_count,
        "n_candidates": 0,
        "A1_filter_only": 0,
        "A2_iou_nms05": 0,
        "A3_global_relation": 0,
        "A4_group_no_spatial": 0,
        "A5_no_adaptive_dedup": 0,
        "A8_full": 0,
        "O1_oracle_category": 0,
        "O2_oracle_dedup": 0,
        "O3_proposal_cover": 0,
    }


def evaluate_one(d, category_head, relation_head, text_prototypes, device, cover_sets=None):
    h, w = int(d["height"]), int(d["width"])
    image_area = float(h * w)
    gt_count = int(d.get("gt_count", 0))
    z = d["z"].float()
    bbox_np = d["bbox"].float().numpy()
    n_cand = z.shape[0]
    empty = {
        "gt_count": gt_count,
        "n_candidates": n_cand,
        "A1_filter_only": 0,
        "A2_iou_nms05": 0,
        "A3_global_relation": 0,
        "A4_group_no_spatial": 0,
        "A5_no_adaptive_dedup": 0,
        "A8_full": 0,
        "O1_oracle_category": 0,
        "O2_oracle_dedup": 0,
        "O3_proposal_cover": 0,
    }
    if n_cand == 0:
        return empty

    probs = get_category_probs(category_head, z, text_prototypes, device)
    top_conf = probs.max(axis=1)
    top_class = probs.argmax(axis=1)
    vidx = valid_indices(d, top_conf)
    n_filtered = int(((np.asarray(d["valid"]) > 0) & (top_conf < CONF)).sum())
    row = dict(empty)
    row["n_conf_filtered"] = n_filtered
    row["A1_filter_only"] = len(vidx)
    if len(vidx) == 0:
        return row

    A_sem = semantic_affinity(probs)
    A_inst = relation_matrices(d, z, probs, top_conf, relation_head, device)

    row["A2_iou_nms05"] = nms_count(vidx, bbox_np, top_conf, top_class, iou_thr=0.5)
    row["A3_global_relation"] = dedup_count(
        [vidx], A_inst, probs, bbox_np, image_area, n_cand, adaptive=True
    )

    groups_no_refine = cluster_no_refine(probs, A_sem, vidx)
    gg_no_refine = [[vidx[i] for i in g] for g in groups_no_refine]
    row["A4_group_no_spatial"] = dedup_count(
        gg_no_refine, A_inst, probs, bbox_np, image_area, n_cand, adaptive=True
    )

    groups = cluster_full(probs, A_sem, bbox_np, vidx, image_area)
    gg = [[vidx[i] for i in g] for g in groups]
    row["A5_no_adaptive_dedup"] = dedup_count(
        gg, A_inst, probs, bbox_np, image_area, n_cand, adaptive=False
    )
    row["A8_full"] = dedup_count(
        gg, A_inst, probs, bbox_np, image_area, n_cand, adaptive=True
    )
    row["n_groups_A8"] = len(groups)

    if cover_sets is None:
        return row

    covered = [i for i, s in enumerate(cover_sets) if s]
    covered_dots = set().union(*[cover_sets[i] for i in covered]) if covered else set()
    row["O3_proposal_cover"] = len(covered_dots)

    if covered:
        n_classes = probs.shape[1]
        oprobs = np.zeros((n_cand, n_classes), dtype=np.float32)
        matched_class = np.asarray(d["matched_class"])
        for i in covered:
            c = int(matched_class[i])
            if 0 <= c < n_classes:
                oprobs[i, c] = 1.0
        oA_sem = semantic_affinity(oprobs)
        oA_inst = relation_matrices(d, z, oprobs, oprobs.max(axis=1), relation_head, device)
        ogroups = cluster_full(oprobs, oA_sem, bbox_np, covered, image_area)
        ogg = [[covered[i] for i in g] for g in ogroups]
        row["O1_oracle_category"] = dedup_count(
            ogg, oA_inst, oprobs, bbox_np, image_area, n_cand, adaptive=True
        )
        row["O2_oracle_dedup"] = oracle_dedup_count(
            gg, cover_sets, probs, bbox_np, image_area, n_cand
        )
    return row


def load_cache_for_scenario(stem, scenario, dirs):
    file_name = f"{stem}.pt"
    if scenario == "final_multires":
        p = dirs["final"] / file_name
        if p.exists():
            if (dirs["mr100"] / file_name).exists():
                cache_source = "final_mr100"
            elif (dirs["mr51"] / file_name).exists():
                cache_source = "final_mr51"
            else:
                cache_source = "final_fast"
            return torch.load(p, map_location="cpu", weights_only=False), p, cache_source
        fallback = dirs["tiled100"] / file_name
        if fallback.exists():
            return (
                torch.load(fallback, map_location="cpu", weights_only=False),
                fallback,
                "tiled100_fallback_missing_final",
            )
    elif scenario == "no_51_100_multires":
        p = dirs["mr100"] / file_name
        if p.exists():
            return torch.load(p, map_location="cpu", weights_only=False), p, "mr100"
        fallback = dirs["tiled100"] / file_name
        if fallback.exists():
            return (
                torch.load(fallback, map_location="cpu", weights_only=False),
                fallback,
                "tiled100_fallback_missing_mr100",
            )
        p = dirs["fast"] / file_name
        if p.exists():
            return torch.load(p, map_location="cpu", weights_only=False), p, "fast"
    elif scenario == "no_100plus_multires":
        p = dirs["mr51"] / file_name
        if p.exists():
            return torch.load(p, map_location="cpu", weights_only=False), p, "mr51"
        p = dirs["fast"] / file_name
        if p.exists():
            return torch.load(p, map_location="cpu", weights_only=False), p, "fast"
    else:
        raise ValueError(scenario)
    return None, p, "missing_cache_zero_candidate"


def aggregate_rows(rows, keys):
    gts = [r["gt_count"] for r in rows]
    return {
        key: {
            "overall": metrics([r[key] for r in rows], gts),
            "perbin": perbin(rows, key),
        }
        for key in keys
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="result/logs/fsc147_multires_component_ablation.json")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--skip-oracle", action="store_true")
    args = ap.parse_args()

    dirs = {
        "final": CACHE_ROOT / "fsc147_test_multires_all",
        "fast": CACHE_ROOT / "fsc147_test_fast",
        "mr100": CACHE_ROOT / "fsc147_test_multires",
        "mr51": CACHE_ROOT / "fsc147_test_multires_51_100",
        "tiled100": CACHE_ROOT / "fsc147_test_tiled",
        "tiled51": CACHE_ROOT / "fsc147_test_tiled_51_100",
    }
    names = load_test_names(DEFAULT_SPLIT)
    if args.limit > 0:
        names = names[: args.limit]
    ann = json.loads(DEFAULT_ANN.read_text())

    print("[load] models")
    category_fast = load_category_head(REPO / "result/checkpoints/category_cosine_fast.pt", args.device)
    relation_fast = load_relation_head(
        REPO / "result/checkpoints/fsc147_relation_best.pt", args.device
    )
    category_pts32 = load_category_head(REPO / "result/checkpoints/category_cosine_pts32.pt", args.device)
    relation_pts32 = load_relation_head(
        REPO / "result/checkpoints/fsc147_relation_pts32_best.pt", args.device
    )
    text_prototypes = torch.nn.functional.normalize(
        torch.load(
            REPO / "result/checkpoints/text_prototypes_fsc147.pt",
            map_location="cpu",
            weights_only=False,
        ).float(),
        dim=-1,
    ).to(args.device)

    out = {
        "config": {
            "dataset": "FSC147 test",
            "n_test_names": len(names),
            "tau_inst": TAU_INST,
            "tau_affinity": TAU_AFF,
            "conf_threshold": CONF,
            "category_fast_ckpt": "result/checkpoints/category_cosine_fast.pt",
            "relation_fast_ckpt": "result/checkpoints/fsc147_relation_best.pt",
            "category_pts32_ckpt": "result/checkpoints/category_cosine_pts32.pt",
            "relation_pts32_ckpt": "result/checkpoints/fsc147_relation_pts32_best.pt",
            "text_prototypes": "result/checkpoints/text_prototypes_fsc147.pt",
            "cache_final": str(dirs["final"]),
            "cache_no_51_100": f"{dirs['mr100']} over {dirs['fast']}",
            "cache_no_100plus": f"{dirs['mr51']} over {dirs['fast']}",
            "fallback_policy": (
                "If a FSC147 test image is missing from the merged multires cache but has "
                "a 100+ tiled cache, evaluate that tiled cache with the pts32 heads. If "
                "the scenario intentionally removes that frontend and no fast cache exists, "
                "include the image as a zero-candidate prediction instead of skipping it."
            ),
            "note": "A8 1189-image anchor is result/logs/fsc147_multires_extended.json.",
        }
    }

    def model_for_cache(cache_source):
        use_pts32 = (
            cache_source in {"final_mr100", "final_mr51", "mr100", "mr51"}
            or str(cache_source).startswith("tiled")
        )
        if use_pts32:
            return category_pts32, relation_pts32, "pts32"
        return category_fast, relation_fast, "fast"

    backend_keys = [
        "A1_filter_only",
        "A2_iou_nms05",
        "A3_global_relation",
        "A4_group_no_spatial",
        "A5_no_adaptive_dedup",
        "A8_full",
    ]
    oracle_keys = ["O1_oracle_category", "O2_oracle_dedup", "O3_proposal_cover"]

    # Main backend/component table on the final 12.74 cache.
    rows = []
    t0 = time.time()
    for i, name in enumerate(names):
        stem = Path(name).stem
        d, cache_path, cache_source = load_cache_for_scenario(stem, "final_multires", dirs)
        if d is None:
            row = empty_prediction_row(stem, ann)
            model_source = "none"
        else:
            category_head, relation_head, model_source = model_for_cache(cache_source)
            cover_sets = None
            if not args.skip_oracle:
                cover_sets = source_cover_sets(stem, d, ann, dirs, cache_source)
            row = evaluate_one(d, category_head, relation_head, text_prototypes, args.device, cover_sets)
        row["file_name"] = name
        row["cache_path"] = str(cache_path)
        row["cache_source"] = cache_source
        row["model_source"] = model_source
        rows.append(row)
        if (i + 1) % 100 == 0:
            print(f"  [final] {i+1}/{len(names)} elapsed={time.time()-t0:.1f}s", flush=True)

    keys = backend_keys if args.skip_oracle else backend_keys + oracle_keys
    out["final_multires"] = {
        "n_images": len(rows),
        "variants": aggregate_rows(rows, keys),
        "rows": rows,
    }

    # Frontend ablations. Only the full backend is needed here.
    for scenario, label in [
        ("no_51_100_multires", "A6_no_51_100_multires"),
        ("no_100plus_multires", "A7_no_100plus_multires"),
    ]:
        srows = []
        t1 = time.time()
        for i, name in enumerate(names):
            stem = Path(name).stem
            d, cache_path, cache_source = load_cache_for_scenario(stem, scenario, dirs)
            if d is None:
                row = empty_prediction_row(stem, ann)
                model_source = "none"
            else:
                category_head, relation_head, model_source = model_for_cache(cache_source)
                row = evaluate_one(d, category_head, relation_head, text_prototypes, args.device, None)
            srows.append({
                "file_name": name,
                "cache_path": str(cache_path),
                "cache_source": cache_source,
                "model_source": model_source,
                "gt_count": row["gt_count"],
                label: row["A8_full"],
                "n_candidates": row["n_candidates"],
            })
            if (i + 1) % 200 == 0:
                print(f"  [{label}] {i+1}/{len(names)} elapsed={time.time()-t1:.1f}s", flush=True)
        out[scenario] = {
            "n_images": len(srows),
            "variants": aggregate_rows(srows, [label]),
            "rows": srows,
        }

    anchor_path = REPO / "result/logs/fsc147_multires_extended.json"
    if anchor_path.exists():
        anchor = json.loads(anchor_path.read_text())
        got = out["final_multires"]["variants"]["A8_full"]["overall"]
        ref = anchor["overall"]
        if args.limit > 0:
            status = "SKIPPED_LIMIT"
        elif abs(got["MAE"] - ref["MAE"]) < 0.01 and abs(got["RMSE"] - ref["RMSE"]) < 0.01:
            status = "OK"
        else:
            status = "MISMATCH"
        out["anchor"] = {
            "source": str(anchor_path),
            "reference": ref,
            "got": got,
            "mae_abs_diff": abs(got["MAE"] - ref["MAE"]),
            "rmse_abs_diff": abs(got["RMSE"] - ref["RMSE"]),
            "status": status,
        }
        print(f"[anchor] A8 MAE={got['MAE']:.6f} vs ref={ref['MAE']:.6f} "
              f"status={out['anchor']['status']}")

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))
    print(f"wrote {out_path}")

    print("\n=== final_multires variants ===")
    for key in keys:
        m = out["final_multires"]["variants"][key]["overall"]
        print(f"{key:24s} MAE={m['MAE']:8.3f} RMSE={m['RMSE']:8.3f} bias={m['bias']:8.3f}")
    print("\n=== frontend ablations ===")
    for scenario, label in [
        ("no_51_100_multires", "A6_no_51_100_multires"),
        ("no_100plus_multires", "A7_no_100plus_multires"),
    ]:
        m = out[scenario]["variants"][label]["overall"]
        print(f"{label:24s} MAE={m['MAE']:8.3f} RMSE={m['RMSE']:8.3f} bias={m['bias']:8.3f}")


if __name__ == "__main__":
    main()
