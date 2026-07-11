"""OmniCount-191 multi-class ablations for OV-CUD.

This script evaluates prompt-free multi-label outputs against per-class counts.
It intentionally does not use the cached ``valid`` or ``matched_class`` fields
for predicted variants, because those fields are derived from GT dots. Oracle
variants rebuild candidate-to-class matches from the original OmniCount COCO
annotations and decoded cached masks.

Outputs:
  - protocol ablation: total-only vs predicted per-class vs oracle grouping
  - category separation ablation: global / p-dot / bucket / full / oracle
  - dedup ablation: SAM2 / category-only / IoU-NMS / relation / adaptive filter
  - difficulty slices: by #GT classes, super-category, and GT-count bin
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

np.seterr(over="ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.clustering.first_neighbor import (
    category_aware_clustering_with_spatial,
    first_neighbor_clustering,
    spatial_sub_clustering,
)
from code.counting.deduplicate import (
    build_same_instance_components,
    build_same_instance_components_adaptive,
)
from code.counting.representative import select_representatives
from code.heads.relation_head import PairwiseRelationHead, RelationHeadConfig
from code.matrix.pairwise_features import (
    box_geometry,
    build_pairwise_features,
    pairwise_feature_dim,
)
from script.eval_omnicount import count_image_class_agnostic
from script.preprocess_omnicount import load_omnicount_coco


BINS = [
    ("0-10", 0, 10),
    ("11-20", 11, 20),
    ("21-50", 21, 50),
    ("51-100", 51, 100),
    ("100+", 101, 10**9),
]


def bin_of(count: int) -> str:
    for label, lo, hi in BINS:
        if lo <= count <= hi:
            return label
    return "100+"


def class_bucket(n_classes: int) -> str:
    if n_classes <= 1:
        return "1"
    if n_classes == 2:
        return "2"
    if n_classes == 3:
        return "3"
    return "4+"


def load_category_head(ckpt_path: str, device: str):
    from script.train_category_v2 import CosineCategoryHead

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    head = CosineCategoryHead(
        in_dim=ck["in_dim"],
        proj_dim=ck["proj_dim"],
        dropout=ck.get("config", {}).get("dropout", 0.3),
        num_layers=ck.get("config", {}).get("num_layers", 2),
    )
    head.load_state_dict(ck["head"])
    head.to(device).eval()
    return head


def load_relation_head(ckpt_path: str, device: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = RelationHeadConfig(
        feat_dim=ck.get("feat_dim", pairwise_feature_dim(ck["z_dim"])),
        hidden_dim=ck.get("hidden_dim", 512),
        num_layers=ck.get("num_layers", 3),
        dropout=0.1,
    )
    head = PairwiseRelationHead(cfg)
    head.load_state_dict(ck["relation_head"])
    head.to(device).eval()
    return head


def decode_masks(d: dict[str, Any]) -> list[np.ndarray]:
    from pycocotools import mask as mask_utils

    masks = []
    for rle in d.get("masks_rle", []):
        counts = rle["counts"]
        if isinstance(counts, str):
            counts = counts.encode("ascii")
        masks.append(mask_utils.decode({"size": rle["size"], "counts": counts}).astype(bool))
    return masks


def build_entry_map(omnicount_dir: str) -> dict[tuple[str, str], dict[str, Any]]:
    entries = load_omnicount_coco(omnicount_dir)
    return {(e["category"], e["file_name"]): e for e in entries}


def candidate_class_sets(
    d: dict[str, Any],
    entry: dict[str, Any] | None,
    class_to_idx: dict[str, int],
) -> list[set[int]]:
    """Return GT class indices covered by each candidate mask."""
    n = int(d["z"].shape[0])
    out = [set() for _ in range(n)]
    if entry is None or "masks_rle" not in d:
        return out

    h, w = int(d["height"]), int(d["width"])
    masks = decode_masks(d)
    for ci, m in enumerate(masks[:n]):
        for ann in entry.get("annotations", []):
            cname = ann["class_name"]
            if cname not in class_to_idx:
                continue
            x = int(round(float(ann["cx"])))
            y = int(round(float(ann["cy"])))
            if 0 <= x < w and 0 <= y < h and bool(m[y, x]):
                out[ci].add(class_to_idx[cname])
    return out


def bbox_iou_matrix(bbox: np.ndarray) -> np.ndarray:
    n = bbox.shape[0]
    out = np.zeros((n, n), dtype=np.float32)
    if n == 0:
        return out
    xyxy = bbox.copy()
    xyxy[:, 2] = bbox[:, 0] + bbox[:, 2]
    xyxy[:, 3] = bbox[:, 1] + bbox[:, 3]
    area = np.maximum(bbox[:, 2] * bbox[:, 3], 1e-6)
    for i in range(n):
        xx1 = np.maximum(xyxy[i, 0], xyxy[:, 0])
        yy1 = np.maximum(xyxy[i, 1], xyxy[:, 1])
        xx2 = np.minimum(xyxy[i, 2], xyxy[:, 2])
        yy2 = np.minimum(xyxy[i, 3], xyxy[:, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = np.maximum(area[i] + area - inter, 1e-6)
        out[i] = inter / union
    return out


def adaptive_area_keep(
    bbox: np.ndarray,
    image_area: float,
    adaptive_mult: float,
    adaptive_floor: float,
) -> np.ndarray:
    areas = bbox[:, 2] * bbox[:, 3]
    if len(areas) == 0:
        return np.zeros((0,), dtype=bool)
    med = float(np.median(areas))
    cap = max(adaptive_mult * med if med > 0 else image_area, adaptive_floor * image_area)
    keep = areas <= cap
    if not np.any(keep):
        keep[:] = True
    return keep


@torch.no_grad()
def relation_matrices(
    rel_head,
    z: torch.Tensor,
    cat_probs: np.ndarray,
    bbox: torch.Tensor,
    device: str,
    max_rel_candidates: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(z.shape[0])
    A_sem = np.zeros((n, n), dtype=np.float32)
    A_inst = np.zeros((n, n), dtype=np.float32)
    A_part = np.zeros((n, n), dtype=np.float32)
    if n < 2:
        return A_sem, A_inst, A_part

    if n > max_rel_candidates:
        # Keep relation computation bounded. Uncomputed pairs stay at logit 0.
        keep = np.arange(max_rel_candidates)
    else:
        keep = np.arange(n)

    ii, jj = torch.triu_indices(len(keep), len(keep), offset=1, device=device)
    if ii.numel() == 0:
        return A_sem, A_inst, A_part

    keep_t = torch.as_tensor(keep, dtype=torch.long, device=device)
    gi = keep_t[ii]
    gj = keep_t[jj]

    z_d = z.to(device)
    p_d = torch.from_numpy(cat_probs).float().to(device)
    b_d = bbox.to(device)
    geom_ij = box_geometry(b_d[gi], b_d[gj])
    geom_ji = box_geometry(b_d[gj], b_d[gi])
    phi_ij = build_pairwise_features(z_d[gi], z_d[gj], p_d[gi], p_d[gj], b_d[gi], b_d[gj], geom_ij)
    phi_ji = build_pairwise_features(z_d[gj], z_d[gi], p_d[gj], p_d[gi], b_d[gj], b_d[gi], geom_ji)
    out = rel_head(torch.cat([phi_ij, phi_ji], dim=0))
    p = ii.numel()
    sem = 0.5 * (out["sem"][:p] + out["sem"][p:]).cpu().numpy()
    inst = 0.5 * (out["inst"][:p] + out["inst"][p:]).cpu().numpy()
    part = out["part"][:p].cpu().numpy()
    gi_np = gi.cpu().numpy()
    gj_np = gj.cpu().numpy()
    A_sem[gi_np, gj_np] = sem
    A_sem[gj_np, gi_np] = sem
    A_inst[gi_np, gj_np] = inst
    A_inst[gj_np, gi_np] = inst
    A_part[gi_np, gj_np] = part
    return A_sem, A_inst, A_part


def greedy_nms(indices: list[int], bbox: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    order = sorted(indices, key=lambda i: -float(scores[i]))
    keep: list[int] = []
    iou = bbox_iou_matrix(bbox)
    for i in order:
        if all(float(iou[i, j]) < iou_thr for j in keep):
            keep.append(i)
    return keep


def iou_components(indices: list[int], bbox: np.ndarray, iou_thr: float) -> list[list[int]]:
    if len(indices) <= 1:
        return [list(indices)] if indices else []
    iou = bbox_iou_matrix(bbox)
    parent = {i: i for i in indices}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for ai, i in enumerate(indices):
        for j in indices[ai + 1 :]:
            if float(iou[i, j]) >= iou_thr:
                union(i, j)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in indices:
        groups[find(i)].append(i)
    return list(groups.values())


def group_indices(
    mode: str,
    indices: np.ndarray,
    cat_probs: np.ndarray,
    top_class: np.ndarray,
    A_sem: np.ndarray,
    bbox: np.ndarray,
    image_area: float,
    oracle_sets: list[set[int]],
    class_names: list[str],
    tau_aff: float,
) -> list[tuple[str | None, list[int]]]:
    """Return groups as (forced_class_name_or_none, global_indices)."""
    if len(indices) == 0:
        return []
    idx_list = [int(i) for i in indices]
    if mode == "global":
        return [(None, idx_list)]
    if mode == "bucket":
        by_cls: dict[int, list[int]] = defaultdict(list)
        for i in idx_list:
            by_cls[int(top_class[i])].append(i)
        out = []
        for cls_idx, members in by_cls.items():
            subgroups = spatial_sub_clustering(members, bbox, image_area, max_group_size=30)
            cname = class_names[cls_idx] if 0 <= cls_idx < len(class_names) else str(cls_idx)
            out.extend((cname, g) for g in subgroups)
        return out
    if mode == "pdot":
        subp = cat_probs[indices]
        A_group = subp @ subp.T
        local_groups = first_neighbor_clustering(A_group, tau_affinity=tau_aff, top_class=None)
        return [(None, [int(indices[j]) for j in g]) for g in local_groups]
    if mode == "full":
        subp = cat_probs[indices]
        sub_A = A_sem[np.ix_(indices, indices)]
        local_groups = category_aware_clustering_with_spatial(
            subp,
            sub_A,
            bbox[indices],
            image_area,
            tau_affinity=tau_aff,
            max_group_size=30,
            use_bucketing=True,
        )
        return [(None, [int(indices[j]) for j in g]) for g in local_groups]
    if mode == "oracle":
        by_cls = defaultdict(list)
        for i in idx_list:
            for cls_idx in oracle_sets[i]:
                by_cls[int(cls_idx)].append(i)
        return [
            (class_names[cls_idx] if 0 <= cls_idx < len(class_names) else str(cls_idx), members)
            for cls_idx, members in sorted(by_cls.items())
        ]
    raise ValueError(f"unknown group mode: {mode}")


def reps_for_group(
    members: list[int],
    dedup: str,
    A_inst: np.ndarray,
    A_part: np.ndarray,
    cat_probs: np.ndarray,
    bbox: np.ndarray,
    image_area: float,
    tau_inst: float,
    nms_iou: float,
    min_rep_conf: float,
) -> list[int]:
    if len(members) == 0:
        return []
    if dedup == "none":
        return list(members)
    if dedup == "iou":
        components = iou_components(members, bbox, nms_iou)
    elif dedup == "relation":
        if len(members) > 20:
            components = build_same_instance_components_adaptive(
                members, A_inst, base_tau=tau_inst, use_greedy=True, max_comp_size=5
            )
        else:
            components = build_same_instance_components(members, A_inst, tau_inst=tau_inst)
    else:
        raise ValueError(f"unknown dedup mode: {dedup}")
    return select_representatives(
        components,
        A_part,
        cat_probs,
        bbox,
        image_area,
        min_category_conf=min_rep_conf,
    )


def add_count(dst: Counter, cname: str, value: int) -> None:
    if value > 0:
        dst[cname] += int(value)


def predict_variant(
    variant: dict[str, Any],
    prepared: dict[str, Any],
    args,
    class_names: list[str],
) -> tuple[dict[str, int] | None, int]:
    n = prepared["n"]
    bbox = prepared["bbox_np"]
    image_area = prepared["image_area"]
    top_conf = prepared["top_conf"]
    top_class = prepared["top_class"]
    cat_probs = prepared["cat_probs"]
    A_sem = prepared["A_sem"]
    A_inst = prepared["A_inst"]
    A_part = prepared["A_part"]
    oracle_sets = prepared["oracle_sets"]

    if variant["kind"] == "total_sam2":
        return None, n

    base_keep = np.ones(n, dtype=bool)
    if variant.get("area_filter", False):
        base_keep &= adaptive_area_keep(
            bbox,
            image_area,
            adaptive_mult=args.adaptive_mult,
            adaptive_floor=args.adaptive_floor,
        )

    if variant["kind"] == "total_class_agnostic":
        r = count_image_class_agnostic(
            prepared["d"],
            tau_inst=variant.get("tau_inst", 0.5),
            tau_aff=variant.get("tau_aff", 0.3),
            max_area_ratio=variant.get("max_area_ratio", 1.0),
            adaptive_mult=args.adaptive_mult,
            adaptive_floor=args.adaptive_floor,
        )
        return None, int(r["pred_count"])

    if variant["kind"] == "category_only":
        keep = base_keep & (top_conf >= args.conf_threshold)
        pred = Counter()
        for i in np.where(keep)[0]:
            add_count(pred, class_names[int(top_class[i])], 1)
        return dict(pred), int(sum(pred.values()))

    if variant["kind"] == "nms_by_class":
        keep = base_keep & (top_conf >= args.conf_threshold)
        pred = Counter()
        for cls_idx in np.unique(top_class[keep]):
            members = np.where(keep & (top_class == cls_idx))[0].tolist()
            kept = greedy_nms(members, bbox, top_conf, args.nms_iou)
            add_count(pred, class_names[int(cls_idx)], len(kept))
        return dict(pred), int(sum(pred.values()))

    if variant["kind"] == "grouped":
        if variant["group_mode"] == "oracle":
            keep = base_keep & np.array([bool(s) for s in oracle_sets], dtype=bool)
        else:
            keep = base_keep & (top_conf >= args.conf_threshold)
        indices = np.where(keep)[0]
        groups = group_indices(
            variant["group_mode"],
            indices,
            cat_probs,
            top_class,
            A_sem,
            bbox,
            image_area,
            oracle_sets,
            class_names,
            args.tau_aff,
        )
        pred = Counter()
        for forced_cls, members in groups:
            reps = reps_for_group(
                members,
                variant.get("dedup", "relation"),
                A_inst,
                A_part,
                cat_probs,
                bbox,
                image_area,
                args.tau_inst,
                args.nms_iou,
                0.0 if forced_cls is not None else args.min_rep_conf,
            )
            if not reps:
                continue
            if forced_cls is not None:
                add_count(pred, forced_cls, len(reps))
            elif variant.get("assign", "group") == "rep":
                for r in reps:
                    add_count(pred, class_names[int(top_class[r])], 1)
            else:
                rep_classes = [int(top_class[r]) for r in reps]
                cls_idx = Counter(rep_classes).most_common(1)[0][0]
                add_count(pred, class_names[cls_idx], len(reps))
        return dict(pred), int(sum(pred.values()))

    raise ValueError(f"unknown variant kind: {variant['kind']}")


def total_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        return {"n_images": 0, "MAE": None, "RMSE": None, "bias": None}
    pred = np.array([r["pred_total"] for r in rows], dtype=float)
    gt = np.array([r["gt_total"] for r in rows], dtype=float)
    err = pred - gt
    return {
        "n_images": int(len(rows)),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "mean_GT": float(np.mean(gt)),
        "mean_Pred": float(np.mean(pred)),
    }


def multiclass_metrics(rows: list[dict[str, Any]], class_names: list[str]) -> dict[str, Any] | None:
    if not rows or rows[0].get("pred_counts") is None:
        return None

    per_class = {}
    rmse_all = []
    mae_all = []
    rmse_nz = []
    mae_nz = []
    for cname in class_names:
        pred = np.array([r["pred_counts"].get(cname, 0) for r in rows], dtype=float)
        gt = np.array([r["gt_counts"].get(cname, 0) for r in rows], dtype=float)
        err = pred - gt
        pos = gt > 0
        cls = {
            "n_pos": int(pos.sum()),
            "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err**2))),
            "bias": float(np.mean(err)),
        }
        rmse_all.append(cls["RMSE"])
        mae_all.append(cls["MAE"])
        if pos.any():
            cls["MAE_nz"] = float(np.mean(np.abs(err[pos])))
            cls["RMSE_nz"] = float(np.sqrt(np.mean(err[pos] ** 2)))
            rmse_nz.append(cls["RMSE_nz"])
            mae_nz.append(cls["MAE_nz"])
        per_class[cname] = cls

    return {
        "mMAE": float(np.mean(mae_all)),
        "mRMSE": float(np.mean(rmse_all)),
        "mMAE_nz": float(np.mean(mae_nz)) if mae_nz else None,
        "mRMSE_nz": float(np.mean(rmse_nz)) if rmse_nz else None,
        "per_class": per_class,
    }


def summarize_variant(rows: list[dict[str, Any]], class_names: list[str]) -> dict[str, Any]:
    summary = {"total": total_metrics(rows)}
    mc = multiclass_metrics(rows, class_names)
    if mc is not None:
        summary.update({k: v for k, v in mc.items() if k != "per_class"})
        summary["per_class"] = mc["per_class"]
    return summary


def slice_summaries(rows: list[dict[str, Any]], class_names: list[str]) -> dict[str, Any]:
    out = {}
    for key in ["nclass_bucket", "supercategory", "gt_bin"]:
        by = defaultdict(list)
        for r in rows:
            by[r[key]].append(r)
        out[key] = {label: summarize_variant(rs, class_names) for label, rs in sorted(by.items())}
    return out


def variant_specs() -> dict[str, list[dict[str, Any]]]:
    protocol = [
        {
            "name": "protocol_class_agnostic_default",
            "kind": "total_class_agnostic",
            "tau_inst": 0.5,
            "tau_aff": 0.3,
            "max_area_ratio": 1.0,
        },
        {
            "name": "protocol_class_agnostic_adaptive",
            "kind": "total_class_agnostic",
            "tau_inst": 0.5,
            "tau_aff": 0.3,
            "max_area_ratio": -1.0,
        },
        {"name": "protocol_predicted_groups", "kind": "grouped", "group_mode": "full", "dedup": "relation"},
        {"name": "protocol_oracle_class_grouping", "kind": "grouped", "group_mode": "oracle", "dedup": "relation"},
    ]
    category = [
        {"name": "catsep_global_no_class_group", "kind": "grouped", "group_mode": "global", "dedup": "relation", "assign": "rep"},
        {"name": "catsep_pdot_grouping", "kind": "grouped", "group_mode": "pdot", "dedup": "relation"},
        {"name": "catsep_category_bucket", "kind": "grouped", "group_mode": "bucket", "dedup": "relation"},
        {"name": "catsep_full_category_relation", "kind": "grouped", "group_mode": "full", "dedup": "relation"},
        {"name": "catsep_oracle_category", "kind": "grouped", "group_mode": "oracle", "dedup": "relation"},
    ]
    dedup = [
        {"name": "dedup_sam2_only_total", "kind": "total_sam2"},
        {"name": "dedup_category_only", "kind": "category_only"},
        {"name": "dedup_iou_nms", "kind": "nms_by_class"},
        {"name": "dedup_relation_head", "kind": "grouped", "group_mode": "full", "dedup": "relation"},
        {
            "name": "dedup_relation_head_adaptive_filter",
            "kind": "grouped",
            "group_mode": "full",
            "dedup": "relation",
            "area_filter": True,
        },
    ]
    all_specs = protocol + category + dedup
    # Preserve first occurrence by variant name.
    seen = set()
    unique = []
    for v in all_specs:
        if v["name"] not in seen:
            unique.append(v)
            seen.add(v["name"])
    return {"protocol": protocol, "category_separation": category, "dedup": dedup, "all": unique}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/home/czp/ws_yiyang/ovcud_cache/omnicount_test")
    ap.add_argument("--omnicount-dir", default="/home/czp/official_code/dataset/omnicount/OmniCount-191")
    ap.add_argument("--category-ckpt", default="result/checkpoints/category_coco80_cosine.pt")
    ap.add_argument("--relation-ckpt", default="result/checkpoints/fsc147_relation_pts32_best.pt")
    ap.add_argument("--text-prototypes", default="result/checkpoints/text_prototypes_omnicount.pt")
    ap.add_argument("--class-names", default="result/checkpoints/omnicount_class_names.json")
    ap.add_argument("--out", default="result/logs/omnicount_multiclass_ablation_sample.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--min-classes", type=int, default=1)
    ap.add_argument("--conf-threshold", type=float, default=0.2)
    ap.add_argument("--tau-inst", type=float, default=0.99)
    ap.add_argument("--tau-aff", type=float, default=0.1)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--min-rep-conf", type=float, default=0.05)
    ap.add_argument("--adaptive-mult", type=float, default=8.0)
    ap.add_argument("--adaptive-floor", type=float, default=0.08)
    ap.add_argument("--max-rel-candidates", type=int, default=220)
    ap.add_argument("--save-per-image", action="store_true")
    args = ap.parse_args()

    device = args.device
    class_names = json.load(open(args.class_names))
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    entry_map = build_entry_map(args.omnicount_dir)

    files = sorted(Path(args.cache_dir).glob("*.pt"))
    selected = []
    for fp in files:
        d0 = torch.load(fp, map_location="cpu", weights_only=False)
        if len(d0.get("unique_classes") or []) >= args.min_classes:
            selected.append(fp)
        if args.limit > 0 and len(selected) >= args.limit:
            break

    print(f"[data] cache={args.cache_dir}")
    print(f"[data] selected={len(selected)} min_classes={args.min_classes} limit={args.limit}")
    print(f"[model] category={args.category_ckpt}")
    print(f"[model] relation={args.relation_ckpt}")
    print(f"[model] prototypes={args.text_prototypes}")

    category_head = load_category_head(args.category_ckpt, device)
    relation_head = load_relation_head(args.relation_ckpt, device)
    text_proto = F.normalize(
        torch.load(args.text_prototypes, map_location=device, weights_only=False).float(), dim=-1
    )

    specs = variant_specs()
    rows_by_variant: dict[str, list[dict[str, Any]]] = {v["name"]: [] for v in specs["all"]}

    t0 = time.time()
    for fi, fp in enumerate(selected):
        d = torch.load(fp, map_location="cpu", weights_only=False)
        category = d.get("category", "unknown")
        file_name = d.get("file_name", fp.name)
        entry = entry_map.get((category, file_name))
        gt_counts = {str(k): int(v) for k, v in dict(d.get("class_counts", {})).items()}
        gt_total = int(d.get("gt_count", sum(gt_counts.values())))
        n_gt_classes = len([v for v in gt_counts.values() if v > 0])

        z = d["z"].float()
        bbox_t = d["bbox"].float()
        bbox_np = bbox_t.numpy()
        n = int(z.shape[0])
        image_area = float(int(d["height"]) * int(d["width"]))

        with torch.no_grad():
            logits = category_head(z.to(device), text_proto)
            cat_probs = torch.softmax(logits, dim=-1).cpu().numpy()
        top_conf = cat_probs.max(axis=1) if n else np.zeros((0,), dtype=np.float32)
        top_class = cat_probs.argmax(axis=1) if n else np.zeros((0,), dtype=np.int64)
        A_sem, A_inst, A_part = relation_matrices(
            relation_head,
            z,
            cat_probs,
            bbox_t,
            device,
            max_rel_candidates=args.max_rel_candidates,
        )
        oracle_sets = candidate_class_sets(d, entry, class_to_idx)

        prepared = {
            "d": d,
            "n": n,
            "bbox_np": bbox_np,
            "image_area": image_area,
            "top_conf": top_conf,
            "top_class": top_class,
            "cat_probs": cat_probs,
            "A_sem": A_sem,
            "A_inst": A_inst,
            "A_part": A_part,
            "oracle_sets": oracle_sets,
        }

        common = {
            "file": file_name,
            "supercategory": category,
            "gt_counts": gt_counts,
            "gt_total": gt_total,
            "n_gt_classes": n_gt_classes,
            "nclass_bucket": class_bucket(n_gt_classes),
            "gt_bin": bin_of(gt_total),
            "n_candidates": n,
        }
        for variant in specs["all"]:
            pred_counts, pred_total = predict_variant(variant, prepared, args, class_names)
            row = dict(common)
            row["variant"] = variant["name"]
            row["pred_counts"] = pred_counts
            row["pred_total"] = int(pred_total)
            rows_by_variant[variant["name"]].append(row)

        if (fi + 1) % 50 == 0 or (fi + 1) == len(selected):
            elapsed = time.time() - t0
            key = "protocol_predicted_groups"
            m = total_metrics(rows_by_variant[key])
            print(
                f"  [{fi+1}/{len(selected)}] {key} MAE={m['MAE']:.2f} "
                f"rate={(fi+1)/max(elapsed, 1e-6):.2f}/s"
            )

    summaries = {name: summarize_variant(rows, class_names) for name, rows in rows_by_variant.items()}
    slices = {name: slice_summaries(rows, class_names) for name, rows in rows_by_variant.items()}

    output = {
        "config": {
            "cache_dir": args.cache_dir,
            "omnicount_dir": args.omnicount_dir,
            "category_ckpt": args.category_ckpt,
            "relation_ckpt": args.relation_ckpt,
            "text_prototypes": args.text_prototypes,
            "class_names": args.class_names,
            "device": device,
            "limit": args.limit,
            "min_classes": args.min_classes,
            "conf_threshold": args.conf_threshold,
            "tau_inst": args.tau_inst,
            "tau_aff": args.tau_aff,
            "nms_iou": args.nms_iou,
        },
        "variant_groups": {
            k: [v["name"] for v in vals] for k, vals in specs.items() if k != "all"
        },
        "summaries": summaries,
        "slices": slices,
    }
    if args.save_per_image:
        output["per_image"] = rows_by_variant

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"[save] {args.out}")

    print("\nProtocol:")
    for v in specs["protocol"]:
        s = summaries[v["name"]]
        print(
            f"  {v['name']:<38} totalMAE={s['total']['MAE']:.3f} "
            f"mRMSE={s.get('mRMSE')} mRMSE_nz={s.get('mRMSE_nz')}"
        )
    print("\nCategory separation:")
    for v in specs["category_separation"]:
        s = summaries[v["name"]]
        print(
            f"  {v['name']:<38} totalMAE={s['total']['MAE']:.3f} "
            f"mRMSE={s.get('mRMSE')} mRMSE_nz={s.get('mRMSE_nz')}"
        )
    print("\nDedup:")
    for v in specs["dedup"]:
        s = summaries[v["name"]]
        print(
            f"  {v['name']:<38} totalMAE={s['total']['MAE']:.3f} "
            f"mRMSE={s.get('mRMSE')} mRMSE_nz={s.get('mRMSE_nz')}"
        )


if __name__ == "__main__":
    main()
