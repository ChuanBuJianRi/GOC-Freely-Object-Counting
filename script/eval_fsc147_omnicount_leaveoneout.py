"""Full M1-M6 leave-one-out evaluation on FSC-147 and OmniCount-191.

The learned weights are those used by the FSC-147 12.67 main method. The
datasets keep their native evaluation protocols:

* FSC-147: full 1,190-image total-count MAE/RMSE. This reproduces the current
  cache-compatible protocol, including its GT-dot-derived ``valid`` field.
* OmniCount-191: full 1,957-image prompt-free semantic groups. Predictions do
  not read cached ``valid``/``matched_class`` fields. We report total-count,
  present-class, mRMSE, and mRMSE-nz metrics.

M1-M5 each remove one component from M6:
  M1 - relation-head dedup -> category grouping + box-IoU NMS
  M2 - adaptive density/confidence filtering -> fixed pool + no conf filter
  M3 - COCO relation pretraining -> dot-supervised pts32 relation head
  M4 - high-resolution candidate/head training -> pts16 pool and heads
  M5 - T4 extreme-density rescue
  M6 - full model
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from script.ablation_fsc147_multires_components import (  # noqa: E402
    CACHE_ROOT,
    CONF,
    DEFAULT_ANN,
    DEFAULT_SPLIT,
    TAU_AFF,
    TAU_INST,
    cluster_full,
    dedup_count,
    gt_count_from_ann,
    load_test_names,
    relation_matrices as fsc_relation_matrices,
    semantic_affinity,
)
from script.eval_carpk import get_category_probs  # noqa: E402
from script.eval_omnicount_multiclass_ablation import (  # noqa: E402
    bin_of,
    class_bucket,
    greedy_nms,
    group_indices,
    load_category_head,
    load_relation_head,
    multiclass_metrics,
    relation_matrices as omni_relation_matrices,
    reps_for_group,
    total_metrics,
)


VARIANTS = ("m1", "m2", "m3", "m4", "m5", "m6")
FSC_BINS = (
    ("0-10", 0, 10),
    ("11-20", 11, 20),
    ("21-50", 21, 50),
    ("51-100", 51, 100),
    ("100+", 101, 10**9),
)

CKPT = {
    "category_fast": REPO / "result/checkpoints/category_cosine_fast.pt",
    "category_pts32": REPO / "result/checkpoints/category_cosine_pts32.pt",
    "relation_fast": REPO / "result/checkpoints/fsc147_relation_best.pt",
    "relation_pts32": REPO / "result/checkpoints/fsc147_relation_pts32_best.pt",
    "relation_dotonly": REPO / "result/checkpoints/fsc147_relation_pts32_dotonly_m3.pt",
    "relation_exp5c": REPO / "result/checkpoints/fsc147_relation_exp5c.pt",
    "proto_fsc": REPO / "result/checkpoints/text_prototypes_fsc147.pt",
    "proto_omni": REPO / "result/checkpoints/text_prototypes_omnicount.pt",
    "omni_classes": REPO / "result/checkpoints/omnicount_class_names.json",
}

FSC_CACHE = {
    "final": CACHE_ROOT / "fsc147_test_multires_all",
    "fast": CACHE_ROOT / "fsc147_test_fast",
    "mr100": CACHE_ROOT / "fsc147_test_multires",
    "mr51": CACHE_ROOT / "fsc147_test_multires_51_100",
    "rescue": CACHE_ROOT / "fsc147_rescue_7611_4x4_ov25_bbox",
}

DEFINITIONS = {
    "m1": "M6 minus RH: learned relation dedup is replaced by class-aware box-IoU NMS@0.5.",
    "m2": "M6 minus ADF: fixed base candidate density and confidence threshold 0; T4 trigger retained.",
    "m3": "M6 minus CP: pts32 relation head is FSC-dot-supervised only; all other weights/config fixed.",
    "m4": "M6 minus HR: pts16 candidate pool and pts16 category/relation heads; T4 trigger retained.",
    "m5": "M6 minus T4: disable the 4x4 zero-candidate rescue and keep the original multires frontend.",
    "m6": "Full learned model and inference policy, including the no-GT fast-zero -> 4x4 T4 rescue.",
}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scalar_metrics(pred: list[float], gt: list[float]) -> dict[str, float | int | None]:
    if not gt:
        return {"n": 0, "MAE": None, "RMSE": None, "bias": None}
    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    err = p - g
    return {
        "n": int(len(g)),
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt(np.square(err).mean())),
        "bias": float(err.mean()),
        "mean_gt": float(g.mean()),
        "mean_pred": float(p.mean()),
    }


def bootstrap_ci(
    pred: list[float], gt: list[float], n_boot: int, seed: int
) -> dict[str, Any]:
    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    err = p - g
    rng = np.random.default_rng(seed)
    maes = np.empty(n_boot, dtype=np.float64)
    rmses = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, len(err), len(err))
        sample = err[idx]
        maes[i] = np.abs(sample).mean()
        rmses[i] = np.sqrt(np.square(sample).mean())
    return {
        "n_boot": n_boot,
        "seed": seed,
        "MAE_95CI": [float(x) for x in np.quantile(maes, [0.025, 0.975])],
        "RMSE_95CI": [float(x) for x in np.quantile(rmses, [0.025, 0.975])],
    }


def paired_mae_delta_ci(
    pred: list[float], ref: list[float], gt: list[float], n_boot: int, seed: int
) -> dict[str, Any]:
    p = np.asarray(pred, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    per_image_delta = np.abs(p - g) - np.abs(r - g)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, len(g), len(g))
        samples[i] = per_image_delta[idx].mean()
    return {
        "delta_MAE_vs_M6": float(per_image_delta.mean()),
        "delta_MAE_95CI": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
    }


def load_models(device: str) -> dict[str, Any]:
    for name, path in CKPT.items():
        if name == "omni_classes":
            continue
        if not path.exists():
            raise FileNotFoundError(f"missing required asset: {path}")
    print("[model] loading category/relation heads", flush=True)
    models = {
        "category_fast": load_category_head(str(CKPT["category_fast"]), device),
        "category_pts32": load_category_head(str(CKPT["category_pts32"]), device),
        "relation_fast": load_relation_head(str(CKPT["relation_fast"]), device),
        "relation_pts32": load_relation_head(str(CKPT["relation_pts32"]), device),
        "relation_dotonly": load_relation_head(str(CKPT["relation_dotonly"]), device),
        "relation_exp5c": load_relation_head(str(CKPT["relation_exp5c"]), device),
        "proto_fsc": F.normalize(
            torch.load(CKPT["proto_fsc"], map_location=device, weights_only=False).float(),
            dim=-1,
        ),
        "proto_omni": F.normalize(
            torch.load(CKPT["proto_omni"], map_location=device, weights_only=False).float(),
            dim=-1,
        ),
    }
    return models


def fsc_source(stem: str) -> tuple[str, str]:
    file_name = f"{stem}.pt"
    if (FSC_CACHE["mr100"] / file_name).exists():
        return "pts32", "final_mr100"
    if (FSC_CACHE["mr51"] / file_name).exists():
        return "pts32", "final_mr51"
    return "fast", "final_fast"


def load_pt(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def grouped_box_nms_count(
    groups: list[list[int]], bbox: np.ndarray, scores: np.ndarray, iou_threshold: float
) -> int:
    total = 0
    for group in groups:
        kept: list[int] = []
        for i in sorted(group, key=lambda index: -float(scores[index])):
            xi, yi, wi, hi = bbox[i]
            duplicate = False
            for j in kept:
                xj, yj, wj, hj = bbox[j]
                x1, y1 = max(xi, xj), max(yi, yj)
                x2, y2 = min(xi + wi, xj + wj), min(yi + hi, yj + hj)
                inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                union = wi * hi + wj * hj - inter
                if union > 0 and inter / union >= iou_threshold:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(i)
        total += len(kept)
    return total


def fsc_predict(
    d: dict[str, Any] | None,
    category_head,
    relation_head,
    text_proto: torch.Tensor,
    device: str,
    conf_threshold: float,
    dedup_mode: str,
) -> tuple[int, int, int]:
    if d is None:
        return 0, 0, 0
    z = d["z"].float()
    n = int(z.shape[0])
    if n == 0:
        return 0, 0, 0
    bbox = d["bbox"].float().numpy()
    image_area = float(int(d["height"]) * int(d["width"]))
    probs = get_category_probs(category_head, z, text_proto, device)
    top_conf = probs.max(axis=1)
    valid = np.asarray(d["valid"]) > 0
    indices = [i for i in range(n) if valid[i] and top_conf[i] >= conf_threshold]
    if not indices:
        return 0, n, 0

    A_sem = semantic_affinity(probs)
    local_groups = cluster_full(probs, A_sem, bbox, indices, image_area)
    groups = [[indices[j] for j in group] for group in local_groups]
    if dedup_mode == "box_iou":
        pred = grouped_box_nms_count(groups, bbox, top_conf, iou_threshold=0.5)
        return int(pred), n, len(indices)

    A_inst = fsc_relation_matrices(d, z, probs, top_conf, relation_head, device)
    pred = dedup_count(groups, A_inst, probs, bbox, image_area, n, adaptive=True)
    return int(pred), n, len(indices)


def evaluate_fsc147(
    models: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    names = load_test_names(DEFAULT_SPLIT)
    if args.limit > 0:
        names = names[: args.limit]
    ann = json.loads(DEFAULT_ANN.read_text())
    rows_by_variant = {name: [] for name in VARIANTS}
    t0 = time.time()

    for index, image_name in enumerate(names):
        stem = Path(image_name).stem
        gt = gt_count_from_ann(stem, ann)
        d_fast = load_pt(FSC_CACHE["fast"] / f"{stem}.pt")
        d_final = load_pt(FSC_CACHE["final"] / f"{stem}.pt")
        fast_zero = d_fast is None or int(d_fast["z"].shape[0]) == 0
        d_rescue = load_pt(FSC_CACHE["rescue"] / f"{stem}.pt") if fast_zero else None
        rescue_applied = fast_zero and d_rescue is not None
        d_main = d_rescue if rescue_applied else d_final
        main_source, final_cache_source = fsc_source(stem)
        if rescue_applied:
            main_source = "pts32"
            final_cache_source = "rescue_4x4"

        main_cat = models[f"category_{main_source}"]
        main_rel = models[f"relation_{main_source}"]

        # M1: keep grouping/filter/frontend, replace learned dedup with IoU NMS.
        pred, n, n_valid = fsc_predict(
            d_main, main_cat, None, models["proto_fsc"], args.device, CONF, "box_iou"
        )
        rows_by_variant["m1"].append({
            "file_name": image_name, "gt_count": gt, "pred_count": pred,
            "n_candidates": n, "n_valid": n_valid, "cache_source": final_cache_source,
            "t4_applied": rescue_applied,
        })

        # M2: fixed fast pool and no confidence filter, while retaining T4.
        d_m2 = d_rescue if rescue_applied else d_fast
        source_m2 = "pts32" if rescue_applied else "fast"
        pred, n, n_valid = fsc_predict(
            d_m2,
            models[f"category_{source_m2}"],
            models[f"relation_{source_m2}"],
            models["proto_fsc"],
            args.device,
            0.0,
            "relation",
        )
        rows_by_variant["m2"].append({
            "file_name": image_name, "gt_count": gt, "pred_count": pred,
            "n_candidates": n, "n_valid": n_valid,
            "cache_source": "rescue_4x4" if rescue_applied else "fixed_fast",
            "t4_applied": rescue_applied,
        })

        # M3: only the pts32 relation head is replaced with the dot-only head.
        rel_m3 = models["relation_dotonly"] if main_source == "pts32" else models["relation_fast"]
        pred, n, n_valid = fsc_predict(
            d_main, main_cat, rel_m3, models["proto_fsc"], args.device, CONF, "relation"
        )
        rows_by_variant["m3"].append({
            "file_name": image_name, "gt_count": gt, "pred_count": pred,
            "n_candidates": n, "n_valid": n_valid, "cache_source": final_cache_source,
            "t4_applied": rescue_applied,
        })

        # M4: pts16 heads/pool everywhere; T4 remains available on fast-zero images.
        d_m4 = d_rescue if rescue_applied else d_fast
        pred, n, n_valid = fsc_predict(
            d_m4,
            models["category_fast"],
            models["relation_exp5c"],
            models["proto_fsc"],
            args.device,
            CONF,
            "relation",
        )
        rows_by_variant["m4"].append({
            "file_name": image_name, "gt_count": gt, "pred_count": pred,
            "n_candidates": n, "n_valid": n_valid,
            "cache_source": "rescue_4x4_pts16_heads" if rescue_applied else "fixed_fast_pts16",
            "t4_applied": rescue_applied,
        })

        # M5: identical learned model but no T4 rescue.
        source_m5, cache_m5 = fsc_source(stem)
        pred, n, n_valid = fsc_predict(
            d_final,
            models[f"category_{source_m5}"],
            models[f"relation_{source_m5}"],
            models["proto_fsc"],
            args.device,
            CONF,
            "relation",
        )
        rows_by_variant["m5"].append({
            "file_name": image_name, "gt_count": gt, "pred_count": pred,
            "n_candidates": n, "n_valid": n_valid, "cache_source": cache_m5,
            "t4_applied": False,
        })

        # M6: full current main method.
        pred, n, n_valid = fsc_predict(
            d_main, main_cat, main_rel, models["proto_fsc"], args.device, CONF, "relation"
        )
        rows_by_variant["m6"].append({
            "file_name": image_name, "gt_count": gt, "pred_count": pred,
            "n_candidates": n, "n_valid": n_valid, "cache_source": final_cache_source,
            "t4_applied": rescue_applied,
        })

        if (index + 1) % 100 == 0 or index + 1 == len(names):
            m6 = scalar_metrics(
                [r["pred_count"] for r in rows_by_variant["m6"]],
                [r["gt_count"] for r in rows_by_variant["m6"]],
            )
            print(
                f"  [FSC147 {index+1}/{len(names)}] M6 MAE={m6['MAE']:.3f} "
                f"rate={(index+1)/max(time.time()-t0, 1e-6):.2f}/s",
                flush=True,
            )

    summaries = {}
    m6_pred = [r["pred_count"] for r in rows_by_variant["m6"]]
    gts = [r["gt_count"] for r in rows_by_variant["m6"]]
    for offset, name in enumerate(VARIANTS):
        rows = rows_by_variant[name]
        if len(rows) != len(names) or len({row["file_name"] for row in rows}) != len(names):
            raise RuntimeError(f"FSC147 {name} output is incomplete or has duplicate image IDs")
        pred = [r["pred_count"] for r in rows]
        per_bin = {}
        for label, lo, hi in FSC_BINS:
            selected = [r for r in rows if lo <= r["gt_count"] <= hi]
            per_bin[label] = scalar_metrics(
                [r["pred_count"] for r in selected],
                [r["gt_count"] for r in selected],
            )
        summaries[name] = {
            "definition": DEFINITIONS[name],
            "metrics": scalar_metrics(pred, gts),
            "per_gt_count_bin": per_bin,
            "bootstrap": bootstrap_ci(pred, gts, args.bootstrap, args.seed + offset),
            "paired_vs_m6": paired_mae_delta_ci(
                pred, m6_pred, gts, args.bootstrap, args.seed + 100 + offset
            ),
        }

    if args.limit <= 0:
        m6 = summaries["m6"]["metrics"]
        if abs(float(m6["MAE"]) - 12.668067226890756) > 1e-9:
            raise RuntimeError(f"FSC147 M6 anchor mismatch: {m6}")
        if abs(float(m6["RMSE"]) - 113.71112226924458) > 1e-9:
            raise RuntimeError(f"FSC147 M6 RMSE anchor mismatch: {m6}")
        changed_m5 = [
            left["file_name"]
            for left, right in zip(rows_by_variant["m5"], rows_by_variant["m6"])
            if left["pred_count"] != right["pred_count"]
        ]
        if changed_m5 != ["7611.jpg"]:
            raise RuntimeError(f"FSC147 M5/M6 must differ only on 7611.jpg: {changed_m5}")

    return {
        "config": {
            "dataset": "FSC147 test full",
            "num_images": len(names),
            "protocol": "cache-compatible; cache valid is GT-dot-derived",
            "tau_inst": TAU_INST,
            "tau_affinity": TAU_AFF,
            "conf_threshold": CONF,
            "t4_trigger": "fast n_candidates == 0",
            "t4_rescue_cache": str(FSC_CACHE["rescue"]),
        },
        "summaries": summaries,
        "per_image": rows_by_variant,
    }


def omni_category_probs(head, z: torch.Tensor, proto: torch.Tensor, device: str) -> np.ndarray:
    if z.shape[0] == 0:
        return np.zeros((0, int(proto.shape[0])), dtype=np.float32)
    with torch.no_grad():
        logits = head(z.to(device), proto)
        return torch.softmax(logits, dim=-1).cpu().numpy()


def prepare_omni(
    d: dict[str, Any], category_head, relation_head, proto: torch.Tensor, args
) -> dict[str, Any]:
    z = d["z"].float()
    bbox_t = d["bbox"].float()
    probs = omni_category_probs(category_head, z, proto, args.device)
    if relation_head is None:
        n = int(z.shape[0])
        A_sem = (probs @ probs.T).astype(np.float32) * 10.0
        A_inst = np.zeros((n, n), dtype=np.float32)
        A_part = np.zeros((n, n), dtype=np.float32)
    else:
        A_sem, A_inst, A_part = omni_relation_matrices(
            relation_head, z, probs, bbox_t, args.device, args.max_rel_candidates
        )
    return {
        "n": int(z.shape[0]),
        "bbox_np": bbox_t.numpy(),
        "image_area": float(int(d["height"]) * int(d["width"])),
        "cat_probs": probs,
        "top_conf": probs.max(axis=1) if len(probs) else np.zeros(0, dtype=np.float32),
        "top_class": probs.argmax(axis=1) if len(probs) else np.zeros(0, dtype=np.int64),
        "A_sem": A_sem,
        "A_inst": A_inst,
        "A_part": A_part,
        "oracle_sets": [set() for _ in range(int(z.shape[0]))],
    }


def omni_predict(
    prepared: dict[str, Any], class_names: list[str], args, conf: float, dedup: str
) -> dict[str, int]:
    n = prepared["n"]
    if n == 0:
        return {}
    keep = np.where(prepared["top_conf"] >= conf)[0]
    groups = group_indices(
        "full",
        keep,
        prepared["cat_probs"],
        prepared["top_class"],
        prepared["A_sem"],
        prepared["bbox_np"],
        prepared["image_area"],
        prepared["oracle_sets"],
        class_names,
        args.tau_affinity,
    )
    pred: Counter[str] = Counter()
    for _, members in groups:
        if dedup == "box_iou":
            reps = greedy_nms(
                members, prepared["bbox_np"], prepared["top_conf"], args.nms_iou
            )
        else:
            reps = reps_for_group(
                members,
                "relation",
                prepared["A_inst"],
                prepared["A_part"],
                prepared["cat_probs"],
                prepared["bbox_np"],
                prepared["image_area"],
                args.tau_inst,
                args.nms_iou,
                args.min_rep_conf,
            )
        if not reps:
            continue
        classes = [int(prepared["top_class"][r]) for r in reps]
        cls_idx = Counter(classes).most_common(1)[0][0]
        pred[class_names[cls_idx]] += len(reps)
    return dict(pred)


def present_class_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gt, pred = [], []
    for row in rows:
        for name, count in row["gt_counts"].items():
            if count > 0:
                gt.append(float(count))
                pred.append(float(row["pred_counts"].get(name, 0)))
    return scalar_metrics(pred, gt)


def compact_omni_summary(
    rows: list[dict[str, Any]], class_names: list[str], args, seed: int
) -> dict[str, Any]:
    total = total_metrics(rows)
    mc = multiclass_metrics(rows, class_names)
    pred_total = [r["pred_total"] for r in rows]
    gt_total = [r["gt_total"] for r in rows]
    slices = {}
    for key in ("nclass_bucket", "supercategory", "gt_bin"):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row[key])].append(row)
        slices[key] = {
            label: {
                "num_images": len(selected),
                "total": total_metrics(selected),
                "present_class": present_class_metrics(selected),
            }
            for label, selected in sorted(grouped.items())
        }
    return {
        "total": total,
        "present_class": present_class_metrics(rows),
        "mMAE": mc["mMAE"],
        "mRMSE": mc["mRMSE"],
        "mMAE_nz": mc["mMAE_nz"],
        "mRMSE_nz": mc["mRMSE_nz"],
        "per_class": mc["per_class"],
        "bootstrap": bootstrap_ci(pred_total, gt_total, args.bootstrap, seed),
        "slices": slices,
    }


def evaluate_omnicount(
    models: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    cache32 = Path(args.omni_cache_pts32)
    cache16 = Path(args.omni_cache_pts16)
    all_files32 = sorted(cache32.glob("*.pt"))
    all_names16 = {path.name for path in cache16.glob("*.pt")}
    if args.limit <= 0 and (
        len(all_files32) != 1957
        or len(all_names16) != 1957
        or {path.name for path in all_files32} != all_names16
    ):
        raise RuntimeError("OmniCount pts16/pts32 caches must contain the same 1,957 files")
    files = all_files32
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"no OmniCount cache files in {cache32}")
    class_names = json.loads(CKPT["omni_classes"].read_text())
    rows_by_variant = {name: [] for name in VARIANTS}
    zero_pts32: list[str] = []
    t0 = time.time()

    for index, path32 in enumerate(files):
        d32 = load_pt(path32)
        d16 = load_pt(cache16 / path32.name)
        if d16 is None:
            raise FileNotFoundError(f"M4 pts16 cache missing: {cache16 / path32.name}")
        if int(d32["z"].shape[0]) == 0:
            zero_pts32.append(path32.name)

        full = prepare_omni(
            d32, models["category_pts32"], models["relation_pts32"], models["proto_omni"], args
        )
        no_relation = prepare_omni(
            d32, models["category_pts32"], None, models["proto_omni"], args
        )
        dotonly = prepare_omni(
            d32, models["category_pts32"], models["relation_dotonly"], models["proto_omni"], args
        )
        pts16 = prepare_omni(
            d16, models["category_fast"], models["relation_exp5c"], models["proto_omni"], args
        )

        predictions = {
            "m1": omni_predict(no_relation, class_names, args, args.conf_threshold, "box_iou"),
            "m2": omni_predict(full, class_names, args, 0.0, "relation"),
            "m3": omni_predict(dotonly, class_names, args, args.conf_threshold, "relation"),
            "m4": omni_predict(pts16, class_names, args, args.conf_threshold, "relation"),
            "m6": omni_predict(full, class_names, args, args.conf_threshold, "relation"),
        }
        predictions["m5"] = dict(predictions["m6"])

        gt_counts = {str(k): int(v) for k, v in dict(d32["class_counts"]).items()}
        gt_total = int(d32.get("gt_count", sum(gt_counts.values())))
        common = {
            "file_name": str(d32.get("file_name", path32.name)),
            "supercategory": str(d32.get("category", "unknown")),
            "gt_counts": gt_counts,
            "gt_total": gt_total,
            "n_gt_classes": sum(v > 0 for v in gt_counts.values()),
            "nclass_bucket": class_bucket(sum(v > 0 for v in gt_counts.values())),
            "gt_bin": bin_of(gt_total),
            "n_candidates_pts32": int(d32["z"].shape[0]),
            "n_candidates_pts16": int(d16["z"].shape[0]),
        }
        for name in VARIANTS:
            row = dict(common)
            row["variant"] = name
            row["pred_counts"] = predictions[name]
            row["pred_total"] = int(sum(predictions[name].values()))
            row["t4_applied"] = False
            rows_by_variant[name].append(row)

        if (index + 1) % 50 == 0 or index + 1 == len(files):
            m6 = total_metrics(rows_by_variant["m6"])
            print(
                f"  [OmniCount {index+1}/{len(files)}] M6 MAE={m6['MAE']:.3f} "
                f"rate={(index+1)/max(time.time()-t0, 1e-6):.2f}/s",
                flush=True,
            )

    summaries = {}
    m6_pred = [r["pred_total"] for r in rows_by_variant["m6"]]
    gts = [r["gt_total"] for r in rows_by_variant["m6"]]
    for offset, name in enumerate(VARIANTS):
        rows = rows_by_variant[name]
        if len(rows) != len(files) or len({row["file_name"] for row in rows}) != len(files):
            raise RuntimeError(f"OmniCount {name} output is incomplete or has duplicate image IDs")
        summaries[name] = {
            "definition": DEFINITIONS[name],
            **compact_omni_summary(rows, class_names, args, args.seed + offset),
            "paired_vs_m6": paired_mae_delta_ci(
                [r["pred_total"] for r in rows],
                m6_pred,
                gts,
                args.bootstrap,
                args.seed + 100 + offset,
            ),
        }

    if len(files) == 1957:
        if zero_pts32:
            raise RuntimeError(f"unexpected OmniCount T4 trigger candidates: {zero_pts32}")
        mismatch = any(
            left["pred_counts"] != right["pred_counts"]
            or left["pred_total"] != right["pred_total"]
            for left, right in zip(rows_by_variant["m5"], rows_by_variant["m6"])
        )
        if mismatch:
            raise RuntimeError(
                "OmniCount M5 must equal M6 because the T4 trigger fires on 0/1957 images"
            )

    return {
        "config": {
            "dataset": "OmniCount-191 test full",
            "num_images": len(files),
            "num_classes": len(class_names),
            "protocol": "prompt-free predicted semantic groups; cached valid/matched_class unused",
            "cache_pts32": str(cache32),
            "cache_pts16": str(cache16),
            "conf_threshold": args.conf_threshold,
            "conf_threshold_source": (
                "0.1 is the pre-existing 2026-07-08 OmniCount full protocol; "
                "0.2 is retained only as a cross-domain sensitivity audit"
            ),
            "tau_inst": args.tau_inst,
            "tau_affinity": args.tau_affinity,
            "t4_triggered_images": 0,
        },
        "summaries": summaries,
        "per_image": rows_by_variant,
    }


def compact_dataset_summary(result: dict[str, Any], dataset: str) -> dict[str, Any]:
    variants = {}
    for name, summary in result["summaries"].items():
        if dataset == "fsc147":
            variants[name] = {
                "metrics": summary["metrics"],
                "paired_vs_m6": summary["paired_vs_m6"],
            }
        else:
            variants[name] = {
                "total": summary["total"],
                "present_class": summary["present_class"],
                "mRMSE": summary["mRMSE"],
                "mRMSE_nz": summary["mRMSE_nz"],
                "paired_vs_m6": summary["paired_vs_m6"],
            }
    return {"config": result["config"], "variants": variants}


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    print(f"[save] {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=("both", "fsc147", "omnicount"), default="both")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260710)
    ap.add_argument("--conf-threshold", type=float, default=0.2)
    ap.add_argument("--tau-inst", type=float, default=0.99)
    ap.add_argument("--tau-affinity", type=float, default=0.1)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--min-rep-conf", type=float, default=0.05)
    ap.add_argument("--max-rel-candidates", type=int, default=220)
    ap.add_argument(
        "--omni-cache-pts32", default="/home/czp/ws_yiyang/ovcud_cache/omnicount_test"
    )
    ap.add_argument(
        "--omni-cache-pts16", default="/home/czp/ws_yiyang/ovcud_cache/omnicount_test_pts16"
    )
    ap.add_argument(
        "--out-prefix", default="result/logs/fsc147_omnicount_leaveoneout_full"
    )
    args = ap.parse_args()

    models = load_models(args.device)
    outputs: dict[str, Any] = {}
    prefix = REPO / args.out_prefix
    suffix = "" if args.limit <= 0 else f"_limit{args.limit}"

    if args.dataset in {"both", "fsc147"}:
        outputs["fsc147"] = evaluate_fsc147(models, args)
        write_json(Path(f"{prefix}_fsc147{suffix}.json"), outputs["fsc147"])
    if args.dataset in {"both", "omnicount"}:
        outputs["omnicount"] = evaluate_omnicount(models, args)
        write_json(Path(f"{prefix}_omnicount{suffix}.json"), outputs["omnicount"])

    summary_path = Path(f"{prefix}_summary{suffix}.json")
    summary = {
        "date": "2026-07-10",
        "variant_definitions": DEFINITIONS,
        "checkpoint_sha256": {
            name: file_sha256(path) for name, path in CKPT.items() if name != "omni_classes"
        },
        "datasets": {
            name: compact_dataset_summary(result, name) for name, result in outputs.items()
        },
    }
    if summary_path.exists():
        previous = json.loads(summary_path.read_text())
        if previous.get("checkpoint_sha256") == summary["checkpoint_sha256"]:
            summary["datasets"] = {**previous.get("datasets", {}), **summary["datasets"]}
    write_json(summary_path, summary)

    print("\n=== Leave-one-out summary ===")
    for dataset, result in outputs.items():
        print(dataset)
        for name in VARIANTS:
            if dataset == "fsc147":
                m = result["summaries"][name]["metrics"]
                print(f"  {name.upper()}: MAE={m['MAE']:.3f} RMSE={m['RMSE']:.3f}")
            else:
                s = result["summaries"][name]
                print(
                    f"  {name.upper()}: total MAE={s['total']['MAE']:.3f} "
                    f"RMSE={s['total']['RMSE']:.3f} mRMSE={s['mRMSE']:.4f} "
                    f"mRMSE-nz={s['mRMSE_nz']:.4f}"
                )


if __name__ == "__main__":
    main()
