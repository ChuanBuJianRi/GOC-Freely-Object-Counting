"""Audit FSC-147 vocabulary leakage while retaining the historical GT oracle.

This is a diagnostic experiment, not a valid no-GT evaluation.  It reproduces
the FSC-147 12.67 path, including the GT-dot-derived ``valid`` candidate gate,
GT-selected multiresolution cache routing, the old relation heads, and the T4
rescue.  The controlled variants remove held-out class-name prototypes and,
in the strictest category-side variant, replace the old category heads with
official-train-only heads.

The key comparison is ``legacy_no_test118`` versus ``legacy_full147``: the
prototype tensors are identical row-for-row and only the 29 official-test
class rows are removed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from script.ablation_fsc147_multires_components import (  # noqa: E402
    CONF,
    DEFAULT_ANN,
    DEFAULT_SPLIT,
    TAU_AFF,
    TAU_INST,
    gt_count_from_ann,
    load_test_names,
)
from script.eval_fsc147_omnicount_leaveoneout import (  # noqa: E402
    FSC_BINS,
    FSC_CACHE,
    bootstrap_ci,
    file_sha256,
    fsc_predict,
    fsc_source,
    load_pt,
    paired_mae_delta_ci,
    scalar_metrics,
)
from script.eval_omnicount_multiclass_ablation import (  # noqa: E402
    load_category_head,
    load_relation_head,
)


VARIANTS = (
    "legacy_full147",
    "legacy_no_test118",
    "legacy_train89_subset",
    "trainonly_category_full147_oldproto",
    "trainonly_category_no_test118_oldproto",
    "trainonly_category_train89_oldproto",
    "trainonly_category_train89",
)
REFERENCE = "legacy_full147"

CKPT = {
    "legacy_category_fast": REPO / "result/checkpoints/category_cosine_fast.pt",
    "legacy_category_pts32": REPO / "result/checkpoints/category_cosine_pts32.pt",
    "train_category_fast": (
        REPO / "result/checkpoints/cp_strict/category_pts16_trainvocab.pt"
    ),
    "train_category_pts32": (
        REPO / "result/checkpoints/cp_strict/category_pts32_trainvocab.pt"
    ),
    "relation_fast": REPO / "result/checkpoints/fsc147_relation_best.pt",
    "relation_pts32": REPO / "result/checkpoints/fsc147_relation_pts32_best.pt",
    "prototype_full": REPO / "result/checkpoints/text_prototypes_fsc147.pt",
    "prototype_full_meta": (
        REPO / "result/checkpoints/text_prototypes_fsc147_categories.json"
    ),
    "prototype_train": (
        REPO / "result/checkpoints/text_prototypes_fsc147_train89.pt"
    ),
    "prototype_train_meta": (
        REPO / "result/checkpoints/text_prototypes_fsc147_train89.json"
    ),
}

DEFAULT_CLASS_MAP = Path(
    "/home/czp/official_code/dataset/FSC147/ImageClasses_FSC147.txt"
)
DEFAULT_OUTPUT = (
    REPO / "result/logs/fsc147_leaky_filter_vocab_ablation_full1190.json"
)

DEFINITIONS = {
    "legacy_full147": (
        "Historical 12.67 anchor: old category/relation heads and all 147 "
        "FSC class-name prototypes."
    ),
    "legacy_no_test118": (
        "Exact vocabulary-row ablation: old heads and old prototype vectors, "
        "with only the 29 official-test class rows removed (train+val 118 remain)."
    ),
    "legacy_train89_subset": (
        "Old heads and old prototype vectors, restricted to the 89 official-train "
        "class rows; no validation/test class-name row remains."
    ),
    "trainonly_category_full147_oldproto": (
        "Official-train-only category heads with all 147 rows from the old prototype "
        "bank. This is the same-head vocabulary reference, not a no-leak variant."
    ),
    "trainonly_category_no_test118_oldproto": (
        "Official-train-only category heads with the old prototype bank after exactly "
        "removing the 29 official-test rows."
    ),
    "trainonly_category_train89_oldproto": (
        "Official-train-only category heads with only the 89 official-train rows from "
        "the old prototype bank."
    ),
    "trainonly_category_train89": (
        "Official-train-only category heads and directly encoded train-89 "
        "prototypes; old relation heads and all historical GT-assisted inference "
        "components are retained."
    ),
}


def load_class_map(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        image_name, class_name = line.split("\t", 1)
        mapping[image_name] = class_name.strip()
    return mapping


def split_vocabulary_audit(
    split_path: Path, class_map_path: Path, full_meta_path: Path
) -> dict[str, Any]:
    split = json.loads(split_path.read_text())
    class_by_image = load_class_map(class_map_path)
    meta = json.loads(full_meta_path.read_text())
    categories = sorted(meta["categories"], key=lambda row: int(row["contiguous_id"]))
    if [int(row["contiguous_id"]) for row in categories] != list(range(147)):
        raise RuntimeError("full FSC prototype metadata is not contiguous 0..146")
    id_by_name = {row["name"]: int(row["contiguous_id"]) for row in categories}

    split_names: dict[str, list[str]] = {}
    split_ids: dict[str, list[int]] = {}
    for split_name in ("train", "val", "test"):
        missing_images = [name for name in split[split_name] if name not in class_by_image]
        if missing_images:
            raise RuntimeError(f"class map is missing {split_name} images: {missing_images[:5]}")
        names = sorted({class_by_image[name] for name in split[split_name]})
        missing_categories = [name for name in names if name not in id_by_name]
        if missing_categories:
            raise RuntimeError(
                f"prototype metadata is missing {split_name} classes: {missing_categories}"
            )
        split_names[split_name] = names
        split_ids[split_name] = sorted(id_by_name[name] for name in names)

    sets = {name: set(ids) for name, ids in split_ids.items()}
    if {name: len(ids) for name, ids in sets.items()} != {
        "train": 89,
        "val": 29,
        "test": 29,
    }:
        raise RuntimeError(f"unexpected FSC split class counts: {sets}")
    if (sets["train"] & sets["val"]) or (sets["train"] & sets["test"]) or (
        sets["val"] & sets["test"]
    ):
        raise RuntimeError("FSC official train/val/test classes are not disjoint")
    if sets["train"] | sets["val"] | sets["test"] != set(range(147)):
        raise RuntimeError("FSC official class split does not cover all 147 classes")

    return {
        "class_by_image": class_by_image,
        "id_to_name": {int(row["contiguous_id"]): row["name"] for row in categories},
        "split_class_ids": split_ids,
        "split_class_names": split_names,
        "pairwise_intersections": {
            "train_val": len(sets["train"] & sets["val"]),
            "train_test": len(sets["train"] & sets["test"]),
            "val_test": len(sets["val"] & sets["test"]),
        },
    }


def load_models_and_vocab(device: str, audit: dict[str, Any]) -> dict[str, Any]:
    for path in CKPT.values():
        if not path.exists():
            raise FileNotFoundError(f"missing required asset: {path}")

    full = F.normalize(
        torch.load(CKPT["prototype_full"], map_location=device, weights_only=False).float(),
        dim=-1,
    )
    train_direct = F.normalize(
        torch.load(CKPT["prototype_train"], map_location=device, weights_only=False).float(),
        dim=-1,
    )
    train_meta = json.loads(CKPT["prototype_train_meta"].read_text())
    train_ids = audit["split_class_ids"]["train"]
    test_ids = set(audit["split_class_ids"]["test"])
    no_test_ids = [index for index in range(147) if index not in test_ids]
    if train_meta["global_class_ids"] != train_ids:
        raise RuntimeError("train-89 prototype metadata disagrees with official split")
    if tuple(full.shape) != (147, 512) or tuple(train_direct.shape) != (89, 512):
        raise RuntimeError(
            f"unexpected prototype shape: full={tuple(full.shape)}, "
            f"train={tuple(train_direct.shape)}"
        )

    train_subset = full[torch.as_tensor(train_ids, device=full.device)]
    cosine = torch.sum(train_subset * train_direct, dim=-1)
    prototype_comparison = {
        "selected_vs_direct_max_abs": float((train_subset - train_direct).abs().max()),
        "selected_vs_direct_mean_abs": float((train_subset - train_direct).abs().mean()),
        "selected_vs_direct_cosine_mean": float(cosine.mean()),
        "selected_vs_direct_cosine_min": float(cosine.min()),
        "selected_vs_direct_cosine_max": float(cosine.max()),
        "exactly_equal": bool(torch.equal(train_subset, train_direct)),
    }

    return {
        "legacy_category_fast": load_category_head(
            str(CKPT["legacy_category_fast"]), device
        ),
        "legacy_category_pts32": load_category_head(
            str(CKPT["legacy_category_pts32"]), device
        ),
        "train_category_fast": load_category_head(
            str(CKPT["train_category_fast"]), device
        ),
        "train_category_pts32": load_category_head(
            str(CKPT["train_category_pts32"]), device
        ),
        "relation_fast": load_relation_head(str(CKPT["relation_fast"]), device),
        "relation_pts32": load_relation_head(str(CKPT["relation_pts32"]), device),
        "prototypes": {
            "legacy_full147": full,
            "legacy_no_test118": full[
                torch.as_tensor(no_test_ids, device=full.device)
            ],
            "legacy_train89_subset": train_subset,
            "trainonly_category_full147_oldproto": full,
            "trainonly_category_no_test118_oldproto": full[
                torch.as_tensor(no_test_ids, device=full.device)
            ],
            "trainonly_category_train89_oldproto": train_subset,
            "trainonly_category_train89": train_direct,
        },
        "prototype_ids": {
            "legacy_full147": list(range(147)),
            "legacy_no_test118": no_test_ids,
            "legacy_train89_subset": train_ids,
            "trainonly_category_full147_oldproto": list(range(147)),
            "trainonly_category_no_test118_oldproto": no_test_ids,
            "trainonly_category_train89_oldproto": train_ids,
            "trainonly_category_train89": train_ids,
        },
        "prototype_comparison": prototype_comparison,
    }


def category_head_for_variant(
    models: dict[str, Any], variant: str, source: str
):
    family = "train" if variant.startswith("trainonly_category_") else "legacy"
    return models[f"{family}_category_{source}"]


def per_count_bin(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for label, lo, hi in FSC_BINS:
        selected = [row for row in rows if lo <= row["gt_count"] <= hi]
        output[label] = scalar_metrics(
            [row["pred_count"] for row in selected],
            [row["gt_count"] for row in selected],
        )
    return output


def per_class_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["class_name"]].append(row)
    return {
        class_name: scalar_metrics(
            [row["pred_count"] for row in class_rows],
            [row["gt_count"] for row in class_rows],
        )
        for class_name, class_rows in sorted(grouped.items())
    }


def paired_diagnostics(
    rows: list[dict[str, Any]], reference_rows: list[dict[str, Any]], n_boot: int, seed: int
) -> dict[str, Any]:
    if [row["file_name"] for row in rows] != [
        row["file_name"] for row in reference_rows
    ]:
        raise RuntimeError("paired rows have different image order")
    pred = np.asarray([row["pred_count"] for row in rows], dtype=np.float64)
    ref = np.asarray([row["pred_count"] for row in reference_rows], dtype=np.float64)
    gt = np.asarray([row["gt_count"] for row in rows], dtype=np.float64)
    delta_pred = pred - ref
    delta_error = np.abs(pred - gt) - np.abs(ref - gt)
    changed = np.flatnonzero(delta_pred != 0)
    top = sorted(changed, key=lambda i: -abs(float(delta_pred[i])))[:20]
    output = paired_mae_delta_ci(
        pred.tolist(), ref.tolist(), gt.tolist(), n_boot, seed
    )
    output["delta_MAE_vs_reference"] = output.pop("delta_MAE_vs_M6")
    output.update(
        {
            "changed_images": int(len(changed)),
            "unchanged_images": int(len(rows) - len(changed)),
            "improved_images": int(np.sum(delta_error < 0)),
            "worsened_images": int(np.sum(delta_error > 0)),
            "equal_error_images": int(np.sum(delta_error == 0)),
            "prediction_increased": int(np.sum(delta_pred > 0)),
            "prediction_decreased": int(np.sum(delta_pred < 0)),
            "mean_prediction_delta": float(delta_pred.mean()),
            "max_abs_prediction_delta": float(np.abs(delta_pred).max()),
            "top_prediction_changes": [
                {
                    "file_name": rows[i]["file_name"],
                    "class_name": rows[i]["class_name"],
                    "gt_count": int(gt[i]),
                    "reference_pred": int(ref[i]),
                    "variant_pred": int(pred[i]),
                    "prediction_delta": int(delta_pred[i]),
                    "absolute_error_delta": int(delta_error[i]),
                }
                for i in top
            ],
        }
    )
    return output


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    audit = split_vocabulary_audit(DEFAULT_SPLIT, args.class_map, CKPT["prototype_full_meta"])
    models = load_models_and_vocab(args.device, audit)
    names = load_test_names(DEFAULT_SPLIT)
    if args.limit > 0:
        names = names[: args.limit]
    ann = json.loads(DEFAULT_ANN.read_text())
    rows_by_variant = {variant: [] for variant in VARIANTS}
    start = time.time()

    for index, image_name in enumerate(names):
        stem = Path(image_name).stem
        gt = gt_count_from_ann(stem, ann)
        d_fast = load_pt(FSC_CACHE["fast"] / f"{stem}.pt")
        d_final = load_pt(FSC_CACHE["final"] / f"{stem}.pt")
        fast_zero = d_fast is None or int(d_fast["z"].shape[0]) == 0
        d_rescue = load_pt(FSC_CACHE["rescue"] / f"{stem}.pt") if fast_zero else None
        rescue_applied = fast_zero and d_rescue is not None
        d_main = d_rescue if rescue_applied else d_final
        source, cache_source = fsc_source(stem)
        if rescue_applied:
            source = "pts32"
            cache_source = "rescue_4x4"

        relation_head = models[f"relation_{source}"]
        for variant in VARIANTS:
            pred, n_candidates, n_valid = fsc_predict(
                d_main,
                category_head_for_variant(models, variant, source),
                relation_head,
                models["prototypes"][variant],
                args.device,
                CONF,
                "relation",
            )
            rows_by_variant[variant].append(
                {
                    "file_name": image_name,
                    "class_name": audit["class_by_image"][image_name],
                    "gt_count": gt,
                    "pred_count": pred,
                    "n_candidates": n_candidates,
                    "n_valid": n_valid,
                    "cache_source": cache_source,
                    "t4_applied": rescue_applied,
                }
            )

        if (index + 1) % args.progress_every == 0 or index + 1 == len(names):
            ref_rows = rows_by_variant[REFERENCE]
            metrics = scalar_metrics(
                [row["pred_count"] for row in ref_rows],
                [row["gt_count"] for row in ref_rows],
            )
            print(
                f"[{index + 1}/{len(names)}] {REFERENCE} MAE={metrics['MAE']:.4f} "
                f"rate={(index + 1) / max(time.time() - start, 1e-6):.2f} image/s",
                flush=True,
            )

    reference_rows = rows_by_variant[REFERENCE]
    reference_pred = [row["pred_count"] for row in reference_rows]
    gt = [row["gt_count"] for row in reference_rows]
    summaries = {}
    for offset, variant in enumerate(VARIANTS):
        rows = rows_by_variant[variant]
        if len(rows) != len(names) or len({row["file_name"] for row in rows}) != len(names):
            raise RuntimeError(f"{variant} is incomplete or contains duplicate image IDs")
        pred = [row["pred_count"] for row in rows]
        summaries[variant] = {
            "definition": DEFINITIONS[variant],
            "prototype_count": len(models["prototype_ids"][variant]),
            "prototype_global_ids": models["prototype_ids"][variant],
            "test_class_rows_retained": len(
                set(models["prototype_ids"][variant])
                & set(audit["split_class_ids"]["test"])
            ),
            "validation_class_rows_retained": len(
                set(models["prototype_ids"][variant])
                & set(audit["split_class_ids"]["val"])
            ),
            "metrics": scalar_metrics(pred, gt),
            "per_gt_count_bin": per_count_bin(rows),
            "per_test_class": per_class_metrics(rows),
            "bootstrap": bootstrap_ci(pred, gt, args.bootstrap, args.seed + offset),
            "paired_vs_legacy_full147": paired_diagnostics(
                rows,
                reference_rows,
                args.bootstrap,
                args.seed + 100 + offset,
            ),
        }
        if variant.startswith("trainonly_category_"):
            summaries[variant]["paired_vs_same_head_full147_oldproto"] = (
                paired_diagnostics(
                    rows,
                    rows_by_variant["trainonly_category_full147_oldproto"],
                    args.bootstrap,
                    args.seed + 200 + offset,
                )
            )

    if args.limit <= 0:
        anchor = summaries[REFERENCE]["metrics"]
        expected_mae = 12.668067226890756
        expected_rmse = 113.71112226924458
        if abs(float(anchor["MAE"]) - expected_mae) > 1e-9 or abs(
            float(anchor["RMSE"]) - expected_rmse
        ) > 1e-9:
            raise RuntimeError(f"historical 12.67 anchor mismatch: {anchor}")

    serializable_audit = {key: value for key, value in audit.items() if key != "class_by_image"}
    serializable_audit["id_to_name"] = {
        str(key): value for key, value in audit["id_to_name"].items()
    }
    return {
        "experiment": (
            "FSC-147 historical GT-assisted vocabulary leakage ablation; "
            "diagnostic only, not a no-GT result"
        ),
        "config": {
            "dataset": "FSC-147 official full test",
            "num_images": len(names),
            "device": args.device,
            "protocol": (
                "Retains GT-dot-derived valid gate, GT-selected MR cache routing, "
                "old relation heads, confidence filtering, and fast-zero T4 rescue"
            ),
            "conf_threshold": CONF,
            "tau_inst": TAU_INST,
            "tau_affinity": TAU_AFF,
            "t4_trigger": "fast raw candidate count == 0",
            "reference_variant": REFERENCE,
            "bootstrap_resamples": args.bootstrap,
            "bootstrap_seed": args.seed,
        },
        "definitions": DEFINITIONS,
        "vocabulary_audit": serializable_audit,
        "prototype_comparison": models["prototype_comparison"],
        "assets": {
            str(key): {
                "path": str(path),
                "sha256": file_sha256(path),
            }
            for key, path in CKPT.items()
        },
        "cache_paths": {key: str(path) for key, path in FSC_CACHE.items()},
        "summaries": summaries,
        "per_image": rows_by_variant,
        "elapsed_seconds": time.time() - start,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--class-map", type=Path, default=DEFAULT_CLASS_MAP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.output}", flush=True)
    for variant in VARIANTS:
        metrics = result["summaries"][variant]["metrics"]
        paired = result["summaries"][variant]["paired_vs_legacy_full147"]
        print(
            f"{variant}: MAE={metrics['MAE']:.6f} RMSE={metrics['RMSE']:.6f} "
            f"bias={metrics['bias']:.6f} changed={paired['changed_images']} "
            f"delta_MAE={paired['delta_MAE_vs_reference']:+.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
