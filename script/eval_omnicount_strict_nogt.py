"""Evaluate the FSC strict no-GT frozen model on OmniCount test-full.

All 1,957 predictions are written to a prediction-only artifact before the
isolated OmniCount count target shard is opened. No OmniCount class name,
count, box, or annotation is available to the prediction loop.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from script.eval_fsc147_strict_nogt import (  # noqa: E402
    SEEDS,
    assert_exact_cache,
    bootstrap,
    code_asset_sha256,
    count_prepared,
    load_models,
    load_safe_cache,
    metric,
    prepare_sample,
    sha256,
)
from script.export_omnicount_strict_protocol import (  # noqa: E402
    TARGET_SCHEMA,
    read_manifest,
)


DEFAULT_CACHE_ROOT = Path("/home/czp/ws_yiyang/ovcud_cache")
EXPECTED_TILED_SOURCE = "tiled-nogt-2x2-overlap0.25-bbox"


def total_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    return metric(
        [float(row["pred_count"]) for row in rows],
        [float(row["gt_count"]) for row in rows],
    )


def grouped_metrics(
    rows: list[dict[str, Any]], key: Callable[[dict[str, Any]], str]
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[key(row)].append(row)
    return {
        label: total_metrics(selected)
        for label, selected in sorted(grouped.items())
    }


def class_count_bucket(row: dict[str, Any]) -> str:
    count = int(row["gt_class_count"])
    return "4+" if count >= 4 else str(count)


def gt_count_bucket(row: dict[str, Any]) -> str:
    count = int(row["gt_count"])
    if count <= 5:
        return "01-05"
    if count <= 10:
        return "06-10"
    if count <= 20:
        return "11-20"
    if count <= 50:
        return "21-50"
    return "51+"


def candidate_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for key in ("raw_fast_candidates", "raw_tiled_candidates"):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        result[key] = {
            "sum": int(values.sum()),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9)),
            "max": int(values.max()),
            "zero": int(np.sum(values == 0)),
        }
    return result


def load_targets_after_predictions(
    target_path: Path, manifest_path: Path, sample_ids: list[str]
) -> dict[str, dict[str, Any]]:
    payload = json.loads(target_path.read_text())
    targets = payload.get("targets", {})
    if (
        payload.get("schema") != TARGET_SCHEMA
        or payload.get("split") != "test"
        or payload.get("image_count") != len(sample_ids)
        or payload.get("manifest_sha256") != sha256(manifest_path)
        or payload.get("prediction_process_must_not_load_this_file") is not True
        or set(targets) != set(sample_ids)
    ):
        raise RuntimeError(f"invalid or non-exact isolated target shard: {target_path}")
    return targets


def summarize_run(
    rows: list[dict[str, Any]], bootstrap_iterations: int, bootstrap_seed: int
) -> dict[str, Any]:
    worst = sorted(
        rows,
        key=lambda row: (-abs(row["pred_count"] - row["gt_count"]), row["sample_id"]),
    )[:20]
    return {
        "metrics": total_metrics(rows),
        "bootstrap": bootstrap(rows, bootstrap_iterations, bootstrap_seed),
        "per_gt_class_count": grouped_metrics(rows, class_count_bucket),
        "single_vs_multi": grouped_metrics(
            rows, lambda row: "single-class" if row["gt_class_count"] == 1 else "multi-class"
        ),
        "per_supercategory": grouped_metrics(rows, lambda row: row["supercategory"]),
        "per_gt_count_bin": grouped_metrics(rows, gt_count_bucket),
        "routing": dict(Counter(row["source"] for row in rows)),
        "worst_20": [
            {
                "sample_id": row["sample_id"],
                "file_name": row["file_name"],
                "supercategory": row["supercategory"],
                "gt_class_count": row["gt_class_count"],
                "gt_count": row["gt_count"],
                "pred_count": row["pred_count"],
                "signed_error": row["pred_count"] - row["gt_count"],
                "raw_tiled_candidates": row["raw_tiled_candidates"],
            }
            for row in worst
        ],
    }


def omnicount_code_sha256() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        REPO / "script/preprocess_omnicount_tiled_nogt.py",
        REPO / "script/export_omnicount_strict_protocol.py",
        REPO / "script/export_omnicount_inference_cache.py",
    )
    return {str(path.relative_to(REPO)): sha256(path) for path in paths}


def run_predict(args: argparse.Namespace) -> None:
    manifest = read_manifest(args.manifest)
    entries = manifest["entries"]
    if len(entries) != args.expected_images:
        raise RuntimeError(f"expected {args.expected_images} images, found {len(entries)}")
    names = [row["file_name"] for row in entries]
    assert_exact_cache(args.fast_cache, names)
    assert_exact_cache(args.tiled_cache, names)

    frozen = json.loads(args.frozen_config.read_text())
    if (
        frozen.get("frozen") is not True
        or frozen.get("protocol", {}).get("test_read") is not False
        or frozen.get("selected", {}).get("route") != {"mode": "always_tiled"}
    ):
        raise RuntimeError("this evaluator requires the FSC validation-frozen always-tiled config")
    models = load_models(args.checkpoint_dir, args.device)
    if models["sha256"] != frozen.get("assets", {}).get("sha256"):
        raise RuntimeError("model hashes differ from the FSC-147 strict frozen configuration")
    if code_asset_sha256() != frozen.get("assets", {}).get("code_sha256"):
        raise RuntimeError("shared prediction code differs from the FSC-147 frozen configuration")

    tiled_config = frozen["selected"]["tiled_config"]
    prediction_rows: dict[str, list[dict[str, Any]]] = {str(seed): [] for seed in SEEDS}
    fast_zero_names = []
    started = time.time()
    for index, entry in enumerate(entries):
        stem = entry["sample_id"]
        fast = load_safe_cache(args.fast_cache / f"{stem}.pt")
        tiled = load_safe_cache(args.tiled_cache / f"{stem}.pt")
        if fast["file_name"] != entry["file_name"] or tiled["file_name"] != entry["file_name"]:
            raise RuntimeError(f"manifest/cache filename mismatch for {stem}")
        if tiled.get("source_cache") != EXPECTED_TILED_SOURCE:
            raise RuntimeError(f"unexpected tiled recipe for {stem}: {tiled.get('source_cache')}")
        if len(fast["z"]) == 0:
            fast_zero_names.append(entry["file_name"])
            continue

        prepared = prepare_sample(tiled, models, "tiled", args.device)
        base_row = {
            "sample_id": stem,
            "file_name": entry["file_name"],
            "source": "tiled2x2",
            "raw_fast_candidates": int(len(fast["z"])),
            "raw_tiled_candidates": int(prepared["n"]),
        }
        for seed in SEEDS:
            prediction_rows[str(seed)].append(
                base_row | {"pred_count": count_prepared(prepared, seed, tiled_config)}
            )
        if (index + 1) % 100 == 0:
            print(
                f"[prediction] {index+1}/{len(entries)} "
                f"elapsed={(time.time()-started)/60:.1f}m",
                flush=True,
            )

    if fast_zero_names:
        raise RuntimeError(
            "the FSC main method requires a train-selected 4x4 rescue for raw-fast-zero "
            f"images; generate that cache before evaluation: {fast_zero_names}"
        )
    if any(len(rows) != len(entries) for rows in prediction_rows.values()):
        raise RuntimeError("prediction rows are incomplete")

    primary_seed = int(frozen["selected"]["primary_seed"])
    # This process has no target-shard argument and persists image-only predictions only.
    prediction_payload = {
        "schema": "omnicount-strict-prediction-only-v1",
        "date": "2026-07-11",
        "dataset": "OmniCount-191 test full",
        "image_count": len(entries),
        "manifest_path": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "frozen_config_path": str(args.frozen_config),
        "frozen_config_sha256": sha256(args.frozen_config),
        "prediction_gt_fields": [],
        "target_file_loaded": False,
        "tiled_config": tiled_config,
        "route": frozen["selected"]["route"],
        "primary_seed": primary_seed,
        "protocol": {
            "same_model_as_fsc147_strict_mae_26p4992": True,
            "fsc147_reference": {
                "images": 1190,
                "MAE": 26.499159663865544,
                "RMSE": 129.68557969115892,
            },
            "prediction_input": "image only",
            "prediction_gt_fields": [],
            "omnicount_threshold_tuning": False,
            "selection_split": "FSC-147 official validation only",
            "text_vocabulary": "fixed 89 FSC-147 official-train classes; no OmniCount class names",
            "relation_initialization": "scratch; COCO relation pretraining forbidden",
            "frontend": "pts32 2x2 tiles, overlap=0.25, bbox merge",
        },
        "candidate_stats": candidate_stats(prediction_rows[str(primary_seed)]),
        "assets": {
            "model_paths": models["paths"],
            "model_sha256": models["sha256"],
            "frozen_config_path": str(args.frozen_config),
            "frozen_config_sha256": sha256(args.frozen_config),
            "shared_code_sha256": code_asset_sha256(),
            "omnicount_code_sha256": omnicount_code_sha256(),
            "manifest_path": str(args.manifest),
            "manifest_sha256": sha256(args.manifest),
            "fast_cache": str(args.fast_cache),
            "tiled_cache": str(args.tiled_cache),
        },
        "rows": prediction_rows,
        "elapsed_seconds": time.time() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(prediction_payload, ensure_ascii=False, indent=2) + "\n"
    )
    prediction_sha = sha256(args.out)
    print(f"[prediction-only] sha256={prediction_sha} -> {args.out}", flush=True)
    print(json.dumps({
        "images": len(entries),
        "primary_seed": primary_seed,
        "mean_prediction": float(np.mean([
            row["pred_count"] for row in prediction_rows[str(primary_seed)]
        ])),
        "fast_zero": 0,
    }, indent=2))


def run_score(args: argparse.Namespace) -> None:
    started = time.time()
    manifest = read_manifest(args.manifest)
    entries = manifest["entries"]
    if len(entries) != args.expected_images:
        raise RuntimeError(f"expected {args.expected_images} images, found {len(entries)}")
    sample_ids = [row["sample_id"] for row in entries]
    predictions = json.loads(args.predictions.read_text())
    prediction_rows = predictions.get("rows", {})
    if (
        predictions.get("schema") != "omnicount-strict-prediction-only-v1"
        or predictions.get("image_count") != len(entries)
        or predictions.get("manifest_sha256") != sha256(args.manifest)
        or predictions.get("prediction_gt_fields") != []
        or predictions.get("target_file_loaded") is not False
        or set(prediction_rows) != {str(seed) for seed in SEEDS}
    ):
        raise RuntimeError(f"invalid prediction-only artifact: {args.predictions}")
    for seed in SEEDS:
        rows = prediction_rows[str(seed)]
        if (
            len(rows) != len(entries)
            or [row["sample_id"] for row in rows] != sample_ids
            or any(set(row) != {
                "sample_id", "file_name", "source", "raw_fast_candidates",
                "raw_tiled_candidates", "pred_count",
            } for row in rows)
        ):
            raise RuntimeError(f"non-exact or unsafe prediction rows for seed {seed}")

    # Only this scoring process accepts and opens the isolated target shard.
    targets = load_targets_after_predictions(args.targets, args.manifest, sample_ids)
    runs = {}
    joined_rows: dict[str, list[dict[str, Any]]] = {}
    for seed in SEEDS:
        key = str(seed)
        rows = []
        for prediction in prediction_rows[key]:
            target = targets[prediction["sample_id"]]
            rows.append(prediction | {
                "gt_count": int(target["gt_total"]),
                "gt_class_count": int(target["gt_class_count"]),
                "supercategory": str(target["supercategory"]),
            })
        joined_rows[key] = rows
        runs[key] = summarize_run(rows, args.bootstrap, args.bootstrap_seed + seed)

    primary_seed = int(predictions["primary_seed"])
    primary_key = str(primary_seed)
    maes = np.asarray([runs[str(seed)]["metrics"]["MAE"] for seed in SEEDS])
    rmses = np.asarray([runs[str(seed)]["metrics"]["RMSE"] for seed in SEEDS])
    output = {
        "date": "2026-07-11",
        "dataset": "OmniCount-191 test full",
        "protocol": {
            **predictions["protocol"],
            "omnicount_annotations_loaded": "score process only, after prediction process exited",
            "process_isolation": "predict CLI has no target-shard argument",
            "route": predictions["route"],
            "tiled_config": predictions["tiled_config"],
            "classwise_metrics": (
                "not reported: the fixed FSC train-89 ontology is not the OmniCount 93-class "
                "ontology; reporting mRMSE/mRMSE-nz would require an OmniCount class list/prompt "
                "or a separately declared anonymous matching protocol"
            ),
        },
        "image_count": len(entries),
        "primary_seed": primary_seed,
        "primary": runs[primary_key],
        "aggregate": {
            "MAE_mean": float(maes.mean()),
            "MAE_std": float(maes.std(ddof=1)),
            "RMSE_mean": float(rmses.mean()),
            "RMSE_std": float(rmses.std(ddof=1)),
        },
        "runs": runs,
        "candidate_stats": predictions["candidate_stats"],
        "assets": predictions["assets"] | {
            "target_path": str(args.targets),
            "target_sha256": sha256(args.targets),
            "prediction_only_path": str(args.predictions),
            "prediction_only_sha256": sha256(args.predictions),
        },
        "elapsed_seconds": time.time() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "primary_seed": primary_seed,
        "primary": output["primary"]["metrics"],
        "aggregate": output["aggregate"],
    }, indent=2))
    print(f"[save] {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    predict = subparsers.add_parser("predict")
    predict.add_argument("--manifest", type=Path, required=True)
    predict.add_argument(
        "--fast-cache", type=Path,
        default=DEFAULT_CACHE_ROOT / "omnicount_strict_nogt_fast",
    )
    predict.add_argument(
        "--tiled-cache", type=Path,
        default=DEFAULT_CACHE_ROOT / "omnicount_strict_nogt_tiled2x2",
    )
    predict.add_argument(
        "--checkpoint-dir", type=Path,
        default=REPO / "result/checkpoints/cp_strict",
    )
    predict.add_argument(
        "--frozen-config", type=Path,
        default=REPO / "result/logs/fsc147_strict_nogt_val_selection.json",
    )
    predict.add_argument("--expected-images", type=int, default=1957)
    predict.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    predict.add_argument("--out", type=Path, required=True)

    score = subparsers.add_parser("score")
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--targets", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--expected-images", type=int, default=1957)
    score.add_argument("--bootstrap", type=int, default=5000)
    score.add_argument("--bootstrap-seed", type=int, default=20260711)
    score.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    if args.phase == "predict":
        run_predict(args)
    else:
        run_score(args)


if __name__ == "__main__":
    main()
