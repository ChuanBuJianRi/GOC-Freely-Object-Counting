"""Strict no-GT FSC-147 validation selection and frozen full-test evaluation.

Prediction functions only receive inference-v1 caches containing image-derived
features. Ground-truth annotations are joined after every prediction is frozen.
Relation heads must be FSC-dot-supervised scratch runs with no COCO provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from script import ablation_fsc147_multires_components as ab  # noqa: E402
from script.eval_carpk import get_category_probs  # noqa: E402
from script.eval_omnicount_multiclass_ablation import (  # noqa: E402
    load_category_head,
    load_relation_head,
)
from script.train_candidate_filter import (  # noqa: E402
    CandidateFilter,
    bbox_features,
)


DEFAULT_SPLIT = Path("/home/czp/official_code/dataset/FSC147/Train_Test_Val_FSC_147.json")
DEFAULT_ANN = Path("/home/czp/official_code/dataset/FSC147/annotation_FSC147_384.json")
DEFAULT_CACHE_ROOT = Path("/home/czp/ws_yiyang/ovcud_cache")
SEEDS = (17, 42, 73)
SAFE_FIELDS = {"schema", "img_id", "file_name", "z", "bbox", "height", "width", "source_cache"}
FORBIDDEN_FIELDS = {
    "valid", "purity", "coverage", "iou", "matched_class", "matched_instance_id",
    "gt_count", "class_name", "is_part", "is_countable", "points",
}
GT_BINS = (("0-10", 0, 10), ("11-20", 11, 20), ("21-50", 21, 50),
           ("51-100", 51, 100), ("100+", 101, 10**9))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_key(config: dict[str, float]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def load_safe_cache(path: Path) -> dict[str, Any]:
    d = torch.load(path, map_location="cpu", weights_only=False)
    unexpected = set(d) - SAFE_FIELDS
    if d.get("schema") != "fsc147-inference-v1" or unexpected or set(d) & FORBIDDEN_FIELDS:
        raise RuntimeError(
            f"unsafe inference cache {path}: unexpected={sorted(unexpected)} "
            f"forbidden={sorted(set(d) & FORBIDDEN_FIELDS)}"
        )
    z = d["z"].float()
    bbox = d["bbox"].float()
    if z.ndim != 2 or bbox.shape != (len(z), 4):
        raise RuntimeError(f"invalid tensors in {path}")
    return d


def assert_exact_cache(cache_dir: Path, names: list[str]) -> None:
    expected = {Path(name).stem for name in names}
    actual = {path.stem for path in cache_dir.glob("*.pt")}
    if expected != actual:
        raise RuntimeError(
            f"cache must exactly match split: {cache_dir}; "
            f"missing={sorted(expected-actual)} extra={sorted(actual-expected)}"
        )


def load_filter(path: Path, device: str) -> CandidateFilter:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    manifest = checkpoint.get("data_manifest", {})
    if manifest.get("official_split") != "train" or manifest.get("test_images_loaded") != 0:
        raise RuntimeError(f"candidate filter is not official-train-only: {path}")
    model = CandidateFilter(
        z_dim=checkpoint["z_dim"], geom_dim=checkpoint["geom_dim"],
        hidden_dim=checkpoint["hidden_dim"], dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["candidate_filter"])
    return model.to(device).eval()


def load_models(checkpoint_dir: Path, device: str) -> dict[str, Any]:
    paths = {
        "category_fast": checkpoint_dir / "category_pts16_trainonly.pt",
        "category_tiled": checkpoint_dir / "category_pts32_trainonly.pt",
        "filter_fast": REPO / "result/checkpoints/fsc147_candidate_filter_pts16_trainonly.pt",
        "filter_tiled": REPO / "result/checkpoints/fsc147_candidate_filter_pts32_trainonly.pt",
        "prototypes": REPO / "result/checkpoints/text_prototypes_fsc147.pt",
    }
    for seed in SEEDS:
        paths[f"relation_fast_{seed}"] = checkpoint_dir / f"relation_pts16_scratch_seed{seed}.pt"
        paths[f"relation_tiled_{seed}"] = checkpoint_dir / f"relation_pts32_scratch_seed{seed}.pt"
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    for name in ("category_fast", "category_tiled"):
        checkpoint = torch.load(paths[name], map_location="cpu", weights_only=False)
        if checkpoint.get("data_manifest", {}).get("official_split", {}).get("split_key") != "train":
            raise RuntimeError(f"category head is not official-train-only: {paths[name]}")
    for seed in SEEDS:
        for resolution in ("fast", "tiled"):
            path = paths[f"relation_{resolution}_{seed}"]
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if checkpoint.get("pretrained_from"):
                raise RuntimeError(f"COCO-pretrained relation checkpoint is forbidden: {path}")
            if checkpoint.get("data_manifest", {}).get("official_split", {}).get("split_key") != "train":
                raise RuntimeError(f"relation head is not official-train-only: {path}")

    models: dict[str, Any] = {
        "category_fast": load_category_head(str(paths["category_fast"]), device),
        "category_tiled": load_category_head(str(paths["category_tiled"]), device),
        "filter_fast": load_filter(paths["filter_fast"], device),
        "filter_tiled": load_filter(paths["filter_tiled"], device),
        "prototypes": F.normalize(
            torch.load(paths["prototypes"], map_location=device, weights_only=False).float(), dim=-1
        ),
        "paths": {name: str(path) for name, path in paths.items()},
        "sha256": {name: sha256(path) for name, path in paths.items()},
    }
    for seed in SEEDS:
        models[f"relation_fast_{seed}"] = load_relation_head(
            str(paths[f"relation_fast_{seed}"]), device
        )
        models[f"relation_tiled_{seed}"] = load_relation_head(
            str(paths[f"relation_tiled_{seed}"]), device
        )
    return models


@torch.no_grad()
def prepare_sample(d: dict[str, Any], models: dict[str, Any], resolution: str,
                   device: str) -> dict[str, Any]:
    z = d["z"].float()
    bbox_t = d["bbox"].float()
    bbox = bbox_t.numpy()
    n = len(z)
    if n == 0:
        return {"n": 0, "file_name": d["file_name"]}
    category = models[f"category_{resolution}"]
    candidate_filter = models[f"filter_{resolution}"]
    probs = get_category_probs(category, z, models["prototypes"], device)
    category_confidence = probs.max(axis=1)
    geometry = bbox_features(bbox_t, int(d["height"]), int(d["width"]))
    filter_probability = torch.sigmoid(
        candidate_filter(z.to(device), geometry.to(device))
    ).cpu().numpy()
    relation = {}
    for seed in SEEDS:
        relation[seed] = ab.relation_matrices(
            d, z, probs, category_confidence,
            models[f"relation_{resolution}_{seed}"], device,
        )
    return {
        "n": n,
        "file_name": d["file_name"],
        "probs": probs,
        "category_confidence": category_confidence,
        "filter_probability": filter_probability,
        "bbox": bbox,
        "image_area": float(int(d["height"]) * int(d["width"])),
        "semantic_affinity": ab.semantic_affinity(probs),
        "relation": relation,
    }


def count_prepared(prepared: dict[str, Any], seed: int, config: dict[str, float]) -> int:
    n = prepared["n"]
    if n == 0:
        return 0
    indices = [
        index for index in range(n)
        if prepared["filter_probability"][index] >= config["filter_threshold"]
        and prepared["category_confidence"][index] >= config["category_threshold"]
    ]
    if not indices:
        return 0
    local_groups = ab.cluster_full(
        prepared["probs"], prepared["semantic_affinity"], prepared["bbox"],
        indices, prepared["image_area"],
    )
    groups = [[indices[index] for index in group] for group in local_groups]
    previous = ab.TAU_INST
    try:
        ab.TAU_INST = config["tau_inst"]
        return int(ab.dedup_count(
            groups, prepared["relation"][seed], prepared["probs"], prepared["bbox"],
            prepared["image_area"], n, adaptive=True,
        ))
    finally:
        ab.TAU_INST = previous


def metric(predictions: list[float], targets: list[float]) -> dict[str, float | int]:
    pred = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    error = pred - target
    return {
        "n": int(len(target)),
        "MAE": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt(np.square(error).mean())),
        "bias": float(error.mean()),
        "mean_pred": float(pred.mean()),
        "mean_gt": float(target.mean()),
    }


def load_targets(names: list[str], annotation_path: Path) -> list[int]:
    # This function is intentionally separate from every prediction function.
    annotation = json.loads(annotation_path.read_text())
    return [ab.gt_count_from_ann(Path(name).stem, annotation) for name in names]


def grid() -> list[dict[str, float]]:
    return [
        {"filter_threshold": filter_threshold, "category_threshold": category_threshold,
         "tau_inst": tau_inst}
        for filter_threshold in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4)
        for category_threshold in (0.0, 0.1, 0.2, 0.3)
        for tau_inst in (0.99, 0.999)
    ]


def validation_frontend_predictions(names: list[str], cache_dir: Path, models: dict[str, Any],
                                    resolution: str, device: str) -> tuple[dict[str, Any], list[int]]:
    configurations = grid()
    predictions = {
        config_key(config): {str(seed): [] for seed in SEEDS}
        for config in configurations
    }
    raw_candidates = []
    started = time.time()
    for index, name in enumerate(names):
        d = load_safe_cache(cache_dir / f"{Path(name).stem}.pt")
        prepared = prepare_sample(d, models, resolution, device)
        raw_candidates.append(prepared["n"])
        for config in configurations:
            key = config_key(config)
            for seed in SEEDS:
                predictions[key][str(seed)].append(count_prepared(prepared, seed, config))
        if (index + 1) % 250 == 0:
            print(
                f"[{resolution}] validation {index+1}/{len(names)} "
                f"elapsed={(time.time()-started)/60:.1f}m", flush=True,
            )
    return predictions, raw_candidates


def summarize_configs(predictions: dict[str, Any], targets: list[int]) -> dict[str, Any]:
    summary = {}
    for key, per_seed in predictions.items():
        seed_metrics = {seed: metric(pred, targets) for seed, pred in per_seed.items()}
        summary[key] = {
            "seed_metrics": seed_metrics,
            "MAE_mean": float(np.mean([row["MAE"] for row in seed_metrics.values()])),
            "RMSE_mean": float(np.mean([row["RMSE"] for row in seed_metrics.values()])),
        }
    return summary


def select_config(summary: dict[str, Any]) -> str:
    return min(
        summary,
        key=lambda key: (
            summary[key]["MAE_mean"], summary[key]["RMSE_mean"],
            json.loads(key)["filter_threshold"],
            json.loads(key)["category_threshold"],
            abs(json.loads(key)["tau_inst"] - 0.99),
        ),
    )


def select_route(fast_predictions: dict[str, list[int]], tiled_predictions: dict[str, list[int]],
                 targets: list[int]) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates: dict[str, Any] = {}
    policies: list[dict[str, Any]] = [
        {"mode": "always_fast"}, {"mode": "always_tiled"},
    ] + [{"mode": "predicted_count", "threshold": threshold} for threshold in (10, 20, 30, 40, 50, 75, 100)]
    for policy in policies:
        seed_metrics = {}
        seed_predictions = {}
        seed_routed = {}
        for seed in SEEDS:
            key = str(seed)
            combined = []
            routed = 0
            for fast, tiled in zip(fast_predictions[key], tiled_predictions[key]):
                use_tiled = policy["mode"] == "always_tiled" or (
                    policy["mode"] == "predicted_count" and fast >= policy["threshold"]
                )
                combined.append(tiled if use_tiled else fast)
                routed += int(use_tiled)
            seed_predictions[key] = combined
            seed_metrics[key] = metric(combined, targets)
            seed_routed[key] = routed
        policy_key = json.dumps(policy, sort_keys=True, separators=(",", ":"))
        candidates[policy_key] = {
            "policy": policy,
            "seed_metrics": seed_metrics,
            "seed_routed_tiled": seed_routed,
            "MAE_mean": float(np.mean([row["MAE"] for row in seed_metrics.values()])),
            "RMSE_mean": float(np.mean([row["RMSE"] for row in seed_metrics.values()])),
            "predictions": seed_predictions,
        }
    selected_key = min(
        candidates,
        key=lambda key: (candidates[key]["MAE_mean"], candidates[key]["RMSE_mean"], key),
    )
    return candidates[selected_key]["policy"], candidates


def run_validation(args: argparse.Namespace) -> None:
    split = json.loads(args.split_file.read_text())
    names = list(split["val"])
    assert_exact_cache(args.val_fast_cache, names)
    assert_exact_cache(args.val_tiled_cache, names)
    models = load_models(args.checkpoint_dir, args.device)

    fast_predictions, fast_raw = validation_frontend_predictions(
        names, args.val_fast_cache, models, "fast", args.device
    )
    tiled_predictions, tiled_raw = validation_frontend_predictions(
        names, args.val_tiled_cache, models, "tiled", args.device
    )
    # GT is loaded only after all validation predictions have been produced.
    targets = load_targets(names, args.annotation)
    fast_summary = summarize_configs(fast_predictions, targets)
    tiled_summary = summarize_configs(tiled_predictions, targets)
    fast_key = select_config(fast_summary)
    tiled_key = select_config(tiled_summary)
    route, route_sweep = select_route(
        fast_predictions[fast_key], tiled_predictions[tiled_key], targets
    )
    selected_route_key = json.dumps(route, sort_keys=True, separators=(",", ":"))
    output = {
        "date": "2026-07-11",
        "frozen": True,
        "protocol": {
            "prediction_gt_fields": [],
            "inference_cache_schema": "fsc147-inference-v1",
            "training_labels": "official-train FSC dots only",
            "relation_initialization": "scratch; COCO pretraining forbidden",
            "selection_split": "official val 1286",
            "test_read": False,
            "seeds": list(SEEDS),
            "grid": grid(),
            "route_grid": [row["policy"] for row in route_sweep.values()],
            "t4_policy": "if raw fast candidate count is zero, use predeclared 4x4 image-only cache",
        },
        "selected": {
            "fast_config": json.loads(fast_key),
            "tiled_config": json.loads(tiled_key),
            "route": route,
            "validation_metrics": route_sweep[selected_route_key]["seed_metrics"],
            "validation_MAE_mean": route_sweep[selected_route_key]["MAE_mean"],
            "validation_RMSE_mean": route_sweep[selected_route_key]["RMSE_mean"],
            "validation_routed_tiled": route_sweep[selected_route_key]["seed_routed_tiled"],
        },
        "frontend_summary": {"fast": fast_summary, "tiled": tiled_summary},
        "route_sweep": {
            key: {name: value for name, value in row.items() if name != "predictions"}
            for key, row in route_sweep.items()
        },
        "raw_candidate_stats": {
            "fast_mean": float(np.mean(fast_raw)), "tiled_mean": float(np.mean(tiled_raw)),
            "fast_zero": int(np.sum(np.asarray(fast_raw) == 0)),
            "tiled_zero": int(np.sum(np.asarray(tiled_raw) == 0)),
        },
        "assets": {"paths": models["paths"], "sha256": models["sha256"]},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(f"[selected] {json.dumps(output['selected'], ensure_ascii=False, indent=2)}")
    print(f"[save] {args.out}")


def per_bin(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for label, low, high in GT_BINS:
        selected = [row for row in rows if low <= row["gt_count"] <= high]
        result[label] = metric(
            [row["pred_count"] for row in selected], [row["gt_count"] for row in selected]
        )
    return result


def bootstrap(rows: list[dict[str, Any]], iterations: int, seed: int) -> dict[str, Any]:
    pred = np.asarray([row["pred_count"] for row in rows], dtype=np.float64)
    gt = np.asarray([row["gt_count"] for row in rows], dtype=np.float64)
    error = pred - gt
    rng = np.random.default_rng(seed)
    maes = np.empty(iterations)
    rmses = np.empty(iterations)
    for index in range(iterations):
        sample = error[rng.integers(0, len(error), len(error))]
        maes[index] = np.abs(sample).mean()
        rmses[index] = np.sqrt(np.square(sample).mean())
    return {
        "iterations": iterations, "seed": seed,
        "MAE_95CI": [float(value) for value in np.quantile(maes, (0.025, 0.975))],
        "RMSE_95CI": [float(value) for value in np.quantile(rmses, (0.025, 0.975))],
    }


def run_test(args: argparse.Namespace) -> None:
    frozen = json.loads(args.frozen_config.read_text())
    if not frozen.get("frozen") or frozen.get("protocol", {}).get("test_read") is not False:
        raise RuntimeError("test requires a validation-frozen config that has not read test metrics")
    split = json.loads(args.split_file.read_text())
    names = list(split["test"])
    assert_exact_cache(args.test_fast_cache, names)
    assert_exact_cache(args.test_tiled_cache, names)
    models = load_models(args.checkpoint_dir, args.device)
    if models["sha256"] != frozen["assets"]["sha256"]:
        raise RuntimeError("checkpoint hashes differ from validation-frozen config")
    fast_config = frozen["selected"]["fast_config"]
    tiled_config = frozen["selected"]["tiled_config"]
    route = frozen["selected"]["route"]

    prediction_rows = {str(seed): [] for seed in SEEDS}
    started = time.time()
    for index, name in enumerate(names):
        stem = Path(name).stem
        fast = prepare_sample(
            load_safe_cache(args.test_fast_cache / f"{stem}.pt"), models, "fast", args.device
        )
        tiled = prepare_sample(
            load_safe_cache(args.test_tiled_cache / f"{stem}.pt"), models, "tiled", args.device
        )
        rescue = None
        if fast["n"] == 0:
            rescue_path = args.test_rescue_cache / f"{stem}.pt"
            if not rescue_path.exists():
                raise FileNotFoundError(f"zero-fast image lacks image-only T4 cache: {rescue_path}")
            rescue = prepare_sample(load_safe_cache(rescue_path), models, "tiled", args.device)
        for seed in SEEDS:
            fast_prediction = count_prepared(fast, seed, fast_config)
            use_tiled = route["mode"] == "always_tiled" or (
                route["mode"] == "predicted_count" and fast_prediction >= route["threshold"]
            )
            source = "tiled2x2" if use_tiled else "fast"
            prediction = count_prepared(tiled, seed, tiled_config) if use_tiled else fast_prediction
            if rescue is not None:
                source = "t4_4x4_fast_zero"
                prediction = count_prepared(rescue, seed, tiled_config)
            prediction_rows[str(seed)].append({
                "file_name": name, "pred_count": prediction, "source": source,
                "raw_fast_candidates": fast["n"], "raw_tiled_candidates": tiled["n"],
            })
        if (index + 1) % 250 == 0:
            print(f"[test prediction] {index+1}/{len(names)} elapsed={(time.time()-started)/60:.1f}m", flush=True)

    # Test annotations enter only after predictions for all 1,190 images exist.
    targets = load_targets(names, args.annotation)
    runs = {}
    for seed in SEEDS:
        key = str(seed)
        rows = []
        for prediction, gt in zip(prediction_rows[key], targets):
            rows.append(prediction | {"gt_count": gt})
        runs[key] = {
            "metrics": metric([row["pred_count"] for row in rows], targets),
            "per_gt_bin": per_bin(rows),
            "bootstrap": bootstrap(rows, args.bootstrap, args.bootstrap_seed + seed),
            "routing": dict(Counter(row["source"] for row in rows)),
            "rows": rows,
        }
    maes = np.asarray([runs[str(seed)]["metrics"]["MAE"] for seed in SEEDS])
    rmses = np.asarray([runs[str(seed)]["metrics"]["RMSE"] for seed in SEEDS])
    output = {
        "date": "2026-07-11",
        "protocol": frozen["protocol"] | {
            "test_read": "only after all predictions were materialized",
            "test_images": len(names),
            "frozen_config_sha256": sha256(args.frozen_config),
        },
        "selected": frozen["selected"],
        "aggregate": {
            "MAE_mean": float(maes.mean()),
            "MAE_std": float(maes.std(ddof=1)),
            "RMSE_mean": float(rmses.mean()),
            "RMSE_std": float(rmses.std(ddof=1)),
        },
        "runs": runs,
        "assets": frozen["assets"],
        "elapsed_seconds": time.time() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output["aggregate"], indent=2))
    print(f"[save] {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("validate", "test"))
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--annotation", type=Path, default=DEFAULT_ANN)
    parser.add_argument("--checkpoint-dir", type=Path, default=REPO / "result/checkpoints/cp_strict")
    parser.add_argument("--val-fast-cache", type=Path, default=DEFAULT_CACHE_ROOT / "fsc147_nogt_val_fast")
    parser.add_argument("--val-tiled-cache", type=Path, default=DEFAULT_CACHE_ROOT / "fsc147_nogt_val_tiled2x2")
    parser.add_argument("--test-fast-cache", type=Path, default=DEFAULT_CACHE_ROOT / "fsc147_nogt_test_fast")
    parser.add_argument("--test-tiled-cache", type=Path, default=DEFAULT_CACHE_ROOT / "fsc147_nogt_test_tiled2x2")
    parser.add_argument("--test-rescue-cache", type=Path, default=DEFAULT_CACHE_ROOT / "fsc147_nogt_test_rescue4x4")
    parser.add_argument("--frozen-config", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260711)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "validate":
        run_validation(args)
    else:
        if args.frozen_config is None:
            parser.error("test phase requires --frozen-config")
        run_test(args)


if __name__ == "__main__":
    main()
