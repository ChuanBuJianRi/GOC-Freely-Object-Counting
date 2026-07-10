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
DEFAULT_RESCUE_SELECTION = REPO / "result/configs/fsc147_train_rescue_selection.json"
DEFAULT_VAL_TARGETS = REPO / "result/configs/fsc147_val_count_targets.json"
SEEDS = (17, 42, 73)
SAFE_FIELDS = {"schema", "img_id", "file_name", "z", "bbox", "height", "width", "source_cache"}
FORBIDDEN_FIELDS = {
    "valid", "purity", "coverage", "iou", "matched_class", "matched_instance_id",
    "gt_count", "class_name", "is_part", "is_countable", "points",
}
GT_BINS = (("0-10", 0, 10), ("11-20", 11, 20), ("21-50", 21, 50),
           ("51-100", 51, 100), ("100+", 101, 10**9))
CODE_ASSETS = (
    Path(__file__).resolve(),
    REPO / "script/ablation_fsc147_multires_components.py",
    REPO / "script/train_candidate_filter.py",
    REPO / "script/train_category_v2.py",
    REPO / "code/clustering/first_neighbor.py",
    REPO / "code/counting/deduplicate.py",
    REPO / "code/counting/representative.py",
    REPO / "code/matrix/pairwise_features.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_asset_sha256() -> dict[str, str]:
    return {str(path.relative_to(REPO)): sha256(path) for path in CODE_ASSETS}


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
        "category_fast": checkpoint_dir / "category_pts16_trainvocab.pt",
        "category_tiled": checkpoint_dir / "category_pts32_trainvocab.pt",
        "filter_fast": REPO / "result/checkpoints/fsc147_candidate_filter_pts16_trainonly.pt",
        "filter_tiled": REPO / "result/checkpoints/fsc147_candidate_filter_pts32_trainonly.pt",
        "prototypes": REPO / "result/checkpoints/text_prototypes_fsc147_train89.pt",
        "prototype_metadata": REPO / "result/checkpoints/text_prototypes_fsc147_train89.json",
    }
    for seed in SEEDS:
        paths[f"relation_fast_{seed}"] = (
            checkpoint_dir / f"relation_pts16_trainvocab_scratch_seed{seed}.pt"
        )
        paths[f"relation_tiled_{seed}"] = (
            checkpoint_dir / f"relation_pts32_trainvocab_scratch_seed{seed}.pt"
        )
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    prototype_metadata = json.loads(paths["prototype_metadata"].read_text())
    if (
        prototype_metadata.get("schema") != "fsc147-train-vocabulary-v1"
        or prototype_metadata.get("official_split") != "train"
        or prototype_metadata.get("test_images_loaded") != 0
        or prototype_metadata.get("validation_images_loaded") != 0
        or prototype_metadata.get("nontrain_class_rows_retained") != 0
        or prototype_metadata.get("out_prototypes_sha256") != sha256(paths["prototypes"])
    ):
        raise RuntimeError("text prototype metadata is not strict official-train-only")
    for name in ("category_fast", "category_tiled"):
        checkpoint = torch.load(paths[name], map_location="cpu", weights_only=False)
        if checkpoint.get("data_manifest", {}).get("official_split", {}).get("split_key") != "train":
            raise RuntimeError(f"category head is not official-train-only: {paths[name]}")
        vocabulary = checkpoint.get("data_manifest", {}).get("prototype_vocabulary", {})
        if (
            vocabulary.get("official_split") != "train"
            or vocabulary.get("test_images_loaded") != 0
            or vocabulary.get("validation_images_loaded") != 0
            or vocabulary.get("nontrain_class_rows_retained") != 0
            or vocabulary.get("prototype_sha256") != sha256(paths["prototypes"])
            or vocabulary.get("metadata_sha256") != sha256(paths["prototype_metadata"])
            or vocabulary.get("global_class_ids") != prototype_metadata.get("global_class_ids")
        ):
            raise RuntimeError(f"category vocabulary is not strict train-only: {paths[name]}")
    for seed in SEEDS:
        for resolution in ("fast", "tiled"):
            path = paths[f"relation_{resolution}_{seed}"]
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if checkpoint.get("pretrained_from", "missing") is not None:
                raise RuntimeError(f"COCO-pretrained relation checkpoint is forbidden: {path}")
            if checkpoint.get("train_config", {}).get("require_train_only_vocabulary") is not True:
                raise RuntimeError(f"relation strict-vocabulary guard was not enabled: {path}")
            if checkpoint.get("data_manifest", {}).get("official_split", {}).get("split_key") != "train":
                raise RuntimeError(f"relation head is not official-train-only: {path}")
            dot_audit = checkpoint.get("data_manifest", {}).get("dot_instance_label_audit", {})
            if (
                dot_audit.get("same_dot_pairs", 0) <= 0
                or dot_audit.get("valid_with_negative_id") != 0
                or dot_audit.get("invalid_with_nonnegative_id") != 0
                or dot_audit.get("id_outside_gt_count") != 0
            ):
                raise RuntimeError(f"relation checkpoint lacks valid FSC dot-label audit: {path}")
            assets = checkpoint.get("data_manifest", {}).get("training_assets", {})
            if (
                assets.get("category_checkpoint_sha256")
                != sha256(paths[f"category_{resolution}"])
                or assets.get("text_prototypes_sha256") != sha256(paths["prototypes"])
            ):
                raise RuntimeError(f"relation training assets differ from strict assets: {path}")

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


def load_rescue_selection(path: Path, models: dict[str, Any]) -> dict[str, Any]:
    selection = json.loads(path.read_text())
    protocol = selection.get("protocol", {})
    if (
        selection.get("frozen") is not True
        or protocol.get("selection_split") != "official-train fast-zero images"
        or protocol.get("validation_images_loaded") != 0
        or protocol.get("test_images_loaded") != 0
        or protocol.get("prediction_gt_fields") != []
        or selection.get("assets", {}).get("candidate_filter_sha256")
        != models["sha256"]["filter_tiled"]
        or selection.get("selected", {}).get("recipe") not in {"2x2", "3x3", "4x4"}
    ):
        raise RuntimeError("rescue frontend was not selected by the strict train-only protocol")
    return selection


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
    relation_ranking_score = category_confidence * filter_probability
    relation = {}
    for seed in SEEDS:
        relation[seed] = ab.relation_matrices(
            d, z, probs, relation_ranking_score,
            models[f"relation_{resolution}_{seed}"], device,
        )
    return {
        "n": n,
        "file_name": d["file_name"],
        "probs": probs,
        "category_confidence": category_confidence,
        "filter_probability": filter_probability,
        "relation_ranking_score": relation_ranking_score,
        "bbox": bbox,
        "image_area": float(int(d["height"]) * int(d["width"])),
        "semantic_affinity": ab.semantic_affinity(probs),
        "relation": relation,
    }


def groups_for_gate(prepared: dict[str, Any], filter_threshold: float,
                    category_threshold: float) -> list[list[int]]:
    n = prepared["n"]
    if n == 0:
        return []
    indices = [
        index for index in range(n)
        if prepared["filter_probability"][index] >= filter_threshold
        and prepared["category_confidence"][index] >= category_threshold
    ]
    if not indices:
        return []
    local_groups = ab.cluster_full(
        prepared["probs"], prepared["semantic_affinity"], prepared["bbox"],
        indices, prepared["image_area"],
    )
    return [[indices[index] for index in group] for group in local_groups]


def count_groups(prepared: dict[str, Any], groups: list[list[int]], seed: int,
                 tau_inst: float) -> int:
    if not groups:
        return 0
    previous = ab.TAU_INST
    try:
        ab.TAU_INST = tau_inst
        return int(ab.dedup_count(
            groups, prepared["relation"][seed], prepared["probs"], prepared["bbox"],
            prepared["image_area"], prepared["n"], adaptive=True,
        ))
    finally:
        ab.TAU_INST = previous


def count_prepared(prepared: dict[str, Any], seed: int, config: dict[str, float]) -> int:
    groups = groups_for_gate(
        prepared, config["filter_threshold"], config["category_threshold"]
    )
    return count_groups(prepared, groups, seed, config["tau_inst"])


def count_prepared_grid(prepared: dict[str, Any],
                        configurations: list[dict[str, float]]) -> dict[str, dict[str, int]]:
    counts = {config_key(config): {} for config in configurations}
    group_cache: dict[tuple[float, float], list[list[int]]] = {}
    for config in configurations:
        gate = (config["filter_threshold"], config["category_threshold"])
        if gate not in group_cache:
            group_cache[gate] = groups_for_gate(prepared, *gate)
        key = config_key(config)
        for seed in SEEDS:
            counts[key][str(seed)] = count_groups(
                prepared, group_cache[gate], seed, config["tau_inst"]
            )
    return counts


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


def load_test_targets_after_predictions(names: list[str], annotation_path: Path) -> list[int]:
    # Full annotation access is restricted to the post-prediction test metric phase.
    annotation = json.loads(annotation_path.read_text())
    return [ab.gt_count_from_ann(Path(name).stem, annotation) for name in names]


def load_isolated_targets(names: list[str], target_path: Path, split_key: str) -> list[int]:
    shard = json.loads(target_path.read_text())
    targets = shard.get("targets", {})
    if (
        shard.get("schema") != "fsc147-count-target-v1"
        or shard.get("official_split") != split_key
        or shard.get("nonselected_images_loaded") != 0
        or set(targets) != set(names)
    ):
        raise RuntimeError(f"target shard is not exact isolated {split_key} data: {target_path}")
    return [int(targets[name]) for name in names]


def grid() -> list[dict[str, float]]:
    return [
        {"filter_threshold": filter_threshold, "category_threshold": category_threshold,
         "tau_inst": tau_inst}
        for filter_threshold in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        for category_threshold in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
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
        sample_counts = count_prepared_grid(prepared, configurations)
        for key, per_seed in sample_counts.items():
            for seed, count in per_seed.items():
                predictions[key][seed].append(count)
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
    rescue_selection = load_rescue_selection(args.rescue_selection, models)

    fast_predictions, fast_raw = validation_frontend_predictions(
        names, args.val_fast_cache, models, "fast", args.device
    )
    tiled_predictions, tiled_raw = validation_frontend_predictions(
        names, args.val_tiled_cache, models, "tiled", args.device
    )
    # Only the isolated val shard is loaded after all validation predictions.
    targets = load_isolated_targets(names, args.val_targets, "val")
    fast_summary = summarize_configs(fast_predictions, targets)
    tiled_summary = summarize_configs(tiled_predictions, targets)
    fast_key = select_config(fast_summary)
    tiled_key = select_config(tiled_summary)
    route, route_sweep = select_route(
        fast_predictions[fast_key], tiled_predictions[tiled_key], targets
    )
    selected_route_key = json.dumps(route, sort_keys=True, separators=(",", ":"))
    selected_seed_metrics = route_sweep[selected_route_key]["seed_metrics"]
    primary_seed = min(
        (str(seed) for seed in SEEDS),
        key=lambda seed: (
            selected_seed_metrics[seed]["MAE"],
            selected_seed_metrics[seed]["RMSE"],
            int(seed),
        ),
    )
    output = {
        "date": "2026-07-10",
        "frozen": True,
        "protocol": {
            "prediction_gt_fields": [],
            "inference_cache_schema": "fsc147-inference-v1",
            "training_supervision": (
                "official-train image-level classes for category head; official-train "
                "dots for candidate filter and shared-dot instance relation"
            ),
            "relation_outputs_used": "instance branch only; part-whole matrix is fixed zero",
            "text_vocabulary": "89 official-train classes; no val/test class names",
            "relation_initialization": "scratch; COCO pretraining forbidden",
            "selection_split": "official val 1286",
            "selection_target_shard": str(args.val_targets),
            "selection_target_shard_sha256": sha256(args.val_targets),
            "test_read": False,
            "seeds": list(SEEDS),
            "grid": grid(),
            "route_grid": [row["policy"] for row in route_sweep.values()],
            "fast_zero_rescue": (
                f"if raw fast candidate count is zero, use train-selected "
                f"{rescue_selection['selected']['recipe']} image-only tiled cache"
            ),
        },
        "selected": {
            "fast_config": json.loads(fast_key),
            "tiled_config": json.loads(tiled_key),
            "route": route,
            "primary_seed": int(primary_seed),
            "primary_seed_selection": "minimum validation MAE, then RMSE; never test",
            "validation_metrics": selected_seed_metrics,
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
        "assets": {
            "paths": models["paths"],
            "sha256": models["sha256"],
            "code_sha256": code_asset_sha256(),
        },
        "rescue_selection": {
            "path": str(args.rescue_selection),
            "sha256": sha256(args.rescue_selection),
            "selected": rescue_selection["selected"],
            "protocol": rescue_selection["protocol"],
        },
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
    if args.tile_plan is None:
        raise RuntimeError("test phase requires --tile-plan")
    tile_plan = json.loads(args.tile_plan.read_text())
    if (
        tile_plan.get("frozen_config_sha256") != sha256(args.frozen_config)
        or tile_plan.get("test_annotations_read") is not False
        or tile_plan.get("prediction_gt_fields") != []
    ):
        raise RuntimeError("tile plan was not generated from the supplied frozen config")
    tile_names = list(tile_plan["tile_names"])
    fast_zero_plan = list(tile_plan["fast_zero_names"])
    if (
        len(tile_names) != len(set(tile_names))
        or len(fast_zero_plan) != len(set(fast_zero_plan))
        or not set(tile_names).issubset(names)
        or not set(fast_zero_plan).issubset(names)
    ):
        raise RuntimeError("tile plan contains names outside the official test split")
    assert_exact_cache(args.test_tiled_cache, tile_names)
    assert_exact_cache(args.test_rescue_cache, fast_zero_plan)
    tile_stems = {Path(name).stem for name in tile_names}
    models = load_models(args.checkpoint_dir, args.device)
    if models["sha256"] != frozen["assets"]["sha256"]:
        raise RuntimeError("checkpoint hashes differ from validation-frozen config")
    if code_asset_sha256() != frozen["assets"].get("code_sha256"):
        raise RuntimeError("prediction code differs from validation-frozen config")
    rescue_selection = load_rescue_selection(args.rescue_selection, models)
    if sha256(args.rescue_selection) != frozen.get("rescue_selection", {}).get("sha256"):
        raise RuntimeError("train-only rescue selection differs from validation-frozen config")
    expected_rescue_source = rescue_selection["selected"]["expected_source_cache"]
    fast_config = frozen["selected"]["fast_config"]
    tiled_config = frozen["selected"]["tiled_config"]
    route = frozen["selected"]["route"]

    prediction_rows = {str(seed): [] for seed in SEEDS}
    observed_fast_zero = []
    started = time.time()
    for index, name in enumerate(names):
        stem = Path(name).stem
        fast = prepare_sample(
            load_safe_cache(args.test_fast_cache / f"{stem}.pt"), models, "fast", args.device
        )
        tiled = None
        if stem in tile_stems:
            tiled = prepare_sample(
                load_safe_cache(args.test_tiled_cache / f"{stem}.pt"), models, "tiled", args.device
            )
        rescue = None
        if fast["n"] == 0:
            observed_fast_zero.append(name)
            rescue_path = args.test_rescue_cache / f"{stem}.pt"
            if not rescue_path.exists():
                raise FileNotFoundError(f"zero-fast image lacks image-only T4 cache: {rescue_path}")
            rescue_sample = load_safe_cache(rescue_path)
            if rescue_sample.get("source_cache") != expected_rescue_source:
                raise RuntimeError(f"test rescue recipe differs from train selection: {rescue_path}")
            rescue = prepare_sample(rescue_sample, models, "tiled", args.device)
        for seed in SEEDS:
            fast_prediction = count_prepared(fast, seed, fast_config)
            use_tiled = route["mode"] == "always_tiled" or (
                route["mode"] == "predicted_count" and fast_prediction >= route["threshold"]
            )
            if rescue is not None:
                source = f"rescue_{rescue_selection['selected']['recipe']}_fast_zero"
                prediction = count_prepared(rescue, seed, tiled_config)
            elif use_tiled:
                if tiled is None:
                    raise RuntimeError(
                        f"routed image is absent from frozen tile plan: {name}, seed={seed}"
                    )
                source = "tiled2x2"
                prediction = count_prepared(tiled, seed, tiled_config)
            else:
                source = "fast"
                prediction = fast_prediction
            prediction_rows[str(seed)].append({
                "file_name": name, "pred_count": prediction, "source": source,
                "raw_fast_candidates": fast["n"],
                "raw_tiled_candidates": None if tiled is None else tiled["n"],
            })
        if (index + 1) % 250 == 0:
            print(f"[test prediction] {index+1}/{len(names)} elapsed={(time.time()-started)/60:.1f}m", flush=True)

    if sorted(observed_fast_zero) != sorted(fast_zero_plan):
        raise RuntimeError("fast-zero observations differ from the frozen image-only tile plan")

    # Test annotations enter only after predictions for all 1,190 images exist.
    targets = load_test_targets_after_predictions(names, args.annotation)
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
        "date": "2026-07-10",
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
        "primary_seed": frozen["selected"]["primary_seed"],
        "primary": {
            key: value
            for key, value in runs[str(frozen["selected"]["primary_seed"])].items()
            if key != "rows"
        },
        "runs": runs,
        "assets": frozen["assets"],
        "elapsed_seconds": time.time() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output["aggregate"], indent=2))
    print(f"[save] {args.out}")


def run_test_plan(args: argparse.Namespace) -> None:
    frozen = json.loads(args.frozen_config.read_text())
    if not frozen.get("frozen") or frozen.get("protocol", {}).get("test_read") is not False:
        raise RuntimeError("test planning requires an untouched validation-frozen config")
    split = json.loads(args.split_file.read_text())
    names = list(split["test"])
    assert_exact_cache(args.test_fast_cache, names)
    models = load_models(args.checkpoint_dir, args.device)
    if models["sha256"] != frozen["assets"]["sha256"]:
        raise RuntimeError("checkpoint hashes differ from validation-frozen config")
    if code_asset_sha256() != frozen["assets"].get("code_sha256"):
        raise RuntimeError("prediction code differs from validation-frozen config")
    fast_config = frozen["selected"]["fast_config"]
    route = frozen["selected"]["route"]
    routed_by_seed = {str(seed): [] for seed in SEEDS}
    fast_zero = []
    for index, name in enumerate(names):
        stem = Path(name).stem
        prepared = prepare_sample(
            load_safe_cache(args.test_fast_cache / f"{stem}.pt"), models, "fast", args.device
        )
        if prepared["n"] == 0:
            fast_zero.append(name)
            continue
        for seed in SEEDS:
            prediction = count_prepared(prepared, seed, fast_config)
            use_tiled = route["mode"] == "always_tiled" or (
                route["mode"] == "predicted_count" and prediction >= route["threshold"]
            )
            if use_tiled:
                routed_by_seed[str(seed)].append(name)
        if (index + 1) % 250 == 0:
            print(f"[test plan] {index+1}/{len(names)}", flush=True)
    union = sorted(set().union(*(set(value) for value in routed_by_seed.values())))
    output = {
        "date": "2026-07-10",
        "prediction_gt_fields": [],
        "test_annotations_read": False,
        "frozen_config": str(args.frozen_config),
        "frozen_config_sha256": sha256(args.frozen_config),
        "route": route,
        "tile_names": union,
        "tile_count": len(union),
        "routed_by_seed": routed_by_seed,
        "fast_zero_names": fast_zero,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"tile_count": len(union), "fast_zero": fast_zero}, indent=2))
    print(f"[save] {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("validate", "plan-test", "test"))
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--annotation", type=Path, default=DEFAULT_ANN)
    parser.add_argument("--val-targets", type=Path, default=DEFAULT_VAL_TARGETS)
    parser.add_argument("--checkpoint-dir", type=Path, default=REPO / "result/checkpoints/cp_strict")
    parser.add_argument("--val-fast-cache", type=Path, default=DEFAULT_CACHE_ROOT / "fsc147_nogt_val_fast")
    parser.add_argument("--val-tiled-cache", type=Path, default=DEFAULT_CACHE_ROOT / "fsc147_nogt_val_tiled2x2")
    parser.add_argument("--test-fast-cache", type=Path, default=DEFAULT_CACHE_ROOT / "fsc147_nogt_test_fast")
    parser.add_argument("--test-tiled-cache", type=Path, default=DEFAULT_CACHE_ROOT / "fsc147_nogt_test_tiled2x2")
    parser.add_argument("--test-rescue-cache", type=Path, default=DEFAULT_CACHE_ROOT / "fsc147_nogt_test_rescue4x4")
    parser.add_argument("--rescue-selection", type=Path, default=DEFAULT_RESCUE_SELECTION)
    parser.add_argument("--frozen-config", type=Path)
    parser.add_argument("--tile-plan", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260711)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "validate":
        run_validation(args)
    elif args.phase == "plan-test":
        if args.frozen_config is None:
            parser.error("plan-test phase requires --frozen-config")
        run_test_plan(args)
    else:
        if args.frozen_config is None or args.tile_plan is None:
            parser.error("test phase requires --frozen-config and --tile-plan")
        run_test(args)


if __name__ == "__main__":
    main()
