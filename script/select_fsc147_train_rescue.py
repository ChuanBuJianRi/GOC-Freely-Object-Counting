"""Select a fast-zero rescue frontend using official-train failures only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from script.eval_fsc147_strict_nogt import (  # noqa: E402
    assert_exact_cache,
    load_filter,
    load_safe_cache,
    metric,
    sha256,
)
from script.train_candidate_filter import bbox_features  # noqa: E402


DEFAULT_SPLIT = Path("/home/czp/official_code/dataset/FSC147/Train_Test_Val_FSC_147.json")
DEFAULT_IMAGES = REPO / "result/configs/fsc147_train_fast_zero_images.json"
DEFAULT_FILTER = REPO / "result/checkpoints/fsc147_candidate_filter_pts32_trainonly.pt"
DEFAULT_TARGETS = REPO / "result/configs/fsc147_train_fast_zero_count_targets.json"


def parse_cache(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("cache must be RECIPE=PATH")
    recipe, path = value.split("=", 1)
    if recipe not in {"2x2", "3x3", "4x4"}:
        raise argparse.ArgumentTypeError("recipe must be 2x2, 3x3, or 4x4")
    return recipe, Path(path)


@torch.no_grad()
def predict_valid_candidates(sample: dict, model, device: str, threshold: float) -> int:
    z = sample["z"].float()
    if len(z) == 0:
        return 0
    geometry = bbox_features(
        sample["bbox"].float(), int(sample["height"]), int(sample["width"])
    )
    probability = torch.sigmoid(model(z.to(device), geometry.to(device)))
    return int((probability >= threshold).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--images-file", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--candidate-filter", type=Path, default=DEFAULT_FILTER)
    parser.add_argument("--cache", action="append", type=parse_cache, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    names = list(json.loads(args.images_file.read_text()))
    split = json.loads(args.split_file.read_text())
    if not names or not set(names).issubset(split["train"]):
        raise RuntimeError("rescue selection images must be official-train only")
    caches = dict(args.cache)
    if set(caches) != {"2x2", "3x3", "4x4"}:
        raise RuntimeError("exactly one 2x2, 3x3, and 4x4 cache is required")
    for cache_dir in caches.values():
        assert_exact_cache(cache_dir, names)

    model = load_filter(args.candidate_filter, args.device)
    predictions: dict[str, list[int]] = {recipe: [] for recipe in caches}
    raw_candidates: dict[str, list[int]] = {recipe: [] for recipe in caches}
    source_cache_values: dict[str, set[str]] = {recipe: set() for recipe in caches}
    for recipe, cache_dir in caches.items():
        for name in names:
            sample = load_safe_cache(cache_dir / f"{Path(name).stem}.pt")
            raw_candidates[recipe].append(len(sample["z"]))
            source_cache_values[recipe].add(str(sample["source_cache"]))
            predictions[recipe].append(
                predict_valid_candidates(sample, model, args.device, args.threshold)
            )

    # The physically isolated train-only shard enters after all predictions.
    target_shard = json.loads(args.targets.read_text())
    if (
        target_shard.get("schema") != "fsc147-count-target-v1"
        or target_shard.get("official_split") != "train"
        or target_shard.get("nonselected_images_loaded") != 0
        or set(target_shard.get("targets", {})) != set(names)
    ):
        raise RuntimeError("rescue target shard is not isolated official-train data")
    targets = [int(target_shard["targets"][name]) for name in names]
    recipe_order = {"2x2": 0, "3x3": 1, "4x4": 2}
    results = {}
    for recipe in caches:
        results[recipe] = {
            "metrics": metric(predictions[recipe], targets),
            "raw_candidate_mean": float(np.mean(raw_candidates[recipe])),
            "source_cache_values": sorted(source_cache_values[recipe]),
            "rows": [
                {
                    "file_name": name,
                    "pred_valid_candidates": prediction,
                    "raw_candidates": raw,
                    "gt_count": target,
                }
                for name, prediction, raw, target in zip(
                    names, predictions[recipe], raw_candidates[recipe], targets
                )
            ],
        }
    selected_recipe = min(
        results,
        key=lambda recipe: (
            results[recipe]["metrics"]["MAE"],
            results[recipe]["metrics"]["RMSE"],
            recipe_order[recipe],
        ),
    )
    expected_sources = results[selected_recipe]["source_cache_values"]
    if len(expected_sources) != 1:
        raise RuntimeError("selected rescue cache does not have one stable source recipe")
    output = {
        "date": "2026-07-10",
        "frozen": True,
        "protocol": {
            "selection_split": "official-train fast-zero images",
            "selection_images": names,
            "validation_images_loaded": 0,
            "test_images_loaded": 0,
            "prediction_gt_fields": [],
            "gt_read": "isolated official-train target shard, after all frontend predictions",
            "selection_metric": "candidate-filter count MAE; RMSE and lower tile count tie-break",
            "candidate_filter_threshold": args.threshold,
        },
        "selected": {
            "recipe": selected_recipe,
            "expected_source_cache": expected_sources[0],
            "metrics": results[selected_recipe]["metrics"],
        },
        "results": results,
        "assets": {
            "candidate_filter": str(args.candidate_filter),
            "candidate_filter_sha256": sha256(args.candidate_filter),
            "target_shard": str(args.targets),
            "target_shard_sha256": sha256(args.targets),
            "cache_paths": {recipe: str(path) for recipe, path in caches.items()},
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output["selected"], ensure_ascii=False, indent=2))
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
