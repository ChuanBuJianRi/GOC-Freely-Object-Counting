"""Strict multi-seed CP ablation for the FSC-147 relation heads.

The relation heads and fixed category heads must be trained from official
FSC-147 ``train`` images only. A single ``tau_inst`` is selected for each
paired pts16/pts32 model on the official validation split, then frozen for the
full 1,190-image test split.

This remains a cache-compatible component audit: the existing FSC caches use
GT-dot-derived ``valid`` labels, and the high-density cache route is reproduced
with its existing test-cache membership policy.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
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

from script import ablation_fsc147_multires_components as ab
from script.eval_omnicount_multiclass_ablation import (
    load_category_head,
    load_relation_head,
)


DEFAULT_SPLIT = Path("/home/czp/official_code/dataset/FSC147/Train_Test_Val_FSC_147.json")
DEFAULT_ANN = Path("/home/czp/official_code/dataset/FSC147/annotation_FSC147_384.json")
DEFAULT_CACHE_ROOT = Path("/home/czp/ws_yiyang/ovcud_cache")
DEFAULT_CKPT_DIR = REPO / "result/checkpoints/cp_strict"
DEFAULT_TAUS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.985, 0.99, 0.995, 0.997, 0.999)
GT_BINS = (("0-10", 0, 10), ("11-20", 11, 20), ("21-50", 21, 50),
           ("51-100", 51, 100), ("100+", 101, 10**9))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(pred: list[float], gt: list[float]) -> dict[str, float | int]:
    if not gt:
        return {"n": 0, "MAE": None, "RMSE": None, "bias": None,
                "mean_pred": None, "mean_gt": None}
    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    err = p - g
    return {
        "n": int(len(g)),
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt(np.square(err).mean())),
        "bias": float(err.mean()),
        "mean_pred": float(p.mean()),
        "mean_gt": float(g.mean()),
    }


def per_bin(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for label, lo, hi in GT_BINS:
        selected = [row for row in rows if lo <= row["gt_count"] <= hi]
        result[label] = metrics(
            [row["pred_count"] for row in selected],
            [row["gt_count"] for row in selected],
        )
    return result


def prepare_count(d, category_head, relation_head, text_proto, device: str):
    if d is None or int(d["z"].shape[0]) == 0:
        return None
    z = d["z"].float()
    bbox = d["bbox"].float().numpy()
    probs = ab.get_category_probs(category_head, z, text_proto, device)
    top_conf = probs.max(axis=1)
    indices = ab.valid_indices(d, top_conf, conf=ab.CONF)
    if not indices:
        return None
    groups_local = ab.cluster_full(
        probs,
        ab.semantic_affinity(probs),
        bbox,
        indices,
        float(int(d["height"]) * int(d["width"])),
    )
    groups = [[indices[index] for index in group] for group in groups_local]
    relation = ab.relation_matrices(d, z, probs, top_conf, relation_head, device)
    return {
        "groups": groups,
        "relation": relation,
        "probs": probs,
        "bbox": bbox,
        "image_area": float(int(d["height"]) * int(d["width"])),
        "n": int(z.shape[0]),
    }


def count_at_tau(prepared, tau: float) -> int:
    if prepared is None:
        return 0
    previous = ab.TAU_INST
    try:
        ab.TAU_INST = tau
        return int(ab.dedup_count(
            prepared["groups"],
            prepared["relation"],
            prepared["probs"],
            prepared["bbox"],
            prepared["image_area"],
            prepared["n"],
            adaptive=True,
        ))
    finally:
        ab.TAU_INST = previous


def checkpoint_path(root: Path, resolution: str, condition: str, seed: int) -> Path:
    return root / f"relation_{resolution}_{condition}_seed{seed}.pt"


def validate_checkpoint_pair(paths: list[Path], condition: str, seed: int) -> dict[str, Any]:
    manifests = []
    info = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        manifest = checkpoint.get("data_manifest")
        if not manifest or manifest.get("official_split", {}).get("split_key") != "train":
            raise RuntimeError(f"checkpoint is not an official-train-only run: {path}")
        manifests.append(manifest)
        pretrained = checkpoint.get("pretrained_from")
        if condition == "cp" and not pretrained:
            raise RuntimeError(f"CP checkpoint has no pretrained provenance: {path}")
        if condition == "scratch" and pretrained:
            raise RuntimeError(f"scratch checkpoint unexpectedly uses pretraining: {path}")
        info[path.name] = {
            "sha256": sha256(path),
            "epoch": int(checkpoint["epoch"]),
            "val_metrics": checkpoint["val_metrics"],
            "pretrained_from": pretrained,
            "train_config": checkpoint.get("train_config"),
            "data_manifest": manifest,
        }
    return info


def load_cache(path: Path):
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def validation_predictions(
    names: list[str], ann: dict[str, Any], cache_root: Path,
    val_multires_dir: Path,
    category_fast, category_pts32, relation_fast, relation_pts32,
    text_proto, taus: tuple[float, ...], device: str,
) -> tuple[dict[float, list[int]], list[int], dict[str, int]]:
    predictions = {tau: [] for tau in taus}
    gt = []
    routing = defaultdict(int)
    for index, name in enumerate(names):
        stem = Path(name).stem
        gt_count = ab.gt_count_from_ann(stem, ann)
        use_pts32 = gt_count > 50
        resolution = "pts32" if use_pts32 else "pts16"
        cache_dir = val_multires_dir if use_pts32 else cache_root / "fsc147_train_fast"
        d = load_cache(cache_dir / f"{stem}.pt")
        if use_pts32 and d is None:
            raise FileNotFoundError(
                f"validation multires cache missing for high-density image: {stem}"
            )
        prepared = prepare_count(
            d,
            category_pts32 if use_pts32 else category_fast,
            relation_pts32 if use_pts32 else relation_fast,
            text_proto,
            device,
        )
        for tau in taus:
            predictions[tau].append(count_at_tau(prepared, tau))
        gt.append(gt_count)
        routing[resolution] += 1
        if (index + 1) % 250 == 0:
            print(f"    validation {index+1}/{len(names)}", flush=True)
    return predictions, gt, dict(routing)


def choose_tau(predictions: dict[float, list[int]], gt: list[int]) -> tuple[float, dict[str, Any]]:
    sweep = {str(tau): metrics(pred, gt) for tau, pred in predictions.items()}
    selected = min(
        predictions,
        key=lambda tau: (
            sweep[str(tau)]["MAE"],
            sweep[str(tau)]["RMSE"],
            abs(tau - 0.99),
            -tau,
        ),
    )
    return selected, sweep


def test_predictions(
    names: list[str], ann: dict[str, Any], cache_root: Path,
    category_fast, category_pts32, relation_fast, relation_pts32,
    text_proto, tau: float, device: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    final_dir = cache_root / "fsc147_test_multires_all"
    fast_dir = cache_root / "fsc147_test_fast"
    mr100_dir = cache_root / "fsc147_test_multires"
    mr51_dir = cache_root / "fsc147_test_multires_51_100"
    rescue_dir = cache_root / "fsc147_rescue_7611_4x4_ov25_bbox"
    rows = []
    routing = defaultdict(int)
    for index, name in enumerate(names):
        stem = Path(name).stem
        fast = load_cache(fast_dir / f"{stem}.pt")
        fast_zero = fast is None or int(fast["z"].shape[0]) == 0
        rescue = load_cache(rescue_dir / f"{stem}.pt") if fast_zero else None
        if rescue is not None:
            d, resolution, source = rescue, "pts32", "rescue_4x4"
        else:
            d = load_cache(final_dir / f"{stem}.pt")
            if (mr100_dir / f"{stem}.pt").exists():
                resolution, source = "pts32", "final_mr100"
            elif (mr51_dir / f"{stem}.pt").exists():
                resolution, source = "pts32", "final_mr51"
            else:
                resolution, source = "pts16", "final_fast"
        prepared = prepare_count(
            d,
            category_pts32 if resolution == "pts32" else category_fast,
            relation_pts32 if resolution == "pts32" else relation_fast,
            text_proto,
            device,
        )
        rows.append({
            "file_name": name,
            "gt_count": ab.gt_count_from_ann(stem, ann),
            "pred_count": count_at_tau(prepared, tau),
            "resolution": resolution,
            "cache_source": source,
        })
        routing[source] += 1
        if (index + 1) % 250 == 0:
            print(f"    test {index+1}/{len(names)}", flush=True)
    return rows, dict(routing)


def aggregate_seed_results(results: dict[str, dict[str, Any]], seeds: list[int], n_boot: int, seed: int):
    aggregate = {}
    for condition in ("cp", "scratch"):
        maes = np.asarray([results[f"{condition}_seed{s}"]["test_metrics"]["MAE"] for s in seeds])
        rmses = np.asarray([results[f"{condition}_seed{s}"]["test_metrics"]["RMSE"] for s in seeds])
        aggregate[condition] = {
            "MAE_mean": float(maes.mean()),
            "MAE_std": float(maes.std(ddof=1)) if len(maes) > 1 else 0.0,
            "RMSE_mean": float(rmses.mean()),
            "RMSE_std": float(rmses.std(ddof=1)) if len(rmses) > 1 else 0.0,
            "selected_taus": [results[f"{condition}_seed{s}"]["selected_tau"] for s in seeds],
        }

    deltas = []
    cp_errors, scratch_errors = [], []
    for current_seed in seeds:
        cp_rows = results[f"cp_seed{current_seed}"]["test_rows"]
        scratch_rows = results[f"scratch_seed{current_seed}"]["test_rows"]
        if [r["file_name"] for r in cp_rows] != [r["file_name"] for r in scratch_rows]:
            raise RuntimeError("paired test row order mismatch")
        gt = np.asarray([row["gt_count"] for row in cp_rows], dtype=np.float64)
        cp = np.asarray([row["pred_count"] for row in cp_rows], dtype=np.float64)
        scratch = np.asarray([row["pred_count"] for row in scratch_rows], dtype=np.float64)
        cp_error = np.abs(cp - gt)
        scratch_error = np.abs(scratch - gt)
        cp_errors.append(cp_error)
        scratch_errors.append(scratch_error)
        deltas.append(float((cp_error - scratch_error).mean()))

    per_image_delta = np.stack(cp_errors).mean(axis=0) - np.stack(scratch_errors).mean(axis=0)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        sample = rng.integers(0, len(per_image_delta), len(per_image_delta))
        boot[index] = per_image_delta[sample].mean()
    aggregate["paired_cp_minus_scratch"] = {
        "per_seed_delta_MAE": deltas,
        "mean_delta_MAE": float(np.mean(deltas)),
        "seed_std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
        "image_bootstrap_95CI": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "bootstrap_definition": "image bootstrap of the seed-averaged paired absolute-error delta",
    }
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--val-multires-cache", type=Path,
        default=DEFAULT_CACHE_ROOT / "fsc147_val_multires_51plus",
    )
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--annotation", type=Path, default=DEFAULT_ANN)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 73])
    parser.add_argument("--taus", type=float, nargs="+", default=list(DEFAULT_TAUS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260710)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--out", type=Path, default=REPO / "result/logs/cp_strict_multiseed.json.gz")
    args = parser.parse_args()

    split = json.loads(args.split_file.read_text())
    val_names = list(split["val"])
    test_names = list(split["test"])
    if args.limit > 0:
        val_names = val_names[:args.limit]
        test_names = test_names[:args.limit]
    if args.limit <= 0 and (len(val_names) != 1286 or len(test_names) != 1190):
        raise RuntimeError("unexpected official FSC-147 validation/test split size")
    ann = json.loads(args.annotation.read_text())
    expected_val_multires = {
        Path(name).stem for name in val_names
        if ab.gt_count_from_ann(Path(name).stem, ann) > 50
    }
    available_val_multires = {
        path.stem for path in args.val_multires_cache.glob("*.pt")
    }
    if args.limit <= 0 and expected_val_multires != available_val_multires:
        raise RuntimeError(
            "official-val multires cache must exactly match the 386 GT>50 validation images"
        )
    taus = tuple(sorted(set(args.taus)))

    category_paths = {
        "pts16": args.checkpoint_dir / "category_pts16_trainonly.pt",
        "pts32": args.checkpoint_dir / "category_pts32_trainonly.pt",
    }
    for path in category_paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        manifest = checkpoint.get("data_manifest", {})
        if manifest.get("official_split", {}).get("split_key") != "train":
            raise RuntimeError(f"category checkpoint is not train-only: {path}")

    category_fast = load_category_head(str(category_paths["pts16"]), args.device)
    category_pts32 = load_category_head(str(category_paths["pts32"]), args.device)
    text_proto = F.normalize(
        torch.load(
            REPO / "result/checkpoints/text_prototypes_fsc147.pt",
            map_location=args.device,
            weights_only=False,
        ).float(),
        dim=-1,
    )

    results = {}
    checkpoint_info = {}
    manifest_reference = {}
    started = time.time()
    for condition in ("cp", "scratch"):
        for model_seed in args.seeds:
            key = f"{condition}_seed{model_seed}"
            paths = [
                checkpoint_path(args.checkpoint_dir, "pts16", condition, model_seed),
                checkpoint_path(args.checkpoint_dir, "pts32", condition, model_seed),
            ]
            print(f"[{key}] loading heads", flush=True)
            checkpoint_info[key] = validate_checkpoint_pair(paths, condition, model_seed)
            for resolution, path in zip(("pts16", "pts32"), paths):
                manifest = checkpoint_info[key][path.name]["data_manifest"]
                signature = (
                    manifest["train_sha256"], manifest["model_val_sha256"],
                    manifest["split_seed"],
                )
                if resolution in manifest_reference and manifest_reference[resolution] != signature:
                    raise RuntimeError(
                        f"paired training split mismatch for {resolution}: {key}"
                    )
                manifest_reference[resolution] = signature
            relation_fast = load_relation_head(str(paths[0]), args.device)
            relation_pts32 = load_relation_head(str(paths[1]), args.device)
            val_pred, val_gt, val_routing = validation_predictions(
                val_names, ann, args.cache_root, args.val_multires_cache,
                category_fast, category_pts32, relation_fast, relation_pts32,
                text_proto, taus, args.device,
            )
            selected_tau, val_sweep = choose_tau(val_pred, val_gt)
            print(f"[{key}] selected tau={selected_tau} on validation", flush=True)
            test_rows, test_routing = test_predictions(
                test_names, ann, args.cache_root,
                category_fast, category_pts32, relation_fast, relation_pts32,
                text_proto, selected_tau, args.device,
            )
            test_summary = metrics(
                [row["pred_count"] for row in test_rows],
                [row["gt_count"] for row in test_rows],
            )
            results[key] = {
                "condition": condition,
                "seed": model_seed,
                "selected_tau": selected_tau,
                "validation_sweep": val_sweep,
                "validation_routing": val_routing,
                "test_metrics": test_summary,
                "test_per_gt_bin": per_bin(test_rows),
                "test_routing": test_routing,
                "test_rows": test_rows,
            }
            print(
                f"[{key}] test MAE={test_summary['MAE']:.4f} "
                f"RMSE={test_summary['RMSE']:.4f}",
                flush=True,
            )

    aggregate = aggregate_seed_results(
        results, args.seeds, args.bootstrap, args.bootstrap_seed
    )
    output = {
        "date": "2026-07-10",
        "protocol": {
            "objective": "strict CP vs scratch relation-head ablation",
            "official_train_only": True,
            "model_selection": "fixed internal 10% split of official train; split_seed=20260710",
            "threshold_selection": "official FSC-147 validation only",
            "validation_frontend": (
                "GT<=50: pts16; GT>50: pts16 + 2x2 tiled pts32 merged with bbox-IoU@0.5"
            ),
            "validation_multires_cache": str(args.val_multires_cache),
            "test_selection": "no test metric is read before tau is frozen",
            "tau_grid": list(taus),
            "confidence_threshold": ab.CONF,
            "tau_affinity": ab.TAU_AFF,
            "seeds": args.seeds,
            "category_checkpoints": {name: str(path) for name, path in category_paths.items()},
            "category_sha256": {name: sha256(path) for name, path in category_paths.items()},
            "coco_pretrained_checkpoint": str(REPO / "result/checkpoints/coco_relation_1152.pt"),
            "coco_pretrained_sha256": sha256(
                REPO / "result/checkpoints/coco_relation_1152.pt"
            ),
            "limitation": (
                "This isolates CP without train/test overlap, but keeps the existing "
                "GT-dot-derived valid cache and cache-membership density routing."
            ),
        },
        "checkpoint_info": checkpoint_info,
        "runs": results,
        "aggregate": aggregate,
        "elapsed_seconds": time.time() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix == ".gz":
        with gzip.open(args.out, "wt", encoding="utf-8") as handle:
            json.dump(output, handle, ensure_ascii=False, indent=2)
    else:
        args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(f"[save] {args.out}")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
