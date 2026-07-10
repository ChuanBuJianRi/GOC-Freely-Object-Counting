"""Validate and summarize the MCAC full-2115 leave-one-out reruns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


VARIANTS = ("m1", "m2", "m3", "m4", "m5", "m6")
EXPECTED_OVERLAYS = {"m1": 3, "m2": 0, "m3": 3, "m4": 3, "m5": 0, "m6": 3}
GT_BINS = (
    ("1-10", 1, 10),
    ("11-50", 11, 50),
    ("51-100", 51, 100),
    ("101-200", 101, 200),
    ("201-300", 201, 300),
)


def metrics(pred, gt):
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    err = pred - gt
    return {
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt(np.square(err).mean())),
        "bias": float(err.mean()),
        "n": int(len(err)),
    }


def pair_rows(data):
    rows = []
    for image in data["results"]:
        for row in image["per_class"]:
            rows.append({"img_id": image["img_id"], **row})
    return rows


def bootstrap_by_image(data, n_boot, seed):
    rng = np.random.default_rng(seed)
    images = data["results"]
    mae, rmse = [], []
    for _ in range(n_boot):
        sampled = rng.integers(0, len(images), size=len(images))
        errors = []
        for idx in sampled:
            errors.extend(
                row["pred"] - row["gt"] for row in images[int(idx)]["per_class"]
            )
        errors = np.asarray(errors, dtype=np.float64)
        mae.append(float(np.abs(errors).mean()))
        rmse.append(float(np.sqrt(np.square(errors).mean())))
    return {
        "unit": "image (all GT classes from a sampled image kept together)",
        "n_boot": n_boot,
        "seed": seed,
        "MAE_ci95": [float(x) for x in np.percentile(mae, [2.5, 97.5])],
        "RMSE_ci95": [float(x) for x in np.percentile(rmse, [2.5, 97.5])],
    }


def image_error_map(data, total=False):
    out = {}
    for image in data["results"]:
        if total:
            value = abs(image["total_pred"] - image["total_gt"])
        else:
            value = sum(abs(row["pred"] - row["gt"]) for row in image["per_class"])
        out[image["img_id"]] = value
    return out


def compare_errors(candidate, baseline):
    improved = sum(candidate[k] < baseline[k] for k in baseline)
    worsened = sum(candidate[k] > baseline[k] for k in baseline)
    return {
        "improved_images": improved,
        "worsened_images": worsened,
        "tied_images": len(baseline) - improved - worsened,
    }


def slices(data):
    by_classes = {}
    for n_cls in range(5):
        images = [x for x in data["results"] if len(x["per_class"]) == n_cls]
        pairs = [row for image in images for row in image["per_class"]]
        total = metrics(
            [x["total_pred"] for x in images], [x["total_gt"] for x in images]
        )
        by_classes[str(n_cls)] = {
            "num_images": len(images),
            "per_class_metrics": (
                metrics([x["pred"] for x in pairs], [x["gt"] for x in pairs])
                if pairs else None
            ),
            "total_count_metrics": total,
        }

    pairs = pair_rows(data)
    by_gt_count = {}
    for name, lo, hi in GT_BINS:
        selected = [x for x in pairs if lo <= x["gt"] <= hi]
        by_gt_count[name] = metrics(
            [x["pred"] for x in selected], [x["gt"] for x in selected]
        )
    return {"by_num_gt_classes": by_classes, "by_gt_class_count": by_gt_count}


def validate(variant, data):
    assert data["config"]["variant"] == variant
    assert data["num_images"] == 2115
    assert data["num_image_class_pairs"] == 3630
    assert data["config"]["n_overlay_images"] == EXPECTED_OVERLAYS[variant]
    ids = [x["img_id"] for x in data["results"]]
    assert len(ids) == len(set(ids)) == 2115

    pairs = pair_rows(data)
    recomputed = metrics([x["pred"] for x in pairs], [x["gt"] for x in pairs])
    for key in ("MAE", "RMSE", "bias"):
        assert abs(recomputed[key] - data["per_class_metrics"][key]) < 1e-10

    zero = next(x for x in data["results"] if x["img_id"] == "2277443934862561")
    assert zero["total_gt"] == zero["total_pred"] == 0
    assert zero["per_class"] == []


def strict_summary(data, n_boot, seed):
    pairs = pair_rows(data)
    return {
        "config": data["config"],
        "per_class_metrics": data["per_class_metrics"],
        "total_count_metrics": data["total_count_metrics"],
        "bootstrap_95ci": bootstrap_by_image(data, n_boot, seed),
        "slices": slices(data),
        "zero_target_row": next(
            x for x in data["results"] if x["img_id"] == "2277443934862561"
        ),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs-dir", default="result/logs")
    ap.add_argument("--out", default="result/logs/mcac_full2115_leaveoneout_summary.json")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260710)
    args = ap.parse_args()

    logs = Path(args.logs_dir)
    data = {}
    for variant in VARIANTS:
        path = logs / f"mcac_full2115_{variant}_12p67.json"
        data[variant] = json.loads(path.read_text())
        validate(variant, data[variant])

    base = data["m6"]
    base_pair_errors = image_error_map(base)
    base_total_errors = image_error_map(base, total=True)
    summary = {
        "dataset": "MCAC test full 2,115 images",
        "protocol": base["config"]["protocol"],
        "num_images": 2115,
        "num_image_class_pairs": 3630,
        "zero_target_image": "2277443934862561",
        "protocol_audit": {
            "leave_one_out_candidate_filter": "gt_dot_valid",
            "leave_one_out_uses_gt_at_inference": True,
            "reason": "cache valid is computed from whether a SAM candidate covers a GT dot",
            "publication_status": "diagnostic only; not a strict prompt-free inference result",
        },
        "variants": {},
    }

    for variant in VARIANTS:
        item = data[variant]
        pc = item["per_class_metrics"]
        tc = item["total_count_metrics"]
        summary["variants"][variant] = {
            "config": item["config"],
            "per_class_metrics": pc,
            "total_count_metrics": tc,
            "delta_per_class_vs_m6": {
                "MAE": float(pc["MAE"] - base["per_class_metrics"]["MAE"]),
                "RMSE": float(pc["RMSE"] - base["per_class_metrics"]["RMSE"]),
            },
            "per_image_class_error_vs_m6": compare_errors(
                image_error_map(item), base_pair_errors
            ),
            "per_image_total_error_vs_m6": compare_errors(
                image_error_map(item, total=True), base_total_errors
            ),
            "bootstrap_95ci": bootstrap_by_image(item, args.n_boot, args.seed),
            "slices": slices(item),
        }

    rescue_ids = [
        x["img_id"] for x in base["results"] if x["cache_source"] == "overlay"
    ]
    summary["rescue"] = {
        "trigger": "fast n_candidates == 0 (no GT used)",
        "triggered_images": rescue_ids,
        "m5_m6_predictions_identical": all(
            (a["total_pred"], a["per_class"]) == (b["total_pred"], b["per_class"])
            for a, b in zip(data["m5"]["results"], data["m6"]["results"])
        ),
        "m6_trigger_rows": [
            x for x in base["results"] if x["img_id"] in rescue_ids
        ],
    }

    strict = {}
    for variant in ("m5", "m6"):
        path = logs / f"mcac_strict_nogt_full2115_{variant}_12p67.json"
        item = json.loads(path.read_text())
        assert item["config"]["variant"] == variant
        assert item["config"]["candidate_filter"] == "all"
        assert item["config"]["uses_gt_candidate_filter"] is False
        assert item["num_images"] == 2115
        assert item["num_image_class_pairs"] == 3630
        strict[variant] = item
    summary["strict_no_gt_sanity"] = {
        "m5": strict_summary(strict["m5"], args.n_boot, args.seed),
        "m6": strict_summary(strict["m6"], args.n_boot, args.seed),
        "m5_m6_per_class_predictions_identical": all(
            a["per_class"] == b["per_class"]
            for a, b in zip(strict["m5"]["results"], strict["m6"]["results"])
        ),
        "changed_total_prediction_rows": [
            {
                "img_id": a["img_id"],
                "gt": a["total_gt"],
                "m5_pred": a["total_pred"],
                "m6_pred": b["total_pred"],
            }
            for a, b in zip(strict["m5"]["results"], strict["m6"]["results"])
            if a["total_pred"] != b["total_pred"]
        ],
    }

    occam_path = logs / "mcac_occam_full2115_sharedpts32.json"
    occam = json.loads(occam_path.read_text())
    assert occam["num_images"] == 2115
    assert occam["num_image_class_pairs"] == 3630
    summary["baselines"] = {
        "abc123_published": {
            "MAE": 9.52,
            "RMSE": 17.64,
            "source": "arXiv:2309.04820v2 MCAC test table",
            "protocol_note": "prompt-free, density-map supervised, anonymous density heads",
        },
        "occam_shared_pts32_local": {
            key: occam[key] for key in (
                "method", "num_images", "num_image_class_pairs", "MAE", "RMSE",
                "MAE_count_optimal_matching_diag", "total_count_MAE", "total_count_RMSE",
                "config",
            )
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"validated M1-M6; wrote {out}")


if __name__ == "__main__":
    main()
