"""Re-run the released ABC123 checkpoint on the official MCAC test protocol.

This adapter deliberately reuses ActiveVisionLab/ABC123's MCAC_Dataset so the
crop, occlusion filtering, precomputed density maps, and class compaction match
the released code. The model forward uses the previously audited einops-free
wrapper in eval_abc123_baseline.py; its count and density outputs are bit-equal
to the released ABC123 module for the same input and checkpoint.

The released config has eval_batch_size=2 and drop_last=True while the local
test split has 2,115 images. One full pass therefore reports both:
  - full_2115: every local test image;
  - official_drop_last: the exact prefix retained by the released DataLoader.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, SequentialSampler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from script.eval_abc123_baseline import load_abc123


GT_BINS = (
    ("1-10", 1, 10),
    ("11-50", 11, 50),
    ("51-100", 51, 100),
    ("101-200", 101, 200),
    ("201-300", 201, 300),
)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def abc_repo_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def min_cost_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact rectangular assignment for five prediction heads and <=4 GT classes."""
    n_pred, n_gt = cost.shape
    if n_gt == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    if n_gt > n_pred:
        raise ValueError(f"cannot match {n_gt} GT classes to {n_pred} heads")

    best_rows = None
    best_cost = float("inf")
    cols = np.arange(n_gt, dtype=np.int64)
    for rows in itertools.permutations(range(n_pred), n_gt):
        value = float(sum(cost[rows[c], c] for c in range(n_gt)))
        if value < best_cost:
            best_cost = value
            best_rows = rows
    return np.asarray(best_rows, dtype=np.int64), cols


def match_one(
    pred_counts_raw: torch.Tensor,
    pred_density_raw: torch.Tensor,
    gt_counts: torch.Tensor,
    gt_density: torch.Tensor,
    gtd_scale: float,
) -> dict:
    gt_nonzero = torch.nonzero(gt_counts > 0, as_tuple=False).flatten()
    n_gt = int(gt_nonzero.numel())
    head_counts = (pred_counts_raw / gtd_scale).detach().cpu().float().numpy()

    if n_gt == 0:
        return {
            "gt_counts": [],
            "pred_counts": [],
            "pred_head_indices": [],
            "gt_compact_indices": [],
            "head_counts": [float(x) for x in head_counts],
        }

    pred_flat = pred_density_raw.flatten(1)
    gt_flat = gt_density[gt_nonzero].flatten(1).to(pred_flat)
    pred_norm = F.normalize(pred_flat, dim=1)
    gt_norm = F.normalize(gt_flat, dim=1)
    cost = torch.cdist(pred_norm, gt_norm, p=1).square().detach().cpu().numpy()
    pred_idx, compact_gt_idx = min_cost_assignment(cost)

    gt_values = gt_counts[gt_nonzero][torch.from_numpy(compact_gt_idx)].cpu().numpy()
    pred_values = head_counts[pred_idx]
    return {
        "gt_counts": [float(x) for x in gt_values],
        "pred_counts": [float(x) for x in pred_values],
        "pred_head_indices": [int(x) for x in pred_idx],
        "gt_compact_indices": [int(x) for x in compact_gt_idx],
        "head_counts": [float(x) for x in head_counts],
    }


def basic_metrics(pred: list[float], gt: list[float]) -> dict:
    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    if len(g) == 0:
        return {"MAE": None, "RMSE": None, "bias": None, "n": 0}
    err = p - g
    return {
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt(np.square(err).mean())),
        "bias": float(err.mean()),
        "n": int(len(err)),
        "mean_gt": float(g.mean()),
        "mean_pred": float(p.mean()),
    }


def summarize(rows: list[dict]) -> dict:
    pred_pairs = [p for row in rows for p in row["pred_counts"]]
    gt_pairs = [g for row in rows for g in row["gt_counts"]]
    matched_total_pred = [sum(row["pred_counts"]) for row in rows]
    all_head_total_pred = [sum(row["head_counts"]) for row in rows]
    gt_total = [sum(row["gt_counts"]) for row in rows]

    per_gt_bin = {}
    pair_records = [
        (g, p)
        for row in rows
        for g, p in zip(row["gt_counts"], row["pred_counts"])
    ]
    for name, lo, hi in GT_BINS:
        selected = [(g, p) for g, p in pair_records if lo <= g <= hi]
        per_gt_bin[name] = basic_metrics(
            [p for g, p in selected], [g for g, p in selected]
        )

    by_num_classes = {}
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[len(row["gt_counts"])].append(row)
    for n_cls in range(5):
        selected = grouped.get(n_cls, [])
        by_num_classes[str(n_cls)] = {
            "num_images": len(selected),
            "per_class_metrics": basic_metrics(
                [p for row in selected for p in row["pred_counts"]],
                [g for row in selected for g in row["gt_counts"]],
            ),
            "matched_total_metrics": basic_metrics(
                [sum(row["pred_counts"]) for row in selected],
                [sum(row["gt_counts"]) for row in selected],
            ),
            "all_head_total_metrics": basic_metrics(
                [sum(row["head_counts"]) for row in selected],
                [sum(row["gt_counts"]) for row in selected],
            ),
        }

    g = np.asarray(gt_pairs, dtype=np.float64)
    p = np.asarray(pred_pairs, dtype=np.float64)
    err = p - g
    return {
        "num_images": len(rows),
        "num_image_class_pairs": len(gt_pairs),
        "per_class_metrics": basic_metrics(pred_pairs, gt_pairs),
        "NAE": float(np.mean(np.abs(err) / g)) if len(g) else None,
        "SRE": float(np.sqrt(np.mean(np.square(err) / g))) if len(g) else None,
        "matched_total_metrics": basic_metrics(matched_total_pred, gt_total),
        "all_head_total_metrics": basic_metrics(all_head_total_pred, gt_total),
        "by_num_gt_classes": by_num_classes,
        "by_gt_class_count": per_gt_bin,
    }


def load_official_dataset(abc_root: Path, mcac_root: Path):
    sys.path.insert(0, str(abc_root))
    from data import MCAC_Dataset

    cfg = yaml.safe_load((abc_root / "configs/_DEFAULT.yml").read_text())
    cfg.update(yaml.safe_load((abc_root / "configs/ABC123test.yml").read_text()))
    cfg["data_path"] = str(mcac_root.resolve()) + os.sep
    return MCAC_Dataset(cfg, train=False), cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--abc-root", default="/tmp/ABC123")
    ap.add_argument("--checkpoint", default="/tmp/ABC123/checkpoints/model_chkpt.ckpt")
    ap.add_argument("--mcac-root", default="/home/czp/ljs/dataset/MCAC")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--gtd-scale", type=float, default=400.0)
    ap.add_argument(
        "--legacy-tensor-resize",
        action="store_true",
        help="Use antialias=False to emulate torchvision 0.13 Tensor Resize.",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="result/logs/abc123_mcac_test_full2115_official.json")
    args = ap.parse_args()

    abc_root = Path(args.abc_root)
    checkpoint = Path(args.checkpoint)
    dataset, official_cfg = load_official_dataset(abc_root, Path(args.mcac_root))
    if args.legacy_tensor_resize:
        from torchvision.transforms import Resize

        dataset.resize_im = Resize(
            (official_cfg["img_size"][0], official_cfg["img_size"][1]),
            antialias=False,
        )
    if args.limit > 0:
        dataset.im_ids = dataset.im_ids[: args.limit]
    loader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    print(f"[data] images={len(dataset)} batch={args.batch_size} order=official os.listdir")
    print(f"[data] first={dataset.im_ids[0]} last={dataset.im_ids[-1]}")
    model = load_abc123(abc_root, checkpoint, args.device)
    print("[model] released ABC123 checkpoint loaded", flush=True)

    rows = []
    t0 = time.time()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            images, _, _, gt_density, gt_counts, image_ids = batch
            images = images.to(args.device, non_blocking=True)
            pred_counts_raw, pred_density_raw = model(images)

            for i in range(images.shape[0]):
                matched = match_one(
                    pred_counts_raw[i],
                    pred_density_raw[i],
                    gt_counts[i],
                    gt_density[i],
                    args.gtd_scale,
                )
                matched["image_id"] = str(int(image_ids[i]))
                matched["gt_total"] = float(sum(matched["gt_counts"]))
                matched["pred_matched_total"] = float(sum(matched["pred_counts"]))
                matched["pred_all_head_total"] = float(sum(matched["head_counts"]))
                rows.append(matched)

            if (batch_idx + 1) % 50 == 0 or len(rows) == len(dataset):
                m = summarize(rows)["per_class_metrics"]
                rate = len(rows) / max(time.time() - t0, 1e-6)
                print(
                    f"  [{len(rows)}/{len(dataset)}] pairs={m['n']} "
                    f"MAE={m['MAE']:.3f} RMSE={m['RMSE']:.3f} rate={rate:.1f}/s",
                    flush=True,
                )

    retained = (len(rows) // args.batch_size) * args.batch_size
    official_rows = rows[:retained]
    dropped_rows = rows[retained:]
    report = {
        "config": {
            "method": "ABC123 released checkpoint",
            "abc_repo": str(abc_root),
            "abc_repo_commit": abc_repo_commit(abc_root),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "official_config": "configs/ABC123test.yml",
            "crop_size": official_cfg["MCAC_crop_size"],
            "occ_limit": official_cfg["MCAC_occ_limit"],
            "matching": "L2-normalized density maps; Hungarian L1 cost squared",
            "gtd_scale": args.gtd_scale,
            "batch_size": args.batch_size,
            "released_drop_last": bool(official_cfg["drop_last"]),
            "device": args.device,
            "torch_version": torch.__version__,
            "legacy_tensor_resize": args.legacy_tensor_resize,
        },
        "published_reference": {
            "source": "arXiv:2309.04820v2 MCAC test table",
            "MAE": 9.52,
            "RMSE": 17.64,
        },
        "full_2115": summarize(rows),
        "official_drop_last": {
            **summarize(official_rows),
            "dropped_image_ids": [row["image_id"] for row in dropped_rows],
        },
        "results": rows,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("\n=== ABC123 MCAC reproduction ===")
    for name in ("full_2115", "official_drop_last"):
        m = report[name]["per_class_metrics"]
        print(
            f"{name}: images={report[name]['num_images']} pairs={m['n']} "
            f"MAE={m['MAE']:.4f} RMSE={m['RMSE']:.4f} bias={m['bias']:+.4f}"
        )
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
