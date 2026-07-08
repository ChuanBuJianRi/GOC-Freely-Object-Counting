"""Evaluate the official ABC123 checkpoint on local counting datasets.

ABC123 is prompt-free/exemplar-free at inference, but it is a dense-map
regression model trained with count/density supervision. The paper reports
FSC147 with two subclass-combination rules:
  - sum: sum all predicted density heads
  - max_density: pixelwise max over predicted density heads, then sum

This script loads the official ActiveVisionLab/ABC123 checkpoint and evaluates
the same prompt-free total-count outputs on FSC147, CARPK, PUCPR+, and
OmniCount-191. It intentionally does not vendor the ABC123 repository; pass
--abc-root to a local clone.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image


BINS = [
    ("0-10", 0, 10),
    ("11-20", 11, 20),
    ("21-50", 21, 50),
    ("51-100", 51, 100),
    ("100+", 101, 10**9),
]


def bin_of(count: float) -> str:
    for lab, lo, hi in BINS:
        if lo <= count <= hi:
            return lab
    return "100+"


class CountingHeadMultiLinearLayersSeparateSum(torch.nn.Module):
    """Einops-free equivalent of ABC123's official counting head."""

    def __init__(
        self,
        feature_dim: int = 384,
        channels: list[str] | tuple[str, ...] = ("5", "32"),
        resolution: int = 28,
        padding_mode: str = "replicate",
    ):
        super().__init__()
        self.num_counts = int(channels[0])
        num_ups = 4
        self.count_feat_dim = int(2**num_ups)
        self.channels = channels
        self.resolution = resolution

        self.net = torch.nn.Sequential()
        self.net.add_module(
            "linear_0", torch.nn.Linear(feature_dim, self.num_counts * self.count_feat_dim)
        )

        self.ups = torch.nn.Sequential()
        in_channels = [int(self.count_feat_dim / (2**i)) for i in range(num_ups)]
        out_channels = [int(self.count_feat_dim / (2 ** (i + 1))) for i in range(num_ups)]
        for i in range(num_ups):
            self.ups.add_module(
                f"conv_{i}",
                torch.nn.Conv2d(
                    in_channels[i],
                    out_channels[i],
                    7,
                    padding=3,
                    padding_mode=padding_mode,
                ),
            )
            if i != num_ups - 1:
                self.ups.add_module(f"relu_{i}", torch.nn.ReLU())
                self.ups.add_module(f"upsample_{i}", torch.nn.UpsamplingBilinear2d(scale_factor=2))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b * h * w, c)
        x = self.net(x)
        x = x.view(b, h, w, self.num_counts, self.count_feat_dim)
        x = x.permute(0, 3, 4, 1, 2).reshape(
            b * self.num_counts, self.count_feat_dim, h, w
        )

        density = self.ups(x)
        density = density.view(b, self.num_counts, density.shape[-2], density.shape[-1])
        counts = density.flatten(2).sum(dim=-1)
        return counts, density


class ABC123Counter(torch.nn.Module):
    def __init__(self, abc_root: Path, pretrained_backbone: bool = False):
        super().__init__()
        sys.path.insert(0, str(abc_root))
        from models.backbone_vit import ViTExtractor

        vit_config = {
            "base_model": "vit_small_patch8_224_dino",
            "facet": "token",
            "layer": 11,
            "bin": False,
            "stride": 8,
            "pretrained": pretrained_backbone,
        }
        self.counting_backbone = ViTExtractor(vit_config)
        self.counting_head = CountingHeadMultiLinearLayersSeparateSum(
            feature_dim=384,
            channels=("5", "32"),
            resolution=28,
            padding_mode="replicate",
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.counting_backbone(x)
        return self.counting_head(feats)


def load_abc123(abc_root: Path, checkpoint: Path, device: str) -> ABC123Counter:
    model = ABC123Counter(abc_root=abc_root, pretrained_backbone=False)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"[load] first missing: {missing[:5]}")
        if unexpected:
            print(f"[load] first unexpected: {unexpected[:5]}")
    model.to(device).eval()
    return model


def image_to_tensor(
    path: str | Path,
    image_size: int = 224,
    normalize_imagenet: bool = False,
) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    ten = torch.from_numpy(arr).permute(2, 0, 1)
    if normalize_imagenet:
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=ten.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=ten.dtype).view(3, 1, 1)
        ten = (ten - mean) / std
    return ten


@torch.no_grad()
def predict_abc123(
    model: ABC123Counter,
    image_paths: list[str],
    device: str,
    gtd_scale: float = 400.0,
    normalize_imagenet: bool = False,
) -> list[dict]:
    batch = torch.stack(
        [image_to_tensor(p, normalize_imagenet=normalize_imagenet) for p in image_paths],
        dim=0,
    ).to(device)
    counts, density = model(batch)
    counts = counts / gtd_scale
    density = density / gtd_scale
    sum_pred = counts.sum(dim=1)
    max_density_pred = density.max(dim=1).values.flatten(1).sum(dim=1)
    head_max_pred = counts.max(dim=1).values
    head_min_abs_pred = counts.abs().min(dim=1).values

    out = []
    for i in range(len(image_paths)):
        head_counts = counts[i].detach().cpu().float().numpy()
        out.append(
            {
                "pred_sum": float(sum_pred[i].detach().cpu()),
                "pred_max_density": float(max_density_pred[i].detach().cpu()),
                "pred_head_max": float(head_max_pred[i].detach().cpu()),
                "pred_head_min_abs": float(head_min_abs_pred[i].detach().cpu()),
                "head_counts": [float(x) for x in head_counts],
            }
        )
    return out


def load_fsc147(
    fsc147_dir: Path,
    split: str = "test",
    exclude_count_over: int = -1,
) -> list[dict]:
    ann = json.loads((fsc147_dir / "annotation_FSC147_384.json").read_text())
    splits = json.loads((fsc147_dir / "Train_Test_Val_FSC_147.json").read_text())
    names = splits[split]
    entries = []
    for name in names:
        if name not in ann:
            continue
        gt = len(ann[name]["points"])
        if exclude_count_over > 0 and gt > exclude_count_over:
            continue
        img_path = fsc147_dir / "images_384_VarV2" / name
        if not img_path.exists():
            continue
        entries.append(
            {
                "id": name,
                "img_path": str(img_path),
                "gt_count": int(gt),
                "dataset": "FSC147",
                "split": split,
            }
        )
    return entries


def read_bbox_count(path: Path) -> int:
    if not path.exists():
        return 0
    return len([line for line in path.read_text().splitlines() if line.strip()])


def resolve_image(img_dir: Path, stem: str) -> Path | None:
    p = img_dir / stem
    if p.exists():
        return p
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def load_car_dataset(devkit_data: Path, split: str, dataset_name: str) -> list[dict]:
    img_dir = devkit_data / "Images"
    ann_dir = devkit_data / "Annotations"
    if split == "all":
        image_paths = sorted(
            p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        names = [p.stem for p in image_paths]
    else:
        split_file = devkit_data / "ImageSets" / f"{split}.txt"
        names = [x.strip() for x in split_file.read_text().splitlines() if x.strip()]
        image_paths = [resolve_image(img_dir, name) for name in names]

    entries = []
    for name, img_path in zip(names, image_paths):
        if img_path is None or not img_path.exists():
            continue
        gt = read_bbox_count(ann_dir / f"{name}.txt")
        entries.append(
            {
                "id": name,
                "img_path": str(img_path),
                "gt_count": int(gt),
                "dataset": dataset_name,
                "split": split,
            }
        )
    return entries


def load_omnicount(omnicount_dir: Path, split: str = "test") -> list[dict]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from script.preprocess_omnicount import load_omnicount_coco

    raw = load_omnicount_coco(str(omnicount_dir))
    entries = []
    for item in raw:
        if split != "all" and item.get("split", "test") != split:
            # Current loader returns test cache entries; keep this guard for future use.
            pass
        entries.append(
            {
                "id": item["file_name"],
                "img_path": item["img_path"],
                "gt_count": int(item["total_count"]),
                "dataset": "OmniCount-191",
                "split": "test",
                "category": item.get("category", ""),
            }
        )
    return entries


def metrics_for(results: list[dict], pred_key: str) -> dict:
    if not results:
        return {"n_images": 0, "MAE": None, "RMSE": None, "bias": None}
    preds = np.array([r[pred_key] for r in results], dtype=np.float64)
    gts = np.array([r["gt_count"] for r in results], dtype=np.float64)
    err = preds - gts
    return {
        "n_images": len(results),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "mean_GT": float(np.mean(gts)),
        "mean_Pred": float(np.mean(preds)),
        "nMAE": float(np.mean(np.abs(err)) / max(float(np.mean(gts)), 1.0)),
    }


def summarize(results: list[dict]) -> dict:
    variants = {
        "sum": "pred_sum",
        "max_density": "pred_max_density",
        "head_max": "pred_head_max",
        "best_head_oracle": "pred_best_head_oracle",
    }
    summary = {name: metrics_for(results, key) for name, key in variants.items()}

    per_bin = {}
    by_bin: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_bin[bin_of(r["gt_count"])].append(r)
    for lab, _, _ in BINS:
        rs = by_bin.get(lab, [])
        if not rs:
            continue
        per_bin[lab] = {name: metrics_for(rs, key) for name, key in variants.items()}

    per_category = {}
    if any("category" in r for r in results):
        by_cat: dict[str, list[dict]] = defaultdict(list)
        for r in results:
            by_cat[r.get("category", "unknown")].append(r)
        for cat, rs in sorted(by_cat.items()):
            per_category[cat] = {name: metrics_for(rs, key) for name, key in variants.items()}

    return {"summary": summary, "per_bin": per_bin, "per_category": per_category}


def batched(seq: list[dict], batch_size: int) -> Iterable[list[dict]]:
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["fsc147", "carpk", "pucpr", "omnicount"])
    ap.add_argument("--abc-root", default="/tmp/ABC123")
    ap.add_argument("--checkpoint", default="/tmp/ABC123/checkpoints/model_chkpt.ckpt")
    ap.add_argument("--fsc147-dir", default="/home/czp/official_code/dataset/FSC147")
    ap.add_argument("--carpk-data", default="/home/czp/official_code/datasets/CARPK_devkit/data")
    ap.add_argument("--pucpr-data", default="/home/czp/official_code/datasets/PUCPR+_devkit/data")
    ap.add_argument("--omnicount-dir", default="/home/czp/official_code/dataset/omnicount/OmniCount-191")
    ap.add_argument("--split", default="test")
    ap.add_argument("--exclude-count-over", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gtd-scale", type=float, default=400.0)
    ap.add_argument("--normalize-imagenet", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.dataset == "fsc147":
        entries = load_fsc147(Path(args.fsc147_dir), args.split, args.exclude_count_over)
    elif args.dataset == "carpk":
        entries = load_car_dataset(Path(args.carpk_data), args.split, "CARPK")
    elif args.dataset == "pucpr":
        entries = load_car_dataset(Path(args.pucpr_data), args.split, "PUCPR+")
    else:
        entries = load_omnicount(Path(args.omnicount_dir), args.split)

    if args.limit > 0:
        entries = entries[: args.limit]

    print(f"[data] dataset={args.dataset} split={args.split} n={len(entries)}")
    print(f"[load] ABC123 root={args.abc_root}")
    model = load_abc123(Path(args.abc_root), Path(args.checkpoint), args.device)
    print("[load] model ready")

    results = []
    t0 = time.time()
    for bi, batch_entries in enumerate(batched(entries, args.batch_size)):
        preds = predict_abc123(
            model,
            [e["img_path"] for e in batch_entries],
            args.device,
            gtd_scale=args.gtd_scale,
            normalize_imagenet=args.normalize_imagenet,
        )
        for entry, pred in zip(batch_entries, preds):
            row = dict(entry)
            row.update(pred)
            heads = np.array(row["head_counts"], dtype=np.float64)
            row["pred_best_head_oracle"] = float(heads[np.argmin(np.abs(heads - row["gt_count"]))])
            row["best_head_idx_oracle"] = int(np.argmin(np.abs(heads - row["gt_count"])))
            results.append(row)

        if (bi + 1) % 10 == 0 or len(results) == len(entries):
            elapsed = time.time() - t0
            rate = len(results) / max(elapsed, 1e-6)
            m_sum = metrics_for(results, "pred_sum")
            m_max = metrics_for(results, "pred_max_density")
            print(
                f"  [{len(results)}/{len(entries)}] "
                f"sum_MAE={m_sum['MAE']:.2f} max_MAE={m_max['MAE']:.2f} "
                f"rate={rate:.1f}/s"
            )

    report = {
        "config": {
            "dataset": args.dataset,
            "split": args.split,
            "exclude_count_over": args.exclude_count_over,
            "abc_root": args.abc_root,
            "checkpoint": args.checkpoint,
            "gtd_scale": args.gtd_scale,
            "normalize_imagenet": args.normalize_imagenet,
            "batch_size": args.batch_size,
            "device": args.device,
        },
        **summarize(results),
        "results": results,
    }

    print("\n=== ABC123 results ===")
    for name, m in report["summary"].items():
        print(
            f"{name:<18} n={m['n_images']} MAE={m['MAE']:.3f} "
            f"RMSE={m['RMSE']:.3f} bias={m['bias']:+.3f} "
            f"mean_pred={m['mean_Pred']:.2f}"
        )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"[save] {out}")


if __name__ == "__main__":
    main()
