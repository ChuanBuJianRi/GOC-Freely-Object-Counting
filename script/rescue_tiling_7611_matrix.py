"""Extreme-density rescue tiling matrix for FSC147 7611.jpg.

This is a targeted diagnostic for the GT=2560 FSC147 test image that was
missing from the original multires cache. It keeps the counting heads fixed and
varies only the tiled proposal frontend.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from script import preprocess_fsc147_tiled as tiled  # noqa: E402
from script.ablation_fsc147_multires_components import (  # noqa: E402
    decode_cover_sets_from_masks,
    evaluate_one,
    load_category_head,
    load_relation_head,
)


FSC_DIR = Path("/home/czp/official_code/dataset/FSC147")
CACHE_ROOT = Path("/home/czp/ws_yiyang/ovcud_cache")


RESCUE_CONFIGS = [
    {
        "name": "2x2_tiled_ov25_bbox",
        "tiles": 2,
        "overlap": 0.25,
        "upscale": False,
        "cache_dir": "fsc147_rescue_7611_2x2_ov25_bbox",
    },
    {
        "name": "3x3_tiled_ov25_bbox",
        "tiles": 3,
        "overlap": 0.25,
        "upscale": False,
        "cache_dir": "fsc147_rescue_7611_3x3_ov25_bbox",
    },
    {
        "name": "4x4_tiled_ov25_bbox",
        "tiles": 4,
        "overlap": 0.25,
        "upscale": False,
        "cache_dir": "fsc147_rescue_7611_4x4_ov25_bbox",
    },
    {
        "name": "3x3_overlap50_bbox",
        "tiles": 3,
        "overlap": 0.50,
        "upscale": False,
        "cache_dir": "fsc147_rescue_7611_3x3_ov50_bbox",
    },
    {
        "name": "4x4_overlap50_bbox",
        "tiles": 4,
        "overlap": 0.50,
        "upscale": False,
        "cache_dir": "fsc147_rescue_7611_4x4_ov50_bbox",
    },
    {
        "name": "2x2_upscaled_ov25_bbox",
        "tiles": 2,
        "overlap": 0.25,
        "upscale": True,
        "cache_dir": "fsc147_rescue_7611_2x2_upscale_ov25_bbox",
    },
]


EXISTING_REFS = [
    ("2x2_tiled_existing_mask", CACHE_ROOT / "fsc147_test_tiled" / "7611.pt"),
    ("3x3_tiled_existing_mask", CACHE_ROOT / "fsc147_test_tiled_3x3" / "7611.pt"),
    (
        "2x2_upscaled_existing_mask",
        CACHE_ROOT / "fsc147_test_tiled_2x2_upscale" / "7611.pt",
    ),
]


def load_class_info(file_name: str):
    cats = json.loads((REPO / "result/checkpoints/text_prototypes_fsc147_categories.json").read_text())[
        "categories"
    ]
    name_to_idx = {c["name"]: c["contiguous_id"] for c in cats}
    img_to_class = {}
    with (FSC_DIR / "ImageClasses_FSC147.txt").open() as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                img_to_class[parts[0]] = parts[1]
    class_name = img_to_class[file_name]
    return class_name, name_to_idx[class_name]


def high_frequency_score(image: np.ndarray) -> dict:
    gray = image.astype(np.float32).mean(axis=2) / 255.0
    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx * gx + gy * gy)
    return {
        "gradient_mean": float(grad.mean()),
        "gradient_p95": float(np.percentile(grad, 95)),
        "gradient_var": float(grad.var()),
    }


def bbox_stats(d: dict) -> dict:
    n = int(d["z"].shape[0])
    if n == 0:
        return {
            "mean_bbox_area_ratio": 0.0,
            "median_bbox_area_ratio": 0.0,
            "p95_bbox_area_ratio": 0.0,
        }
    h, w = int(d["height"]), int(d["width"])
    areas = (d["bbox"][:, 2] * d["bbox"][:, 3]).float().numpy() / float(h * w)
    return {
        "mean_bbox_area_ratio": float(areas.mean()),
        "median_bbox_area_ratio": float(np.median(areas)),
        "p95_bbox_area_ratio": float(np.percentile(areas, 95)),
    }


def evaluate_cache(cache_path, name, ann_entry, category_head, relation_head, text_prototypes, device):
    d = torch.load(cache_path, map_location="cpu", weights_only=False)
    cover_sets = decode_cover_sets_from_masks(d, ann_entry.get("points", []))
    row = evaluate_one(d, category_head, relation_head, text_prototypes, device, cover_sets)
    covered = set()
    for s in cover_sets:
        covered.update(s)
    gt = int(row["gt_count"])
    return {
        "name": name,
        "cache_path": str(cache_path),
        "n_candidates": int(row["n_candidates"]),
        "n_valid": int((d["valid"] > 0).sum().item()) if d["valid"].numel() else 0,
        "A1_filter_only": int(row["A1_filter_only"]),
        "A8_full": int(row["A8_full"]),
        "O1_oracle_category": int(row["O1_oracle_category"]),
        "O2_oracle_dedup": int(row["O2_oracle_dedup"]),
        "O3_proposal_cover": int(row["O3_proposal_cover"]),
        "covered_dots": int(len(covered)),
        "proposal_cover_rate": float(len(covered) / max(gt, 1)),
        "n_conf_filtered": int(row.get("n_conf_filtered", 0)),
        "n_groups_A8": int(row.get("n_groups_A8", 0)),
        **bbox_stats(d),
    }


def process_config(cfg, image, file_name, ann_entry, class_idx, class_name, amg, encoder, args):
    out_dir = Path(args.cache_root) / cfg["cache_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(file_name).stem}.pt"
    if out_path.exists() and not args.force:
        return out_path, 0.0, "cached"

    t0 = time.time()
    result = tiled.process_image_tiled(
        image,
        file_name,
        ann_entry,
        class_idx,
        class_name,
        amg,
        encoder,
        cfg["tiles"],
        cfg["overlap"],
        upscale=cfg["upscale"],
        merge_mode=args.merge_mode,
    )
    if result is None:
        raise RuntimeError(f"no candidates for {cfg['name']}")
    torch.save(result, out_path)
    return out_path, time.time() - t0, "processed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="7611.jpg")
    ap.add_argument("--out", default="result/logs/fsc147_rescue_tiling_7611_matrix.json")
    ap.add_argument("--cache-root", default=str(CACHE_ROOT))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pts-per-side", type=int, default=32)
    ap.add_argument("--merge-mode", choices=["bbox", "mask"], default="bbox")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    device = args.device
    file_name = args.image
    ann = json.loads((FSC_DIR / "annotation_FSC147_384.json").read_text())
    ann_entry = ann[file_name]
    class_name, class_idx = load_class_info(file_name)
    image_path = FSC_DIR / "images_384_VarV2" / file_name
    image = np.array(Image.open(image_path).convert("RGB"))

    # SAM2 config paths are relative to the official_code working directory.
    os.chdir("/home/czp/official_code")
    print(f"[init] image={file_name} class={class_name} gt={len(ann_entry['points'])}")
    print(f"[init] build SAM2 pts={args.pts_per_side}")
    amg = tiled.build_sam2_amg(device, args.pts_per_side)
    print("[init] build DINOv2")
    encoder = tiled.DINOv2RegionEncoder(device=device)

    print("[init] load counting heads")
    category_head = load_category_head(REPO / "result/checkpoints/category_cosine_pts32.pt", device)
    relation_head = load_relation_head(REPO / "result/checkpoints/fsc147_relation_pts32_best.pt", device)
    text_prototypes = torch.nn.functional.normalize(
        torch.load(
            REPO / "result/checkpoints/text_prototypes_fsc147.pt",
            map_location="cpu",
            weights_only=False,
        ).float(),
        dim=-1,
    ).to(device)

    rows = []
    for cfg in RESCUE_CONFIGS:
        print(
            f"[run] {cfg['name']} tiles={cfg['tiles']} overlap={cfg['overlap']} "
            f"upscale={cfg['upscale']} merge={args.merge_mode}",
            flush=True,
        )
        cache_path, elapsed, status = process_config(
            cfg, image, file_name, ann_entry, class_idx, class_name, amg, encoder, args
        )
        row = evaluate_cache(
            cache_path, cfg["name"], ann_entry, category_head, relation_head, text_prototypes, device
        )
        row.update({
            "tiles": cfg["tiles"],
            "overlap": cfg["overlap"],
            "upscale": bool(cfg["upscale"]),
            "merge_mode": args.merge_mode,
            "status": status,
            "preprocess_elapsed_s": float(elapsed),
        })
        rows.append(row)
        print(
            f"  -> cand={row['n_candidates']} valid={row['n_valid']} "
            f"pred={row['A8_full']} O3={row['O3_proposal_cover']} "
            f"cover={row['proposal_cover_rate']:.1%} elapsed={elapsed:.1f}s",
            flush=True,
        )

    refs = []
    for name, path in EXISTING_REFS:
        if path.exists():
            refs.append(
                evaluate_cache(path, name, ann_entry, category_head, relation_head, text_prototypes, device)
            )

    fast_path = Path(args.cache_root) / "fsc147_test_fast" / f"{Path(file_name).stem}.pt"
    fast_summary = None
    if fast_path.exists():
        d_fast = torch.load(fast_path, map_location="cpu", weights_only=False)
        fast_summary = {
            "cache_path": str(fast_path),
            "n_candidates": int(d_fast["z"].shape[0]),
            "n_valid": int((d_fast["valid"] > 0).sum().item()) if d_fast["valid"].numel() else 0,
            **bbox_stats(d_fast),
        }

    out = {
        "image": file_name,
        "gt_count": len(ann_entry["points"]),
        "class_name": class_name,
        "device": device,
        "pts_per_side": args.pts_per_side,
        "merge_mode": args.merge_mode,
        "trigger_proxies": {
            "image_area": int(image.shape[0] * image.shape[1]),
            "high_frequency": high_frequency_score(image),
            "fast_cache": fast_summary,
        },
        "rescue_configs": rows,
        "existing_references": refs,
    }

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
