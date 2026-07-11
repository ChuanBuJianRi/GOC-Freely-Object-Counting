"""Build OmniCount 2x2 tiled caches from an image-only test manifest."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from code.encoders.dinov2_encoder import DINOv2RegionEncoder  # noqa: E402
from script.export_omnicount_strict_protocol import read_manifest  # noqa: E402
from script.preprocess_fsc147_tiled import build_sam2_amg  # noqa: E402
from script.preprocess_fsc147_tiled_nogt import process_image  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--omnicount-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tiles", type=int, default=2)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--pts-per-side", type=int, default=32)
    parser.add_argument(
        "--points-per-batch", type=int, default=64,
        help="SAM2 execution batch size; does not change the sampled point grid",
    )
    parser.add_argument("--merge-mode", choices=("mask", "bbox"), default="bbox")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid shard configuration")

    manifest = read_manifest(args.manifest)
    entries = manifest["entries"]
    shard_entries = entries[args.shard_index::args.num_shards]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[init] shard={args.shard_index}/{args.num_shards} "
        f"images={len(shard_entries)}/{len(entries)} annotations_loaded=0",
        flush=True,
    )
    amg = build_sam2_amg(args.device, args.pts_per_side)
    amg.points_per_batch = args.points_per_batch
    encoder = DINOv2RegionEncoder(device=args.device)
    print(f"[init] SAM2 points_per_batch={args.points_per_batch}", flush=True)
    processed = skipped = errors = 0
    started = time.time()

    for index, row in enumerate(shard_entries):
        stem = row["sample_id"]
        output_path = args.out_dir / f"{stem}.pt"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        image_path = args.omnicount_dir / row["relative_path"]
        try:
            image = np.array(Image.open(image_path).convert("RGB"))
            sample = process_image(
                image, row["file_name"], amg, encoder,
                args.tiles, args.overlap, args.merge_mode,
            )
            if set(sample) & {
                "valid", "purity", "coverage", "iou", "matched_class",
                "matched_instance_id", "gt_count", "class_name", "points",
            }:
                raise RuntimeError("image-only preprocessor emitted a forbidden GT field")
            torch.save(sample, output_path)
            processed += 1
        except Exception as error:
            errors += 1
            print(f"[error] {row['relative_path']}: {error}", flush=True)
        if (index + 1) % 25 == 0:
            print(
                f"[{index+1}/{len(shard_entries)}] processed={processed} "
                f"skipped={skipped} errors={errors} "
                f"elapsed={(time.time()-started)/60:.1f}m",
                flush=True,
            )
    print(
        f"[done] processed={processed} skipped={skipped} errors={errors} "
        f"elapsed={(time.time()-started)/60:.1f}m -> {args.out_dir}",
        flush=True,
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
