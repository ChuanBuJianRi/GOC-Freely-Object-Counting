"""Build tiled FSC-147 candidate caches without loading annotations or labels."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.candidates.crops import build_three_crops
from code.encoders.dinov2_encoder import DINOv2RegionEncoder
from script.preprocess_fsc147_tiled import (
    bbox_iou_xywh,
    build_sam2_amg,
    compute_tiles,
    merge_masks_by_bbox,
    merge_masks_by_iou,
)


def empty_sample(name: str, height: int, width: int) -> dict:
    return {
        "schema": "fsc147-inference-v1",
        "img_id": Path(name).stem,
        "file_name": name,
        "z": torch.empty((0, 1152), dtype=torch.float32),
        "bbox": torch.empty((0, 4), dtype=torch.float32),
        "height": int(height),
        "width": int(width),
        "source_cache": "tiled-nogt-empty",
    }


def process_image(image: np.ndarray, name: str, amg, encoder: DINOv2RegionEncoder,
                  tiles: int, overlap: float, merge_mode: str) -> dict:
    height, width = image.shape[:2]
    masks: list[np.ndarray] = []
    bboxes: list[list[float]] = []
    for y1, x1, y2, x2 in compute_tiles(height, width, tiles, overlap):
        tile = image[y1:y2, x1:x2]
        tile_h, tile_w = tile.shape[:2]
        for result in amg.generate(tile):
            mask = np.asarray(result["segmentation"]).astype(np.uint8)
            area = float(mask.sum())
            area_ratio = area / max(tile_h * tile_w, 1)
            if area == 0 or not 1e-4 < area_ratio < 0.95:
                continue
            ys, xs = np.where(mask)
            if xs.size == 0:
                continue
            local_x1, local_y1 = int(xs.min()), int(ys.min())
            local_x2, local_y2 = int(xs.max()) + 1, int(ys.max()) + 1
            if local_x2 - local_x1 < 4 or local_y2 - local_y1 < 4:
                continue
            global_mask = np.zeros((height, width), dtype=np.uint8)
            global_mask[y1:y2, x1:x2] = mask
            masks.append(global_mask)
            bboxes.append([
                float(x1 + local_x1), float(y1 + local_y1),
                float(local_x2 - local_x1), float(local_y2 - local_y1),
            ])
    if not masks:
        return empty_sample(name, height, width)

    if merge_mode == "bbox":
        keep = merge_masks_by_bbox(bboxes, iou_thresh=0.7)
    else:
        keep = merge_masks_by_iou(masks, bboxes, iou_thresh=0.7)
    masks = [masks[index] for index in keep]
    bboxes = [bboxes[index] for index in keep]

    # A second, stricter pass matches the historical tiled recipe and remains
    # image-only because it uses candidate geometry/masks exclusively.
    order = sorted(range(len(masks)), key=lambda index: -float(masks[index].sum()))
    keep = []
    for index in order:
        duplicate = False
        for previous in keep:
            if merge_mode == "bbox":
                duplicate = bbox_iou_xywh(bboxes[index], bboxes[previous]) > 0.9
            else:
                intersection = float(np.logical_and(masks[index], masks[previous]).sum())
                union = float(np.logical_or(masks[index], masks[previous]).sum())
                duplicate = union > 0 and intersection / union > 0.9
            if duplicate:
                break
        if not duplicate:
            keep.append(index)
    keep.sort()
    masks = [masks[index] for index in keep]
    bboxes = [bboxes[index] for index in keep]
    if not masks:
        return empty_sample(name, height, width)

    masked_crops, box_crops, context_crops = [], [], []
    for mask, bbox in zip(masks, bboxes):
        x, y, box_w, box_h = bbox
        masked, box, context = build_three_crops(
            image, mask, (x, y, x + box_w, y + box_h)
        )
        masked_crops.append(masked)
        box_crops.append(box)
        context_crops.append(context)
    z = encoder.encode_views(masked_crops, box_crops, context_crops, batch_size=64)
    return {
        "schema": "fsc147-inference-v1",
        "img_id": Path(name).stem,
        "file_name": name,
        "z": z.float(),
        "bbox": torch.tensor(bboxes, dtype=torch.float32),
        "height": int(height),
        "width": int(width),
        "source_cache": f"tiled-nogt-{tiles}x{tiles}-overlap{overlap}-{merge_mode}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--split-key", choices=("val", "test"), required=True)
    parser.add_argument("--img-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tiles", type=int, default=2)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--pts-per-side", type=int, default=32)
    parser.add_argument("--merge-mode", choices=("mask", "bbox"), default="bbox")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid shard configuration")

    names = list(json.loads(args.split_file.read_text())[args.split_key])
    shard_names = names[args.shard_index::args.num_shards]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[init] split={args.split_key} shard={args.shard_index}/{args.num_shards} "
        f"images={len(shard_names)}/{len(names)}",
        flush=True,
    )
    amg = build_sam2_amg(args.device, args.pts_per_side)
    encoder = DINOv2RegionEncoder(device=args.device)
    processed = skipped = errors = 0
    started = time.time()
    for index, name in enumerate(shard_names):
        out = args.out_dir / f"{Path(name).stem}.pt"
        if out.exists():
            skipped += 1
            continue
        image_path = args.img_dir / name
        try:
            image = np.array(Image.open(image_path).convert("RGB"))
            sample = process_image(
                image, name, amg, encoder, args.tiles, args.overlap, args.merge_mode
            )
            torch.save(sample, out)
            processed += 1
        except Exception as error:
            errors += 1
            print(f"[error] {name}: {error}", flush=True)
        if (index + 1) % 25 == 0:
            print(
                f"[{index+1}/{len(shard_names)}] processed={processed} skipped={skipped} "
                f"errors={errors} elapsed={(time.time()-started)/60:.1f}m",
                flush=True,
            )
    print(
        f"[done] processed={processed} skipped={skipped} errors={errors} "
        f"elapsed={(time.time()-started)/60:.1f}m -> {args.out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
