"""Build per-mask count labels for P1: Counting as Closed-Set Classification.

For each training image's SAM2 masks, count how many GT dots fall inside each mask's bbox.
This serves as weak supervision for training a per-mask count bin classifier.

The classifier addresses the over-merge problem: when a SAM2 mask covers multiple
small objects (common in dense scenes), the pipeline incorrectly counts it as 1.

Usage:
    python script/build_p1_count_labels.py \
        --cache-dir /home/czp/ws_yiyang/ovcud_cache/fsc147_train_fast \
        --ann /home/czp/official_code/dataset/FSC147/annotation_FSC147_384.json \
        --out /home/czp/ws_yiyang/ovcud_cache/p1_count_labels.pt
"""

import argparse, json, os, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def count_dots_in_bbox(dots_xy, bbox_xywh):
    """Count how many dots fall inside a bbox (xywh format).

    Args:
        dots_xy: (N, 2) array of dot coordinates [x, y]
        bbox_xywh: (4,) array [x, y, w, h]

    Returns:
        int: number of dots inside the bbox
    """
    x, y, w, h = bbox_xywh
    inside = (
        (dots_xy[:, 0] >= x)
        & (dots_xy[:, 0] <= x + w)
        & (dots_xy[:, 1] >= y)
        & (dots_xy[:, 1] <= y + h)
    )
    return int(inside.sum())


def dot_count_to_bin(count: int) -> int:
    """Map exact dot count to bin index.

    Bins:
        0: {1}       — single object
        1: {2-3}     — small cluster
        2: {4-7}     — medium cluster
        3: {8-15}    — large cluster
        4: {16+}     — very large cluster
    """
    if count <= 1:
        return 0
    elif count <= 3:
        return 1
    elif count <= 7:
        return 2
    elif count <= 15:
        return 3
    else:
        return 4


BIN_NAMES = ["{1}", "{2-3}", "{4-7}", "{8-15}", "{16+}"]
BIN_EXPECTED = [1.0, 2.5, 5.5, 11.5, 20.0]  # Expected count per bin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/home/czp/ws_yiyang/ovcud_cache/fsc147_train_fast")
    ap.add_argument("--ann", default="/home/czp/official_code/dataset/FSC147/annotation_FSC147_384.json")
    ap.add_argument("--out", default="/home/czp/ws_yiyang/ovcud_cache/p1_count_labels.pt")
    ap.add_argument("--max-images", type=int, default=0, help="Limit images for debugging (0=all)")
    args = ap.parse_args()

    ann = json.load(open(args.ann))
    cache_files = sorted(
        [f for f in os.listdir(args.cache_dir) if f.endswith(".pt")]
    )
    if args.max_images > 0:
        cache_files = cache_files[: args.max_images]

    print(f"Processing {len(cache_files)} cache files...")

    all_z = []  # DINOv2 features
    all_bbox_feats = []  # Bbox geometric features
    all_bins = []  # Count bin labels
    all_counts = []  # Exact dot counts (for analysis)
    all_valid = []  # Whether the mask is valid
    all_img_ids = []  # Debug: which image

    n_masks_total = 0
    n_masks_with_dots = 0
    bin_counter = Counter()

    t_start = time.time()
    for i, fname in enumerate(cache_files):
        img_id = fname.replace(".pt", "")
        cache_path = os.path.join(args.cache_dir, fname)

        try:
            d = torch.load(cache_path, map_location="cpu", weights_only=False)
        except Exception:
            continue

        # Find annotation entry
        file_name = d.get("file_name", f"{img_id}.jpg")
        entry = ann.get(file_name)
        if entry is None:
            # Try matching by numeric ID
            for key in ann:
                if key.startswith(img_id):
                    entry = ann[key]
                    break
        if entry is None:
            continue

        dots = np.array(entry["points"], dtype=np.float32)  # (N_dots, 2)
        z = d["z"].float().numpy()  # (N_masks, 1152)
        bbox = d["bbox"].float().numpy()  # (N_masks, 4) xywh
        valid = d.get("valid", torch.ones(z.shape[0])).numpy()
        h_img, w_img = int(d["height"]), int(d["width"])

        n_masks = z.shape[0]
        for j in range(n_masks):
            n_masks_total += 1
            dot_count = count_dots_in_bbox(dots, bbox[j])

            # Bbox geometric features
            area = bbox[j, 2] * bbox[j, 3]
            aspect = bbox[j, 2] / max(bbox[j, 3], 1e-6)
            norm_x = bbox[j, 0] / max(w_img, 1)
            norm_y = bbox[j, 1] / max(h_img, 1)
            norm_w = bbox[j, 2] / max(w_img, 1)
            norm_h = bbox[j, 3] / max(h_img, 1)

            bbox_feat = np.array(
                [area, aspect, norm_x, norm_y, norm_w, norm_h,
                 np.log(max(area, 1.0))],
                dtype=np.float32,
            )

            bin_idx = dot_count_to_bin(dot_count)

            all_z.append(z[j])
            all_bbox_feats.append(bbox_feat)
            all_bins.append(bin_idx)
            all_counts.append(dot_count)
            all_valid.append(float(valid[j] > 0))
            all_img_ids.append(img_id)

            if dot_count > 0:
                n_masks_with_dots += 1
            bin_counter[bin_idx] += 1

        if (i + 1) % 500 == 0:
            dt = time.time() - t_start
            print(
                f"  [{i+1}/{len(cache_files)}] "
                f"masks={n_masks_total}, with_dots={n_masks_with_dots} "
                f"({100*n_masks_with_dots/max(n_masks_total,1):.1f}%), "
                f"rate={n_masks_total/dt:.0f} masks/s"
            )

    dt = time.time() - t_start
    print(f"\nDone: {len(cache_files)} images, {n_masks_total} masks, {dt:.1f}s")
    print(f"Masks with >=1 dot: {n_masks_with_dots} ({100*n_masks_with_dots/max(n_masks_total,1):.1f}%)")
    print(f"\nBin distribution:")
    for bin_idx in range(5):
        pct = 100 * bin_counter[bin_idx] / max(n_masks_total, 1)
        print(f"  {BIN_NAMES[bin_idx]}: {bin_counter[bin_idx]} ({pct:.1f}%)")

    # Save
    result = {
        "z": torch.from_numpy(np.stack(all_z, axis=0)).float(),
        "bbox_feats": torch.from_numpy(np.stack(all_bbox_feats, axis=0)).float(),
        "bins": torch.tensor(all_bins, dtype=torch.long),
        "counts": torch.tensor(all_counts, dtype=torch.long),
        "valid": torch.tensor(all_valid, dtype=torch.float32),
        "img_ids": all_img_ids,
        "bin_names": BIN_NAMES,
        "bin_expected": BIN_EXPECTED,
        "n_images": len(cache_files),
        "n_masks": n_masks_total,
    }

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    torch.save(result, args.out)
    print(f"\nSaved to: {args.out}")
    print(f"File size: {os.path.getsize(args.out) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
