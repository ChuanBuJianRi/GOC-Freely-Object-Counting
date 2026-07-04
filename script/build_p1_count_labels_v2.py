"""Build per-mask count labels with UNIQUE dot-to-mask assignment (v2).

V1 issue: counting all dots inside each bbox caused double-counting from
overlapping masks. Multiple candidates overlapped the same dot, inflating
per-mask count labels.

V2 fix: each GT dot is assigned to exactly ONE mask — the smallest bbox
that contains it. This prevents double-counting and gives a better
approximation of true per-mask object count.

Usage:
    python script/build_p1_count_labels_v2.py \
        --cache-dir /home/czp/ws_yiyang/ovcud_cache/fsc147_train_fast \
        --ann /home/czp/official_code/dataset/FSC147/annotation_FSC147_384.json \
        --out /home/czp/ws_yiyang/ovcud_cache/p1_count_labels_v2.pt
"""

import argparse, json, os, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def assign_dots_to_masks_unique(dots, bboxes):
    """Assign each dot to exactly ONE mask (smallest bbox containing it).

    This prevents the double-counting problem of V1 where a dot inside
    overlapping bboxes was counted for all of them.

    Args:
        dots: (N, 2) array of dot coordinates [x, y]
        bboxes: (M, 4) array of bboxes [x, y, w, h]

    Returns:
        per_mask_counts: (M,) array of unique dot counts per mask
    """
    M = len(bboxes)
    per_mask_counts = np.zeros(M, dtype=np.int32)
    if M == 0 or len(dots) == 0:
        return per_mask_counts

    # Precompute bbox areas for tie-breaking
    areas = bboxes[:, 2] * bboxes[:, 3]

    for dot_idx in range(len(dots)):
        x, y = dots[dot_idx]

        # Find all masks containing this dot
        inside = (
            (bboxes[:, 0] <= x)
            & (bboxes[:, 0] + bboxes[:, 2] >= x)
            & (bboxes[:, 1] <= y)
            & (bboxes[:, 1] + bboxes[:, 3] >= y)
        )

        if inside.any():
            # Assign to the smallest bbox (most precise match)
            inside_indices = np.where(inside)[0]
            best_idx = inside_indices[areas[inside_indices].argmin()]
            per_mask_counts[best_idx] += 1

    return per_mask_counts


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
BIN_EXPECTED = [1.0, 2.5, 5.5, 11.5, 20.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/home/czp/ws_yiyang/ovcud_cache/fsc147_train_fast")
    ap.add_argument("--ann", default="/home/czp/official_code/dataset/FSC147/annotation_FSC147_384.json")
    ap.add_argument("--out", default="/home/czp/ws_yiyang/ovcud_cache/p1_count_labels_v2.pt")
    ap.add_argument("--max-images", type=int, default=0)
    args = ap.parse_args()

    ann = json.load(open(args.ann))
    cache_files = sorted([f for f in os.listdir(args.cache_dir) if f.endswith(".pt")])
    if args.max_images > 0:
        cache_files = cache_files[: args.max_images]

    print(f"Processing {len(cache_files)} cache files (V2: unique dot assignment)...")

    all_z = []
    all_bbox_feats = []
    all_bins = []
    all_counts = []
    all_valid = []
    all_img_ids = []

    n_masks_total = 0
    n_masks_with_dots = 0
    n_dots_assigned = 0
    n_dots_total = 0
    bin_counter = Counter()

    t_start = time.time()
    for i, fname in enumerate(cache_files):
        img_id = fname.replace(".pt", "")
        cache_path = os.path.join(args.cache_dir, fname)

        try:
            d = torch.load(cache_path, map_location="cpu", weights_only=False)
        except Exception:
            continue

        file_name = d.get("file_name", f"{img_id}.jpg")
        entry = ann.get(file_name)
        if entry is None:
            for key in ann:
                if key.startswith(img_id):
                    entry = ann[key]
                    break
        if entry is None:
            continue

        dots = np.array(entry["points"], dtype=np.float32)
        z = d["z"].float().numpy()
        bbox = d["bbox"].float().numpy()
        valid = d.get("valid", torch.ones(z.shape[0])).numpy()
        h_img, w_img = int(d["height"]), int(d["width"])

        n_dots_total += len(dots)

        # Unique dot-to-mask assignment (key change from V1)
        per_mask_counts = assign_dots_to_masks_unique(dots, bbox)
        n_dots_assigned += int(per_mask_counts.sum())

        n_masks = z.shape[0]
        for j in range(n_masks):
            n_masks_total += 1
            dot_count = int(per_mask_counts[j])

            # Bbox geometric features (same 7-dim as V1)
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
                f"dot_assign_rate={100*n_dots_assigned/max(n_dots_total,1):.1f}%, "
                f"rate={n_masks_total/dt:.0f} masks/s"
            )

    dt = time.time() - t_start
    print(f"\nDone: {len(cache_files)} images, {n_masks_total} masks, {dt:.1f}s")
    print(f"Total GT dots: {n_dots_total}")
    print(f"Dots assigned to a mask: {n_dots_assigned} ({100*n_dots_assigned/max(n_dots_total,1):.1f}%)")
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
        "version": "v2-unique-assignment",
    }

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    torch.save(result, args.out)
    print(f"\nSaved to: {args.out}")
    print(f"File size: {os.path.getsize(args.out) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
