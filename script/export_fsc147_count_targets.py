"""Export a split-isolated FSC-147 count-target shard.

Only files named by the requested split/list are opened. A legacy cache can
provide its stored count, or an individual density-map file can provide the
integral (equivalent to the source dot count). The combined annotation JSON is
intentionally unsupported so non-selected GT cannot enter the process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


DEFAULT_SPLIT = Path("/home/czp/official_code/dataset/FSC147/Train_Test_Val_FSC_147.json")


def names_sha256(names: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(names)).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--split-key", choices=("train", "val", "test"), required=True)
    parser.add_argument("--images-file", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-cache", type=Path)
    source.add_argument("--density-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    split = json.loads(args.split_file.read_text())
    if args.images_file:
        names = list(json.loads(args.images_file.read_text()))
    else:
        names = list(split[args.split_key])
    if not names or not set(names).issubset(split[args.split_key]):
        raise RuntimeError("target names are not a subset of the declared split")

    targets: dict[str, int] = {}
    source_files = []
    for name in names:
        stem = Path(name).stem
        if args.source_cache:
            path = args.source_cache / f"{stem}.pt"
            sample = torch.load(path, map_location="cpu", weights_only=False)
            if "gt_count" not in sample:
                raise RuntimeError(f"cache has no gt_count: {path}")
            count = int(sample["gt_count"])
        else:
            path = args.density_dir / f"{stem}.npy"
            count = int(round(float(np.load(path).sum())))
        targets[name] = count
        source_files.append(str(path))

    output = {
        "schema": "fsc147-count-target-v1",
        "official_split": args.split_key,
        "split_file": str(args.split_file),
        "declared_images": len(names),
        "names_sha256": names_sha256(names),
        "nonselected_images_loaded": 0,
        "source_type": "legacy split cache gt_count" if args.source_cache else "density integral",
        "source_root": str(args.source_cache or args.density_dir),
        "source_files": source_files,
        "targets": targets,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(f"[save] split={args.split_key} images={len(names)} -> {args.out}")


if __name__ == "__main__":
    main()
