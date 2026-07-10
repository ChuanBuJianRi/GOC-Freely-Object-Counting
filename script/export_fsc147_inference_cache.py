"""Export FSC-147 caches with an inference-only, GT-free schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


ALLOWED_FIELDS = {"schema", "img_id", "file_name", "z", "bbox", "height", "width", "source_cache"}
FORBIDDEN_FIELDS = {
    "valid", "purity", "coverage", "iou", "matched_class", "matched_instance_id",
    "gt_count", "class_name", "is_part", "is_countable",
}


def safe_sample(d: dict, source: Path, stem: str) -> dict:
    z = d["z"].float()
    bbox = d["bbox"].float()
    if z.ndim != 2 or bbox.shape != (len(z), 4):
        raise RuntimeError(f"invalid candidate tensors in {source}")
    sample = {
        "schema": "fsc147-inference-v1",
        "img_id": stem,
        "file_name": d.get("file_name", f"{stem}.jpg"),
        "z": z,
        "bbox": bbox,
        "height": int(d["height"]),
        "width": int(d["width"]),
        "source_cache": str(source),
    }
    if set(sample) - ALLOWED_FIELDS or set(sample) & FORBIDDEN_FIELDS:
        raise RuntimeError("inference schema contains forbidden fields")
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--fallback-dir", type=Path)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--split-key", choices=("train", "val", "test"))
    parser.add_argument("--images-file", type=Path, help="JSON filename list for a predeclared trigger set")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--allow-missing-as-empty", action="store_true")
    args = parser.parse_args()

    if args.images_file:
        names = list(json.loads(args.images_file.read_text()))
    elif args.split_file and args.split_key:
        split = json.loads(args.split_file.read_text())
        names = list(split[args.split_key])
    else:
        parser.error("provide --images-file or both --split-file and --split-key")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    for index, name in enumerate(names):
        stem = Path(name).stem
        candidates = [args.source_dir / f"{stem}.pt"]
        if args.fallback_dir:
            candidates.append(args.fallback_dir / f"{stem}.pt")
        source = next((path for path in candidates if path.exists()), None)
        if source is None:
            missing.append(name)
            if not args.allow_missing_as_empty:
                continue
            raise RuntimeError(
                f"cannot create a dimensionally valid empty cache without a fallback: {name}"
            )
        d = torch.load(source, map_location="cpu", weights_only=False)
        torch.save(safe_sample(d, source, stem), args.out_dir / f"{stem}.pt")
        if (index + 1) % 500 == 0:
            print(f"[export] {index+1}/{len(names)}", flush=True)
    actual = {path.stem for path in args.out_dir.glob("*.pt")}
    expected = {Path(name).stem for name in names}
    if actual != expected:
        raise RuntimeError(
            f"inference cache mismatch: missing={sorted(expected-actual)} extra={sorted(actual-expected)}"
        )
    print(f"[done] {len(actual)} files -> {args.out_dir}; source-missing={missing}")


if __name__ == "__main__":
    main()
