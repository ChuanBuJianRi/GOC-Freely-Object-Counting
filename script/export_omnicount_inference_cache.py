"""Copy OmniCount image-derived tensors into the strict inference-only schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from script.export_fsc147_inference_cache import safe_sample  # noqa: E402
from script.export_omnicount_strict_protocol import read_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_manifest(args.manifest)
    entries = manifest["entries"]
    expected = {row["sample_id"] for row in entries}
    actual_source = {path.stem for path in args.source_dir.glob("*.pt")}
    if actual_source != expected:
        raise RuntimeError(
            f"source cache mismatch: missing={sorted(expected-actual_source)} "
            f"extra={sorted(actual_source-expected)}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(entries):
        stem = row["sample_id"]
        source = args.source_dir / f"{stem}.pt"
        data = torch.load(source, map_location="cpu", weights_only=False)
        sample = safe_sample(data, source, stem)
        if sample["file_name"] != row["file_name"]:
            raise RuntimeError(f"cache/manifest filename mismatch: {source}")
        torch.save(sample, args.out_dir / source.name)
        if (index + 1) % 500 == 0:
            print(f"[safe export] {index+1}/{len(entries)}", flush=True)

    actual = {path.stem for path in args.out_dir.glob("*.pt")}
    if actual != expected:
        raise RuntimeError("strict inference cache does not exactly match the manifest")
    print(f"[done] {len(actual)} inference-only files -> {args.out_dir}")


if __name__ == "__main__":
    main()
