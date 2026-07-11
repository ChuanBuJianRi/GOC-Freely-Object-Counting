"""Export an image-only OmniCount manifest and an isolated count target shard.

The ``manifest`` phase scans JPEG files only and never opens annotation files.
The ``targets`` phase is separate and is used only after predictions are frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("/home/czp/official_code/dataset/omnicount/OmniCount-191")
CATEGORIES = ("Birds", "Fruits", "Pets", "Satellite", "Supermarket", "Urban", "Wild")
MANIFEST_SCHEMA = "omnicount-test-image-manifest-v1"
TARGET_SCHEMA = "omnicount-test-count-target-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    entries = payload.get("entries", [])
    if (
        payload.get("schema") != MANIFEST_SCHEMA
        or payload.get("split") != "test"
        or payload.get("annotation_files_opened") != 0
        or payload.get("ground_truth_values_retained") != 0
        or payload.get("image_count") != len(entries)
    ):
        raise RuntimeError(f"invalid image-only manifest: {path}")
    sample_ids = [str(row["sample_id"]) for row in entries]
    file_names = [str(row["file_name"]) for row in entries]
    relative_paths = [str(row["relative_path"]) for row in entries]
    if (
        len(sample_ids) != len(set(sample_ids))
        or len(file_names) != len(set(file_names))
        or len(relative_paths) != len(set(relative_paths))
    ):
        raise RuntimeError("manifest identifiers are not unique")
    return payload


def export_manifest(args: argparse.Namespace) -> None:
    entries: list[dict[str, str]] = []
    for category in CATEGORIES:
        test_dir = args.omnicount_dir / category / "test"
        if not test_dir.is_dir():
            raise FileNotFoundError(test_dir)
        image_paths = sorted(
            path for path in test_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
        )
        for path in image_paths:
            entries.append({
                "sample_id": path.stem,
                "file_name": path.name,
                "relative_path": path.relative_to(args.omnicount_dir).as_posix(),
            })

    sample_ids = [row["sample_id"] for row in entries]
    file_names = [row["file_name"] for row in entries]
    if len(entries) != args.expected_images:
        raise RuntimeError(f"expected {args.expected_images} JPEGs, found {len(entries)}")
    if len(sample_ids) != len(set(sample_ids)) or len(file_names) != len(set(file_names)):
        raise RuntimeError("OmniCount test filenames/stems are not globally unique")

    payload = {
        "schema": MANIFEST_SCHEMA,
        "dataset": "OmniCount-191",
        "split": "test",
        "image_count": len(entries),
        "dataset_root": str(args.omnicount_dir),
        "selection": "all .jpg/.jpeg files under the seven test directories",
        "annotation_files_opened": 0,
        "ground_truth_values_retained": 0,
        "prediction_gt_fields": [],
        "entries": entries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"[manifest] {len(entries)} image-only rows -> {args.out}")
    print(f"[manifest] sha256={sha256(args.out)}")


def export_targets(args: argparse.Namespace) -> None:
    manifest = read_manifest(args.manifest)
    entries = manifest["entries"]
    expected_by_file = {row["file_name"]: row for row in entries}
    targets: dict[str, dict[str, Any]] = {}
    annotation_assets = []

    for category in CATEGORIES:
        annotation_path = args.omnicount_dir / category / "test" / "_annotations.coco.json"
        annotation = json.loads(annotation_path.read_text())
        category_names = {int(row["id"]): str(row["name"]) for row in annotation["categories"]}
        image_names = {int(row["id"]): str(row["file_name"]) for row in annotation["images"]}
        counts_by_image: dict[int, Counter[str]] = {
            image_id: Counter() for image_id in image_names
        }
        for row in annotation["annotations"]:
            image_id = int(row["image_id"])
            class_id = int(row["category_id"])
            if image_id not in counts_by_image or class_id not in category_names:
                raise RuntimeError(f"invalid annotation reference in {annotation_path}")
            counts_by_image[image_id][category_names[class_id]] += 1

        for image_id, file_name in image_names.items():
            manifest_row = expected_by_file.get(file_name)
            if manifest_row is None:
                raise RuntimeError(f"annotation image is absent from manifest: {file_name}")
            relative_parent = Path(manifest_row["relative_path"]).parts[0]
            if relative_parent != category:
                raise RuntimeError(f"super-category mismatch for {file_name}")
            class_counts = counts_by_image[image_id]
            if not class_counts:
                raise RuntimeError(f"test image has no count target: {file_name}")
            sample_id = manifest_row["sample_id"]
            targets[sample_id] = {
                "file_name": file_name,
                "supercategory": category,
                "gt_total": int(sum(class_counts.values())),
                "gt_class_count": len(class_counts),
                "class_counts": dict(sorted(class_counts.items())),
            }
        annotation_assets.append({
            "path": str(annotation_path),
            "sha256": sha256(annotation_path),
            "images": len(image_names),
            "annotations": len(annotation["annotations"]),
        })

    expected_ids = {row["sample_id"] for row in entries}
    if set(targets) != expected_ids:
        raise RuntimeError(
            f"target shard mismatch: missing={sorted(expected_ids-set(targets))} "
            f"extra={sorted(set(targets)-expected_ids)}"
        )
    payload = {
        "schema": TARGET_SCHEMA,
        "dataset": "OmniCount-191",
        "split": "test",
        "image_count": len(targets),
        "manifest_path": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "prediction_process_must_not_load_this_file": True,
        "annotation_assets": annotation_assets,
        "targets": targets,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"[targets] {len(targets)} isolated rows -> {args.out}")
    print(f"[targets] sha256={sha256(args.out)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--omnicount-dir", type=Path, default=DEFAULT_ROOT)
    manifest_parser.add_argument("--expected-images", type=int, default=1957)
    manifest_parser.add_argument("--out", type=Path, required=True)

    target_parser = subparsers.add_parser("targets")
    target_parser.add_argument("--omnicount-dir", type=Path, default=DEFAULT_ROOT)
    target_parser.add_argument("--manifest", type=Path, required=True)
    target_parser.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    if args.phase == "manifest":
        export_manifest(args)
    else:
        export_targets(args)


if __name__ == "__main__":
    main()
