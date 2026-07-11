"""Build an FSC-147 text vocabulary from official-train labels only.

Class IDs and names are discovered exclusively from official-train cache files.
Validation/test image labels, class mappings, and full-dataset prototype banks
are never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


DEFAULT_TEMPLATES = (
    "a photo of a {}.",
    "a close-up photo of a {}.",
    "a photo of a single {}.",
    "an image of a {}.",
    "a cropped photo of a {}.",
)


DEFAULT_SPLIT = Path("/home/czp/official_code/dataset/FSC147/Train_Test_Val_FSC_147.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--out-prototypes", type=Path, required=True)
    parser.add_argument("--out-metadata", type=Path, required=True)
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    split = json.loads(args.split_file.read_text())
    train_names = list(split["train"])
    global_id_to_name: dict[int, str] = {}
    used_names = []
    missing_names = []
    for name in train_names:
        path = args.train_cache / f"{Path(name).stem}.pt"
        if not path.exists():
            missing_names.append(name)
            continue
        sample = torch.load(path, map_location="cpu", weights_only=False)
        labels = {int(value) for value in sample["matched_class"].view(-1).tolist()}
        if len(labels) != 1:
            raise RuntimeError(f"expected one image-level class in {path}, got {sorted(labels)}")
        global_id = labels.pop()
        class_name = str(sample["class_name"])
        previous = global_id_to_name.setdefault(global_id, class_name)
        if previous != class_name:
            raise RuntimeError(
                f"inconsistent official-train class mapping for {global_id}: "
                f"{previous!r} vs {class_name!r}"
            )
        used_names.append(name)

    global_class_ids = sorted(global_id_to_name)
    if not global_class_ids:
        raise RuntimeError("no official-train classes were discovered")
    class_names = [global_id_to_name[global_id] for global_id in global_class_ids]
    import open_clip

    model = open_clip.create_model(
        args.model_name, pretrained=args.pretrained, device=args.device
    ).eval()
    tokenizer = open_clip.get_tokenizer(args.model_name)
    prompts = [
        template.format(class_name)
        for class_name in class_names
        for template in DEFAULT_TEMPLATES
    ]
    encoded = []
    with torch.no_grad():
        for start in range(0, len(prompts), args.batch_size):
            tokens = tokenizer(prompts[start:start + args.batch_size]).to(args.device)
            features = model.encode_text(tokens).float()
            encoded.append(torch.nn.functional.normalize(features, dim=-1).cpu())
    prompt_features = torch.cat(encoded).view(len(class_names), len(DEFAULT_TEMPLATES), -1)
    selected = torch.nn.functional.normalize(prompt_features.mean(dim=1), dim=-1)

    args.out_prototypes.parent.mkdir(parents=True, exist_ok=True)
    args.out_metadata.parent.mkdir(parents=True, exist_ok=True)
    torch.save(selected, args.out_prototypes)
    metadata = {
        "schema": "fsc147-train-vocabulary-v1",
        "official_split": "train",
        "split_file": str(args.split_file),
        "declared_train_images": len(train_names),
        "loaded_train_images": len(used_names),
        "missing_train_images": missing_names,
        "test_images_loaded": 0,
        "validation_images_loaded": 0,
        "prototype_generation": "direct OpenCLIP encoding of official-train class names",
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "prompt_templates": list(DEFAULT_TEMPLATES),
        "out_prototypes": str(args.out_prototypes),
        "out_prototypes_sha256": sha256(args.out_prototypes),
        "global_class_ids": global_class_ids,
        "categories": [
            {
                "local_id": local_id,
                "global_id": global_id,
                "name": global_id_to_name[global_id],
            }
            for local_id, global_id in enumerate(global_class_ids)
        ],
        "retained_class_count": len(global_class_ids),
        "nontrain_class_rows_retained": 0,
        "selection_rule": "global class IDs observed in official-train cache files only",
    }
    args.out_metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(
        f"[done] train_images={len(used_names)}/{len(train_names)} "
        f"classes={len(global_class_ids)} tensor={tuple(selected.shape)}"
    )
    print(f"[save] {args.out_prototypes}")
    print(f"[save] {args.out_metadata}")


if __name__ == "__main__":
    main()
