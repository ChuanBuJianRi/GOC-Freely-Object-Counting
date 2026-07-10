"""Run the train-only, validation-calibrated CP multi-seed experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SPLIT_FILE = Path("/home/czp/official_code/dataset/FSC147/Train_Test_Val_FSC_147.json")
CACHE_ROOT = Path("/home/czp/ws_yiyang/ovcud_cache")
PRETRAINED = REPO / "result/checkpoints/coco_relation_1152.pt"
TEXT_PROTO = REPO / "result/checkpoints/text_prototypes_fsc147.pt"


def run_logged(command: list[str], log_path: Path, dry_run: bool) -> None:
    print("\n$ " + " ".join(command), flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 73])
    parser.add_argument("--split-seed", type=int, default=20260710)
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=REPO / "result/checkpoints/cp_strict")
    parser.add_argument("--log-dir", type=Path,
                        default=REPO / "result/logs/cp_strict_training")
    parser.add_argument(
        "--stages", nargs="+", choices=("category", "relation", "val-cache", "eval"),
        default=["category", "relation", "val-cache", "eval"],
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    categories = {
        "pts16": {
            "cache": CACHE_ROOT / "fsc147_train_fast",
            "checkpoint": args.checkpoint_dir / "category_pts16_trainonly.pt",
        },
        "pts32": {
            "cache": CACHE_ROOT / "fsc147_train_pts32",
            "checkpoint": args.checkpoint_dir / "category_pts32_trainonly.pt",
        },
    }

    if "category" in args.stages:
        for resolution, config in categories.items():
            checkpoint = config["checkpoint"]
            if checkpoint.exists() and not args.force:
                print(f"[skip] {checkpoint}")
                continue
            command = [
                python, "script/train_category_v2.py",
                "--data_dir", str(config["cache"]),
                "--text_prototypes", str(TEXT_PROTO),
                "--out_ckpt", str(checkpoint),
                "--proj_dim", "512", "--dropout", "0.3", "--num_layers", "2",
                "--temperature", "0.07", "--label_smoothing", "0.1",
                "--epochs", "40", "--batch_size", "256", "--lr", "0.001",
                "--weight_decay", "0.001", "--warmup_epochs", "3",
                "--val_ratio", "0.15", "--patience", "15",
                "--seed", "42", "--split_seed", str(args.split_seed),
                "--split_file", str(SPLIT_FILE), "--split_key", "train",
                "--num_workers", "4", "--deterministic", "--device", args.device,
            ]
            run_logged(command, args.log_dir / f"category_{resolution}.log", args.dry_run)

    if "relation" in args.stages:
        for model_seed in args.seeds:
            for resolution, config in categories.items():
                if not config["checkpoint"].exists() and not args.dry_run:
                    raise FileNotFoundError(config["checkpoint"])
                for condition in ("cp", "scratch"):
                    checkpoint = (
                        args.checkpoint_dir
                        / f"relation_{resolution}_{condition}_seed{model_seed}.pt"
                    )
                    if checkpoint.exists() and not args.force:
                        print(f"[skip] {checkpoint}")
                        continue
                    command = [
                        python, "script/train_relation_1152.py",
                        "--data_dir", str(config["cache"]),
                        "--category_ckpt", str(config["checkpoint"]),
                        "--text_prototypes", str(TEXT_PROTO),
                        "--save_ckpt", str(checkpoint),
                        "--hidden_dim", "512", "--num_layers", "3", "--dropout", "0.1",
                        "--max_cand", "64", "--max_pairs", "4096",
                        "--lambda_sem", "1.0", "--lambda_inst", "1.0",
                        "--neg_ratio", "5.0", "--pos_weight", "8.0",
                        "--epochs", "40", "--lr", "0.0005", "--val_frac", "0.1",
                        "--seed", str(model_seed), "--split_seed", str(args.split_seed),
                        "--split_file", str(SPLIT_FILE), "--split_key", "train",
                        "--deterministic", "--device", args.device,
                    ]
                    if condition == "cp":
                        command.extend(["--pretrained", str(PRETRAINED)])
                    run_logged(
                        command,
                        args.log_dir / f"relation_{resolution}_{condition}_seed{model_seed}.log",
                        args.dry_run,
                    )

    if "val-cache" in args.stages:
        tiled_dir = CACHE_ROOT / "fsc147_val_tiled_51plus"
        multires_dir = CACHE_ROOT / "fsc147_val_multires_51plus"
        tiled_command = [
            python, "script/preprocess_fsc147_tiled.py",
            "--split-file", str(SPLIT_FILE), "--split-key", "val", "--min-count", "51",
            "--out-dir", str(tiled_dir), "--tiles", "2", "--overlap", "0.25",
            "--pts-per-side", "32", "--device", args.device,
        ]
        run_logged(tiled_command, args.log_dir / "validation_tiled_cache.log", args.dry_run)
        merge_command = [
            python, "script/build_multires_cache.py",
            "--cache-16", str(CACHE_ROOT / "fsc147_train_fast"),
            "--cache-32", str(tiled_dir),
            "--out-dir", str(multires_dir), "--iou-thresh", "0.5",
        ]
        run_logged(merge_command, args.log_dir / "validation_multires_merge.log", args.dry_run)

    if "eval" in args.stages:
        command = [
            python, "script/eval_cp_pretraining_multiseed.py",
            "--checkpoint-dir", str(args.checkpoint_dir),
            "--cache-root", str(CACHE_ROOT),
            "--val-multires-cache", str(CACHE_ROOT / "fsc147_val_multires_51plus"),
            "--split-file", str(SPLIT_FILE),
            "--annotation", str(
                Path("/home/czp/official_code/dataset/FSC147/annotation_FSC147_384.json")
            ),
            "--seeds", *[str(seed) for seed in args.seeds],
            "--device", args.device,
            "--out", str(REPO / "result/logs/cp_strict_multiseed.json.gz"),
        ]
        run_logged(command, args.log_dir / "evaluation.log", args.dry_run)

    manifest = {
        "seeds": args.seeds,
        "split_seed": args.split_seed,
        "split_file": str(SPLIT_FILE),
        "checkpoint_dir": str(args.checkpoint_dir),
        "stages": args.stages,
    }
    if not args.dry_run:
        (args.log_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
