"""Train an FSC-147 candidate-validity predictor from official train dots only.

The target is the legacy cache ``valid`` label (a candidate covers at least one
training dot). At inference, the predictor replaces direct access to that label.
Official validation/test files are never loaded by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_SPLIT = Path("/home/czp/official_code/dataset/FSC147/Train_Test_Val_FSC_147.json")


def names_sha256(names: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(names)).encode("utf-8")).hexdigest()


def bbox_features(bbox: torch.Tensor, height: int, width: int) -> torch.Tensor:
    bbox = bbox.float()
    x, y, bw, bh = bbox.unbind(dim=1)
    w = max(float(width), 1.0)
    h = max(float(height), 1.0)
    area = (bw * bh) / (w * h)
    aspect = torch.log((bw + 1.0) / (bh + 1.0)).clamp(-5.0, 5.0)
    border = ((x <= 1.0) | (y <= 1.0) | (x + bw >= w - 1.0) | (y + bh >= h - 1.0)).float()
    return torch.stack((x / w, y / h, bw / w, bh / h, area, aspect, border), dim=1)


class CandidateFilter(nn.Module):
    def __init__(self, z_dim: int = 1152, geom_dim: int = 7, hidden_dim: int = 256,
                 dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(z_dim + geom_dim),
            nn.Linear(z_dim + geom_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((z, geom), dim=1)).squeeze(1)


def load_partition(cache_dir: Path, names: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    zs: list[torch.Tensor] = []
    geoms: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    used: list[str] = []
    for name in names:
        path = cache_dir / f"{Path(name).stem}.pt"
        if not path.exists():
            continue
        d = torch.load(path, map_location="cpu", weights_only=False)
        z = d["z"].float()
        if z.ndim != 2 or z.shape[0] == 0:
            continue
        valid = d["valid"].float()
        if len(valid) != len(z):
            raise RuntimeError(f"candidate/label length mismatch: {path}")
        zs.append(z)
        geoms.append(bbox_features(d["bbox"], int(d["height"]), int(d["width"])))
        labels.append((valid > 0).float())
        used.append(name)
    if not zs:
        raise RuntimeError(f"no training candidates loaded from {cache_dir}")
    return torch.cat(zs), torch.cat(geoms), torch.cat(labels), used


@torch.no_grad()
def evaluate(model: CandidateFilter, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    tp = fp = tn = fn = n = 0
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    for z, geom, label in loader:
        z, geom, label = z.to(device), geom.to(device), label.to(device)
        logits = model(z, geom)
        loss_sum += float(criterion(logits, label))
        pred = logits >= 0
        target = label > 0.5
        tp += int((pred & target).sum())
        fp += int((pred & ~target).sum())
        tn += int((~pred & ~target).sum())
        fn += int((~pred & target).sum())
        n += len(label)
    return {
        "loss": loss_sum / max(n, 1),
        "accuracy_at_0.5": (tp + tn) / max(n, 1),
        "precision_at_0.5": tp / max(tp + fp, 1),
        "recall_at_0.5": tp / max(tp + fn, 1),
        "n": n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=20260711)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--model-val-ratio", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)

    split = json.loads(args.split_file.read_text())
    official_train = list(split["train"])
    generator = torch.Generator().manual_seed(args.split_seed)
    permutation = torch.randperm(len(official_train), generator=generator).tolist()
    n_model_val = max(1, int(len(official_train) * args.model_val_ratio))
    model_val_names = [official_train[index] for index in permutation[:n_model_val]]
    train_names = [official_train[index] for index in permutation[n_model_val:]]

    print(f"[data] loading official-train candidates from {args.cache_dir}", flush=True)
    z_train, geom_train, y_train, used_train = load_partition(args.cache_dir, train_names)
    z_val, geom_val, y_val, used_val = load_partition(args.cache_dir, model_val_names)
    print(
        f"[data] train images={len(used_train)} candidates={len(y_train)} positive={float(y_train.mean()):.3f}; "
        f"model-val images={len(used_val)} candidates={len(y_val)} positive={float(y_val.mean()):.3f}",
        flush=True,
    )

    train_loader = DataLoader(
        TensorDataset(z_train, geom_train, y_train), batch_size=args.batch_size,
        shuffle=True, generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        TensorDataset(z_val, geom_val, y_val), batch_size=args.batch_size, shuffle=False,
    )
    model = CandidateFilter(
        z_dim=z_train.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss()
    best_loss = float("inf")
    best: dict | None = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        n_seen = 0
        for z, geom, label in train_loader:
            z, geom, label = z.to(args.device), geom.to(args.device), label.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(z, geom)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(label)
            n_seen += len(label)
        scheduler.step()
        val_metrics = evaluate(model, val_loader, args.device)
        row = {"epoch": epoch, "train_loss": loss_sum / max(n_seen, 1), **val_metrics}
        history.append(row)
        print(
            f"[epoch {epoch:02d}] train={row['train_loss']:.4f} val={row['loss']:.4f} "
            f"P={row['precision_at_0.5']:.3f} R={row['recall_at_0.5']:.3f}",
            flush=True,
        )
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best = {
                "candidate_filter": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "z_dim": int(z_train.shape[1]),
                "geom_dim": int(geom_train.shape[1]),
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "epoch": epoch,
                "model_val_metrics": val_metrics,
                "train_config": vars(args) | {"cache_dir": str(args.cache_dir), "split_file": str(args.split_file), "out": str(args.out)},
                "data_manifest": {
                    "official_split": "train",
                    "declared_images": len(official_train),
                    "train_image_count": len(used_train),
                    "model_val_image_count": len(used_val),
                    "train_names_sha256": names_sha256(used_train),
                    "model_val_names_sha256": names_sha256(used_val),
                    "split_seed": args.split_seed,
                    "label_source": "candidate covers >=1 official-train FSC dot",
                    "test_images_loaded": 0,
                },
                "history": history.copy(),
            }
    if best is None:
        raise RuntimeError("training produced no checkpoint")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best, args.out)
    print(f"[save] {args.out} best_epoch={best['epoch']} val_loss={best_loss:.4f}")


if __name__ == "__main__":
    main()
