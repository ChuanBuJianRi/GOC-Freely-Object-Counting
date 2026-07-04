"""Train a per-mask count bin classifier for P1: Counting as Closed-Set Classification.

The classifier predicts which count interval a SAM2 mask belongs to:
    {1}, {2-3}, {4-7}, {8-15}, {16+}

This addresses the over-merge problem where one mask covers multiple small objects.

Input: DINOv2 3-view features (1152-dim) + bbox geometric features (7-dim)
Output: 5-bin classification

Usage:
    python script/train_count_classifier.py \
        --labels /home/czp/ws_yiyang/ovcud_cache/p1_count_labels.pt \
        --out result/checkpoints/count_classifier.pt
"""

import argparse, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class CountBinClassifier(nn.Module):
    """Predict count bin from DINOv2 features + bbox geometric features."""

    def __init__(self, in_dim=1152, bbox_dim=7, hidden_dim=256, num_bins=5, dropout=0.3):
        super().__init__()
        total_in = in_dim + bbox_dim
        self.net = nn.Sequential(
            nn.Linear(total_in, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_bins),
        )
        self.num_bins = num_bins

    def forward(self, z, bbox_feats):
        x = torch.cat([z, bbox_feats], dim=-1)
        return self.net(x)

    def predict_bin(self, z, bbox_feats):
        """Predict bin indices and expected counts."""
        logits = self.forward(z, bbox_feats)
        probs = F.softmax(logits, dim=-1)
        bins = logits.argmax(dim=-1)
        return bins, probs


BIN_EXPECTED = torch.tensor([1.0, 2.5, 5.5, 11.5, 20.0], dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="/home/czp/ws_yiyang/ovcud_cache/p1_count_labels.pt")
    ap.add_argument("--out", default="result/checkpoints/count_classifier.pt")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[load] Loading labels from {args.labels}...")
    data = torch.load(args.labels, map_location="cpu", weights_only=False)
    z_all = data["z"].float()  # (N, 1152)
    bbox_all = data["bbox_feats"].float()  # (N, 7)
    bins_all = data["bins"].long()  # (N,)
    valid_all = data["valid"].float()  # (N,)
    counts_all = data["counts"].long()  # (N,)

    # Only use valid masks
    valid_mask = valid_all > 0.5
    z_all = z_all[valid_mask]
    bbox_all = bbox_all[valid_mask]
    bins_all = bins_all[valid_mask]
    counts_all = counts_all[valid_mask]

    n_total = z_all.shape[0]
    print(f"[data] {n_total} valid masks out of {valid_all.shape[0]} total")

    # Normalize bbox features
    bbox_mean = bbox_all.mean(dim=0, keepdim=True)
    bbox_std = bbox_all.std(dim=0, keepdim=True).clamp(min=1e-6)
    bbox_all = (bbox_all - bbox_mean) / bbox_std

    # Split train/val
    n_val = int(n_total * args.val_split)
    indices = torch.randperm(n_total)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    z_train, bbox_train, bins_train = z_all[train_idx], bbox_all[train_idx], bins_all[train_idx]
    z_val, bbox_val, bins_val = z_all[val_idx], bbox_all[val_idx], bins_all[val_idx]

    # Compute class weights for imbalanced bins
    bin_counts = torch.bincount(bins_train, minlength=5).float()
    class_weights = n_total / (5 * bin_counts)
    class_weights = class_weights.clamp(min=0.1, max=10.0)
    print(f"[data] Train: {z_train.shape[0]}, Val: {z_val.shape[0]}")
    print(f"[data] Bin distribution (train): {bin_counts.tolist()}")
    print(f"[data] Class weights: {class_weights.tolist()}")

    train_ds = TensorDataset(z_train, bbox_train, bins_train)
    val_ds = TensorDataset(z_val, bbox_val, bins_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # Model
    model = CountBinClassifier().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights.to(args.device))

    best_val_acc = 0
    best_epoch = 0

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        t0 = time.time()

        for z_b, bbox_b, bins_b in train_loader:
            z_b = z_b.to(args.device)
            bbox_b = bbox_b.to(args.device)
            bins_b = bins_b.to(args.device)

            logits = model(z_b, bbox_b)
            loss = ce_loss(logits, bins_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * z_b.size(0)
            preds = logits.argmax(dim=-1)
            train_correct += (preds == bins_b).sum().item()
            train_total += z_b.size(0)

        scheduler.step()
        train_acc = train_correct / max(train_total, 1)
        train_loss_avg = train_loss / max(train_total, 1)

        # Val
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for z_b, bbox_b, bins_b in val_loader:
                z_b = z_b.to(args.device)
                bbox_b = bbox_b.to(args.device)
                bins_b = bins_b.to(args.device)

                logits = model(z_b, bbox_b)
                loss = ce_loss(logits, bins_b)

                val_loss += loss.item() * z_b.size(0)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == bins_b).sum().item()
                val_total += z_b.size(0)

                all_val_preds.append(preds.cpu())
                all_val_labels.append(bins_b.cpu())

        val_acc = val_correct / max(val_total, 1)
        val_loss_avg = val_loss / max(val_total, 1)

        # Per-bin accuracy
        all_preds = torch.cat(all_val_preds)
        all_labels = torch.cat(all_val_labels)
        per_bin = {}
        for b in range(5):
            mask = all_labels == b
            if mask.sum() > 0:
                per_bin[b] = (all_preds[mask] == b).float().mean().item()
            else:
                per_bin[b] = 0.0

        dt = time.time() - t0
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch + 1

        marker = " *" if is_best else ""
        print(
            f"[e{epoch+1:3d}] "
            f"train_loss={train_loss_avg:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss_avg:.4f} val_acc={val_acc:.3f} "
            f"| bins: " + " ".join(f"b{b}={per_bin[b]:.2f}" for b in range(5))
            + f" | {dt:.1f}s{marker}"
        )

    print(f"\n[best] epoch={best_epoch}, val_acc={best_val_acc:.4f}")

    # Save checkpoint
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "bbox_mean": bbox_mean,
            "bbox_std": bbox_std,
            "in_dim": 1152,
            "bbox_dim": 7,
            "hidden_dim": 256,
            "num_bins": 5,
            "bin_names": ["{1}", "{2-3}", "{4-7}", "{8-15}", "{16+}"],
            "bin_expected": BIN_EXPECTED.tolist(),
            "val_acc": best_val_acc,
            "best_epoch": best_epoch,
        },
        args.out,
    )
    print(f"Saved to: {args.out}")


if __name__ == "__main__":
    main()
