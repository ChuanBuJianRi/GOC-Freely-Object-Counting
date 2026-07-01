"""Train LinearPrototypeHead (closed-set) on FSC147 for P1-2 C6 ablation.

C6: closed-set head vs text-prototype head → 验证 open-vocabulary 设定价值。
"""

from __future__ import annotations

import argparse, math, os, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 512, hidden_dim: int = 512,
                 num_layers: int = 2, dropout: float = 0.3, normalize: bool = False):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [proj_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)
        self.normalize = normalize

    def forward(self, z):
        h = self.mlp(z)
        if self.normalize:
            h = F.normalize(h, dim=-1)
        return h


class LinearPrototypeHead(nn.Module):
    """Closed-set classifier: logit_i = W · proj(z_i) + b. No text prototypes."""
    def __init__(self, in_dim: int, num_classes: int, proj_dim: int = 512,
                 bias: bool = True, dropout: float = 0.3, num_layers: int = 2):
        super().__init__()
        self.proj = ProjectionHead(in_dim, proj_dim, dropout=dropout, num_layers=num_layers, normalize=False)
        self.classifier = nn.Linear(proj_dim, num_classes, bias=bias)

    @property
    def num_classes(self):
        return self.classifier.out_features

    def forward(self, z, text_prototypes=None):
        h = self.proj(z)
        return self.classifier(h)


class CachedDataset(Dataset):
    def __init__(self, data_dir: str, files: list | None = None, min_purity: float = 0.0):
        if files is not None:
            self.files = list(files)
        else:
            self.files = sorted(Path(data_dir).glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"no .pt files in {data_dir}")
        self._index = []
        self._cache = {}
        for fi, f in enumerate(self.files):
            d = torch.load(f, map_location="cpu")
            n = int(d["z"].shape[0])
            purity = d.get("purity", torch.ones(n))
            for i in range(n):
                if float(purity[i]) >= min_purity:
                    self._index.append((fi, i))

    @property
    def num_classes_seen(self):
        classes = set()
        for fi, i in self._index:
            d = self._load(fi)
            classes.add(int(d["matched_class"][i]))
        return len(classes)

    def _load(self, fi):
        if fi not in self._cache:
            self._cache[fi] = torch.load(self.files[fi], map_location="cpu")
        return self._cache[fi]

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        fi, i = self._index[idx]
        d = self._load(fi)
        return {
            "z": d["z"][i].float(),
            "matched_class": int(d["matched_class"][i]),
            "purity": float(d["purity"][i]),
            "valid": float(d["valid"][i]),
        }


def collate(batch):
    return {
        "z": torch.stack([b["z"] for b in batch]),
        "matched_class": torch.tensor([b["matched_class"] for b in batch], dtype=torch.long),
        "purity": torch.tensor([b["purity"] for b in batch]),
        "valid": torch.tensor([b["valid"] for b in batch]),
    }


def compute_loss(logits, targets, sample_weight, label_smoothing=0.1):
    nC = logits.size(1)
    smooth_targets = torch.zeros_like(logits)
    smooth_targets.fill_(label_smoothing / (nC - 1))
    smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - label_smoothing)
    log_p = F.log_softmax(logits, dim=-1)
    ce = -(smooth_targets * log_p).sum(dim=-1)
    denom = sample_weight.sum().clamp_min(1e-6)
    return (sample_weight * ce).sum() / denom


@torch.no_grad()
def evaluate(head, loader, device):
    head.eval()
    correct = total = 0
    correct_top3 = 0
    for batch in loader:
        z = batch["z"].to(device)
        labels = batch["matched_class"].to(device)
        valid = (batch["valid"] > 0).to(device)
        logits = head(z)
        pred = logits.argmax(dim=-1)
        top3 = logits.topk(min(3, logits.size(1)), dim=-1).indices
        m = valid
        correct += int(((pred == labels) & m).sum())
        correct_top3 += int(((top3 == labels.unsqueeze(1)).any(dim=1) & m).sum())
        total += int(m.sum())
    head.train()
    return {"top1": correct / max(total, 1), "top3": correct_top3 / max(total, 1), "n_valid": total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_ckpt", default="result/checkpoints/category_fsc147_linear.pt")
    ap.add_argument("--num_classes", type=int, default=147)
    ap.add_argument("--proj_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--warmup_epochs", type=int, default=3)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device

    all_files = sorted(Path(args.data_dir).glob("*.pt"))
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(all_files), generator=g).tolist()
    n_val = max(1, int(len(all_files) * args.val_ratio))
    val_files = [all_files[i] for i in perm[:n_val]]
    train_files = [all_files[i] for i in perm[n_val:]]
    train_ds = CachedDataset(args.data_dir, files=train_files)
    val_ds = CachedDataset(args.data_dir, files=val_files)
    print(f"[data] train: {len(train_ds)} candidates from {len(train_files)} images")
    print(f"[data] val:   {len(val_ds)} candidates from {len(val_files)} images")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=args.num_workers,
                              pin_memory=(device != "cpu"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=args.num_workers,
                            pin_memory=(device != "cpu"))

    sample_z = train_ds[0]["z"]
    in_dim = sample_z.shape[0]
    print(f"[model] in_dim={in_dim}, num_classes={args.num_classes}")

    head = LinearPrototypeHead(
        in_dim=in_dim, num_classes=args.num_classes, proj_dim=args.proj_dim,
        dropout=args.dropout, num_layers=args.num_layers,
    ).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[model] params={n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(ep):
        if ep < args.warmup_epochs:
            return (ep + 1) / args.warmup_epochs
        progress = (ep - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_top1 = 0.0
    best_state = None
    no_improve = 0
    history = []

    t0 = time.time()
    for epoch in range(args.epochs):
        head.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            z = batch["z"].to(device)
            labels = batch["matched_class"].to(device)
            w = (batch["purity"] * batch["valid"]).to(device)
            logits = head(z)
            loss = compute_loss(logits, labels, w, label_smoothing=args.label_smoothing)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        sched.step()
        avg_loss = total_loss / max(n_batches, 1)
        val_metrics = evaluate(head, val_loader, device)
        lr_now = opt.param_groups[0]["lr"]
        msg = (f"[epoch {epoch+1:2d}/{args.epochs}] loss={avg_loss:.4f} "
               f"val_top1={val_metrics['top1']:.4f} val_top3={val_metrics['top3']:.4f} "
               f"lr={lr_now:.2e}")
        history.append({"epoch": epoch+1, "loss": avg_loss, **val_metrics})
        cur = val_metrics["top1"]
        if cur > best_top1:
            best_top1 = cur
            no_improve = 0
            best_state = {
                "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                "in_dim": in_dim, "proj_dim": args.proj_dim, "num_classes": args.num_classes,
                "val_top1": best_top1, "val_top3": val_metrics["top3"],
                "epoch": epoch + 1, "config": vars(args),
            }
            msg += " *"
        else:
            no_improve += 1
        print(msg)
        if args.patience > 0 and no_improve >= args.patience:
            print(f"Early stop at epoch {epoch+1} (best val_top1={best_top1:.4f})")
            break

    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed/60:.1f} min, best val_top1={best_top1:.4f}")

    if best_state is not None:
        os.makedirs(os.path.dirname(args.out_ckpt) if os.path.dirname(args.out_ckpt) else ".", exist_ok=True)
        best_state["history"] = history
        torch.save(best_state, args.out_ckpt)
        print(f"Checkpoint saved to {args.out_ckpt}")

    train_metrics = evaluate(head, train_loader, device)
    print(f"Final train top1={train_metrics['top1']:.4f} top3={train_metrics['top3']:.4f}")
    print(f"Best val    top1={best_top1:.4f} (gap={train_metrics['top1']-best_top1:+.4f})")


if __name__ == "__main__":
    main()
