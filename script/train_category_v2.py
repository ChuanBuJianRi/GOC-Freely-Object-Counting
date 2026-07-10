"""改进版 Category Head 训练（cosine-only，强正则化）。

核心改动 vs 原版：
    1. 只用 TextPrototypeCosineHead（去除过拟合的 closed branch）
    2. 更高 dropout (0.3) + weight decay (1e-3)
    3. Label smoothing 0.1
    4. 按类别留出 held-out validation（真正的 open-vocab 评估）
    5. 支持 384 和 1152 维特征

用法：
    python script/train_category_v2.py \
        --data_dir /path/to/cache --text_prototypes /path/to/tp.pt \
        --out_ckpt result/checkpoints/category_cosine_v2.pt

预估时间（RTX 4090）：
    126K candidates, batch=512, 30 epochs → ~15 min
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# --------------------------------------------------------------------------- #
# 轻量 Projection Head
# --------------------------------------------------------------------------- #
class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 512, hidden_dim: int = 512,
                 num_layers: int = 2, dropout: float = 0.3):
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

    def forward(self, z):
        return F.normalize(self.mlp(z), dim=-1)


class CosineCategoryHead(nn.Module):
    """纯 cosine 分类头（开放词表，无可学习 closed branch）。"""
    def __init__(self, in_dim: int, proj_dim: int = 512, temperature: float = 0.07,
                 learnable_temp: bool = True, dropout: float = 0.3, num_layers: int = 2):
        super().__init__()
        self.proj = ProjectionHead(in_dim, proj_dim, dropout=dropout, num_layers=num_layers)
        if learnable_temp:
            self.log_temp = nn.Parameter(torch.tensor(math.log(temperature)))
        else:
            self.register_buffer("log_temp", torch.tensor(math.log(temperature)))

    def get_temp(self):
        return self.log_temp.exp().clamp(0.01, 0.5)

    def forward(self, z, text_prototypes):
        h = self.proj(z)  # [B, D]
        t = F.normalize(text_prototypes, dim=-1)  # [C, D]
        return (h @ t.t()) / self.get_temp()  # [B, C]


# --------------------------------------------------------------------------- #
# 数据集
# --------------------------------------------------------------------------- #
class CachedDataset(Dataset):
    def __init__(self, data_dir: str, files: Optional[list] = None, min_purity: float = 0.0):
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

    @staticmethod
    def split_by_class(data_dir: str, val_ratio: float = 0.15, seed: int = 42):
        files = sorted(Path(data_dir).glob("*.pt"))
        file_cls = {}
        for f in files:
            d = torch.load(f, map_location="cpu")
            file_cls[f] = d.get("class_name", "__unknown__")
        classes = sorted(set(file_cls.values()))
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(classes), generator=g).tolist()
        n_val = max(1, int(len(classes) * val_ratio))
        heldout = {classes[i] for i in perm[:n_val]}
        train_files = [f for f in files if file_cls[f] not in heldout]
        val_files = [f for f in files if file_cls[f] in heldout]
        return (CachedDataset(data_dir, files=train_files),
                CachedDataset(data_dir, files=val_files),
                sorted(heldout))

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


def filter_split_files(all_files, split_file: str | None, split_key: str):
    if not split_file:
        return all_files, None
    split_path = Path(split_file)
    split = json.loads(split_path.read_text())
    if split_key not in split:
        raise KeyError(f"split key {split_key!r} not found in {split_path}")
    allowed = {Path(name).stem for name in split[split_key]}
    selected = [path for path in all_files if path.stem in allowed]
    if not selected:
        raise RuntimeError(f"no cache files matched {split_key!r} in {split_path}")
    return selected, {
        "split_file": str(split_path),
        "split_key": split_key,
        "declared_images": len(allowed),
        "matched_cache_files": len(selected),
        "missing_cache_files": sorted(allowed - {path.stem for path in selected}),
    }


def file_id_sha256(files) -> str:
    payload = "\n".join(sorted(path.stem for path in files)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def collate(batch):
    return {
        "z": torch.stack([b["z"] for b in batch]),
        "matched_class": torch.tensor([b["matched_class"] for b in batch], dtype=torch.long),
        "purity": torch.tensor([b["purity"] for b in batch]),
        "valid": torch.tensor([b["valid"] for b in batch]),
    }


# --------------------------------------------------------------------------- #
# 损失
# --------------------------------------------------------------------------- #
def compute_loss(logits, targets, sample_weight, class_weight=None, label_smoothing=0.1, focal_gamma=0.0):
    """加权 CE + label smoothing + 可选 focal loss。"""
    nC = logits.size(1)
    smooth_targets = torch.zeros_like(logits)
    smooth_targets.fill_(label_smoothing / (nC - 1))
    smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - label_smoothing)

    log_p = F.log_softmax(logits, dim=-1)
    ce = -(smooth_targets * log_p).sum(dim=-1)

    if focal_gamma > 0:
        pt = torch.exp(-ce)
        ce = ((1 - pt) ** focal_gamma) * ce

    if class_weight is not None:
        ce = ce * class_weight[targets]

    denom = sample_weight.sum().clamp_min(1e-6)
    return (sample_weight * ce).sum() / denom


# --------------------------------------------------------------------------- #
# 评估
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(head, loader, text_proto, device):
    head.eval()
    correct = total = 0
    correct_top3 = 0
    for batch in loader:
        z = batch["z"].to(device)
        labels = batch["matched_class"].to(device)
        valid = (batch["valid"] > 0).to(device)
        logits = head(z, text_proto)
        pred = logits.argmax(dim=-1)
        top3 = logits.topk(min(3, logits.size(1)), dim=-1).indices
        m = valid
        correct += int(((pred == labels) & m).sum())
        correct_top3 += int(((top3 == labels.unsqueeze(1)).any(dim=1) & m).sum())
        total += int(m.sum())
    head.train()
    return {
        "top1": correct / max(total, 1),
        "top3": correct_top3 / max(total, 1),
        "n_valid": total,
    }


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="缓存目录（384 或 1152 维）")
    ap.add_argument("--text_prototypes", required=True, help="文本原型 .pt")
    ap.add_argument("--out_ckpt", default="result/checkpoints/category_cosine_v2.pt")
    ap.add_argument("--proj_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--min_purity", type=float, default=0.0, help="过滤 purity 低于此值的候选")
    ap.add_argument("--focal_gamma", type=float, default=0.0, help=">0 时使用 focal loss (gamma)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--warmup_epochs", type=int, default=3)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--split_seed", type=int, default=None,
        help="data split seed; defaults to --seed for backward compatibility",
    )
    ap.add_argument("--split_file", default=None, help="optional official split JSON")
    ap.add_argument("--split_key", default="train", help="key selected from --split_file")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    device = args.device

    # 数据：按图划分（让模型见所有类，与原始训练一致）
    import torch as _torch
    all_files = sorted(Path(args.data_dir).glob("*.pt"))
    all_files, official_split = filter_split_files(
        all_files, args.split_file, args.split_key
    )
    split_seed = args.seed if args.split_seed is None else args.split_seed
    g = _torch.Generator().manual_seed(split_seed)
    perm = _torch.randperm(len(all_files), generator=g).tolist()
    n_val = max(1, int(len(all_files) * args.val_ratio))
    val_files = [all_files[i] for i in perm[:n_val]]
    train_files = [all_files[i] for i in perm[n_val:]]
    train_ds = CachedDataset(args.data_dir, files=train_files, min_purity=args.min_purity)
    val_ds = CachedDataset(args.data_dir, files=val_files, min_purity=args.min_purity)
    heldout = []
    print(f"[data] train: {len(train_ds)} candidates from {len(train_files)} images")
    print(f"[data] val:   {len(val_ds)} candidates from {len(val_files)} images")
    print(f"[data] model_seed={args.seed} split_seed={split_seed}")
    if official_split:
        print(f"[data] official split: {official_split}")

    loader_gen = _torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=args.num_workers,
                              pin_memory=(device != "cpu"), generator=loader_gen)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=args.num_workers,
                            pin_memory=(device != "cpu"))

    # 文本原型
    tp = torch.load(args.text_prototypes, map_location="cpu", weights_only=False)
    tp = F.normalize(tp.float(), dim=-1).to(device)
    num_classes = tp.shape[0]
    print(f"[model] num_classes={num_classes}, proj_dim={args.proj_dim}")

    # 确定 in_dim
    sample_z = train_ds[0]["z"]
    in_dim = sample_z.shape[0]
    print(f"[model] in_dim={in_dim} (feature dimension)")

    # 模型
    head = CosineCategoryHead(
        in_dim=in_dim, proj_dim=args.proj_dim,
        temperature=args.temperature, learnable_temp=True,
        dropout=args.dropout, num_layers=args.num_layers,
    ).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[model] params={n_params/1e6:.2f}M")

    # 优化器
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # LR schedule: warmup + cosine
    def lr_lambda(ep):
        if ep < args.warmup_epochs:
            return (ep + 1) / args.warmup_epochs
        progress = (ep - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # 训练
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

            logits = head(z, tp)
            loss = compute_loss(logits, labels, w, label_smoothing=args.label_smoothing, focal_gamma=args.focal_gamma)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            n_batches += 1

        sched.step()
        avg_loss = total_loss / max(n_batches, 1)

        # 验证
        val_metrics = evaluate(head, val_loader, tp, device)
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
                "in_dim": in_dim, "proj_dim": args.proj_dim, "num_classes": num_classes,
                "val_top1": best_top1, "val_top3": val_metrics["top3"],
                "epoch": epoch + 1, "config": vars(args),
                "data_manifest": {
                    "official_split": official_split,
                    "split_seed": split_seed,
                    "selected_count": len(all_files),
                    "selected_sha256": file_id_sha256(all_files),
                    "train_count": len(train_files),
                    "train_sha256": file_id_sha256(train_files),
                    "model_val_count": len(val_files),
                    "model_val_sha256": file_id_sha256(val_files),
                },
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

    # 保存
    if best_state is not None:
        os.makedirs(os.path.dirname(args.out_ckpt) if os.path.dirname(args.out_ckpt) else ".", exist_ok=True)
        best_state["history"] = history
        torch.save(best_state, args.out_ckpt)
        print(f"Checkpoint saved to {args.out_ckpt}")

    # 最终在 training set 上测一次（检测过拟合程度）
    train_metrics = evaluate(head, train_loader, tp, device)
    print(f"Final train top1={train_metrics['top1']:.4f} top3={train_metrics['top3']:.4f}")
    print(f"Best val    top1={best_top1:.4f} (gap={train_metrics['top1']-best_top1:+.4f})")


if __name__ == "__main__":
    main()
