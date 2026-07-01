"""Train CosineCategoryHead on COCO data for P1-2 ablation.

用途：P1-2 分类头消融 — 在 COCO（通用域）上训练投影头，然后与
     FSC147-trained 投影头对比，检验 FSC147 domain adaptation 是否必要。

用法:
    # 训练 COCO 80-class head
    python script/train_category_coco.py \
        --data_dir /home/czp/ws_yiyang/ovcud_cache/coco_val_3view \
        --text_prototypes result/checkpoints/text_prototypes_coco80.pt \
        --out_ckpt result/checkpoints/category_coco80_cosine.pt

    # 训练 COCO→LVIS 映射 head (用 COCO 数据但映射到 LVIS 1203 类)
    # 需要先创建 COCO→LVIS class mapping
    python script/train_category_coco.py \
        --data_dir /home/czp/ws_yiyang/ovcud_cache/coco_val_3view \
        --text_prototypes /home/czp/ws_yiyang/ovcud_cache/text_prototypes_lvis1203.pt \
        --class_mapping result/checkpoints/coco_to_lvis_mapping.pt \
        --out_ckpt result/checkpoints/category_coco2lvis_cosine.pt
"""

from __future__ import annotations

import argparse, json, math, os, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Copy model defs from train_category_v2 (same architecture for fair comparison)
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


class CachedDataset(Dataset):
    def __init__(self, data_dir: str, files: list | None = None,
                 class_mapping: dict | None = None, min_purity: float = 0.0):
        if files is not None:
            self.files = list(files)
        else:
            self.files = sorted(Path(data_dir).glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"no .pt files in {data_dir}")
        self.class_mapping = class_mapping  # optional: COCO idx -> LVIS idx
        self._index = []
        self._cache = {}
        for fi, f in enumerate(self.files):
            d = torch.load(f, map_location="cpu")
            n = int(d["z"].shape[0])
            purity = d.get("purity", torch.ones(n))
            valid = d.get("valid", torch.ones(n))
            for i in range(n):
                if float(purity[i]) >= min_purity and float(valid[i]) > 0:
                    mc = int(d["matched_class"][i])
                    if self.class_mapping is not None:
                        mc = self.class_mapping.get(mc, -1)
                    if mc >= 0:
                        self._index.append((fi, i, mc))

    @property
    def num_classes_seen(self):
        classes = set()
        for fi, i, mc in self._index:
            classes.add(mc)
        return len(classes)

    def _load(self, fi):
        if fi not in self._cache:
            self._cache[fi] = torch.load(self.files[fi], map_location="cpu")
        return self._cache[fi]

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        fi, i, mc = self._index[idx]
        d = self._load(fi)
        return {
            "z": d["z"][i].float(),
            "matched_class": mc,
            "purity": float(d["purity"][i]),
            "valid": 1.0,
        }


def collate(batch):
    return {
        "z": torch.stack([b["z"] for b in batch]),
        "matched_class": torch.tensor([b["matched_class"] for b in batch], dtype=torch.long),
        "purity": torch.tensor([b["purity"] for b in batch]),
        "valid": torch.tensor([b["valid"] for b in batch]),
    }


def compute_loss(logits, targets, sample_weight, class_weight=None, label_smoothing=0.1, focal_gamma=0.0):
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


def build_coco_to_lvis_mapping(coco_categories_json, lvis_categories_json):
    """Build mapping from COCO class_idx (0-79) to LVIS contiguous_id (0-1202).

    COCO categories are a strict subset of LVIS, matched by name.
    """
    import json
    with open(coco_categories_json) as f:
        coco_cats = json.load(f)
    with open(lvis_categories_json) as f:
        lvis_cats = json.load(f)

    # Extract name lists
    if isinstance(coco_cats, dict):
        coco_names = [c["name"] for c in coco_cats.get("categories", coco_cats)]
    else:
        coco_names = [c["name"] if isinstance(c, dict) else c for c in coco_cats]

    if isinstance(lvis_cats, dict):
        lvis_list = lvis_cats.get("categories", lvis_cats)
    else:
        lvis_list = lvis_cats

    lvis_name_to_id = {}
    for c in lvis_list:
        name = c["name"] if isinstance(c, dict) else c
        cid = c.get("contiguous_id", c.get("id", 0)) if isinstance(c, dict) else 0
        lvis_name_to_id[name.lower().strip()] = cid

    mapping = {}
    matched = 0
    for coco_idx, name in enumerate(coco_names):
        lvis_id = lvis_name_to_id.get(name.lower().strip(), -1)
        if lvis_id >= 0:
            mapping[coco_idx] = lvis_id
            matched += 1

    print(f"[mapping] COCO→LVIS: {matched}/{len(coco_names)} categories matched")
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--text_prototypes", required=True)
    ap.add_argument("--out_ckpt", default="result/checkpoints/category_coco_cosine.pt")
    ap.add_argument("--class_mapping", default=None, help=".pt file with COCO→target class mapping dict")
    ap.add_argument("--proj_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.07)
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

    # Load class mapping if provided
    class_mapping = None
    if args.class_mapping:
        class_mapping = torch.load(args.class_mapping, map_location="cpu", weights_only=False)
        if isinstance(class_mapping, dict):
            class_mapping = {int(k): int(v) for k, v in class_mapping.items()}
            print(f"[mapping] Loaded {len(class_mapping)} entries")

    # Data
    all_files = sorted(Path(args.data_dir).glob("*.pt"))
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(all_files), generator=g).tolist()
    n_val = max(1, int(len(all_files) * args.val_ratio))
    val_files = [all_files[i] for i in perm[:n_val]]
    train_files = [all_files[i] for i in perm[n_val:]]
    train_ds = CachedDataset(args.data_dir, files=train_files, class_mapping=class_mapping)
    val_ds = CachedDataset(args.data_dir, files=val_files, class_mapping=class_mapping)
    print(f"[data] train: {len(train_ds)} candidates from {len(train_files)} images")
    print(f"[data] val:   {len(val_ds)} candidates from {len(val_files)} images")
    print(f"[data] train classes seen: {train_ds.num_classes_seen}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=args.num_workers,
                              pin_memory=(device != "cpu"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=args.num_workers,
                            pin_memory=(device != "cpu"))

    # Text prototypes
    tp = torch.load(args.text_prototypes, map_location="cpu", weights_only=False)
    tp = F.normalize(tp.float(), dim=-1).to(device)
    num_classes = tp.shape[0]
    print(f"[model] num_classes={num_classes}, proj_dim={args.proj_dim}")

    # Determine in_dim
    sample_z = train_ds[0]["z"]
    in_dim = sample_z.shape[0]
    print(f"[model] in_dim={in_dim}")

    # Model
    head = CosineCategoryHead(
        in_dim=in_dim, proj_dim=args.proj_dim,
        temperature=args.temperature, learnable_temp=True,
        dropout=args.dropout, num_layers=args.num_layers,
    ).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[model] params={n_params/1e6:.2f}M")

    # Optimizer
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(ep):
        if ep < args.warmup_epochs:
            return (ep + 1) / args.warmup_epochs
        progress = (ep - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # Training loop
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
            loss = compute_loss(logits, labels, w, label_smoothing=args.label_smoothing)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            n_batches += 1

        sched.step()
        avg_loss = total_loss / max(n_batches, 1)

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

    train_metrics = evaluate(head, train_loader, tp, device)
    print(f"Final train top1={train_metrics['top1']:.4f} top3={train_metrics['top3']:.4f}")
    print(f"Best val    top1={best_top1:.4f} (gap={train_metrics['top1']-best_top1:+.4f})")


if __name__ == "__main__":
    main()
