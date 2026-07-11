"""COCO 预训练关系头：利用实例分割 mask 提供精确 same-instance 标签。

与 FSC147 dot-based 弱监督相比：
    - same-instance: GT instance_id (from mask IoU matching) → 精确
    - same-category: GT category_id → 精确
    - part-whole: mask containment + same-instance → 精确软标签

预训练后 fine-tune 到 FSC147。

用法:
    python script/train_relation_coco.py --data_dir ... --out_ckpt result/checkpoints/coco_relation_1152.pt
"""

from __future__ import annotations

import argparse, math, os, sys, time
from pathlib import Path
from typing import Optional

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from code.heads.relation_head import PairwiseRelationHead, RelationHeadConfig
from code.matrix.pairwise_features import build_pairwise_features, pairwise_feature_dim, box_geometry


class RelationDataset(Dataset):
    def __init__(self, data_dir, files=None, min_cand=4):
        all_files = list(files) if files else sorted(Path(data_dir).glob("*.pt"))
        self.files = [f for f in all_files if torch.load(f, map_location="cpu")["z"].shape[0] >= min_cand]
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        d = torch.load(self.files[idx], map_location="cpu")
        return {"z": d["z"].float(), "matched_class": d["matched_class"].long(),
                "matched_instance_id": d["matched_instance_id"].long(),
                "valid": d["valid"].float(), "purity": d["purity"].float(),
                "coverage": d["coverage"].float(), "bbox": d["bbox"].float()}
def collate_single(batch): return batch[0]


def build_pairs(sample, cat_probs, max_cand, max_pairs, device, generator, neg_ratio=0.0):
    z = sample["z"].to(device); cls = sample["matched_class"].to(device)
    inst = sample["matched_instance_id"].to(device); valid = sample["valid"].to(device)
    purity = sample["purity"].to(device); coverage = sample["coverage"].to(device)
    box = sample["bbox"].to(device); p = cat_probs.to(device)
    n = z.shape[0]

    if n > max_cand:
        keep = torch.topk(purity, min(max_cand, n)).indices
        z, cls, inst, valid, purity, coverage, box, p = (
            z[keep], cls[keep], inst[keep], valid[keep], purity[keep], coverage[keep], box[keep], p[keep])
        n = len(keep)

    ii, jj = torch.triu_indices(n, n, offset=1, device=device)
    if ii.numel() == 0: return None
    if ii.numel() > max_pairs:
        sel = torch.randperm(ii.numel(), generator=generator, device=device)[:max_pairs]
        ii, jj = ii[sel], jj[sel]

    geom_ij = box_geometry(box[ii], box[jj])
    geom_ji = box_geometry(box[jj], box[ii])
    phi_ij = build_pairwise_features(z[ii], z[jj], p[ii], p[jj], box[ii], box[jj], geom=geom_ij)
    phi_ji = build_pairwise_features(z[jj], z[ii], p[jj], p[ii], box[jj], box[ii], geom=geom_ji)

    vi, vj = valid[ii] > 0.5, valid[jj] > 0.5
    both_valid = (vi & vj)

    # COCO: 精确 same-category 标签
    y_sem = ((cls[ii] == cls[jj]) & vi & vj).float()

    # COCO: 精确 same-instance 标签 (GT instance_id from mask matching)
    same_inst = (inst[ii] == inst[jj]) & (inst[ii] >= 0) & (inst[jj] >= 0)
    y_inst = (same_inst & vi & vj).float()

    # COCO: 精确 part-whole 软标签
    # i 是 j 的部件 = same_inst * containment_i_in_j * completeness_gap
    cont_i_in_j = geom_ij[:, 1]  # bbox containment proxy
    completeness_gap = (coverage[jj] - coverage[ii]).clamp_min(0.0)
    y_part = same_inst.float() * cont_i_in_j * completeness_gap

    w = (purity[ii] * valid[ii]) * (purity[jj] * valid[jj])

    # 负采样
    if neg_ratio > 0:
        valid_pair = both_valid
        pos = ((y_sem > 0.5) | (y_inst > 0.5)) & valid_pair
        neg = valid_pair & (~pos)
        n_pos = int(pos.sum())
        if n_pos > 0:
            neg_idx = neg.nonzero(as_tuple=True)[0]
            keep_neg = min(neg_idx.numel(), int(neg_ratio * n_pos))
            if neg_idx.numel() > keep_neg:
                pm = torch.randperm(neg_idx.numel(), generator=generator, device=device)[:keep_neg]
                neg_idx = neg_idx[pm]
            sel = torch.cat([pos.nonzero(as_tuple=True)[0], neg_idx])
            phi_ij, phi_ji = phi_ij[sel], phi_ji[sel]
            y_sem, y_inst, y_part = y_sem[sel], y_inst[sel], y_part[sel]
            w = w[sel]

    return phi_ij, phi_ji, y_sem, y_inst, y_part, w


def run_epoch(rel_head, cat_probs_fn, loader, z_dim, args, device, opt=None, gen=None):
    train = opt is not None; rel_head.train(train)
    agg = {"loss": 0.0, "L_sem": 0.0, "L_inst": 0.0, "L_part": 0.0,
           "sem_pos": 0.0, "inst_pos": 0.0}
    n_steps = 0
    for sample in loader:
        p = cat_probs_fn(sample).to(device)
        nr = args.neg_ratio if train else 0.0
        built = build_pairs(sample, p, args.max_cand, args.max_pairs, device, gen, neg_ratio=nr)
        if built is None: continue
        phi_ij, phi_ji, y_sem, y_inst, y_part, w = built
        P = phi_ij.shape[0]

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            out = rel_head(torch.cat([phi_ij, phi_ji], dim=0))
            sem = 0.5 * (out["sem"][:P] + out["sem"][P:])
            inst = 0.5 * (out["inst"][:P] + out["inst"][P:])
            part = out["part"][:P]
            pw = torch.tensor(args.pos_weight, device=device) if args.pos_weight > 0 else None
            l_sem = F.binary_cross_entropy_with_logits(sem, y_sem, pos_weight=pw, reduction="none")
            l_sem = (w * l_sem).sum() / w.sum().clamp_min(1e-6)
            l_inst = F.binary_cross_entropy_with_logits(inst, y_inst, pos_weight=pw, reduction="none")
            l_inst = (w * l_inst).sum() / w.sum().clamp_min(1e-6)
            l_part = F.binary_cross_entropy_with_logits(part, y_part, reduction="none")
            l_part = (w * l_part).sum() / w.sum().clamp_min(1e-6)
            loss = args.lambda_sem * l_sem + args.lambda_inst * l_inst + args.lambda_part * l_part

        if train: opt.zero_grad(); loss.backward(); opt.step()

        for k, v in [("loss", loss), ("L_sem", l_sem), ("L_inst", l_inst), ("L_part", l_part)]:
            agg[k] += float(v.detach())
        with torch.no_grad():
            sm = w > 0.5
            if sm.sum() > 0:
                agg["sem_pos"] += float(y_sem[sm].mean())
                agg["inst_pos"] += float(y_inst[sm].mean())
        n_steps += 1
    return {k: v / max(n_steps, 1) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--save_ckpt", default="result/checkpoints/coco_relation_1152.pt")
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_cand", type=int, default=64)
    ap.add_argument("--max_pairs", type=int, default=4096)
    ap.add_argument("--lambda_sem", type=float, default=1.0)
    ap.add_argument("--lambda_inst", type=float, default=1.0)
    ap.add_argument("--lambda_part", type=float, default=1.0)
    ap.add_argument("--neg_ratio", type=float, default=5.0)
    ap.add_argument("--pos_weight", type=float, default=2.0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed); gen = torch.Generator(device=args.device).manual_seed(args.seed)
    device = args.device

    all_files = sorted(Path(args.data_dir).glob("*.pt"))
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(all_files), generator=g).tolist()
    n_val = max(1, int(len(all_files)*args.val_frac))
    val_files = [all_files[i] for i in perm[:n_val]]
    train_files = [all_files[i] for i in perm[n_val:]]
    train_ds = RelationDataset(args.data_dir, files=train_files)
    val_ds = RelationDataset(args.data_dir, files=val_files)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_single)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_single)
    print(f"[data] train={len(train_ds)} val={len(val_ds)}")

    # 用 GT one-hot 作为 category probs（oracle）
    # 扫描所有文件获取实际类别数
    n_cls = 0
    for f in train_ds.files:
        mc = torch.load(f, map_location="cpu")["matched_class"]
        if mc.numel() > 0:
            n_cls = max(n_cls, int(mc.max()))
    n_cls += 1
    print(f"[model] using ORACLE category probs ({n_cls} classes)")
    def oracle_cat_probs(sample):
        cls = sample["matched_class"]
        probs = torch.zeros(len(cls), n_cls)
        for i, c in enumerate(cls.tolist()):
            if c >= 0: probs[i, c] = 1.0
        return probs

    z_dim = torch.load(train_ds.files[0], map_location="cpu")["z"].shape[1]
    feat_dim = pairwise_feature_dim(z_dim)
    rel_head = PairwiseRelationHead(RelationHeadConfig(
        feat_dim=feat_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout
    )).to(device)
    print(f"[model] relation head: feat_dim={feat_dim} params={sum(p.numel() for p in rel_head.parameters())/1e6:.2f}M")

    opt = torch.optim.AdamW(rel_head.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    t0 = time.time(); best_loss = float("inf")
    for epoch in range(args.epochs):
        tr = run_epoch(rel_head, oracle_cat_probs, train_loader, z_dim, args, device, opt, gen)
        ev = run_epoch(rel_head, oracle_cat_probs, val_loader, z_dim, args, device)
        sched.step()
        msg = (f"[epoch {epoch+1:2d}/{args.epochs}] "
               f"train loss={tr['loss']:.4f} sem_pos={tr['sem_pos']*100:.1f}% inst_pos={tr['inst_pos']*100:.1f}% || "
               f"val loss={ev['loss']:.4f} sem_pos={ev['sem_pos']*100:.1f}% inst_pos={ev['inst_pos']*100:.1f}%")
        print(msg)
        if ev["loss"] < best_loss:
            best_loss = ev["loss"]
            os.makedirs(os.path.dirname(args.save_ckpt) if os.path.dirname(args.save_ckpt) else ".", exist_ok=True)
            torch.save({
                "relation_head": rel_head.state_dict(), "feat_dim": feat_dim, "z_dim": z_dim,
                "hidden_dim": args.hidden_dim, "num_layers": args.num_layers,
                "epoch": epoch+1, "val_metrics": ev,
            }, args.save_ckpt)
            print(f"  -> saved {args.save_ckpt}")
    print(f"\nDone in {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
