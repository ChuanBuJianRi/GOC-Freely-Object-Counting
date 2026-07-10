"""训练 1152-dim 关系头（适配统一三路特征）。

基于 ws_yiyang 的 train_relation.py 改写，去掉 DDP，适配新缓存格式。

预估时间 (RTX 4090, 6142 图, ~50 候选/图): ~30 min
"""

from __future__ import annotations

import argparse, hashlib, json, math, os, random, sys, time
from pathlib import Path
from typing import Optional

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.heads.relation_head import PairwiseRelationHead, RelationHeadConfig
from code.matrix.pairwise_features import build_pairwise_features, pairwise_feature_dim, box_geometry


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class RelationDataset(Dataset):
    def __init__(self, data_dir: str, files: Optional[list] = None, min_cand: int = 2):
        all_files = list(files) if files is not None else sorted(Path(data_dir).glob("*.pt"))
        self.files = [f for f in all_files if torch.load(f, map_location="cpu")["z"].shape[0] >= min_cand]
        if not self.files: raise RuntimeError("no image with >= min_cand candidates")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        d = torch.load(self.files[idx], map_location="cpu")
        return {
            "z": d["z"].float(), "matched_class": d["matched_class"].long(),
            "matched_instance_id": d["matched_instance_id"].long(),
            "valid": d["valid"].float(), "purity": d["purity"].float(),
            "coverage": d["coverage"].float(), "bbox": d["bbox"].float(),
        }


def collate_single(batch): return batch[0]


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Category head wrapper
# ---------------------------------------------------------------------------
@torch.no_grad()
def get_category_probs(head, z, tp, device):
    from script.train_category_v2 import CosineCategoryHead
    if isinstance(head, CosineCategoryHead):
        return torch.softmax(head(z.to(device), tp.to(device)), dim=-1).cpu()
    else:
        from code.training.train_category import forward_logits
        logits, _ = forward_logits(head, z.to(device), tp.to(device), None)
        return torch.softmax(logits, dim=-1).cpu()


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------
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

    box_g_ij = box_geometry(box[ii], box[jj])
    box_g_ji = box_geometry(box[jj], box[ii])
    geom_ij, geom_ji = box_g_ij, box_g_ji

    phi_ij = build_pairwise_features(z[ii], z[jj], p[ii], p[jj], box[ii], box[jj], geom=geom_ij)
    phi_ji = build_pairwise_features(z[jj], z[ii], p[jj], p[ii], box[jj], box[ii], geom=geom_ji)

    vi, vj = valid[ii] > 0.5, valid[jj] > 0.5
    y_sem = ((cls[ii] == cls[jj]) & vi & vj).float()
    same_inst = (inst[ii] == inst[jj]) & (inst[ii] >= 0) & (inst[jj] >= 0)
    y_inst = (same_inst & vi & vj).float()
    w = (purity[ii] * valid[ii]) * (purity[jj] * valid[jj])

    # 平衡负采样
    if neg_ratio > 0:
        valid_pair = (vi & vj).float() > 0.5
        pos = ((y_sem > 0.5) | (y_inst > 0.5)) & valid_pair
        neg = valid_pair & (~pos)
        n_pos = int(pos.sum())
        if n_pos > 0:
            neg_idx = neg.nonzero(as_tuple=True)[0]
            keep_neg = min(neg_idx.numel(), int(neg_ratio * n_pos))
            if neg_idx.numel() > keep_neg:
                perm = torch.randperm(neg_idx.numel(), generator=generator, device=device)[:keep_neg]
                neg_idx = neg_idx[perm]
            sel = torch.cat([pos.nonzero(as_tuple=True)[0], neg_idx])
            phi_ij, phi_ji = phi_ij[sel], phi_ji[sel]
            y_sem, y_inst = y_sem[sel], y_inst[sel]
            w = w[sel]

    return phi_ij, phi_ji, y_sem, y_inst, w


def run_epoch(rel_head, cat_head, tp, loader, z_dim, args, device, opt=None, gen=None):
    train = opt is not None
    rel_head.train(train)
    agg = {"loss": 0.0, "L_sem": 0.0, "L_inst": 0.0, "sem_acc": 0.0, "sem_rec": 0.0,
           "inst_acc": 0.0, "inst_rec": 0.0, "inst_prec": 0.0}
    # 累积混淆矩阵计数（跨 batch 微平均，正样本极稀疏，逐图平均不可靠）
    tp_i = fp_i = fn_i = tn_i = 0
    tp_s = fn_s = 0
    n_steps = 0
    for sample in loader:
        p = get_category_probs(cat_head, sample["z"], tp, device)
        nr = args.neg_ratio if train else 0.0
        built = build_pairs(sample, p, args.max_cand, args.max_pairs, device, gen, neg_ratio=nr)
        if built is None: continue
        phi_ij, phi_ji, y_sem, y_inst, w = built

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            P = phi_ij.shape[0]
            out = rel_head(torch.cat([phi_ij, phi_ji], dim=0))
            sem = 0.5 * (out["sem"][:P] + out["sem"][P:])
            inst = 0.5 * (out["inst"][:P] + out["inst"][P:])
            pw = torch.tensor(args.pos_weight, device=device) if args.pos_weight > 0 else None
            l_sem = F.binary_cross_entropy_with_logits(sem, y_sem, pos_weight=pw, reduction="none")
            l_sem = (w * l_sem).sum() / w.sum().clamp_min(1e-6)
            l_inst = F.binary_cross_entropy_with_logits(inst, y_inst, pos_weight=pw, reduction="none")
            l_inst = (w * l_inst).sum() / w.sum().clamp_min(1e-6)
            loss = args.lambda_sem * l_sem + args.lambda_inst * l_inst

        if train:
            opt.zero_grad(); loss.backward(); opt.step()

        for k, v in [("loss", loss), ("L_sem", l_sem), ("L_inst", l_inst)]:
            agg[k] += float(v.detach())
        # Metrics: 用 valid 掩码（w>0.5 因 purity 极小恒为空，见 diag_relation_metrics.py）
        with torch.no_grad():
            sm = w > 0
            if sm.sum() > 0:
                ip = (torch.sigmoid(inst[sm]) > 0.5); it = y_inst[sm] > 0.5
                tp_i += int((ip & it).sum()); fp_i += int((ip & ~it).sum())
                fn_i += int((~ip & it).sum()); tn_i += int((~ip & ~it).sum())
                sp = (torch.sigmoid(sem[sm]) > 0.5); st = y_sem[sm] > 0.5
                tp_s += int((sp & st).sum()); fn_s += int((~sp & st).sum())
        n_steps += 1

    out = {k: v / max(n_steps, 1) for k, v in agg.items()}
    # 微平均 precision/recall（正样本稀疏，微平均比逐图平均可靠）
    out["inst_rec"] = tp_i / max(tp_i + fn_i, 1)
    out["inst_prec"] = tp_i / max(tp_i + fp_i, 1)
    out["inst_acc"] = (tp_i + tn_i) / max(tp_i + fp_i + fn_i + tn_i, 1)
    out["sem_rec"] = tp_s / max(tp_s + fn_s, 1)
    out["inst_pos"] = tp_i + fn_i
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--category_ckpt", required=True)
    ap.add_argument("--text_prototypes", required=True)
    ap.add_argument("--save_ckpt", default="result/checkpoints/fsc147_relation_1152.pt")
    ap.add_argument("--pretrained", default=None, help="预训练权重路径（如 COCO pretrained）")
    ap.add_argument(
        "--require_train_only_vocabulary", action="store_true",
        help="reject pretrained initialization and non-train-only category vocabularies",
    )
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_cand", type=int, default=64)
    ap.add_argument("--max_pairs", type=int, default=4096)
    ap.add_argument("--lambda_sem", type=float, default=1.0)
    ap.add_argument("--lambda_inst", type=float, default=1.0)
    ap.add_argument("--neg_ratio", type=float, default=5.0)
    ap.add_argument("--pos_weight", type=float, default=3.0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--split_seed", type=int, default=None,
        help="data split seed; defaults to --seed for backward compatibility",
    )
    ap.add_argument("--split_file", default=None, help="optional official split JSON")
    ap.add_argument("--split_key", default="train", help="key selected from --split_file")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    gen = torch.Generator(device=args.device).manual_seed(args.seed)
    device = args.device

    # Data split (by file)
    all_files = sorted(Path(args.data_dir).glob("*.pt"))
    all_files, official_split = filter_split_files(
        all_files, args.split_file, args.split_key
    )
    split_seed = args.seed if args.split_seed is None else args.split_seed
    g = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(len(all_files), generator=g).tolist()
    n_val = max(1, int(len(all_files) * args.val_frac))
    val_files = [all_files[i] for i in perm[:n_val]]
    train_files = [all_files[i] for i in perm[n_val:]]
    train_ds = RelationDataset(args.data_dir, files=train_files)
    val_ds = RelationDataset(args.data_dir, files=val_files)
    loader_gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, collate_fn=collate_single,
        generator=loader_gen,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_single)
    print(
        f"[data] selected={len(all_files)} train={len(train_ds)} val={len(val_ds)} "
        f"model_seed={args.seed} split_seed={split_seed}"
    )
    if official_split:
        print(f"[data] official split: {official_split}")

    # Category head
    from script.train_category_v2 import CosineCategoryHead
    category_path = Path(args.category_ckpt)
    prototype_path = Path(args.text_prototypes)
    ck = torch.load(category_path, map_location=device, weights_only=False)
    vocabulary = ck.get("data_manifest", {}).get("prototype_vocabulary")
    if args.require_train_only_vocabulary:
        if args.pretrained:
            raise RuntimeError("strict train-only relation training forbids --pretrained")
        if not vocabulary or (
            vocabulary.get("official_split") != "train"
            or vocabulary.get("test_images_loaded") != 0
            or vocabulary.get("validation_images_loaded") != 0
            or vocabulary.get("nontrain_class_rows_retained") != 0
        ):
            raise RuntimeError("category checkpoint does not use a strict train-only vocabulary")
        if vocabulary.get("prototype_sha256") != file_sha256(prototype_path):
            raise RuntimeError("relation prototype differs from the category training prototype")
    cat_head = CosineCategoryHead(in_dim=ck["in_dim"], proj_dim=ck["proj_dim"], dropout=0.3, num_layers=2)
    cat_head.load_state_dict(ck["head"]); cat_head.to(device).eval()
    for p in cat_head.parameters(): p.requires_grad_(False)

    tp = F.normalize(
        torch.load(prototype_path, map_location=device, weights_only=False).float(), dim=-1
    )

    # Relation head
    z_dim = torch.load(train_ds.files[0], map_location="cpu")["z"].shape[1]
    feat_dim = pairwise_feature_dim(z_dim)
    rel_head = PairwiseRelationHead(RelationHeadConfig(
        feat_dim=feat_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout
    )).to(device)
    n_params = sum(p.numel() for p in rel_head.parameters())
    print(f"[model] relation head: feat_dim={feat_dim} params={n_params/1e6:.2f}M")

    if args.pretrained:
        pt_ck = torch.load(args.pretrained, map_location=device)
        rel_head.load_state_dict(pt_ck["relation_head"])
        print(f"[model] loaded pretrained weights from {args.pretrained}")

    opt = torch.optim.AdamW(rel_head.parameters(), lr=args.lr, weight_decay=1e-4)

    t0 = time.time()
    best_loss = float("inf")
    for epoch in range(args.epochs):
        tr = run_epoch(rel_head, cat_head, tp, train_loader, z_dim, args, device, opt, gen)
        ev = run_epoch(rel_head, cat_head, tp, val_loader, z_dim, args, device)

        msg = (f"[epoch {epoch+1:2d}/{args.epochs}] "
               f"train loss={tr['loss']:.4f} inst_P={tr['inst_prec']*100:.0f}% inst_R={tr['inst_rec']*100:.0f}% || "
               f"val loss={ev['loss']:.4f} inst_P={ev['inst_prec']*100:.0f}% inst_R={ev['inst_rec']*100:.0f}% "
               f"sem_R={ev['sem_rec']*100:.0f}% (inst_pos={int(ev['inst_pos'])})")
        print(msg)

        if ev["loss"] < best_loss:
            best_loss = ev["loss"]
            os.makedirs(os.path.dirname(args.save_ckpt) if os.path.dirname(args.save_ckpt) else ".", exist_ok=True)
            torch.save({
                "relation_head": rel_head.state_dict(), "optimizer": opt.state_dict(),
                "feat_dim": feat_dim, "z_dim": z_dim, "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers, "category_ckpt": args.category_ckpt,
                "epoch": epoch + 1, "val_metrics": ev,
                "pretrained_from": args.pretrained,
                "train_config": vars(args),
                "data_manifest": {
                    "official_split": official_split,
                    "split_seed": split_seed,
                    "selected_count": len(all_files),
                    "selected_sha256": file_id_sha256(all_files),
                    "partition_train_count": len(train_files),
                    "partition_model_val_count": len(val_files),
                    "train_count": len(train_ds.files),
                    "train_sha256": file_id_sha256(train_ds.files),
                    "model_val_count": len(val_ds.files),
                    "model_val_sha256": file_id_sha256(val_ds.files),
                    "training_assets": {
                        "category_checkpoint": str(category_path),
                        "category_checkpoint_sha256": file_sha256(category_path),
                        "text_prototypes": str(prototype_path),
                        "text_prototypes_sha256": file_sha256(prototype_path),
                        "prototype_vocabulary": vocabulary,
                    },
                },
            }, args.save_ckpt)
            print(f"  -> saved {args.save_ckpt}")

    print(f"\nDone in {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
