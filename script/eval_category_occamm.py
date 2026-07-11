"""在 OCCAM-M 测试缓存上评估已训分类头（不重训）。

加载 result/checkpoints/category_hybrid.pt + 文本原型，在 fsc147_test_occamm
（新候选配方，含 z + matched_class + purity + valid）上评估：

  - top1            : valid>0 候选上的分类准确率（对齐 train_category.evaluate 口径）
  - top5            : 同上 top5
  - purity 加权 top1: 按 purity*valid 加权（高质量候选权重更高）
  - 分类 oracle     : 在「候选→GT 匹配」已给定 matched_class 的前提下，
                      分类头对 valid 候选能达到的 top1 = 当前分类头的实际上限指标本身，
                      这里额外报告「若只看 purity>0.7 的高纯净候选」的 top1，作为
                      候选质量充分时分类头的近似 oracle 上界。

仅评估，不训练，不写权重。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.heads.category_head import build_category_head, CategoryHeadConfig
from code.training.train_category import forward_logits


def _infer_proj_kwargs(state: dict) -> dict:
    """从 checkpoint 的 ProjectionHead MLP 键推断 num_layers / 是否含 dropout。

    层序列：每个隐藏块 = Linear, GELU[, Dropout]，最后一层只有 Linear。
    含 dropout 时块跨度为 3（Linear@k, GELU@k+1, Dropout@k+2），否则为 2。
    用 open_head.proj.mlp 的最大 Linear 索引推断结构。
    """
    lin_idx = sorted(
        int(k.split(".")[3])
        for k in state
        if k.startswith("open_head.proj.mlp.") and k.endswith(".weight")
    )
    if not lin_idx:
        # 非 hybrid（cosine/margin/linear）：proj.mlp.*
        lin_idx = sorted(
            int(k.split(".")[2])
            for k in state
            if k.startswith("proj.mlp.") and k.endswith(".weight")
        )
    stride = 1
    if len(lin_idx) >= 2:
        stride = lin_idx[1] - lin_idx[0]   # 2=无dropout, 3=有dropout
    num_layers = len(lin_idx)
    dropout = 0.1 if stride >= 3 else 0.0
    return {"dropout": dropout, "num_layers": num_layers}


def load_head(ckpt_path: str, device: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    proj_kwargs = _infer_proj_kwargs(ck["head"])
    ht = ck["head_type"]
    if ht == "hybrid":
        extra = {"open_kwargs": {"proj_kwargs": proj_kwargs},
                 "closed_kwargs": {"proj_kwargs": proj_kwargs}}
    else:
        extra = {"proj_kwargs": proj_kwargs}
    cfg = CategoryHeadConfig(
        head_type=ht,
        in_dim=ck["in_dim"],
        proj_dim=ck["proj_dim"],
        num_classes=ck["num_classes"],
        extra=extra,
    )
    head = build_category_head(cfg)
    head.load_state_dict(ck["head"])
    head.to(device).eval()
    print(f"[load] inferred proj_kwargs={proj_kwargs}")
    return head, ck


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="result/checkpoints/category_hybrid.pt")
    ap.add_argument("--cache-dir", default="/home/czp/ws_yiyang/ovcud_cache/fsc147_test_occamm")
    ap.add_argument("--text-prototypes", default="result/checkpoints/text_prototypes_fsc147.pt")
    ap.add_argument("--images-file", default=None, help="可选：只评这些文件名（json 列表）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="result/logs/eval_category_occamm.json")
    args = ap.parse_args()

    head, ck = load_head(args.ckpt, args.device)
    print(f"[load] head={ck['head_type']} in_dim={ck['in_dim']} num_classes={ck['num_classes']} "
          f"训练时 eval_top1={ck['metrics'].get('eval_top1'):.4f}")

    tp = torch.load(args.text_prototypes, map_location="cpu", weights_only=False)
    tp = torch.nn.functional.normalize(tp.float(), dim=-1).to(args.device)

    wanted = set(json.load(open(args.images_file))) if args.images_file else None
    files = sorted(f for f in os.listdir(args.cache_dir) if f.endswith(".pt"))

    n_valid = 0
    top1 = top5 = 0
    w_sum = w_top1 = 0.0
    hp_n = hp_top1 = 0     # 高纯净 purity>0.7
    n_imgs = 0

    for fn in files:
        d = torch.load(os.path.join(args.cache_dir, fn), map_location="cpu", weights_only=False)
        file_name = d.get("file_name") or fn
        if wanted is not None and file_name not in wanted:
            continue
        z = d["z"].float().to(args.device)
        labels = torch.as_tensor(np.asarray(d["matched_class"]), dtype=torch.long, device=args.device)
        valid = torch.as_tensor(np.asarray(d["valid"]), dtype=torch.float32, device=args.device) > 0
        purity = torch.as_tensor(np.asarray(d["purity"]), dtype=torch.float32, device=args.device)
        if z.numel() == 0 or int(valid.sum()) == 0:
            continue
        n_imgs += 1

        with torch.no_grad():
            logits, _ = forward_logits(head, z, tp, None)
        pred = logits.argmax(dim=-1)
        top5_idx = logits.topk(min(5, logits.size(1)), dim=-1).indices

        m = valid
        nv = int(m.sum())
        n_valid += nv
        c1 = (pred == labels) & m
        top1 += int(c1.sum())
        in5 = (top5_idx == labels.unsqueeze(1)).any(dim=1) & m
        top5 += int(in5.sum())

        w = (purity * m.float())
        w_sum += float(w.sum())
        w_top1 += float((w * (pred == labels).float()).sum())

        hp = m & (purity > 0.7)
        hp_n += int(hp.sum())
        hp_top1 += int(((pred == labels) & hp).sum())

    res = {
        "n_images": n_imgs,
        "n_valid_cands": n_valid,
        "top1": top1 / max(n_valid, 1),
        "top5": top5 / max(n_valid, 1),
        "purity_weighted_top1": w_top1 / max(w_sum, 1e-9),
        "high_purity(>0.7)_n": hp_n,
        "high_purity(>0.7)_top1": hp_top1 / max(hp_n, 1),
        "train_eval_top1_ref": ck["metrics"].get("eval_top1"),
    }

    print()
    print(f"=== 已训 hybrid 分类头 @ OCCAM-M 测试缓存 ({args.cache_dir}) ===")
    print(f"  评估图数        : {res['n_images']}")
    print(f"  valid 候选数    : {res['n_valid_cands']}")
    print(f"  top1            : {res['top1']:.4f}   (训练时旧候选 eval_top1={res['train_eval_top1_ref']:.4f})")
    print(f"  top5            : {res['top5']:.4f}")
    print(f"  purity 加权 top1: {res['purity_weighted_top1']:.4f}")
    print(f"  高纯净>0.7 top1 : {res['high_purity(>0.7)_top1']:.4f}  (n={res['high_purity(>0.7)_n']})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\n结果写入 {args.out}")


if __name__ == "__main__":
    main()
