"""分支隔离评估：定位 hybrid 分类头在未见测试类 top1=0 的根因。

hybrid: score = alpha * open_logits + (1-alpha) * closed_logits
  - open_head  : 文本原型 cosine（开放词表，理论上能泛化到未见类）
  - closed_head: linear 分类器（只学过训练类，未见类必错）

对同一批 valid 候选分别用 full / open_only / closed_only 打分算 top1：
  - open_only 显著 > full  -> closed 分支拖累未见类，优化方向是压制/去掉 closed
  - open_only 也低         -> 特征/原型层面问题（看真值类在 open 的平均排名）
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

from code.heads.category_head import HybridCategoryHead  # noqa: E402
import eval_category_occamm as E  # noqa: E402


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="result/checkpoints/category_hybrid.pt")
    ap.add_argument("--cache-dir", default="/home/czp/ws_yiyang/ovcud_cache/fsc147_test_occamm")
    ap.add_argument("--text-prototypes", default="result/checkpoints/text_prototypes_fsc147.pt")
    ap.add_argument("--images-file", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--out", default="result/logs/eval_head_branches.json")
    args = ap.parse_args()

    head, ck = E.load_head(args.ckpt, args.device)
    assert isinstance(head, HybridCategoryHead), "本脚本针对 hybrid 头"
    alpha = float(head.get_alpha().detach())
    print(f"[load] hybrid alpha={alpha:.4f} num_closed={head.num_closed_classes}")

    tp = torch.load(args.text_prototypes, map_location="cpu", weights_only=False)
    tp = torch.nn.functional.normalize(tp.float(), dim=-1).to(args.device)

    wanted = set(json.load(open(args.images_file))) if args.images_file else None
    files = sorted(f for f in os.listdir(args.cache_dir) if f.endswith(".pt"))
    if args.limit > 0:
        files = files[: args.limit]

    nv = 0
    hit = {"full": 0, "open": 0, "closed": 0}
    open_top5 = 0
    rank_sum = 0
    for fn in files:
        d = torch.load(os.path.join(args.cache_dir, fn), map_location="cpu", weights_only=False)
        file_name = d.get("file_name") or fn
        if wanted is not None and file_name not in wanted:
            continue
        z = d["z"].float().to(args.device)
        labels = torch.as_tensor(np.asarray(d["matched_class"]), dtype=torch.long, device=args.device)
        valid = torch.as_tensor(np.asarray(d["valid"]), dtype=torch.float32, device=args.device) > 0
        if z.numel() == 0 or int(valid.sum()) == 0:
            continue

        open_logits = head.open_head(z, tp)
        closed = head.closed_head(z)
        C = open_logits.size(1)
        if C > closed.size(1):
            pad = z.new_zeros(z.size(0), C - closed.size(1))
            closed_full = torch.cat([closed, pad], dim=1)
        else:
            closed_full = closed
        full = alpha * open_logits + (1 - alpha) * closed_full

        m = valid
        nv += int(m.sum())
        for name, lg in [("full", full), ("open", open_logits), ("closed", closed_full)]:
            hit[name] += int(((lg.argmax(-1) == labels) & m).sum())

        top5 = open_logits.topk(min(5, C), dim=-1).indices
        open_top5 += int(((top5 == labels.unsqueeze(1)).any(1) & m).sum())
        order = open_logits.argsort(dim=-1, descending=True)
        rank = (order == labels.unsqueeze(1)).float().argmax(dim=-1)
        rank_sum += float((rank.float() * m.float()).sum())

    res = {
        "alpha": alpha,
        "n_valid": nv,
        "top1_full": hit["full"] / max(nv, 1),
        "top1_open_only": hit["open"] / max(nv, 1),
        "top1_closed_only": hit["closed"] / max(nv, 1),
        "top5_open_only": open_top5 / max(nv, 1),
        "open_mean_gt_rank": rank_sum / max(nv, 1),
    }
    print()
    print(f"=== branch isolation @ {args.cache_dir} (valid={nv}) ===")
    print(f"  alpha(open weight)   : {res['alpha']:.4f}")
    print(f"  top1 full            : {res['top1_full']:.4f}")
    print(f"  top1 open_only(a=1)  : {res['top1_open_only']:.4f}")
    print(f"  top1 closed_only     : {res['top1_closed_only']:.4f}")
    print(f"  top5 open_only       : {res['top5_open_only']:.4f}")
    print(f"  gt rank in open(mean): {res['open_mean_gt_rank']:.1f}  (0=best)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\nresult -> {args.out}")


if __name__ == "__main__":
    main()
