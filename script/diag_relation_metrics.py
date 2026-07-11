"""诊断 val inst_rec=0 的根因：检查 same-instance 正样本数量、purity 权重掩码分布。

用法:
    python script/diag_relation_metrics.py --data_dir /home/czp/ws_yiyang/ovcud_cache/fsc147_train_fast
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--n_files", type=int, default=200)
    args = ap.parse_args()

    files = sorted(Path(args.data_dir).glob("*.pt"))[: args.n_files]
    tot_pairs = 0
    tot_inst_pos = 0
    tot_sem_pos = 0
    tot_w_gt_half = 0          # w>0.5 的对数
    tot_inst_pos_w_gt_half = 0  # w>0.5 且 same-inst 的对数
    tot_valid_cand = 0
    tot_cand = 0
    imgs_with_inst_pos = 0
    purity_stats = []

    for f in files:
        d = torch.load(f, map_location="cpu")
        cls = d["matched_class"].long()
        inst = d["matched_instance_id"].long()
        valid = d["valid"].float()
        purity = d["purity"].float()
        n = cls.shape[0]
        if n < 2:
            continue
        tot_cand += n
        tot_valid_cand += int((valid > 0.5).sum())
        purity_stats.append(purity)

        ii, jj = torch.triu_indices(n, n, offset=1)
        vi, vj = valid[ii] > 0.5, valid[jj] > 0.5
        same_inst = (inst[ii] == inst[jj]) & (inst[ii] >= 0) & (inst[jj] >= 0)
        y_inst = (same_inst & vi & vj)
        y_sem = (cls[ii] == cls[jj]) & vi & vj
        w = (purity[ii] * valid[ii]) * (purity[jj] * valid[jj])
        w_mask = w > 0.5

        tot_pairs += ii.numel()
        tot_inst_pos += int(y_inst.sum())
        tot_sem_pos += int(y_sem.sum())
        tot_w_gt_half += int(w_mask.sum())
        tot_inst_pos_w_gt_half += int((y_inst & w_mask).sum())
        if int(y_inst.sum()) > 0:
            imgs_with_inst_pos += 1

    purity_all = torch.cat(purity_stats)
    print(f"=== 诊断 {len(files)} 张图 ({args.data_dir}) ===")
    print(f"候选总数: {tot_cand}, valid 候选: {tot_valid_cand} ({tot_valid_cand/max(tot_cand,1)*100:.1f}%)")
    print(f"purity 分布: min={purity_all.min():.3f} mean={purity_all.mean():.3f} "
          f"max={purity_all.max():.3f}  >0.7 占比={float((purity_all>0.7).float().mean())*100:.1f}%")
    print(f"\n候选对总数: {tot_pairs}")
    print(f"  same-instance 正样本: {tot_inst_pos} ({tot_inst_pos/max(tot_pairs,1)*100:.3f}%)")
    print(f"  same-category 正样本: {tot_sem_pos} ({tot_sem_pos/max(tot_pairs,1)*100:.3f}%)")
    print(f"  含 inst 正样本的图: {imgs_with_inst_pos}/{len(files)}")
    print(f"\n=== w>0.5 掩码 (val 指标只在这些对上计算) ===")
    print(f"  w>0.5 的对: {tot_w_gt_half} ({tot_w_gt_half/max(tot_pairs,1)*100:.1f}%)")
    print(f"  w>0.5 且 same-inst 的对: {tot_inst_pos_w_gt_half} "
          f"({tot_inst_pos_w_gt_half/max(tot_w_gt_half,1)*100:.4f}% of w>0.5)")
    if tot_inst_pos_w_gt_half == 0:
        print("\n  [结论] w>0.5 掩码内 same-instance 正样本为 0 → inst_rec 恒为 0/1=0 是统计问题！")
    else:
        print(f"\n  [结论] w>0.5 掩码内有 {tot_inst_pos_w_gt_half} 个 inst 正样本，inst_rec 应可正常计算")


if __name__ == "__main__":
    main()
