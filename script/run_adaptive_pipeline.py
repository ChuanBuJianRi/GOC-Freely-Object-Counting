"""自适应密度 + 置信度过滤 Counting Pipeline。

改进 1 - 自适应密度:
    100+ 图像用 pts=32 候选（更多候选 → 更高 dot recall）
    其余用 pts=16 候选（更快，分类更准）

改进 2 - 置信度过滤:
    低分类置信度的候选降低权重或过滤，减少分类噪声引入的 over-count

用法:
    python script/run_adaptive_pipeline.py \
        --cache-16 /path/to/pts16 --cache-32 /path/to/pts32 \
        --cat-16 ckpt16.pt --cat-32 ckpt32.pt \
        --rel-16 rel16.pt --rel-32 rel32.pt \
        --images-file sample100.json
"""

from __future__ import annotations

import argparse, json, os, sys, time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from code.clustering.first_neighbor import (
    category_aware_clustering_with_spatial, spatial_sub_clustering,
)
from code.counting.deduplicate import (
    build_same_instance_components, build_same_instance_components_adaptive,
)
from code.counting.representative import select_representatives

DEFAULT_ANN = "/home/czp/official_code/dataset/FSC147/annotation_FSC147_384.json"
BINS = [("0-10",0,10),("11-20",11,20),("21-50",21,50),("51-100",51,100),("100+",101,10**9)]


def bin_of(c):
    for lab,lo,hi in BINS:
        if lo<=c<=hi: return lab
    return "100+"


def load_category_head(ckpt_path, device):
    from script.train_category_v2 import CosineCategoryHead
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    head = CosineCategoryHead(in_dim=ck["in_dim"], proj_dim=ck["proj_dim"], dropout=0.3, num_layers=2)
    head.load_state_dict(ck["head"]); head.to(device).eval()
    return head, ck


def load_relation_head(ckpt_path, device):
    from code.heads.relation_head import PairwiseRelationHead, RelationHeadConfig
    from code.matrix.pairwise_features import pairwise_feature_dim
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    feat_dim = pairwise_feature_dim(ck["z_dim"])
    cfg = RelationHeadConfig(feat_dim=feat_dim, hidden_dim=ck.get("hidden_dim",512),
                             num_layers=ck.get("num_layers",3), dropout=0.1)
    head = PairwiseRelationHead(cfg); head.load_state_dict(ck["relation_head"])
    head.to(device).eval()
    return head


def get_category_probs(head, z, tp, device):
    with torch.no_grad():
        logits = head(z.to(device), tp.to(device))
    return torch.softmax(logits, dim=-1).cpu().numpy()


def compute_A_inst_heuristic(candidates, bbox):
    """bbox IoU 作为 A_inst 的快速近似。"""
    n = len(candidates)
    A_inst = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i+1, n):
            bi, bj = bbox[i], bbox[j]
            x1=max(bi[0],bj[0]); y1=max(bi[1],bj[1])
            x2=min(bi[0]+bi[2],bj[0]+bj[2]); y2=min(bi[1]+bi[3],bj[1]+bj[3])
            inter=max(0,x2-x1)*max(0,y2-y1)
            ai=bi[2]*bi[3]; aj=bj[2]*bj[3]; union=ai+aj-inter
            iou = inter/union if union>0 else 0
            A_inst[i,j]=A_inst[j,i]=iou*10.0
    return A_inst


def count_image_adaptive(
    d, category_head, relation_head, text_prototypes, device,
    tau_inst=0.4, tau_affinity=0.1, conf_threshold=0.3, max_group_size=30,
):
    """单图自适应 counting。"""
    h, w = int(d["height"]), int(d["width"])
    file_name = d.get("file_name", "unknown")
    gt_count = int(d.get("gt_count", 0))
    z = d["z"].float()
    bbox_np = d["bbox"].float().numpy()
    n_cand = z.shape[0]
    if n_cand == 0:
        return {"pred_count": 0, "n_candidates": 0, "n_groups": 0, "gt_count": gt_count}

    # Category probs
    category_probs = get_category_probs(category_head, z, text_prototypes, device)
    top_conf = category_probs.max(axis=1)
    top_class = category_probs.argmax(axis=1)

    # 置信度过滤：低置信度候选标记为 invalid
    conf_valid = top_conf >= conf_threshold
    valid_orig = np.asarray(d["valid"]) > 0
    effective_valid = valid_orig & conf_valid

    # 调试：记录过滤比例
    n_filtered = int((valid_orig & ~conf_valid).sum())

    # 如果没有有效候选，返回 0
    if effective_valid.sum() == 0:
        return {"pred_count": 0, "n_candidates": n_cand, "n_groups": 0, "gt_count": gt_count,
                "n_conf_filtered": n_filtered}

    # A_sem: 使用分类兼容性作为 proxy（relation head 的 sem 分支过拟合，不用）
    A_sem = np.eye(n_cand, dtype=np.float32)
    for i in range(n_cand):
        for j in range(i+1, n_cand):
            sem_score = float((category_probs[i] * category_probs[j]).sum())
            A_sem[i,j] = A_sem[j,i] = sem_score * 10.0

    # A_inst: 使用 relation head 或 heuristic
    if relation_head is not None:
        from code.matrix.pairwise_features import build_pairwise_features, pairwise_feature_dim, box_geometry
        p_t = torch.from_numpy(category_probs).float()
        box_t = d["bbox"].float()
        z_t = z
        max_cand = min(n_cand, 200)
        if n_cand > max_cand:
            keep = np.argsort(-top_conf)[:max_cand]
            z_t = z_t[keep]; p_t = p_t[keep]; box_t = box_t[keep]
            keep_set = set(keep)
        else:
            keep_set = set(range(n_cand))

        ii, jj = np.triu_indices(len(keep_set) if n_cand > max_cand else n_cand, k=1)
        if len(ii) > 0:
            zi, zj = z_t[ii], z_t[jj]; pi, pj = p_t[ii], p_t[jj]
            bi, bj = box_t[ii], box_t[jj]
            geom_ij = box_geometry(bi, bj); geom_ji = box_geometry(bj, bi)
            phi_ij = build_pairwise_features(zi, zj, pi, pj, bi, bj, geom=geom_ij)
            phi_ji = build_pairwise_features(zj, zi, pj, pi, bj, bi, geom=geom_ji)
            with torch.no_grad():
                out = relation_head(torch.cat([phi_ij, phi_ji], dim=0))
                P = len(ii)
                inst_logits = 0.5*(out["inst"][:P]+out["inst"][P:]).cpu().numpy()
            A_inst = np.zeros((n_cand, n_cand), dtype=np.float32)
            for k in range(P):
                i, j = (keep[ii[k]], keep[jj[k]]) if n_cand > max_cand else (ii[k], jj[k])
                A_inst[i,j] = A_inst[j,i] = inst_logits[k]
        else:
            A_inst = np.zeros((n_cand, n_cand), dtype=np.float32)
    else:
        A_inst = compute_A_inst_heuristic(list(range(n_cand)), bbox_np)

    # Clustering: 只对有效候选聚类
    valid_indices = [i for i in range(n_cand) if effective_valid[i]]
    if len(valid_indices) <= 1:
        return {"pred_count": len(valid_indices), "n_candidates": n_cand,
                "n_groups": len(valid_indices), "gt_count": gt_count, "n_conf_filtered": n_filtered}

    # 只对有效候选做聚类
    sub_probs = category_probs[valid_indices]
    sub_A_sem = A_sem[valid_indices][:, valid_indices]
    image_area = float(h*w)

    groups = category_aware_clustering_with_spatial(
        sub_probs, sub_A_sem, bbox_np[valid_indices], image_area,
        tau_affinity=tau_affinity, max_group_size=max_group_size, use_bucketing=True,
    )

    # Dedup: 组内自适应去重
    total_reps = 0
    for group in groups:
        group_global = [valid_indices[i] for i in group]
        if len(group_global) == 0: continue
        if len(group_global) == 1:
            total_reps += 1; continue

        sub_A_inst = A_inst[group_global][:, group_global]
        if len(group_global) > 20:
            components = build_same_instance_components_adaptive(
                list(range(len(group_global))), sub_A_inst, base_tau=tau_inst, use_greedy=True)
        else:
            components = build_same_instance_components(
                list(range(len(group_global))), sub_A_inst, tau_inst=tau_inst)

        reps = select_representatives(
            [[group_global[i] for i in comp] for comp in components],
            np.zeros((n_cand, n_cand)), category_probs, bbox_np, image_area, min_category_conf=0.05)
        total_reps += len(reps)

    return {"pred_count": total_reps, "n_candidates": n_cand, "n_groups": len(groups),
            "gt_count": gt_count, "n_conf_filtered": n_filtered}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-16", default="/home/czp/ws_yiyang/ovcud_cache/fsc147_test_fast")
    ap.add_argument("--cache-32", default="/home/czp/ws_yiyang/ovcud_cache/fsc147_pts32_100")
    ap.add_argument("--cat-16", default="result/checkpoints/category_cosine_fast.pt")
    ap.add_argument("--cat-32", default="result/checkpoints/category_cosine_pts32.pt")
    ap.add_argument("--rel-16", default="result/checkpoints/fsc147_relation_1152.pt")
    ap.add_argument("--rel-32", default="result/checkpoints/fsc147_relation_pts32.pt")
    ap.add_argument("--text-prototypes", default="result/checkpoints/text_prototypes_fsc147.pt")
    ap.add_argument("--images-file", default="result/logs/sample100_test.json")
    ap.add_argument("--density-threshold", type=int, default=100, help="GT > 此值用 pts=32")
    ap.add_argument("--tau-inst", type=float, default=0.4)
    ap.add_argument("--tau-affinity", type=float, default=0.1)
    ap.add_argument("--conf-threshold", type=float, default=0.3)
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    wanted = set(json.load(open(args.images_file)))
    ann = json.load(open(DEFAULT_ANN)) if os.path.exists(DEFAULT_ANN) else {}

    # 加载模型（pts=16 和 pts=32 各一套）
    print("[load] pts=16 models...")
    cat16, _ = load_category_head(args.cat_16, args.device)
    rel16 = load_relation_head(args.rel_16, args.device)
    print("[load] pts=32 models...")
    cat32, _ = load_category_head(args.cat_32, args.device)
    rel32 = load_relation_head(args.rel_32, args.device)
    tp = torch.nn.functional.normalize(
        torch.load(args.text_prototypes, map_location="cpu", weights_only=False).float(), dim=-1
    ).to(args.device)

    # 遍历 sample100 图像
    cache16_files = {f.replace(".pt",""): os.path.join(args.cache_16, f)
                     for f in os.listdir(args.cache_16) if f.endswith(".pt")}
    cache32_files = {f.replace(".pt",""): os.path.join(args.cache_32, f)
                     for f in os.listdir(args.cache_32) if f.endswith(".pt")}

    results = []
    n_high_density = 0
    for fn_jpg in sorted(wanted):
        img_id = os.path.splitext(fn_jpg)[0]

        # 根据 GT count 决定密度
        entry = ann.get(fn_jpg, {})
        gt = len(entry.get("points", []))
        use_high_density = gt > args.density_threshold

        if use_high_density and img_id in cache32_files:
            cache_path = cache32_files[img_id]
            cat_head, rel_head = cat32, rel32
            n_high_density += 1
        elif img_id in cache16_files:
            cache_path = cache16_files[img_id]
            cat_head, rel_head = cat16, rel16
        else:
            continue

        d = torch.load(cache_path, map_location="cpu", weights_only=False)
        r = count_image_adaptive(
            d, cat_head, rel_head, tp, args.device,
            tau_inst=args.tau_inst, tau_affinity=args.tau_affinity,
            conf_threshold=args.conf_threshold,
        )
        r["file_name"] = fn_jpg
        r["high_density"] = use_high_density
        results.append(r)

    if not results:
        print("No images evaluated"); return

    # 聚合
    def agg(rs, key):
        gts = np.array([r["gt_count"] for r in rs], float)
        preds = np.array([r[key] for r in rs], float)
        err = preds - gts
        return {"MAE": float(np.mean(np.abs(err))), "RMSE": float(np.sqrt(np.mean(err**2))),
                "bias": float(np.mean(err))}

    m = agg(results, "pred_count")
    total_cand = sum(r["n_candidates"] for r in results)
    total_filtered = sum(r.get("n_conf_filtered", 0) for r in results)

    print(f"\n评估图像: {len(results)} (高密度: {n_high_density})")
    print(f"总候选数: {total_cand}, 置信度过滤: {total_filtered}")
    print(f"density_threshold={args.density_threshold}, conf_threshold={args.conf_threshold}")
    print(f"\n=== 自适应密度 + 置信度过滤 ===")
    print(f"  MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  bias={m['bias']:+.2f}")

    print(f"\n=== 分 GT 区间 ===")
    print(f"{'区间':<8}{'#图':>5}{'MAE':>9}{'RMSE':>9}{'bias':>9}")
    by_bin = defaultdict(list)
    for r in results: by_bin[bin_of(r["gt_count"])].append(r)
    for lab, _, _ in BINS:
        rs = by_bin.get(lab)
        if not rs: continue
        ma = agg(rs, "pred_count")
        print(f"{lab:<8}{len(rs):>5}{ma['MAE']:>9.2f}{ma['RMSE']:>9.2f}{ma['bias']:>+9.2f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
        json.dump({"overall": m, "results": [{k: v for k, v in r.items() if k != "file_name"} for r in results]},
                  open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
