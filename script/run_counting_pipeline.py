"""端到端 Counting Pipeline（OV-CUD Phase 3-4）。

加载候选缓存，跑 category head → relation head → clustering → dedup → counting，
输出 per-image count 并与 GT 对比。

支持多种 oracle 模式用于诊断：
    --oracle-category  : 用 GT matched_class 替代分类头
    --oracle-dedup     : 用 dot-based 二分图分配替代 relation head dedup
    --oracle-all       : 两者都用（理论上界）

首次运行默认使用简化的启发式 dedup（无 relation head 时），
后续可加载真实 fsc147_relation.pt 获得更好的结果。

用法：
    # Oracle all（理论上界）
    python script/run_counting_pipeline.py --cache-dir ... --oracle-all

    # 使用真实分类头 + 启发式 dedup
    python script/run_counting_pipeline.py --cache-dir ... --category-ckpt ... --text-prototypes ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from code.clustering.first_neighbor import (
    category_aware_clustering, spatial_sub_clustering,
    category_aware_clustering_with_spatial,
)
from code.counting.deduplicate import (
    build_same_instance_components, build_same_instance_components_adaptive,
)
from code.counting.representative import select_representatives

DEFAULT_ANN = "/home/czp/official_code/dataset/FSC147/annotation_FSC147_384.json"

BINS = [
    ("0-10", 0, 10),
    ("11-20", 11, 20),
    ("21-50", 21, 50),
    ("51-100", 51, 100),
    ("100+", 101, 10 ** 9),
]


def bin_of(c: int) -> str:
    for lab, lo, hi in BINS:
        if lo <= c <= hi:
            return lab
    return "100+"


def decode_masks(masks_rle, h, w):
    return [mask_utils.decode(r).astype(bool) for r in masks_rle]


# --------------------------------------------------------------------------- #
# Oracle dedup: dot-based bipartite assignment
# --------------------------------------------------------------------------- #
def oracle_dedup_count(
    valid_indices: List[int],
    masks: List[np.ndarray],
    pts_int: List[Tuple[int, int]],
    matched_classes: np.ndarray,
) -> int:
    """用 dot-based 贪心二分图分配做 oracle dedup counting。

    每个 dot 最多分配一个候选，每个候选最多分配一个 dot。
    返回 assigned dot 数量 = 计数。
    """
    # 取 valid 候选
    valid_masks = [masks[i] for i in valid_indices]
    valid_classes = [int(matched_classes[i]) for i in valid_indices]

    # 构建候选-dot 覆盖关系
    h, w = valid_masks[0].shape if valid_masks else (0, 0)
    cand_dots = []
    for m in valid_masks:
        dots = set()
        for di, (xi, yi) in enumerate(pts_int):
            if 0 <= xi < w and 0 <= yi < h and m[yi, xi]:
                dots.add(di)
        cand_dots.append(dots)

    # 按类分组
    class_to_cands = defaultdict(list)
    for ci, cls in enumerate(valid_classes):
        if cls >= 0:
            class_to_cands[cls].append(ci)

    total_count = 0
    for cls, cand_indices in class_to_cands.items():
        # 贪心：小候选优先（更精确），每个候选分配一个未分配的 dot
        assigned_dots = set()
        # 按候选面积升序（小mask更精确）
        sorted_cands = sorted(cand_indices, key=lambda ci: len(valid_masks[ci].nonzero()[0]))

        for ci in sorted_cands:
            available = cand_dots[ci] - assigned_dots
            if available:
                # 选面积最小的那个（precision）
                best_dot = min(available, key=lambda d: 1)  # any dot
                assigned_dots.add(best_dot)

        total_count += len(assigned_dots)

    return total_count


# --------------------------------------------------------------------------- #
# Heuristic dedup (no relation head)
# --------------------------------------------------------------------------- #
def heuristic_dedup_count(
    group_indices: List[int],
    bbox: np.ndarray,
    masks: Optional[List[np.ndarray]] = None,
    tau_iou: float = 0.5,
) -> int:
    """启发式去重（无 relation head 时的 fallback）。

    用 bbox IoU 或 mask IoU 近似 A_inst：
    两个候选 IoU > tau_iou → 同实例 → 合并。
    每 component 选一个代表 → 计数。
    """
    n = len(group_indices)
    if n == 0:
        return 0
    if n == 1:
        return 1

    # 构建 A_inst 近似
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a in range(n):
        for b in range(a + 1, n):
            gi, gj = group_indices[a], group_indices[b]
            if masks is not None:
                # 用 mask IoU
                ma, mb = masks[gi], masks[gj]
                inter = float(np.logical_and(ma, mb).sum())
                union_area = float(np.logical_or(ma, mb).sum())
                iou = inter / union_area if union_area > 0 else 0.0
            else:
                # 用 bbox IoU
                ba, bb = bbox[gi], bbox[gj]
                x1 = max(ba[0], bb[0])
                y1 = max(ba[1], bb[1])
                x2 = min(ba[0] + ba[2], bb[0] + bb[2])
                y2 = min(ba[1] + ba[3], bb[1] + bb[3])
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                area_a = ba[2] * ba[3]
                area_b = bb[2] * bb[3]
                union_area = area_a + area_b - inter
                iou = inter / union_area if union_area > 0 else 0.0

            if iou > tau_iou:
                union(a, b)

    # 统计 component 数量
    roots = set(find(i) for i in range(n))
    return len(roots)


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def count_image(
    d: dict,
    *,
    category_head=None,
    relation_head=None,
    text_prototypes=None,
    oracle_category: bool = False,
    oracle_dedup: bool = False,
    tau_affinity: float = 0.3,
    tau_inst: float = 0.5,
    device: str = "cpu",
    fsc147_ann: Optional[dict] = None,
) -> dict:
    """对单张图像跑完整 pipeline。

    Returns:
        dict with keys: pred_count, n_candidates, n_groups, n_components, gt_count, file_name
    """
    h, w = int(d["height"]), int(d["width"])
    file_name = d.get("file_name", "unknown")
    gt_count = int(d.get("gt_count", 0))

    z = d["z"].float().to(device)
    bbox_np = d["bbox"].float().numpy()  # [N, 4] XYWH
    n_cand = z.shape[0]

    if n_cand == 0:
        return {"pred_count": 0, "n_candidates": 0, "n_groups": 0, "n_components": 0,
                "gt_count": gt_count, "file_name": file_name}

    # Step 1: Category probabilities
    if oracle_category:
        # 使用 GT matched_class → one-hot
        matched_class = np.asarray(d["matched_class"])
        valid = np.asarray(d["valid"]) > 0
        num_classes = int(matched_class.max()) + 1 if len(matched_class) > 0 else 1
        category_probs = np.zeros((n_cand, max(num_classes, 1)), dtype=np.float32)
        for i in range(n_cand):
            if valid[i] and matched_class[i] >= 0:
                category_probs[i, int(matched_class[i])] = 1.0
            else:
                category_probs[i, 0] = 1.0  # background
    elif category_head is not None:
        # 使用真实分类头
        with torch.no_grad():
            from code.training.train_category import forward_logits
            logits, _ = forward_logits(category_head, z, text_prototypes, None)
            probs = torch.softmax(logits, dim=-1)
        category_probs = probs.cpu().numpy()
    else:
        # 无分类头：全归为一个 unknown group
        category_probs = np.zeros((n_cand, 2), dtype=np.float32)
        category_probs[:, 0] = 1.0  # all "unknown"

    # Step 2: Relation matrices (A_sem, A_inst, A_part)
    if oracle_dedup:
        A_sem = np.eye(n_cand, dtype=np.float32)
        A_inst = np.zeros((n_cand, n_cand), dtype=np.float32)
        A_part = np.zeros((n_cand, n_cand), dtype=np.float32)
    elif relation_head is not None and category_probs.shape[1] > 1:
        # 使用真实 relation head
        from code.matrix.pairwise_features import build_pairwise_features, pairwise_feature_dim, box_geometry
        from code.heads.relation_head import PairwiseRelationHead

        # 准备输入
        z_t = z.to(device)
        p_t = torch.from_numpy(category_probs).float().to(device)
        box_t = d["bbox"].float().to(device)  # [N, 4] XYWH

        # 限制候选数（取 top-K by category confidence）
        max_cand = min(n_cand, 200)
        if n_cand > max_cand:
            conf = category_probs.max(axis=1)
            keep = np.argsort(-conf)[:max_cand]
            z_t = z_t[keep]
            p_t = p_t[keep]
            box_t = box_t[keep]
            n_use = max_cand
        else:
            keep = np.arange(n_cand)
            n_use = n_cand

        # 构建所有候选对的 phi_ij
        ii, jj = np.triu_indices(n_use, k=1)
        if len(ii) > 0:
            zi, zj = z_t[ii], z_t[jj]
            pi, pj = p_t[ii], p_t[jj]
            bi, bj = box_t[ii], box_t[jj]
            geom_ij = box_geometry(bi, bj)
            phi_ij = build_pairwise_features(zi, zj, pi, pj, bi, bj, geom=geom_ij)
            phi_ji = build_pairwise_features(zj, zi, pj, pi, bj, bi, geom=box_geometry(bj, bi))

            with torch.no_grad():
                out = relation_head(torch.cat([phi_ij, phi_ji], dim=0))
                P = phi_ij.shape[0]
                sem_logits = 0.5 * (out["sem"][:P] + out["sem"][P:]).cpu().numpy()
                inst_logits = 0.5 * (out["inst"][:P] + out["inst"][P:]).cpu().numpy()
                part_logits = out["part"][:P].cpu().numpy()

            # 填充到完整矩阵
            A_sem = np.zeros((n_cand, n_cand), dtype=np.float32)
            A_inst = np.zeros((n_cand, n_cand), dtype=np.float32)
            A_part = np.zeros((n_cand, n_cand), dtype=np.float32)
            for k in range(P):
                i, j = keep[ii[k]], keep[jj[k]]
                A_sem[i, j] = A_sem[j, i] = sem_logits[k]
                A_inst[i, j] = A_inst[j, i] = inst_logits[k]
                A_part[i, j] = part_logits[k]
        else:
            A_sem = np.eye(n_cand, dtype=np.float32)
            A_inst = np.zeros((n_cand, n_cand), dtype=np.float32)
            A_part = np.zeros((n_cand, n_cand), dtype=np.float32)
    else:
        # 启发式：用 bbox IoU 作为 same-instance proxy
        A_sem = np.eye(n_cand, dtype=np.float32)
        A_inst = np.zeros((n_cand, n_cand), dtype=np.float32)
        # 预计算 bbox IoU
        for i in range(n_cand):
            for j in range(i + 1, n_cand):
                bi, bj = bbox_np[i], bbox_np[j]
                x1 = max(bi[0], bj[0]); y1 = max(bi[1], bj[1])
                x2 = min(bi[0]+bi[2], bj[0]+bj[2]); y2 = min(bi[1]+bi[3], bj[1]+bj[3])
                inter = max(0, x2-x1) * max(0, y2-y1)
                area_i = bi[2]*bi[3]; area_j = bj[2]*bj[3]
                union = area_i + area_j - inter
                iou = inter / union if union > 0 else 0.0
                A_inst[i, j] = iou * 10.0
                A_inst[j, i] = A_inst[i, j]
                A_sem[i, j] = iou * 10.0
                A_sem[j, i] = A_sem[i, j]
        A_part = np.zeros((n_cand, n_cand), dtype=np.float32)

    # Step 3: Clustering (with spatial sub-clustering for large groups)
    image_area = float(h * w)
    max_group_size = 30  # 超过此大小做空间拆分

    if oracle_category:
        # 直接用 GT 类别分组，但对大组做空间拆分
        valid = np.asarray(d["valid"]) > 0
        matched_class = np.asarray(d["matched_class"])
        raw_groups = []
        class_to_indices = defaultdict(list)
        for i in range(n_cand):
            if valid[i] and matched_class[i] >= 0:
                class_to_indices[int(matched_class[i])].append(i)
        for cls, indices in class_to_indices.items():
            if len(indices) > max_group_size:
                sub = spatial_sub_clustering(indices, bbox_np, image_area, max_group_size)
                raw_groups.extend(sub)
            else:
                raw_groups.append(indices)
        groups = raw_groups
    else:
        groups = category_aware_clustering_with_spatial(
            category_probs, A_sem, bbox_np, image_area,
            tau_affinity=tau_affinity,
            max_group_size=max_group_size,
            use_bucketing=True,
        )

    # Step 4: Dedup + Counting
    if oracle_dedup:
        # 使用 dot-based oracle
        masks_rle = d["masks_rle"]
        masks = decode_masks(masks_rle, h, w)
        valid = np.asarray(d["valid"]) > 0
        valid_indices = [i for i in range(n_cand) if valid[i]]

        # 获取 dot annotations
        ann_entry = fsc147_ann.get(file_name) if fsc147_ann else None
        if ann_entry and ann_entry.get("points"):
            pts = np.asarray(ann_entry["points"], dtype=np.float64)
            pts_int = [(int(round(float(x))), int(round(float(y)))) for x, y in pts
                       if 0 <= int(round(float(x))) < w and 0 <= int(round(float(y))) < h]
        else:
            pts_int = []

        if pts_int:
            pred_count = oracle_dedup_count(
                valid_indices, masks, pts_int, np.asarray(d["matched_class"])
            )
        else:
            pred_count = 0
        n_components = 0  # oracle 不按 component 计
    else:
        # 使用 A_inst 或启发式 dedup
        masks = None
        if "masks_rle" in d:
            masks = decode_masks(d["masks_rle"], h, w)

        total_components = 0
        total_reps = 0
        for group in groups:
            if len(group) == 0:
                continue
            if len(group) == 1:
                total_components += 1
                total_reps += 1
                continue

            # 自适应去重：大组用贪心+更高阈值
            if len(group) > 20:
                components = build_same_instance_components_adaptive(
                    group, A_inst, base_tau=tau_inst, use_greedy=True, max_comp_size=5)
            else:
                components = build_same_instance_components(group, A_inst, tau_inst=tau_inst)
            reps = select_representatives(
                components, A_part, category_probs, bbox_np, image_area,
                min_category_conf=0.05,
            )
            total_components += len(components)
            total_reps += len(reps)

        pred_count = total_reps
        n_components = total_components

    return {
        "pred_count": pred_count,
        "n_candidates": n_cand,
        "n_groups": len(groups),
        "n_components": n_components if not oracle_dedup else 0,
        "gt_count": gt_count,
        "file_name": file_name,
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="OV-CUD End-to-End Counting Pipeline")
    ap.add_argument("--cache-dir", required=True, help="候选缓存目录")
    ap.add_argument("--ann", default=DEFAULT_ANN)
    ap.add_argument("--images-file", default=None)
    ap.add_argument("--category-ckpt", default=None, help="分类头 checkpoint")
    ap.add_argument("--relation-ckpt", default=None, help="关系头 checkpoint (fsc147_relation.pt)")
    ap.add_argument("--text-prototypes", default=None, help="文本原型")
    ap.add_argument("--oracle-category", action="store_true")
    ap.add_argument("--oracle-dedup", action="store_true")
    ap.add_argument("--oracle-all", action="store_true")
    ap.add_argument("--tau-affinity", type=float, default=0.3)
    ap.add_argument("--tau-inst", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if args.oracle_all:
        args.oracle_category = True
        args.oracle_dedup = True

    # Load models if needed
    category_head = None
    text_prototypes = None
    if not args.oracle_category and args.category_ckpt:
        ck = torch.load(args.category_ckpt, map_location="cpu", weights_only=False)
        head_type = ck.get("head_type", "cosine")

        # Check if it's a cosine-only head (v2 format)
        if head_type == "cosine" or "config" in ck:
            from script.train_category_v2 import CosineCategoryHead
            in_dim = ck.get("in_dim", 1152)
            proj_dim = ck.get("proj_dim", 512)
            category_head = CosineCategoryHead(in_dim=in_dim, proj_dim=proj_dim, dropout=0.3, num_layers=2)
            category_head.load_state_dict(ck["head"])
            category_head.to(args.device).eval()
            print(f"[load] category head: cosine in_dim={in_dim}")
        else:
            from code.heads.category_head import build_category_head, CategoryHeadConfig
            proj_kwargs = {"dropout": 0.0, "num_layers": 2}
            extra = {}
            if head_type == "hybrid":
                extra = {"open_kwargs": {"proj_kwargs": proj_kwargs}, "closed_kwargs": {"proj_kwargs": proj_kwargs}}
            else:
                extra = {"proj_kwargs": proj_kwargs}
            cfg = CategoryHeadConfig(
                head_type=head_type, in_dim=ck["in_dim"],
                proj_dim=ck["proj_dim"], num_classes=ck["num_classes"],
                extra=extra,
            )
            category_head = build_category_head(cfg)
            category_head.load_state_dict(ck["head"])
            category_head.to(args.device).eval()
            print(f"[load] category head: {head_type} num_classes={ck['num_classes']} in_dim={ck['in_dim']}")

    # Load relation head if provided
    relation_head = None
    if args.relation_ckpt:
        from code.heads.relation_head import PairwiseRelationHead, RelationHeadConfig
        from code.matrix.pairwise_features import pairwise_feature_dim

        rel_ck = torch.load(args.relation_ckpt, map_location="cpu", weights_only=False)
        feat_dim = pairwise_feature_dim(rel_ck["z_dim"])
        rel_cfg = RelationHeadConfig(
            feat_dim=feat_dim, hidden_dim=rel_ck["hidden_dim"],
            num_layers=rel_ck["num_layers"], dropout=0.1,
        )
        relation_head = PairwiseRelationHead(rel_cfg)
        relation_head.load_state_dict(rel_ck["relation_head"])
        relation_head.to(args.device).eval()
        print(f"[load] relation head: feat_dim={feat_dim} hidden={rel_ck['hidden_dim']} layers={rel_ck['num_layers']}")

    if not args.oracle_category and args.text_prototypes:
        text_prototypes = torch.load(args.text_prototypes, map_location="cpu", weights_only=False)
        text_prototypes = torch.nn.functional.normalize(text_prototypes.float(), dim=-1).to(args.device)

    # Load annotations (for dot-based oracle)
    fsc147_ann = json.load(open(args.ann)) if args.oracle_dedup else None

    # Run
    wanted = set(json.load(open(args.images_file))) if args.images_file else None
    files = sorted(f for f in os.listdir(args.cache_dir) if f.endswith(".pt"))
    if args.limit > 0:
        files = files[:args.limit]

    results = []
    for fn in files:
        d = torch.load(os.path.join(args.cache_dir, fn), map_location="cpu", weights_only=False)
        file_name = d.get("file_name") or fn
        if wanted is not None and file_name not in wanted:
            continue

        r = count_image(
            d,
            category_head=category_head,
            relation_head=relation_head,
            text_prototypes=text_prototypes,
            oracle_category=args.oracle_category,
            oracle_dedup=args.oracle_dedup,
            tau_affinity=args.tau_affinity,
            tau_inst=args.tau_inst,
            device=args.device,
            fsc147_ann=fsc147_ann,
        )
        results.append(r)

    if not results:
        print("没有可评估的图像")
        return

    # Evaluate
    def agg(rs, key):
        gts = np.array([r["gt_count"] for r in rs], float)
        preds = np.array([r[key] for r in rs], float)
        err = preds - gts
        return {
            "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "bias": float(np.mean(err)),
        }

    m = agg(results, "pred_count")
    n_total = sum(r["n_candidates"] for r in results)
    n_groups = sum(r["n_groups"] for r in results)
    n_comp = sum(r["n_components"] for r in results)

    print(f"评估图像: {len(results)}")
    print(f"总候选数: {n_total}, 总 group 数: {n_groups}, 总 component 数: {n_comp}")
    print()
    oracle_tag = ""
    if args.oracle_category and args.oracle_dedup:
        oracle_tag = " [ORACLE CATEGORY + DEDUP]"
    elif args.oracle_category:
        oracle_tag = " [ORACLE CATEGORY]"
    elif args.oracle_dedup:
        oracle_tag = " [ORACLE DEDUP]"
    print(f"=== 计数结果{oracle_tag} ===")
    print(f"  MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  bias={m['bias']:+.2f}")

    print()
    print(f"=== 分 GT 区间 MAE ===")
    print(f"{'区间':<8}{'#图':>5}{'MAE':>9}{'RMSE':>9}{'bias':>9}")
    by_bin = defaultdict(list)
    for r in results:
        by_bin[bin_of(r["gt_count"])].append(r)
    for lab, _, _ in BINS:
        rs = by_bin.get(lab)
        if not rs:
            continue
        ma = agg(rs, "pred_count")
        print(f"{lab:<8}{len(rs):>5}{ma['MAE']:>9.2f}{ma['RMSE']:>9.2f}{ma['bias']:>+9.2f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
        json.dump({"overall": m, "by_bin": {lab: agg(by_bin.get(lab, []), "pred_count") for lab, _, _ in BINS if by_bin.get(lab)}, "results": results}, open(args.out, "w"), indent=2, ensure_ascii=False)
        print(f"\n结果写入 {args.out}")


if __name__ == "__main__":
    main()
