"""Oracle 诊断（D2 + D3）：评估分类头和 dedup 各自的上界。

D2 - Oracle Category:
    用 GT 类别标签（来自缓存 matched_class）做 candidate grouping。
    即：同一 GT 类的 candidate -> 同一 semantic group。

D3 - Oracle Dedup (dot-based):
    用 dot id 做 same-instance component：
    包含同一个 dot 的 candidates -> 同一 instance（一个 count）。

结合使用：
    D2_cat + D3_dedup: GT 类别分组 + dot-based dedup = 计数理论上界
    D2_cat only: 用 GT 类别分组，但无 dedup（每个 group 的 candidate 数 = count）
    No oracle: 所有 candidate 算一个 group（无类别信息）
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from pycocotools import mask as mask_utils

DEFAULT_ANN = "/home/czp/official_code/dataset/FSC147/annotation_FSC147_384.json"

BINS: List[Tuple[str, int, int]] = [
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


def decode_masks(masks_rle: List[dict], h: int, w: int) -> List[np.ndarray]:
    out = []
    for r in masks_rle:
        m = mask_utils.decode(r).astype(bool)
        if m.shape == (h, w):
            out.append(m)
    return out


def cand_dot_map(masks: List[np.ndarray], pts_int: List[Tuple[int, int]]) -> List[List[int]]:
    """每个候选覆盖哪些 dot（返回 dot 索引列表）。"""
    hits: List[List[int]] = []
    for m in masks:
        hit_dots = []
        for di, (xi, yi) in enumerate(pts_int):
            if m[yi, xi]:
                hit_dots.append(di)
        hits.append(hit_dots)
    return hits


def count_oracle_dedup(
    masks: List[np.ndarray],
    pts_int: List[Tuple[int, int]],
    matched_classes: np.ndarray,
    valid: np.ndarray,
) -> Tuple[int, int, int, int]:
    """Oracle category + oracle dedup counting.

    返回 (pred_count, dot_covered, dot_total, num_groups)

    算法：
    1. 只用 valid > 0 的 candidate
    2. 按 GT matched_class 分组
    3. 组内按 dot-based same-instance dedup:
       - 两个 candidate 共享至少一个 dot -> 同一 instance component
       - 每个 connected component = 一个 count
    4. pred_count = sum over groups of num_components
    """
    cand_dots = cand_dot_map(masks, pts_int)
    n_cand = len(masks)

    # Valid candidates
    valid_idx = [i for i in range(n_cand) if valid[i] > 0]

    # Group by GT class
    class_to_cands: Dict[int, List[int]] = defaultdict(list)
    for i in valid_idx:
        cls = int(matched_classes[i])
        if cls >= 0:
            class_to_cands[cls].append(i)

    # Compute dot coverage (for reference)
    all_covered_dots = set()
    for i in valid_idx:
        all_covered_dots.update(cand_dots[i])

    # For each class group, build same-instance components
    total_count = 0
    total_components = 0
    used_dots_for_counting = set()

    for cls, cands_in_class in class_to_cands.items():
        # Build adjacency: two candidates share a dot -> same instance
        n = len(cands_in_class)
        if n == 0:
            continue

        # Union-Find on candidates
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

        # For each dot, find all candidates that cover it and union them
        dot_to_cands: Dict[int, List[int]] = defaultdict(list)
        for local_idx, ci in enumerate(cands_in_class):
            for dot_id in cand_dots[ci]:
                dot_to_cands[dot_id].append(local_idx)

        for dot_id, local_indices in dot_to_cands.items():
            for j in range(1, len(local_indices)):
                union(local_indices[0], local_indices[j])

        # Count unique components
        component_roots = set(find(i) for i in range(n))
        count_for_class = len(component_roots)
        total_count += count_for_class
        total_components += count_for_class

    return total_count, len(all_covered_dots), len(pts_int), len(class_to_cands)


def count_oracle_category_only(
    masks: List[np.ndarray],
    pts_int: List[Tuple[int, int]],
    matched_classes: np.ndarray,
    valid: np.ndarray,
) -> Tuple[int, int, int]:
    """Oracle category only (no dedup): 每个 valid candidate 算 1。

    按 GT 类别分组后，直接数 valid candidate 数量。
    这模拟了"类别完美但无去重"的情况。
    """
    n_cand = len(masks)
    valid_idx = [i for i in range(n_cand) if valid[i] > 0]

    class_to_cands: Dict[int, List[int]] = defaultdict(list)
    for i in valid_idx:
        cls = int(matched_classes[i])
        if cls >= 0:
            class_to_cands[cls].append(i)

    all_covered_dots = set()
    for i in valid_idx:
        cand_dots_i = []
        for di, (xi, yi) in enumerate(pts_int):
            if masks[i][yi, xi]:
                cand_dots_i.append(di)
        all_covered_dots.update(cand_dots_i)

    # Simple: count = number of valid candidates in each class, summed
    total_count = sum(len(cands) for cands in class_to_cands.values())

    return total_count, len(all_covered_dots), len(pts_int)


def main() -> None:
    ap = argparse.ArgumentParser(description="Oracle category + dedup 诊断")
    ap.add_argument("--cache-dir", required=True, help="候选缓存目录（每图一个 .pt，含 matched_class/valid/masks_rle）")
    ap.add_argument("--ann", default=DEFAULT_ANN)
    ap.add_argument("--images-file", default=None, help="可选：只评这些文件名")
    ap.add_argument("--out", default="", help="输出 json")
    ap.add_argument("--limit", type=int, default=-1, help="最多处理 N 张图")
    args = ap.parse_args()

    ann = json.load(open(args.ann))
    wanted = set(json.load(open(args.images_file))) if args.images_file else None

    files = sorted(f for f in os.listdir(args.cache_dir) if f.endswith(".pt"))
    if args.limit > 0:
        files = files[: args.limit]

    rows = []
    for fn in files:
        d = torch.load(os.path.join(args.cache_dir, fn), map_location="cpu", weights_only=False)
        file_name = d.get("file_name") or f"{d.get('img_id')}.jpg"
        if wanted is not None and file_name not in wanted:
            continue
        entry = ann.get(file_name)
        if entry is None or not entry.get("points"):
            continue

        h, w = int(d["height"]), int(d["width"])
        masks = decode_masks(d["masks_rle"], h, w)
        pts = np.asarray(entry["points"], dtype=np.float64)

        pts_int = []
        for x, y in pts:
            xi, yi = int(round(float(x))), int(round(float(y)))
            if 0 <= xi < w and 0 <= yi < h:
                pts_int.append((xi, yi))
        gt = len(pts_int)
        if gt == 0:
            continue

        if len(masks) == 0:
            continue

        matched_classes = np.asarray(d["matched_class"])
        valid = np.asarray(d["valid"])

        # D2+D3: oracle category + oracle dedup
        pred_cat_dedup, dots_cov, dots_tot, n_groups = count_oracle_dedup(
            masks, pts_int, matched_classes, valid
        )

        # D2 only: oracle category, no dedup (each valid candidate = 1 count)
        pred_cat_only, _, _ = count_oracle_category_only(
            masks, pts_int, matched_classes, valid
        )

        # Count oracle-A (covered dots)
        cand_dots = cand_dot_map(masks, pts_int)
        all_covered = set()
        for i in range(len(masks)):
            if valid[i] > 0:
                all_covered.update(cand_dots[i])

        rows.append({
            "file": file_name,
            "gt": gt,
            "n_cand": len(masks),
            "n_valid": int(valid.sum()),
            "pred_cover": len(all_covered),           # Oracle-A
            "pred_cat_only": pred_cat_only,            # D2: perfect category, no dedup
            "pred_cat_dedup": pred_cat_dedup,          # D2+D3: perfect category + dot dedup
            "dot_recall": len(all_covered) / gt,
            "n_groups": n_groups,
        })

    if not rows:
        print("没有可评估的图像")
        return

    def agg(rs, key):
        gts = np.array([r["gt"] for r in rs], float)
        preds = np.array([r[key] for r in rs], float)
        err = preds - gts
        return {
            "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "bias": float(np.mean(err)),
        }

    print(f"评估图像数: {len(rows)}")
    print()
    print("=== 整体计数 Oracle 对比 ===")
    for name, key in [
        ("Oracle-A 覆盖上界", "pred_cover"),
        ("D2: oracle category (no dedup)", "pred_cat_only"),
        ("D2+D3: oracle cat + dot dedup", "pred_cat_dedup"),
    ]:
        m = agg(rows, key)
        print(f"  {name:<35} MAE={m['MAE']:7.2f}  RMSE={m['RMSE']:7.2f}  bias={m['bias']:+7.2f}")

    print()
    print("=== 分 GT 区间 MAE ===")
    header = f"{'区间':<8}{'#图':>5}{'覆盖MAE':>9}{'CatOnly':>9}{'CatDedup':>9}"
    print(header)
    print("-" * len(header))
    by_bin: Dict[str, list] = defaultdict(list)
    for r in rows:
        by_bin[bin_of(r["gt"])].append(r)
    for lab, _, _ in BINS:
        rs = by_bin.get(lab)
        if not rs:
            continue
        a = agg(rs, "pred_cover")
        b = agg(rs, "pred_cat_only")
        c = agg(rs, "pred_cat_dedup")
        print(f"{lab:<8}{len(rs):>5}{a['MAE']:>9.2f}{b['MAE']:>9.2f}{c['MAE']:>9.2f}")

    # Key diagnostic comparison
    print()
    print("=== 诊断结论 ===")
    m_cover = agg(rows, "pred_cover")
    m_cat = agg(rows, "pred_cat_only")
    m_dedup = agg(rows, "pred_cat_dedup")

    print(f"候选召回上界 (Oracle-A):       MAE={m_cover['MAE']:.2f}")
    print(f"+ oracle category (no dedup):   MAE={m_cat['MAE']:.2f}  (delta={m_cat['MAE']-m_cover['MAE']:+.2f})")
    print(f"+ oracle category + dot dedup:  MAE={m_dedup['MAE']:.2f}  (delta={m_dedup['MAE']-m_cat['MAE']:+.2f})")
    print()
    print("解读:")
    print("  Oracle-A -> D2(delta)  : 分类错误引入的额外误差（当前 image-level class 完美，主要看候选是否按类分桶正确）")
    print("  D2 -> D2+D3(delta)     : dedup 错误引入的额外误差（dot-based oracle dedup 消除去重误差）")
    print("  D2+D3 残差 vs Oracle-A : 主要来自 group 内部的 instance component Union-Find 逻辑")

    if args.out:
        json.dump({
            "overall": {
                "cover": m_cover,
                "cat_only": m_cat,
                "cat_dedup": m_dedup,
            },
            "rows": rows,
        }, open(args.out, "w"), indent=2, ensure_ascii=False)
        print(f"\n明细写入 {args.out}")


if __name__ == "__main__":
    main()
