"""Representative Selection（OV-CUD §13.3）。

在每个 same-instance component 内选出一个代表候选。
representative score 综合类别置信度、几何完整性和 part-whole 方向。
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


def representative_score(
    candidate_idx: int,
    component_indices: List[int],
    category_probs: np.ndarray,    # [N, C]
    A_part: np.ndarray,            # [N, N] part-whole relation (directional: A_part[i,j]=P(i is part of j))
    bbox: np.ndarray,              # [N, 4] XYWH
    image_area: float,
) -> float:
    """计算单个候选的代表得分。

    RepScore = category_confidence + completeness_score
             + mean(A_part[*, i])   (高入度 → 更完整)
             - mean(A_part[i, *])   (高出度 → 可能是局部)

    Args:
        candidate_idx: 候选索引
        component_indices: 同 component 内所有候选索引（用于归一化）
        category_probs: [N, C] 类别分布
        A_part: [N, N] part-whole 关系矩阵
        bbox: [N, 4] bbox XYWH
        image_area: 图像面积（用于归一化）

    Returns:
        得分（越高越好）
    """
    # 类别置信度（top class）
    cat_conf = float(category_probs[candidate_idx].max())

    # 几何完整度（bbox 面积，越大越可能完整）
    x, y, w, h = bbox[candidate_idx]
    area = w * h
    # Normalize to ~[0, 1] relative to image
    completeness = min(area / (image_area * 0.5), 1.0)

    # Part-whole 得分
    all_indices = list(range(A_part.shape[0]))
    outgoing = float(np.mean([A_part[candidate_idx, j] for j in all_indices if j != candidate_idx])) if len(all_indices) > 1 else 0.0
    incoming = float(np.mean([A_part[j, candidate_idx] for j in all_indices if j != candidate_idx])) if len(all_indices) > 1 else 0.0

    score = cat_conf + 0.5 * completeness + incoming - outgoing
    return score


def choose_representative(
    component_indices: List[int],
    A_part: np.ndarray,
    category_probs: np.ndarray,
    bbox: np.ndarray,
    image_area: float,
) -> int:
    """从 component 中选出最佳代表候选。

    Returns:
        代表候选的全局索引
    """
    if len(component_indices) == 1:
        return component_indices[0]

    best_idx = component_indices[0]
    best_score = -float("inf")
    for ci in component_indices:
        score = representative_score(
            ci, component_indices, category_probs, A_part, bbox, image_area
        )
        if score > best_score:
            best_score = score
            best_idx = ci
    return best_idx


def select_representatives(
    components: List[List[int]],
    A_part: np.ndarray,
    category_probs: np.ndarray,
    bbox: np.ndarray,
    image_area: float,
    min_category_conf: float = 0.1,
) -> List[int]:
    """从所有 components 中选出代表候选列表。

    Args:
        components: 各 component 的候选索引列表
        A_part: [N, N] part-whole 矩阵
        category_probs: [N, C]
        bbox: [N, 4]
        image_area: 图像面积
        min_category_conf: 最低类别置信度（低于此值的不作为代表）

    Returns:
        representative_indices: 代表候选索引列表
    """
    representatives = []
    for comp in components:
        rep = choose_representative(comp, A_part, category_probs, bbox, image_area)
        conf = float(category_probs[rep].max())
        if conf >= min_category_conf:
            representatives.append(rep)
    return representatives


__all__ = ["representative_score", "choose_representative", "select_representatives"]
