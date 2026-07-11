"""Category-Aware First-Neighbor Clustering（OV-CUD §11）。

基于类别兼容性 + same-category 关系得到 coarse semantic groups。

核心逻辑：
    1. 按 top-1 class 分桶（避免不同类被强行连在一起）
    2. 桶内使用 A_group 做 first-neighbor clustering
    3. 若 max A_group < tau_affinity，保留为 singleton / unknown
    4. connected components → coarse semantic groups
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch


def build_group_affinity(
    category_probs: torch.Tensor,   # [N, C]  已 softmax
    A_sem: torch.Tensor,            # [N, N]  same-category relation scores (logits)
    tau_sem: float = 0.5,
) -> torch.Tensor:
    """构造语义分组 affinity 矩阵。

    A_group[i,j] = category_compatibility(i,j) * sigmoid(A_sem[i,j])

    category_compatibility = p_i · p_j（类别分布点积）
    """
    cat_compat = (category_probs @ category_probs.t()).clamp(0.0, 1.0)
    sem_prob = torch.sigmoid(A_sem)
    A_group = cat_compat * sem_prob
    return A_group


def first_neighbor_clustering(
    A_group: np.ndarray,
    tau_affinity: float = 0.3,
    top_class: Optional[np.ndarray] = None,
) -> List[List[int]]:
    """First-neighbor clustering，返回 connected components。

    每个节点最多连一条边（到最相似的邻居），避免强制合并。

    Args:
        A_group: [N, N] affinity 矩阵（对称）
        tau_affinity: 最小 affinity 阈值
        top_class: [N] 每个节点的 top-1 类别（可选，用于分桶）

    Returns:
        groups: list of list of indices
    """
    n = A_group.shape[0]
    if n == 0:
        return []

    # 如果提供了 top_class，先按类分桶再在桶内聚类
    if top_class is not None:
        all_groups = []
        unique_classes = np.unique(top_class)
        for cls in unique_classes:
            mask = top_class == cls
            indices = np.where(mask)[0]
            if len(indices) <= 1:
                if len(indices) == 1:
                    all_groups.append([int(indices[0])])
                continue
            sub_A = A_group[indices][:, indices]
            sub_groups = _cluster_single_bucket(sub_A, tau_affinity)
            for g in sub_groups:
                all_groups.append([int(indices[i]) for i in g])
        return all_groups
    else:
        return _cluster_single_bucket(A_group, tau_affinity)


def _cluster_single_bucket(
    A_group: np.ndarray,
    tau_affinity: float,
) -> List[List[int]]:
    """单桶内 first-neighbor clustering。"""
    n = A_group.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    # 构建有向图：每个节点连到最相似邻居（如果 > tau_affinity）
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        scores = A_group[i].copy()
        scores[i] = -np.inf
        j = int(scores.argmax())
        if A_group[i, j] >= tau_affinity:
            adj[i, j] = True
            adj[j, i] = True  # 对称

    # Connected components
    visited = np.zeros(n, dtype=bool)
    groups = []
    for i in range(n):
        if visited[i]:
            continue
        # BFS
        comp = []
        stack = [i]
        visited[i] = True
        while stack:
            v = stack.pop()
            comp.append(v)
            neighbors = np.where(adj[v] & ~visited)[0]
            for nb in neighbors:
                visited[nb] = True
                stack.append(nb)
        groups.append(comp)

    return groups


def category_aware_clustering(
    category_probs: np.ndarray,   # [N, C]
    A_sem: np.ndarray,            # [N, N]
    tau_affinity: float = 0.3,
    tau_sem: float = 0.5,
    use_bucketing: bool = True,
) -> Tuple[List[List[int]], np.ndarray]:
    """完整的 category-aware clustering pipeline。

    Returns:
        groups: list of group index lists
        A_group: [N, N] 使用的 affinity 矩阵
    """
    N = category_probs.shape[0]
    if N == 0:
        return [], np.zeros((0, 0))

    # Build affinity
    cat_compat = category_probs @ category_probs.T
    sem_prob = 1.0 / (1.0 + np.exp(-A_sem))  # sigmoid
    A_group = cat_compat * sem_prob

    # Top class for bucketing
    top_class = None
    if use_bucketing:
        top_class = category_probs.argmax(axis=1)

    groups = first_neighbor_clustering(A_group, tau_affinity, top_class)

    return groups, A_group


def spatial_sub_clustering(
    group_indices: List[int],
    bbox: np.ndarray,              # [N, 4] XYWH
    image_area: float,
    max_group_size: int = 30,
    spatial_threshold: float = 0.15,  # 相对图像对角线的距离阈值
) -> List[List[int]]:
    """对大 group 做空间 sub-clustering，拆分为空间上更紧凑的子组。

    原理：空间距离远的候选不太可能是同一实例，预先拆分可以：
        1. 减少去重阶段的 Union-Find 链式效应
        2. 降低 pairwise 计算量
        3. 提高 representative selection 质量

    Args:
        group_indices: group 内的候选索引
        bbox: [N, 4] XYWH 格式
        image_area: 图像面积 (h*w)
        max_group_size: 超过此大小的 group 才拆分
        spatial_threshold: bbox 中心距离阈值（相对图像对角线）

    Returns:
        sub_groups: list of sub-group index lists
    """
    n = len(group_indices)
    if n <= max_group_size:
        return [list(group_indices)]

    # 计算 bbox 中心
    centers = np.zeros((n, 2), dtype=np.float32)
    for i, idx in enumerate(group_indices):
        x, y, w, h = bbox[idx]
        centers[i] = [x + w / 2, y + h / 2]

    # 相对距离阈值（图像对角线 × spatial_threshold）
    diag = np.sqrt(image_area)
    threshold = diag * spatial_threshold

    # 构建空间邻接矩阵
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(((centers[i] - centers[j]) ** 2).sum())
            if dist < threshold:
                adj[i, j] = True
                adj[j, i] = True

    # Connected components
    visited = np.zeros(n, dtype=bool)
    sub_groups = []
    for i in range(n):
        if visited[i]:
            continue
        comp = []
        stack = [i]
        visited[i] = True
        while stack:
            v = stack.pop()
            comp.append(group_indices[v])
            neighbors = np.where(adj[v] & ~visited)[0]
            for nb in neighbors:
                visited[nb] = True
                stack.append(nb)
        sub_groups.append(comp)

    return sub_groups


def category_aware_clustering_with_spatial(
    category_probs: np.ndarray,
    A_sem: np.ndarray,
    bbox: np.ndarray,
    image_area: float,
    tau_affinity: float = 0.3,
    max_group_size: int = 30,
    spatial_threshold: float = 0.15,
    use_bucketing: bool = True,
) -> List[List[int]]:
    """带空间 sub-clustering 的完整聚类 pipeline。"""
    N = category_probs.shape[0]
    if N == 0:
        return []

    # Step 1: Category-aware first-neighbor clustering
    cat_compat = category_probs @ category_probs.T
    sem_prob = 1.0 / (1.0 + np.exp(-A_sem))
    A_group = cat_compat * sem_prob

    top_class = category_probs.argmax(axis=1) if use_bucketing else None
    coarse_groups = first_neighbor_clustering(A_group, tau_affinity, top_class)

    # Step 2: Spatial sub-clustering for large groups
    refined_groups = []
    for group in coarse_groups:
        if len(group) > max_group_size:
            sub = spatial_sub_clustering(group, bbox, image_area, max_group_size, spatial_threshold)
            refined_groups.extend(sub)
        else:
            refined_groups.append(group)

    return refined_groups


__all__ = [
    "build_group_affinity",
    "first_neighbor_clustering",
    "category_aware_clustering",
    "spatial_sub_clustering",
    "category_aware_clustering_with_spatial",
]
