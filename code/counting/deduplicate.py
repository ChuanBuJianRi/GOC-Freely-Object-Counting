"""Same-Instance Deduplication（OV-CUD §13.2）。

在每个 semantic group 内，根据 A_inst 构建 same-instance components。
每个 component 最多贡献一个 count。
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


def build_same_instance_components(
    group_indices: List[int],
    A_inst: np.ndarray,          # [N, N] same-instance relation scores (logits)
    tau_inst: float = 0.5,
) -> List[List[int]]:
    """在 group 内构建 same-instance components。

    两个 candidate 属于同一 instance 当且仅当 sigmoid(A_inst[i,j]) >= tau_inst。
    用 Union-Find 做连通分量。

    Args:
        group_indices: 该 group 内的 candidate 索引列表
        A_inst: [N, N] same-instance 关系 logits
        tau_inst: sigmoid 阈值

    Returns:
        components: list of component index lists（每个 component 是 group_indices 的子集）
    """
    n = len(group_indices)
    if n == 0:
        return []
    if n == 1:
        return [list(group_indices)]

    # Union-Find
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
            score = float(1.0 / (1.0 + np.exp(-A_inst[gi, gj])))
            if score >= tau_inst:
                union(a, b)

    # 收集 components
    root_to_members = {}
    for i in range(n):
        r = find(i)
        root_to_members.setdefault(r, []).append(group_indices[i])

    return list(root_to_members.values())


def greedy_dedup(
    group_indices: List[int],
    A_inst: np.ndarray,
    tau_inst: float = 0.5,
    max_comp_size: int = 5,
) -> List[List[int]]:
    """贪心去重：按质量排序，依次分配候选到已有 component。

    避免 Union-Find 链式效应：每个候选只能加入最相似的 component，
    且 component 大小受 max_comp_size 限制。

    Args:
        group_indices: group 内的候选索引
        A_inst: [N, N] same-instance logits
        tau_inst: sigmoid 阈值
        max_comp_size: 每个 component 最大候选数

    Returns:
        components
    """
    n = len(group_indices)
    if n <= 1:
        return [list(group_indices)]

    # 计算所有 pair 的 sigmoid 分数
    scores = {}
    for a in range(n):
        for b in range(a + 1, n):
            gi, gj = group_indices[a], group_indices[b]
            s = float(1.0 / (1.0 + np.exp(-A_inst[gi, gj])))
            if s >= tau_inst:
                scores[(a, b)] = s

    # 按分数降序处理 pair
    components = [[i] for i in range(n)]  # 初始：每个候选独立
    comp_of = list(range(n))  # 候选→component index

    for (a, b), s in sorted(scores.items(), key=lambda x: -x[1]):
        ca, cb = comp_of[a], comp_of[b]
        if ca == cb:
            continue
        # 合并限制：不能超过 max_comp_size
        if len(components[ca]) + len(components[cb]) <= max_comp_size:
            # merge cb into ca
            for idx in components[cb]:
                comp_of[idx] = ca
            components[ca].extend(components[cb])
            components[cb] = []

    # 收集非空 components
    result = []
    for comp in components:
        if comp:
            result.append([group_indices[i] for i in comp])
    return result


def adaptive_tau(group_size: int, base_tau: float = 0.5, max_tau: float = 0.95) -> float:
    """自适应阈值：大 group 用更高阈值，防止链式合并。

    tau = base_tau + (max_tau - base_tau) * min(group_size / 100, 1.0)
    """
    return base_tau + (max_tau - base_tau) * min(group_size / 100.0, 1.0)


def build_same_instance_components_adaptive(
    group_indices: List[int],
    A_inst: np.ndarray,
    base_tau: float = 0.5,
    max_tau: float = 0.95,
    use_greedy: bool = True,
    max_comp_size: int = 5,
) -> List[List[int]]:
    """自适应去重：根据 group 大小调整 tau_inst，可选贪心模式。"""
    n = len(group_indices)
    tau = adaptive_tau(n, base_tau, max_tau)

    if use_greedy and n > 10:
        return greedy_dedup(group_indices, A_inst, tau, max_comp_size)
    else:
        return build_same_instance_components(group_indices, A_inst, tau)


__all__ = [
    "build_same_instance_components",
    "greedy_dedup",
    "adaptive_tau",
    "build_same_instance_components_adaptive",
]
