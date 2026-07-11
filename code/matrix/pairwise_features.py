"""候选对特征 phi_ij 构造（OV-CUD Stage 2，对应 2_training_plan.md §3.1）。

训练与推理共用。给定两个候选的区域特征 z、类别分布 p、几何框 box，输出拼接特征：

    phi_ij = [ z_i, z_j, |z_i - z_j|, z_i * z_j,
               cosine(z_i, z_j),
               p_i · p_j,                       # 来自冻结的 Category Head
               box_iou,
               containment_i_in_j, containment_j_in_i,
               center_distance, scale_ratio, area_ratio ]

注意：第一版离线缓存只存了候选 bbox（未存 mask），故几何项用 **bbox** 计算
（mask IoU/containment 用 box 近似）。bbox 约定为 COCO 风格 XYWH。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# 几何标量个数：iou, cont_i_in_j, cont_j_in_i, center_dist, scale_ratio, area_ratio
_NUM_GEOM = 6
# 额外标量：cosine(z_i,z_j), p_i·p_j
_NUM_SCALAR = 2


def pairwise_feature_dim(z_dim: int) -> int:
    """phi_ij 维度 = 4*z_dim（z_i,z_j,|diff|,prod）+ 2 标量 + 6 几何。"""
    return 4 * z_dim + _NUM_SCALAR + _NUM_GEOM


def _xywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    x, y, w, h = box.unbind(-1)
    return torch.stack([x, y, x + w, y + h], dim=-1)


def box_geometry(box_i: torch.Tensor, box_j: torch.Tensor) -> torch.Tensor:
    """成对 bbox 几何特征。box_*: [P,4] XYWH -> [P,6]。"""
    a = _xywh_to_xyxy(box_i)
    b = _xywh_to_xyxy(box_j)
    area_i = (box_i[..., 2] * box_i[..., 3]).clamp_min(1e-6)
    area_j = (box_j[..., 2] * box_j[..., 3]).clamp_min(1e-6)

    lt = torch.maximum(a[..., :2], b[..., :2])
    rb = torch.minimum(a[..., 2:], b[..., 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[..., 0] * wh[..., 1]
    union = (area_i + area_j - inter).clamp_min(1e-6)

    iou = inter / union
    cont_i_in_j = inter / area_i           # i 被 j 包含的比例
    cont_j_in_i = inter / area_j

    ci = (a[..., :2] + a[..., 2:]) * 0.5
    cj = (b[..., :2] + b[..., 2:]) * 0.5
    center_dist = (ci - cj).norm(dim=-1) / (area_i.sqrt() + area_j.sqrt()).clamp_min(1e-6)

    si, sj = area_i.sqrt(), area_j.sqrt()
    scale_ratio = torch.minimum(si, sj) / torch.maximum(si, sj).clamp_min(1e-6)
    area_ratio = torch.minimum(area_i, area_j) / torch.maximum(area_i, area_j).clamp_min(1e-6)

    return torch.stack(
        [iou, cont_i_in_j, cont_j_in_i, center_dist, scale_ratio, area_ratio], dim=-1
    )


def build_pairwise_features(
    z_i: torch.Tensor, z_j: torch.Tensor,    # [P, D]
    p_i: torch.Tensor, p_j: torch.Tensor,    # [P, C]  类别分布（已 softmax）
    box_i: torch.Tensor, box_j: torch.Tensor,  # [P, 4] XYWH
    geom: "torch.Tensor | None" = None,        # [P, 6] 预算几何，缺省用 bbox 计算
) -> torch.Tensor:
    """构造 phi_ij，返回 [P, pairwise_feature_dim(D)]。

    geom 若提供（如用真实 mask 算的 iou/containment），则直接使用，
    否则回退到 bbox 几何（box_geometry）。
    """
    cos = F.cosine_similarity(z_i, z_j, dim=-1, eps=1e-6).unsqueeze(-1)
    p_dot = (p_i * p_j).sum(dim=-1, keepdim=True)
    if geom is None:
        geom = box_geometry(box_i, box_j)
    return torch.cat(
        [z_i, z_j, (z_i - z_j).abs(), z_i * z_j, cos, p_dot, geom], dim=-1
    )
