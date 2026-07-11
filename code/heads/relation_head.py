"""Pairwise Relation Head（OV-CUD Stage 2，对应 2_training_plan.md §3.1）。

输入候选对特征 phi_ij，输出三个分支：

    A_sem[i,j]  : 同语义类别概率（对称）
    A_inst[i,j] : 同一真实实例概率（对称）
    A_part[i,j] : i 是 j 的部件的概率（有向）

对称分支在训练/推理时对 (i,j) 与 (j,i) 双向取平均；有向分支保留方向。
本 head 不含 SAM2/DINOv2/Category Head 任何参数（它们在 Stage 2 全部冻结）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class RelationHeadConfig:
    feat_dim: int            # = pairwise_feature_dim(z_dim)
    hidden_dim: int = 512
    num_layers: int = 3
    dropout: float = 0.1


class PairwiseRelationHead(nn.Module):
    def __init__(self, cfg: RelationHeadConfig) -> None:
        super().__init__()
        dims = [cfg.feat_dim] + [cfg.hidden_dim] * (cfg.num_layers - 1)
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.GELU()]
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
        self.trunk = nn.Sequential(*layers)
        self.sem = nn.Linear(cfg.hidden_dim, 1)
        self.inst = nn.Linear(cfg.hidden_dim, 1)
        self.part = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, phi: torch.Tensor) -> dict[str, torch.Tensor]:
        """phi: [P, feat_dim] -> 各 [P] 的 logit。"""
        h = self.trunk(phi)
        return {
            "sem": self.sem(h).squeeze(-1),
            "inst": self.inst(h).squeeze(-1),
            "part": self.part(h).squeeze(-1),
        }


__all__ = ["RelationHeadConfig", "PairwiseRelationHead"]
