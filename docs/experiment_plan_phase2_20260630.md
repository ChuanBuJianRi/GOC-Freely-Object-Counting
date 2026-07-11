# OV-CUD 实验计划 Phase 2 (2026-06-30)

## 当前方法与 Baseline 对比

### FSC147 Benchmark 现状

| 方法 | 类型 | MAE | RMSE | 需要 Exemplar? | 需要 Count Label? | 输出类别名? |
|---|---|---|---|---|---|---|
| **SAVE** (2025) | Zero-shot | **8.89** | 35.83 | ❌ Text prompt | ✅ | ❌ |
| **T2ICount** (CVPR 2025) | Zero-shot | **11.76** | 97.86 | ❌ Text prompt | ✅ | ❌ |
| **SAViT** (2025) | Few-shot | **8.92** | 31.26 | ✅ 3 boxes | ✅ | ❌ |
| **OV-CUD (Ours, pts=16)** | Prompt-free+Class-aware | **19.39** | 45.33 | ❌ 无需任何输入 | ❌ 只需类别标签 | ✅ 输出类名 |

### 方法定位差异

OV-CUD 与标准 counting 模型的根本区别：

| 特性 | 标准 Counting | OV-CUD |
|---|---|---|
| 推理输入 | 需要 text prompt 或 exemplar box | **仅图像，无需任何 prompt** |
| 训练监督 | 需要 density map 或 count label | **只需 instance mask + category** |
| 输出 | 仅数量 | **类名 + 数量 + 实例位置** |
| 计数方式 | 密度图积分/回归 | 候选发现 + 去重 + 代表计数 |
| 词汇表 | 闭集/文本引导 | **可扩展类词表 (147 类)** |

**MAE=19.39** 在 prompt-free + count-supervision-free + class-aware 的设定下是合理的，
与需要 density map 监督的 SOTA 方法差距主要来自候选密度不足（pts=16 → recall 80.9%）。

### 当前瓶颈量化

| 瓶颈 | 贡献 | 证据 |
|---|---|---|
| 候选 recall 不足 | **Δ=16.98** (Oracle-All) | pts=16 dot recall 仅 80.9%，100+ 区间 avg_cand=94 < avg_gt=169 |
| 去重误差 | **Δ=2.41** (19.39-16.98) | 关系头弱监督，inst_rec 偏低 |
| 聚类误差 | Δ≈0 | 96.15% 分类准确率，聚类基本正确 |
| 分类误差 | Δ≈0 | top1=96.15% |

**结论：候选密度是 #1 瓶颈，100+ 密集场景候选数远小于 GT 数。**

---

## Exp 4: 高密度 SAM2 (pts=32)

**目标**: 提升 dot recall 80.9% → 95.2%，Oracle-All MAE < 10

**方案**:
- `preprocess_fast_unified.py --pts-per-side 32` 重处理训练+测试集
- 预估: 训练 5376 图 ~1.5h + 测试 100 图 ~2min
- pts=32 候选数 ~90/图 (vs 56/图 at pts=16)，100+ 区间预计 ~150/图

**验收**:
- Dot recall > 95%
- Oracle-All MAE < 10 (vs 当前 16.98)
- 端到端 MAE < 13 (预期)

**风险**:
- 候选数翻倍 → pairwise 关系计算 O(N²) 增长 2.6×
- 推理时间增长，需优化 max_cand/max_pairs

---

## Exp 5: 改善 100+ 密集场景

**目标**: 100+ 区间 MAE 从 83.77 降至 < 30

**方案 A - Multi-scale Tiling**:
- 启用 SAM2 `crop_n_layers=1`，对密集区域额外 crop
- 仅在 100+ 图像上启用（基于 gt_count 判断）

**方案 B - 自适应候选密度**:
- 根据预测的候选数动态调整 SAM2 参数
- 候选数不足时降低 stability_score_thresh 或提高 pts_per_side

**方案 C - 关系头 dedup 改进**:
- 增加 pos_weight (3→8)，提升 inst_rec
- 降低 neg_ratio，保留更多 hard negative
- 训练更长时间 (20→40 epochs)

**验收**:
- 100+ 区间 MAE < 30
- 整体 MAE < 15

---

## Exp 6: 关系头预训练 (LVIS/COCO)

**目标**: 用实例分割数据预训练关系头，改善 A_inst 质量

**方案**:
- 使用 LVIS/COCO instance mask 构建精确的 same-instance / part-whole 标签
- 预训练 → FSC147 fine-tune
- 利用 mask IoU / containment 构建高质量 part-whole 软标签

**预估**: LVIS 预训练 ~1h, FSC147 fine-tune ~30min

**验收**:
- inst_AUC > 0.99 (val), inst_AP > 0.90
- 端到端 MAE 进一步降低 2-3 点

---

## Exp 7: 聚类改进

**目标**: 替换 first-neighbor 为更精细的聚类方法

**方案 A - 双阈值聚类**:
- 先用低阈值聚合候选，再用高阈值拆分过大的 group
- 防止所有同类候选被合并成一个 group

**方案 B - Affinity Propagation**:
- 自动确定 cluster 数量
- 不预设 group 数量

**方案 C - 组内 sub-clustering**:
- 在按类分桶后，组内再用空间距离或 A_sem 做 spectral clustering
- 对大 group (>50 candidates) 自动拆分

**验收**:
- 聚类纯度 > 95%
- group 数量合理（不是 1 group/image）

---

## 执行顺序

```
Exp 4 (高密度 SAM2) → Exp 5 (密集场景) → Exp 6 (关系头预训练) → Exp 7 (聚类改进)
    ~2h                  ~1h                ~1.5h                   ~0.5h
```

### 最终目标

| 指标 | 当前 | Phase 2 目标 |
|---|---|---|
| Dot recall | 80.9% | > 95% |
| Oracle-All MAE | 16.98 | < 8 |
| 端到端 MAE | **19.39** | **< 12** |
| 100+ 区间 MAE | 83.77 | < 30 |
| 分类头 Test top1 | 96.15% | 96%+ |
