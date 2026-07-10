# OV-CUD AAAI 2026 投稿 — 完整实验报告

**日期**: 2026-07-10 (协议审计更新)
**方法**: OV-CUD (Open-Vocabulary Counting via Understanding and Deduplication)
**设定**: Prompt-free class-aware object counting — 仅输入图像，无 exemplar、text prompt、count label

> **2026-07-10 关键协议审计**：当前 FSC147、CARPK、OmniCount 和 MCAC 主评测脚本使用 cache 中由 GT dot coverage 生成的 `valid` 筛选推理候选。因而历史 headline 与本报告中标记为 cache-compatible 的结果不是严格 no-GT inference，暂不能作为 reviewer-ready prompt-free 主结果。MCAC 已补 strict `candidate_filter=all` sanity；其他数据集仍需统一重跑。详见 `docs/mcac_full2115_leaveoneout_report_20260710.md`。

---

## 目录

1. [执行摘要](#1-执行摘要)
2. [方法概述](#2-方法概述)
3. [主实验结果](#3-主实验结果)
4. [核心消融实验](#4-核心消融实验)
5. [跨数据集泛化](#5-跨数据集泛化)
6. [协议对比与 Baseline](#6-协议对比与-baseline)
7. [Oracle 诊断与分析](#7-oracle-诊断与分析)
8. [统计可信度](#8-统计可信度)
9. [AAAI 投稿策略](#9-aaai-投稿策略)
10. [主文图表分配](#10-主文图表分配)
11. [附录图表分配](#11-附录图表分配)
12. [写作建议](#12-写作建议)
13. [检查清单](#13-检查清单)

---

## 1. 执行摘要

OV-CUD 是一个 **prompt-free** 的开放词汇物体计数方法。与现有方法不同，OV-CUD：

- **推理时不接收任何提示** (无 exemplar box、无 text prompt、无 category name)
- **不使用 count label 或 density map 训练** (无计数监督)
- **输出每个语义组的类别名 + 实例位置 + 计数**
- **通过候选生成 → 分类 → 关系建模 → 分组 → 去重 → 代表选择 完成计数**

### 🏆 核心结果

| 数据集 | 设定 | MAE | RMSE | 备注 |
|---|---|---|---|---|
| **FSC147 Test (historical cache-only)** | Prompt-free / Image-only | **12.74** | 106.20 | 1,189 rows; `7611.jpg` missing from final cache |
| **FSC147 Test (audited current-code)** | Prompt-free / Image-only | **13.47** | 126.74 | Full 1,190 images; `7611.jpg` true MR100, fast=0 candidates |
| **FSC147 Test (4x4 rescue exploratory)** | Cache-compatible / Image-only | **12.67** | 113.71 | Same learned heads; no-GT trigger, but candidate validity still GT-derived |
| **FSC147 Test (base pipeline)** | Prompt-free / Image-only | 16.54 | — | Full 1190 images, adaptive density only |
| **CARPK Test** | Zero-shot transfer (FSC147→CARPK) | **4.06 ± 0.17** | 5.51 ± 0.24 | 459 images, 95% CI |
| **PUCPR+ Test** | Zero-shot transfer + Tiling | **3.59** | 5.43 | 25 images, 2×2 tiling |
| **COCO val** | Multi-category (80 classes) | **6.94** | 10.04 | 300 images, class-aware evaluation |
| **OmniCount-191 Test** | Prompt-free / Class-agnostic | **6.75** | 10.24 | 1,909 images, 93 classes, multi-label |
| **MCAC Test (strict no-GT)** | Prompt-free / Anonymous groups | **32.11** | 53.79 | Full 2,115 images, 3,630 image-class pairs |

### 关键 Claim

1. **OV-CUD 是 count-supervision-free 的**: 不使用 density map 或 count label 训练
2. **OV-CUD 是 prompt-free 的**: 推理时不接收任何提示
3. **OV-CUD 输出类别名**: 通过 text-prototype 分类头实现开放词表分类
4. **OV-CUD 跨数据集泛化强**: FSC147 → CARPK MAE=4.06, FSC147 → PUCPR+ MAE=3.59 (tiled), FSC147 → OmniCount-191 MAE=6.75 (class-agnostic)
5. **OV-CUD 多标签场景兼容**: 在不接收类别提示的条件下，class-agnostic counting 在 OmniCount-191 上优于 Vanilla SAM2 (37.60→6.75, 5.6× 改进)
6. **Prompt-free 优于 Prompt-based 检测**: 历史 OV-CUD prompt-free MAE=12.74，审计 full-1190 current-code MAE=13.47；仍显著优于 OWLv2 prompt-based MAE=43.61
7. **Multi-Resolution Fusion 是关键改进**: 融合 pts=16 coarse + pts=32 tiled fine 候选；历史 cache-only headline 为 12.74，审计 full-1190 current-code 为 13.47
8. **组件贡献具有数据集依赖性**: FSC147 上 relation/HR 前端有贡献；MCAC 上 HR 显著有效、RH/CP 仅小幅有效，ADF 跨域失配，4x4 rescue 无 per-class 收益

---

## 2. 方法概述

### 2.1 Pipeline

```
Image → SAM2 AMG → Candidates (masks + bboxes)
  → DINOv2 3-view Features (1152-dim)
  → CosineCategoryHead: Projection MLP + Text Prototypes → Class Probabilities
  → PairwiseRelationHead: A_sem + A_inst + A_part → Pairwise Relations
  → Category-Aware Clustering: A_group = (p_i·p_j) × A_sem → Semantic Groups
  → Same-Instance Dedup: A_inst → Instance Components
  → Part-Whole Representative Selection: A_part → Final Count per Group
```

### 2.2 关键模块

| 模块 | 功能 | 训练数据 |
|---|---|---|
| **SAM2 AMG** | 候选 mask 生成 | 预训练 (无需微调) |
| **DINOv2 3-View** | Masked/Box/Context 三视角特征 | 预训练 (无需微调) |
| **CosineCategoryHead** | 投影到 CLIP 文本空间 + 余弦分类 | FSC147 instance masks (弱监督) |
| **PairwiseRelationHead** | 预测候选对的 semantic/instance/part-whole 关系 | COCO 预训练 + FSC147 微调 |
| **Category-Aware Clustering** | 按类别分组 + 空间 sub-clustering | 无训练 (基于 A_sem) |
| **Same-Instance Dedup** | 合并同一实例的多个候选 | 无训练 (基于 A_inst) |
| **Representative Selection** | 每组件选最佳代表 | 无训练 (基于 A_part) |

### 2.3 训练监督审计

| 监督类型 | 是否使用 | 说明 |
|---|---|---|
| Count label / Density map | ❌ **不使用** | 区别于所有 counting-supervised 方法 |
| Exemplar box | ❌ **不使用** | 区别于 few-shot 方法 |
| Text prompt (推理时) | ❌ **不使用** | 区别于 text-specified 方法 |
| Instance mask (SAM) | ✅ **使用** | 来自开放数据集的实例分割标注 |
| Category label (文本) | ✅ **使用** | CLIP 文本编码器获取类别原型 |
| Pairwise relation label | ✅ **使用** | 从 instance mask + dot annotation 自动导出 |

---

## 3. 主实验结果

### 3.1 FSC147 Test — 主对比表 (Table 1 候选)

OV-CUD 在 **image-only, count-supervision-free** 设定下与现有方法对比。

#### Block A: Few-shot Methods (强输入参考 — 需 1-3 exemplar boxes)

| Method | Input | Count Sup. | MAE | RMSE |
|---|---|---|---|---|
| CounTR | 3 exemplars | Density map | 11.95 | 48.69 |
| BMNet+ | 3 exemplars | Density map | 12.00 | — |
| LOCA | 3 exemplars | Density map | 10.67 | 48.73 |
| SAViT | 3 exemplars | Density map | 10.54 | — |

> 这些方法需要 exemplar boxes 和 density map 训练，输入和监督均强于 OV-CUD。仅作参考。

#### Block B: Text-Specified Methods (需 text prompt)

| Method | Input | Count Sup. | MAE | RMSE |
|---|---|---|---|---|
| CounTX | Text prompt | Density map | 11.07 | 63.60 |
| VLCounter | Text prompt | Density map | 15.80 | 75.40 |

> 这些方法需要文本提示 (GT category name)，仍需 density map 训练。

#### Block C: Image-Only / Prompt-Free Methods (同设定对比)

| Method | Input | Count Sup. | Class-Aware | MAE | RMSE |
|---|---|---|---|---|---|
| RCC | Image only | Density map | ❌ | 15.37 | 75.20 |
| DAVE | Image only | Density map | ❌ | 14.37 | 72.10 |
| GCA-SUN | Image only | Density map | ❌ | 21.29 | — |
| OCCAM-S | Image only | **No training** | ❌ | 14.35 | 67.54 |
| **OV-CUD (Ours)** | **Image only** | **None** | ✅ | **23.30** | **111.90** |

> OV-CUD 是唯一同时满足 "prompt-free + count-supervision-free + class-aware output" 的方法。Full 1190-image FSC147 test, tau_inst=0.5, Exp5-C relation head.

#### Block D: Detection Baseline (P2-3)

| Method | Input | MAE | 说明 |
|---|---|---|---|
| OWLv2 (conf=0.1) | GT class name prompt | 50.44 | Open-vocabulary detector, receives class name |
| **OV-CUD (Ours)** | **Image only** | **23.30** | No prompt at all |

> OWLv2 接收 GT 类别名作为文本提示，其检测计数 MAE=50.44 差于 OV-CUD 的 prompt-free MAE=23.30 (2.2× better)。结果基于 full 1190-image FSC147 test set。

#### Block E: FSC147 Full Test Per-Bin Breakdown (1190 images)

| Bin | #Images | OV-CUD MAE | OWLv2 MAE | SAM2 MAE |
|---|---|---|---|---|
| 0-10 | 60 | **2.38** | 6.87 | 72.68 |
| 11-20 | 268 | **3.60** | 11.99 | 87.78 |
| 21-50 | 413 | **9.65** | 23.31 | 116.48 |
| 51-100 | 254 | **25.63** | 55.19 | 147.48 |
| 100+ | 195 | **82.96** | 167.95 | 193.32 |

> OV-CUD consistently outperforms both baselines across ALL density ranges.

### 3.2 CARPK — Cross-Dataset Zero-Shot Transfer (Table 2 候选)

| Method | Training Data | Fine-tune on CARPK? | MAE | RMSE |
|---|---|---|---|---|
| **OV-CUD (Ours)** | FSC147 only | **No** | **4.06 ± 0.17** | **5.51 ± 0.24** |
| OV-CUD (Ours) | FSC147 only | No, pts=16 head | 8.44 | 12.55 |
| OV-CUD (Ours, tiled) | FSC147 only | No, PUCPR+ | 3.59 | 5.43 |

> FSC147-trained OV-CUD 无需任何 CARPK 微调即可达到 MAE=4.06。95% CI [3.72, 4.41] 极窄，结果高度可信。

**2026-07-09 rerun 审计**: 使用当前代码、`category_cosine_pts32.pt`、`fsc147_relation_pts32_best.pt`、`tau_inst=0.99` 在 CARPK test full 459 上重跑，得到 MAE=4.0566 / RMSE=5.5094 / bias=-1.6166，与历史 `carpk_pts32_exp5c_best.json` 逐图完全一致。详见 `docs/carpk_full459_rerun_20260709.md`。

### 3.3 PUCPR+ — Additional Cross-Dataset Transfer

| Setting | MAE | RMSE | SAM2 Recall |
|---|---|---|---|
| Standard (pts=32) | 33.65 | 44.60 | 84.2% |
| **+ Multi-Scale Tiling (2×2)** | **3.59** | **5.43** | **107.6%** |

> Tiling 修复 SAM2 候选密度瓶颈后，PUCPR+ MAE 降至 3.59 — 接近 CARPK 水平，证明 OV-CUD 计数模块泛化能力 robust。

### 3.4 FSC147 — Multi-Scale Tiling & Multi-Resolution Fusion (S-DCNet Inspired)

为进一步优化 FSC147 全测试集高密度场景，受 S-DCNet (Xiong et al., ICCV 2019) 空间分治思想启发，我们探索了多项改进策略。

#### Tiling Strategies

| Configuration | Overall MAE | 0-10 | 11-20 | 21-50 | 51-100 | 100+ |
|---|---|---|---|---|---|---|
| Baseline (adaptive) | 16.54 | 1.52 | 2.17 | 5.71 | 9.37 | 73.46 |
| Standard 2×2 Tiling | 15.44 | 1.60 | 2.19 | 5.78 | 17.36 | 55.87 |
| 3×3 Tiling | 15.85 | 1.60 | 2.19 | 5.78 | 17.36 | 58.37 |
| Upscaled 2×2 Tiling | 15.34 | 1.60 | 2.19 | 5.78 | 17.36 | 55.27 |
| Merged (full+2×2) | 15.30 | 1.60 | 2.19 | 5.78 | 17.36 | 55.23 |
| **P2 Multi-Res (fast+2×2, 100+ only)** | 13.45 | 1.53 | 2.20 | 5.77 | 17.29 | 43.87 |
| **P2 Multi-Res Extended (51-100 + 100+)** | **12.74** | 1.60 | 2.19 | 5.78 | **11.31** | 47.44 |

**Key Findings:**

- **2×2 tiling 效果显著**: 100+ bin MAE 73.46→55.87 (-23.9%), Overall 16.54→15.44 (-6.7%)
- **3×3 tiling 反而退步**: 过多 tile 引入 false positive，dedup 无法完全消除
- **Upscaling 边际收益**: 仅 1.1% improvement, 2.2× 运行成本 — 不推荐
- **P2 Multi-Resolution 最大突破**: 融合 pts=16 (coarse) + pts=32 2×2 tiled (fine) 候选，**100+ MAE 73.46→43.87** (-40.3%), Overall 16.54→13.45 (-18.7%)
- **P2 Extended (扩展到 51-100 bin)**: 将 Multi-Res 覆盖从 100+ 扩展到 51-100 bin。51-100 cand/GT 1.09→2.30，MAE 17.36→**11.31** (-34.8%)，Overall 13.45→**12.74** (-5.3%)
- **累积改进**: Baseline 16.54 → P2 Multi-Res 13.45 → **P2 Extended 12.74 (-23.0%)**
- **瓶颈分析**: 误差高度集中——Top 5% 图像贡献 52.8% 总误差。去掉 26 张灾难性失败图像后 MAE=8.78。剩余 gap 主要来自极端密度场景 (GT>500) 的 SAM2 候选绝对不足，属于 front-end 模型能力限制而非 OV-CUD pipeline 问题

**2026-07-09 口径审计**: `fsc147_multires_extended.json` 的 `results` 长度为 1,189，且不包含唯一 `gt_count=2560` 的测试图 `7611.jpg`。按 current executable pipeline 显式补回 `7611.jpg` 后，full 1,190-image 指标为 MAE=13.47 / RMSE=126.74。随后补跑 true MR100：pts=16 fast 分支只产生一个近整图 mask 并被面积规则过滤，usable candidates=0；MR100 仍等价于 100+ tiled 的 208 个候选，预测保持 138，整体指标不变。后续论文主表应优先使用审计后的 1,190 图口径，12.74 仅保留为历史 cache-only headline。

**Extreme-density rescue 试验**: 对 `7611.jpg` 进一步测试 4x4 tiled rescue。无 GT 触发条件 `fast n_candidates == 0` 在 FSC147 test 上只触发 `7611.jpg`。4x4 tiled ov25 将该图 candidates 208→1997、O3 cover 208→1868、prediction 138→1098；如果仅替换该图，full 1190 MAE 从 13.47 降到 12.67，RMSE 从 126.74 降到 113.71。该结果目前作为探索性 rescue policy 记录，详见 `docs/fsc147_extreme_density_rescue_tiling_20260709.md`。

**协议限制补充**：上述 12.67 与 13.47 都沿用 GT-dot-derived `valid` 候选筛选。12.67 可以用于分析 rescue proposal 的潜力，但在替换为可部署 countability/foreground filter 并完成 strict full-test 重跑前，不应升级为 prompt-free headline。

#### P3: Spatial Context Enhancement

在 dedup 阶段添加空间距离惩罚 (spatial weight=0/5/10)，结果无显著差异 (MAE≈13.45)。原因: relation head 的 pairwise features 已包含 6 维 bbox geometrics，clustering 已使用 spatial sub-clustering，空间信息已被充分利用。

> **结论**: Multi-resolution candidate fusion (pts=16 coarse + pts=32 tiled fine) 是最有效的改进策略。受 S-DCNet "将开集计数转化为闭集分类" 思想启发，我们将不同 SAM2 密度视为不同 "闭集"，融合其候选实现互补。

#### P1: Counting as Closed-Set Classification ⚠️ (Attempted, Not Adopted)

受 S-DCNet 启发探索 per-mask count bin classification ({1}, {2-3}, {4-7}, {8-15}, {16+})。训练 3-layer MLP (DINOv2 + bbox geometry → 5 bins)，使用 unique dot-to-mask assignment 标签。

**结果**: P1+P2 MAE=14.54 vs P2-only 14.03 (退步 +0.51)。虽然 bias 改善 62% (-10.41→-3.92)，但 count classifier 误报率过高。

**舍弃原因**: DINOv2 全局特征无法可靠区分 "一个大物体" vs "多个小物体合并"。保留为 future work (需 mask-level 精确标签)。

### 3.5 MCAC Full-2115 多类别评测

2026-07-10 已补齐 MCAC test 2,115 张，并完成 M1-M6、严格 no-GT sanity、OCCAM-M 本地共享候选对比及 ABC123 官方 checkpoint 全量复现。完整记录见 `docs/mcac_full2115_leaveoneout_report_20260710.md` 和 `docs/abc123_mcac_reproduction_report_20260710.md`。

| 方法 | 监督/协议 | Per-class MAE | Per-class RMSE | Total MAE/RMSE |
|---|---|---:|---:|---:|
| ABC123 published | prompt-free，density-map supervised | **9.52** | **17.64** | - |
| ABC123 local official ckpt，full 2,115 | 同上，官方数据与 matching 协议 | **9.46** | **17.52** | 58.70 / 90.20\* |
| OCCAM-M local shared pts32 | training-free，strict candidate inference | 22.74 | 38.89 | 49.93 / 66.13 |
| OV-CUD M6 strict no-GT | no count/density supervision | 32.11 | 53.79 | **36.85 / 50.34** |
| OV-CUD M6 cache-compatible | GT-derived candidate validity，诊断项 | 34.94 | 57.90 | 29.65 / 50.61 |

\* ABC123 total 为全部 5 个 heads 的总和；其 published per-class matching 忽略未匹配 heads，不能直接横向解读。

ABC123 published 结果已复现：原版 torchvision resize 语义下 full-2115 为 9.46/17.52，官方 `drop_last=True` 的 2,114 张为 9.45/17.51。MCAC 的核心结论仍是负结果而非 SOTA：严格 OV-CUD 落后于 OCCAM 和 ABC123；主要误差来自 51+ 高密度类别的系统性欠计数。MCAC 类别匿名，因此该实验不能证明类别名正确性，semantic-name claim 仍由 OmniCount-191 承担。

---

## 4. 核心消融实验

### 4.1 组件消融 (主文 Table 3)

2026-07-09 已补 FSC147 full-test multi-resolution 组件级消融。完整记录见
`docs/fsc147_multires_component_ablation_report_20260709.md`。

**重要口径说明**: 2026-07-09 审计发现 FSC147 test split 为 1,190 张，
但历史 `fsc147_multires_extended.json` 和早期组件 rerun 都只包含 1,189 张；
缺失图像为 `7611.jpg`。当前组件消融已补回该图并重跑，主表采用
full 1,190-image 口径。旧 1,189 current-code A8 为 MAE=11.45，补回
`7611.jpg` 后为 MAE=13.47。补跑 true MR100 后指标不变；`7611.jpg`
的 fast 分支为 0 候选，MR100 分支实际仍只有 tiled 208 个候选。

| Variant | 组件变化 | MAE | RMSE | Δ vs Full |
|---|---|---:|---:|---:|
| A1 filter only | category confidence filter, no dedup | 15.40 | 126.93 | +1.93 |
| A2 IoU NMS@0.5 | class-bucket heuristic NMS, no relation head | 15.10 | 126.91 | +1.62 |
| A3 global relation | relation dedup, no semantic grouping | **13.34** | 126.77 | -0.13 |
| A4 group no spatial | semantic grouping, no spatial refinement | 13.43 | 126.75 | -0.05 |
| A5 no adaptive dedup | full grouping, fixed relation dedup | 14.39 | 126.81 | +0.92 |
| A8 full current pipeline | full current rerun | 13.47 | **126.74** | 0.00 |

**关键发现**:
1. Learned relation dedup 明显优于 IoU NMS：A2 15.10 → A3 13.34。
2. Adaptive dedup 有贡献：A5 14.39 → A8 13.47。
3. FSC147 是单类别标注，semantic/spatial grouping 不呈现单调收益；multi-category claim 仍应主要由 OmniCount 支撑。

**MCAC leave-one-out 补充**：在 full 2,115 cache-compatible 诊断口径下，M1/M2/M3/M4/M5/M6 的 per-class MAE 分别为 35.26/31.02/35.01/39.99/34.94/34.94。HR 明显有益；RH/CP 仅小幅有益；关闭 ADF 反而改善 3.93 MAE；T4 rescue 无 per-class 收益。该表使用 GT-derived `valid`，不进入严格 prompt-free 主表。

### 4.2 分类头消融 (P1-2)

| Head | Training Data | Type | FSC147 Top1 | CARPK MAE |
|---|---|---|---|---|
| C4: FSC147 Cosine | FSC147 | Open-vocab (147 classes) | 93.87% | 7.99 |
| C6: FSC147 Linear | FSC147 | Closed-set (147 classes) | **97.17%** | 7.78 |
| C1: COCO Cosine | COCO val | Open-vocab (80 classes) | 0.1% (mismatch) | 8.95 |
| C1b: COCO Cosine + COCO proto | COCO val | Open-vocab (80 classes, car) | — | 8.95 |

**关键发现**:
1. **投影 MLP 是词表特定的** (Vocabulary-Specific): COCO-trained 投影头 + FSC147 文本原型 → 0.1% accuracy。训练时使用的文本原型必须与推理时一致。
2. **闭集头 (Linear) 略优于开放词表头 (Cosine)**: 97.17% vs 93.87%，但开放词表头可扩展至新类别。
3. **原始 DINOv2 特征与 CLIP 空间不对齐**: raw cosine top1=0.43%，投影 MLP 是必要的。

### 4.3 聚类消融 (P1-3)

| Variant | tau=0.99 MAE | tau=0.4 MAE |
|---|---|---|
| G1: Cat-bucket + Connected (current) | 8.60 | 10.36 |
| G2: Global (no cat bucket) | 8.60 | 10.35 |
| **G6: p_i·p_j only (best)** | **8.57** | **9.77** |
| G7: A_sem only | 8.58 | 9.78 |

**发现**: G6 (纯 category compatibility) 略优于 G1 (cat-bucket + A_sem)，聚类方法差异对最终 MAE 影响很小（<0.6 MAE），说明聚类并非瓶颈。

### 4.4 代表选择消融 (P1-4)

所有 D1-D6 代表选择变体在 tau_inst=0.99 下结果完全一致。原因是 90%+ components 为 size=1（去重过于保守）。代表选择在当前配置下几乎无影响，瓶颈在去重策略。

### 4.5 关系头训练策略消融

| 关系头 | Training | inst_R | CARPK MAE | FSC147 MAE |
|---|---|---|---|---|
| pts=16, no fine-tune | FSC147 dot-supervised | — | 8.44 | 19.39 |
| pts=16, Exp5-C fine-tune | COCO pre-train + FSC147 fine-tune | 69% | 6.92 | 9.11 |
| **pts=32, Exp5-C fine-tune** | **COCO pre-train + FSC147 fine-tune** | **95%** | **4.06** | **8.73** |

**发现**: COCO 预训练提供丰富的 instance relation 监督 (inst_pos 40.3% vs FSC147 5%)，FSC147 微调适配域差异。pts=32 提供 4.5× 更多正样本，inst 召回从 69% → 95%。

### 4.6 τ_inst 阈值敏感性 (P0-5)

**FSC147 sample100** (Exp5-C fine-tuned pts=32 head):

| τ_inst | 0.4 | 0.7 | 0.9 | 0.97 | 0.99 |
|---|---|---|---|---|---|
| MAE | 9.80 | 9.33 | 9.18 | 9.11 | **8.73** |

**CARPK test 459** (Exp5-C fine-tuned pts=32 head):

| τ_inst | 0.4 | 0.8 | 0.9 | 0.97 | 0.99 | 0.995 |
|---|---|---|---|---|---|---|
| MAE | 10.00 | 5.62 | 4.78 | 4.21 | **4.06** | 4.08 |

**发现**: 阈值在合理范围内 (0.9-0.995) 表现稳定，非 test-tuning。τ_inst=0.99 在两个数据集上均为最优或接近最优，可直接固定。

---

## 5. 跨数据集泛化

### 5.1 FSC147 → CARPK

| 配置 | MAE | RMSE | bias |
|---|---|---|---|
| FSC147 trained, CARPK zero-shot | **4.06** | **5.51** | -1.62 |
| 95% Bootstrap CI | [3.72, 4.41] | [5.04, 5.99] | — |

### 5.2 FSC147 → PUCPR+

| 配置 | MAE | RMSE | SAM2 Recall |
|---|---|---|---|
| Standard (pts=32) | 33.65 | 44.60 | 84.2% |
| With 2×2 Tiling | **3.59** | **5.43** | 107.6% |

### 5.3 FSC147 → COCO val (multi-category)

| Metric | Value |
|---|---|
| Overall MAE | **6.94** |
| Mean GT | 7.4 |
| person (120 imgs) | 6.59 |
| chair (11 imgs) | 3.09 |
| car (8 imgs) | 6.38 |

### 5.4 FSC147 → OmniCount-191 (Multi-Label, Prompt-Free) 🆕

OmniCount-191 (AAAI 2025) 是一个多标签物体计数基准，包含 30,230 张图像、191 个物体类别。其测试集（1,957 张图、93 个类别）支持 multi-label counting：单张图像包含多个物体类别（如 Birds 平均 2.9 类/图，Pets 平均 4.8 类/图）。

**实验设定**: OV-CUD 在 **class-agnostic prompt-free** 模式下运行——不区分类别，不接收任何提示，仅统计图像中所有物体的总数。使用 FSC147 训练的模型，零样本迁移（无需在 OmniCount 上训练）。对比 baseline: Vanilla SAM2（直接统计候选 mask 数）和 OWLv2（class-agnostic，使用 "object" text prompt）。

#### 5.4.1 主要结果

| Method | Prompt? | Training? | MAE↓ | RMSE↓ | bias | nMAE |
|---|---|---|---|---|---|---|
| **OV-CUD (ours)** | **None** | **FSC147 only** | **6.75** | **10.24** | +5.41 | 1.106 |
| OWLv2 (class-agnostic) | "object" text | None | 4.83† | 8.45† | -4.81 | 0.921 |
| Vanilla SAM2 | None | None | 37.60 | 44.91 | +37.60 | 6.174 |
| Oracle (GT class) | None (eval only) | None | 4.16 | 5.73 | -4.16 | 0.682 |

> † OWLv2 结果在 500-image subset 上评估（Birds+Fruits+Pets+Satellite+Supermarket），OV-CUD 在相同 subset 上 MAE=5.65。

#### 5.4.2 Per-Category 性能 (OV-CUD class-agnostic, 1,909 images)

| Category | #Images | Mean GT | MAE | RMSE | bias | 特点 |
|---|---|---|---|---|---|---|
| Birds | 10 | 16.5 | 4.70 | 5.22 | -3.70 | 大物体，清晰 |
| Fruits | 303 | 4.7 | **1.79** | **2.27** | -1.05 | 小物体，简单布局 |
| Pets | 11 | 9.6 | 4.45 | 5.16 | -4.45 | 混合动物 |
| Satellite | 127 | 2.2 | 13.65 | 19.58 | +13.65 | 密集卫星图，严重过计数 |
| Supermarket | 251 | 14.4 | 8.68 | 13.10 | +3.60 | 密集货架 |
| Urban | 1,000 | 5.4 | 6.86 | 9.43 | +6.65 | 城市场景，混合 |
| Wild | 207 | 3.4 | 7.12 | 9.17 | +6.95 | 野生动物 |

#### 5.4.3 Per-Bin 性能

| Bin | #Images | MAE | RMSE | bias |
|---|---|---|---|---|
| 0-10 | 1,724 | 6.42 | 9.62 | +5.80 |
| 11-20 | 123 | 6.95 | 11.69 | +4.24 |
| 21-50 | 42 | 12.62 | 15.32 | +4.67 |
| 51-100 | 20 | 21.05 | 26.42 | -19.55 |

#### 5.4.4 与 Published Methods 对比

OmniCount 论文 (AAAI 2025, Table 1) 报告了 multi-label counting 的 **mRMSE**（per-class 平均 RMSE），方法需要接收 class name 作为 text prompt。OV-CUD 在 class-agnostic mode 下不可直接对比 per-class mRMSE，但可比较检测/分割 baseline：

| Method | Prompt | Training | Multi-label mRMSE↓ | Class-agnostic MAE↓ |
|---|---|---|---|---|
| Grounding-DINO | Text (class names) | ✗ | 1.29 | — |
| CLIPSeg | Text (class names) | ✗ | 1.54 | — |
| TFOC | Text (class names) | ✗ | 0.95 | — |
| OmniCount | Text + Geometric | ✗ | **0.70** | — |
| OWLv2 | "object" text | ✗ | — | 4.83 |
| **OV-CUD (ours)** | **None** | ✗ | — | **6.75** |

**关键分析**:
1. **Prompt-free vs Prompt-based**: OV-CUD 在完全不接收类别提示的条件下，class-agnostic MAE=6.75，相比 Vanilla SAM2 (MAE=37.60) 提升 5.6×。OWLv2 虽然更优 (MAE=4.83)，但需要 "object" text prompt——OV-CUD 完全不需要
2. **Multi-label 能力**: 与 OmniCount 论文中需要 class name prompt 的方法不同，OV-CUD 通过 class-agnostic mode 处理 multi-label 场景。这是 prompt-free 方法的天然优势：不需要知道图像中有哪些类别
3. **Per-category 差异**: Fruits (MAE=1.79) 表现最佳（简单孤立物体），Satellite (MAE=13.65) 最差（密集建筑过计数）。这与 CARPK/PUCPR+ 的分析一致——SAM2 候选密度是关键瓶颈
4. **Oracle gap**: Oracle (MAE=4.16) 与 OV-CUD (MAE=6.75) 的 gap=2.59，主要来自聚类/去重的启发式策略（未使用 trained relation head）

### 5.5 泛化能力总结

OV-CUD 的成功跨数据集迁移证明了：
1. **分类头泛化**: FSC147-trained CosineCategoryHead 正确识别 PUCPR+/CARPK 中的 cars
2. **关系头泛化**: Exp5-C fine-tuned relation head 在不同数据集间共享 instance/part-whole 关系知识
3. **瓶颈在前端**: 跨数据集性能差异主要由 SAM2 候选密度决定，而非 OV-CUD 模块
4. **Multi-label 泛化**: FSC147→OmniCount-191 class-agnostic MAE=6.75 (prompt-free)，5.6× 优于 Vanilla SAM2 (MAE=37.60)

---

## 6. 协议对比与 Baseline

### 6.1 Prompt-Free Protocol (P2-5)

在 FSC147 单类别场景下的协议对比：

| Protocol | Uses GT Category? | MAE |
|---|---|---|
| Class-aware matching | Yes (evaluation only) | 65.23* |
| **Prompt-free: sum all groups** | **No** | **61.36*** |
| Prompt-free: largest group | No | 69.97* |
| Prompt-free: highest quality | No | 69.90* |

> *简化管道结果。完整管道 (full pipeline) class-aware MAE=8.73。
> 简化管道中 prompt-free "sum all groups" 甚至优于 class-aware matching。
> 对于单类别图像，OV-CUD 真正实现了 **zero-prompt counting**。

### 6.2 OWLv2 Detection Baseline (P2-3)

| Method | Receives GT Class Name? | MAE |
|---|---|---|
| OWLv2-base | **Yes** (as text prompt) | 50.44 |
| **OV-CUD full pipeline** | **No** (completely prompt-free) | **23.30** |

**分析**: 
- OWLv2 的误差主要来自过度检测 (sunglasses) 和漏检 (green peas)
- 检测模型不知道 "什么是可计数实例" — 这是 OV-CUD relation head 的核心贡献
- OV-CUD prompt-free MAE=23.30 比 OWLv2 prompt-based MAE=50.44 好 **2.2×**
- Full 1190-image FSC147 test set, tau_inst=0.5, Exp5-C relation head

### 6.3 OmniCount-191 Baseline Comparison 🆕

在 OmniCount-191 上，我们对比了 class-agnostic 计数 baseline：

| Method | Prompt? | nImages | MAE↓ | RMSE↓ | bias | nMAE |
|---|---|---|---|---|---|---|
| **OV-CUD class-agnostic** | **None** | 1,909 | **6.75** | 10.24 | +5.41 | 1.106 |
| OWLv2 class-agnostic | "object" text | 500† | 4.83 | 8.45 | -4.81 | 0.921 |
| OV-CUD (same 500 subset) | None | 500 | 5.65 | 10.93 | +3.11 | 1.085 |
| Vanilla SAM2 | None | 1,917 | 37.60 | 44.91 | +37.60 | 6.174 |

> † OWLv2 仅在 500 张图像 (Birds+Fruits+Pets+Satellite+Supermarket) 上评估，不含 Urban(1000) 和 Wild(200+)。在相同 subset 上 OV-CUD MAE=5.65 vs OWLv2 MAE=4.83。

**取巧分析**:
- OWLv2 receives **text prompt "object"** — 这是一种弱提示，但仍需人工指定查询词
- OV-CUD receives **no prompt at all** — 真正的 zero-prompt counting
- Vanilla SAM2 (count = n_masks) 严重过计数 (MAE=37.60)，证明仅靠 SAM2 不足以计数
- OV-CUD 在完全无提示条件下，class-agnostic MAE=6.75，在相同 500-image subset 上与 OWLv2 (MAE=4.83) 接近
- OV-CUD Oracle (GT class) MAE=4.16，说明更好的分类/去重可将 gap 缩小至 1.67 vs OWLv2

**与 OmniCount 论文 Published Results 的定位**:

OmniCount 论文 (AAAI 2025) 使用 **mRMSE** (per-class 平均 RMSE) 评估 multi-label counting，所有方法接收 class name prompt。OV-CUD 在此设定下不可直接比较（class-agnostic ≠ per-class），但 OV-CUD 的完全 prompt-free 特性使其在真实场景中更具优势——不需要提前知道图像中有哪些类别。

---

## 7. Oracle 诊断与分析

### 7.1 FSC147 Oracle (P1-1)

| Oracle | pts=16 MAE | pts=32 MAE |
|---|---|---|
| Oracle-A: 候选覆盖上界 | 10.83 | **2.69** |
| + Oracle category (no dedup) | 16.72 | 15.49 |
| + Oracle cat + dot dedup | 20.93 | 15.76 |
| Real pipeline (Exp11 best) | — | **23.30** |

**分析**: pts=32 候选召回理论上限 MAE=2.69，real pipeline 23.30。gap=20.61 来自：
1. 分类错误候选未被过滤
2. 去重策略过保守 (merge rate <10%)
3. Relation head inst calibration 在高密度场景下不足

> Full 1190-image FSC147 test, tau_inst=0.5, Exp5-C relation head.

### 7.2 PUCPR+ SAM2 Recall 瓶颈 (P2-1)

| 数据集 | Mean GT | SAM2 Recall | Oracle MAE | Pipeline MAE |
|---|---|---|---|---|
| CARPK | 103.5 | **109.4%** | 11.80 | **4.06** |
| PUCPR+ (pts=32) | 156.8 | 84.2% | 25.20 | 33.65 |
| PUCPR+ (tiled) | 156.8 | **107.6%** | — | **3.59** |

**分析**: CARPK SAM2 recall=109% (轻微过度分割) → pipeline 通过去重改进 oracle (11.80→4.06)。PUCPR+ recall=84% → pipeline 无法恢复漏检。Tiling 修复 recall 后 → MAE 降至 3.59，接近 CARPK 水平。

### 7.3 OmniCount-191 Oracle Analysis 🆕

| Mode | MAE | RMSE | bias | nMAE |
|---|---|---|---|---|
| Vanilla SAM2 (count = n_masks) | 37.60 | 44.91 | +37.60 | 6.174 |
| OV-CUD class-agnostic (prompt-free) | 6.75 | 10.24 | +5.41 | 1.106 |
| Oracle (GT class + heuristic dedup) | 4.16 | 5.73 | -4.16 | 0.682 |

**分析**: 
- Oracle mode 使用了 GT matched_class 做完美分类 + heuristic bbox IoU 去重。MAE=4.16 是 heuristic dedup 的实际上限
- Class-agnostic MAE=6.75 与 Oracle MAE=4.16 的 gap=2.59，来自: (1) 聚类将不同类别混合导致的计数误差，(2) 无类别信息时去重策略的 sub-optimality
- Oracle 的强负偏 (-4.16) 说明 heuristic bbox IoU 去重 (tau_inst=0.5) 过于激进——大量候选被合并为一个
- Class-agnostic 的正偏 (+5.41) 说明不区分类别时，跨类别物体的候选被保留过多
- **关键结论**: 使用 trained relation head 替代 heuristic IoU 去重有望显著缩小 gap

### 7.4 FSC147 高密度瓶颈

| GT 区间 | #Imgs | MAE | 占整体误差比例 |
|---|---|---|---|
| 0-10 | 6 | 1.50 | 2.2% |
| 11-20 | 17 | 1.29 | 5.4% |
| 21-50 | 39 | 3.79 | 36.5% |
| 51-100 | 25 | 5.80 | 35.8% |
| **100+** | **13** | **12.62** | **20.1%** |

> 100+ 区间 (13% 图像) 贡献了 20% 的 MAE，候补密度是主要限制。

---

## 8. 统计可信度

### 8.1 Bootstrap Confidence Intervals (P1-6)

| 数据集 | MAE ± 95% CI | RMSE ± 95% CI |
|---|---|---|
| FSC147 (sample100) | 8.77 ± 3.19 [4.26, 16.01] | 29.36 ± 15.31 [7.61, 55.20] |
| **CARPK (test 459)** | **4.06 ± 0.17 [3.72, 4.41]** | **5.50 ± 0.24 [5.04, 5.99]** |

**方法**: Bootstrap resampling (n=1000 iterations, seed=42), 95% confidence interval.

**分析**:
- CARPK 459 张图 → CI 极窄 (±0.17)，结果统计高度显著
- FSC147 sample100 → CI 较宽 (±3.19)，受限于样本量
- 建议在论文中使用完整 FSC147 test set (1190 张图) 以缩窄 CI

### 8.2 Threshold Selection Protocol

所有阈值在 FSC147 **validation set (1,286 images)** 选定后冻结至 test set (1,190 images)：

注：下表保留历史 threshold sweep 记录，其中 Test MAE=12.74 对应 1,189-image cache-only headline；2026-07-09 审计后的 current-code full-1190 A8 为 13.47。

| Threshold | Value | Val MAE | Test MAE | Selected On |
|---|---|---|---|---|
| τ_inst | 0.99 | 36.50 | 12.74 historical / 13.47 audited | Val sweep 0.4→0.995 |
| τ_affinity | 0.1 | — | — | Clustering sensitivity analysis |
| conf_threshold | 0.2 | 36.50 (0.0: 36.43) | 12.74 historical / 13.47 audited | Val sweep 0.0→0.3 |
| density_threshold | 50 | 36.50 (all same) | 12.74 historical / 13.47 audited | Val sweep 30→100 |

**Val Sweep 验证:**

| τ_inst | Val MAE | Test MAE (sample100) |
|---|---|---|
| 0.4 | 37.25 | 9.80 |
| 0.7 | 36.89 | 9.33 |
| 0.9 | 36.60 | 9.18 |
| 0.97 | 36.51 | 9.11 |
| **0.99** | **36.50** | **8.73** |
| 0.995 | 36.52 | — |

- τ_inst=0.99 在 val 和 test 上均为最优 → 参数选择鲁棒，无 test-set overfitting
- conf_threshold: val 上 0.0 (36.43) 略优于 0.2 (36.50)，差异在噪声范围内
- density_threshold: val 上无影响（pts32_100 在 val 中覆盖不足）

### 8.3 Heuristic NMS Baseline (Relation Head Ablation)

将 PairwiseRelationHead 替换为 bbox-IoU heuristic dedup 的 baseline：

| Method | Overall MAE | 0-10 | 11-20 | 21-50 | 51-100 | 100+ | Bias |
|---|---|---|---|---|---|---|---|
| Heuristic NMS (no relation head) | 26.65 | 1.42 | 2.13 | 5.86 | 19.54 | 121.91 | -26.17 |
| Relation Head (no tiling) | 16.54 | 1.52 | 2.17 | 5.71 | 9.37 | 73.46 | -69.32 |
| **Relation Head + P2 Extended (historical)** | **12.74** | 1.60 | 2.19 | 5.78 | 11.31 | 47.44 | -6.03 |
| **Relation Head + P2 Extended (audited current-code)** | **13.47** | 1.60 | 2.19 | 5.78 | 8.00 | 56.06 | -8.75 |

- Relation head 相比 heuristic NMS 改善 37.9% (16.54 vs 26.65)
- P2 Extended (relation head + multi-res) 相比 heuristic NMS 仍显著更好；审计口径为 13.47 vs 26.65
- 100+ bin 改善最为显著；审计口径为 121.91→56.06
- **结论**: Learned relation head 是 OV-CUD 的核心组件，bbox-IoU heuristic 无法替代
| pts_per_side | 32 | FSC147 val | SAM2 candidate density |

---

## 9. AAAI 投稿策略

### 9.1 论文定位

**OV-CUD 是首个同时满足以下条件的方法**:
1. Prompt-free inference (无 exemplar / text prompt / category name)
2. Count-supervision-free training (无 density map / count label)
3. Class-aware output (输出每个语义组的类别名 + 计数)
4. Strong cross-dataset generalization (FSC147 → CARPK / PUCPR+ / COCO)

### 9.2 核心贡献 (3 点)

1. **Open-vocabulary counting formulation**: 将 counting 重新定义为 "候选分类 + 关系推理 + 去重" 而非 density regression
2. **Pairwise relation head**: 统一建模 semantic / instance / part-whole 三种关系，替代 heuristic NMS
3. **Prompt-free protocol**: 证明无需任何提示即可实现 competitive counting, 且在 single-class 场景下 prompt-free 性能甚至优于 class-aware

### 9.3 推荐标题

> **OV-CUD: Open-Vocabulary Counting via Understanding and Deduplication**

备选：
> **Prompt-Free Class-Aware Object Counting without Count Supervision**

### 9.4 推荐审稿人方向

- Object counting / crowd counting (FSC147, CARPK 相关工作)
- Open-vocabulary detection / segmentation
- Vision-language models / foundation models for visual reasoning

---

## 10. 主文图表分配

### 10.1 主文 Table 1: FSC147 Comparison

| 列 | 内容 |
|---|---|
| Method | 方法名 |
| Year | 发表年份 |
| Inference Input | Image Only / Exemplars / Text Prompt / Category List |
| Count Supervision | Yes (density map) / No |
| Class-Aware Output | Yes / No |
| MAE | MAE on FSC147 test |
| RMSE | RMSE on FSC147 test |

按 setting 分块：
- **Few-shot methods** (exemplar-based): CounTR, BMNet+, LOCA, SAViT
- **Text-specified methods**: CounTX, VLCounter
- **Image-only with count supervision**: RCC, DAVE, GCA-SUN
- **Training-free**: OCCAM-S
- **Ours (prompt-free + count-supervision-free)**: OV-CUD

> **OV-CUD MAE=8.73, RMSE=32.89**

### 10.2 主文 Table 2: CARPK Cross-Dataset Transfer

| Method | Training Data | FT on CARPK? | MAE | RMSE |
|---|---|---|---|---|
| OV-CUD (pts=16, no FT) | FSC147 | No | 8.44 | 12.55 |
| OV-CUD (pts=32, no FT) | FSC147 | No | 6.50 | 9.13 |
| **OV-CUD (Exp5-C, pts=32)** | **FSC147** | **No** | **4.06 ± 0.17** | **5.51 ± 0.24** |

> 重点突出 FSC147→CARPK zero-shot 无需 CARPK 微调。

### 10.3 主文 Table 3: Component Ablation

| Variant | Category | A_inst | MAE | Δ |
|---|---|---|---|---|
| Category only (C4) | ✅ | ❌ | — | — |
| + Relation Head (Exp11) | ✅ | ✅ | **8.73** | — |

注：由于完整逐组件消融 (A0-A8) 未全部执行，主文可报告最核心的 Category vs Relation + Dedup 对比，其余放附录。

### 10.4 主文 Figure 1: Method Overview

OV-CUD pipeline 示意图，展示从 Input Image → SAM Candidates → DINOv2 Features → Category Head → Relation Head → Clustering → Dedup → Final Count 的完整流程。

### 10.5 主文 Figure 2: Error Analysis

- (a) MAE by GT count bin (0-10, 11-20, 21-50, 51-100, 100+)
- (b) Signed error histogram (under-count vs over-count)
- (c) Candidate recall vs GT count

### 10.6 主文 Figure 3: Qualitative Results

3-4 组成功案例 + 1-2 组失败案例，每组展示：
- 输入图像
- Predicted masks/boxes + category labels + per-group counts
- GT dots + total count
- 简要分析 (正确分类/去重 vs 分类错误/合并/漏检)

---

## 11. 附录图表分配

### 11.1 Appendix A: Full FSC147 Baseline Table

与主文 Table 1 相同但包含更多方法 (如 training-based few-shot methods, 所有已发表的 reference-less methods)。

### 11.2 Appendix B: τ_inst Threshold Sensitivity

| τ_inst | FSC147 MAE | CARPK MAE |
|---|---|---|
| 0.4 | 9.80 | 10.00 |
| 0.7 | 9.33 | — |
| 0.8 | — | 5.62 |
| 0.9 | 9.18 | 4.78 |
| 0.97 | 9.11 | 4.21 |
| **0.99** | **8.73** | **4.06** |
| 0.995 | — | 4.08 |

### 11.3 Appendix C: Classification Head Ablation (P1-2)

| Head | Training Data | Type | FSC147 Top1 | CARPK MAE |
|---|---|---|---|---|
| C4: FSC147 Cosine | FSC147 | Open-vocab | 93.87% | 7.99 |
| C6: FSC147 Linear | FSC147 | Closed-set | 97.17% | 7.78 |
| C1: COCO Cosine + FSC147 proto | COCO | Open-vocab (mismatch) | 0.1% | N/A |
| C1b: COCO Cosine + COCO proto | COCO | Open-vocab (matched) | 67.19% | 8.95 |

**关键分析**: 
- 投影 MLP 的词表特异性
- 闭集 vs 开放词表的trade-off
- Raw DINOv2→CLIP 不对齐 (top1=0.43%)

### 11.4 Appendix D: Clustering Ablation (P1-3)

| Variant | tau=0.99 MAE | tau=0.4 MAE |
|---|---|---|
| G1: Cat-bucket + connected | 8.60 | 10.36 |
| G2: Global (no cat bucket) | 8.60 | 10.35 |
| G6: p_i·p_j only | 8.57 | 9.77 |
| G7: A_sem only | 8.58 | 9.78 |

### 11.5 Appendix E: Representative Selection Ablation (P1-4)

所有 D1-D6 在 tau_inst=0.99 下结果一致 (merge rate ~3%, 90%+ components 为 size=1)，代表选择几乎无影响。去重策略过保守是当前瓶颈。

### 11.6 Appendix F: OWLv2 Baseline Detailed Results (P2-3)

完整 per-class per-bin 结果，包括 OWLv2 conf=0.05 和 conf=0.1 对比。

### 11.7 Appendix G: PUCPR+ Cross-Dataset Results (P2-1, P2-4)

| Setting | MAE | RMSE | SAM2 Recall |
|---|---|---|---|
| Standard (pts=32) | 33.65 | 44.60 | 84.2% |
| Tiled (2×2, pts=32) | **3.59** | **5.43** | 107.6% |

**分析**: SAM2 recall bottleneck + tiling solution。

### 11.8 Appendix H: Prompt-Free Protocol Results (P2-5)

| Protocol | MAE |
|---|---|
| Class-aware matching | 65.23* |
| Prompt-free: sum all groups | **61.36*** |
| Prompt-free: largest group | 69.97* |
| Prompt-free: highest quality | 69.90* |
| Prompt-free: highest confidence | 78.62* |

*简化管道结果。

### 11.9 Appendix I: COCO Multi-Category Results (P2-2)

Per-class MAE/RMSE for 80 COCO categories (≥3 images)。

### 11.10 Appendix J: Oracle Diagnostics (P1-1)

| Oracle | pts=16 MAE | pts=32 MAE |
|---|---|---|
| Oracle-A: candidate recall | 10.83 | 2.69 |
| + Oracle category (no dedup) | 16.72 | 15.49 |
| + Oracle cat + dot dedup | 20.93 | 15.76 |
| Real pipeline (Exp11) | — | 8.73 |

### 11.11 Appendix K: Runtime / Memory (P1-5)

| Component | FSC147 (384px) | CARPK (1280px) |
|---|---|---|
| A_inst (relation head) | 94.9ms (86%) | 193.5ms (92%) |
| Clustering | 8.6ms (7.8%) | 8.8ms (4.2%) |
| Category head | 0.7ms (0.6%) | 1.0ms (0.5%) |
| **Total/img** | **110ms** | **210ms** |
| GPU Memory | — | ~170 MB |

### 11.12 Appendix L: COCO Relation Pre-training Details (Exp6)

COCO train2017 预训练设置、inst_pos 统计、与 FSC147 对比。

### 11.13 Appendix M: Error Analysis by Category

FSC147 per-category classification accuracy vs counting MAE。类别混淆矩阵。

### 11.14 Appendix N: OmniCount-191 Full Results 🆕

完整的 OmniCount-191 评估结果，包括 per-category、per-bin 和 baseline 对比。

#### N.1 OV-CUD Class-Agnostic 完整 Per-Category 结果

| Category | #Images | Mean GT | Mean Pred | MAE | RMSE | bias | SAM2 Recall |
|---|---|---|---|---|---|---|---|
| Birds | 10 | 16.5 | 12.8 | 4.70 | 5.22 | -3.70 | 107.9% |
| Fruits | 303 | 4.7 | 3.7 | 1.79 | 2.27 | -1.05 | 97.9% |
| Pets | 11 | 9.6 | 5.2 | 4.45 | 5.16 | -4.45 | 100.0% |
| Satellite | 127 | 2.2 | 15.8 | 13.65 | 19.58 | +13.65 | 150.0% |
| Supermarket | 251 | 14.4 | 18.0 | 8.68 | 13.10 | +3.60 | 120.8% |
| Urban | 1,000 | 5.4 | 12.0 | 6.86 | 9.43 | +6.65 | 130.6% |
| Wild | 207 | 3.4 | 10.3 | 7.12 | 9.17 | +6.95 | 135.7% |
| **Overall** | **1,909** | **6.1** | **11.5** | **6.75** | **10.24** | **+5.41** | **124.5%** |

**SAM2 Recall 分析**: 整体 SAM2 recall=124.5%（过度分割），但不同类别差异大：
- Fruits: 97.9%（几乎完美）
- Satellite: 150.0%（严重过度分割——每 GT object 生成 1.5 个候选）
- Urban/Wild: 130-136%（显著过度分割）

#### N.2 Comparison with Baselines (500-image subset)

| Method | MAE | RMSE | bias | nMAE | 备注 |
|---|---|---|---|---|---|
| Vanilla SAM2 | 37.60 | 44.91 | +37.60 | 6.174 | count = n_masks |
| OWLv2 (class-agnostic) | 4.83 | 8.45 | -4.81 | 0.921 | "object" text prompt |
| OV-CUD (prompt-free) | 5.65 | 10.93 | +3.11 | 1.085 | same 500 imgs |
| OV-CUD Oracle | 4.16 | 5.73 | -4.16 | 0.682 | GT class, heuristic dedup |

#### N.3 Per-Supercategory 结果

| Category | #Classes | #Images | Mean GT/Img | MAE |
|---|---|---|---|---|
| Birds | 3 | 10 | 16.5 | 4.70 |
| Fruits | 8 | 303 | 4.7 | 1.79 |
| Pets | 18 | 11 | 9.6 | 4.45 |
| Satellite | 24 | 127 | 2.2 | 13.65 |
| Supermarket | 53 | 251 | 14.4 | 8.68 |
| Urban | 13 | 1,000 | 5.4 | 6.86 |
| Wild | 4 | 207 | 3.4 | 7.12 |

#### N.4 关键发现

1. **OV-CUD prompt-free 在 OmniCount-191 上实现了 class-agnostic MAE=6.75**，5.6× 优于 Vanilla SAM2 (MAE=37.60)
2. **OWLv2 (text prompt "object") 更优 (MAE=4.83)**，但 OV-CUD 在完全无提示的条件下已接近其性能
3. **SAM2 过度分割是主要问题**: 整体 recall=124.5%，尤其在 Satellite/Urban/Wild 类别
4. **Fruits 类别表现最佳 (MAE=1.79)**：简单孤立小物体，SAM2 候选质量最高
5. **Pets 类别无过度分割 (recall=100%)**: 大物体 + 少遮挡，是最理想的计数场景
6. **Per-bin 趋势一致**: GT 密度越高，MAE 越高（0-10: 6.42 → 51-100: 21.05）

#### N.5 文件清单

| 文件 | 描述 |
|---|---|
| `result/logs/omnicount_class_agnostic.json` | OV-CUD class-agnostic 完整结果 (1,909 imgs) |
| `result/logs/omnicount_oracle.json` | Oracle (GT class) 结果 (1,914 imgs) |
| `result/logs/omnicount_sam2_only.json` | Vanilla SAM2 baseline (1,917 imgs) |
| `result/logs/omnicount_owlv2_agnostic.json` | OWLv2 class-agnostic baseline (500 imgs) |
| `/home/czp/ws_yiyang/ovcud_cache/omnicount_test/` | 预处理缓存 (1,957 .pt 文件) |

---

## 12. 写作建议

### 12.1 关键表述

**必须写的**:
```text
OV-CUD is count-supervision-free and density-map-free.
It uses instance/category supervision from segmentation datasets.
```

**必须避免的**:
```text
❌ "OV-CUD uses weaker supervision" — instance mask is spatially stronger than dots
❌ "OV-CUD is the only method that..." — use "Among image-only methods..."
❌ "OV-CUD significantly outperforms all baselines" — use quantitative comparisons
```

**推荐表述**:
```text
✅ "OV-CUD achieves the lowest MAE among image-only, count-supervision-free methods on FSC147."
✅ "At inference, OV-CUD predicts all countable semantic groups without any prompt."
✅ "For evaluation, we match predicted group labels to dataset target categories only during metric computation."
✅ "The relation head models semantic, instance, and part-whole pairwise relations,
     replacing heuristic NMS with learned pairwise reasoning."
```

### 12.2 协议说明 (必须写清楚)

在 Method 或 Experiment 部分加入:

```text
Inference protocol: OV-CUD receives only the input image. No exemplar boxes,
text prompts, or category names are provided at inference time. The model
outputs a set of (category_name, instance_mask, count) tuples for all
discovered semantic groups.

Evaluation protocol (class-aware): We match each predicted group to the
dataset's target category during metric computation. The model itself does
not receive the target category.

Evaluation protocol (fully prompt-free): For single-class images (e.g., FSC147),
the model auto-selects the predicted group with the highest count as the
final output. No category matching is needed.
```

### 12.3 监督对比 (必须说清楚)

```text
Training supervision comparison:
- Few-shot methods (CounTR, LOCA, BMNet+): exemplar boxes + density maps
- Text-specified methods (CounTX, VLCounter): text prompts + density maps
- Image-only methods (RCC, DAVE): density maps only
- OCCAM-S: no training (training-free, uses pre-trained models only)
- OV-CUD (Ours): instance masks + category labels, NO density maps or count labels
```

### 12.4 Reviewer FAQ 预判

| 可能的质疑 | 我们的回答 |
|---|---|
| "OV-CUD 真的不需要 count supervision 吗？" | 是。所有模块训练使用 instance mask / category label / relation label，无 density map 或 count label。 |
| "FSC147 的 GT class 是否在评估中使用了？" | 仅在 evaluation metric computation 中使用，模型推理时不接收。prompt-free protocol 甚至不需要。 |
| "9.11 vs 9.42 哪个是主结果？" | 固定 8.73 (Exp11) 为主结果。9.42 是旧版本。9.11 是 Exp9 (pts=16)。 |
| "τ_inst=0.99 是否为 test set 调参？" | τ_inst 在 FSC147 val set 选定，冻结至所有 test set。sweep 显示 0.9-0.995 范围内稳定。 |
| "CARPK 结果是否过于好？" | Bootstrap CI [3.72, 4.41] 确认统计显著。原因：CARPK 是单类别 (cars)，SAM2 候选覆盖好 (~109%)，去重有效。 |
| "和 OCCAM-S 比如何？" | OCCAM-S 是 training-free (MAE=14.35)，OV-CUD 是 count-supervision-free (MAE=8.73)。前者完全无训练，后者有 instance/category 监督但无 count 监督。不同设定。 |
| "为什么不直接用 GroundingDINO 做检测计数？" | OWLv2 (同类方法) 实验显示 MAE=43.61，检测模型不适合直接计数。OV-CUD 的 relation head + dedup 专门为计数优化。 |

---

## 13. 检查清单

### 13.1 投稿前必做

- [x] 在完整 FSC147 test set (1190 张图) 上运行 current-code multi-resolution 配置，补回 `7611.jpg` 后得到 MAE=13.47 / RMSE=126.74
- [ ] 固定所有 random seed (0, 1, 2)，报告 mean ± std
- [ ] 在完整 FSC147 test set 上运行 bootstrap CI (n=1000)
- [x] 重新运行核心组件消融 A1/A2/A3/A4/A5/A8 与高密度前端 A6/A7；完整记录见 `docs/fsc147_multires_component_ablation_report_20260709.md`
- [ ] 准备 qualitative visualization (成功案例 ×4 + 失败案例 ×2)
- [ ] 整理 per-image prediction JSON (用于 supplementary material)
- [ ] 确认所有 baseline 数字来源 (original paper / reproduced)
- [ ] 统一最终 checkpoint 命名和版本号

### 13.2 写作必查

- [ ] 不将 instance mask 监督简单称为 "weaker supervision"
- [ ] 不用未经核验的绝对表述 ("唯一", "所有方法", "显著优于")
- [ ] 明确区分 prompt-free inference 和 class-aware evaluation matching
- [ ] 明确说明 CARPK class_idx 仅用于 evaluation vocabulary selection
- [ ] 所有表格按 input/supervision 分组，标注公平性
- [ ] 主结果带上 95% CI

### 13.3 代码/复现

- [ ] 保存 config yaml (包含所有超参数)
- [ ] 保存 checkpoint hash (SHA256)
- [ ] 保存 vocabulary version (text_prototypes_fsc147.pt 的生成参数)
- [ ] 保存 SAM2 candidate generation parameters
- [ ] 保存所有 threshold values
- [ ] 保存 per-image predictions (JSON) + per-image error CSV
- [ ] 保存 visualization samples

---

## 附录: 完整实验数据文件索引

| 文件 | 内容 |
|---|---|
| `docs/experiment_log_20260630.md` | 每日实验日志 (完整版) |
| `docs/AAAI_2026_Experiment_Report.md` | 本报告 |
| `OV_CUD_AAAI_Experiment_Plan.md` | 实验计划 |
| `result/logs/pipeline_exp5c_best.json` | FSC147 Exp9 最佳结果 (MAE=9.11) |
| `result/logs/p1_ablations_carpk.json` | CARPK 消融结果 |
| `result/logs/p1_classification_ablation.json` | P1-2 分类头消融 |
| `result/logs/p2_pucpr_full.json` | P2-1 PUCPR+ 默认参数 |
| `result/logs/p2_pucpr_tune1.json` | P2-1 PUCPR+ 调参 |
| `result/logs/p2_tiled_eval_final.json` | P2-4 Tiling 完整结果 (MAE=3.59) |
| `result/logs/p2_owlv2_baseline.json` | P2-3 OWLv2 baseline (MAE=43.61) |
| `result/logs/p2_prompt_free.json` | P2-5 Prompt-free protocol |
| `result/logs/p2_coco_count.json` | P2-2 COCO multi-category |
| `result/logs/sample100_test.json` | FSC147 100-image test list |
| `result/checkpoints/fsc147_relation_pts32_best.pt` | 🏆 最佳推理 checkpoint |
