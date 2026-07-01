# OV-CUD AAAI 投稿实验补充计划

**版本**: v1.0  
**日期**: 2026-07-01  
**目标**: 围绕当前 OV-CUD 在 FSC147 与 CARPK 上的结果，规划 AAAI 投稿前必须补充的消融实验、对比实验、诊断实验与论文呈现方案。

---

## 0. 当前结果与需要先统一的数字

### 0.1 当前核心结果

当前已有结果可以形成论文主线：

| 数据集 | 设置 | 推理输入 | 训练监督 | MAE | RMSE | 备注 |
|---|---|---:|---:|---:|---:|---|
| FSC147 Test | Prompt-free / class-aware | 仅图像 | Instance mask + category / relation supervision | 9.11 | 32.87 | Exp9: relation head fine-tune + `tau_inst=0.97` |
| CARPK Test | zero-shot transfer | 仅图像 | FSC147 训练模型，无 CARPK 微调 | 6.50 | 9.13 | class_idx=car / cars |

### 0.2 投稿前必须统一的记录

你当前 benchmark 文档里存在几处数字不一致，投稿前必须固定最终版本：

| 项目 | 当前出现的数值 | 需要处理 |
|---|---|---|
| FSC147 MAE | 9.11 / 9.42 | 固定最终 checkpoint、固定阈值、重新导出一版唯一结果 |
| FSC147 RMSE | 32.87 / 33.18 | 同上 |
| OV-CUD latest | 9.11 / 32.87 | 作为主表结果或 appendix 结果二选一 |
| CARPK class name | car / cars / class_idx=29 | 统一 vocabulary entry 与 evaluation protocol |

**建议**: 论文主文只保留一个主结果，例如：

```text
OV-CUD achieves 9.11 MAE / 32.87 RMSE on FSC147 Test and 6.50 MAE / 9.13 RMSE on CARPK zero-shot transfer.
```

如果 9.42 是旧版本结果，放入 ablation 或直接删除。

---

## 1. 论文主张与实验支撑关系

AAAI 投稿时不要只堆结果，要让每个实验支撑一个明确 claim。

| 论文主张 | 必须支撑的实验 |
|---|---|
| OV-CUD 是 prompt-free 的 class-aware counting 方法 | 输入协议说明 + prompt-free protocol comparison |
| OV-CUD 不使用 count label / density map 训练 | supervision table + training label audit |
| OV-CUD 能输出类别名、实例位置和 count | qualitative visualization + JSON output examples |
| OV-CUD 在 FSC147 上优于 prompt-free / reference-less 方法 | FSC147 主对比表 |
| OV-CUD 在 CARPK 上 zero-shot transfer 强 | CARPK cross-dataset comparison |
| 关系头、instance dedup 和 part-whole representative selection 是关键 | component ablations |
| 性能不是靠 test set 调参得到的 | validation-only threshold selection + frozen test protocol |
| 高 count 区间主要受 proposal recall 限制 | candidate recall / upper-bound diagnostics |

---

## 2. 最重要的协议澄清实验

这是 AAAI 审稿最可能质疑的地方，必须优先补。

### 2.1 Prompt-free evaluation protocol

FSC147 的标准任务通常默认有目标类别或 exemplar，而 OV-CUD 是输出所有语义组。必须明确：**模型推理时不输入 exemplar、text prompt 或 target class；evaluation 阶段如何把输出 group 和 GT count 对齐。**

建议报告两个协议：

| 协议 | 是否使用 GT 类别进行 scoring | 用途 |
|---|---:|---|
| **Class-aware protocol** | 是，仅用于 evaluation matching | 主结果。模型输出所有类别，评估时取与 GT target category 对齐的 predicted group |
| **Fully prompt-free single-count protocol** | 否 | 更严格补充结果。模型必须自动选择一个最终 count，例如最高 group quality / 最大 repeated group / 最大 countable group |

### 2.2 必做实验

| 实验 ID | 实验 | 目的 | 期望结论 |
|---|---|---|---|
| P1 | Class-aware scoring: 用 GT category name 只在 evaluation 阶段匹配输出 group | 证明 class-aware 输出可评估 | 主结果稳定 |
| P2 | Fully prompt-free scoring: 不用 GT category，自动选择最高质量 group | 证明真正无提示能力 | 可能低于 P1，但仍优于 reference-less baseline 更有说服力 |
| P3 | Oracle group selection: 从所有 predicted groups 中选 count error 最小的 group | 衡量 group discovery 上限 | 如果 P3 远好于 P1/P2，说明 label/group selection 是瓶颈 |
| P4 | Category-hit-conditioned MAE | 看分类命中对 count 的影响 | 分析 class-aware matching 是否主导误差 |

### 2.3 写作建议

不要写成“模型知道 FSC147 要数哪个类”。应该写成：

```text
At inference time, OV-CUD predicts all countable semantic groups without any prompt. For class-aware evaluation on FSC147, we match the predicted group label to the dataset target category only during metric computation.
```

---

## 3. 主对比实验计划

### 3.1 FSC147 Test 主表

建议拆成四个对比块，不要混在一张表里直接排名。

#### Block A: Few-shot methods

这些方法需要 1-3 个 exemplar boxes，输入强于 OV-CUD。

| 方法类型 | 推理输入 | 训练监督 | 是否直接公平 |
|---|---|---|---:|
| LOCA / CounTR / BMNet+ / GeCo / SAViT / SMFENet | 3 exemplars | density map / count supervision | 否，作为强输入参考 |

#### Block B: Text-specified / zero-shot methods

这些方法需要 text prompt 或类别描述。

| 方法类型 | 推理输入 | 训练监督 | 是否直接公平 |
|---|---|---|---:|
| CounTX / T2ICount / SAVE / VLCounter | text prompt / category description | density map / count supervision | 否，作为 prompt-based 参考 |

#### Block C: Reference-less / prompt-free methods

这是最关键的同设定对比。

| 方法类型 | 推理输入 | 训练监督 | 是否直接公平 |
|---|---|---|---:|
| RCC / CounTR zero-shot / LOCA zero-shot / DAVE / MAFEA / GCA-SUN | 仅图像 | density map / count supervision | 接近公平，但它们使用 count supervision |
| OCCAM-S | 仅图像 | 无训练 | 接近公平，但不输出类别名 |
| OV-CUD | 仅图像 | instance mask + category / relation labels | 主方法 |

#### Block D: Detection / foundation-model baselines

建议新增一组你自己能跑的 baseline，增强说服力：

| Baseline | 输入 | 实现方式 | 目的 |
|---|---|---|---|
| SAM2 + DINOv2 clustering | 仅图像 | 候选 feature 聚类 + component count | 证明不是 SAM2 本身带来的效果 |
| SAM2 + classification head + NMS | 仅图像 | 不用 relation head，只按类别聚合和 NMS | 证明 relation/dedup 必要 |
| GroundingDINO / OWLv2 / YOLO-World + category list | text/category list | 对 FSC147 target class 做检测计数 | 强 prompt 检测式 baseline，用于说明 prompt-free 的差异 |
| CLIP/SigLIP crop classifier + SAM2 | 类别文本或全 vocab | SAM 候选 + open-vocab crop classification + NMS | 对比你训练分类头的价值 |

**优先级**: SAM2 + classification head + NMS 是必须做的内部强 baseline。

---

## 4. 核心消融实验

### 4.1 总体组件消融

这是主文最重要的 ablation table。

| 实验 ID | Variant | Category Head | A_sem | A_inst | A_part | Clustering | Dedup | Rep. selection | 目的 |
|---|---|---:|---:|---:|---:|---|---|---|---|
| A0 | Raw SAM count | ❌ | ❌ | ❌ | ❌ | 无 | 无 | 无 | 下界，证明 raw proposals 不能直接计数 |
| A1 | Category only | ✅ | ❌ | ❌ | ❌ | 按类别分桶 | 无 | 最高分类置信 | 证明分类头单独不够 |
| A2 | Category + NMS | ✅ | ❌ | ❌ | ❌ | 按类别分桶 | box/mask NMS | NMS kept | 对比传统去重 |
| A3 | + semantic relation | ✅ | ✅ | ❌ | ❌ | A_group clustering | 无 | 最高分类置信 | 证明 A_sem 对 semantic group 有贡献 |
| A4 | + instance relation | ✅ | ✅ | ✅ | ❌ | A_group clustering | A_inst components | 组件内最高分类置信 | 证明 same-instance dedup 是关键 |
| A5 | + part-whole relation | ✅ | ✅ | ✅ | ✅ | A_group clustering | A_inst components | A_part rep score | 完整模型 |
| A6 | Full minus refinement | ✅ | ✅ | ✅ | ✅ | A_group clustering | A_inst components | A_part rep score | 证明 refinement 的贡献 |
| A7 | Full with oracle class | Oracle | ✅ | ✅ | ✅ | A_group clustering | A_inst components | A_part rep score | 分类上限 |
| A8 | Full with oracle dedup | ✅ | ✅ | Oracle | ✅ | A_group clustering | Oracle components | A_part rep score | dedup 上限 |

建议指标：

```text
FSC147 MAE / RMSE / NAE
CARPK MAE / RMSE
Class hit rate
Duplicate rate
Miss rate
Part-as-instance error rate
```

### 4.2 关系头消融

| 实验 ID | Variant | 目的 |
|---|---|---|
| R1 | Relation head 不微调，只用 instance segmentation pretrain | 看 FSC147 adaptation 是否必要 |
| R2 | 只训练 A_sem | 看 semantic relation 是否独立有效 |
| R3 | 只训练 A_inst | 看 dedup 是否独立有效 |
| R4 | 只训练 A_part | 看代表选择是否独立有效 |
| R5 | A_sem + A_inst | 验证 part-whole 之前的主要增益 |
| R6 | A_sem + A_inst + A_part | 完整关系头 |
| R7 | 共享 backbone vs 三个独立 relation heads | 看模型复杂度和性能取舍 |

### 4.3 `tau_inst` 和阈值敏感性

你当前最优是 `tau_inst=0.97`，必须证明不是 test set 调参。

| 实验 ID | Sweep | 范围 | 数据集 | 目的 |
|---|---|---|---|---|
| T1 | `tau_inst` | 0.90, 0.93, 0.95, 0.97, 0.98, 0.99 | FSC147 val/test | 证明阈值稳定 |
| T2 | `tau_affinity` | 0.20-0.80 | FSC147 val/test | 语义聚类阈值敏感性 |
| T3 | class confidence threshold | 0.10-0.70 | FSC147 val/test | 分类过滤敏感性 |
| T4 | representative score weights | grid / random search on val | FSC147 val/test | 防止 hand-tuned overfit |
| T5 | group quality threshold | 0.10-0.80 | FSC147 val/test | 控制 precision/recall |

**要求**: 所有阈值必须在 validation set 上选定，然后冻结到 test set。

### 4.4 SAM2 candidate generation 消融

当前高密度场景瓶颈可能来自 candidate recall。需要系统验证。

| 实验 ID | Variant | 参数 | 指标 | 目的 |
|---|---|---|---|---|
| S1 | pts=16 | 较低候选密度 | MAE / runtime / candidate recall | 看低密度下限 |
| S2 | pts=32 | 当前主结果 | 同上 | 主设置 |
| S3 | pts=64 | 高候选密度 | 同上 | 看高密度是否改善 100+ |
| S4 | max_candidates=128 | 当前旧限制 | 同上 | 证明 100+ 受限 |
| S5 | max_candidates=256 | 中等候选上限 | 同上 | 推荐主设置候选 |
| S6 | max_candidates=512 | 高上限 | 同上 | 看性能上限和速度代价 |
| S7 | multi-scale tiling | local SAM + merge | 同上 | 针对密集小目标 |
| S8 | no candidate filtering | 不过滤明显噪声 | MAE / duplicate rate | 证明 filtering 必要 |
| S9 | aggressive filtering | 强过滤 | MAE / miss rate | 看 recall-precision trade-off |

建议额外报告：

```text
average #candidates / image
average #representatives / image
dot recall / box recall
runtime / image
GPU memory
```

### 4.5 分类头消融

| 实验 ID | Variant | 训练数据 | 目的 |
|---|---|---|---|
| C1 | LVIS-only classifier | LVIS | 检验不使用 FSC147 的基础泛化 |
| C2 | LVIS + FSC147 exemplar fine-tune | LVIS + FSC147 exemplars | 检验 FSC147 domain adaptation |
| C3 | LVIS + FSC147 dot-mined pseudo candidates | LVIS + pseudo candidates | 检验弱监督候选挖掘的贡献 |
| C4 | FSC147-only classifier | FSC147 | 看是否过拟合 FSC147，CARPK 是否下降 |
| C5 | LVIS replay vs no replay | LVIS replay ratio 0/10/25/50% | 防止 catastrophic forgetting |
| C6 | closed-set head vs text-prototype head | same features | 验证 open-vocabulary 设定价值 |
| C7 | top-1 class vs top-k category compatibility | k=1/3/5 | 看 `p_i dot p_j` 是否优于 hard label |

关键分析：

```text
classification hit rate vs final MAE
per-category classification accuracy
per-category count error
class confusion matrix
```

### 4.6 Clustering 消融

| 实验 ID | Variant | 目的 |
|---|---|---|
| G1 | category bucket + connected components | 当前主方法 |
| G2 | global first-neighbor without category bucket | 证明 category-aware 的必要性 |
| G3 | category bucket + first-neighbor | 对比 connected components 策略 |
| G4 | FINCH clustering | 对比 OCCAM-S 风格聚类 |
| G5 | spectral clustering / agglomerative clustering | 验证聚类算法是否关键 |
| G6 | `A_group = p_i dot p_j` only | 分类相容性 alone |
| G7 | `A_group = A_sem` only | relation alone |
| G8 | `A_group = (p_i dot p_j) * A_sem` | 当前完整 affinity |

### 4.7 Representative selection 消融

| 实验 ID | Variant | 目的 |
|---|---|---|
| D1 | 每个 component 选最大 mask | 简单 baseline |
| D2 | 每个 component 选最高 class confidence | 分类置信 baseline |
| D3 | 每个 component 选最高 SAM score | proposal baseline |
| D4 | class confidence + completeness | 几何完整性贡献 |
| D5 | class confidence + A_part incoming/outgoing | part-whole 贡献 |
| D6 | full RepScore | 主方法 |
| D7 | residual part filtering off | 证明残余 part 过滤必要 |

---

## 5. Oracle 诊断实验

这部分建议放 appendix，但对定位瓶颈非常重要。

### 5.1 FSC147 dot-based upper bound

FSC147 没有实例 mask，但有 dot annotation，可以做弱 oracle。

| 实验 ID | Oracle | 实现 | 解释 |
|---|---|---|---|
| O1 | Candidate recall upper bound | 每个 GT dot 是否被至少一个 SAM candidate 覆盖 | proposal 理论上限 |
| O2 | Oracle class | 覆盖 target dot 的 candidate 强制赋 GT target class | 分类瓶颈上限 |
| O3 | Oracle instance component | 包含同一个 dot 的 candidates 归为同一 instance | A_inst / dedup 上限 |
| O4 | Oracle representative | 每个 dot 选择一个最合理 candidate | representative selection 上限 |
| O5 | Oracle group selection | 从 predicted groups 选择 count 最接近 GT 的 group | group discovery / label selection 上限 |

### 5.2 必须按 count bin 报告

| GT 区间 | O1 candidate recall | O2 oracle-class MAE | O3 oracle-dedup MAE | Full MAE | 结论 |
|---|---:|---:|---:|---:|---|
| 0-10 |  |  |  |  | 低 count 是否受分类影响 |
| 11-20 |  |  |  |  |  |
| 21-50 |  |  |  |  |  |
| 51-100 |  |  |  |  |  |
| 100+ |  |  |  |  | proposal 是否是主瓶颈 |

### 5.3 CARPK oracle 诊断

CARPK 有 bbox，可以做更强 oracle：

| 实验 ID | Oracle | 实现 |
|---|---|---|
| CO1 | proposal recall | SAM candidate 与 GT car bbox IoU / center coverage |
| CO2 | oracle category | car candidates 强制 class=car |
| CO3 | oracle dedup | candidate 匹配到同一 GT bbox 即同一 instance |
| CO4 | under-count source analysis | 漏检 car vs 被合并 car vs 被过滤 car |

---

## 6. Cross-dataset 与泛化实验

### 6.1 CARPK 结果需要补充的实验

当前 CARPK zero-shot MAE=6.50 很强，建议围绕它做完整故事。

| 实验 ID | 实验 | 目的 |
|---|---|---|
| X1 | FSC147-trained OV-CUD -> CARPK test, no fine-tune | 当前主结果，证明 zero-shot transfer |
| X2 | LVIS-only OV-CUD -> CARPK test | 看 CARPK 成绩是否依赖 FSC147 训练 |
| X3 | FSC147-trained classifier + relation head without FSC adaptation | 分离分类与关系贡献 |
| X4 | CARPK val fine-tune classification head only | 看少量域内调参能否逼近 few-shot SOTA |
| X5 | CARPK val fine-tune relation threshold only | 看阈值域适配收益 |
| X6 | CARPK train fine-tune full head | 上限，不作为主方法 |
| X7 | CARPK -> PUCPR+ zero-shot transfer | 如果时间允许，验证 car domain robustness |

### 6.2 额外数据集建议

| 数据集 | 优先级 | 原因 | 注意事项 |
|---|---:|---|---|
| CARPK | P0 | 已经有强结果，必须完善 | 单类、俯拍，适合 cross-dataset |
| PUCPR+ | P1 | car counting 互补视角 | 数据小，放 appendix 即可 |
| COCO-count subset | P1 | instance mask + category，适合 class-aware evaluation | 不是标准 counting benchmark，需要清楚说明自建 protocol |
| LVIS-count subset | P2 | 与训练词表一致，可做 diagnostic | 容易被质疑 train/test overlap，要谨慎 |
| ShanghaiTech Part A/B | P3 | 经典 crowd counting | 人群太密，instance counting 不一定适合 |

---

## 7. Error analysis 计划

### 7.1 基础误差分解

除了 MAE/RMSE，建议每张图记录：

```text
signed_error = pred_count - gt_count
absolute_error = abs(pred_count - gt_count)
relative_error = abs(pred_count - gt_count) / max(gt_count, 1)
```

按以下维度统计：

| 维度 | 分组方式 | 要回答的问题 |
|---|---|---|
| GT count | 0-10, 11-20, 21-50, 51-100, 100+ | 高密度是否主要 under-count |
| object size | small / medium / large | 小目标是否漏检 |
| category frequency | head / medium / tail | 长尾类别是否差 |
| candidate count | low / medium / high | SAM 候选数量是否限制 performance |
| class hit | hit / miss | 分类命中对最终 count 的影响 |
| group quality | high / low | 置信度是否可用于拒识 |

### 7.2 必做图表

| 图表 | 内容 | 位置 |
|---|---|---|
| Error vs GT count bin | MAE/RMSE/NAE by bin | 主文 |
| Signed error histogram | under-count / over-count 偏置 | 主文或 appendix |
| Candidate recall vs GT count | 解释高密度瓶颈 | 主文 |
| Class hit rate vs MAE | 分类瓶颈分析 | appendix |
| Runtime vs #candidates | 实用性分析 | appendix |
| Qualitative success cases | 输出类别名 + mask/box + count | 主文 |
| Qualitative failure cases | 漏检、合并、part-as-instance、错类 | 主文 |

---

## 8. 统计显著性与复现要求

### 8.1 随机种子

至少报告 3 个 seed：

```text
seed = 0, 1, 2
```

如果训练成本太高，至少对最后的分类头/关系头 fine-tune 做 3 seed，并报告：

```text
mean ± std for MAE/RMSE
```

### 8.2 Confidence interval

对 FSC147 Test 和 CARPK Test 做 bootstrap：

```text
bootstrap resampling images, 1000 samples
report 95% CI for MAE and RMSE
```

主表可以写：

```text
9.11 ± 0.xx MAE
```

### 8.3 Threshold selection discipline

投稿前必须写清楚：

```text
All thresholds are selected on the validation set and kept fixed on the test set.
```

并保留阈值表：

| Threshold | Value | Selected on | Used for |
|---|---:|---|---|
| `tau_inst` | 0.97 | FSC147 val | same-instance components |
| `tau_affinity` | TBD | FSC147 val | semantic grouping |
| `tau_cls` | TBD | FSC147 val | candidate filtering |
| `pts` | 32 | FSC147 val | SAM2 candidate density |

---

## 9. Baseline 数值核验计划

当前 benchmark 文档使用“原论文声称的精度，未复现”。这可以作为内部分析，但论文投稿时必须严谨。

### 9.1 必须核验的内容

| 项目 | 操作 |
|---|---|
| FSC147 split | 确认所有 baseline 使用相同 test split |
| Exemplar 数量 | 区分 1-shot / 3-shot / zero-shot / no-prompt |
| Training supervision | 区分 density map、dot/count label、instance mask、no training |
| Input modality | exemplar bbox、text prompt、category name、image only |
| Metric | MAE/RMSE 是否同一公式 |
| Result source | 官方 paper / official code / reproduced |

### 9.2 表格标注建议

主表中每个方法加这些列：

```text
Input: Image / Exemplar / Text / Category list
Count supervision: Yes / No
Instance supervision: Yes / No
Class-aware output: Yes / No
Reproduced: Official / Ours
```

### 9.3 谨慎表述

避免使用未经充分核验的绝对表述：

```text
当前唯一
显著优于所有公开方法
监督更弱
```

更稳的写法：

```text
Among image-only methods reported on FSC147, OV-CUD achieves the lowest MAE under our count-supervision-free setting.
```

关于监督强弱，建议写：

```text
OV-CUD is count-supervision-free and density-map-free, but uses instance/category supervision from segmentation datasets.
```

不要简单写“instance mask + category 比 density map 更弱”。严格来说，instance mask 在空间标注强度上通常强于 dot/density supervision。

---

## 10. 论文主表和附录表建议

### 10.1 主文 Table 1: FSC147 comparison

包含：

```text
Method / Year / Inference input / Count supervision / Class-aware output / MAE / RMSE
```

按 setting 分块：

```text
Few-shot
Text-specified
Image-only with count supervision
Training-free
Ours
```

### 10.2 主文 Table 2: CARPK zero-shot transfer

包含：

```text
Method / Training data / Inference input / Fine-tune on CARPK? / MAE / RMSE
```

重点突出：

```text
FSC147 -> CARPK, no fine-tune
```

### 10.3 主文 Table 3: Component ablation

建议只保留 6-8 行最核心 variant：

```text
Category only
+ A_sem
+ A_inst
+ A_part
+ refinement
Full
Oracle class
Oracle dedup
```

### 10.4 Appendix tables

| Appendix Table | 内容 |
|---|---|
| A1 | 完整 FSC147 baseline table |
| A2 | 完整 CARPK baseline table |
| A3 | threshold sensitivity |
| A4 | SAM2 candidate ablation |
| A5 | clustering ablation |
| A6 | representative selection ablation |
| A7 | per-category FSC147 results |
| A8 | runtime / memory |

---

## 11. 实验执行优先级

### P0: 必须完成，决定投稿可信度

| 优先级 | 实验 | 预计产出 |
|---:|---|---|
| P0-1 | 固定最终 checkpoint 和唯一数字 | 主结果可复现 |
| P0-2 | FSC147 protocol clarification: class-aware vs fully prompt-free | 防止 reviewer 质疑输入设定 |
| P0-3 | Component ablation: category / A_sem / A_inst / A_part | 支撑方法贡献 |
| P0-4 | SAM candidate density / max_candidates ablation | 解释高 count 区间瓶颈 |
| P0-5 | `tau_inst` 和主要阈值 sweep on val | 防止 test tuning 质疑 |
| P0-6 | CARPK zero-shot 结果复核 | 支撑泛化 claim |
| P0-7 | Qualitative visualization | 证明输出类别名和实例位置 |

### P1: 强烈建议完成，提升论文竞争力

| 优先级 | 实验 | 预计产出 |
|---:|---|---|
| P1-1 | Oracle diagnostics | 定位瓶颈，增强分析深度 |
| P1-2 | Classification head ablation | 说明 FSC147 adaptation 与 LVIS pretraining 的贡献 |
| P1-3 | Clustering ablation | 证明 category-aware matrix clustering 必要 |
| P1-4 | Representative selection ablation | 证明 part-whole 关系有效 |
| P1-5 | Runtime / memory | 说明可用性 |
| P1-6 | Bootstrap CI / 3 seeds | 提升统计可信度 |

### P2: 时间允许再做

| 优先级 | 实验 | 预计产出 |
|---:|---|---|
| P2-1 | PUCPR+ zero-shot | 额外 car counting 泛化 |
| P2-2 | COCO-count subset | class-aware multi-category counting |
| P2-3 | GroundingDINO / YOLO-World prompt baseline | 对比 foundation detection |
| P2-4 | Multi-scale tiling | 高密度优化版本 |
| P2-5 | Fully prompt-free single-count result | 最严格无提示 protocol |

---

## 12. 推荐实验矩阵

### 12.1 第一轮: 1-2 天内完成

| 实验 | FSC147 | CARPK | 输出 |
|---|---:|---:|---|
| Final Exp9 rerun | ✅ | ✅ | 固定主结果 |
| Component ablation A1-A5 | ✅ | ✅ | 主文 ablation |
| `tau_inst` sweep | ✅ val/test | ✅ test | 阈值曲线 |
| pts / max_candidates sweep | ✅ | ✅ | proposal 分析 |
| signed error by bin | ✅ | ✅ | 错误分析图 |

### 12.2 第二轮: 3-5 天内完成

| 实验 | FSC147 | CARPK | 输出 |
|---|---:|---:|---|
| Oracle diagnostics | ✅ | ✅ | appendix 分析 |
| Classification ablation C1-C5 | ✅ | ✅ | 说明训练策略 |
| Clustering ablation G1-G8 | ✅ | 可选 | 支撑 matrix clustering |
| Representative selection D1-D7 | ✅ | ✅ | 支撑 A_part |
| Runtime/memory | ✅ | ✅ | appendix |

### 12.3 第三轮: 投稿前增强

| 实验 | 数据集 | 输出 |
|---|---|---|
| Fully prompt-free single-count protocol | FSC147 | 更严格无提示结果 |
| PUCPR+ transfer | PUCPR+ | 额外泛化 |
| COCO-count subset | COCO val | multi-category class-aware result |
| 3-seed / bootstrap CI | FSC147 + CARPK | 统计可信度 |

---

## 13. 推荐最终叙事

论文叙事可以这样组织：

1. 现有 counting 方法通常依赖 exemplar、text prompt 或 count/density supervision。
2. 我们提出 OV-CUD：仅输入图像，自动生成候选、分类、建模候选关系、分组、去重并输出 class-aware counts。
3. 方法不训练 count regressor，也不使用 density map；计数来自 refined semantic group 中的代表实例数量。
4. 在 FSC147 上，OV-CUD 在 image-only / count-supervision-free 设定下达到强性能。
5. 在 CARPK 上，FSC147 训练模型无需 CARPK 微调即可达到 6.50 MAE，证明跨域泛化。
6. 消融显示：classification、semantic relation、same-instance dedup、part-whole representative selection 都不可或缺。
7. 诊断显示：高密度场景的主要瓶颈是 candidate recall 和 proposal density，而不是 count regression 缺失。

---

## 14. 投稿前检查清单

### 14.1 实验完整性

- [ ] 主结果唯一且可复现
- [ ] FSC147 protocol 说清楚
- [ ] CARPK zero-shot 设置说清楚
- [ ] 所有阈值在 validation set 选定
- [ ] 主对比表按 input/supervision 分组
- [ ] 至少一个 strong internal baseline: SAM2 + classification + NMS
- [ ] 组件消融完整
- [ ] candidate recall / high-count analysis 完整
- [ ] 失败案例可视化完整

### 14.2 写作风险控制

- [ ] 不把 instance mask 监督简单称为 weaker supervision
- [ ] 不用未经核验的“唯一”或“所有方法”绝对表述
- [ ] 明确区分 prompt-free inference 和 class-aware evaluation matching
- [ ] 明确说明 OV-CUD 输出所有 groups，而不是只输出一个 count
- [ ] 明确说明 CARPK class_idx 只用于 evaluation 或固定 vocabulary selection，不是人类 prompt

### 14.3 代码与复现

- [ ] 保存 config yaml
- [ ] 保存 checkpoint hash
- [ ] 保存 vocab version
- [ ] 保存 SAM2 candidate generation parameters
- [ ] 保存 threshold values
- [ ] 保存 per-image predictions
- [ ] 保存 per-image error CSV
- [ ] 保存 visualization samples

---

## 15. 建议优先做的 10 个实验

如果时间有限，只做下面 10 个：

1. **Final Exp9 rerun**: 固定 FSC147 9.11 / 32.87 与 CARPK 6.50 / 9.13。
2. **Protocol comparison**: class-aware scoring vs fully prompt-free group selection。
3. **SAM2 + classification + NMS baseline**。
4. **Component ablation**: category only, +A_sem, +A_inst, +A_part, full。
5. **`tau_inst` sweep**: 证明 0.97 不是 test tuning。
6. **candidate density sweep**: pts=16/32/64，max_candidates=128/256/512。
7. **oracle candidate recall**: FSC147 dot recall 与 CARPK bbox recall。
8. **classification ablation**: LVIS-only vs LVIS+FSC147 fine-tune vs pseudo-candidate。
9. **CARPK zero-shot复核 + under-count 分析**。
10. **qualitative success/failure visualization**。

---

## 16. 最终建议

当前结果已经足够形成强论文雏形，但 AAAI 投稿前最关键的不是再追求一个小数点提升，而是补齐以下三类证据：

1. **协议公平性证据**: prompt-free 到底如何评估，是否使用 GT 类别只在 scoring 阶段。
2. **方法必要性证据**: 分类头、关系头、A_inst、A_part、refinement 每个组件都要有消融。
3. **瓶颈解释证据**: 高密度场景为什么错，candidate recall 和 oracle upper bound 必须给出来。

只要这三类证据补齐，FSC147 + CARPK 的当前结果已经具备很强的投稿竞争力。
