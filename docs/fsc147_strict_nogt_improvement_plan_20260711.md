# FSC-147 Strict No-GT 主方法分阶段改进计划

**制定日期**：2026-07-11

**状态**：仅完成计划设计；尚未生成新 cache、尚未训练、尚未运行任何新评测

**当前基线**：FSC-147 official test full 1,190，validation 预选 relation seed 73，MAE=26.4992 / RMSE=129.6856

**目标协议**：prompt-free、density-map-free、FSC point/dot-supervised、official-train-only、COCO relation pretraining-free、严格 image-only inference

## 1. 计划目的

当前 strict no-GT 主结果已经消除了 test `valid`、GT count 路由、训练 split overlap 和 test 类名访问，但性能从历史 GT-assisted 的 12.67 回落到 26.50。现有诊断表明主要问题不是 CP，而是：

1. category/filter/relation 在 full-image pts32 candidates 上训练，主推理却使用 2x2 tiled candidates；
2. train-89 closed-set category confidence 被同时当成 objectness gate，无法可靠处理 FSC held-out classes；
3. 稀疏图系统性过计数，高密度图系统性漏计，一个全局阈值无法兼顾；
4. 当前 learned relation 在 validation 上未超过固定 IoU-NMS；
5. fast-zero rescue 只覆盖 `7611` 类型，无法捕获 `1123`、`6860` 等非零但 proposal 严重不足的图像。

本计划按可归因、低泄露风险、由小到大的顺序推进。每个阶段只改变一个主要因素，未通过阶段验收不得进入下一阶段，也不得读取 FSC test 选择配置。

## 2. 不可违反的数据边界

### 2.1 Split 用途

| 数据 | 允许用途 | 禁止用途 |
|---|---|---|
| Official train 3,659 | 候选标注、模型训练、内部 fit/dev/audit、密度与 tiling policy 学习 | 不得混入 val/test 文件 |
| Official validation 1,286 | 阶段候选冻结后的外部验证、最终超参数与 route 选择 | 不得反向加入训练；不得逐样本人工修正规则 |
| Official test 1,190 | 所有阶段结束、配置和代码哈希冻结后的最终一次评测 | 不得调阈值、选 seed、选 tile、设计 trigger 或决定模型结构 |
| OmniCount/MCAC/CARPK/PUCPR+ | 最终跨数据集与多类别复核 | 不得反向优化 FSC 特定参数后再称 zero-shot |

FSC test 已在历史开发中被多次查看，因此后续只能保证“计算路径 no-GT”，不能重新宣称 untouched blind test。投稿前必须再选择一个未触碰的数据集或预留新 holdout 复核主要结论。

### 2.2 Official-train 内部划分

在第一项训练实验前固定并提交一个不可变 manifest：

- `train-fit`：80%，只用于梯度训练；
- `train-dev`：10%，只用于 early stopping 和局部阈值选择；
- `train-audit`：10%，阶段内锁定，只在候选方案确定后评估一次；
- split seed 固定为 `20260711`；
- 按 official-train class 做分层，保证同一类别的图像不会因样本数量差异集中到单一 partition；
- manifest 记录图像列表、SHA-256、类别分布和 GT count 分桶分布。

同一阶段不得根据 `train-audit` 结果继续改结构；若失败，必须新建下一实验 ID 并重新走 dev/audit 流程。

### 2.3 物理隔离要求

1. 训练脚本只能接收 physically isolated official-train dot/class shard，不能解析包含 val/test 的完整 annotation JSON。
2. Validation/test inference cache 继续严格限定为：

   `schema, img_id, file_name, z, bbox, height, width, source_cache`。

3. Train labeled cache 可含 dots 派生标签，但目录、schema 和加载器必须与 inference cache 分离。
4. Validation/test GT 只在该 split 的全部预测生成并冻结后进入 metric join。
5. 每个 checkpoint 必须保存 split manifest、上游资产哈希、训练 seed、代码哈希和 `test_images_loaded=0`。
6. 任何 test cache 文件是否存在、test filename 特例或历史 test GT bin 都不得参与路由。
7. 词表只能来自 official train 或预先声明的外部通用词表；不得引入 FSC val/test 类名列表。

## 3. 统一评估与报告规范

每个阶段统一报告：

- MAE / RMSE / bias；
- GT count bins：0-10、11-20、21-50、51-100、100+；
- raw/filter/objectness/joint/dedup 后的 candidate count 与保留率；
- proposal-capacity diagnostic、singleton recall、duplicate multiplicity；
- 三个训练 seeds 的 mean±sample std；
- 与固定 IoU-NMS 的 paired per-image bootstrap 95% CI；
- 参数量、cache 成本、训练时间和推理时间；
- no-GT 审计：训练/验证/测试图像读取数、禁止字段扫描、资产 SHA-256。

阶段验收默认优先看 `train-audit` 和 official validation MAE，同时要求：

1. MAE 改善至少 1.0，或 paired bootstrap 95% CI 不跨 0；
2. 不能以 100+ 显著恶化换取低密度改善，反之亦然；
3. 三 seeds 方向一致；
4. 不能增加新的 test-specific trigger；
5. 若复杂模块未稳定超过简单 baseline，优先保留简单 baseline。

## 4. Phase 0：协议与诊断基础设施

### 4.1 目标

建立后续所有训练共用的数据与审计基础，避免每个实验重新定义口径。

### 4.2 待办

1. 生成并提交 official-train `fit/dev/audit` manifest。
2. 导出 physically isolated train dots + image-level class shard。
3. 新增 train-only tiled labeled cache schema，与 inference-v1 schema 分离。
4. 在候选生成时保留 train-only 诊断标签：
   - 每个 candidate 覆盖的完整 dot ID set；
   - `dot_count`；
   - background / singleton / multi-dot；
   - tile ID、是否位于 tile border、SAM image-derived quality scores；
   - mask RLE 仅保存在 train/eval diagnostic cache，不进入 inference cache。
5. 新增统一 stage evaluator，能够在不把 GT 传入预测函数的前提下报告各阶段统计。
6. 冻结当前 strict baseline 的代码、配置、模型和 validation 输出作为 `B0`。

### 4.3 验收

- train cache 文件集合与 official train manifest 精确一致；
- val/test 图像读取数均为 0；
- 从 train labeled cache 导出的 safe cache 与直接 image-only 生成结果在 `z/bbox` 上逐 tensor 一致；
- 泄露单元测试能拒绝 `valid/points/gt_count/matched_*` 进入 inference loader。

## 5. Phase 1：前端匹配的 Candidate Filter

这是计划中的第一项实际实验。目的仅是验证“full-image 训练、tiled 推理”的分布错位，不同时更换 label 定义、category head 或 relation head。

### 5.1 数据生成

- 只处理 official train；
- SAM2 配方与 strict 主推理完全一致：pts32、2x2、overlap=0.25、bbox merge；
- 使用与 `preprocess_fsc147_tiled_nogt.py` 相同的候选过滤、两阶段去重、three-view DINOv2 encoding；
- 候选和特征先由图像生成，之后才从 isolated train-dot shard加入标签；
- 不读取 official val/test annotation，不生成 test 新 cache。

### 5.2 严格对照矩阵

| ID | Filter 训练候选 | Label | 其余组件 | 目的 |
|---|---|---|---|---|
| F0 | full-image pts32 | any-dot binary | 当前 train-89 category + scratch relation | 当前 strict baseline |
| F1 | **2x2 tiled pts32** | **any-dot binary** | 与 F0 完全相同 | 只测 frontend matching |
| F2 | full + tiled 混合 | any-dot binary | 与 F0 完全相同 | 检查混合训练是否兼顾 fast/tiled |

F1 是第一优先实验；只有 F1 完成后才决定是否运行 F2。

### 5.3 固定训练配方

- CandidateFilter 结构保持 1152+7 -> 256 -> 128 -> 1；
- BCE loss、batch size 2,048、AdamW `lr=5e-4`、weight decay `1e-4`；
- seeds：17、42、73；
- early stopping 只看 train-dev BCE/F1；
- 第一张比较表固定使用旧 inference threshold 0.3，不重新调参；
- 第二张表允许在 train-dev 上从预注册集合选择 threshold：

  `{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7}`。

- category threshold 固定 0.4、`tau_inst=0.99`、route=`always_tiled`、relation seed 固定为当前 validation primary 73；
- official validation 只在 F1 checkpoint 和 threshold 冻结后运行一次。

### 5.4 必报诊断

- train-full、train-tiled、val-tiled 的 filter score quantiles；
- threshold 下 P/R/F1、singleton recall、multi-dot pass rate；
- joint gate retention；
- full counting MAE/RMSE/bias 和 per-bin；
- F1-F0 paired per-image bootstrap；
- 不读取 FSC test。

### 5.5 Phase 1 决策规则

| 结果 | 决策 |
|---|---|
| F1 在 train-audit 与 val 都稳定改善 | F1 成为后续 objectness 基线 |
| F1 只改善 val、不改善 train-audit | 判定存在 val 适配风险，停止升级 |
| F1 改善 filter P/R，但 counting 不改善 | 问题主要在 label 定义或下游 gate，进入 Phase 2，但不替换主模型 |
| F1 三 seeds 方向不一致 | 增加正负采样与校准研究，不进入下一阶段 |
| F1 显著伤害 100+ | 不接受，即使 overall MAE 下降 |

## 6. Phase 2：Objectness 与 Semantics 解耦

### 6.1 动机

当前 `max(train-89 category probability)` 同时承担 objectness gate 和语义分组。由于 FSC train/test 类别互斥，该分数不是可靠的开放集 objectness。

### 6.2 变体

| ID | 质量标签 | 推理 gate | 语义用途 |
|---|---|---|---|
| O0 | any-dot binary | filter × category confidence | Phase 1 最优基线 |
| O1 | background / any-dot | objectness only | category 只用于 grouping |
| O2 | background / singleton / multi-dot | singleton-quality | category 只用于 grouping |
| O3 | objectness + singleton count 多任务 | calibrated quality score | category 只用于 grouping |

不得直接在 test 上关闭 category threshold。所有 objectness threshold、loss weight 和 class balance 只在 train-dev 选择。

### 6.3 验收

- O2/O3 必须降低低密度 false positives，同时不降低 100+ singleton recall；
- official validation 至少优于 F1 1.0 MAE；
- 输出需证明 improvement 来自 objectness，而不是引入 test vocabulary。

## 7. Phase 3：Relation 与 Dedup 重做

### 7.1 必须先回答的问题

当前 fixed IoU-NMS(0.3) 在 validation 的诊断 MAE 为 29.33，而 learned relation 为 30.00。Relation 只有稳定超过 NMS 才能保留为 accuracy contribution。

### 7.2 改进项

1. 使用 tiled train duplicates 训练，不再使用 full-image-only pair distribution。
2. same-instance 正标签改为两个候选的 covered-dot sets 有交集，而不是只比较第一个 dot ID。
3. 对 multi-dot candidates 单独标注，不把其第一个 dot 当成完整 instance identity。
4. 去掉或重标定受 train-89 OOD 影响的 `p_i·p_j` pair feature。
5. 用 spatial kNN / overlap graph 覆盖全部候选，取消全局 top-200 截断。
6. 显式修复 `adaptive_tau(base=0.99, max=0.95)` 的语义冲突，并在 train-dev 预注册比较：
   - fixed 0.99；
   - density-conditioned tau；
   - IoU-NMS 0.3；
   - learned relation。

### 7.3 变体与停止条件

| ID | Dedup | 条件 |
|---|---|---|
| R0 | IoU-NMS 0.3 | 简单 baseline |
| R1 | 当前 scratch relation | strict baseline |
| R2 | tiled shared-dot-set relation | 核心候选 |
| R3 | R2 + full local graph | 高密度扩展 |

若 R2/R3 在三 seeds 和 paired CI 上不能稳定超过 R0，则主方法采用 R0，relation head 降为未采用探索，不再保留贡献 claim。

## 8. Phase 4：Image-Conditioned Density Router

### 8.1 目标

解决低密度过计数和高密度漏计方向相反的问题，不再使用一个全局 gate。

### 8.2 允许的 image-only 输入

- raw fast/tiled candidate count；
- candidate bbox area、尺度、边界比例分布；
- filter/objectness score quantiles；
- tile 间 count consistency 和 overlap duplicate ratio；
- 图像频率、纹理、熵；
- DINO global image embedding；
- SAM image-derived stability/predicted-IoU statistics。

不得使用 GT count、GT bin、cache membership 或 filename 特例作为推理输入。

### 8.3 两种监督口径

| Router | 训练目标 | 论文表述 |
|---|---|---|
| D1 | train image-only proxy / self-supervised clustering | 不新增 count supervision |
| D2 | official-train dot count 派生 density bin | point/count-supervised，density-map-free |

D1、D2 必须分表报告，不能混用监督 claim。若 D2 明显更好，论文主设定必须诚实更新为 count-informed point supervision。

### 8.4 Router 输出

- sparse：更强 objectness threshold + 更积极 NMS/dedup；
- medium：2x2 标准配置；
- dense：降低 gate、减弱 dedup；
- extreme：切换 4x4/6x6 或 Phase 5 分支。

所有 route threshold 只在 train-dev 选择，train-audit 锁定复核，官方 val 最后一次确认。

## 9. Phase 5：极端密度专用分支

### 9.1 动机

`1123`、`6860` 等图像 fast candidates 非零，因此当前 fast-zero trigger 无法识别。只增加 4x4 也未必能覆盖 GT 2,000-3,700 的场景。

### 9.2 前端矩阵

| ID | Frontend | 使用范围 |
|---|---|---|
| E0 | 2x2 pts32 | 当前标准 |
| E1 | 4x4 pts32 | dense/extreme |
| E2 | 6x6 pts32 | extreme |
| E3 | 4x4/6x6 + upscale | 小目标诊断 |
| E4 | point-supervised patch count / point detector | SAM proposal capacity不足时 |

Tile recipe 必须在全部 high-density official-train fit/dev 样本上选择，不能再用单个 fast-zero train image，也不能根据 `7611` 结果决定。

E4 不使用 density map，但使用 dot 派生 patch count 或 point supervision；若采用，论文不得声称 count-supervision-free。

## 10. Phase 6：真正的 Prompt-Free 多类别模块

FSC-147 每图单类别且 train/test 类别互斥，不能单独证明多类别命名能力。本阶段与 FSC total count 改进分离。

1. Objectness 与语义分组完全解耦；
2. 用 DINO/CLIP visual embedding 做 open-set grouping，不以 train-89 top-1 bucket 作为硬边界；
3. 类别命名作为 grouping 后的独立步骤；
4. 词表只能使用预声明外部通用词表，或输出 anonymous groups；
5. OmniCount 报 total + per-class，MCAC 使用 Hungarian spatial matching；
6. 与 ABC123、OCCAM、prompted OmniCount methods 分协议比较。

## 11. Phase 7：严格组件消融与跨数据集复核

当 Phase 1-6 的最终组件冻结后，重新定义 strict leave-one-out：

| ID | 从完整模型移除 | 替代 |
|---|---|---|
| A1 | tiled-trained objectness | full-image filter |
| A2 | singleton-quality target | any-dot binary |
| A3 | density router | fixed global policy |
| A4 | high-density frontend | 2x2 only |
| A5 | learned relation | val-selected IoU-NMS |
| A6 | open-set grouping | one global bucket / argmax bucket |

所有行使用同一 safe cache、同一 split manifest、同一 seed policy；每行只改变一个组件。先跑 train-audit/official val，主表方案冻结后才考虑最终 test。

跨数据集必须使用同一新 checkpoint 重跑 CARPK、PUCPR+、OmniCount、MCAC，旧 FSC/COCO checkpoint 数字继续标记为 legacy。

## 12. Phase 8：最终冻结与一次性 Test

只有满足以下条件才允许进入：

1. 模型结构、所有 thresholds、route、tile recipes、seed selection 已由 train-dev/audit + official val 冻结；
2. 所有 checkpoints、代码、词表、manifest、cache recipe 已记录 SHA-256；
3. test plan 由 image-only cache 生成，记录 `test_annotations_read=false`；
4. inference loader 对 forbidden GT fields fail-closed；
5. 预测进程先为全部 1,190 张生成结果，之后 metric 进程才读取 isolated test targets；
6. test 运行后不允许调参再重跑；若失败，只能如实报告或转到新 benchmark 开发。

最终报告包含 primary seed、三 seed mean±std、bootstrap CI、per-bin、极端失败和完整逐图 JSON。

## 13. 推荐执行顺序与预计成本

| 顺序 | 阶段 | 主要成本 | 是否需要训练 | 是否允许读 FSC test GT |
|---:|---|---|---|---|
| 1 | Phase 0 数据隔离与 train tiled cache | 高：SAM2/DINO 全 train 预处理 | 否 | 否 |
| 2 | Phase 1 tiled candidate filter | 低 | 是 | 否 |
| 3 | Phase 2 singleton objectness | 低-中 | 是 | 否 |
| 4 | Phase 3 relation/NMS | 中 | 是 | 否 |
| 5 | Phase 4 density router | 中 | 是 | 否 |
| 6 | Phase 5 extreme frontend | 高 | 可选 | 否 |
| 7 | Phase 6 multi-category | 中-高 | 可选 | 否 |
| 8 | Phase 7 strict LOO/cross-dataset | 高 | 视变体而定 | 否 |
| 9 | Phase 8 final test | 中 | 否 | 仅预测冻结后计算指标 |

## 14. 第一项实验启动前检查清单

- [ ] 本文计划已提交，实验矩阵和停止条件不再根据 test 修改。
- [ ] Official-train fit/dev/audit manifest 已生成并审核。
- [ ] Isolated train dot/class shard 已生成，manifest 显示 val/test loaded=0。
- [ ] Train tiled labeled cache 脚本与 no-GT inference generator 做候选 tensor 一致性测试。
- [ ] F0/F1 的模型结构、loss、seeds、threshold grid 已冻结。
- [ ] Category/relation/route 在 Phase 1 中保持不变。
- [ ] Official validation 只在 F1 完成并冻结后运行。
- [ ] FSC test cache、annotation 和历史逐图结果不进入训练/选择进程。
- [ ] 输出目录、JSON 命名和 SHA-256 manifest 已预先确定。

## 15. 预定产出命名

计划后续使用以下命名，便于审计：

- `result/configs/fsc147_train_fit_dev_audit_20260711.json`
- `result/configs/fsc147_train_points_isolated.json`
- `result/checkpoints/fsc147_candidate_filter_tiled_seed{17,42,73}.pt`
- `result/logs/fsc147_phase1_filter_train_audit.json`
- `result/logs/fsc147_phase1_filter_val_frozen.json`
- `docs/fsc147_phase1_tiled_filter_report_20260711.md`

在本文状态从“计划”改为“执行中”之前，不得创建上述实验结果文件。
