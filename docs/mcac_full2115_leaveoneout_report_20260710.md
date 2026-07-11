# MCAC 2,115 张全量评测与 Leave-One-Out 消融报告

> **2026-07-11 状态更正**：本文 MCAC 推理虽不读取 MCAC GT，但使用历史 FSC/COCO checkpoint，不再能称为与当前 strict FSC-147 CP-free 主模型相同；数值保留为 legacy checkpoint 记录。

**日期**：2026-07-10

**数据集**：MCAC test full 2,115 images

**对应计划**：`docs/experiment_plan_ablation_mcac_20260709(1).md`
**FSC147 参考主结果**：MAE=12.67 / RMSE=113.71（同一学习权重，增加极端密度 4x4 rescue policy）

## 1. 执行结论

本轮已完成以下工作：

1. 补回此前缺失的 `2277443934862561`，pts16/pts32 cache 均由 2,114 补齐到 2,115。
2. 使用 FSC147 12.67 对应的同一组学习权重和阈值，重跑 MCAC M1-M6 全量消融。
3. 将 FSC147 的无 GT 触发规则 `fast n_candidates == 0 -> 4x4 tiled rescue` 迁移到 MCAC；共触发 3 张。
4. 修复 OCCAM GT 类别解析不一致，完成共享 pts32 候选上的 2,115 张 OCCAM-M 全量复现。
5. 审计发现现有主评测代码使用 GT-dot-derived `valid` 筛选候选；因此补跑严格 `candidate_filter=all` 的无 GT M6/M5 sanity。
6. 使用 ABC123 官方 checkpoint 和官方 MCAC 数据协议重跑 full 2,115，得到 9.46 / 17.52，成功复现 published 9.52 / 17.64。

最重要的结果如下：

| 方法/口径 | 是否在推理候选筛选中使用 GT | Per-class MAE | Per-class RMSE | Total MAE | Total RMSE |
|---|---:|---:|---:|---:|---:|
| ABC123 published | 否；但使用 density-map 监督 | **9.52** | **17.64** | - | - |
| ABC123 local official ckpt，full 2,115 | 否；但使用 density-map 监督 | **9.46** | **17.52** | 58.70\* | 90.20\* |
| OCCAM-M local，shared pts32 | 否 | 22.74 | 38.89 | 49.93 | 66.13 |
| Ours M6，严格 no-GT | 否 | 32.11 | 53.79 | **36.85** | **50.34** |
| Ours M6，现有 cache-compatible | **是** | 34.94 | 57.90 | 29.65 | 50.61 |

\* ABC123 total 指标为全部 5 个 heads 的预测总和；published per-class matching 会忽略未匹配的额外 heads，因此这两个口径不等价。

结论不能写成“我们在 MCAC 上达到 SOTA”：

- 严格 no-GT 的 ours 明显落后于 OCCAM-M local 和 ABC123 published。
- 当前 cache-compatible M1-M6 可以用于内部组件诊断，但由于使用 GT-derived `valid`，不能作为严格 prompt-free 主表。
- OCCAM 的 per-class 指标优于 ours，但 total-count 更差，说明 OCCAM 存在较多未匹配额外 cluster；MCAC 的 per-class matching 会忽略这些额外输出。
- MCAC 类别没有自然语言名。本实验只验证匿名多组计数，不能用来证明“输出正确类别名”；类别名 claim 仍应由 OmniCount-191 支撑。

## 2. “FSC147 12.67 同一模型”核对

FSC147 12.67 不是新训练 checkpoint，而是在审计后的 13.47 模型上增加 4x4 rescue：

| 模块 | 配置 |
|---|---|
| pts32 category head | `result/checkpoints/category_cosine_pts32.pt` |
| pts32 relation head | `result/checkpoints/fsc147_relation_pts32_best.pt` |
| fast category head | `result/checkpoints/category_cosine_fast.pt` |
| fast relation head | `result/checkpoints/fsc147_relation_best.pt` |
| text prototypes | `result/checkpoints/text_prototypes_fsc147.pt` |
| `tau_inst` | 0.99 |
| `tau_affinity` | 0.1 |
| `conf_threshold` | 0.2 |
| rescue trigger | fast cache `n_candidates == 0`，不使用 GT |
| rescue frontend | SAM2 pts32，4x4 tiles，overlap=0.25，bbox merge |

MCAC M6 使用同一 pts32 category/relation checkpoint 和相同阈值。根据预注册 MCAC adapter，普通图使用中心裁剪后的全图 pts32 cache；仅对无 GT 触发的 3 张应用 4x4 overlay。没有在 MCAC 上训练或微调模型。

需要注意：FSC147 12.67 是在 13.47 full-1190 结果上替换 `7611.jpg` 预测得到的 rescue policy 结果，而且同样依赖 GT-derived `valid`。本轮审计后，12.67 应暂时标为 cache/oracle-filtered exploratory result，而不是已完成部署级验证的 strict prompt-free headline。

## 3. 数据与 2,115 张完整性修复

MCAC test 目录共有 2,115 张。此前两套 cache 都只有 2,114 张，唯一缺失项为：

```text
2277443934862561
```

该图文件完整，但在 ABC123 test 协议 `center-crop672 + occlusion<70` 下：

- 原始类别：Plant；
- crop 内对象数：191；
- 191 个对象的 occlusion 均不小于 70；
- 最终 GT 为 0 类、0 个目标。

旧预处理器把“零有效类别”当成失败并跳过。本轮改为保留合法零目标图：

| 项目 | 修复前 | 修复后 |
|---|---:|---:|
| pts32 cache | 2,114 | 2,115 |
| pts16 cache | 2,114 | 2,115 |
| per-class image-class pairs | 3,630 | 3,630 |
| total-count images | 2,114 | 2,115 |

它不增加 per-class 分母，但必须进入 total-count 诊断。严格 no-GT M6 在该图预测 85 个实例，这说明保留零目标图对假阳性审计很重要。

数据按有效 GT 类别数分布为：0 类 1 张、1 类 1,013 张、2 类 760 张、3 类 267 张、4 类 74 张。

## 4. 评测协议

### 4.1 共用数据协议

- 图像：中心裁剪 672x672；
- GT：仅保留 `occlusion_crop672 < 70` 的目标；
- 输出：匿名预测实例按 top-1 FSC147 prototype bucket 分组；prototype 只作为匿名分组键，不解释为 MCAC 语义类名；
- 匹配：预测 bucket 与 GT 类别按 mask-union 覆盖 GT dot 数做 Hungarian 最大匹配；
- 零覆盖匹配视为未匹配，未匹配 GT 类预测为 0；
- 多余预测 bucket 不进入 per-class MAE/RMSE，但会进入 total-count 诊断。

### 4.2 协议审计：GT-derived `valid`

`script/preprocess_mcac.py` 中的 `valid` 定义是“候选 mask 是否覆盖至少一个 GT dot”。旧版 `script/eval_mcac.py` 又直接用：

```python
valid_orig = np.asarray(d["valid"]) > 0
```

筛选推理候选。相同模式也存在于当前 FSC147、CARPK 和 OmniCount 主评测脚本。因此：

- 本轮 M1-M6 的 cache-compatible 表严格复现项目现有口径；
- 该表包含 GT 候选筛选泄漏，只能作为组件诊断；
- 新增 `--candidate-filter all` 后，GT 只用于最终指标匹配，才是严格 no-GT inference sanity。

这项问题不会因为严格 no-GT 的 per-class MAE偶然更低而消失。GT-derived 筛选仍违反 prompt-free 协议，并会隐藏额外分组和零目标图上的假阳性。

## 5. MCAC M1-M6 Leave-One-Out

### 5.1 变体定义

| 变体 | 唯一主要变化 |
|---|---|
| M1 | relation dedup 替换为 class-bucket box-IoU NMS@0.5 |
| M2 | 关闭 ADF：无 confidence filter；MCAC 无常规 density routing，且不启用 rescue overlay |
| M3 | relation head 换为 `fsc147_relation_pts32_dotonly_m3.pt` |
| M4 | pts16 cache + `category_cosine_fast.pt` + `fsc147_relation_exp5c.pt`；保留 pts16 rescue overlay |
| M5 | 保留 M6 heads/阈值，关闭 4x4 rescue overlay |
| M6 | 完整模型，pts32 base + 3-image pts32 rescue overlay |

### 5.2 Cache-compatible 全量结果

以下表与当前 FSC147 12.67 代码口径一致，但使用 GT-derived candidate validity，只作内部消融。

| 变体 | Per-class MAE | Per-class RMSE | Delta MAE vs M6 | Total MAE | Total RMSE |
|---|---:|---:|---:|---:|---:|
| M1 - RH | 35.26 | 58.30 | +0.31 | 30.55 | 51.68 |
| M2 - ADF | **31.02** | **53.13** | **-3.93** | **16.72** | **27.41** |
| M3 - CP | 35.01 | 57.97 | +0.07 | 29.82 | 50.71 |
| M4 - HR | 39.99 | 65.79 | +5.04 | 51.73 | 83.04 |
| M5 - T4 | 34.94 | 57.90 | 0.00 | 29.65 | 50.61 |
| M6 full | 34.94 | 57.90 | 0.00 | 29.65 | 50.61 |

M6 image-bootstrap 95% CI：

- MAE：[33.35, 36.66]
- RMSE：[55.48, 60.20]

M2 MAE 95% CI 为 [29.55, 32.61]，与 M6 不重叠。M2 在 1,678 张图上降低 per-image class error，仅 174 张变差，因此不是小样本噪声。

### 5.3 组件结论

1. **RH 有小幅正贡献**：M1 比 M6 差 0.31 MAE；838 张变差、345 张改善、932 张不变。
2. **ADF 在 MCAC 上是负贡献**：关闭固定 `conf_threshold=0.2` 后 MAE 改善 3.93。该阈值在 FSC147 有效，但在 MCAC 域迁移中过滤过强。
3. **CP 边际贡献很小**：M3 只比 M6 差 0.07 MAE，与 FSC147 上 M3/M6 接近的结论一致。
4. **HR 是最明显的正向组件**：pts16 M4 比 pts32 M6 差 5.04 MAE，1,440 张图变差。
5. **T4 在 MCAC 当前触发集上无 per-class 收益**：M5/M6 的 2,115 张 per-class 预测完全一致。

因此论文不能写“所有组件在 MCAC 上都有效”。诚实结论是 HR 显著有效、RH/CP 小幅有效、ADF 跨域失配、当前 4x4 rescue 无有效覆盖。

## 6. 4x4 Rescue 分析

无 GT 触发条件在 MCAC 上命中 3 张：

| 图像 | GT | pts32 4x4 candidates | valid dot-cover candidates | M5 per-class pred | M6 per-class pred |
|---|---:|---:|---:|---:|---:|
| `6521150169351191` | 6 | 1 | 0 | 0 | 0 |
| `8834542134687416` | 20 | 17 | 0 | 0 | 0 |
| `9186053451918968` | 5 | 0 | 0 | 0 | 0 |

严格 no-GT 下，前两张的 total prediction 从 0 变为 1 和 9，但这些实例均未覆盖 GT dot，所以 per-class 匹配仍为 0。第三张仍无候选。

结论：触发规则可以发现全图 frontend failure，但 FSC147 上有效的 4x4 配置不能直接解决 MCAC 这三张。失败原因是 tile 内 SAM proposal 仍没有覆盖目标，而不是后端 relation dedup 把正确候选删掉。

## 7. 严格 No-GT Sanity

### 7.1 总指标

| 变体 | Per-class MAE | Per-class RMSE | bias | Total MAE | Total RMSE | Total bias |
|---|---:|---:|---:|---:|---:|---:|
| M5 strict，no rescue | 32.108 | 53.786 | -27.155 | 36.854 | 50.341 | +16.217 |
| M6 strict，4x4 rescue | 32.108 | 53.786 | -27.155 | 36.850 | 50.340 | +16.222 |

M6 strict 的 image-bootstrap 95% CI：MAE [30.66, 33.66]，RMSE [51.47, 56.06]。

严格 no-GT per-class 指标比 GT-valid 口径略好，是因为一些不直接覆盖 dot 的候选在分组/代表选择后提供了更接近 GT 的 group count；这不代表 GT 筛选合理。total-count 同时显示大量额外实例，且零目标图预测 85，说明 per-class matching 会隐藏假阳性。

### 7.2 按每图类别数

| GT 类别数 | 图像数 | Per-class MAE | Per-class RMSE | Total MAE |
|---:|---:|---:|---:|---:|
| 0 | 1 | - | - | 85.00 |
| 1 | 1,013 | 39.01 | 66.76 | 30.87 |
| 2 | 760 | 27.58 | 48.56 | 42.09 |
| 3 | 267 | 33.90 | 50.20 | 43.18 |
| 4 | 74 | 26.86 | 35.96 | 41.42 |

错误没有随类别数单调增加；单类别图反而更难，主要因为该切片包含更多高计数类别。因此 MCAC 上的主要瓶颈不是“图中类别越多越差”，而是高密度类别的候选覆盖和欠计数。

### 7.3 按单类 GT count

| 单类 GT count | pairs | MAE | RMSE | bias |
|---|---:|---:|---:|---:|
| 1-10 | 890 | 5.64 | 10.50 | +1.75 |
| 11-50 | 1,470 | 13.52 | 17.10 | -7.53 |
| 51-100 | 732 | 41.63 | 45.94 | -39.19 |
| 101-200 | 368 | 90.89 | 99.02 | -88.69 |
| 201-300 | 170 | 163.16 | 168.38 | -163.16 |

从 51 个目标开始出现严重系统性欠计数。最差 pair 为图像 `6128446514113359`：GT=293，prediction=24，绝对误差 269。

## 8. 多类别方法对比

| 方法 | Prompt | 计数监督 | 输出 | 候选/协议 | Per-class MAE | Per-class RMSE |
|---|---|---|---|---|---:|---:|
| ABC123 published | 无 | density map + count | 匿名 density heads | 论文 MCAC test | **9.52** | **17.64** |
| ABC123 local official ckpt | 无 | density map + count | 匿名 density heads | full 2,115，官方数据协议 | **9.46** | **17.52** |
| OCCAM-M local | 无 | 无训练 | 匿名 clusters | shared pts32 SAM + local FINCH | 22.74 | 38.89 |
| Ours M6 strict | 无 | 无 count/density supervision | FSC prototype anonymous buckets | all candidates + Hungarian | 32.11 | 53.79 |

说明：

- ABC123 published 数字直接引用 arXiv:2309.04820v2；local 行使用官方 checkpoint、官方 `MCAC_Dataset` 和原版 torchvision resize 语义。完整 2,115 张为 9.46/17.52，官方 `drop_last=True` 的 2,114 张为 9.45/17.51，说明 published 9.52/17.64 已成功复现。
- ABC123 per-class matching 只评估与非零 GT density map 匹配的 heads，忽略额外 heads。其全部 5 heads total MAE/RMSE 为 58.70/90.20；零目标图 `2277443934862561` 的全 head 预测总数为 401.20，但不进入 per-class 分母。
- OCCAM 行是本地共享候选适配，不是 OCCAM 论文公开 MCAC 数字。它复用与 ours 相同的 pts32 SAM pool，再运行 OCCAM area filtering、mask-IoU dedup、ResNet50 和 FINCH。
- OCCAM spatial matching MAE/RMSE 为 22.74/38.89；count-optimal matching 诊断 MAE 为 16.28。
- OCCAM total-count 为 49.93/66.13，差于 ours strict 的 36.85/50.34，说明其额外 cluster 较多；仅看 per-class matching 会忽略这些 cluster。
- 逐图 class-error 对比中，ours 优于 OCCAM 658 张，差于 OCCAM 1,370 张，87 张持平。
- OmniCount prompted 行本轮未跑，不应填入 MCAC 主表。

## 9. 对论文 Claim 的影响

### 可以保留

1. 相同 FSC147 学习权重可以零训练迁移到 MCAC，并输出多个匿名 group。
2. HR/pts32 候选对 MCAC 有明确正贡献。
3. 方法不使用 count label 或 density map 训练；这一监督差异相对 ABC123 仍成立。

### 必须降级或改写

1. 当前 12.67、CARPK、OmniCount 等 headline 评测使用 GT-derived `valid`，在完成 strict no-GT 全量重跑前，不能继续称为严格 prompt-free 部署结果。
2. 不能声称所有组件都有用：ADF 在 MCAC 显著反向，T4 在当前 MCAC 触发集上没有 per-class 收益。
3. 不能声称 MCAC 证明“类别名输出正确”：MCAC 类别匿名，FSC147 prototype 仅作为 bucket key。
4. 不能声称 ours 在 MCAC 优于 OCCAM 或 ABC123。

### 下一步优先级

1. 用可部署的 countability/foreground head 替代 GT-derived `valid`，然后重跑 FSC147、CARPK、OmniCount、MCAC 全部主结果。
2. 只在 MCAC train/val 上选择 confidence threshold；不能根据 test 上 M2 更好直接调参并重报 test。
3. 分离“额外预测 group 惩罚”和“GT class 匹配误差”，主表同时给 per-class 与 total-count。
4. 若要正式引用 OCCAM，补跑原生 OCCAM AMG 配方；当前 shared-proposal 行保留为受控本地 baseline。
5. 针对 51+ count 类别改进 proposal recall，当前误差主要来自高密度欠计数，而不是类别数本身。

## 10. 复现实验产物

| 文件 | 内容 |
|---|---|
| `script/preprocess_mcac.py` | 保留零目标样本，补齐 full 2,115 |
| `script/preprocess_mcac_rescue_tiled.py` | 无 GT 触发的 MCAC 4x4 rescue overlay |
| `script/eval_mcac.py` | M1-M6、overlay、完整性校验及 strict candidate filter |
| `script/eval_mcac_occam.py` | OCCAM shared-cache MCAC 全量评测 |
| `script/eval_abc123_mcac_official.py` | ABC123 官方 MCAC 协议 full/drop-last 双口径复现 |
| `script/summarize_mcac_full2115.py` | 指标复算、CI、切片和协议校验 |
| `docs/abc123_mcac_reproduction_report_20260710.md` | ABC123 MCAC 独立复现与协议审计报告 |
| `result/logs/abc123_mcac_reproduction_summary_20260710.json` | ABC123 复现机器可读汇总 |
| `result/logs/abc123_mcac_test_full2115_official{,_legacyresize}_20260710.json.gz` | ABC123 两种 resize 口径逐图结果 |
| `result/logs/mcac_full2115_leaveoneout_summary.json` | 本报告机器可读汇总 |
| `result/logs/mcac_full2115_m{1..6}_12p67.json.gz` | cache-compatible M1-M6 逐图原始结果（Git 压缩归档） |
| `result/logs/mcac_strict_nogt_full2115_m{5,6}_12p67.json.gz` | strict no-GT 逐图结果（Git 压缩归档） |
| `result/logs/mcac_occam_full2115_sharedpts32.json.gz` | OCCAM 逐图结果（Git 压缩归档） |

所有全量输出均通过以下断言：2,115 个唯一 image ID、3,630 个有效 image-class pair、M3 metadata 正确标记为 `m3`、零目标图存在、指标由逐图记录复算一致。
