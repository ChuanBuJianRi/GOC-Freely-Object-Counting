# FSC-147 Full + OmniCount-191 Full 组件 Leave-One-Out 消融报告

**日期**：2026-07-10

**FSC147 legacy engineering anchor**：MAE=12.67 / RMSE=113.71（旧训练 split 含 official-test 图）

**测试集**：FSC-147 test full 1,190 + OmniCount-191 test full 1,957

**实验计划**：`docs/experiment_plan_ablation_fsc147_omnicount_20260710.md`

> **CP 口径更正**：下表 M3 是历史工程诊断，不是真正 leave-one-out。它只替换 pts32 relation head，fast 分支仍使用 CP head，而且旧 category/relation 训练目录混入 official val/test。CP 的正式结论改用本文第 3.1 节的 official-train-only 三随机种子配对实验。

## 1. 执行结论

本轮完成了统一 M1-M6 leave-one-out 实现，并在两个完整测试集上运行。最重要结果如下：

| Variant | 移除组件 | FSC147 MAE | FSC147 RMSE | OmniCount total MAE | OmniCount total RMSE | mRMSE | mRMSE-nz |
|---|---|---:|---:|---:|---:|---:|---:|
| M1 | RH | 14.29 | 113.91 | 4.68 | 8.46 | 0.457 | 3.911 |
| M2 | ADF | 26.97 | 124.52 | 12.73 | 17.30 | 0.842 | 3.864 |
| M3† | CP（legacy partial/leaky） | 12.59 | 113.69 | 4.68 | 8.46 | 0.457 | 3.911 |
| M4 | HR | 27.18 | 125.83 | 5.64 | 9.72 | **0.424** | 3.914 |
| M5 | T4 | 13.47 | 126.74 | 4.68 | 8.46 | 0.457 | 3.911 |
| **M6** | **完整模型** | **12.67** | **113.71** | **4.68** | **8.46** | 0.457 | **3.911** |

不能把这张表解释成“所有组件在所有数据集都有效”：

1. **RH、ADF、HR、T4 在 FSC147 上有正贡献**；其中 ADF/HR 的影响最大。
2. **严格 CP 消融显示最终 MAE 等效**：official-train-only 三 seed 下 CP 为 13.9706±0.0933，scratch 为 13.9714±0.1214，paired Δ=-0.00084，95% CI 跨 0。旧 M3 的 -0.073 不再参与判断。
3. **OmniCount 上只有 ADF 和 HR 改变 total-count 结果**；M1/M3†/M5 与 M6 逐图完全相同。
4. OmniCount 的 `mRMSE` 会被大量 GT=0 类别影响。M4 的 mRMSE 更低，不代表其 total counting 更好；其 total MAE 实际从 4.68 退化到 5.64。

## 2. 旧消融口径修复

原 FSC147 M1-M5 以 no-rescue M6=13.47 为基线，而主方法后来升级为 T4 rescue 后的 12.67。旧表直接用 M1-M5 减 12.67 会混合两个推理策略。

本轮修复为：

- M1-M4 均保留 `fast n_candidates==0 -> 4x4 rescue`；
- 只有 M5 关闭 T4；
- `7611.jpg` 在 M1/M2/M3/M4/M6 上使用同一份 4x4 cache；
- 所有变体包含完整 1,190 张，不再使用 sample100 或 1,189-image cache。

该修复解释了 M2/M4 与旧记录的变化：旧 M2/M4 约为 28.5，本轮保留 T4 后分别为 26.97/27.18。

## 3. 模型与参数一致性

| 模块 | M6 配置 |
|---|---|
| fast category | `category_cosine_fast.pt` |
| fast relation | `fsc147_relation_best.pt` |
| pts32 category | `category_cosine_pts32.pt` |
| pts32 relation | `fsc147_relation_pts32_best.pt` |
| M3 dot-only relation | `fsc147_relation_pts32_dotonly_m3.pt`（旧实验只替换 pts32） |
| M4 pts16 relation | `fsc147_relation_exp5c.pt` |
| FSC vocabulary | `text_prototypes_fsc147.pt` |
| OmniCount vocabulary | `text_prototypes_omnicount.pt`，93 类 |
| `tau_inst` / `tau_affinity` | 0.99 / 0.1 |
| FSC confidence | 0.2 |
| OmniCount confidence | 0.1，沿用 2026-07-08 已建立协议 |
| box-IoU baseline | 0.5 |

OmniCount 只替换文本原型，不重新训练 category/relation head。M4 新生成 pts16 full cache，共 1,957 个文件，与 pts32 cache 文件名集合完全一致；候选数范围 1-137，均含 `points_per_side=16` metadata。

### 3.1 CP official-train-only 配对补充

旧训练脚本直接 glob 约 6,143 个覆盖 train/val/test 的 cache 后随机划分。按旧 seed42 复算，category fast/pts32 分别有 1,002/1,018 张 official-test 图进入 train，relation fast/pts32 分别有 1,066/1,075 张进入 train。旧 M3 又只替换 pts32 relation，因此 `12.595 vs 12.668` 不能回答 CP 是否有效。

修复后对 pts16/pts32 同时训练 CP 与 scratch，模型 seed 为 17/42/73，训练文件仅来自 official train；每个完整模型只在 official val 选择 `tau_inst`，再冻结到 full test：

| Condition | Test MAE mean±std | Test RMSE mean±std | MAE w/o 7611 | Val-selected tau |
|---|---:|---:|---:|---:|
| CP | **13.9706 ± 0.0933** | **111.2808 ± 0.0174** | **12.9840 ± 0.0934** | 0.999 / 0.999 / 0.999 |
| Scratch | 13.9714 ± 0.1214 | 111.2902 ± 0.0112 | 12.9843 ± 0.1223 | 0.999 / 0.999 / 0.999 |

配对 `CP-Scratch` ΔMAE 为 `+0.0151/+0.0210/-0.0387`，均值 **-0.00084±0.03288**，逐图 bootstrap 95% CI **[-0.04398,+0.04314]**。CP 改善训练 relation loss/precision，但未转化为 test counting MAE。完整记录见 `docs/fsc147_cp_strict_multiseed_report_20260710.md`。

## 4. FSC-147 Full 结果

### 4.1 总体与 paired delta

| Variant | MAE | RMSE | Bias | ΔMAE vs M6 | Paired ΔMAE 95% CI | 改变图像数 |
|---|---:|---:|---:|---:|---:|---:|
| M1 - RH | 14.293 | 113.911 | -5.586 | +1.625 | [1.280, 1.998] | 710 |
| M2 - ADF | 26.966 | 124.523 | -25.655 | +14.297 | [11.558, 17.245] | 475 |
| M3† - CP | 12.595 | 113.686 | -8.146 | -0.073 | [-0.220, 0.071] | 378 |
| M4 - HR | 27.184 | 125.825 | -25.910 | +14.516 | [11.863, 17.399] | 446 |
| M5 - T4 | 13.475 | 126.742 | -8.754 | +0.807 | [0.000, 2.420] | 1 |
| **M6 full** | **12.668** | **113.711** | **-7.947** | 0.000 | [0.000, 0.000] | - |

解释：

- **RH 有稳定贡献**：box-IoU heuristic 使 MAE 增加 1.63，paired CI 不跨 0。
- **ADF 与 HR 是主要贡献项**：分别增加 14.30 和 14.52 MAE。两者都影响高密度候选覆盖，因此数值接近，但不是同一个开关。
- **旧 M3 不作 CP 因果解释**：它的差值来自 partial/leaky 配置。第 3.1 节的严格结果同样不支持 CP 提升 MAE，但也不支持 CP 有害；合理定位只能是训练初始化/关系校准选择。
- **T4 只改变一张图**：M5/M6 仅 `7611.jpg` 不同，因此普通 image bootstrap 的 CI 下界为 0。它对极端密度有效，但统计证据集中在一个样本。

### 4.2 GT count 分桶

| Variant | 0-10 | 11-20 | 21-50 | 51-100 | 100+ |
|---|---:|---:|---:|---:|---:|
| M1 | 1.52 | 2.15 | 5.65 | 13.63 | 54.08 |
| M2 | 1.62 | 2.18 | 5.79 | 18.16 | 125.14 |
| M3† | 1.60 | 2.19 | 5.78 | 7.56 | 51.26 |
| M4 | 1.60 | 2.19 | 5.78 | 18.24 | 126.38 |
| M5 | 1.60 | 2.19 | 5.78 | 8.00 | 56.06 |
| **M6** | **1.60** | **2.19** | **5.78** | 8.00 | **51.14** |

ADF/HR 的退化几乎全部来自 51+ 高密度图；M1 的主要退化集中在 51-100，说明 relation dedup 在中高密度候选重叠时更重要。

## 5. OmniCount-191 Full 结果

### 5.1 总体指标

| Variant | Total MAE | Total RMSE | Bias | Present-class MAE/RMSE | mRMSE | mRMSE-nz | ΔMAE vs M6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 - RH | 4.677 | 8.463 | -4.356 | 3.128 / 6.960 | 0.457 | 3.911 | 0.000 |
| M2 - ADF | 12.734 | 17.301 | +12.346 | **3.095 / 6.957** | 0.842 | **3.864** | +8.058 |
| M3† - CP | 4.677 | 8.463 | -4.356 | 3.128 / 6.960 | 0.457 | 3.911 | 0.000 |
| M4 - HR | 5.637 | 9.717 | -5.577 | 3.143 / 6.961 | **0.424** | 3.914 | +0.960 |
| M5 - T4 | 4.677 | 8.463 | -4.356 | 3.128 / 6.960 | 0.457 | 3.911 | 0.000 |
| **M6 full** | **4.677** | **8.463** | **-4.356** | 3.128 / 6.960 | 0.457 | 3.911 | 0.000 |

M6 的 4.676546/8.463454 与已有 `omnicount_multiclass_ablation_full1957_fsc147head_conf01.json` 精确一致，说明新统一实现没有改变既有 OmniCount 主协议。

### 5.2 组件解释

- **M1=M6，0/1,957 图变化**：85,387 个原始候选经 `conf=0.1` 后剩 3,282 个；其 semantic groups 全部是 singleton（最大 group size=1），最大同 top-class pair 的 `p_i·p_j=0.08675 < tau_affinity=0.1`。relation head 根本没有进入可改变 count 的组内去重，RH 的贡献不能由该 OmniCount 配置支撑。
- **M2 明显过计数**：关闭 confidence filter 后 mean prediction 从 1.68 增至 18.38，total bias 从 -4.36 变为 +12.35。ADF 对跨域假阳性控制非常关键。
- **M3=M6，0/1,957 图变化**：这是上述 singleton semantic groups 的结构性结果，不是 CP 与无 CP 模型能力相同的独立证据；正式 CP 判断只使用第 3.1 节严格 FSC 配对实验。
- **M4 改变 1,381/1,957 图**：pts16 使 total MAE 增加 0.96，paired 95% CI [0.867, 1.061]，HR 有稳定贡献。
- **M5=M6，0/1,957 图变化**：OmniCount pts32 cache 没有零候选图，T4 policy 从未触发。这是 policy 的预期行为，不代表 T4 在 FSC 极端密度图上无效。

### 5.3 按每图类别数

| GT 类别数 | 图像数 | M6 MAE/RMSE | M2 MAE/RMSE | M4 MAE/RMSE |
|---:|---:|---:|---:|---:|
| 1 | 1,124 | 4.49 / 10.02 | 14.67 / 19.31 | 5.36 / 11.43 |
| 2 | 311 | 4.17 / 5.02 | 13.14 / 16.37 | 5.43 / 6.10 |
| 3 | 186 | 6.02 / 7.12 | 12.73 / 16.52 | 7.51 / 8.47 |
| 4+ | 336 | 5.03 / 5.45 | 5.90 / 9.84 | 5.73 / 6.25 |

ADF 的过计数在 1-3 类图最严重；HR 在所有类别数切片上均改善 total count。类别数不是唯一难度来源，高密度 Supermarket/Birds 仍主导误差。

### 5.4 按 GT total count

| GT count | 图像数 | M6 MAE/RMSE | M2 MAE/RMSE | M4 MAE/RMSE |
|---|---:|---:|---:|---:|
| 0-10 | 1,772 | 3.15 / 3.81 | 11.78 / 15.66 | 3.86 / 4.56 |
| 11-20 | 123 | 10.97 / 11.51 | 17.09 / 22.76 | 13.06 / 13.36 |
| 21-50 | 42 | 26.38 / 27.44 | 34.45 / 40.80 | 32.17 / 33.16 |
| 51-100 | 20 | 55.80 / 57.67 | 24.90 / 29.73 | 61.80 / 63.20 |

M2 在 51-100 上 total MAE 更低，是因为无过滤产生大量候选，偶然缓解总数欠计；但其类别名仍错误，present-class prediction 在该切片为 0，因此不能据此选择 M2。

## 6. 指标陷阱与置信度敏感性

OmniCount 有 93 类，大量 image-class 条目为 GT=0：

- M4 输出更少，降低了 GT=0 类别上的假阳性，因此 `mRMSE 0.424 < 0.457`；
- 但 M4 total MAE 5.64 明显差于 M6 的 4.68；
- M2 的 `mRMSE-nz=3.864` 略优于 M6 3.911，但 total MAE 恶化 8.06，并产生大量错误类别输出。

因此论文不能只报 mRMSE。建议 OmniCount 主表同时报告：

1. total MAE/RMSE；
2. mRMSE 与 mRMSE-nz；
3. present-class MAE/RMSE；
4. predicted total/bias。

跨域阈值敏感性：

| OmniCount M6 confidence | Total MAE | Total RMSE | Mean prediction | mRMSE | mRMSE-nz |
|---:|---:|---:|---:|---:|---:|
| **0.1，既有主协议** | **4.68** | **8.46** | 1.677 | 0.457 | **3.911** |
| 0.2，FSC 参数直接迁移 | 5.98 | 10.15 | 0.049 | **0.407** | 3.919 |

`conf=0.2` 几乎使模型全零。0.1 是本轮前已存在的 OmniCount adapter，不是看完本轮 test 后选择；0.2 仅作为跨域 calibration 证据。

## 7. 论文结论

### 可以写

1. RH 对 FSC147 有稳定贡献，box-IoU 无法替代 learned relation dedup。
2. ADF 和 HR 是两个数据集上最稳定、最明确的组件；HR 的 OmniCount paired CI 不跨 0。
3. T4 是保守的极端失败 rescue：FSC147 只触发一张并显著改善该图，OmniCount 无触发时无副作用。
4. 同一 FSC147 学习权重可在 OmniCount 输出 prompt-free semantic groups；M6 精确复现 4.68/8.46。

### 不能写

1. 不能声称 CP 提升或损害最终 MAE；严格三 seed 结果为统计和实际等效。
2. 不能声称 RH 在 OmniCount 上有可测收益；M1/M6 逐图一致。
3. 不能用 OmniCount mRMSE 单独判断组件优劣。
4. FSC147 表仍是 GT-derived `valid` 的 cache-compatible 口径，strict no-GT full 重跑仍待补。

## 8. 复现实验产物

| 文件 | 内容 |
|---|---|
| `script/eval_fsc147_omnicount_leaveoneout.py` | 双数据集 M1-M6、切片、bootstrap、paired CI |
| `script/preprocess_omnicount.py` | 支持完整 pts16 cache 与零候选占位 |
| `result/logs/fsc147_omnicount_leaveoneout_full_summary.json` | 主机器可读汇总 |
| `result/logs/fsc147_omnicount_leaveoneout_full_fsc147.json.gz` | FSC147 六变体逐图结果 |
| `result/logs/fsc147_omnicount_leaveoneout_full_omnicount.json.gz` | OmniCount 六变体逐图结果 |
| `result/logs/fsc147_omnicount_leaveoneout_full_summary_conf02.json` | OmniCount conf=0.2 敏感性汇总 |
| `docs/fsc147_cp_strict_multiseed_report_20260710.md` | CP/scratch official-train-only 三 seed 完整报告 |
| `result/logs/cp_strict_multiseed_summary.json` | 严格 CP 实验可读汇总 |
| `result/logs/cp_strict_multiseed.json.gz` | 严格 CP 实验逐阈值、逐图结果 |

所有主输出通过以下断言：FSC 1,190 个唯一 ID；OmniCount 1,957 个唯一 ID；M6 anchors 精确一致；M5=M6 on OmniCount；FSC M5/M6 只改变 `7611.jpg`；指标可由逐图结果复算。
