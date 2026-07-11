# FSC-147 GT-assisted 词表泄露对照实验报告

**日期**：2026-07-11

**状态**：full test 1,190 已完成；逐图结果、5,000 次 bootstrap 与资产哈希已冻结

**性质**：协议归因实验，不是 strict no-GT 主结果

## 1. 结论先行

本实验回答两个不同问题，结论不能混在一起：

1. **冻结旧 12.67 category head 后直接删掉 test 类原型，会严重退化。**旧 147 类 head 的 MAE 从 `12.6681` 升至 `57.5790`。这说明旧 checkpoint 与完整 147 类词表强耦合，不能在不重新校准 head 的情况下直接裁词表。
2. **test 类名本身不是 GT-assisted 低 MAE 的必要条件。**换成 official-train-only category head，并只保留 official-train 89 类原型后，在其余历史泄露全部保留的条件下得到 **MAE=13.1109 / RMSE=109.6685**。相对旧 12.67，配对 MAE 只增加 `0.4429`，95% CI 为 `[-0.4950, 1.0412]`，没有显示本质性退化。
3. 在同一个 train-only category head、同一旧原型族下，词表从 `147 -> 118 -> 89` 行时，MAE 反而从 `15.5479 -> 14.1218 -> 13.0849`。加入 held-out 类名没有帮助当前 total-count，反而造成置信度和分组失配。

因此，可以较有把握地消除一个局部担心：**历史 12.67 的低 MAE 并不要求推理时知道 FSC test 类名**。但更大的担心没有消失：`13.11` 仍直接使用 test GT-derived `valid`、GT-selected MR 路由、旧 relation head 和 test-informed T4，不能进入论文主表。当前可报告主结果仍是 strict no-GT 的 `26.50 / 129.69`。

## 2. 实验问题与固定协议

FSC-147 official split 的类别互斥：

| Split | Images | Unique classes | 与其他 split 类别交集 |
|---|---:|---:|---:|
| train | 3,659 | 89 | 0 |
| validation | 1,286 | 29 | 0 |
| test | 1,190 | 29 | 0 |

所有变体固定以下历史 12.67 组件：

- full test 1,190，同一图像顺序和 annotation；
- cache `valid > 0`，该字段由 test GT dot 是否落入候选得到；
- 通过 MR51/MR100 cache membership 进行 GT-count-informed 路由；
- `conf_threshold=0.2`、`tau_inst=0.99`、`tau_affinity=0.1`；
- 旧 `fsc147_relation_best.pt` / `fsc147_relation_pts32_best.pt`；
- fast raw candidate 为 0 时使用 4x4 T4 rescue；全量中仅 `7611.jpg` 触发；
- category grouping、relation dedup 和所有后处理不变。

本实验故意保留上述泄露，以隔离“词表/类别 head”一侧的影响。任何一行都不能标为 no-GT。

## 3. 变体定义

| 变体 | Category head | Prototype bank | Test 类原型行 | 作用 |
|---|---|---:|---:|---|
| L147 | 旧 147 类、训练目录有 test overlap | 旧 full-147 | 29 | 精确复现历史 12.67 |
| L118 | 同 L147 | 旧原型删除 test 29 行 | 0 | 同一旧 head 的纯 test-row 删除 |
| L89 | 同 L147 | 旧原型仅 train 89 行 | 0 | 同一旧 head 删除全部 held-out 行 |
| C147 | official-train-only head | 旧 full-147 | 29 | clean head 的 same-head 词表参考 |
| C118 | 同 C147 | 旧原型删除 test 29 行 | 0 | clean head 的纯 test-row 删除 |
| C89-old | 同 C147 | 旧原型仅 train 89 行 | 0 | clean head，仅 train rows |
| C89-direct | 同 C147 | official-train 类名直接编码 | 0 | 最终“无泄露词表、保留 GT oracle”配置 |

`C89-old` 与 `C89-direct` 用于检查原型生成本身的影响。两套 train-89 原型并非逐元素相同：平均 cosine 为 `0.994405`，最小为 `0.989360`，因此不能把原型重编码混进纯裁行对照。

## 4. Full-test 结果

### 4.1 旧 head：直接裁词失败

| 变体 | MAE | RMSE | Bias | 相对 L147 的配对 MAE 差 | 95% CI | 变化图像 |
|---|---:|---:|---:|---:|---:|---:|
| **L147** | **12.6681** | 113.7111 | -7.9471 | 0 | [0, 0] | 0 |
| L118 | 57.5790 | 145.6918 | -57.5151 | +44.9109 | [41.6905, 48.3471] | 1,172 |
| L89 | 58.3202 | 146.0983 | -58.2529 | +45.6521 | [42.3538, 49.0639] | 1,178 |

L118 的 mean prediction 从 `58.30` 降到 `8.73`；1,169 张图预测下降，1,148 张图误差变差。这不是“test 类名对计数提供了 45 MAE 的真实信息”，而是旧 head/阈值对完整旧词表的严重校准依赖。

### 4.2 Train-only head：test 类名不必要

| 变体 | MAE | RMSE | Bias | 相对 C147 的配对 MAE 差 | 95% CI |
|---|---:|---:|---:|---:|---:|
| C147 | 15.5479 | 114.7505 | -13.0639 | 0 | [0, 0] |
| C118 | 14.1218 | 111.7693 | -10.7202 | **-1.4261** | [-1.9404, -1.0756] |
| C89-old | **13.0849** | **109.5305** | -8.6496 | **-2.4630** | [-3.4330, -1.7857] |
| C89-direct | 13.1109 | 109.6685 | -8.6370 | **-2.4370** | [-3.3950, -1.7697] |

进一步的配对结果：

- `C89-old - C118`：MAE `-1.0370`，95% CI `[-1.5025, -0.6941]`；删除 validation 29 行仍改善。
- `C89-direct - C89-old`：MAE `+0.0261`，95% CI `[-0.0235, 0.0815]`；直接重编码与旧行裁剪对计数几乎无差别。
- `C89-direct - L147`：MAE `+0.4429`，95% CI `[-0.4950, 1.0412]`；两者 RMSE 为 `109.6685` 与 `113.7111`。

最后一项不是纯单变量对照，因为同时替换了 category head；它回答的是更实际的问题：在保留历史 GT filtering 的情况下，是否存在一个完全不含 val/test 类名、且性能接近 12.67 的 category 配置。答案是肯定的。

### 4.3 GT-count 分桶

| 变体 | 0-10 MAE | 11-20 | 21-50 | 51-100 | 100+ |
|---|---:|---:|---:|---:|---:|
| L147 | **1.60** | **2.19** | **5.78** | **8.00** | 51.14 |
| C147 | 2.30 | 3.34 | 8.13 | 10.60 | 58.56 |
| C118 | 1.95 | 2.86 | 7.22 | 9.67 | 53.76 |
| C89-old | 1.78 | 2.53 | 6.52 | 9.29 | **49.93** |
| C89-direct | 1.78 | 2.52 | 6.50 | 9.25 | 50.18 |

C89-direct 在低、中密度段比 L147 略差，但 100+ 段略好。总 RMSE 降低主要受极端高密度图改善影响，不能解释成整体统计显著优于 L147。

## 5. 为什么旧 head 裁词会崩，而 clean head 不会

旧 category head 使用 147 类原型训练，训练 cache 又包含大量 official-test 图。对 test 图候选，正确 test prototype 往往是高置信度 winner。删掉该行后，剩余类别 logits 变得分散，许多候选的最大 softmax 低于固定 `0.2`，于是即使候选先通过 GT-derived `valid`，仍被 category confidence gate 删除。

典型样本如下：

| Image | GT | L147 pred / valid | L118 pred / valid | C89-direct pred / valid |
|---|---:|---:|---:|---:|
| `6281.jpg` | 675 | 670 / 670 | 21 / 21 | 657 / 658 |
| `687.jpg` | 548 | 493 / 497 | 30 / 30 | 473 / 476 |
| `7611.jpg` | 2,560 | 1,098 / 1,099 | 629 / 629 | 1,550 / 1,550 |
| `1123.jpg` | 3,701 | 164 / 165 | 79 / 79 | 163 / 163 |

表中的 `valid` 是同时满足 cache `valid > 0` 与 `top_conf >= 0.2` 的有效候选数。Train-only head 从训练阶段起就在 89 类空间内校准，因此不会在删除 test winner 后出现同样的置信度坍缩。更重要的是，GT `valid` 已经替模型完成最困难的前景筛选；category 模块在这里主要影响置信度、匿名分组和去重，而不是必须输出真实 test 类名。因此一个候选即使被映射到错误的 train 类，只要同类候选的分组结构稳定，total count 仍可接近历史结果。

## 6. 对当前方法的含义

### 6.1 可以下的结论

- 在 GT-assisted 历史 pipeline 中，去掉所有 val/test 类名并使用匹配的 train-only category head 后，MAE 仍为 `13.11`；test 类名不是低 MAE 的必要条件。
- 旧 checkpoint 的 `147 -> 118` 崩溃是 checkpoint/vocabulary calibration failure，不能拿来证明 class-name prompting 对 count 有巨大贡献。
- filtering 确实解释了为什么无 test 类名仍能计数：GT oracle 已把大多数背景候选排除，category 只需提供足够稳定的置信度和分组。

### 6.2 不能下的结论

- 不能说“原 12.67 同一个冻结模型去掉词表后完全不变”；同一旧 head 的纯裁行实验明确失败。
- 不能把 `13.11` 写成 prompt-free/no-GT 主结果。它仍读取 test GT-derived `valid`，并使用 GT-informed cache 路由。
- 不能由本实验得出 category 模块在 strict no-GT 下不重要。移除 GT oracle 后，candidate filter 与 category confidence 的误差会重新成为主要瓶颈，strict 结果仍是 `26.50`。
- C89-direct 仍保留旧 relation head；该权重的训练目录有 official-test overlap。这里只清理了 category vocabulary/head 一侧，不是完整 train-only 模型。

## 7. 复现与资产

运行命令：

```bash
python3 script/eval_fsc147_leaky_filter_vocab_ablation.py \
  --bootstrap 5000 --progress-every 50
```

关键文件：

- 评估入口：`script/eval_fsc147_leaky_filter_vocab_ablation.py`
- 原始逐图输出：`result/logs/fsc147_leaky_filter_vocab_ablation_full1190.json`
- 仓库同步压缩包：`result/logs/fsc147_leaky_filter_vocab_ablation_full1190.json.gz`
- 原始 JSON SHA-256：`86a60693f4f3b8523690c2897fb1f7f72acd7c1e6ec605a93cd5e140355bd166`
- 同步压缩包 SHA-256：`bdf359f6a4a08037e5f4c7c6600ed30750bea9e6574fa8f86e3c3245592c0b1b`
- 大小：原始 JSON 2,240,594 bytes；压缩包 124,883 bytes
- 旧 prototype：`result/checkpoints/text_prototypes_fsc147.pt`
- train-only prototype：`result/checkpoints/text_prototypes_fsc147_train89.pt`
- train-only category：`result/checkpoints/cp_strict/category_pts16_trainvocab.pt`、`category_pts32_trainvocab.pt`

完整性检查：7 个变体均为 1,190 个唯一 image ID；所有 metric 已由逐图预测独立复算；L147 精确通过历史锚点断言 `12.668067226890756 / 113.71112226924458`。
