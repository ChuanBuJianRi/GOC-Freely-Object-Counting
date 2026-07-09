# ABC123 复现实验记录

**日期**: 2026-07-08；2026-07-09 补充 FSC147 full-test 切片分析与复现差异排查
**分支**: `ljs`
**论文**: ABC Easy as 123: A Blind Counter for Exemplar-Free Multi-Class Class-agnostic Counting, arXiv:2309.04820v2
**官方仓库**: https://github.com/ActiveVisionLab/ABC123

---

## 1. 方法配置判断

ABC123 的推理输入是 **image-only**，不需要 exemplar box，也不需要 text/class prompt。因此它可以算 **prompt-free / exemplar-free**。

但它不是 dense-map-free 方法。它的核心是：

- DINO ViT-S/8 backbone；
- 5 个 counting heads，每个 head 输出一张 density map；
- 每个 head 的 count 由 density map 积分得到；
- 训练使用 GT density map 的 pixel-wise loss；
- 多类别无序输出通过 Hungarian matching 对齐到 GT density maps；
- 论文明确说明：quantitative evaluation 使用同样的 GT-density matching；真实部署时没有 GT density，因此 matching 不可用。

所以 ABC123 在我们的论文中应标注为：

```text
prompt-free / exemplar-free, but dense-map-supervised and dense-map-output.
No semantic class names. No per-class category labels without user interpretation.
```

它可以作为 prompt-free dense regression baseline，但不能作为 “dense map-free” 同范式 baseline。

---

## 2. 论文 published 数字核对

论文源文件来自 arXiv v2：`tables/FSC147_results_table.tex`。

FSC147 表格标题是：

```text
Comparison to SOTA methods when trained on MCAC and applied to the cases in FSC147 with fewer than 300 objects.
```

也就是说，论文 FSC147 数字不是官方 full test，而是剔除了 GT count > 300 的高密度图。

| Method | Shots | Sub-Class Combine | Val MAE | Val RMSE | Test MAE | Test RMSE | 备注 |
|---|---:|---|---:|---:|---:|---:|---|
| ABC123 | 0 | Max | 19.56 | 46.71 | 22.43 | 47.35 | FSC147, GT<300 subset |
| ABC123 | 0 | Sum | **11.13** | **34.47** | **11.75** | **33.41** | FSC147, GT<300 subset |

用户提到的 “FSC147 9 点多” 不在 v2 的 FSC147 table 里。论文中 `9.52 / 17.64` 出现在 MCAC test table，对应 ABC123 在 MCAC 上的结果，不是 FSC147。

---

## 3. 本地复现实验设置

官方仓库克隆到：

```bash
/tmp/ABC123
```

官方 README 权重下载地址：

```text
https://www.robots.ox.ac.uk/~mahobley/ABC123/model_chkpt.zip
```

解压后 checkpoint：

```bash
/tmp/ABC123/checkpoints/model_chkpt.ckpt
```

Checkpoint metadata：

| 项目 | 值 |
|---|---:|
| epoch | 86 |
| global_step | 102254 |
| best val_DDP_MAE | 8.9605 |
| best path | `blendersnsm009all_backboneall_intcount_mhat5/...` |

新增评估脚本：

```bash
script/eval_abc123_baseline.py
```

脚本复用官方 `models/backbone_vit.py`，并在本项目内等价实现 counting head，避免额外 `einops/pytorch_lightning` 依赖。推理设置：

- 输入 resize 到 `224x224`；
- 不做 ImageNet normalization，保持官方 `data.py` 的 ToTensor + Resize 口径；
- `gtd_scale=400`；
- 输出四种读数：
  - `sum`: 5 个 density heads 全部积分求和，接近论文 FSC 的 Sub-Class Combine=Sum；
  - `max_density`: 逐像素取 5 个 density heads 的最大值再积分，接近论文 Sub-Class Combine=Max；
  - `head_max`: 取单个最大 count head；
  - `best_head_oracle`: 选择最接近 GT count 的 head，只作诊断，不能用于主比较。

---

## 4. FSC147 复现结果

2026-07-09 重新跑 full test，结果与 2026-07-08 记录一致：

```bash
python3 script/eval_abc123_baseline.py \
  --dataset fsc147 --split test \
  --batch-size 64 --device cuda \
  --out result/logs/abc123_fsc147_test_full1190.json
```

### 4.1 FSC147 test full 1,190

| Protocol | 图像数 | MAE | RMSE | bias | mean GT | mean pred |
|---|---:|---:|---:|---:|---:|---:|
| ABC123 sum | 1,190 | 48.19 | 151.11 | -20.51 | 66.25 | 45.74 |
| ABC123 max_density | 1,190 | 43.89 | 150.71 | -37.00 | 66.25 | 29.25 |
| ABC123 head_max | 1,190 | 45.80 | 151.34 | -42.02 | 66.25 | 24.23 |
| ABC123 best_head_oracle | 1,190 | 44.55 | 151.03 | -43.58 | 66.25 | 22.67 |

结果文件：

```bash
result/logs/abc123_fsc147_test_full1190.json
```

### 4.2 FSC147 test, GT<=300

本地 FSC147 test split 中 GT>300 的图像有 23 张，因此 `GT<=300` 后为 1,167 张。

| Protocol | 图像数 | MAE | RMSE | bias | mean GT | mean pred |
|---|---:|---:|---:|---:|---:|---:|
| ABC123 sum | 1,167 | 37.51 | 58.96 | -9.46 | 54.51 | 45.05 |
| ABC123 max_density | 1,167 | 32.63 | 55.48 | -25.60 | 54.51 | 28.91 |
| ABC123 head_max | 1,167 | 34.54 | 56.99 | -30.69 | 54.51 | 23.82 |
| ABC123 best_head_oracle | 1,167 | 33.27 | 56.13 | -32.28 | 54.51 | 22.23 |

结果文件：

```bash
result/logs/abc123_fsc147_test_le300.json
```

### 4.3 与论文 FSC147 table 的差距

| Protocol | Paper test MAE | 本地 test MAE | 差距 |
|---|---:|---:|---:|
| ABC123 Sum, GT<300 | 11.75 | 37.51 | +25.76 |
| ABC123 Max, GT<300 | 22.43 | 32.63 | +10.20 |

论文可视化示例也没有复现。例如 `4756.jpg` 在论文图文件名中是 GT=14、Pred=14.7；本地官方 checkpoint 输出 `sum=2.8`。这说明差距不是指标聚合问题，而是官方 README checkpoint / FSC adapter / 图像预处理 / split 口径中至少有一项与论文未公开流程不同。

因此本文档不把 ABC123 写成 “复现 paper FSC147 11.75 成功”，而写成：

```text
official README checkpoint reproduction on local FSC147 protocol.
Published FSC147 paper table is cited separately and marked as not reproduced locally.
```

### 4.4 Full vs GT<300 vs GT>300

本地 test split 没有恰好 GT=300 的图，因此 `GT<300` 与脚本中的 `GT<=300` 在本次统计中等价。ABC123 论文只给 `GT<300` 口径，但 full test 的高密度图对结论影响很大。

| Subset | 图像数 | mean GT | ABC123 sum MAE/RMSE | ABC123 max_density MAE/RMSE | 主要结论 |
|---|---:|---:|---:|---:|---|
| Full test | 1,190 | 66.25 | 48.19 / 151.11 | 43.89 / 150.71 | 全量口径，RMSE 很高 |
| GT<300 | 1,167 | 54.51 | 37.51 / 58.96 | 32.63 / 55.48 | 仍未复现论文 11.75 |
| GT>300 | 23 | 661.87 | 590.34 / 1002.52 | 615.34 / 1009.48 | 极端欠计数 |

关键观察：

- `GT>300` 只有 23 张，占 full test 的 **1.93%**。
- 但这 23 张贡献了 ABC123 `sum` 的 **23.7% 绝对误差** 和 **85.1% 平方误差**。
- 因此 ABC123 论文只报 `GT<300` 是一个很强的评估限制；full-test RMSE 不能从 paper table 推断。
- 即使在 `GT<300` 子集，本地 official checkpoint 仍明显弱于 paper table：`sum` MAE `37.51` vs paper `11.75`。

### 4.5 按 GT count bin 的 full-test 切片

| GT bin | 图像数 | mean GT | sum MAE/RMSE/bias | max_density MAE/RMSE/bias |
|---|---:|---:|---:|---:|
| 0-10 | 60 | 9.15 | 9.76 / 12.75 / +7.50 | 3.78 / 5.28 / -0.23 |
| 11-20 | 268 | 14.81 | 14.44 / 25.75 / +9.55 | 8.96 / 17.69 / +0.35 |
| 21-50 | 413 | 33.67 | 22.79 / 30.38 / +1.59 | 18.36 / 24.01 / -11.22 |
| 51-100 | 254 | 72.57 | 47.30 / 57.40 / -12.46 | 41.56 / 49.52 / -32.34 |
| 101-300 | 172 | 155.57 | 104.02 / 124.14 / -67.10 | 100.65 / 124.02 / -99.43 |
| 301+ | 23 | 661.87 | 590.34 / 1002.52 / -581.04 | 615.34 / 1009.48 / -615.34 |

这个切片显示 ABC123 的 official checkpoint 在低计数图上还能工作，尤其 `max_density` 在 `0-20` 区间相对稳定；但从 `51-100` 开始系统性欠计数，`301+` 基本失效。`sum` 在低计数区间偏过计数，在高计数区间偏欠计数；`max_density` 更保守，因此低计数更好、高计数更差。

### 4.6 最大误差样例

| Image | GT | sum pred | max_density pred | sum error |
|---|---:|---:|---:|---:|
| `1123.jpg` | 3701 | 0.6 | 0.5 | -3700.4 |
| `7611.jpg` | 2560 | 0.0 | 0.0 | -2560.0 |
| `2159.jpg` | 621 | 84.4 | 67.6 | -536.6 |
| `6860.jpg` | 512 | 7.1 | 2.2 | -504.9 |
| `7473.jpg` | 508 | 7.4 | 6.5 | -500.6 |

这些样例解释了 full-test RMSE 为什么接近 `151`：并不是全体样本都差，而是少数超高密度图的预测几乎塌到 0 或几十。

### 4.7 对论文比较的建议

ABC123 可以在论文中引用，但建议拆成两行：

| Method | Protocol | MAE/RMSE | 用途 |
|---|---|---:|---|
| ABC123 paper | FSC147 GT<300, published, dense-map supervised | 11.75 / 33.41 | 说明 prompt-free dense regression 的 published reference |
| ABC123 local official ckpt | FSC147 full 1,190, official README checkpoint | 48.19 / 151.11 | 说明本地 full-test 复现与真实全量压力 |

不要把 ABC123 paper `GT<300` 数字直接与 OV-CUD full-test `1,190` 数字当作同协议比较。更稳妥的写法是：

```text
ABC123 reports strong prompt-free dense-regression results on the FSC147 subset with fewer than 300 objects. On our full FSC147 test evaluation using the released checkpoint, the method suffers from severe undercounting on the 300+ images; these 1.93% images account for 85.1% of the squared error. We therefore report ABC123 as a dense-map-supervised reference rather than a same-protocol full-test baseline.
```

### 4.8 复现差异排查

针对“为什么与 ABC123 论文 FSC147 数字差很多，是否是我们复现有问题”，2026-07-09 做了逐项核查。结论是：目前没有发现我们 wrapper 的明显实现错误；更准确的判断是，官方仓库没有释放论文 FSC147 table 所需的完整评估 adapter / 权重 / split 口径，导致本地只能报告 “official README checkpoint reproduction”，不能声称复现了 paper FSC147 11.75。

| 排查项 | 结果 | 结论 |
|---|---|---|
| 官方 checkpoint | 使用 README 下载的 `/tmp/ABC123/checkpoints/model_chkpt.ckpt`；`epoch=86`，metadata `best val_DDP_MAE=8.9605` | 权重来源正确 |
| 权重加载 | 官方 `ABC123` 类与本地 wrapper 均为 `missing=0, unexpected=0` | 不是部分权重漏加载 |
| 模型结构 | 官方 `ABC123test.yml` 为 `counting_head=5_32`、`gtd_scale=400`、`upsample_padding_mode=replicate` | 与本地脚本一致 |
| counting head | 官方 `models/counting_head.py` 与本地 einops-free 实现在同一随机/真实特征上 `count max diff=0.0`、`density max diff=0.0` | head 实现等价 |
| 官方完整 forward | 直接实例化官方 `models/ABC123.py`，与本地 wrapper 在 `4756.jpg`、`1971.jpg` 上 `feature/count/density max diff=0.0` | 本地 wrapper 与官方 forward 完全一致 |
| 预处理 | 官方 `ToTensor()+Resize(224)` 与本地 PIL resize 后转 tensor，在 test 前 200 张上 mean abs pred delta：`sum=0.0679`、`max_density=0.0338` | resize 顺序差异可以忽略 |
| ImageNet normalization | 官方 `data.py` 没有对输入调用 Normalize；额外加 normalization 会使前 200 张 MAE 变差 | 不应加 ImageNet normalization |
| `gtd_scale` | 官方 test config 明确为 `400`；不除以 400 会让输出 count 放大 400 倍 | scale 处理正确 |
| 官方仓库支持 | `/tmp/ABC123/data.py` 只支持 MCAC/MCAC-M1；README 只提供 MCAC/MCAC-M1 testing 命令 | FSC147 评估 adapter 未公开 |
| FSC147 split 口径 | 本地 test 中 `GT>300` 为 `23/1190=1.93%`；论文文字称 test exclusion 为 `1.1%`，val exclusion `3.0%` 与本地 val `3.03%` 接近 | test split/版本口径可能不完全一致 |
| 论文可视化样例 | 15 张 FSC examples 均在本地 val split；论文视觉预测 MAE 约 `1.21`，本地 official checkpoint `sum` MAE 约 `28.63`、`max_density` MAE 约 `17.85` | 差距不是 full-test 高密度图或聚合指标导致 |

论文 FSC examples 的关键样例：

| Image | Paper GT | Paper pred | Local sum | Local max_density |
|---|---:|---:|---:|---:|
| `1971.jpg` | 19 | 19.5 | 67.9 | 26.0 |
| `3266.jpg` | 20 | 20.2 | 59.1 | 31.7 |
| `4756.jpg` | 14 | 14.7 | 2.8 | 2.0 |
| `7265.jpg` | 42 | 40.7 | 127.2 | 86.4 |

这组检查说明：即使不看 full test 的 `GT>300` 极端图，官方 README checkpoint 在论文展示的 val examples 上也不能复现论文预测。因此主要差异更可能来自以下几类：

1. 论文 FSC147 evaluation 使用了未公开的 sub-class combine / FSC adapter，而不是仓库 README 的 MCAC testing pipeline；
2. README example weights 可能不是生成 FSC147 table / figures 的同一 checkpoint；
3. FSC147/FSC133 数据版本或 test split 口径与本地 `Train_Test_Val_FSC_147.json` 有差异；
4. 旧环境差异例如 PyTorch 1.12 / timm 1.0.3 可能带来小数值变化，但无法解释 `11.75 -> 37.51` 这种 MAE 级别差距。

因此主文建议写成：

```text
We audited the released ABC123 checkpoint and code path. The local wrapper exactly matches the official forward pass and uses the official test configuration, but the released repository does not contain the FSC147 evaluation adapter used for the published table. We therefore cite the published ABC123 FSC147 (<300) result separately and report our released-checkpoint rerun as a reproducibility audit, not as a successful reproduction of the paper number.
```

---

## 5. 多数据集全量评估

| Dataset | Split / Protocol | 图像数 | ABC123 sum MAE/RMSE | ABC123 max_density MAE/RMSE | 主要现象 |
|---|---|---:|---:|---:|---|
| FSC147 | test full | 1,190 | 48.19 / 151.11 | 43.89 / 150.71 | 高密度图极大拉高 RMSE |
| FSC147 | test GT<=300 | 1,167 | 37.51 / 58.96 | 32.63 / 55.48 | 未复现论文 11.75 |
| CARPK | test full | 459 | 94.11 / 100.63 | 97.13 / 103.97 | 对俯拍停车场严重欠计数 |
| PUCPR+ | all 125 | 125 | 104.60 / 143.43 | 112.74 / 151.38 | 高密度车流欠计数严重 |
| PUCPR+ | test 25 | 25 | 125.13 / 167.10 | 132.61 / 174.35 | 标准 test split 很小 |
| OmniCount-191 | test full | 1,957 | 9.51 / 25.37 | 7.27 / 18.06 | total count 尚可，但无类别输出 |

结果文件：

```bash
result/logs/abc123_fsc147_test_full1190.json
result/logs/abc123_fsc147_test_le300.json
result/logs/abc123_carpk_test_full459.json
result/logs/abc123_pucpr_all125.json
result/logs/abc123_pucpr_test25.json
result/logs/abc123_omnicount_test_full1957.json
```

---

## 6. 与 OV-CUD 的对比定位

| Dataset | Method | 推理输入 | 训练监督 | 输出 | MAE | RMSE | 备注 |
|---|---|---|---|---|---:|---:|---|
| FSC147 test full | OV-CUD | image only | instance/category | count + class-aware candidates | **12.74** | **106.20** | full 1,190 |
| FSC147 test full | ABC123 local sum | image only | density map/count | total count only | 48.19 | 151.11 | official checkpoint reproduction |
| CARPK test | OV-CUD | image only | instance/category | count + candidates | **4.06** | **5.51** | zero-shot transfer |
| CARPK test | ABC123 local sum | image only | density map/count | total count only | 94.11 | 100.63 | severe undercount |
| OmniCount-191 test | OV-CUD predicted groups | image only | instance/category | per-class counts | **4.68** | **8.46** | mRMSE=0.457 |
| OmniCount-191 test | ABC123 local max_density | image only | density map/count | total count only | 7.27 | 18.06 | no per-class output |

ABC123 的价值是证明 “不需要 prompt 的 dense-map regression” 方向存在；但它不回答我们的核心 claim：

1. **dense map-free**：ABC123 训练和输出都依赖 density map；
2. **multi-category semantic output**：ABC123 的 heads 没有类别名，无法输出 OmniCount per-class count；
3. **跨数据集实例泛化**：在 CARPK/PUCPR+ 上 official checkpoint 欠计数明显。

---

## 7. 论文写作建议

建议把 ABC123 放到 baseline 表的 “Prompt-free but dense-map-supervised” 组，而不是放到 few-shot 或同范式组。

推荐表述：

```text
ABC123 is an exemplar-free/prompt-free dense regression counter. It predicts multiple density maps and uses density-map supervision, while OV-CUD is dense-map-free and produces semantic class-wise counts from image-only input. We cite ABC123's published FSC147 (<300 objects) numbers separately, but our local reproduction with the official released checkpoint does not match those numbers; therefore we report the official-checkpoint reproduction as an implementation audit rather than a claimed paper reproduction.
```

不建议写：

```text
ABC123 is a 3-shot method.
ABC123 is dense-map-free.
We reproduced ABC123 FSC147 11.75.
ABC123 and OV-CUD are the same protocol.
```

---

## 8. 命令记录

```bash
python3 script/eval_abc123_baseline.py \
  --dataset fsc147 --split test \
  --batch-size 64 --device cuda \
  --out result/logs/abc123_fsc147_test_full1190.json

python3 script/eval_abc123_baseline.py \
  --dataset fsc147 --split test --exclude-count-over 300 \
  --batch-size 64 --device cuda \
  --out result/logs/abc123_fsc147_test_le300.json

python3 script/eval_abc123_baseline.py \
  --dataset carpk --split test \
  --batch-size 64 --device cuda \
  --out result/logs/abc123_carpk_test_full459.json

python3 script/eval_abc123_baseline.py \
  --dataset pucpr --split all \
  --batch-size 64 --device cuda \
  --out result/logs/abc123_pucpr_all125.json

python3 script/eval_abc123_baseline.py \
  --dataset omnicount --split test \
  --batch-size 64 --device cuda \
  --out result/logs/abc123_omnicount_test_full1957.json
```
