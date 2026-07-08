# ABC123 复现实验记录

**日期**: 2026-07-08
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
