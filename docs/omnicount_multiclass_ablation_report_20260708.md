# OmniCount-191 多类别消融实验记录

> **2026-07-11 状态更正**：本文 OmniCount 数字来自历史 FSC/COCO checkpoint，不再能称为与当前 strict FSC-147 CP-free 主模型相同。实验数值保留，但 same-model claim 需等待新 checkpoint full-1,957 重跑。

**日期**: 2026-07-08
**分支**: `ljs`
**目标**: 使用当前 FSC147 主结果 MAE=12.67 的同一组 OV-CUD 学习模型，在 OmniCount-191 上验证 prompt-free 多类别计数能力，并补充 published prompted methods 与本地 OWLv2 class-aware baseline。

> **2026-07-10 更新**：12.67 与原 12.74 使用同一组学习权重，差别是 FSC147 完整 1,190 张加入 T4 rescue。OmniCount M1-M6 full leave-one-out 见 `docs/fsc147_omnicount_leaveoneout_report_20260710.md`。

---

## 1. 结论摘要

这组 OmniCount-191 实验是 **全量 test cache 1,957 张图像**，不是 500 张子集。500 张仅用于脚本调试和 OWLv2 阈值选择。

最重要结论：

1. **同一 FSC147 学习模型可以迁移到 OmniCount-191 多类别场景**。只替换 OmniCount 的 93 类文本原型，不重新训练模型。
2. **OV-CUD prompt-free predicted class groups** 在 1,957 张图上达到 **MAE=4.68 / RMSE=8.46 / mRMSE=0.457 / mRMSE-nz=3.911**。
3. 该结果优于 OV-CUD class-agnostic total count 的 **MAE=6.77 / RMSE=10.23**，说明输出类别组不是摆设。
4. 本地补充的 **OWLv2 class-aware** 使用每张图 GT class list 作为 text prompts，在同一批 1,957 张图上达到 **MAE=4.87 / RMSE=9.65 / mRMSE=0.369 / mRMSE-nz=3.657**。
5. 本地补充的 **ABC123 official checkpoint** 是 prompt-free dense-map regression baseline，在 OmniCount-191 total count 上 `max_density` 达到 **MAE=7.27 / RMSE=18.06**，但不能输出类别名或 per-class counts。
6. OmniCount 论文 published methods 是 **class-name prompted** 协议，与 OV-CUD prompt-free 不是同协议；可作为参考表，但不能直接当公平主比较。

---

## 2. 模型一致性审计

OmniCount 多类别实验使用的学习模型与 FSC147 当前主结果 (MAE=12.67) 属于同一组模型：

| 模块 | FSC147 12.67 设置 | OmniCount 多类别设置 | 是否一致 |
|---|---|---|---|
| Category head | `result/checkpoints/category_cosine_pts32.pt` | `result/checkpoints/category_cosine_pts32.pt` | 是 |
| Relation head | `result/checkpoints/fsc147_relation_pts32_best.pt` / `fsc147_relation_pts32_exp5c.pt` | `result/checkpoints/fsc147_relation_pts32_best.pt` | 是 |
| Relation 权重 | Exp5-C epoch 25 | Exp5-C epoch 25 | 是 |
| 候选策略 | FSC147 高密度图使用 multi-resolution cache | OmniCount pts=32 cache | 数据集相关候选缓存 |
| 文本原型 | FSC147 类别原型 | OmniCount-191 93 类原型 | 只替换词表 |

Checkpoint 审计：

| 文件 | SHA256 前缀 | 说明 |
|---|---:|---|
| `category_cosine_pts32.pt` | `43509c6e4ff55498` | 与 FSC147 主结果相同的类别投影头 |
| `fsc147_relation_pts32_best.pt` | `1bb7a1cfd950fde8` | 推理版关系头 |
| `fsc147_relation_pts32_exp5c.pt` | `c0e16479f7d6cc4a` | 训练版关系头，含 optimizer |
| `text_prototypes_omnicount.pt` | `26cd89b7f596a230` | OmniCount 93 类文本原型 |

`fsc147_relation_pts32_best.pt` 与 `fsc147_relation_pts32_exp5c.pt` 的 `relation_head` state dict 完全一致：`max_abs_diff=0.0`，epoch 均为 25。

需要写清楚的 nuance：

- 这里的“同一模型”指 **同一个 category projection head + 同一个 relation head 权重**。
- OmniCount 必须替换文本原型矩阵，因为它是开放词表评估；这不是重新训练模型。
- FSC147 MAE=12.67 使用 FSC147 专用 multi-resolution + T4 rescue；OmniCount 当前使用 pts32 cache，且 T4 trigger 命中 0/1,957。因此候选生成策略不是完全相同，但学习模型相同。

---

## 3. 数据与评估脚本

新增脚本：

```bash
script/eval_omnicount_multiclass_ablation.py
```

该脚本从 OmniCount 原始 COCO-format annotations 重建 per-class GT，而不是直接使用缓存里的 `matched_class`。原因是当前 OmniCount cache 中：

```python
"matched_class": torch.zeros(n_cand, dtype=torch.long)  # class-agnostic
```

预测分支不使用 GT-derived `valid` masks，避免 prompt-free 推理泄漏。Oracle 分支只在上限诊断中解码 SAM mask，并用 GT dot center 做候选-类别匹配。

主命令：

```bash
python3 script/eval_omnicount_multiclass_ablation.py \
  --min-classes 1 \
  --limit -1 \
  --category-ckpt result/checkpoints/category_cosine_pts32.pt \
  --relation-ckpt result/checkpoints/fsc147_relation_pts32_best.pt \
  --text-prototypes result/checkpoints/text_prototypes_omnicount.pt \
  --class-names result/checkpoints/omnicount_class_names.json \
  --conf-threshold 0.1 \
  --out result/logs/omnicount_multiclass_ablation_full1957_fsc147head_conf01.json \
  --device cuda \
  --save-per-image
```

主结果文件：

```bash
result/logs/omnicount_multiclass_ablation_full1957_fsc147head_conf01.json
```

数据统计：

| 项目 | 数值 |
|---|---:|
| 图像数 | 1,957 |
| 测试集中出现的类别数 | 93 |
| 平均 GT count / image | 6.03 |
| 单类别图像 | 1,124 |
| 2+ 类别图像 | 833 |
| 多类别图像比例 | 42.6% |

---

## 4. OV-CUD 协议消融

| Protocol | 推理时提示 | 是否输出 per-class | MAE | RMSE | bias | mRMSE | mRMSE-nz |
|---|---|---:|---:|---:|---:|---:|---:|
| SAM2-only total | 无 | 否 | 37.60 | 44.82 | +37.60 | - | - |
| OV-CUD class-agnostic total | 无 | 否 | 6.77 | 10.23 | +5.47 | - | - |
| OV-CUD predicted class groups | 无 | 是 | **4.68** | **8.46** | -4.36 | 0.457 | 3.911 |
| OV-CUD oracle class grouping | 仅 oracle 诊断 | 是 | **1.12** | **2.22** | -0.65 | 0.111 | 1.047 |

关键结论：

- prompt-free predicted class groups 明显优于 class-agnostic total：**6.77 -> 4.68**。
- Oracle class grouping 上限很高：**4.68 -> 1.12**，说明主要瓶颈是候选类别分配与候选覆盖，而不是“能不能做多类别输出”。
- 这是目前最适合作为主文的 OmniCount 证据：OV-CUD 不接收类别提示，也能输出类别名和对应计数。

---

## 5. 类别分离消融

| Variant | MAE | RMSE | bias | mRMSE | mRMSE-nz |
|---|---:|---:|---:|---:|---:|
| Global, no class grouping | 4.69 | 8.49 | -4.39 | 0.455 | 3.911 |
| `p_i · p_j` grouping | **4.68** | **8.46** | -4.36 | 0.457 | 3.911 |
| Category bucket | 4.69 | 8.48 | -4.38 | 0.455 | 3.911 |
| Full category + relation grouping | **4.68** | **8.46** | -4.36 | 0.457 | 3.911 |
| Oracle category grouping | **1.12** | **2.22** | -0.65 | 0.111 | 1.047 |

解释：

- 在 `conf_threshold=0.1` 下，几种 predicted grouping 变体差异很小。
- Oracle category grouping 大幅提升，说明真正的瓶颈是类别 assignment 和候选覆盖，而不是 bucket / `p_i·p_j` 这类聚类细节。
- 论文中不建议把 OmniCount 作为 relation/category grouping 结构贡献的主要证据；它更适合证明 prompt-free per-class 输出。

---

## 6. 去重消融

| Variant | MAE | RMSE | bias | mRMSE | mRMSE-nz |
|---|---:|---:|---:|---:|---:|
| SAM2-only total | 37.60 | 44.82 | +37.60 | - | - |
| Category-only | **4.68** | **8.46** | -4.36 | 0.457 | 3.911 |
| Category + IoU NMS | 4.69 | 8.49 | -4.39 | 0.455 | 3.910 |
| Category + relation head | **4.68** | **8.46** | -4.36 | 0.457 | 3.911 |
| Relation + adaptive area filter | 4.72 | 8.48 | -4.40 | 0.455 | 3.911 |

解释：

- OmniCount 上从 SAM2-only 到 OV-CUD 的主要收益来自类别过滤和候选筛选。
- 当前阈值已经过滤掉大量重复候选，因此 IoU NMS 与 relation-head dedup 数字接近。
- Relation head 的核心证据仍应放在 FSC147/CARPK 的主消融，不建议在 OmniCount 上强行声称 relation-head 显著贡献。

---

## 7. 难度切片

### 7.1 按每图类别数

OV-CUD predicted class groups：

| GT 类别数 / 图 | 图像数 | MAE | RMSE | bias |
|---|---:|---:|---:|---:|
| 1 | 1,124 | 4.49 | 10.02 | -4.04 |
| 2 | 311 | 4.17 | 5.02 | -3.88 |
| 3 | 186 | 6.02 | 7.12 | -5.86 |
| 4+ | 336 | 5.03 | 5.45 | -5.03 |

2+ 类别图像：

| Protocol | 图像数 | MAE | RMSE | bias |
|---|---:|---:|---:|---:|
| Class-agnostic total | 833 | 4.98 | 8.09 | +3.52 |
| Predicted class groups | 833 | **4.93** | **5.72** | -4.78 |
| Oracle class grouping | 833 | **1.30** | **1.90** | -0.80 |

### 7.2 按 super-category

OV-CUD predicted class groups：

| Super-category | 图像数 | MAE | RMSE | bias |
|---|---:|---:|---:|---:|
| Birds | 10 | 14.30 | 14.56 | -14.30 |
| Fruits | 303 | 4.27 | 4.38 | -4.27 |
| Pets | 11 | 8.45 | 9.06 | -8.45 |
| Satellite | 127 | **1.92** | 3.70 | -0.82 |
| Supermarket | 251 | 11.62 | 20.31 | -11.41 |
| Urban | 1,000 | 3.82 | 4.84 | -3.42 |
| Wild | 255 | 2.50 | 3.24 | -2.37 |

Supermarket 和 Birds 最难。Supermarket 有大量细粒度商品类别且密度高；Birds 图像数少但每图计数较高。

### 7.3 按 GT count bin

OV-CUD predicted class groups：

| GT count bin | 图像数 | MAE | RMSE | bias |
|---|---:|---:|---:|---:|
| 0-10 | 1,772 | 3.15 | 3.81 | -2.79 |
| 11-20 | 123 | 10.97 | 11.51 | -10.97 |
| 21-50 | 42 | 26.38 | 27.44 | -26.38 |
| 51-100 | 20 | 55.80 | 57.67 | -55.80 |

高密度图像明显欠计数，这与 FSC147/CARPK/PUCPR+ 的结论一致：候选密度和小目标覆盖仍是主要瓶颈。

---

## 8. OmniCount 论文 published prompted table

来源：

- AAAI 2025 论文页: https://ojs.aaai.org/index.php/AAAI/article/view/34151
- arXiv 源文件: `arxiv_aaai.tex` 中 Table 1 (`tex_aaai/05_expt.tex`)

这些方法均为 **class-name prompted, not same protocol**，不是 OV-CUD 的 prompt-free 协议。它们可以作为参考表，不应作为同协议公平主表。

| Method | 输入协议 | OmniCount-191 mRMSE | OmniCount-191 mRMSE-nz | 协议标注 |
|---|---|---:|---:|---|
| Grounding-DINO | class names as text prompts | 1.29 | 3.27 | class-name prompted, not same protocol |
| CLIPSeg | class names as text prompts | 1.54 | 4.28 | class-name prompted, not same protocol |
| TFOC | class names / text prompt | 0.95 | 2.89 | class-name prompted, not same protocol |
| OmniCount | class names + semantic-geometric priors | **0.70** | **2.00** | class-name prompted, not same protocol |
| OV-CUD predicted class groups | image only, no class prompt | 0.457 | 3.911 | prompt-free, our stricter protocol |

解读：

- OV-CUD 的 `mRMSE=0.457` 低于 published prompted methods，但这个指标会被大量 GT=0 的类别稀释，不能单独用于强比较。
- `mRMSE-nz=3.911` 更能反映“已出现类别”的误差。它弱于 OmniCount published `2.00`，接近 / 略弱于 Grounding-DINO `3.27` 和 TFOC `2.89`，但我们的输入更严格：不提供类别名。
- 推荐写法：**OV-CUD 在无类别提示的条件下提供 per-class counts；prompted methods 仍在已知类别计数上更强。**

---

## 9. 本地 OWLv2 class-aware baseline

脚本：

```bash
script/run_omnicount_baselines.py
```

为本实验补充了 per-class 保存与 `mRMSE / mRMSE-nz` 统计。

阈值选择：在前 100 张上扫 `conf_threshold`：

| OWLv2 conf | 100 张 MAE | 100 张 RMSE | mRMSE | mRMSE-nz |
|---:|---:|---:|---:|---:|
| 0.1 | 12.75 | 27.36 | 3.341 | 9.177 |
| 0.3 | 2.72 | 6.16 | 0.961 | 2.555 |
| 0.5 | **2.59** | **3.88** | **0.644** | **1.564** |
| 0.7 | 4.33 | 6.05 | 0.925 | 2.245 |

最终 full run 使用 `conf_threshold=0.5`。

命令：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python3 script/run_omnicount_baselines.py \
  --mode class-aware \
  --limit -1 \
  --conf-threshold 0.5 \
  --out result/logs/omnicount_owlv2_classaware_full1957_conf05.json \
  --device cuda
```

结果文件：

```bash
result/logs/omnicount_owlv2_classaware_full1957_conf05.json
```

### 9.1 OWLv2 class-aware 全量结果

| Method | 输入协议 | 图像数 | MAE | RMSE | bias | mRMSE | mRMSE-nz |
|---|---|---:|---:|---:|---:|---:|---:|
| OWLv2 class-aware | 每张图 GT class list 作为 text prompts | 1,957 | 4.87 | 9.65 | -4.78 | **0.369** | **3.657** |
| OV-CUD predicted class groups | image only，无 class prompt | 1,957 | **4.68** | **8.46** | -4.36 | 0.457 | 3.911 |

解读：

- OWLv2 使用 GT class list，输入更强；OV-CUD 完全不接收类别提示。
- 在更强输入下，OWLv2 的 mRMSE / mRMSE-nz 略优于 OV-CUD，但 total MAE / RMSE 略弱于 OV-CUD。
- 这张表非常适合回答 reviewer：“如果用 open-vocabulary detector，并给它类别提示，效果如何？”

### 9.2 OWLv2 按 super-category

| Super-category | 图像数 | MAE | RMSE | bias |
|---|---:|---:|---:|---:|
| Birds | 10 | 10.40 | 10.90 | -10.40 |
| Fruits | 303 | 1.53 | 1.74 | -1.53 |
| Pets | 11 | 6.27 | 7.40 | -6.27 |
| Satellite | 127 | 2.22 | 4.25 | -2.22 |
| Supermarket | 251 | 13.75 | 24.00 | -13.24 |
| Urban | 1,000 | 4.61 | 5.50 | -4.59 |
| Wild | 255 | 2.17 | 3.08 | -2.08 |

### 9.3 OWLv2 按 GT count bin

| GT count bin | 图像数 | MAE | RMSE | bias |
|---|---:|---:|---:|---:|
| 0-10 | 1,772 | 3.06 | 3.90 | -2.97 |
| 11-20 | 123 | 11.41 | 12.00 | -11.39 |
| 21-50 | 42 | 33.76 | 34.77 | -33.76 |
| 51-100 | 20 | 64.50 | 65.86 | -64.50 |

OWLv2 与 OV-CUD 都在高密度 bin 明显欠计数，说明检测式/候选式方法在密集场景仍受 proposal recall 和小目标可见性限制。

---

## 10. 推荐论文表格

### 10.1 同协议主表：Prompt-free / no class list

| Method / Protocol | 输入 | Class-wise output | MAE | RMSE | mRMSE |
|---|---|---:|---:|---:|---:|
| SAM2-only | Image only | 否 | 37.60 | 44.82 | - |
| ABC123 local max_density | Image only | 否 | 7.27 | 18.06 | - |
| OV-CUD class-agnostic | Image only | 否 | 6.77 | 10.23 | - |
| OV-CUD predicted class groups | Image only | 是 | **4.68** | **8.46** | 0.457 |
| OV-CUD oracle class grouping | Oracle eval only | 是 | 1.12 | 2.22 | 0.111 |

### 10.2 参考表：Prompted / class-name specified methods

| Method | 输入 | MAE | RMSE | mRMSE | mRMSE-nz | 说明 |
|---|---|---:|---:|---:|---:|---|
| Grounding-DINO | class names | - | - | 1.29 | 3.27 | OmniCount 论文 published |
| CLIPSeg | class names | - | - | 1.54 | 4.28 | OmniCount 论文 published |
| TFOC | class names / text | - | - | 0.95 | 2.89 | OmniCount 论文 published |
| OmniCount | class names + priors | - | - | **0.70** | **2.00** | OmniCount 论文 published |
| OWLv2 class-aware | GT class list | 4.87 | 9.65 | **0.369** | 3.657 | 本地同 1,957 图 rerun |
| ABC123 local max_density | image only | 7.27 | 18.06 | - | - | prompt-free，但 dense-map regression；无 per-class |
| OV-CUD predicted groups | image only | **4.68** | **8.46** | 0.457 | 3.911 | 无类别提示 |

推荐写法：

> We further compare OV-CUD with class-name prompted references on OmniCount-191. These methods receive the target class names or class list, whereas OV-CUD receives only the image. Despite this stricter protocol, OV-CUD achieves comparable total-count error and produces class-wise counts without prompts.

中文写作口径：

- 主文强调 **同协议 prompt-free 表**。
- prompted methods 放 “reference comparison” 或 appendix。
- 明确写 “not same protocol / stronger input”。
- 不要声称 OV-CUD 在 OmniCount 官方 prompted protocol 上 SOTA。

---

## 11. 局限与下一步

1. **高密度欠计数**：GT>20 的图像误差很大，OmniCount 可能需要复用 FSC147 的 multi-resolution / tiling 候选策略。
2. **类别分配瓶颈**：Oracle class grouping 从 MAE=4.68 降到 1.12，说明更好的类别校准或文本 prompt normalization 有明显空间。
3. **Dedup 贡献被压缩**：`conf=0.1` 已过滤大量重复候选，使 relation dedup 与 IoU NMS 差异很小。
4. **协议差异必须讲清楚**：published OmniCount methods 使用 class-name prompts；OV-CUD 是 image-only prompt-free。
5. **ABC123 不是同范式主 baseline**：它是 prompt-free，但需要 density map/count 监督且输出 dense maps；只能作为 dense-regression reference。

建议后续如有时间补一个实验：

```text
OmniCount 高密度 Supermarket / Birds 子集上跑 multi-resolution candidates，
验证 FSC147 12.74 的 proposal 策略是否也能缓解 OmniCount 欠计数。
```

---

## 12. 文件索引

| 文件 | 内容 |
|---|---|
| `script/eval_omnicount_multiclass_ablation.py` | OV-CUD OmniCount 多类别消融脚本 |
| `script/run_omnicount_baselines.py` | OWLv2 baseline 脚本，已补 mRMSE / mRMSE-nz |
| `script/eval_abc123_baseline.py` | ABC123 official checkpoint 多数据集评估脚本 |
| `result/logs/omnicount_multiclass_ablation_full1957_fsc147head_conf01.json` | OV-CUD 主结果 |
| `result/logs/omnicount_owlv2_classaware_full1957_conf05.json` | OWLv2 class-aware full result |
| `result/logs/abc123_omnicount_test_full1957.json` | ABC123 OmniCount total-count full result |
| `result/logs/omnicount_owlv2_classaware_100_conf05.json` | OWLv2 阈值选择记录 |
| `docs/omnicount_multiclass_ablation_report_20260708.md` | 本中文实验记录 |
