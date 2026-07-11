# OV-CUD AAAI 2026 完整实验报告（审计版）

**最后更新**：2026-07-11

**方法**：OV-CUD / UniCounting

**当前 FSC-147 主协议**：strict no-GT、official-train-only、FSC dot-supervised、COCO relation pretraining-free

> **主结果更正**：历史 `12.67 / 113.71` 使用了 test GT-derived candidate `valid`、GT count 分桶形成的 cache 路由，并且旧 category/relation 训练 cache 混入 official test。该数字不能继续作为 no-GT 主结果。当前通过实现审计和 full-1,190 重跑的主结果是 **MAE=26.50 / RMSE=129.69**（relation seed 73，仅由 validation 预选）。

## 1. 执行摘要

### 1.1 当前可报告结果

| 数据集 / 记录 | 协议 | MAE | RMSE | 状态 |
|---|---|---:|---:|---|
| **FSC-147 test 1,190，primary seed 73** | **strict no-GT，FSC train-only，dot-supervised，CP-free** | **26.50** | **129.69** | 当前主结果 |
| FSC-147，3 relation seeds | 同上；只改变 relation 初始化 | 28.00 ± 1.41 | 129.59 ± 0.16 | 稳定性补充 |
| FSC-147 historical MR+T4 | GT-assisted / split-contaminated | 12.67 | 113.71 | 只能作历史诊断 |
| FSC-147 historical 1,189 | 缺失 `7611.jpg` | 12.74 | 106.20 | 无效主结果 |
| CARPK test 459 | 旧 FSC/COCO checkpoint，CARPK no-GT | 4.06 | 5.51 | 尚未用新 strict checkpoint 重跑 |
| PUCPR+ test 25 | 旧 checkpoint，2x2 tiled | 3.59 | 5.43 | 尚未用新 strict checkpoint 重跑 |
| OmniCount-191 full 1,957 | 旧 checkpoint，prompt-free total count | 6.75 | 10.24 | 不能称为与新 FSC 主模型相同 |
| MCAC full 2,115 | 旧 checkpoint，strict no-MCAC-GT inference | 32.11 | 53.79 | 不能称为与新 FSC 主模型相同 |
| ABC123 MCAC official ckpt | density-supervised，官方 matching | 9.46 | 17.52 | 复现 baseline |

跨数据集数字本身仍可作为各自历史实验记录，但在用本轮 train-only、CP-free checkpoint 重跑之前，不能与 FSC-147 26.50 合并声称“同一模型跨数据集泛化”。

### 1.2 当前允许的 claim

1. 推理时不接收 exemplar、per-image class name prompt 或 test GT。
2. 不使用 density map；relation、candidate filter 使用 official-train FSC dots，category head 使用 official-train image-level 类别。
3. FSC strict 主配置不使用 COCO relation initialization，relation head 从随机初始化训练。
4. validation/test inference cache 只含 image-derived `z/bbox/height/width` 等安全字段。
5. 当前 FSC 主结果覆盖 official test 全部 1,190 张，包括 `7611.jpg`。

### 1.3 当前禁止的 claim

1. 不能再写“FSC-147 MAE=12.67 的 strict prompt-free 主结果”。
2. 不能写 `count-supervision-free`。点标注的数量本身携带 count information，准确表述是 **point/dot-supervised、density-map-free**。
3. 不能用本轮 FSC 结果声称正确输出 held-out test 类别名。严格词表只有 official-train 的 89 类，FSC test 类别不在词表中。
4. 不能声称 strict 模型学习了 part-whole relation。当前推理只使用 instance branch，`A_part` 固定为 0。
5. 不能把 26.50 与 12.67 的差值解释为移除 COCO 的代价；两次协议同时改变了多个泄露相关组件。
6. 不能把本次 FSC test 称为研究过程中的 untouched blind test。项目此前已反复查看该 test，尤其 `7611` 曾启发 rescue 设计。

## 2. 原 12.67 实现审计

| 旧组件 | 问题 | 对预测的影响 | 新处理 |
|---|---|---|---|
| cache `valid` gate | `valid` 由候选是否覆盖 test GT dot 得到 | test GT 直接决定候选保留 | 删除字段；改为 train-dot-supervised MLP |
| MR51/MR100 membership | cache 集合由 test GT count bin 建立 | GT 隐式决定前端路由 | 对全部图生成 fast/tiled；route 只在 val 选择 |
| category heads | 旧 cache 全目录随机切分 | official-test 图进入训练 | 强制 official train 文件白名单 |
| relation heads | 旧 cache overlap + COCO 初始化 | 不满足 train-only、CP-free | FSC dots from scratch，3 seeds |
| 147 类 prototype bank | 含 val/test 独占类名 | 提前访问 held-out label space | 只编码 official-train 89 类 |
| `7611` T4 | recipe 受 test failure 启发 | 存在 researcher-level test familiarity | 仅用 train fast-zero 样本选 2x2/3x3/4x4；同时披露历史 |

旧 `12.67` 的 trigger `fast n_candidates==0` 本身是 image-only，但其余 candidate gate、路由和训练路径不满足 no-GT。仅披露“trigger 不读 GT”不足以挽救整个协议。

## 3. Strict 主方法实现

### 3.1 推理链

```text
Image
  -> SAM2 AMG, pts32 2x2 overlap=0.25 tiled candidates
  -> DINOv2 three-view 1152-d features
  -> train-only candidate-validity MLP
  -> train-89 CosineCategoryHead + fixed OpenCLIP prototypes
  -> category-aware spatial grouping
  -> FSC-dot-supervised scratch relation instance branch
  -> same-instance deduplication
  -> total count
```

pts16 fast cache只用于检测 raw candidate count 是否为 0。Validation 选择了 `always_tiled`，因此其余 1,189 张使用统一 pts32 2x2 前端；`7611.jpg` raw-fast-zero，使用 train-only 选出的 pts32 4x4 rescue。

### 3.2 安全 cache schema

唯一允许字段：

```text
schema, img_id, file_name, z, bbox, height, width, source_cache
```

显式拒绝：

```text
valid, purity, coverage, iou, matched_class, matched_instance_id,
gt_count, class_name, is_part, is_countable, points
```

加载器还检查内部 ID 与文件名一致、图像尺寸为正、`z/bbox` shape 合法、tensor 不含 NaN/Inf，并要求 cache 文件集合与 split 精确相等。

### 3.3 学习组件

| 组件 | 监督与范围 | 结构 / 关键配置 |
|---|---|---|
| Text prototypes | official-train 89 类名；val/test 类名 0 | OpenCLIP ViT-B-32，LAION 权重，5 templates |
| Category pts16/pts32 | official-train image class | 1152->512，2-layer MLP，dropout 0.3，40 epochs，seed 42 |
| Candidate filter pts16/pts32 | 候选是否覆盖 official-train dot | 1159->256->1，dropout 0.2，seed 20260711 |
| Relation pts16/pts32 | 两候选是否分配到同一 official-train dot | hidden 512，3 layers，dropout 0.1，pos_weight 8，40 epochs |
| Relation initialization | 无 COCO | seeds 17/42/73，`pretrained_from=None` |

Relation 训练固定 `max_cand=64`、`max_pairs=4096`、`neg_ratio=5`、AdamW `lr=5e-4`、internal train-val split seed `20260710`。pts16/pts32 dot label 审计分别有 7,380/37,802 个 same-dot positive pairs，非法 ID 均为 0。

### 3.4 Validation-only 参数

| 项目 | 选择结果 |
|---|---|
| fast gate | filter=0.05，category=0，`tau_inst=0.99` |
| tiled gate | filter=0.3，category=0.4，`tau_inst=0.99` |
| grouping | `tau_affinity=0.1`，`max_group_size=30` |
| relation pairs | ranking top 200 candidates |
| route | `always_tiled` |
| primary relation seed | 73，按 val MAE、再 RMSE 预选 |
| fast-zero rescue | train-only 选择的 4x4，overlap=0.25 |

冻结 validation 文件 SHA-256：`8c4f0c61cdf74221470cd41fcc58837be7f39ba37b5f08b50d303b9b3ab5ee16`。测试计划 SHA-256：`10b4a4990da95e9c591dceffb4aa0c5c2a5c09bc834f1e17dfb90df0106a6d8d`。

## 4. Validation 结果

| Route | Val MAE mean | Val RMSE mean | Tiled images |
|---|---:|---:|---:|
| always fast | 36.71 | 121.06 | 0 |
| predicted fast count >=10 | 31.33 | 105.84 | 1,124-1,127 |
| **always tiled** | **30.31** | **102.37** | **1,286** |

| Relation seed | Val MAE | Val RMSE | Bias |
|---:|---:|---:|---:|
| 17 | 30.61 | 102.37 | -13.19 |
| 42 | 30.31 | **102.19** | -13.72 |
| **73** | **30.00** | 102.53 | -15.24 |

Validation 只在全部预测生成后读取物理隔离的 val count shard；该进程不解析包含 test GT 的合并 annotation。

## 5. FSC-147 Full Test 1,190

### 5.1 正式结果

| Relation seed | MAE | RMSE | Bias | MAE bootstrap 95% CI |
|---:|---:|---:|---:|---:|
| 17 | 29.3067 | 129.6714 | +1.0529 | [23.5737, 37.6704] |
| 42 | 28.1882 | **129.4034** | -0.1815 | [22.5041, 36.5172] |
| **73，primary** | **26.4992** | 129.6856 | -3.1748 | [20.7099, 35.2550] |
| **Mean ± sample std** | **27.9980 ± 1.4134** | **129.5868 ± 0.1590** | - | - |

结果 JSON：`result/logs/fsc147_strict_nogt_cp_free_full1190.json`

SHA-256：`a455706c602a3b386488983b94db394e7ba3dd65ca7536aac535aff2121a184c`

### 5.2 Primary seed 分桶

| GT bin | Images | MAE | RMSE | Bias |
|---|---:|---:|---:|---:|
| 0-10 | 60 | 11.08 | 15.81 | +10.22 |
| 11-20 | 268 | 10.73 | 16.35 | +9.65 |
| 21-50 | 413 | 14.53 | 19.52 | +10.76 |
| 51-100 | 254 | 22.99 | 29.63 | +2.29 |
| 100+ | 195 | **82.84** | **316.61** | **-61.55** |

低/中密度明显过计数，高密度明显漏计。Top 1% 图像贡献 27.29% 总绝对误差，Top 5% 贡献 42.64%。

### 5.3 关键失败样本

| Image | GT | Pred | Error | 原因 |
|---|---:|---:|---:|---|
| `1123.jpg` | 3,701 | 140 | -3,561 | 2x2 只有 354 raw candidates，proposal recall 不足 |
| `7611.jpg` | 2,560 | 219 | -2,341 | 4x4 有 1,997 candidates，但联合 gate 只保留 246 |

`7611` 中 filter gate 单独通过 748，category gate 单独通过 584，联合只剩 246，seed 73 relation dedup 后为 219。去掉 `7611` 后 MAE=24.55 / RMSE=110.56；去掉两个最大 outlier 后 MAE=21.58 / RMSE=39.49。该诊断说明 proposal recall 与 train-only category/filter calibration 都是主要瓶颈。

## 6. 为什么不是 12.67

| 协议 | Candidate gate | Routing | Category / relation train split | Vocabulary | Relation init | MAE |
|---|---|---|---|---|---|---:|
| Historical 12.67 | test GT dot oracle | GT count cache bins | 有 official-test overlap | full 147 | COCO + FSC | 12.67 |
| Strict rerun | train-dot MLP prediction | val-frozen image-only | official train only | train 89 | scratch FSC dots | 26.50 |

因此 +13.83 MAE 是完整协议纠正后的差值，不是 CP leave-one-out。此前 official-train-only 的 CP/scratch 三 seed配对差异接近零，已经说明 COCO 初始化不是旧 12.67 的主要来源。若要量化每个修正项，只能在当前 safe cache、train-only data 和 validation-frozen threshold 下逐项重跑；不能恢复 test `valid` 后把结果称为消融。

## 7. 消融实验状态

旧 FSC M1-M6/A1-A8 表依赖 GT-derived `valid` 或旧 checkpoint，不能再进入主文因果表。当前状态：

| 消融 | 当前是否有 strict full-1,190 证据 | 后续要求 |
|---|---|---|
| Relation head -> box-IoU | 否 | 新 safe cache、seed73；阈值只用 val |
| Candidate filter / ADF | 否 | 只比较 train-dot MLP 与预注册 image-only gate |
| Category grouping | 否 | no grouping / train-89 grouping；不可用 test class |
| pts32 2x2 tiling | Val 有 route 证据，test 无严格 LOO | always-fast 与 always-tiled 配置冻结后一次评测 |
| 4x4 fast-zero rescue | test 有单样本主结果，非严格 blind | 报 trigger coverage，并在新未触碰数据集复核 |
| COCO pretraining | 已完成配对实验 | 从主组件表删除，不再作为贡献 |

在这些 strict LOO 完成前，论文 `tab:ablation` 应留空或明确标注 historical GT-assisted，不能沿用 12.67 表格。

## 8. 多类别与跨数据集实验

OmniCount、MCAC、CARPK、PUCPR+ 的已有报告仍保留：

- `docs/omnicount_multiclass_ablation_report_20260708.md`
- `docs/mcac_full2115_leaveoneout_report_20260710.md`
- `docs/abc123_mcac_reproduction_report_20260710.md`
- `docs/carpk_full459_rerun_20260709.md`

但它们使用旧 FSC/COCO checkpoint。若论文要写“与 FSC strict 主结果同一模型”，必须用本轮以下资产重跑：train-89 category heads、train-only candidate filters、scratch relation seed73，以及各数据集 image-only safe cache。MCAC/OmniCount 的多类别匹配协议也必须保持各自报告中的 Hungarian/per-class 定义，不能与 FSC single-class total count 混表。

## 9. 论文表格调整

1. FSC 主表将 OV-CUD 改为 `26.50 / 129.69`，监督列写 `FSC points + image labels`，输入列写 `image only`。
2. `12.67 / 113.71` 移到附录 protocol audit，明确 `GT-assisted candidate oracle`，不与 baseline 排名。
3. 删除 `count-supervision-free`、learned part-whole 和 FSC held-out class-name output 的表述。
4. CP 不再出现在组件贡献表；可在附录报告配对 null result。
5. 旧 CARPK/PUCPR+/OmniCount/MCAC 数字标为 legacy checkpoint，待 strict checkpoint 重跑后再恢复 same-model claim。
6. FSC test familiarity 作为 limitation 披露；新增 benchmark 或未触碰 split 用于最终确认。

## 10. 复现与资产

关键入口：

- `script/build_fsc147_train_vocabulary.py`
- `script/train_category_v2.py`
- `script/train_candidate_filter.py`
- `script/train_relation_1152.py`
- `script/export_fsc147_inference_cache.py`
- `script/preprocess_fsc147_tiled_nogt.py`
- `script/select_fsc147_train_rescue.py`
- `script/eval_fsc147_strict_nogt.py`

正式 test 命令：

```bash
python3 script/eval_fsc147_strict_nogt.py test \
  --frozen-config result/logs/fsc147_strict_nogt_val_selection.json \
  --tile-plan result/configs/fsc147_strict_nogt_test_tile_plan.json \
  --test-tiled-cache /home/czp/ws_yiyang/ovcud_cache/fsc147_nogt_test_tiled2x2 \
  --test-rescue-cache /home/czp/ws_yiyang/ovcud_cache/fsc147_nogt_test_rescue4x4_trainselected \
  --out result/logs/fsc147_strict_nogt_cp_free_full1190.json \
  --device cuda --bootstrap 5000
```

完整模型 SHA、代码 SHA、validation grid、route sweep 和逐图结果分别保存在：

- `result/logs/fsc147_strict_nogt_val_selection.json`
- `result/configs/fsc147_strict_nogt_test_tile_plan.json`
- `result/logs/fsc147_strict_nogt_cp_free_full1190.json`
- `docs/fsc147_strict_nogt_cp_free_report_20260710.md`

## 11. 当前检查清单

- [x] 删除 validation/test inference cache 中所有 GT 字段。
- [x] official-train-only 89 类 prototype、category、candidate filter。
- [x] pts16/pts32 FSC-dot scratch relation，各 3 seeds，无 COCO 初始化。
- [x] 参数、route、primary seed 只由 official validation 选择并冻结。
- [x] full test 1,190 image-only cache 完整审计。
- [x] 所有 1,190 预测生成后才读取 test annotation。
- [x] 逐 seed MAE/RMSE、bootstrap、分桶和失败样本记录。
- [x] 撤销 12.67 的 no-GT 主结果地位。
- [ ] 用新 strict checkpoint 重跑跨数据集与多类别主表。
- [ ] 在当前 strict 协议下重跑 leave-one-out 组件消融。
- [ ] 在未触碰的新 benchmark 上确认 rescue 与高密度结论。
