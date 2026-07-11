# Strict no-GT FSC 主模型 OmniCount-191 Full 复评报告

**日期**：2026-07-11
**分支**：`ljs`
**数据集**：OmniCount-191 test full，1,957 张
**主模型**：FSC-147 strict no-GT / official-train-only / FSC-dot-supervised / COCO relation pretraining-free
**FSC-147 对应结果**：MAE=26.4992 / RMSE=129.6856，primary relation seed 73

## 1. 结论摘要

使用与 FSC-147 strict 主结果完全相同的模型资产、FSC validation 冻结参数和 `always_tiled` 路由，在 OmniCount-191 全量 1,957 张 test 图像上得到：

| Relation seed | MAE | RMSE | Bias | Mean pred | Mean GT |
|---:|---:|---:|---:|---:|---:|
| 17 | 13.5927 | 21.0959 | +13.3209 | 19.3536 | 6.0327 |
| 42 | 13.3700 | 20.7325 | +13.0910 | 19.1237 | 6.0327 |
| **73，FSC-val primary** | **13.2575** | **20.5566** | **+12.9673** | **19.0000** | **6.0327** |
| **3 seeds mean ± sample std** | **13.4067 ± 0.1706** | **20.7950 ± 0.2750** | - | - | - |

Primary seed bootstrap 95% CI：MAE `[12.5764, 13.9612]`，RMSE `[19.1029, 22.0668]`，5,000 次重采样。

本轮最重要的结论不是“strict 模型在 OmniCount 上仍有旧报告的 4.68”，而是：

1. **当前 same-model strict 结果应改为 13.26 / 20.56**。
2. 模型在 1,722/1,957 张图上过计数，mean prediction 是 GT 均值的 3.15 倍，主要问题是明显的跨域正偏。
3. 历史 4.68 / 8.46 使用旧 FSC/COCO checkpoint、OmniCount-93 文本原型、full-image cache 和 `conf=0.1`，不能再称为与 FSC strict 26.50 相同模型/相同配置。
4. 当前固定词表只有 FSC official-train 89 类，不能在不引入 OmniCount 类表的情况下严谨计算 OmniCount 93 类的 mRMSE / mRMSE-nz。本报告只把 total count 和按 GT 难度的 post-prediction 切片作为正式结果。

## 2. 数据与 no-GT 协议

### 2.1 数据统计

| 项目 | 数值 |
|---|---:|
| Test images | 1,957 |
| GT classes | 93 |
| Mean GT count / image | 6.0327 |
| Single-class images | 1,124 |
| Multi-class images | 833 |
| GT class count 2 / 3 / 4+ | 311 / 186 / 336 |

`result/configs/omnicount_test_image_manifest_strict.json` 直接扫描七个 test 目录中的 JPG 文件得到；manifest 记录 `annotation_files_opened=0`、`ground_truth_values_retained=0`、`prediction_gt_fields=[]`。

### 2.2 预测与计分物理分离

本轮不是“同一循环中一边预测一边读 GT”：

1. `predict` 子进程的 CLI 没有 `--targets` 或 annotation 参数。
2. 该进程只读取 image-only manifest、safe cache、FSC-val frozen config 和 train-only checkpoint。
3. 所有 1,957 × 3 seed 预测先写入 `omnicount_strict_nogt_predictions_full1957.json`，其中 `target_file_loaded=false`。
4. 预测进程退出后，独立 `score` 子进程才读取 count target shard 并计算指标。

Prediction-only 每行字段严格为：

```text
sample_id, file_name, source, raw_fast_candidates,
raw_tiled_candidates, pred_count
```

不包含 `gt_count`、`class_counts`、`supercategory`、GT box/dot 或 candidate-validity 标签。

### 2.3 Safe cache 审计

唯一允许的 cache 字段：

```text
schema, img_id, file_name, z, bbox, height, width, source_cache
```

| Cache | Files | Total candidates | Mean | Median | P90 | Max | Zero |
|---|---:|---:|---:|---:|---:|---:|---:|
| pts16 fast，raw-zero trigger only | 1,957 | 52,079 | 26.61 | 23 | 43 | 137 | 0 |
| pts32 2×2，overlap=.25，bbox merge | 1,957 | 221,827 | 113.35 | 104 | 192.4 | 388 | 0 |

两套 cache 均通过 exact filename set、schema、ID、shape 和 finite tensor 检查。OmniCount 上 raw-fast-zero 为 0，因此没有图像触发 FSC train-only 选择的 4×4 rescue；全部 1,957 张都走 frozen `always_tiled` 2×2 前端。

## 3. 模型与冻结配置一致性

| 项目 | 配置 |
|---|---|
| Category head | `cp_strict/category_pts32_trainvocab.pt` |
| Candidate filter | `fsc147_candidate_filter_pts32_trainonly.pt` |
| Relation head | `relation_pts32_trainvocab_scratch_seed{17,42,73}.pt` |
| Primary relation | seed 73，只由 FSC official val MAE/RMSE 选择 |
| Text prototypes | 固定 FSC official-train 89 类；不加载 OmniCount 类名 |
| Relation initialization | scratch，`pretrained_from=None`，禁止 COCO initialization |
| Tiled gate | filter=0.3，category=0.4，`tau_inst=0.99` |
| Grouping | category-aware spatial grouping，`tau_affinity=0.1`，max group 30 |
| Route | `always_tiled`，由 FSC official val 1,286 张选择 |
| OmniCount tuning | 无 threshold / route / seed tuning |

关键 SHA-256：

| Asset | SHA-256 |
|---|---|
| FSC frozen validation config | `8c4f0c61cdf74221470cd41fcc58837be7f39ba37b5f08b50d303b9b3ab5ee16` |
| category pts32 | `5726d7ffd2e0eafd772a65b3963ba975afc21af81910483569812769df330998` |
| candidate filter pts32 | `55d530c3c942ecb873fbbb7433b74ab24c23a2f544f9ff206f79832daec2de11` |
| relation pts32 seed 73 | `ecf09d2967fc10cb901a6fd2858e551a78ca8e84367d8fadafff75af3d142e48` |
| FSC train-89 prototypes | `125a529726a568a19a0b4886d1c8a6661adbd35abb26e94de09bef339a125106` |
| image-only manifest | `c982a19d7e1dd91f9397b601efa765eb95b11de9c7b11e041587d854be4979d5` |
| isolated target shard | `d3febab2e7ea1f11846b2bfa77bff66ba61c4075839cbc65a6ab546dad95d17c` |
| prediction-only result | `ef5f938a6291df9646927430f5a8031f191b4ca446a2e5facdb0797a82502c43` |
| scored result | `667e2e2878dcbcce218f3c9832685d1529e3461bf3e22b15440b08dd1b20d6c6` |

## 4. 难度切片

### 4.1 每图 GT 类别数

| GT classes / image | Images | MAE | RMSE | Bias | Mean pred | Mean GT |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,124 | 14.5827 | 23.4900 | +14.3247 | 20.1343 | 5.8096 |
| 2 | 311 | 15.6174 | 18.8461 | +15.4887 | 21.2444 | 5.7556 |
| 3 | 186 | 14.7097 | 17.9656 | +14.6344 | 22.5000 | 7.8656 |
| 4+ | 336 | **5.8363** | **10.3922** | +5.1696 | 11.1905 | 6.0208 |
| Multi-class 2+ | 833 | 11.4694 | 15.7555 | +11.1357 | 17.4694 | 6.3337 |

4+ 类图像明显较好主要来自数据 composition / super-category 差异，不能解释为“类别越多模型越好”，也不能替代匿名组与 GT 类别的 Hungarian matching。

### 4.2 Super-category

| Super-category | Images | MAE | RMSE | Bias | Mean pred | Mean GT |
|---|---:|---:|---:|---:|---:|---:|
| Birds | 10 | 19.2000 | 19.7484 | +19.2000 | 35.7000 | 16.5000 |
| Fruits | 303 | **2.8713** | 3.8280 | +2.1914 | 6.9076 | 4.7162 |
| Pets | 11 | **2.2727** | 2.7469 | -0.8182 | 8.8182 | 9.6364 |
| Satellite | 127 | 6.7165 | 12.8023 | +5.9291 | 8.1496 | 2.2205 |
| Supermarket | 251 | **27.0279** | **41.9049** | **+26.4303** | 40.8127 | 14.3825 |
| Urban | 1,000 | **14.6580** | 17.8528 | **+14.6020** | 19.9540 | 5.3520 |
| Wild | 255 | 10.0510 | 12.2486 | +9.9647 | 13.3451 | 3.3804 |

Urban 和 Supermarket 分别贡献总绝对误差的 56.50% 和 26.15%，合计 82.65%。这两个域是当前跨域正偏的主要来源。

### 4.3 GT count bin

| GT count | Images | MAE | RMSE | Bias | Mean pred | Mean GT |
|---|---:|---:|---:|---:|---:|---:|
| 1–5 | 1,276 | 8.9208 | 12.9108 | +8.7187 | 11.7022 | 2.9835 |
| 6–10 | 496 | 16.8569 | 22.0878 | +16.5746 | 23.8246 | 7.2500 |
| 11–20 | 123 | 23.4472 | 32.2445 | +22.9431 | 36.5935 | 13.6504 |
| 21–50 | 42 | 52.5476 | 59.2282 | +52.5476 | 86.6667 | 34.1190 |
| 51+ | 20 | 55.5000 | 69.6707 | +50.1000 | 114.6500 | 64.5500 |

与 FSC-147 的高密度漏计不同，OmniCount 几乎所有密度段都表现为过计数，说明这是 domain calibration / candidate multiplicity 问题，不是单纯 proposal recall 不足。

## 5. 错误诊断

### 5.1 方向与相关性

| 诊断 | 数值 |
|---|---:|
| Over / exact / under images | 1,722 / 95 / 140 |
| Pred=0 images | 42 |
| Pred–GT Pearson correlation | 0.7522 |
| Raw tiled candidates–pred correlation | 0.7864 |
| Raw tiled candidates–GT correlation | 0.5725 |
| Top 1% absolute-error share | 8.06% |
| Top 5% absolute-error share | 24.36% |

相对“永远预测 0”的逐图绝对误差，strict 模型只在 493 张更好、126 张相同、1,338 张更差。这不代表零计数器是合理方法，而是提醒 OmniCount 的低均值会让 MAE 对系统性过计数非常敏感。

### 5.2 最大错误样本

| Image | Domain | GT | Pred | Error | Raw tiled candidates |
|---|---|---:|---:|---:|---:|
| `IMG-20211029-WA0020...jpg` | Supermarket | 62 | 212 | +150 | 387 |
| `20211009_181737...jpg` | Supermarket | 16 | 142 | +126 | 302 |
| `85_jpeg...jpg` | Supermarket | 68 | 194 | +126 | 379 |
| `product12-09-2021_10-55-42...jpg` | Supermarket | 5 | 128 | +123 | 296 |
| `97_jpeg...jpg` | Supermarket | 51 | 167 | +116 | 338 |

Top failures 全部来自 Supermarket，且 raw tiled candidates 很高。当前 relation dedup 和 FSC-train filter/category gate 没有充分消除 tile 内部与 tile 间的重复/背景候选。

## 6. 与历史记录和 baseline 的关系

| 方法 / 记录 | 输入与资产协议 | MAE | RMSE | 可否与 strict same-model 直接等同 |
|---|---|---:|---:|---|
| Zero-count sanity | 不看图，仅 sanity | 6.0327 | 10.1942 | 否，不是有效方法 |
| Historical OV-CUD predicted groups | 旧 FSC/COCO heads + OmniCount-93 prototypes + full image | 4.6765 | 8.4635 | **否** |
| Historical OV-CUD class-agnostic | 旧 checkpoint + full image | 6.7726 | 10.2326 | **否** |
| ABC123 local `max_density` | Prompt-free，density-supervised | 7.2701 | 18.0555 | 不同监督/输出协议 |
| OWLv2 class-aware | 每图输入 GT class list prompt | 4.8723 | 9.6504 | Prompted，不同输入协议 |
| **Strict FSC model，本轮** | **固定 train-89，image only，2×2 frozen route** | **13.2575** | **20.5566** | **当前 same-model 正式值** |

历史 predicted-groups 的 mean prediction 只有 1.6771，而本轮是 19.0000；历史 class-agnostic 是 11.4992。候选前端也从历史 full-image pts32 的 43.63 candidates/image 增加到本轮 113.35，约 2.60 倍。因此 4.68 → 13.26 不是一个组件的退化，而是同时包含：

1. strict train-only category/filter/relation 资产替换；
2. OmniCount-93 prototypes 改为固定 FSC train-89 vocabulary；
3. full-image 前端改为 FSC-val 选择的 always-tiled 2×2；
4. 阈值从 OmniCount 历史 `conf=0.1` 改为 FSC-val frozen joint gate。

不能把差值归因给“移除 COCO relation pretraining”，也不能用 OmniCount test 重新调阈值后覆盖本结果。

## 7. 论文结论与后续处理

### 7.1 当前应采用的数字

- 若表述是“FSC strict 26.50 的同一模型零样本迁移到 OmniCount”，使用 **13.26 / 20.56**。
- 旧 4.68 / 8.46 只能标为 legacy OmniCount-adapted vocabulary protocol，不能再写 same strict model。
- 当前 strict 固定词表与 OmniCount 93 类 ontology 不一致，主文暂不报告 mRMSE / mRMSE-nz，也不能用本轮证明正确输出 OmniCount 类名。

### 7.2 后续实验边界

本次结果已经读取 OmniCount test GT，因此任何后续改进必须预先定义并在其他 validation 数据上选择：

1. image-only route：full-image / 2×2 的选择只用候选密度、coverage proxy 或外部 validation，不能按 OmniCount test error 选择；
2. tile-aware duplicate suppression：训练或校准数据不能来自本次 test；
3. vocabulary 协议拆表：固定 FSC train-89 的 same-model transfer 与 OmniCount-93 open-vocabulary prototype replacement 分开报告；
4. 若做匿名多类别指标，预先固定预测 group–GT class 的空间 Hungarian matching，并与 total-count 表分开。

## 8. 复现命令

生成 image-only manifest：

```bash
python3 script/export_omnicount_strict_protocol.py manifest \
  --out result/configs/omnicount_test_image_manifest_strict.json
```

导出只含 image-derived tensor 的 fast cache：

```bash
python3 script/export_omnicount_inference_cache.py \
  --manifest result/configs/omnicount_test_image_manifest_strict.json \
  --source-dir /home/czp/ws_yiyang/ovcud_cache/omnicount_test_pts16 \
  --out-dir /home/czp/ws_yiyang/ovcud_cache/omnicount_strict_nogt_fast
```

生成全量 tiled cache；两个命令分别使用 `--shard-index 0/1`：

```bash
/home/czp/ws_yiyang/FreeCounting/venv/bin/python \
  script/preprocess_omnicount_tiled_nogt.py \
  --manifest result/configs/omnicount_test_image_manifest_strict.json \
  --omnicount-dir /home/czp/official_code/dataset/omnicount/OmniCount-191 \
  --out-dir /home/czp/ws_yiyang/ovcud_cache/omnicount_strict_nogt_tiled2x2 \
  --tiles 2 --overlap 0.25 --pts-per-side 32 --points-per-batch 64 \
  --merge-mode bbox --num-shards 2 --shard-index 0 --device cuda
```

无 GT 预测；该子命令不接受 target 参数：

```bash
/home/czp/ws_yiyang/FreeCounting/venv/bin/python \
  script/eval_omnicount_strict_nogt.py predict \
  --manifest result/configs/omnicount_test_image_manifest_strict.json \
  --out result/logs/omnicount_strict_nogt_predictions_full1957.json \
  --device cuda
```

预测进程退出后再生成/读取隔离 target shard 并计分：

```bash
python3 script/export_omnicount_strict_protocol.py targets \
  --manifest result/configs/omnicount_test_image_manifest_strict.json \
  --out result/configs/omnicount_test_count_targets_strict.json

python3 script/eval_omnicount_strict_nogt.py score \
  --manifest result/configs/omnicount_test_image_manifest_strict.json \
  --targets result/configs/omnicount_test_count_targets_strict.json \
  --predictions result/logs/omnicount_strict_nogt_predictions_full1957.json \
  --bootstrap 5000 \
  --out result/logs/omnicount_strict_nogt_full1957.json
```

正式资产：

- `result/configs/omnicount_test_image_manifest_strict.json`
- `result/configs/omnicount_test_count_targets_strict.json`
- `result/logs/omnicount_strict_nogt_predictions_full1957.json`
- `result/logs/omnicount_strict_nogt_full1957.json`
- `script/export_omnicount_strict_protocol.py`
- `script/export_omnicount_inference_cache.py`
- `script/preprocess_omnicount_tiled_nogt.py`
- `script/eval_omnicount_strict_nogt.py`
