# FSC147 Multi-Resolution 组件级消融实验记录

**日期**: 2026-07-09
**目标**: 审计 FSC147 full-test multi-resolution pipeline 的组件级消融，并修正此前只评估 1,189 张图的问题。FSC147 test split 实际为 1,190 张，其中 `7611.jpg` 在原始图像和标注中存在，但缺失于 final multires cache。

---

## 1. 口径审计结论

| 记录 | 图像数 | MAE | RMSE | bias | 说明 |
|---|---:|---:|---:|---:|---|
| 历史主结果 `fsc147_multires_extended.json` | 1,189 | 12.74 | 106.20 | -6.03 | 文件内 `results` 长度为 1,189，且无 `gt_count=2560` 样本 |
| current-code 组件 rerun | 1,189 | 11.45 | 105.56 | -6.72 | 同一 current-code 路由，但按 cache 存在性跳过 `7611.jpg` |
| **current-code 修正版 rerun** | **1,190** | **13.47** | **126.74** | **-8.75** | 显式补回 `7611.jpg`，本报告主表采用此口径 |

`7611.jpg` 的审计结果：

| 字段 | 值 |
|---|---|
| FSC147 split | test |
| GT count | 2,560 |
| 原图 | `/home/czp/official_code/dataset/FSC147/images_384_VarV2/7611.jpg` |
| 标注 | `/home/czp/official_code/dataset/FSC147/annotation_FSC147_384.json` |
| 缺失 cache | `fsc147_test_fast`, `fsc147_test_multires`, `fsc147_test_multires_all`, `fsc147_test_multires_51_100` |
| 可用 fallback cache | `/home/czp/ws_yiyang/ovcud_cache/fsc147_test_tiled/7611.pt` |
| fallback candidates | 208 |
| A8 prediction | 138 |
| 单图绝对误差 | 2,422 |

结论：FSC147 全量测试应按 **1,190 张** 报告。此前 1,189 张结果是 cache-only 口径，会静默跳过一个极端高密度失败样本，因此不应继续写成 full 1190。

---

## 2. 实验设置

评估数据：

- FSC147 test split: 1,190 张；
- 修正版评估: 1,190 张全部计入；
- `7611.jpg` 使用 100+ tiled cache fallback；
- 在 A7 `no_100plus_multires` 中，因该场景显式移除 100+ 前端且 fast cache 也缺失，`7611.jpg` 按 0 候选预测计入，而不是跳过。

固定参数：

| 参数 | 值 |
|---|---|
| `tau_inst` | 0.99 |
| `tau_affinity` | 0.1 |
| `conf_threshold` | 0.2 |
| final cache | `/home/czp/ws_yiyang/ovcud_cache/fsc147_test_multires_all` |
| base fast cache | `/home/czp/ws_yiyang/ovcud_cache/fsc147_test_fast` |
| 100+ MR cache | `/home/czp/ws_yiyang/ovcud_cache/fsc147_test_multires` |
| 51-100 MR cache | `/home/czp/ws_yiyang/ovcud_cache/fsc147_test_multires_51_100` |

模型口径：

| Cache 来源 | Category head | Relation head |
|---|---|---|
| `final_fast` / `fast` | `category_cosine_fast.pt` | `fsc147_relation_best.pt` |
| `final_mr100` / `final_mr51` / `mr100` / `mr51` / tiled fallback | `category_cosine_pts32.pt` | `fsc147_relation_pts32_best.pt` |

运行命令：

```bash
python3 script/ablation_fsc147_multires_components.py \
  --out result/logs/fsc147_multires_component_ablation_1190.json
```

---

## 3. 组件级消融

| ID | Variant | 目的 |
|---|---|---|
| A1 | category confidence filter only | 只保留 `valid & top_conf>=0.2`，不做去重 |
| A2 | class-bucket IoU NMS@0.5 | 用检测式 heuristic 去重替代 relation head |
| A3 | global relation dedup | 使用 relation head 去重，但不做 semantic/category grouping |
| A4 | group no spatial | semantic/category grouping + relation dedup，但不做 spatial refinement |
| A5 | no adaptive dedup | 使用 spatial refinement，但大 group 不启用 adaptive/greedy dedup |
| A8 | full current pipeline | 完整 current-code rerun |

1190 图主表：

| Variant | MAE | RMSE | bias | 相对 A8 |
|---|---:|---:|---:|---:|
| A1 filter only | 15.40 | 126.93 | -5.57 | +1.93 |
| A2 IoU NMS@0.5 | 15.10 | 126.91 | -6.39 | +1.62 |
| A3 global relation | **13.34** | 126.77 | -9.12 | -0.13 |
| A4 group no spatial | 13.43 | 126.75 | -8.91 | -0.05 |
| A5 no adaptive dedup | 14.39 | 126.81 | -7.14 | +0.92 |
| A8 full | 13.47 | **126.74** | -8.75 | 0.00 |

解读：

1. Relation head 仍明显优于 IoU NMS：A2 `15.10` → A3 `13.34`。
2. Adaptive dedup 仍有稳定贡献：A5 `14.39` → A8 `13.47`。
3. A3/A4/A8 非单调，说明 FSC147 的 single-class 标注不适合单独证明 semantic/category grouping；multi-category claim 应主要由 OmniCount 多类别实验支撑。
4. 1190 口径下 RMSE 明显升高，主要由 `7611.jpg` 这个 GT=2,560 的极端密度样本带来。

---

## 4. 高密度前端消融

| ID | Variant | 说明 |
|---|---|---|
| A6 | no 51-100 multires | 去掉 51-100 bin 的 MR overlay，只保留 100+ MR |
| A7 | no 100+ multires | 去掉 100+ bin 的 MR overlay，只保留 51-100 MR |
| A8 | full | 同时保留 51-100 和 100+ MR |

1190 图结果：

| Variant | MAE | RMSE | bias | 主要影响 |
|---|---:|---:|---:|---|
| A6 no 51-100 MR | 15.66 | 127.21 | -12.80 | 51-100 欠计数明显 |
| A7 no 100+ MR | 26.34 | 142.97 | -23.20 | 100+ 大幅退化 |
| A8 full | **13.47** | **126.74** | -8.75 | 当前 1190 口径最好 |

Per-bin MAE：

| Variant | 0-10 | 11-20 | 21-50 | 51-100 | 100+ |
|---|---:|---:|---:|---:|---:|
| A6 no 51-100 MR | 1.60 | 2.19 | 5.78 | 18.24 | 56.06 |
| A7 no 100+ MR | 1.60 | 2.19 | 5.78 | 8.00 | 134.57 |
| A8 full | **1.60** | **2.19** | **5.78** | **8.00** | **56.06** |

结论：

- 51-100 MR overlay 主要把 51-100 MAE 从 `18.24` 降到 `8.00`；
- 100+ MR overlay 更关键，把 100+ MAE 从 `134.57` 降到 `56.06`；
- 这组结果强支撑 multi-resolution frontend 是最终性能的关键，而不是单纯阈值调参；
- `7611.jpg` 纳入后，100+ bin 更能反映极端密度场景下的 proposal recall 瓶颈。

---

## 5. Oracle 诊断

| ID | Variant | 说明 |
|---|---|---|
| O1 | oracle category | 对覆盖 GT dot 的候选赋 GT class one-hot，再走 learned relation dedup |
| O2 | oracle dot-sharing dedup | 保持预测 grouping，用 GT dot-sharing 替换 same-instance components |
| O3 | proposal cover upper bound | 只统计 final candidates 覆盖到的 unique GT dots |

1190 图结果：

| Variant | MAE | RMSE | bias | 解释 |
|---|---:|---:|---:|---|
| O1 oracle category | 13.17 | 125.55 | -8.99 | 比 A8 只好 `0.30`，分类不是主瓶颈 |
| O2 oracle dot-sharing dedup | 20.12 | 131.10 | -19.90 | 在 MR 合并 cache 上不是有效 upper bound |
| O3 proposal cover upper bound | **7.98** | **95.29** | -7.98 | proposal recall 仍限制极端密度图 |

O3 per-bin：

| Bin | #Images | MAE | RMSE | bias |
|---|---:|---:|---:|---:|
| 0-10 | 60 | 0.70 | 1.49 | -0.70 |
| 11-20 | 268 | 1.42 | 3.19 | -1.42 |
| 21-50 | 413 | 4.13 | 8.67 | -4.13 |
| 51-100 | 254 | 2.57 | 7.21 | -2.57 |
| 100+ | 195 | 34.43 | 234.89 | -34.43 |

结论：

- O1 接近 A8，说明 current pipeline 的主要误差不是类别分类；
- O3 显著低于 A8，说明如果 proposal 覆盖更完整，仍有较大提升空间；
- O2 在 MR cache 上变差，因为 MR 合并后的候选可能一个 mask 覆盖多个 dots，简单 dot-sharing components 会过度合并，不能作为真正 upper bound。论文中建议只作为诊断放入附录。

---

## 6. 写作建议

建议主文使用 1190 审计口径：

```text
On the full 1,190-image FSC147 test split, the current multi-resolution OV-CUD pipeline obtains 13.47 MAE and 126.74 RMSE. The previously logged 12.74 MAE result was produced on a 1,189-image cache-only protocol that omitted 7611.jpg, an extreme-density image with 2,560 objects.
```

组件结论可以写：

```text
Replacing learned relation deduplication with class-bucket IoU NMS increases MAE from 13.47 to 15.10 under the audited 1,190-image protocol. Removing the 100+ multi-resolution frontend causes a much larger degradation (13.47 to 26.34), confirming that dense-scene candidate recall is the dominant bottleneck for extreme counts.
```

需要避免写：

```text
The 12.74 result is full 1,190-image audited FSC147.
Every grouping subcomponent monotonically improves FSC147.
Oracle dot-sharing dedup is a strict upper bound.
```

---

## 7. 产物

| 文件 | 说明 |
|---|---|
| `script/ablation_fsc147_multires_components.py` | full-test multi-resolution 组件消融脚本，已加入缺 cache 图像的显式 fallback/zero-candidate 处理 |
| `result/logs/fsc147_multires_component_ablation.json` | 旧 1,189 图 current-code 结果，保留用于 provenance 对照 |
| `result/logs/fsc147_multires_component_ablation_1190.json` | 修正版 1,190 图结果 |
| `docs/fsc147_multires_component_ablation_report_20260709.md` | 本中文实验记录 |
