# FSC147 Multi-Resolution 组件级消融实验记录

**日期**: 2026-07-09
**目标**: 围绕当前 FSC147 full-test multi-resolution pipeline，补齐 reviewer 最关心的组件级消融：固定同一批 1,189 张可评估 test 图像，依次去掉类别过滤、IoU NMS / relation dedup、semantic grouping、adaptive dedup 和高密度前端 cache，并补充 oracle 诊断。

---

## 1. 重要结论

本次 full rerun 发现一个需要在论文整理前统一的口径问题：

| 项目 | MAE | RMSE | bias | 说明 |
|---|---:|---:|---:|---|
| 历史主结果日志 `fsc147_multires_extended.json` | 12.74 | 106.20 | -6.03 | 之前报告使用的 headline |
| 本次 current-code A8 rerun | **11.45** | **105.56** | -6.72 | 同一 final multires cache，按当前可执行代码重跑 |

Anchor 状态为 `MISMATCH`，MAE 差 `1.29`。逐图比较中，1,189 张里有 406 张预测不同，平均绝对预测差 `2.41`。主要原因不是 cache 缺失，而是历史 `12.74` 文件的生成脚本/中间组合口径与当前可执行脚本不完全一致。

因此本文档的主表采用 **本次 current-code rerun** 作为组件消融 anchor；历史 `12.74` 仍作为此前 headline 记录，不建议在没有统一 provenance 前直接把 `11.45` 替换为论文主数字。

---

## 2. 实验设置

评估数据：

- FSC147 test split: 1,190 张；
- 可评估 cache: 1,189 张；
- 缺失图像: `7611.jpg`；
- GT bins: `0-10 / 11-20 / 21-50 / 51-100 / 100+`。

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

| Cache 类型 | Category head | Relation head |
|---|---|---|
| fast/base | `category_cosine_fast.pt` | `fsc147_relation_best.pt` |
| multi-resolution | `category_cosine_pts32.pt` | `fsc147_relation_pts32_best.pt` |

新增脚本：

```bash
python3 script/ablation_fsc147_multires_components.py \
  --out result/logs/fsc147_multires_component_ablation.json
```

---

## 3. 组件级消融

定义：

| ID | Variant | 目的 |
|---|---|---|
| A1 | category confidence filter only | 只保留 `valid & top_conf>=0.2`，不做去重 |
| A2 | class-bucket IoU NMS@0.5 | 用检测式 heuristic 去重替代 relation head |
| A3 | global relation dedup | 使用 relation head 去重，但不做 semantic/category grouping |
| A4 | group no spatial | semantic/category grouping + relation dedup，但不做 spatial refinement |
| A5 | no adaptive dedup | 使用 spatial refinement，但大 group 不启用 adaptive/greedy dedup |
| A8 | full current pipeline | 完整 current-code rerun |

主表：

| Variant | MAE | RMSE | bias | 相对 A8 |
|---|---:|---:|---:|---:|
| A1 filter only | 13.38 | 105.79 | -3.53 | +1.93 |
| A2 IoU NMS@0.5 | 13.07 | 105.77 | -4.36 | +1.63 |
| A3 global relation | **11.32** | 105.59 | -7.09 | -0.13 |
| A4 group no spatial | 11.40 | 105.57 | -6.88 | -0.05 |
| A5 no adaptive dedup | 12.37 | 105.65 | -5.11 | +0.92 |
| A8 full | 11.45 | **105.56** | -6.72 | 0.00 |

解读：

1. **Relation head 明显优于 IoU NMS**：A2 `13.07` → A3 `11.32`，说明 learned relation dedup 是有贡献的。
2. **Adaptive dedup 有贡献**：A5 `12.37` → A8 `11.45`，大 group 的 adaptive/greedy dedup 带来约 `0.92` MAE 改善。
3. **Semantic/spatial grouping 在 FSC147 single-class 上不是主要收益来源**：A3/A4/A8 差异很小，A3 甚至略优。这不否定 multi-category claim；它说明 FSC147 本身是单类别标注，不适合证明 semantic grouping 的主要价值。OmniCount 多类别表更适合支撑该 claim。

---

## 4. 高密度前端消融

定义：

| ID | Variant | 说明 |
|---|---|---|
| A6 | no 51-100 multires | 去掉 51-100 bin 的 MR overlay，只保留 100+ MR |
| A7 | no 100+ multires | 去掉 100+ bin 的 MR overlay，只保留 51-100 MR |
| A8 | full | 同时保留 51-100 和 100+ MR |

结果：

| Variant | MAE | RMSE | bias | 主要影响 |
|---|---:|---:|---:|---|
| A6 no 51-100 MR | 13.64 | 106.13 | -10.78 | 51-100 欠计数明显 |
| A7 no 100+ MR | 24.21 | 122.26 | -21.07 | 100+ 大幅崩溃 |
| A8 full | **11.45** | **105.56** | -6.72 | 当前最好 |

Per-bin：

| Variant | 0-10 | 11-20 | 21-50 | 51-100 | 100+ |
|---|---:|---:|---:|---:|---:|
| A6 no 51-100 MR | 1.60 | 2.19 | 5.78 | 18.24 | 43.87 |
| A7 no 100+ MR | 1.60 | 2.19 | 5.78 | 8.00 | 122.07 |
| A8 full | **1.60** | **2.19** | **5.78** | **8.00** | **43.87** |

结论：

- 51-100 MR overlay 主要把 51-100 MAE 从 `18.24` 降到 `8.00`；
- 100+ MR overlay 更关键，把 100+ MAE 从 `122.07` 降到 `43.87`；
- 这组结果强支撑 multi-resolution frontend 是最终性能的关键，而不是单纯阈值调参。

---

## 5. Oracle 诊断

定义：

| ID | Variant | 说明 |
|---|---|---|
| O1 | oracle category | 对覆盖 GT dot 的候选赋 GT class one-hot，再走 learned relation dedup |
| O2 | oracle dot-sharing dedup | 保持预测 grouping，用 GT dot-sharing 替换 same-instance components |
| O3 | proposal cover upper bound | 只统计 final candidates 覆盖到的 unique GT dots |

结果：

| Variant | MAE | RMSE | bias | 解释 |
|---|---:|---:|---:|---|
| O1 oracle category | 11.21 | 105.47 | -7.02 | 比 A8 只好 `0.24`，分类不是主瓶颈 |
| O2 oracle dot-sharing dedup | 18.10 | 110.77 | -17.88 | 在 MR 合并 cache 上不是有效 upper bound |
| O3 proposal cover upper bound | **6.01** | **66.60** | -6.01 | proposal recall 仍限制极端密度图 |

O3 per-bin：

| Bin | MAE | RMSE | bias |
|---|---:|---:|---:|
| 0-10 | 0.70 | 1.49 | -0.70 |
| 11-20 | 1.42 | 3.19 | -1.42 |
| 21-50 | 4.13 | 8.67 | -4.13 |
| 51-100 | 2.57 | 7.21 | -2.57 |
| 100+ | 22.48 | 164.14 | -22.48 |

结论：

- O1 接近 A8，说明 current pipeline 的主要误差不是类别分类；
- O3 显著低于 A8，说明如果 proposal 覆盖更完整，仍有较大提升空间；
- O2 在 MR cache 上变差，因为 MR 合并后的候选可能一个 mask 覆盖多个 dots，简单 dot-sharing components 会过度合并，不能作为真正 upper bound。写论文时建议只把 O2 放在诊断/附录，不作为“上界”主张。

---

## 6. 写作建议

主文可以使用的结论：

```text
On the FSC147 full test split, replacing learned relation deduplication with class-bucket IoU NMS increases MAE from 11.45 to 13.07 under the current multi-resolution protocol. Removing the 100+ multi-resolution frontend causes a much larger degradation (11.45 to 24.21), confirming that dense-scene candidate recall is the dominant bottleneck for extreme counts.
```

但需要避免写：

```text
Every grouping subcomponent monotonically improves FSC147.
Oracle dot-sharing dedup is a strict upper bound.
The current rerun exactly reproduces the historical 12.74 log.
```

更稳妥的最终表述：

```text
We report component ablations using the current executable multi-resolution pipeline. This rerun improves over the historical 12.74 log (11.45 MAE), but we keep the historical number as the audited headline until all generation provenance is unified. The ablation trends are stable: learned relation dedup outperforms IoU NMS, adaptive dedup is beneficial, and multi-resolution candidate generation is essential for dense bins.
```

---

## 7. 产物

| 文件 | 说明 |
|---|---|
| `script/ablation_fsc147_multires_components.py` | 新增 full-test multi-resolution 组件消融脚本 |
| `result/logs/fsc147_multires_component_ablation.json` | 全量 1,189 图结果 |
| `docs/fsc147_multires_component_ablation_report_20260709.md` | 本中文实验记录 |
