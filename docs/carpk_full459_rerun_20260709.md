# CARPK Full-Test Current Rerun 记录

**日期**: 2026-07-09
**目的**: 在当前代码和当前 checkpoint 下重跑 CARPK test full 459，确认最近 FSC147 cache/tiling/rescue 相关改动是否影响 CARPK 主结果。

---

## 1. 结论

CARPK full 459 rerun 与历史主日志 **逐图完全一致**，指标没有变化。

| 记录 | nImages | MAE | RMSE | bias | 说明 |
|---|---:|---:|---:|---:|---|
| 历史 `carpk_pts32_exp5c_best.json` | 459 | 4.0566 | 5.5094 | -1.6166 | CARPK 主结果 |
| 当前 rerun `carpk_full459_current_rerun_20260709.json` | 459 | 4.0566 | 5.5094 | -1.6166 | 逐图完全一致 |
| 差值 | 0 | 0.0000 | 0.0000 | 0.0000 | 无变化 |

逐图对比：

- 文件集合完全一致；
- 459 / 459 张的 `pred_count`, `gt_count`, `n_candidates`, `n_groups` 等结果完全一致；
- 因此当前 FSC147 1190 审计、`7611` true MR100、以及 extreme-density rescue tiling 的代码/记录没有改变 CARPK 主评估。

---

## 2. 运行配置

运行命令：

```bash
python3 script/eval_carpk.py \
  --cache-dir /home/czp/ws_yiyang/ovcud_cache/carpk_test \
  --category-ckpt result/checkpoints/category_cosine_pts32.pt \
  --relation-ckpt result/checkpoints/fsc147_relation_pts32_best.pt \
  --text-prototypes result/checkpoints/text_prototypes_fsc147.pt \
  --tau-inst 0.99 \
  --tau-affinity 0.1 \
  --conf-threshold 0.1 \
  --out result/logs/carpk_full459_current_rerun_20260709.json \
  --device cuda
```

固定口径：

| 项目 | 值 |
|---|---|
| cache | `/home/czp/ws_yiyang/ovcud_cache/carpk_test` |
| cache files | 459 |
| category head | `result/checkpoints/category_cosine_pts32.pt` |
| relation head | `result/checkpoints/fsc147_relation_pts32_best.pt` |
| text prototypes | `result/checkpoints/text_prototypes_fsc147.pt` |
| `tau_inst` | 0.99 |
| `tau_affinity` | 0.1 |
| `conf_threshold` | 0.1 |

Checkpoint 审计：

- `fsc147_relation_pts32_best.pt` 与历史日志中记录的 `fsc147_relation_pts32_exp5c.pt` 权重完全一致；
- 两者 `relation_head` state dict 的 max absolute diff 为 0；
- `best.pt` 是带 recommended inference 元信息的推理版。

---

## 3. 指标明细

整体：

| Metric | Value |
|---|---:|
| MAE | 4.0566 |
| RMSE | 5.5094 |
| bias | -1.6166 |
| total candidates | 87,502 |
| confidence filtered | 43 |
| mean candidates / image | 190.64 |
| mean GT / image | 103.49 |
| mean pred / image | 101.87 |

按 GT count bin：

| Bin | #Images | MAE | RMSE | bias |
|---|---:|---:|---:|---:|
| 0-10 | 20 | 0.80 | 0.95 | +0.10 |
| 11-20 | 6 | 1.17 | 1.47 | -0.50 |
| 21-50 | 21 | 4.38 | 5.28 | -4.38 |
| 51-100 | 111 | 4.05 | 5.09 | -2.50 |
| 100+ | 301 | 4.31 | 5.89 | -1.24 |

Top absolute errors：

| Image | GT | Pred | Error | Candidates |
|---|---:|---:|---:|---:|
| `20161225_TPZ_00355.png` | 141 | 120 | -21 | 210 |
| `20161225_TPZ_00375.png` | 115 | 94 | -21 | 171 |
| `20161225_TPZ_00428.png` | 157 | 137 | -20 | 202 |
| `20161225_TPZ_00408.png` | 123 | 105 | -18 | 180 |
| `20161225_TPZ_00343.png` | 129 | 112 | -17 | 182 |

---

## 4. 备注

运行中出现 `numpy exp overflow` warning，来源于 dedup 中对极大负 relation logit 做 sigmoid：

```text
RuntimeWarning: overflow encountered in exp
```

该 warning 不改变最终结果；本次 rerun 与历史日志逐图完全一致，说明 warning 是历史代码中的数值提示，不是本次变化引入的行为差异。

---

## 5. 产物

| 文件 | 说明 |
|---|---|
| `result/logs/carpk_full459_current_rerun_20260709.json` | 当前 CARPK full 459 rerun 结果 |
| `docs/carpk_full459_rerun_20260709.md` | 本中文记录 |
