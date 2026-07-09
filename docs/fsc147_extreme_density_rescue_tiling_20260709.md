# FSC147 极端密度 Rescue Tiling 实验记录

**日期**: 2026-07-09
**目标**: 在不大改 OV-CUD 后端方法的前提下，测试是否可以通过更密的 tiled proposal frontend 缓解 `7611.jpg` 这类极端密度样本的 proposal recall 崩溃问题。

---

## 1. 背景

`7611.jpg` 是 FSC147 test split 中唯一 `GT=2560` 的样本。当前 full 1190 current-code 口径下：

| 项目 | 值 |
|---|---:|
| GT count | 2560 |
| true MR100 candidates | 208 |
| A8 prediction | 138 |
| O3 proposal cover | 208 |
| 单图绝对误差 | 2422 |

此前已确认：补跑 true MR100 后，pts=16 fast 分支没有可用候选，MR100 实际仍等价于 100+ tiled 分支。因此瓶颈不是 fallback 评估口径，而是 dense small-object proposal recall。

---

## 2. 触发条件

Rescue policy 不能使用 GT count。这里先测试一个最保守的无 GT 触发条件：

```text
if fsc147_test_fast/<image>.pt has n_candidates == 0:
    use 4x4 tiled rescue cache
```

在 FSC147 test 1190 张中，该条件只触发 1 张：

| 触发条件 | 触发数 | 触发图像 |
|---|---:|---|
| fast `n_candidates == 0` | 1 / 1190 | `7611.jpg` |
| fast `valid == 0` | 3 / 1190 | `2026.jpg`, `2607.jpg`, `7611.jpg` |
| fast `valid <= 1` | 6 / 1190 | 含 `7611.jpg` |

因此 `n_candidates == 0` 是一个很保守、低误伤的 first-stage trigger。它不使用 GT，只依赖前端是否完全失败。

---

## 3. 单图 Tiling 矩阵

所有配置固定：

- SAM2: `sam2.1_hiera_small`
- `points_per_side=32`
- counting heads: `category_cosine_pts32.pt` + `fsc147_relation_pts32_best.pt`
- merge: `bbox` fast merge，用于避免 4x4/overlap 在 mask pairwise IoU 上 O(n²) 卡死

| 配置 | candidates | valid | A8 pred | abs err | O3 cover | O3 err | cover rate | 预处理耗时 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2x2 tiled ov25 | 208 | 208 | 138 | 2422 | 208 | 2352 | 8.1% | 5.3s |
| 3x3 tiled ov25 | 999 | 997 | 596 | 1964 | 993 | 1567 | 38.8% | 13.9s |
| **4x4 tiled ov25** | **1997** | **1825** | **1098** | **1462** | **1868** | **692** | **73.0%** | 27.4s |
| 3x3 overlap50 | 852 | 850 | 514 | 2046 | 840 | 1720 | 32.8% | 13.4s |
| 4x4 overlap50 | 1103 | 1080 | 665 | 1895 | 1050 | 1510 | 41.0% | 21.8s |
| 2x2 upscaled ov25 | 312 | 312 | 218 | 2342 | 312 | 2248 | 12.2% | 5.6s |

已有 mask-merge reference：

| 参考 cache | candidates | valid | A8 pred | O3 cover |
|---|---:|---:|---:|---:|
| existing 2x2 tiled | 208 | 208 | 138 | 208 |
| existing 3x3 tiled | 993 | 992 | 587 | 993 |
| existing 2x2 upscaled | 312 | 312 | 218 | 312 |

结论：

1. **4x4 tiled ov25 是当前最有效 rescue 配置**：O3 从 208 提升到 1868，A8 pred 从 138 提升到 1098。
2. 3x3 有明显收益，但不足以救回极端密度：pred=596，O3=993。
3. overlap50 在当前 `compute_tiles` 实现下不如普通 ov25；它没有产生更好的有效覆盖。
4. 2x2 upscaled 对 `7611` 帮助有限。
5. 即使 4x4 把 O3 提到 1868，A8 仍只预测 1098，说明在 proposal recall 改善后，后端仍存在高密度欠计数/去重过强问题。

---

## 4. Full-Test Policy 估计

将 full 1190 current-code 结果中仅 `7611.jpg` 替换为 4x4 rescue 结果，得到：

| 口径 | MAE | RMSE | bias | 说明 |
|---|---:|---:|---:|---|
| current-code full 1190 | 13.47 | 126.74 | -8.75 | true MR100, `7611` pred=138 |
| **fast n_candidates==0 → 4x4 rescue** | **12.67** | **113.71** | **-7.95** | 只触发 `7611`, pred=1098 |

Per-bin 变化只影响 100+：

| Bin | MAE after rescue | RMSE after rescue | bias |
|---|---:|---:|---:|
| 0-10 | 1.60 | 2.39 | -0.30 |
| 11-20 | 2.19 | 4.02 | -0.94 |
| 21-50 | 5.78 | 10.24 | -3.71 |
| 51-100 | 8.00 | 12.24 | +1.62 |
| 100+ | 51.14 | 280.12 | -41.37 |

如果 4x4 proposal cover 能被理想后端完全利用，单图 O3=1868 对应 full-test MAE 约为 12.02；这说明 4x4 rescue 仍有后端利用率空间。

---

## 5. 建议

短期可以尝试把 `fast n_candidates == 0 → 4x4 tiled ov25` 作为一个 **极端前端失败 rescue rule**，它不使用 GT，且在 FSC147 test 上只触发 1 张。这个规则可以作为 supplementary/ablation 展示，不建议直接作为主方法 headline，除非在 val/test 上进一步证明不会引入 opportunistic tuning。

下一步建议：

1. 在 val/test 上统计 `fast n_candidates == 0`、`fast valid == 0`、`valid <= 1` 的触发图像，确认是否稳定。
2. 对触发集跑 4x4 rescue，而不是只跑 `7611`。
3. 针对 4x4 rescue cache 研究后端欠计数：例如放宽高密度 group 的 adaptive dedup，或单独调 `conf_threshold` / `tau_inst`。
4. 不建议使用 GT count 或 GT bin 触发 rescue，否则会削弱 prompt-free / count-supervision-free claim。

---

## 6. 产物

| 文件 | 说明 |
|---|---|
| `script/rescue_tiling_7611_matrix.py` | 单图 rescue tiling 矩阵脚本 |
| `script/preprocess_fsc147_tiled.py` | 增加可选 `--merge-mode bbox`，默认仍为原始 mask merge |
| `result/logs/fsc147_rescue_tiling_7611_matrix.json` | 6 配置单图矩阵结果 |
| `result/logs/fsc147_rescue_policy_fast_cand0_4x4.json` | full 1190 替换策略估计结果 |
