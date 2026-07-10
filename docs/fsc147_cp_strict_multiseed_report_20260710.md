# FSC-147 CP 严格多随机种子消融报告

**日期**：2026-07-10

**组件**：CP（COCO relation pretraining）

**状态**：已完成（pts16/pts32 × CP/scratch × 3 seeds，official-val 校准，full-test 1,190）

## 1. 为什么必须重做 M3

旧 M3 的 `12.59` 与 M6 的 `12.67` 不能回答“CP 是否有效”，原因不只是差值很小：

1. 旧 `fsc147_train_fast` 和 `fsc147_train_pts32` 各含约 6,143 个 cache，实际覆盖 train/val/test，而训练脚本直接对整个目录随机划分。
2. 按旧 category seed=42 复算，fast/pts32 分别有 **1,002/1,018 张 official-test 图进入 category train**。
3. 按旧 relation seed=42 复算，fast/pts32 分别有 **1,066/1,075 张 official-test 图进入 relation train**。
4. 旧 M3 只替换 pts32 关系头；741 张 fast 测试图仍使用带 COCO 预训练的 `fsc147_relation_best.pt`，因此不是完整 CP leave-one-out。
5. 旧点估计 `12.595 vs 12.668` 的 paired 95% CI 为 `[-0.220,+0.071]`，统计上也不能说明 CP 有害。

因此，本报告将旧数字标记为 **legacy partial/leaky diagnostic**，不再作为 CP 的因果证据。

## 2. 严格配对协议

### 2.1 数据隔离

| 阶段 | 数据 | 用途 |
|---|---|---|
| Category train/model-val | official train 内固定 85/15 划分 | 训练固定 pts16/pts32 category heads |
| Relation train/model-val | official train 内固定 90/10 划分 | 训练与选择 relation checkpoint |
| Threshold calibration | official val 1,286 | 只选择 `tau_inst` |
| Final evaluation | official test 1,190 | 阈值冻结后只运行一次 |

- official split：`Train_Test_Val_FSC_147.json`。
- data split seed：`20260710`，所有 CP/scratch 与三个模型 seed 共享同一文件集合。
- 模型 seed：`17 / 42 / 73`，预先写入评测脚本。
- category heads 固定，不随 CP/scratch 改变；两者均只使用 official train。
- pts16 cache 匹配 3,656/3,659 张 train 图，缺失 `2737/2979/7454`。
- pts32 cache 匹配 3,657/3,659 张 train 图，缺失 `2737/2979`。

### 2.2 唯一实验变量

每个 resolution × seed 都训练一对模型：

- **CP**：从 `coco_relation_1152.pt` 初始化，再用 FSC train dot supervision 微调。
- **Scratch**：随机初始化，直接用相同 FSC train dot supervision 训练。

其余全部相同：hidden=512、3 layers、dropout=0.1、pos_weight=8、neg_ratio=5、lr=5e-4、40 epochs、best internal-val loss。每一对使用相同文件顺序、pair sampling 和 dropout seed。

### 2.3 Validation-only 阈值选择

- 固定 `conf_threshold=0.2`、`tau_affinity=0.1`。
- 预注册 `tau_inst` grid：`0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.985, 0.99, 0.995, 0.997, 0.999`。
- 每个完整 pts16+pts32 模型只在 official val 上选择一个共同 `tau_inst`。
- 选择顺序：最低 MAE → 最低 RMSE → 最接近旧 0.99 → 更高 tau。
- val 高密度前端与 test 对齐：GT<=50 使用 pts16；GT>50 使用 pts16 + 2x2 tiled pts32、bbox-IoU@0.5 merge。
- 代码和网格在读取 strict test 结果前已推送：`a2728af`、`a62efe3`、`fac02ff`。

## 3. 训练端结果

### 3.1 固定 category heads

| Head | official-train internal-val top-1 | Best epoch |
|---|---:|---:|
| pts16 | 85.86% | 21 |
| pts32 | 85.21% | 24 |

旧 category head 的 validation 随机划分包含 test 图，因此不能与这里的 top-1 直接比较。

### 3.2 Relation model-val，三 seed mean±std

| Resolution | Init | Best loss | Inst precision | Inst recall | Best epoch |
|---|---|---:|---:|---:|---:|
| pts16 | CP | **0.0817 ± 0.0089** | **37.13 ± 1.31%** | 76.65 ± 1.31% | 24.3 ± 2.5 |
| pts16 | Scratch | 0.1141 ± 0.0448 | 28.99 ± 3.50% | **78.91 ± 4.22%** | 27.7 ± 8.1 |
| pts32 | CP | **0.0990 ± 0.0106** | **58.37 ± 1.18%** | 90.92 ± 0.90% | 29.7 ± 7.0 |
| pts32 | Scratch | 0.1073 ± 0.0100 | 53.41 ± 7.22% | **93.07 ± 2.91%** | 28.7 ± 6.7 |

训练端结论：CP 降低 loss、提高 precision，并降低本次 pts16 三 seed 的观察方差；scratch 的 recall 略高。该结果不能替代最终 counting MAE。

## 4. Validation 与 Test 结果

official-val 高密度缓存严格覆盖预期的 386/386 张，缺失和额外文件均为 0。六次运行均由 validation 独立选择 `tau_inst=0.999`，冻结后才读取 test 指标。

| Condition | Seed | Val MAE / selected tau | Test MAE | Test RMSE | Bias | MAE w/o 7611 |
|---|---:|---:|---:|---:|---:|---:|
| CP | 17 | 22.6773 / 0.999 | 13.8765 | 111.2688 | -9.8983 | 12.8898 |
| CP | 42 | 22.6804 / 0.999 | 13.9723 | 111.2729 | -9.7235 | 12.9857 |
| CP | 73 | 22.6625 / 0.999 | 14.0630 | 111.3007 | -9.6597 | 13.0765 |
| Scratch | 17 | 22.7488 / 0.999 | 13.8613 | 111.2918 | -9.9908 | 12.8730 |
| Scratch | 42 | 22.7247 / 0.999 | 13.9513 | 111.2783 | -9.9042 | 12.9647 |
| Scratch | 73 | 22.6874 / 0.999 | 14.1017 | 111.3006 | -9.4176 | 13.1152 |

| Condition | Val MAE mean±std | Test MAE mean±std | Test RMSE mean±std | MAE w/o 7611 mean±std |
|---|---:|---:|---:|---:|
| CP | **22.6734 ± 0.0096** | **13.9706 ± 0.0933** | **111.2808 ± 0.0174** | **12.9840 ± 0.0934** |
| Scratch | 22.7203 ± 0.0310 | 13.9714 ± 0.1214 | 111.2902 ± 0.0112 | 12.9843 ± 0.1223 |

配对的 `CP - Scratch` test ΔMAE 为：seed 17 `+0.0151`、seed 42 `+0.0210`、seed 73 `-0.0387`；均值 `-0.00084 ± 0.03288`。逐图、先对三个 seed 取平均再 bootstrap 5,000 次得到 95% CI **[-0.04398, +0.04314]**，明确跨 0。

### 4.1 密度切片

| GT count | CP MAE mean±std | Scratch MAE mean±std | CP-Scratch |
|---|---:|---:|---:|
| 0-10 | 2.0556 ± 0.0509 | 2.0833 ± 0.0726 | -0.0278 |
| 11-20 | 2.9067 ± 0.0099 | 2.8968 ± 0.0399 | +0.0100 |
| 21-50 | 7.0920 ± 0.0589 | 7.0960 ± 0.0133 | -0.0040 |
| 51-100 | 10.0617 ± 0.2507 | 10.0748 ± 0.3868 | -0.0131 |
| 100+ | 52.5026 ± 0.1922 | 52.4872 ± 0.2341 | +0.0154 |

所有分桶差值均小于 0.03 MAE，没有隐藏的密度区间收益。`7611.jpg` 的 CP 三 seed 均预测 1,373；scratch 为 1,371/1,373/1,373。移除该图后配对结论不变，因此 CP 结论不是由极端离群点造成。

### 4.2 阈值边界诊断

六条 validation MAE 曲线从 0.5 到 0.999 总体单调下降，并全部在预注册网格上界取最优。这说明无泄漏训练头下 relation dedup 仍偏强，`0.999` 不能解释为已找到内部最优点。为避免在看到 test 后继续追分，本轮不扩展网格；后续若研究阈值，应预先声明更高阈值/关闭去重的 validation 网格，并使用新的训练 seed 作为独立确认。

## 5. 结论与决策

1. **CP 对 FSC-147 counting MAE 的边际作用为零**：13.9706 vs 13.9714，Δ=-0.00084，CI 跨 0，三个 seed 的方向也不一致。
2. **不能说 CP 有害**：旧 M3 的 `12.595 < 12.668` 来自 partial/leaky 配置；严格结果没有复现稳定负作用。
3. **CP 有训练端效果，但未转化为 test MAE**：relation model-val loss/precision 更好，official-val MAE 平均改善 0.0469；test 上收益消失。
4. CP 的 MAE seed std 略小（0.093 vs 0.121），但只有三个 seed，不把它写成显著稳定性 claim。
5. 论文组件表不应再把 CP 写成 accuracy-critical module。若保留 COCO 初始化，只能描述为训练初始化/关系校准选择，并明确“最终 MAE 无显著收益”；若追求最简方法，当前证据支持删除 CP。
6. 不把单个最佳 seed 当作主结果，也不因为 test 上某个 tau 更好而回调阈值。

绝对指标方面，严格 train-only 模型约为 13.97，而旧 12.67 权重的训练 cache 含 official-test 图。两者不能用于估计 CP 增益；投稿主结果应最终改为无泄漏、无 GT-derived `valid` 的统一重跑结果，不能继续把 12.67 当成 reviewer-ready headline。

## 6. 仍然存在的协议限制

本实验严格解决了 **train/test overlap、完整 pts16+pts32 CP 移除、多 seed、validation-only 阈值选择**，但仍是 cache-compatible component audit：

1. FSC cache 的 `valid` 来自 GT dot coverage；尚不是 strict no-GT candidate filtering。
2. 当前高密度路由由 GT-count 构建的 cache membership 复现；尚未替换为可部署的预测路由。
3. CP 三 seed 复用同一个 COCO pretrained checkpoint，seed 方差来自 FSC fine-tuning；没有重训三个 COCO pretraining seed。

因此新结果可用于判断 CP 的边际作用，但不能把绝对 MAE 写成 reviewer-ready no-GT headline。

## 7. 复现入口

- `script/train_category_v2.py`：official split-aware category training。
- `script/train_relation_1152.py`：official split-aware paired relation training。
- `script/run_cp_pretraining_multiseed.py`：完整训练/cache/evaluation orchestration。
- `script/eval_cp_pretraining_multiseed.py`：official-val calibration、frozen full-test、bootstrap。
- `result/checkpoints/cp_strict/`：本地 14 个 checkpoint，因体积不提交 Git。
- `result/logs/cp_strict_multiseed_summary.json`：不含逐图行的可读汇总。
- `result/logs/cp_strict_multiseed.json.gz`：逐 seed、逐阈值、逐图机器可读结果。
- 运行耗时：1,312 秒（不含训练和 SAM cache 预处理）；结果文件 65 KB（gzip）。
