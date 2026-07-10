# 实验计划：FSC-147 Full + OmniCount-191 Full Leave-One-Out 消融

**日期**：2026-07-10

**对应论文位置**：`AnonymousSubmission2027.tex` 的组件消融主表与 OmniCount 多类别实验表

> **执行状态**：原 M1-M6 已在 FSC-147 test full 1,190 和 OmniCount-191 test full 1,957 上完成；2026-07-10 另补 CP/scratch × pts16/pts32 × 3 seeds 的无泄漏配对实验。原 M3 只替换 pts32 且旧训练 cache 含 test 图，现降级为 legacy diagnostic。完整结果见 `docs/fsc147_omnicount_leaveoneout_report_20260710.md` 与 `docs/fsc147_cp_strict_multiseed_report_20260710.md`。

---

## 1. 目标与范围

本轮最初仿照 `experiment_plan_ablation_mcac_20260709(1).md` 的 leave-one-out 设计，但后续审计确认原 M3 不满足严格定义；其余推理开关行保留为 12.67 主 checkpoint 下的组件诊断，并另补 CP 严格配对实验。测试集只包括：

1. **FSC-147 test full 1,190**：验证单类别总计数和高密度前端组件。
2. **OmniCount-191 test full 1,957**：验证 prompt-free 多类别分组、per-class 指标和跨域组件行为。

不再使用 FSC147 sample100，不将 MCAC 结果混入这张组件表。M1-M5 每行只从完整配置 M6 中移除一个组件。

## 2. 组件定义

| No. | 移除组件 | 完整配置中的作用 | 移除后的替代 |
|---|---|---|---|
| M1 | RH：学习关系头去重 | same-instance / part-whole relation dedup | 保留类别 grouping，组内改为 box-IoU NMS@0.5 |
| M2 | ADF：自适应密度与置信度过滤 | FSC 使用 dense routing + conf=0.2；OmniCount 使用既有 conf=0.1 adapter | 固定基础候选池，confidence threshold=0；其他组件不变 |
| M3† | CP：COCO relation pretraining | COCO 预训练后 FSC dot-supervised 微调 | 原实现只替换 pts32 关系头；不是完整 leave-one-out，已由严格配对实验取代 |
| M4 | HR：高分辨率候选/训练池 | pts32 candidate pool + pts32 heads | pts16 candidate pool + pts16 category/Exp5-C relation heads |
| M5 | T4：4x4 极端密度 rescue | fast 前端零候选时启用 4x4 tiled proposal | 关闭 T4，保留原 multi-resolution frontend |
| M6 | 完整模型 | RH + ADF + CP + HR + T4 | 无移除 |

## 3. 数据集适配与固定口径

### 3.1 FSC-147 Full

- 图像数：1,190，包含 `7611.jpg`。
- 主结果 anchor：M6 MAE=12.668067 / RMSE=113.711122。
- 学习权重：与 FSC147 12.67 主方法完全相同。
- 阈值：`tau_inst=0.99`、`tau_affinity=0.1`、`conf_threshold=0.2`。
- T4 trigger：`fsc147_test_fast/<image>.pt` 的 `n_candidates==0`，不读取 GT；只触发 `7611.jpg`。
- M1-M4 均保留 T4；只有 M5 关闭 T4。该定义修复旧表把 pre-rescue M1-M5 与 post-rescue M6 混算的问题。
- 协议限制：当前 FSC cache 的 `valid` 来自 GT dot coverage，因此是 cache-compatible 消融，不是 strict no-GT inference。
- 训练隔离审计：旧 fast/pts32 category seed42 分别有 1,002/1,018 张 official-test 图进入 train；旧 relation seed42 分别有 1,066/1,075 张 official-test 图进入 train。因此旧 M3/M6 只保留为工程回归结果，不再支持 CP 因果结论。

### 3.2 OmniCount-191 Full

- 图像数：1,957；93 个测试类别；3,754 个非零 image-class pairs。
- 推理不读取 cache 的 `valid` 或 `matched_class`，GT 只参与最终评测。
- 使用 FSC147 同一 category/relation 学习权重，只替换 OmniCount 93 类文本原型。
- pts32 cache：`/home/czp/ws_yiyang/ovcud_cache/omnicount_test`，1,957/1,957。
- pts16 cache：`/home/czp/ws_yiyang/ovcud_cache/omnicount_test_pts16`，本轮新生成，1,957/1,957。
- OmniCount `conf_threshold=0.1` 沿用 2026-07-08 已建立的 full-test adapter，不在本轮消融后重新调参。
- 额外保留 `conf_threshold=0.2` 作为跨域敏感性审计，不作为主消融行。
- 当前 pts32 cache 没有零候选图，T4 trigger 命中 0/1,957，因此 M5 与 M6 必须逐图一致。

## 4. 待跑矩阵与完成状态

| Variant | FSC-147 full MAE/RMSE | OmniCount total MAE/RMSE | OmniCount mRMSE/mRMSE-nz | 状态 |
|---|---:|---:|---:|---|
| M1 - RH | 14.29 / 113.91 | 4.68 / 8.46 | 0.457 / 3.911 | ✅ full |
| M2 - ADF | 26.97 / 124.52 | 12.73 / 17.30 | 0.842 / 3.864 | ✅ full |
| M3† - CP | 12.59 / 113.69 | 4.68 / 8.46 | 0.457 / 3.911 | ⚠️ legacy partial/leaky |
| M4 - HR | 27.18 / 125.83 | 5.64 / 9.72 | 0.424 / 3.914 | ✅ full |
| M5 - T4 | 13.47 / 126.74 | 4.68 / 8.46 | 0.457 / 3.911 | ✅ full |
| **M6 full** | **12.67 / 113.71** | **4.68 / 8.46** | **0.457 / 3.911** | ✅ full |

## 5. 执行流程

1. ✅ 审计原 M1-M6 定义及 12.67 rescue policy。
2. ✅ 修改 OmniCount 预处理器，零候选图保留占位 cache，不再静默跳过。
3. ✅ 生成 OmniCount pts16 full-1,957 cache，并验证与 pts32 文件名集合完全一致。
4. ✅ 双数据集 smoke test。
5. ✅ FSC147 full M1-M6，M6 硬性 anchor 校验。
6. ✅ OmniCount full M1-M6，验证既有 M6 4.6765/8.4635。
7. ✅ 1,000 次 image bootstrap 和 paired MAE delta CI。
8. ✅ 逐图唯一 ID、图像数、M5=M6 与指标复算检查。
9. ✅ 中文报告、主报告更新及 Git 归档。
10. ✅ CP/scratch 在 official-train-only 数据上各跑 3 seeds，同时训练 pts16/pts32；`tau_inst` 只由 official val 选择，再冻结到 full test。

## 6. 主命令

OmniCount pts16 cache：

```bash
/home/czp/ws_yiyang/FreeCounting/venv/bin/python3 \
  script/preprocess_omnicount.py \
  --pts-per-side 16 \
  --out-dir /home/czp/ws_yiyang/ovcud_cache/omnicount_test_pts16 \
  --device cuda
```

双数据集评测：

```bash
/home/czp/ws_yiyang/FreeCounting/venv/bin/python3 \
  script/eval_fsc147_omnicount_leaveoneout.py \
  --dataset both \
  --conf-threshold 0.1 \
  --bootstrap 1000 \
  --out-prefix result/logs/fsc147_omnicount_leaveoneout_full \
  --device cuda
```

实际执行为先 FSC147、再 OmniCount 两次调用；summary writer 会按相同 checkpoint hash 合并两个数据集，避免长任务失败后整轮重跑。

## 7. 验收条件

- FSC147 每个变体 1,190 个唯一 image ID，M6 精确复现 12.668067/113.711122。
- OmniCount 每个变体 1,957 个唯一 image ID，M6 精确复现既有 4.676546/8.463454。
- OmniCount pts16/pts32 cache 文件名集合完全一致。
- OmniCount M5/M6 逐图相同；FSC147 M5/M6 只允许 T4 触发图变化。
- 所有结果包含配置、逐图预测、总指标、切片、bootstrap 和 paired delta。
- 论文不声称 M1-M5 在两个数据集上都必然退化；OmniCount 的零变化必须如实报告。
- 原 M3 不进入因果消融结论。FSC147 主结果维持 12.67；CP 从主组件表删除，严格配对实验放入补充材料。当前 12.67 checkpoint 仍含 COCO 初始化，真正的 CP-free 主 checkpoint 需要按同一主协议另行重训复评。

## 8. CP 严格配对补充实验

### 8.1 配置

- official train 内固定数据划分 seed `20260710`，official test 进入训练的图像数为 0。
- 模型 seeds：`17 / 42 / 73`。
- CP 与 scratch 每个 seed 都同时训练 pts16、pts32 关系头；固定相同 train-only category heads、文件顺序、pair sampling 与训练超参。
- 每个完整 pts16+pts32 模型在 official val 1,286 张上独立选择一个共同 `tau_inst`，再冻结评估 official test 1,190 张。
- 六次均选择预注册网格上界 `0.999`；不能把该值称为内部最优点。

### 8.2 结果

| Condition | FSC-147 test MAE mean±std | RMSE mean±std | MAE w/o 7611 |
|---|---:|---:|---:|
| CP | **13.9706 ± 0.0933** | **111.2808 ± 0.0174** | **12.9840 ± 0.0934** |
| Scratch | 13.9714 ± 0.1214 | 111.2902 ± 0.0112 | 12.9843 ± 0.1223 |

`CP - Scratch` paired ΔMAE = **-0.00084 ± 0.03288**；逐图 bootstrap 95% CI **[-0.04398, +0.04314]**。三个 seed 的差值为 `+0.0151/+0.0210/-0.0387`，方向不一致。结论是 CP 对最终 counting MAE **无可测边际收益，也无有害证据**。训练端 loss/precision 改善不能替代该 test 结论。

严格实验仍沿用 GT-dot-derived cache `valid` 和 GT-count-derived 高密度 cache membership，因此解决的是 CP 的 train/test overlap 与完整移除问题，不等价于 strict no-GT inference。详细协议、切片与逐图产物见 `docs/fsc147_cp_strict_multiseed_report_20260710.md`。
