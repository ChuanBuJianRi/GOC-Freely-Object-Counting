# 实验计划：Leave-One-Out 消融 + MCAC 多类别对比

**日期**: 2026-07-09
**对应论文位置**: `AnonymousSubmission2027.tex` 中 `tab:ablation`（消融主表）与 `tab:mcac`（MCAC 对比表），两处都有待填的实验单元格。

> **2026-07-10 执行状态**：MCAC full 2,115、M1-M6、OCCAM shared-proposal baseline 和 strict no-GT M5/M6 已完成。审计发现旧主评测使用 GT-dot-derived `valid` 筛选候选；cache-compatible 消融只作诊断，论文 MCAC 对比采用 strict no-GT 行。完整记录见 `docs/mcac_full2115_leaveoneout_report_20260710.md`。

> **主消融范围更新**：论文组件 leave-one-out 主表现只使用 FSC-147 full + OmniCount-191 full；MCAC 留作独立跨域诊断。新计划与结果见 `docs/experiment_plan_ablation_fsc147_omnicount_20260710.md`。

---

## 实验一：组件 Leave-One-Out 消融（M1–M6）

### 1.1 目的

论文消融表已从"累积添加式"改为 **严格 leave-one-out**：M1–M5 每行只从完整模型 M6 中移除一个组件，直接给出每个组件的边际贡献。消融不再跑 sample100，统一改为 **FSC-147 full test 1,190 张 + MCAC test**，各报 MAE / RMSE。

### 1.2 组件定义

| 缩写 | 组件 | 移除后的替代 |
|---|---|---|
| RH | 学习关系头（same-instance / part-whole 去重） | box-IoU heuristic 去重 |
| ADF | 自适应候选密度 + 置信度过滤（density_threshold=50, conf_threshold=0.2） | 固定密度、不过滤 |
| CP | COCO 关系预训练 + FSC-147 微调 | 仅 FSC-147 dot-supervised 训练 |
| HR | 32×32 高分辨率训练候选池 | 16×16 候选池 |
| T4 | 4×4 tiling 高密度候选生成 | 不使用 4×4 tiling，仅保留非 tiled / 原多尺度候选 |

### 1.3 变体配置与待跑清单

| No. | 配置 | 类型 | FSC-147 full | MCAC | 说明 |
|---|---|---|---|---|---|
| M1 | 完整配置，去重换成 box-IoU | 纯推理开关 | ✅ 15.10 / 126.91 | ✅ 35.26 / 58.30 | MCAC 为 cache-compatible 诊断 |
| M2 | 完整配置，关闭 ADF | 纯推理开关 | ✅ 28.50 / 143.38 | ✅ 31.02 / 53.13 | MCAC 上优于 M6，显示阈值跨域失配 |
| M3 | pts32 关系头，仅 dot-supervised（无 COCO 预训练+微调） | 已重训 | ✅ 13.40 / 126.72 | ✅ 35.01 / 57.97 | metadata 已修正为 m3 |
| M4 | pts16 Exp5-C 微调头（= 完整模型减 HR） | 已有 checkpoint | ✅ 28.53 / 143.39 | ✅ 39.99 / 65.79 | HR 在 MCAC 上显著有效 |
| M5 | 完整配置，关闭 4×4 tiling | 纯推理开关 | ✅ 28.84 / 143.54（旧 no-all-tiling 定义） | ✅ 34.94 / 57.90 | MCAC M5 仅关闭 3-image rescue |
| M6 | 完整模型 | 已有 | ✅ 12.67 / 113.71（4x4 rescue exploratory） | ✅ strict 32.11 / 53.79；诊断 34.94 / 57.90 | 同一学习权重，无 MCAC 微调 |

已有 full-test 数字出处：`result/logs/fsc147_multires_extended.json` / 论文 FSC-147 主表。旧的 8.73、9.11 等 sample100 数字只作为历史记录，不再填入消融主表。

**统一口径提醒**：FSC147 M1-M5 是以 pre-rescue M6=13.47 为基线的旧 leave-one-out；12.67 只改变 `7611.jpg` 的 4x4 rescue。不能把 M1-M5 直接与 12.67 计算组件 delta。若论文最终采用 12.67 为 M6，需先重新定义 ADF/T4 的边界并在同一 rescue policy 下重导 FSC147 M1-M5。

### 1.4 注意事项

- 所有变体固定 τ_inst=0.99、同一 full-test 候选 cache、同一分类头，只动被消融的组件；M5 只替换/关闭 4×4 tiling 候选，不改关系头和分类头。
- 正文当前用 16×16 配置下的旧证据（9.42↔19.39、9.11↔9.42、CARPK 8.44↔6.92）临时支撑 M1–M3 的论述，**跑完 full-test leave-one-out 后要替换**。
- 不再跑 FSC-147 sample100 消融，也不再用 sample100 数字支撑主文因果结论。可选输出 bootstrap 95% CI（1000 次重采样），但 CI 基于 full 1,190 张图。
- 参考脚本：`script/run_p1_classification_ablation.py`（消融评估框架）、`script/preprocess_fast_unified.py`（候选 cache）。

---

## 实验二：MCAC 多类别对比

### 2.1 目的

在 MCAC（ABC123 提出的 multi-class class-agnostic counting 基准，每图 1–4 类、每类 1–300 个物体）上，与**仅有的几个真正多类别计数方法**对比。已确认不存在其他有 MCAC 公开数字的多类别方法。

### 2.2 对比表构成（论文 `tab:mcac`）

| 方法 | 协议 | 数字来源 | 状态 |
|---|---|---|---|
| ABC123 | prompt-free，density 监督，匿名 per-class 密度图 | published + 官方 checkpoint 本地全量复现 | ✅ published 9.52 / 17.64；local full-2115 9.46 / 17.52 |
| OCCAM | prompt-free，training-free | 本地 shared-pts32 复现；非论文公开 MCAC 数字 | ✅ 22.74 / 38.89 |
| UniCounting (ours) | prompt-free，count-supervision-free；MCAC 仅匿名分组 | 本地 strict no-GT | ✅ 32.11 / 53.79 |
| OmniCount†（可选） | 输入 GT 类表的 prompted 参考行 | 本地跑或删行 | ⬜ 可选 |

### 2.3 数据与环境

- MCAC 已解压到 `/home/czp/ljs/dataset/MCAC`，本地压缩包为 `/home/czp/ljs/dataset/MCAC.zip`；数据不提交 Git 仓库。
- ABC123 官方 checkpoint 已在 MCAC full 2,115 上复现为 9.46/17.52；官方 `drop_last=True` 的 2,114 张口径为 9.45/17.51。详见 `docs/abc123_mcac_reproduction_report_20260710.md`。
- OCCAM 本地复现入口参考 `result/logs/occam_dot_recall.json` 对应的评估脚本。

### 2.4 评测协议（关键，写论文前先定）

- 指标：per-class counts 的 MAE / RMSE，与 ABC123 论文的 MCAC 表同口径。
- **类别对齐**：MCAC 是合成数据、类别无自然语言名，不能用文本类名匹配。ABC123 的量化协议是把无序预测组与 GT 密度图做匹配（Hungarian）；我们建议对 UniCounting 预测组采用同样的**最优匹配协议**（预测组代表 mask/box 与 GT per-class dot 的空间重合做匹配），并在论文 caption 中注明与 ABC123 协议一致。
- UniCounting 侧只替换文本原型不可行（无类名），直接用 class-agnostic 分组 + 匹配评估即可；这一点与 OmniCount-191 的开放词表协议不同，正文叙述时注意区分。
- 2026-07-10 审计：cache 中 `valid` 来自 GT dot coverage。论文主对比必须使用 `--candidate-filter all`；旧 `gt_dot_valid` 结果只能作为内部诊断。

### 2.5 流程建议

1. ✅ 解压 MCAC，生成并补齐 pts32/pts16 full 2,115 cache。
2. ✅ M6 完整模型跑 MCAC test，另补 strict no-GT M6。
3. ✅ M1/M2/M4/M5 推理开关变体跑 MCAC。
4. ✅ 重训 M3，并完成 FSC147 full + MCAC。
5. ✅ OCCAM shared-pts32 复现跑 MCAC；原生 OCCAM AMG 仍为可选补充。
6. ⬜ OmniCount prompted 参考行未跑，不填主表。
7. ✅ ABC123 官方 checkpoint full-2115 复现，published 指标得到验证。
8. ✅ 原始结果与机器可读汇总输出到 `result/logs/`。

### 2.6 优先级

M6-on-MCAC 和 UniCounting 对比行最优先（两张表共用）；其次 M1/M2/M4/M5（纯推理，便宜）；M3 重训和 OCCAM-on-MCAC 最后。
