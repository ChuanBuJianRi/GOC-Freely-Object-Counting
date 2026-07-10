# FSC-147 Strict No-GT、CP-Free 全量重跑报告

**日期**：2026-07-10

**状态**：执行中；尚未读取新 strict test 指标

## 1. 目的

重新审计 FSC-147 `12.67 / 113.71` 的完整实现路径，删除所有 validation/test GT 参与预测的分支，并将 COCO relation pretraining 替换为 official-train-only FSC dot-supervised scratch relation heads。所有阈值和路由只在 official validation 上选择，配置冻结后才运行 full test 1,190。

## 2. 原 12.67 路径审计

| 模块 | 原配置 | 审计结论 |
|---|---|---|
| Candidate gate | cache `valid` 与 category confidence 取交 | `valid` 由候选是否覆盖 GT dot 决定；test GT 直接改变预测，属于信息泄露 |
| MR routing | 通过 `mr51/mr100` cache membership 选择前端 | cache 集合由 GT count 51-100/100+ 建立，属于隐式 GT 路由 |
| Category heads | `category_cosine_fast/pts32.pt` | 旧训练目录覆盖 train/val/test，训练 split 存在 official-test overlap |
| Text vocabulary | FSC-147 全 147 类原型 | FSC split 类别互斥；其中 val/test 独占的 58 个类名在旧 category 训练中作为负类出现，属于 transductive label-space access |
| Relation heads | `fsc147_relation_best/pts32_best.pt` | 含 `coco_relation_1152.pt` 初始化，且旧 FSC 训练 split 存在 overlap |
| T4 | fast raw candidate count为 0 时使用 4x4 | 触发值本身由图像候选产生，不读取 GT；保留为预声明 failure fallback |
| Test metric | full 1,190，MAE 12.6681 | 只能作为 GT-assisted/cache-compatible 历史结果，不能代表 strict no-GT 性能 |

原预测链的核心泄露代码包括：

- `script/eval_fsc147_full.py:182`：读取 test cache 的 GT-derived `valid`。
- `script/ablation_fsc147_multires_components.py:155-157`：所有 12.67 组件分支共用该门控。
- `script/eval_fsc147_omnicount_leaveoneout.py:197-203`：通过 GT-selected cache 是否存在决定 fast/MR51/MR100。

## 3. 新 strict no-GT 配置

### 3.1 允许与禁止字段

Validation/test inference cache 只允许：

`schema, img_id, file_name, z, bbox, height, width, source_cache`

硬性禁止：

`valid, purity, coverage, iou, matched_class, matched_instance_id, gt_count, class_name, is_part, is_countable, points`

加载器发现任一禁止字段会直接终止。GT annotation 在全部 1,190 张预测生成后才读入并用于计算指标。

### 3.2 学习组件

| 组件 | 数据 | 配置 |
|---|---|---|
| text prototypes | official train 类名 only | OpenCLIP ViT-B-32/LAION；89 类；与 val/test 类名交集为 0 |
| pts16 category | official train only | `cp_strict/category_pts16_trainvocab.pt`；内部 model-val top-1 85.99% |
| pts32 category | official train only | `cp_strict/category_pts32_trainvocab.pt`；内部 model-val top-1 85.02% |
| pts16 relation | official train dots only | `relation_pts16_trainvocab_scratch_seed{17,42,73}.pt`；禁止 `pretrained_from` |
| pts32 relation | official train dots only | `relation_pts32_trainvocab_scratch_seed{17,42,73}.pt`；禁止 `pretrained_from` |
| pts16 candidate filter | official train dots only | 2-layer MLP；内部 model-val P=92.89%，R=93.38% |
| pts32 candidate filter | official train dots only | 2-layer MLP；内部 model-val P=91.77%，R=92.01% |

Candidate filter 只在 train 阶段把“候选覆盖至少一个 train dot”作为监督；validation/test 只使用预测概率。这属于 FSC point supervision，不是 density-map supervision，但不能表述为完全无 point/count information。

原型由 `build_fsc147_train_vocabulary.py` 直接读取 official-train cache 中的类 ID/类名后编码；脚本不读取 `ImageClasses_FSC147.txt`、validation/test image list 对应的类别，也不加载旧 147 类原型。训练器显式完成 global ID 到 89 类 local ID 的映射，并把原型及 metadata 哈希写入 checkpoint。

### 3.3 前端与路由

- fast：所有图统一生成 pts16 full-image safe cache。
- tiled：所有图统一生成 pts32、2x2、overlap=0.25 safe cache；不再只给 GT>50 图生成。
- 路由输入只使用 fast 的预测 count；阈值在 official val 上选择后冻结。
- raw fast candidate count为 0 的触发条件本身不读取 `valid` 或 GT count；但历史 4x4 配方是在查看 test `7611.jpg` 后形成，当前不能直接计入 strict 主结果。
- 不通过 cache 文件是否存在判断密度；val/test fast 与 tiled cache 必须分别精确覆盖完整 split。

T4 是否进入 strict 主结果将只由 official-train 的 3 个 fast-zero failure cases（`2737.jpg`、`2979.jpg`、`7454.jpg`）选择；若 train-side 对照不能支持 4x4，则 strict 主结果关闭 T4，历史 7611 rescue 只作 post-hoc 诊断。

## 4. Validation-only 选择

预注册网格：

- candidate filter threshold：`0, 0.05, 0.1, 0.2, 0.3, 0.4`
- category confidence threshold：`0, 0.1, 0.2, 0.3`
- `tau_inst`：`0.99, 0.999`
- predicted fast count route threshold：`10, 20, 30, 40, 50, 75, 100`，并比较 always-fast/always-tiled

pts16/pts32 frontend 参数先分别按三个 relation seeds 的 validation mean MAE 选择，再选择一个共同路由策略。test 不参与任何选择。

> 待 validation safe tiled cache 完成后填入冻结配置与 validation 指标。

## 5. Full Test 1,190

> 待冻结配置提交后执行。主表将报告三个 scratch relation seeds 的 MAE/RMSE mean±std、逐 seed bootstrap 95% CI、GT count 分桶与 `7611.jpg` 预测。

## 6. 复现入口

- `script/train_candidate_filter.py`：official-train-only candidate filter。
- `script/build_fsc147_train_vocabulary.py`：从 official train 类名直接构建 89 类原型。
- `script/train_category_v2.py --prototype_label_map`：global→train-local 标签映射。
- `script/train_relation_1152.py --require_train_only_vocabulary`：禁止 COCO 初始化并绑定上游资产哈希。
- `script/export_fsc147_inference_cache.py`：物理删除 GT 字段。
- `script/preprocess_fsc147_tiled_nogt.py`：不加载 annotation 的 tiled candidate 生成。
- `script/eval_fsc147_strict_nogt.py`：validation freeze 与 test 两阶段入口。
- `result/configs/fsc147_nogt_fast_zero_images.json`：由 raw fast candidate count 导出的 T4 预计算集合。
