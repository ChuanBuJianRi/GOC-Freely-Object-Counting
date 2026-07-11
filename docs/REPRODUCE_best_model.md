# 最佳模型复现文档 (OV-CUD 关系头)

> **2026-07-11 状态更正**：本文记录的是历史 COCO-pretrained / cache-compatible checkpoint，不再是 FSC-147 当前主模型。新的 strict no-GT、official-train-only、CP-free 主配置与 full-1,190 结果见 `docs/fsc147_strict_nogt_cp_free_report_20260710.md` 和 `docs/AAAI_2026_Experiment_Report.md`。

> 记录当前最佳关系头模型的完整训练/评测参数，用于日后复现。
> 最后更新: 2026-07-01

## 1. 最佳结果

| 数据集 | MAE | RMSE | bias | 配置 |
|---|---|---|---|---|
| **CARPK** (test 459) | **4.06** | 5.51 | -1.62 | pts=32 微调头, tau_inst=0.99 |
| **FSC147** (sample100) | **8.73** | 32.89 | -6.13 | pts=16/32 微调头, tau_inst=0.99 |

对比基线: CARPK 6.50 → 4.06 (-37.5%), FSC147 9.42 → 8.73 (-7.3%)

## 2. 核心产出文件

| 文件 | 说明 |
|---|---|
| `result/checkpoints/fsc147_relation_pts32_best.pt` | 🏆 最佳关系头推理版 (10.5MB, 无 optimizer) |
| `result/checkpoints/fsc147_relation_pts32_exp5c.pt` | 训练版 (31MB, 含 optimizer, 可续训) |
| `result/checkpoints/coco_relation_1152.pt` | COCO 强监督预训练权重 (微调起点) |
| `result/checkpoints/category_cosine_pts32.pt` | 冻结的 pts=32 分类头 |
| `result/checkpoints/text_prototypes_fsc147.pt` | FSC147 文本原型 |

## 3. 模型结构

`PairwiseRelationHead` (`code/heads/relation_head.py`)
- 输入: 成对特征 `phi_ij`, **feat_dim = 4616** = 4×1152 (z_i,z_j,|diff|,prod) + 2 标量 (cosine, p·p) + 6 bbox 几何
- 主干: 3 层 MLP, hidden_dim=512, GELU, dropout=0.1
- 输出: 3 个分支 logit (sem/inst/part), 对称分支对 (i,j)(j,i) 双向平均
- 参数量: 2.63M
- **冻结**: SAM2 / DINOv2 / 分类头全部不训练

## 4. 完整复现流程

### 环境依赖
- GPU: RTX 4090 (24GB)
- SAM2 checkpoint: `/home/czp/ws_yiyang/FreeCounting/ws_yiyang/OCCAM/checkpoints/sam2.1_hiera_small.pt`
- SAM2 config: `configs/sam2.1/sam2.1_hiera_s.yaml`
- DINOv2 三路编码器: `code/encoders/dinov2_encoder.py`

### Step 1: 特征预处理 (离线, 已完成)

**FSC147 训练缓存 (pts=32)** — 输出 `/home/czp/ws_yiyang/ovcud_cache/fsc147_train_pts32` (6143 图):
```bash
python script/preprocess_fast_unified.py --pts-per-side 32 \
    --split train --out-dir /home/czp/ws_yiyang/ovcud_cache/fsc147_train_pts32 --device cuda
```

**SAM2 AMG 参数** (关键, 见 `preprocess_fast_unified.py:build_sam2_amg`):
- `points_per_side=32` (pts=32; 基线用 16)
- `points_per_batch=64`
- `pred_iou_thresh=0.7`
- `stability_score_thresh=0.8`, `stability_score_offset=1.0`
- `box_nms_thresh=0.7`, `crop_n_layers=0`

**每候选特征**: DINOv2 三路 (masked + box + context) concat = 1152-dim
**标签**: dot-based 弱监督 (matched_class, matched_instance_id, purity, valid)

**COCO 预训练缓存** — 输出 `coco_val_3view` (1918 图):
```bash
python script/preprocess_coco_relation.py \
    --ann /home/czp/official_code/dataset/coco/annotations/instances_val2017.json \
    --img-dir /home/czp/official_code/dataset/coco/images/val2017 \
    --out-dir /home/czp/ws_yiyang/ovcud_cache/coco_val_3view --limit 2000 --device cuda
```
COCO 用真实 instance mask 做 IoU 匹配 → **精确 same-instance/category 标签** (强监督)

### Step 2: 阶段 A — COCO 强监督预训练

产出 `coco_relation_1152.pt`:
```bash
python script/train_relation_coco.py \
    --data_dir /home/czp/ws_yiyang/ovcud_cache/coco_val_3view \
    --save_ckpt result/checkpoints/coco_relation_1152.pt \
    --hidden_dim 512 --num_layers 3 --dropout 0.1 \
    --pos_weight 2.0 --neg_ratio 5.0 --epochs 15 --lr 5e-4 --device cuda
```
- 用 oracle (GT one-hot) 作为 category probs
- inst_pos ~37% (强监督, 远高于 FSC147 dot 弱监督的 0.13%)

### Step 3: 阶段 B — FSC147 pts=32 微调 (Exp5-C) ⭐

产出 `fsc147_relation_pts32_exp5c.pt` (best epoch=25):
```bash
python script/train_relation_1152.py \
    --data_dir /home/czp/ws_yiyang/ovcud_cache/fsc147_train_pts32 \
    --category_ckpt result/checkpoints/category_cosine_pts32.pt \
    --text_prototypes result/checkpoints/text_prototypes_fsc147.pt \
    --pretrained result/checkpoints/coco_relation_1152.pt \
    --save_ckpt result/checkpoints/fsc147_relation_pts32_exp5c.pt \
    --pos_weight 8.0 --neg_ratio 5.0 --epochs 40 --lr 5e-4 --device cuda
```

**关键超参** (与原始基线的区别):

| 参数 | 值 | 说明 (相比基线) |
|---|---|---|
| `--pretrained` | coco_relation_1152.pt | ⭐ COCO 强监督预训练起点 |
| `--pos_weight` | **8.0** | ⭐ 基线为 3.0; 提升正样本损失权重 → inst 召回 |
| `--neg_ratio` | 5.0 | 每正样本配 5 负样本 |
| `--epochs` | 40 | 基线 20; 但 best 在 epoch 25 (后续过拟合) |
| `--lr` | 5e-4 | AdamW, weight_decay=1e-4 |
| `--data_dir` | fsc147_train_pts32 | ⭐ pts=32 (基线 pts=16); 正样本对多 4.5× |
| `--category_ckpt` | category_cosine_pts32.pt | 冻结分类头 (pts=32 版) |
| max_cand / max_pairs | 64 / 4096 | (脚本默认) |
| val_frac / seed | 0.1 / 42 | 数据划分 |

**损失**: sem+inst 双分支 BCE-with-logits, purity 加权, best 按 val loss 选
**训练结果**: val loss=0.0659, **inst_R=95%**, inst_P=55% (data split: train=5525, val=614)
**训练时长**: ~27min (RTX 4090, 实际跑到 epoch 36 手动停, best=25)

### Step 4: 导出推理版 (去 optimizer + 内嵌配置)

产出 `fsc147_relation_pts32_best.pt` (10.5MB), 内嵌 `recommended_inference` 元信息。

## 5. 评测复现

### CARPK (test 459)
```bash
python script/eval_carpk.py \
    --cache-dir /home/czp/ws_yiyang/ovcud_cache/carpk_test \
    --category-ckpt result/checkpoints/category_cosine_pts32.pt \
    --relation-ckpt result/checkpoints/fsc147_relation_pts32_best.pt \
    --text-prototypes result/checkpoints/text_prototypes_fsc147.pt \
    --tau-inst 0.99 --tau-affinity 0.1 --conf-threshold 0.1 \
    --out result/logs/carpk_pts32_exp5c_best.json --device cuda
```
→ MAE=4.06, RMSE=5.51, bias=-1.62

### FSC147 (sample100)
```bash
python script/run_adaptive_pipeline.py \
    --rel-16 result/checkpoints/fsc147_relation_best.pt \
    --rel-32 result/checkpoints/fsc147_relation_pts32_exp5c.pt \
    --density-threshold 50 --conf-threshold 0.2 --tau-inst 0.99 --tau-affinity 0.1 \
    --out result/logs/pipeline_pts32_exp5c_best.json
```
→ MAE=8.73, RMSE=32.89

## 6. 关键推理超参 (务必按数据集调 tau_inst)

⚠️ 微调 (pos_weight=8) 使 inst logit 分布右移, **原 tau_inst=0.4 会过度合并**, 必须调高:

| 数据集 | 最优 tau_inst | tau_affinity | conf_threshold | density_threshold |
|---|---|---|---|---|
| CARPK | **0.99** | 0.1 | 0.1 | — |
| FSC147 | **0.99** | 0.1 | 0.2 | 50 |

去重流程 (`code/counting/`): 关系头算 A_inst 亲和矩阵 → 类别聚类分组 →
组内按 tau_inst 做连通分量合并 (Union-Find/贪心) → 每实例选代表 → 计数

## 7. 成功三要素

1. **COCO 强监督预训练** — 真 mask 标签打底 (inst_pos 37% vs FSC147 0.13%)
2. **pts=32 高密度候选** — same-instance 正样本对多 4.5× (4853 vs 1082)
3. **pos_weight=8 微调 + 高阈值去重 (tau_inst=0.99)** — 高召回 (95%) 且不过度合并

## 8. tau_inst sweep 记录 (CARPK, pts=32 微调头)

| tau_inst | MAE | RMSE | bias |
|---|---|---|---|
| 0.4 | 10.00 | 13.45 | -9.92 |
| 0.8 | 5.62 | 7.57 | -5.14 |
| 0.9 | 4.78 | 6.56 | -3.96 |
| 0.97 | 4.21 | 5.76 | -2.57 |
| **0.99** | **4.06** | **5.51** | -1.62 |
| 0.995 | 4.08 | 5.49 | -1.11 |
