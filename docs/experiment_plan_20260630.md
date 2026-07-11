# OV-CUD 实验计划 (2026-06-30)

## 当前状态

### 已解决问题

| 问题 | 状态 | 方案 |
|---|---|---|
| 分类头 train→test 迁移 | ✅ 已解决 | 统一 SAM2 配方 (pts_per_side=16) + 三路 DINOv2 (1152-dim) + 统一 dot matching |
| Train/Test 候选配方不一致 | ✅ 已解决 | `preprocess_fast_unified.py` 统一处理两组数据 |
| 特征维度不足 (384→1152) | ✅ 已解决 | 三路 DINOv2 (masked + box + context) concat |
| 端到端 Pipeline 缺失 | ✅ 已解决 | clustering + dedup + representative selection + counting |

### 当前指标

| 指标 | 旧方案 | 当前方案 |
|---|---|---|
| 分类头 Test top1 | 0.0% | **96.15%** |
| 分类头 Test top5 | 0.07% | **98.52%** |
| 端到端 MAE (Oracle-All) | 3.21 | 16.98 |
| 端到端 MAE (Real Cat + Heuristic Dedup) | 76.69 | **39.64** |

### 剩余瓶颈

1. **去重模块**：旧关系头为 384 维特征训练，不兼容 1152 维。当前 fallback 到 bbox-IoU 启发式，完全不工作
2. **候选密度**：pts_per_side=16 的候选密度低于 OCCAM-M (8px grid)，Oracle-All MAE 16.98 vs 3.21
3. **超参数**：聚类阈值、去重阈值未系统调优

---

## 实验矩阵

### Exp 1: 重新训练关系头 (1152-dim)

**目的**: 替换 bbox-IoU 启发式去重，提供准确的 A_sem / A_inst / A_part

**方案**:
- 使用 `code/training/train_relation.py` (从 ws_yiyang 复制)
- 输入: 1152-dim DINOv2 特征 + 147-class category probs
- Pairwise feature dim: 4×1152 + 2 + 6 = 4616
- 训练数据: `fsc147_train_fast` (统一 SAM2 配方，5376 图)
- 冻结: Category Head (category_cosine_fast.pt)

**超参**:
- hidden_dim=512, num_layers=3, dropout=0.1
- lr=5e-4, batch=1 (按图), epochs=20
- neg_ratio=5.0, pos_weight=3.0
- max_cand=64, max_pairs=4096

**预估时间**: ~30min (RTX 4090)

**验收指标**:
- inst_AUC > 0.95, inst_AP > 0.85
- 端到端 MAE 从 39.64 降至 < 25

---

### Exp 2: 高密度 SAM2 候选

**目的**: 提升候选 dot recall，降低 Oracle-All 理论上限

**方案**:
- 测试 `pts_per_side=24` 和 `pts_per_side=32`
- 与当前 `pts_per_side=16` 对比 dot recall 和 Oracle-All MAE
- 在 sample100 测试子集上快速评测

**预估时间**: 
- 候选生成: ~40min (1190 测试图, ~2s/img for pts=32)
- 评测: ~5min

**验收指标**:
- Oracle-All MAE 从 16.98 降至 < 10
- Dot recall 从 ~95% 提升至 > 97%

**风险**: 候选数增加 → 推理时间增长。需在精度/速度间平衡

---

### Exp 3: 调优聚类/去重超参数

**目的**: 在 Exp 1 (新关系头) + Exp 2 (高密度候选) 基础上，调优关键超参

**方案**:
- `tau_inst` sweep: [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
- `tau_affinity` sweep: [0.1, 0.2, 0.3, 0.5]
- 在 val 集上选最优组合，测试集上报最终结果

**预估时间**: ~10min per sweep × 24 combos ≈ 4h (可并行)

**验收指标**:
- 端到端 MAE 降至目标 < 15
- 定位去重 vs 聚类各自的误差贡献

---

## 执行顺序

```
Exp 1 (关系头重训) → Exp 2 (高密度候选) → Exp 3 (超参调优)
     ↓                      ↓                      ↓
  ~30min                 ~45min                 ~4h
```

### 最终目标

| 指标 | 当前 | Phase 1 目标 | Phase 2 目标 |
|---|---|---|---|
| 分类头 Test top1 | 96.15% | 96%+ | 97%+ |
| 端到端 MAE | 39.64 | < 20 | < 15 |
| Oracle-All MAE | 16.98 | < 10 | < 5 |
| 100+ 区间 MAE | 143.69 | < 60 | < 30 |

---

## 文件清单

### 新增文件
- `script/preprocess_fast_unified.py` — 统一 SAM2-DINOv2 预处理
- `script/train_category_v2.py` — 余弦头训练 (1152-dim)
- `script/run_counting_pipeline.py` — 端到端计数 pipeline
- `script/diag_oracle_category_dedup.py` — Oracle 诊断
- `code/clustering/first_neighbor.py` — First-neighbor 聚类
- `code/counting/deduplicate.py` — Same-instance 去重
- `code/counting/representative.py` — 代表选择
- `code/heads/relation_head.py` — 关系头 (从 ws_yiyang 复制)
- `code/matrix/pairwise_features.py` — 候选对特征构造

### 产出 Checkpoint
- `result/checkpoints/category_cosine_fast.pt` — 统一配方余弦头 (top1=84.7% val, 96.15% test)
- `result/checkpoints/category_cosine_v2.pt` — 早期 384-dim 余弦头
- `result/checkpoints/category_cosine_3view.pt` — 1152-dim 余弦头 (旧 SAM2)

### 缓存数据
- `fsc147_train_fast` — 统一训练缓存 (pts=16, 1152-dim, 5376 图)
- `fsc147_test_fast` — 统一测试缓存 (pts=16, 1152-dim, 6142 图)
