# OV-CUD: Open-Vocabulary Counting Unit Discovery

## 1. 框架评估结论

整体思路成立：先用 SAM2 生成过完备候选，再用区域语义和候选间关系把候选组织成语义计数组，最后在组内做同实例去重并选择代表候选计数。它的核心优势是把 counting 转化为：

```text
候选发现 + 语义命名 + 关系建模 + 代表实例计数
```

而不是直接训练图像级 count regressor 或密度图积分。

需要澄清和修正的点如下：

| 问题 | 风险 | 修订方式 |
|---|---|---|
| `open-vocabulary` 与 `category-supervised` 表述容易冲突 | 如果只训练闭集分类器，就不是严格 open-vocabulary | 明确为 vocabulary-bank-based open-vocabulary：类别文本原型可扩展，训练只学习视觉到文本空间的对齐或轻量 head |
| 内部 text prototype 可能被误解为用户 prompt | 与 prompt-free 定位冲突 | 明确 prompt-free 指推理时用户不提供 prompt；系统内部可使用固定文本模板和词表 |
| First-neighbor clustering 会强制每个候选连边 | 背景、碎片、低置信候选会被并入语义组 | 增加候选过滤、unknown/noise 处理和最小 affinity 保护 |
| 训练时低 purity 候选直接 `argmax GT` 会污染类别标签 | 背景或混合候选被硬分配错误类别 | 增加 valid/ignore 规则：低 purity 候选不参与类别正样本训练，或映射到 unknown/background |
| MDL refinement、deduplication、representative selection 职责重叠 | 文档重复且实现边界不清 | 将 group refinement 只负责语义分组纠错，将 instance counting 负责去重和代表选择 |
| `SameInstanceRedundancy` 放进 group cost 容易误导 split | 同实例重复不应拆出 semantic group | 改为 dedup cost / diagnostic，不作为拆分主要依据 |
| part-whole 方向定义需要统一 | representative score 中方向符号容易写反 | 统一 `A_part[i,j] = P(i is part of j)`，whole 候选应低出度、高入度 |
| 词表混入 part、texture、background | 输出可能把部件或纹理当作目标对象计数 | 区分 countable class、auxiliary class 和 unknown/pattern class |

修订后的主线：

```text
Input image
  -> SAM2 candidates
  -> candidate canonicalization
  -> DINOv2 region features
  -> category probabilities over vocabulary bank
  -> pairwise semantic / instance / part-whole relations
  -> category-aware matrix clustering
  -> consistency refinement for semantic groups
  -> same-instance components
  -> representative selection
  -> class-aware count output
```

---

## 2. 方法定位

目标：给定一张普通 RGB 图像，自动发现图像中所有可数语义组，并输出每组的类别名、数量、实例位置/mask、置信度和分组质量。

推理阶段输入只有图像，不需要：

```text
text prompt
exemplar box / point
用户指定目标类别
image-level count label
```

训练阶段允许使用：

```text
object category labels
instance segmentation masks
detection boxes
region-level semantic labels
```

不使用：

```text
image-level count labels
density maps
count regression labels
```

因此本方法不是 training-free，而是：

```text
Prompt-free
Target-free
Count-supervision-free
Vocabulary-bank-based open-vocabulary semantic counting
```

这里的 open-vocabulary 指系统内部维护可扩展词表，并通过文本原型或视觉-语言对齐支持新增类别；prompt-free 指推理时用户不输入 prompt。

---

## 3. 任务定义

### 3.1 输入

```text
I: RGB image
```

### 3.2 输出

```json
{
  "groups": [
    {
      "class_name": "apple",
      "count": 7,
      "confidence": 0.92,
      "group_quality": 0.88,
      "instances": [
        {
          "box": [x1, y1, x2, y2],
          "mask_id": 3,
          "score": 0.95
        }
      ]
    }
  ]
}
```

### 3.3 与 PF-CUD 的区别

| 模块 | 原 PF-CUD | OV-CUD |
|---|---|---|
| 训练方式 | training-free | category/relation-supervised, count-supervision-free |
| 类别输出 | 不输出具体类别名 | 输出词表类别或 unknown/pattern |
| 候选生成 | SAM / blob / edge | SAM2 proposals |
| 表征 | DINOv2 + shape/color/spatial | DINOv2 region embedding + geometry |
| 分组依据 | rank distance + MST/Otsu | category probability + relation matrix |
| 计数来源 | group size / hypothesis count | refined representatives 数量 |
| 推理输入 | image only | image only |

---

## 4. 总体 Pipeline

```text
1. SAM2 Candidate Proposal
   生成过完备 mask / box 候选

2. Candidate Canonicalization
   为每个候选构造 masked crop、box crop、context crop 和几何特征

3. DINOv2 Region Encoding
   提取候选区域视觉特征

4. Category Prediction Head
   输出候选在 vocabulary bank 上的类别分布

5. Pairwise Relation Head
   输出 same-category、same-instance、part-whole 关系矩阵

6. Category-Aware Matrix Clustering
   基于类别兼容性和 same-category 关系得到粗语义组

7. Consistency Refinement
   合并同类过拆分组，拆分异类误合并组，隔离 unknown/noise 候选

8. Instance Deduplication and Representative Selection
   组内构建 same-instance components，每个真实实例保留一个代表候选

9. Semantic Count Output
   输出 class name、count、instances、confidence、group quality
```

核心计数公式：

```text
count(G) = number of valid representatives in refined semantic group G
```

不是：

```text
raw SAM2 candidate 数量
image-level regression 输出
density map 积分
```

---

## 5. Vocabulary Bank

系统维护内部词表，而不是依赖用户推理时输入类别。

词表分三类：

```text
countable classes:
    apple, orange, person, car, cell, screw, brick, tile, ...

auxiliary classes:
    leaf_part, wheel_part, background_texture, stripe, grid, ...

fallback classes:
    unknown_object, unknown_repeated_pattern, background, noise
```

原则：

```text
countable classes 用于最终计数输出
auxiliary classes 辅助识别局部、纹理和背景，一般不直接作为 count group 输出
unknown_repeated_pattern 可在重复结构明显但语义不确定时输出
background/noise 只用于过滤或诊断
```

文本原型可由 CLIP / SigLIP text encoder 生成：

```text
t_c = TextEncoder("a photo of a {class_name}")
```

这类模板是系统内部固定模板，不是用户 prompt。

---

## 6. Candidate Proposal and Canonicalization

### 6.1 SAM2 Candidate Proposal

SAM2 生成过完备候选集合：

```text
C = {c_1, c_2, ..., c_N}
```

每个候选包含：

```python
Candidate:
    mask: H x W binary mask
    bbox: [x1, y1, x2, y2]
    area: float
    source_score: float
    meta: dict
```

注意：SAM2 candidate 不等于真实 instance。它可能是完整物体、重复 mask、局部部件、碎片、合并物体、背景或纹理区域。因此后续必须依赖 relation head、refinement 和 representative selection。

### 6.2 Candidate Canonicalization

为每个候选构造三种输入：

| 输入 | 定义 | 用途 |
|---|---|---|
| masked crop | 只保留 mask 内部，背景置零 | 强化候选本体外观 |
| box crop | 裁剪 bbox 内图像，不清背景 | 保留局部上下文 |
| context crop | bbox 外扩后裁剪 | 判断完整性、局部-整体关系和邻近实例 |

保存几何特征：

```text
normalized center: cx, cy
normalized width / height
area ratio
aspect ratio
mask area / bbox area
bbox area / image area
compactness
containment candidates
```

### 6.3 Candidate Filtering

推理时先做轻量过滤，避免明显无效候选进入全量 pairwise 计算：

```text
remove extremely small / large masks
remove low SAM score masks
remove near-duplicate masks by high IoU and lower source_score
keep uncertain but plausible repeated units for unknown/pattern handling
```

过滤只处理明显噪声，不替代 relation head。

---

## 7. Region Encoding

对候选 `c_i` 提取三路区域特征：

```text
z_i_mask = DINOv2(masked_crop_i)
z_i_box  = DINOv2(box_crop_i)
z_i_ctx  = DINOv2(context_crop_i)
```

融合为：

```text
z_i = RegionFuse(z_i_mask, z_i_box, z_i_ctx, geometry_i)
```

第一版可使用 MLP：

```text
z_i = MLP([z_i_mask, z_i_box, z_i_ctx, geometry_i])
```

DINOv2 只负责视觉表征，不直接输出类别名或数量。第一版冻结 SAM2、DINOv2 和 text encoder，只训练 category head 与 relation head。

---

## 8. Category Prediction Head

### 8.1 输出

对候选区域 `c_i` 输出：

```text
p_i(c) = P(class = c | c_i)
```

包括：

```python
category_logits_i: [num_classes]
category_probs_i: [num_classes]
top_class_i: str
top_class_score_i: float
is_countable_i: bool
```

### 8.2 Open-Vocabulary Prediction

推荐用视觉区域 embedding 与文本原型匹配：

```text
candidate crop
  -> DINOv2 region encoder
  -> projection head
  -> projected region embedding h_i
  -> cosine(h_i, t_c) / temperature
  -> category logits
```

类别 logit：

```text
logit_i,c = cosine(h_i, t_c) / temperature
```

新增类别时，只需向 vocabulary bank 添加 class name / text prototype；是否需要微调 projection head 取决于新类别与训练分布的差距。

---

## 9. Pairwise Relation Head

### 9.1 关系定义

对候选对 `(c_i, c_j)` 输出：

```text
A_sem[i,j]  = P(c_i and c_j belong to the same semantic class)
A_inst[i,j] = P(c_i and c_j correspond to the same real instance)
A_part[i,j] = P(c_i is part of c_j)
```

其中：

```text
A_sem, A_inst: symmetric
A_part: directional
```

第一版不单独训练 `same_group`，因为 semantic counting group 定义为类别级计数组。同类别不同实例应属于同一个 semantic group，但不能被当成同一个 instance。

### 9.2 标签语义

| 情况 | same category | same instance | part-whole |
|---|---:|---:|---:|
| apple vs orange | 0 | 0 | 0 |
| apple instance 1 vs apple instance 2 | 1 | 0 | 0 |
| 同一 apple 的两个完整重复 mask | 1 | 1 | 低 |
| apple 局部 vs 完整 apple | 1 | 1 | `A_part[part, whole]` 高 |

### 9.3 Pairwise Feature

```text
phi_ij = [
    z_i,
    z_j,
    |z_i - z_j|,
    z_i * z_j,
    cosine(z_i, z_j),
    p_i dot p_j,
    IoU(mask_i, mask_j),
    IoU(box_i, box_j),
    center_distance_ij,
    scale_ratio_ij,
    area_ratio_ij,
    containment_i_in_j,
    containment_j_in_i
]
```

对称关系可用双向输出平均：

```text
A[i,j] = A[j,i] = (score_ij + score_ji) / 2
```

方向关系保留方向：

```text
A_part[i,j] != A_part[j,i]
```

---

## 10. Training Strategy

### 10.1 总体原则

分阶段训练：

```text
Stage 1: train Category Prediction Head / projection head
Stage 2: train Pairwise Relation Head
```

冻结：

```text
SAM2
DINOv2 Region Encoder
Text Encoder
Clustering / Refinement / Counting logic
```

不训练：

```text
count regressor
density predictor
group-level count head
representative head
validity head
```

### 10.2 Candidate-GT Matching

对 SAM2 candidate `M_i` 和 GT instance `G_k` 计算：

```text
IoU_i,k       = |M_i intersect G_k| / |M_i union G_k|
purity_i,k   = |M_i intersect G_k| / |M_i|
coverage_i,k = |M_i intersect G_k| / |G_k|
```

匹配：

```text
k* = argmax_k IoU_i,k
purity_i = purity_i,k*
coverage_i = coverage_i,k*
matched_class_i = class(G_k*)
matched_instance_id_i = id(G_k*)
```

有效性规则：

```text
if max_k purity_i,k < tau_purity:
    candidate is background/noise or ignored
elif max_k coverage_i,k < tau_part and purity_i is high:
    candidate is a possible part candidate
else:
    candidate is a valid semantic candidate
```

低 purity 候选不应硬分配为 `matched_class_i` 的强正样本，否则会污染类别训练和 relation labels。

### 10.3 Stage 1: Category Head

只训练 projection/category head：

```text
L_category = sum_i w_i * CE(p_i, matched_class_i)
```

其中：

```text
w_i = purity_i * valid_i
```

可选：

```text
low-purity candidates -> ignore
background candidates -> background/noise class
part candidates -> matched object class with lower weight, or auxiliary part class if annotated
```

### 10.4 Stage 2: Relation Head

对候选对 `(M_i, M_j)` 训练：

```text
y_sem_ij  = 1[valid_i and valid_j and class_i == class_j]
y_inst_ij = 1[valid_i and valid_j and instance_id_i == instance_id_j]
```

损失：

```text
L_sem  = sum_ij w_i w_j BCE(A_sem[i,j], y_sem_ij)
L_inst = sum_ij w_i w_j BCE(A_inst[i,j], y_inst_ij)
```

Part-whole soft target：

```text
containment_i_in_j = |M_i intersect M_j| / |M_i|
same_inst_ij = 1[instance_id_i == instance_id_j]
completeness_gap_i_to_j = max(0, coverage_j - coverage_i)

y_part_i_to_j =
    same_inst_ij
  * containment_i_in_j
  * completeness_gap_i_to_j
```

```text
L_part = sum_ij w_i w_j BCE(A_part[i,j], y_part_i_to_j)
L_relation = L_sem + L_inst + L_part
```

### 10.5 Pair Sampling

训练 relation head 时采样：

```text
same-instance pairs
same-class different-instance pairs
different-class pairs
part-whole pairs
hard negatives
background/invalid pairs
```

Hard negatives 包括：

```text
外观相似但类别不同
同类别但不同实例
高度重叠但不是同一完整实例
局部候选和完整候选
```

---

## 11. Matrix Clustering

### 11.1 Affinity

推理时构造语义分组 affinity：

```text
category_compatibility_ij = p_i dot p_j
semantic_relation_ij = sigmoid(A_sem[i,j])
A_group_ij = category_compatibility_ij * semantic_relation_ij
```

`A_group` 用于语义分组；`A_inst` 和 `A_part` 不用于语义分组，只用于组内去重、局部过滤和代表选择。

### 11.2 Category-Aware First-Neighbor

为避免不同类别被强行连在一起，推荐先按 top-k 或高置信类别分桶，再在桶内聚类：

```text
1. 过滤低质量或明显 background/noise 候选
2. 按 top-1 class 或 top-k compatible classes 分桶
3. 桶内使用 A_group 做 first-neighbor clustering
4. 若 max_j A_group[i,j] < tau_affinity，则保留 singleton 或 unknown/noise
5. 对 connected components 得到 coarse semantic groups
```

First-neighbor 伪代码：

```python
def first_neighbor_clustering(A_group, tau_affinity):
    graph = zeros_like(A_group)

    for i in range(A_group.shape[0]):
        scores = A_group[i].copy()
        scores[i] = -float("inf")
        j = scores.argmax()

        if scores[j] >= tau_affinity:
            graph[i, j] = 1
            graph[j, i] = 1

    return connected_components(graph)
```

这样保留了不预设 cluster number 的优点，同时避免每个候选被强制合并。

---

## 12. Consistency Refinement

Refinement 不训练，只修正 coarse semantic groups。

一个好的 semantic group 应满足：

```text
组内类别分布一致
组内 semantic affinity 高
跨组同类候选不应过度拆分
background/noise 不应并入 countable group
```

Group cost：

```text
Cost(G) =
    CategoryEntropy(G)
  + RelationInconsistency(G)
  + VisualDispersion(G)
  + NoisePenalty(G)
  + ModelComplexity(G)
```

其中：

```text
p_G(c) = mean_{i in G} p_i(c)
CategoryEntropy(G) = -sum_c p_G(c) log p_G(c)
RelationInconsistency(G) = mean_{i,j in G} -log(A_group_ij + eps)
VisualDispersion(G) = mean_{i in G} distance(z_i, mean_z_G)
NoisePenalty(G) = fraction of low-confidence / auxiliary candidates
ModelComplexity(G) = small penalty for excessive split
```

操作：

```text
Merge:
    if Cost(G_a union G_b) < Cost(G_a) + Cost(G_b)

Split:
    if internal category entropy or relation inconsistency is high,
    and split lowers total cost

Relabel / isolate:
    move low-confidence singleton to unknown/noise
```

`A_inst` 和 `A_part` 不作为主要 split/merge 依据。它们服务于下一步 instance counting，避免把同一语义类别内的重复候选误拆成多个 semantic group。

---

## 13. Instance Counting

### 13.1 Semantic Group vs Instance

Semantic group 是类别级计数组：

```text
Group apple = all apple instances in the image
```

Instance 是组内真实物体实例。计数应在 refined group 内完成：

```text
count(G) = number of same-instance components after representative selection
```

不是：

```text
count(G) = number of raw SAM2 candidates in G
```

### 13.2 Same-Instance Deduplication

在每个 semantic group 内，根据 `A_inst` 构建 same-instance components：

```text
component = a set of candidates likely corresponding to the same real instance
```

每个 component 最多贡献一个 count。

### 13.3 Representative Selection

统一方向定义：

```text
A_part[i,j] = P(candidate_i is part of candidate_j)
```

因此 whole candidate 的特征通常是：

```text
low outgoing part score: mean_j A_part[i,j]
high incoming part score: mean_j A_part[j,i]
larger coverage / area
higher category confidence
```

非训练式代表得分：

```text
RepScore(c_i) =
    category_confidence_i
  + completeness_i
  + mean_j A_part[j,i]
  - mean_j A_part[i,j]
  - duplicate_penalty_i
```

第一版不训练 representative head。代表候选只由类别置信度、几何完整性、part-whole 方向和 same-instance component 内相对质量决定。

### 13.4 Count Flow

```python
def count_group(group, candidates, A_inst, A_part, category_probs):
    components = build_same_instance_components(group, A_inst)

    representatives = []
    for comp in components:
        rep = choose_representative(comp, A_part, category_probs)
        if is_valid_countable(rep):
            representatives.append(rep)

    representatives = remove_residual_part_candidates(representatives, A_part)
    return len(representatives), representatives
```

---

## 14. Group Label Aggregation

每个 candidate 有类别分布 `p_i(c)`。组级类别应优先从 representatives 聚合，而不是从所有 raw candidates 聚合：

```text
p_G(c) = mean_{i in representatives(G)} p_i(c)
class_name(G) = argmax_c p_G(c)
confidence(G) = max_c p_G(c)
```

原因是 raw candidates 可能包含重复、局部、背景和噪声。

Unknown / pattern 处理：

```text
if confidence(G) < tau_label and repeated visual units exist:
    class_name = unknown_repeated_pattern
elif confidence(G) < tau_label:
    class_name = unknown_object
elif top class is auxiliary/background:
    suppress or mark as non-countable
```

示例：

```json
{
  "class_name": "unknown_repeated_pattern",
  "count": 42,
  "confidence": 0.58,
  "note": "visually consistent repeated units, low semantic confidence"
}
```

---

## 15. Inference Pseudocode

```python
def run_ov_cud(image):
    candidates = sam2_generate_candidates(image)
    candidates = filter_obvious_noise(candidates)

    crops = build_candidate_crops(image, candidates)
    region_features = dinov2_encode(crops)

    category_probs = category_head(region_features)

    pairwise_features = build_pairwise_features(
        candidates,
        region_features,
        category_probs,
    )
    relation_outputs = relation_head(pairwise_features)

    A_sem = relation_outputs["same_category"]
    A_inst = relation_outputs["same_instance"]
    A_part = relation_outputs["part_whole"]

    A_group = build_group_affinity(category_probs, A_sem)

    coarse_groups = category_aware_matrix_clustering(
        candidates,
        category_probs,
        A_group,
    )

    refined_groups = consistency_refinement(
        coarse_groups,
        candidates,
        region_features,
        category_probs,
        A_group,
    )

    results = []
    for group in refined_groups:
        count, reps = count_group(
            group,
            candidates,
            A_inst,
            A_part,
            category_probs,
        )
        label = aggregate_group_label(reps, category_probs)
        if is_output_group(label, count):
            results.append(build_group_output(label, count, reps))

    return {"groups": results}
```

---

## 16. Core Data Structures

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class Candidate:
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    area: float
    source_score: Optional[float] = None
    features: Dict[str, np.ndarray] = field(default_factory=dict)
    predictions: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticInstance:
    candidate_index: int
    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    class_name: str
    confidence: float
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticCountGroup:
    class_name: str
    count: int
    instance_indices: List[int]
    candidate_indices: List[int]
    confidence: float
    group_quality: float
    class_distribution: Dict[str, float]
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticCountResult:
    groups: List[SemanticCountGroup]
    instances: List[SemanticInstance]
    candidates: List[Candidate]
    image_shape: tuple[int, int]
    meta: Dict[str, Any] = field(default_factory=dict)
```

---

## 17. Recommended Code Structure

```text
ov_cud/
  proposals/
    sam2_proposal.py

  candidates/
    canonicalize.py
    crops.py
    geometry.py
    filtering.py

  encoders/
    dinov2_encoder.py
    text_encoder.py

  heads/
    category_head.py
    relation_head.py

  matrix/
    affinity.py
    pairwise_features.py

  clustering/
    first_neighbor.py
    connected_components.py

  refinement/
    objective.py
    merge_split.py
    noise.py

  counting/
    deduplicate.py
    representative.py
    semantic_count.py

  training/
    dataset.py
    matching.py
    losses.py
    train_category.py
    train_relation.py

  eval/
    count_metrics.py
    class_aware_metrics.py
    instance_metrics.py

  visualize/
    draw_groups.py
    draw_instances.py

  data.py
  config.py
  pipeline.py
  run_image.py
  run_dataset.py
```

---

## 18. Evaluation

因为输出包含类别名，应使用 class-aware evaluation。

### 18.1 Class-Aware Count Error

对每个类别计算：

```text
pred_count(class)
gt_count(class)
MAE / RMSE / NAE / SRE
```

### 18.2 Semantic Group Accuracy

```text
group class accuracy
top-1 / top-k class accuracy
unknown handling accuracy
countable vs non-countable classification accuracy
```

### 18.3 Instance-Level Diagnostics

```text
box AP
mask AP
instance recall
duplicate rate
false positive rate
false negative rate
part-as-instance error rate
```

最终主指标仍是 semantic count accuracy。

---

## 19. Minimal Viable Version

第一版建议实现：

```text
SAM2 proposals
DINOv2 region encoder
vocabulary-bank category head
pairwise relation head
category-aware first-neighbor clustering
same-instance deduplication
part-whole representative selection
group label aggregation
JSON output
visualization
```

第一版暂不实现：

```text
end-to-end joint training
count regression head
density map
representative head
validity head
complex MDL search
```

---

## 20. Implementation Phases

### Phase 1: Candidate Semantics

目标：

```text
候选区域能输出稳定类别分布
```

实现：

```text
SAM2 candidate generation
candidate crop construction
DINOv2 feature extraction
text prototype category classifier
```

训练：

```text
freeze SAM2 / DINOv2 / text encoder
train category head
loss = L_category
```

### Phase 2: Pairwise Relations

目标：

```text
判断候选两两之间是否同类、同实例、part-whole
```

实现：

```text
pairwise feature construction
relation labels from instance segmentation GT
relation matrix output
```

训练：

```text
freeze SAM2 / DINOv2 / text encoder / category head
train relation head
loss = L_sem + L_inst + L_part
```

### Phase 3: Clustering

目标：

```text
从 A_group 得到 coarse semantic groups
```

实现：

```text
affinity matrix construction
category-aware first-neighbor clustering
connected components
singleton / unknown handling
```

### Phase 4: Refinement and Counting

目标：

```text
修正错分、噪声和局部候选，输出最终 count
```

实现：

```text
category entropy split
same-class merge
same-instance deduplication
part-whole filtering
representative selection
count representatives
```

### Phase 5: Evaluation

目标：

```text
输出 class-aware count metrics 和错误分析
```

实现：

```text
class-aware MAE / RMSE
group label accuracy
duplicate rate
false positive / false negative analysis
part-as-instance error analysis
```
