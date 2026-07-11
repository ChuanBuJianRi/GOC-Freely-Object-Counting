# Image-Only、Point-Supervised、Density-Map-Free 多类别计数实验计划（通俗执行版）

**计划 ID**：`IOPMC-FSC147-v2.1-20260711`

**版本**：v2.1

**日期**：2026-07-11

**状态**：只完成了分析和计划改写；尚未生成 v2.1 cache、尚未训练 v2.1 模型、尚未运行 v2.1 validation/test

**当前基线 B0**：FSC-147 official test 1,190 张，MAE=`26.4992`，RMSE=`129.6856`

**最终目标**：模型只接收一张 RGB 图像，不需要用户提供示例框、示例图片或逐图类别名称；训练可以使用离散点标注，但不训练和不输出计数密度图；一次推理发现多个视觉类别，并输出每一类的实例位置和数量。

## 0. 这次改写了什么

| 版本 | 状态 | 主要内容 |
|---|---|---|
| v1 | 已被取代 | 主要围绕 FSC 单类别总数优化，多类别放在最后 |
| v2.0 | 已被 v2.1 取代 | 把候选生成/筛选设为第一主线，把多类别输出前置 |
| **v2.1** | **当前版本** | 不改变 v2.0 的实验、数值和数据红线；重新组织为“通俗正文 + 技术附录”，让每个阶段先说为什么做、做什么、何时停止 |

历史版本哈希：

- v1：`26e953f84b3faa43e4190603df00f2464c0d37af2353252501032c2e2886d674`
- v2.0：`208f4861dfda5b09e163146afc13943b2ceede8df53176e19ea3e3441ab47976`

如果只想了解研究路线，阅读第 1～13 节即可。准备实际运行实验时，再查看第 14 节及技术附录。

## 1. 一页读懂这份计划

### 1.1 把当前模型想成五道工序

```text
图像
  -> 1. 找：先找出所有可能是物体的候选区域
  -> 2. 选：删除明显不该计数的候选
  -> 3. 合：同一个物体可能被找到多次，只保留一次
  -> 4. 分：把不同物体按视觉类别放进不同组
  -> 5. 数：输出每组数量、实例位置和总数
```

可以把它类比成“入场—安检—去重—分组”：

- 候选生成负责让可能的物体入场；
- 候选筛选像安检，太严会误删，太松会放进很多假候选；
- 同一物体可能拿到多张入场券，去重时只能算一次；
- 最后把留下的实例放进若干暂时没有名称的篮子，再分别计数。

### 1.2 当前最主要的两个问题

结论不是“筛选框决定一切”，而是两个不同问题：

1. **普通和中等密度图：筛选门槛最值得先修。** 当前筛选会同时依赖 Candidate Filter 和 train-89 类别置信度；前者控制假候选，后者对没见过的新类别不可靠。
2. **极高密度图：一开始找到的候选就不够。** 如果只产生 354 个候选，后面无论多聪明，都不可能在“一候选最多算一个”的框架里数出 3,701 个物体。

两个典型例子：

| 图像 | 发生了什么 | 说明 |
|---|---|---|
| `1123.jpg` | GT 3,701；raw candidates 354；final 140 | 主要是“根本没找到足够多物体” |
| `7611.jpg` | GT 2,560；raw 1,997 → joint gate 246 → final 219 | 主要是“找到了不少，但筛选时误删太多” |

### 1.3 为什么先研究筛选，而不是 relation

冻结 validation 上，同一 tiled 候选池、同一模型权重、三 relation seeds mean：

| 筛选方式 | Val MAE | 通俗解释 |
|---|---:|---|
| 两个筛选门槛都不设 | 90.5306 | 假候选和重复候选大量进入计数 |
| 只开 Candidate Filter 门槛 | 40.2742 | Candidate Filter 单独已经解决很多错误 |
| 只开类别置信度门槛 | 50.0218 | 类别门槛也有用，但不适合做开放类别 objectness |
| 两个门槛一起开 | **30.3097** | 当前 B0 joint gate |
| joint gate，只改 relation threshold | 30.8129 | relation threshold 的影响明显较小 |

必须注意：这张表只说明“门槛很敏感”，不是完整模块因果消融。即使把某个阈值设成 0，它的分数仍可能参与后面的候选排序、分组或代表选择。Phase 1 会完成真正的模块解耦实验。

### 1.4 整体路线

| 阶段 | 要回答的通俗问题 | 成功后得到什么 |
|---|---|---|
| Phase 0 | 候选到底在哪一步丢失？ | 一套能追踪每一步候选的“仪表盘”和安全数据划分 |
| Phase 1 | 能否训练出更可靠、对新类别更友好的筛选器？ | tiled-trained countability/singleton-quality 模块 |
| Phase 2 | 高密度区域怎样产生足够多候选？ | 局部高分辨率候选或离散 point/set expert |
| Phase 3 | 怎样分别解决“同一物体去重”和“同类物体分组”？ | same-instance / same-category 双关系结构，或保留简单 NMS |
| Phase 4 | 模型能否真正输出多个类别的数量？ | anonymous groups、每组实例位置和 count |
| Phase 5 | 一张图不同区域能否采用不同计数方式？ | 局部而非整图的 density-aware router |
| Phase 6 | 每个复杂组件真的有用吗？ | 严格 leave-one-out 和跨数据集结果 |
| Phase 7 | 最终结果能否经得起冻结测试？ | 一次性 validation/test 与完整审计产物 |

## 2. 常用术语翻译

| 技术词 | 本计划里的通俗含义 |
|---|---|
| candidate / proposal | 候选物体：模型初步认为可能是物体的区域或离散点 |
| proposal capacity | 候选池容量：当前候选最多可能一一对应多少真实物体 |
| gate | 筛选门槛 |
| Candidate Filter | 候选有效性筛选器 |
| countability | 可计数性：这个候选能否代表一个可数实例 |
| singleton | 只覆盖一个标注点的候选，通常最适合一候选算一个 |
| multi-dot | 同时覆盖多个标注点的候选，不能简单当一个实例 |
| unmatched | 没覆盖当前已标注点；它可能是未标注真实物体，不能直接叫背景 |
| PU learning | 正样本—未标注学习：不把所有 unmatched 强行当负样本 |
| dedup / NMS | 去重：把同一物体的多个重复候选合并 |
| same-instance | 两个候选是不是同一个物体，用于去重 |
| same-category | 两个不同物体是不是同一视觉类别，用于分组 |
| anonymous group | 不要求事先知道类别名称的视觉分组，例如“组 A、组 B” |
| local router | 按图像局部区域选择计数方式的模块 |
| set/query point tokens | 直接预测一组离散物体位置，不生成像素密度图 |
| oracle | 只用于估计上限、实际推理绝不能使用 GT 的理想对照 |
| fail-closed | 一旦发现不允许的 GT 字段就立即停止，而不是悄悄忽略 |

## 3. 最终模型要输出什么

主结果先不强迫模型给类别命名，而是输出匿名组：

```text
组 A：8 个，给出 8 个实例位置
组 B：13 个，给出 13 个实例位置
组 C：4 个，给出 4 个实例位置
总数：25
```

为什么先做匿名组：

- “能不能把不同类别分开并数对”与“能不能正确说出类别名字”是两个问题；
- 固定类别词表不应该反过来决定一个候选是不是物体；
- 匿名组更符合真正的 image-only 主协议；
- 类别命名可以作为冻结后的独立可选分支。

主输出至少包含：每组 ID、视觉 embedding、数量、实例中心/框、分数和总数。精确 JSON schema 见附录 C。

### 3.1 本计划允许的说法

- image-only inference；
- 无用户 exemplar、无逐图 text/class prompt；
- point/dot-supervised；
- 不训练和不输出计数 density map；
- 一次推理输出多个匿名视觉组和各自数量。

### 3.2 本计划不允许夸大的说法

- 不能说完全没有内部 prompt：SAM 自动网格点属于模型内部机制；
- 使用 benchmark 全类别表时不能说 vocabulary-free；
- 使用 dots 就不能说 count-supervision-free；
- 使用 SAM2/DINOv2/CLIP 预训练权重就不能说 prior-free；
- SAM segmentation mask 不是计数 density map，但必须披露模型使用了 mask；
- FSC 的单类别 scalar count 不能单独证明多类别能力。

## 4. 怎样判断模型真的变好了

我们不只看最终 MAE，而要回答四个简单问题。

### 4.1 第一问：一开始有没有找到真实物体

- 有多少真实点至少被一个候选覆盖；
- 有多少真实点能找到 singleton candidate；
- 当前候选池最多能和多少真实物体一一对应。

对应技术指标：`unique_dot_recall`、`singleton_dot_recall`、`matching_capacity`。

### 4.2 第二问：筛选时误删了多少、放进多少假候选

- raw → filter → category → joint 每一步剩多少；
- 每一步新丢失多少仍可恢复的真实点；
- unmatched 候选有多少；
- 低密度图是否因筛选太松而过计数。

### 4.3 第三问：重复候选有没有合并正确

- 同一物体平均产生多少候选；
- 去重有没有把不同物体错误合并；
- learned relation 是否真的超过简单 IoU-NMS。

### 4.4 第四问：多类别有没有分对、数对

- 总数是否正确；
- 每个组的数量是否正确；
- 多余组、漏掉的组都要受惩罚；
- 分组结构是否与真实类别一致；
- 如果启用命名，再单独看命名准确率。

### 4.5 一个新模块怎样才算过关

主要升级必须同时满足：

1. class-dev MAE 至少改善 1.0；
2. paired 95% CI 的改善方向不跨 0；
3. seeds 17、42、73 三次训练方向一致；
4. 改善不能只来自少数与训练类很像的类别；
5. 不能牺牲 100+ 图去换低密度改善，反过来也不行；
6. 多类别 group F1 和额外组错误不能退化；
7. 复杂模块没有稳定超过简单 baseline，就保留简单 baseline。

技术指标的完整清单和计算口径见附录 B。

## 5. 必须遵守的数据红线

正文只需要记住三条：

1. **训练时只能看训练标注。** validation/test GT、类名和单图失败信息不能参与训练、阈值或路由选择。
2. **必须先完成全部预测，再读取 GT 算分。** GT 只能进入独立 metric 进程。
3. **一旦发现禁止字段就停止。** inference loader 遇到 `valid/points/gt_count/matched_*/class_name` 必须 fail-closed。

### 5.1 为什么要按类别隔离开发集

原来的图像随机划分相当于“训练见过猫，开发集还是猫，只是换了照片”；而 FSC test 会出现从未见过的类别。新版按类别整体隔离：

| Partition | 划分 | 用途 |
|---|---|---|
| `fit-train` | 71 个 fit classes 中 90% 图像 | 梯度训练 |
| `fit-stop` | 同 71 类剩余 10% 图像 | early stopping |
| `class-dev` | 完整 9 类 | 选择 threshold、结构和 route |
| `class-audit` | 另外完整 9 类 | 完整模型冻结后只打开一次 |

split seed 固定为 `20260711`。类别分配需平衡图像数、count bins、小目标比例和候选数，并记录 manifest SHA-256。

### 5.2 Validation 和 test

- official validation 已经在 B0 和历史诊断中使用，不能称 untouched；v2.1 完整配置冻结后只允许再做一次冻结确认，不能用结果回调同一版本；
- 若 validation 后继续改模型，必须升级计划版本，并承认 validation 已成为开发证据；
- official test 只在全部配置冻结后执行一次；
- FSC test 历史上已经被查看，因此只能保证 no-GT computation path，不能称 untouched blind test。

### 5.3 多类别训练数据

主 checkpoint 使用 official-train-only synthetic compositions：把不同 train classes 的带点图像组合成多组场景，同步变换 dots、source group 和坐标，不生成 Gaussian density map。只监督已标注 dots；其他物体按 unknown/ignore 处理。

如果以后使用 MCAC/OmniCount train split，必须另建 `domain-supervised` checkpoint 和结果表，不能继续称 official-train-only zero-shot。

## 6. Phase 0：先把候选在哪一步丢失记录清楚

> 一句话：先装“仪表盘”和“数据锁”，还不训练新模型。

### 本阶段要解决什么

现在只能看到最终 count，无法稳定回答“候选没生成出来”还是“生成后被筛掉”。Phase 0 要让每张图都能追踪：

```text
raw 候选
  -> 有效性筛选
  -> 类别门槛
  -> joint gate
  -> 去重
  -> 多类别分组
  -> 最终输出
```

### 具体做什么

1. 生成 class-disjoint manifest；
2. 导出 physically isolated train dots + image-level class shard；
3. 生成与 inference 相同配方的 train tiled labeled cache；
4. 保存每个 train candidate 覆盖的完整 dot IDs、dot_count、singleton/multi-dot、tile/border 和 SAM quality；
5. 建立 prediction-first、GT-later stage evaluator；
6. 将当前只返回一个数字的 `dedup_count` 扩展为返回 groups、representatives、locations 和 total；
7. 不改模型，重新落盘 B0 每一步候选统计；
8. 冻结 B0 代码、配置、模型和预测作为后续 anchor。

### 完成标志

- 四个 class partitions 互斥且覆盖完整 official train；
- labeled cache 去掉标签后的 `z/bbox/quality` 与 image-only generator 完全一致；
- prediction 进程读取 GT 字段数为 0；
- forbidden-field 单元测试通过；
- 每一步 aggregate 都能由逐图 JSON 重算；
- groups 的 count 之和等于原 scalar total。

## 7. Phase 1：训练更可靠的候选筛选器

> 一句话：先把下游固定住，只研究“哪些候选应该留下”。

### 本阶段要解决什么

回答三个问题：

1. Candidate Filter 是否真的是最大的可调模块；
2. 用 full-image candidates 训练、用 tiled candidates 推理是否造成分布错位；
3. 能否避免把所有 unmatched 候选误当背景。

### 第一步：真正拆开 filter 和 category

所有对照使用同一 2x2 candidate pool、固定 IoU-NMS@0.3 和无 category 的代表选择规则。

| ID | 保留什么筛选 | 排序/代表选择 | 用途 |
|---|---|---|---|
| G0 | 不用 learned gate | SAM image-derived quality | 最简单 baseline |
| G1 | 只用 CandidateFilter | filter × SAM quality | 最终 countability 候选 |
| G2 | 只用 train-89 类别置信度 | category confidence | 闭集诊断，不作为最终结构 |
| G3 | Filter + category joint | 当前 joint score | B0-style anchor |
| G4 | GT-dot oracle | oracle quality | 只估计上限，严禁 inference |

G1 必须在 gate、候选排序、representative selection 三处同时移除 category confidence，否则不能叫真正解耦。

### 第二步：比较 full 和 tiled 训练

| ID | 训练候选 | 标签 | 其他条件 |
|---|---|---|---|
| F0 | full-image pts32 | any-dot binary | 固定 G1 |
| F1 | 2x2 tiled pts32 | any-dot binary | 与 F0 完全相同 |
| F2 | full+tiled，1:1 image sampling | any-dot binary | 总训练步数相同 |

F0/F1 必须使用同一新 manifest、同一 seeds、同一训练步数并从头训练，不能拿旧 F0 checkpoint 对比新 F1。

### 第三步：改进标签

只有 F1 确认有效后才继续：

| ID | 标签方式 | 通俗解释 |
|---|---|---|
| Q0 | any-dot BCE | 覆盖任何 dot 就算正样本 |
| Q1 | PU binary | unmatched 默认未知，只采可信负样本 |
| Q2 | unmatched / singleton / multi-dot | 区分候选是否适合“一候选算一个” |
| Q3 | Q2 + `log1p(dot_count)` / ordinal multiplicity 多任务 | 同时估计候选覆盖了几个实例 |

multi-dot 不能统一硬删除；它可以触发高密度分支或提供 multiplicity 信息。

### 主要看什么

- tiled candidate calibration；
- singleton recall 和 matching capacity；
- 低密度假阳性；
- 100+ unique-dot recall；
- 固定 NMS 后的最终 MAE；
- 三 seeds 和 class-level bootstrap。

### 何时停止

- F1 不能同时改善 tiled calibration 和 count MAE：停止“只修训练分布”；
- Q1/Q2 只提高 candidate F1、不提高 capacity/final count：不升级；
- Q2/Q3 让低密度变好但使 100+ unique-dot recall 下降超过 2 个百分点：拒绝；
- G1 不如 G3：继续改善 generic countability calibration，但禁止把 train-89 恢复为最终 objectness gate。

固定训练参数见附录 C。

## 8. Phase 2：让高密度区域产生足够多候选

> 一句话：如果“安检”已经合理，但候选池本身装不下真实物体，就增加局部候选能力。

### 本阶段要解决什么

解决 `1123/6860` 类型的硬上限，同时不回到 density-map regression。

### 实验对照

| ID | 方法 | 用途 |
|---|---|---|
| P0 | 2x2 pts32 | 标准 baseline |
| P1 | 4x4 pts32 | 高密度区域 |
| P2 | 6x6 pts32 | 极高密度区域 |
| P3 | 4x4/6x6 + upscale | 小目标诊断 |
| P4 | patch-wise set/query point tokens | 直接预测离散物体位置 |
| P5 | patch scalar count + discrete group token | point queries 数量仍不够时的辅助专家 |

P4/P5 不生成 Gaussian density map。若某实现内部产生 dense heatmap，必须作为不同协议单列。

### 为什么必须按局部区域选择

一张多类别图里可能同时有稀疏大物体和密集小物体，所以不能只给整张图一个 sparse/dense 标签。Local router 只能读取图像派生信号，例如局部候选数、尺寸、quality quantile、SAM quality、重复率、纹理/entropy/frequency、DINO features、multi-dot probability 和跨尺度一致性。

### 何时停止

- 更密 tiling 不提高 high-density matching capacity：停止继续加 tile；
- capacity 增益小于 2 个百分点但成本超过 2 倍：不进入主模型；
- 只有 P0-P3 仍覆盖不了 train dense cases，才启动 P4/P5；
- dense expert 必须改善 class-dev 100+，且不能增加低密度假阳性。

## 9. Phase 3：分开解决“同一物体”和“同一类别”

> 一句话：先把同一个物体的重复候选合成一个，再把不同实例按类别分组。

### 为什么要拆成两个关系

- `same-instance` 回答“这两个候选是不是同一个物体”，用于去重；
- `same-category` 回答“这两个不同物体是不是同一类”，用于多类别分组。

当前 train-89 类别 bucket 可能在去重前就把两个重复候选分开，因此执行顺序改为：

```text
候选/point tokens
  -> local same-instance consolidation
  -> representatives
  -> same-category grouping
  -> 每组计数
```

### 标签怎样定义

- same-instance 正样本：两个候选都只覆盖同一个 singleton `{d}`；
- 两个可信 singleton 对应不同 dots：负样本；
- 任何 multi-dot pair：ambiguous/ignore，不能再用“covered-dot sets 有交集”直接当正样本；
- same-category 正样本使用相同 official-train class 或 synthetic source group，不同 train classes 才能作为可信负样本；
- 主 relation 不依赖 OOD 敏感的 `p_i·p_j`；
- 使用 local overlap + spatial kNN 覆盖全部候选，取消全局 top-200 截断。

Relation score 的来源和 threshold policy 必须拆成两个变量分别比较，不能把 learned score、NMS 和 density-conditioned threshold 混在同一个实验里。

### 实验对照

| ID | 去重 | 类别分组 | 用途 |
|---|---|---|---|
| R0 | IoU-NMS@0.3 | visual clustering | 简单 baseline |
| R1 | 当前 scratch relation | train-89 grouping | 旧结构诊断 |
| R2 | singleton-safe relation | visual clustering | 只测 learned 去重 |
| R3 | singleton-safe relation | learned same-category | 双关系核心 |
| R4 | R3 + local full graph | R3 | 高密度扩展 |

### 何时停止

- R2/R3/R4 必须在三 seeds、paired CI 和 class-bootstrap 上稳定超过 R0 至少 0.5 MAE；
- over-merge 和 under-merge 都不能恶化；
- 没超过就使用 R0，relation 降为未采用探索；
- 先修复并测试 `adaptive_tau(base=0.99,max=0.95)` 的语义冲突。

## 10. Phase 4：输出真正的多类别计数结果

> 一句话：模型不再只给一个总数，而是给“组 A 有几个、组 B 有几个”，并指出每个实例在哪里。

### 训练数据

使用 official-train-only synthetic multi-category scenes：每图组合 2～5 个 source groups，保留离散点和 source group ID，不生成 density map；fit/dev/audit source classes 严格隔离。

### 实验对照

| ID | 分组方式 | 命名 | 用途 |
|---|---|---|---|
| M0 | one global group | 无 | 只看总数的 control |
| M1 | DINO/CLIP visual clustering | 无 | anonymous baseline |
| M2 | learned same-category graph | 无 | **主匿名多类别模型** |
| M3 | M2 | 预声明通用 frozen vocabulary | 可选命名分支 |
| M4 | benchmark vocabulary | benchmark names | dataset-vocabulary-known 诊断 |

M1/M2 默认只使用 CLIP visual embedding；文本原型不能反过来做 objectness gate。

### 主要看什么

- 每组是否分对、数对；
- 多余组和漏组是否受惩罚；
- total count 是否正确；
- group presence F1、ARI/NMI、per-group count；
- 输出文件是否真的保存 groups、locations 和 counts，而不是只保存总和。

### 何时停止

- group presence F1、ARI/NMI、per-group count 必须同时改善；
- extra predicted groups 必须被惩罚；
- FSC total 变好不能以 grouping collapse 为代价；
- M2 为 primary，M3/M4 永远单独成表。

## 11. Phase 5：让不同区域自动选择合适计数方式

> 一句话：同一张图里的稀疏区域和密集区域可以走不同分支，再统一去重、分组和计数。

| Router | 怎样训练 | 正确论文表述 |
|---|---|---|
| D1 | image-only consistency / proposal stability | 不新增 count target |
| D2 | train dots 派生 local density/capacity target | point/count-informed，density-map-free |

如果 D1 用 GT 决定哪一类 cluster 走哪条 route，它就不能再称无新增 count supervision。D1/D2 必须分表。

统一策略：

- sparse region：严格 countability + NMS/relation；
- medium region：2x2 proposal tokens；
- dense region：放宽 singleton gate，启用 4x4/6x6；
- extreme region：set/query points 或 patch count token；
- 跨 expert 先按 same-instance 去重，再按 same-category 分组。

优先使用 soft/local mixture；硬 route 必须报告边界敏感性。Phase 1～4 全部冻结后才联合选择 D1/D2；完整配置冻结后 class-audit 只打开一次。

## 12. Phase 6：逐项拆掉组件，确认谁真的有效

> 一句话：完整模型冻结后，每次只拿掉一个组件，防止把多个变化混成一个贡献。

| ID | 拿掉什么 | 用什么替代 |
|---|---|---|
| A1 | tiled-trained countability | full-image filter |
| A2 | PU/singleton-quality target | any-dot BCE |
| A3 | proposal-capacity expert | 2x2 only |
| A4 | local router | fixed global policy |
| A5 | learned same-instance | IoU-NMS@0.3 |
| A6 | learned same-category | visual clustering |
| A7 | open-set grouping | one global group |
| A8 | optional naming | anonymous groups |

所有行使用相同 safe cache、manifest、seeds 和 frozen thresholds，一次只改变一项。

随后用同一新 checkpoint 重跑 CARPK、PUCPR+、OmniCount、MCAC：

- 主协议输出 anonymous groups，不读取 benchmark test vocabulary；
- OmniCount 同时报 total、per-group/per-class 和 extra-group penalty；
- MCAC 用带位置输出的 Hungarian spatial matching，并惩罚 unmatched predictions；
- adapter 只能做 resize/schema 转换，不能用 test GT 调阈值；
- 旧 checkpoints 保持 legacy 标记。

## 13. Phase 7：冻结后做最终 validation/test

> 一句话：模型、阈值、route 和代码先全部封存，再生成所有预测，最后才读 GT 算分。

进入条件：

1. fit-stop/class-dev/class-audit 已冻结全部结构和参数；
2. official validation 只做一次 v2.1 冻结确认，不能用于回调本版本；
3. checkpoint、代码、词表、manifest、cache recipe 都有 SHA-256；
4. test plan 只由 image-only cache 生成，记录 `test_annotations_read=false`；
5. inference loader 对禁止 GT 字段 fail-closed；
6. 先为全部 1,190 张生成结构化预测，再由独立 metric 进程读取 targets；
7. test 后不调参重跑，失败只能如实报告或转到新 benchmark。

最终报告必须包含三 seed、image/class bootstrap、各密度区间、候选漏斗、capacity、匿名多类别、extra groups、极端失败和逐图 JSON。

## 14. 现在立即执行的三件事

1. **生成 class-disjoint manifest**：完成 71 fit classes + 9 class-dev + 9 class-audit；
2. **落盘 B0 候选流水**：raw/filter/category/joint/dedup 每一步都保存数量和事后 unique-dot loss；
3. **启动 F0/F1 对照**：同一 manifest、seeds 和训练步数，唯一差异是 full candidates 与 tiled candidates。

在这三件事完成前，不启动 relation 重训，不读取新的 FSC test 配置，也不创建空结果文件。

## 15. 执行顺序和预计成本

| 顺序 | 阶段 | 主要成本 | 是否训练 | Phase 0-6 是否读新 test GT |
|---:|---|---|---:|---:|
| 1 | Phase 0 manifest/cache/telemetry/output schema | 高 | 否 | 不进入模型选择；允许独立审计进程只对冻结 B0 做事后核对 |
| 2 | Phase 1 gate attribution + F0/F1 | 低 | 部分 | 否 |
| 3 | Phase 1 PU/singleton quality | 低-中 | 是 | 否 |
| 4 | Phase 2 capacity/dense tokens | 高 | 可选 | 否 |
| 5 | Phase 3 dual relation/NMS | 中 | 是 | 否 |
| 6 | Phase 4 multi-category grouping | 中 | 是 | 否 |
| 7 | Phase 5 local router/full freeze | 中 | 是 | 否 |
| 8 | Phase 6 LOO/cross-dataset | 高 | 视变体 | 否 |
| 9 | Phase 7 frozen val/final test | 中 | 否 | 仅预测冻结后算指标 |

## 16. 记录和同步规则

每个实验必须有唯一 experiment ID 和 parent baseline，不能覆盖旧输出。完成顺序固定为：

```text
预注册 config
  -> 运行与逐图 prediction artifact
  -> machine-readable summary + SHA-256
  -> 人读阶段报告
  -> 同步 AAAI 总览
```

未运行实验前，禁止创建空 checkpoint、伪日志或空结果报告。完整路径和 JSON 字段见附录 C。

文档同步关系：

| 文档 | 作用 |
|---|---|
| 本计划 | 当前唯一执行计划 |
| `docs/AAAI_2026_Experiment_Report.md` | 同步路线、状态和口径，不改冻结 26.50 |
| `OV_CUD_AAAI_Experiment_Plan.md` | 历史计划，保留旧内容供审计 |
| `docs/fsc147_strict_nogt_cp_free_report_20260710.md` | 冻结 B0 报告，不随新计划改写 |

# 技术附录

## 附录 A：冻结证据和精确诊断数字

### A.1 冻结资产

| 资产 | 用途 | SHA-256 |
|---|---|---|
| `result/logs/fsc147_strict_nogt_cp_free_full1190.json` | B0 test 预测、分桶和逐图 raw count | `a455706c602a3b386488983b94db394e7ba3dd65ca7536aac535aff2121a184c` |
| `result/logs/fsc147_strict_nogt_val_selection.json` | B0 validation 网格和 route | `8c4f0c61cdf74221470cd41fcc58837be7f39ba37b5f08b50d303b9b3ab5ee16` |
| `docs/fsc147_strict_nogt_cp_free_report_20260710.md` | B0 人读报告 | 保持冻结，不改写 |

test 分析只复用上述冻结预测做事后诊断，不产生新 test 阈值、route 或单图规则。

### A.2 Frontend、高密度和 outliers

- official-val three-seed mean：`always_fast=36.7081`，`always_tiled=30.3097`，差 `6.3984 MAE`；
- B0 test 100+ bin：195 张，MAE=`82.8410`，bias=`-61.5487`；绝对误差 `16,154 / 31,534 = 51.23%`；
- `1123`：GT 3,701，2x2 raw 354，final 140；proposal 数量硬误差至少 3,347；
- `7611`：GT 2,560，4x4 raw 1,997 → filter 748 → category 584 → joint 246 → final 219；joint 保留 12.32%；
- 两张图绝对误差 5,902，相当于 full-test 4.96 MAE；
- 事后诊断：selected frontend raw<GT 11 张，对应至少 4.17 MAE；joint 后 candidate<GT 259 张，对应至少 13.21 MAE。两项必须在 Phase 0 独立落盘后才能成为正式证据。

### A.3 Relation 证据边界

- IoU-NMS@0.3 validation diagnostic：MAE=`29.3313`；
- learned relation seed73：MAE=`30.0016`；
- learned relation three-seed mean：MAE=`30.3097`。

NMS 数字目前没有独立冻结 JSON，只能称诊断。Phase 3 正式重跑前，relation 不得作为 accuracy contribution。

## 附录 B：完整指标口径

### B.1 候选漏斗

预测冻结后才 join GT，计算：

- `unique_dot_recall`：至少被一个候选覆盖的 dot 比例；
- `singleton_dot_recall`：至少被一个 singleton candidate 覆盖的 dot 比例；
- `matching_capacity`：candidate-dot 二分图最大一对一匹配数 / GT count；
- `multi_dot_only_recall`：只能由 multi-dot candidates 覆盖的 dot 比例；
- `duplicate_multiplicity`：每个 dot 的候选覆盖数；
- `unmatched_rate`：不覆盖已标注 dot 的候选比例，只能称 unmatched；
- 每一级 candidate/token count、保留率和新丢失的 unique dots；
- proposal 数量下界、matching 下界和候选池 oracle count。

### B.2 计数指标

- MAE / RMSE / bias；
- bins：0-10、11-20、21-50、51-100、100+；
- low-density false positives；
- high-density recall/capacity；
- three seeds mean ± sample std；
- paired per-image bootstrap 与 class-cluster bootstrap 95% CI。

### B.3 多类别指标

- total-count MAE/RMSE；
- matched per-group/per-class MAE/RMSE；
- macro present-class MAE；
- group presence precision/recall/F1；
- ARI/NMI 或 B-cubed grouping F1；
- instance location precision/recall/F1；
- unmatched predicted/GT groups 和 extra-group penalty；
- anonymous Hungarian spatial matching；
- optional naming top-1/top-k accuracy。

只报 Hungarian matched groups、忽略多余 groups，不可作为主结果。

## 附录 C：固定训练、缓存和产物规范

### C.1 Phase 1 固定训练规则

- CandidateFilter anchor：`1152+7 -> 256 -> 128 -> 1`；
- seeds：17、42、73；
- early stopping 唯一主指标：`fit-stop BCE/NLL`；
- threshold 唯一选择指标：class-dev count MAE；
- threshold grid：`{0.1,0.2,0.3,0.4,0.5,0.6,0.7}`；
- downstream 固定 IoU-NMS@0.3；
- Phase 0-6 的训练、预测和模型选择进程不读取 official validation/test GT；Phase 0 可由独立 audit 进程读取 isolated test targets，只复核已经冻结的 B0，不产生新配置。

### C.2 物理隔离

1. 训练进程只能接收 isolated train images/dots/classes；
2. inference cache 只允许 `schema,img_id,file_name,z,bbox,height,width,source_cache` 和纯图像派生 quality；
3. mask RLE、covered-dot sets、group labels 只能进入独立 diagnostic cache；
4. prediction-first、metric-later；
5. loader 遇 `valid,points,gt_count,matched_*,class_name` 必须 fail-closed；
6. checkpoint 记录上游资产、manifest、代码、seed 和 `test_images_loaded=0`。

### C.3 结构化输出 schema

```text
prediction = {
  image_id,
  groups: [
    {
      group_id,
      group_embedding,
      count,
      instances: [{center, bbox, score, source_token}],
      optional_name
    }
  ],
  total_count
}
```

`total_count` 必须等于所有 groups count 之和。

### C.4 预注册配置

v2.1 只重写表达与结构，不改变 v2.0 实验定义，因此以下尚未创建的计划产物继续保留 `v2` 文件标签；文件内部必须记录 `plan_version=v2.1` 和当前计划 SHA-256。

- `result/configs/fsc147_train_class_disjoint_manifest_20260711.json`
- `result/configs/fsc147_train_points_classes_isolated_20260711.json`
- `result/configs/fsc147_candidate_funnel_preregister_v2.json`
- `result/configs/fsc147_multicategory_protocol_v2.json`
- `result/configs/fsc147_final_imageonly_plan_v2.json`

### C.5 Checkpoints、日志和报告

- `result/checkpoints/fsc147_countability_tiled_{variant}_seed{17,42,73}.pt`
- `result/checkpoints/fsc147_same_instance_v2_seed{17,42,73}.pt`
- `result/checkpoints/fsc147_same_category_v2_seed{17,42,73}.pt`
- `result/logs/fsc147_phase0_candidate_funnel_b0.json`
- `result/logs/fsc147_phase1_candidate_core_summary.json`
- `result/logs/fsc147_phase2_capacity_summary.json`
- `result/logs/fsc147_phase4_multicategory_summary.json`
- `result/logs/fsc147_final_v2_predictions.json.gz`
- `docs/fsc147_phase0_candidate_funnel_report_20260711.md`
- `docs/fsc147_phase1_countability_report_20260711.md`
- `docs/fsc147_phase2_capacity_report_20260711.md`
- `docs/fsc147_multicategory_v2_report_20260711.md`
- `docs/fsc147_imageonly_v2_final_report.md`

### C.6 每个冻结 JSON 的必填字段

```text
schema, date, status, frozen, protocol, experiment_id,
parent_experiment_id, split_manifest_sha256, code_sha256,
upstream_assets_sha256, seeds, primary_rule, selected,
prediction_gt_fields, train/val/test_images_loaded,
per_stage_counts, per_stage_unique_dot_metrics,
aggregate, per_image_artifact, artifact_sha256
```

## 附录 D：启动检查清单

- [ ] v2.1 计划、实验矩阵和停止条件已提交，且不再根据 test 修改；
- [ ] class-disjoint manifest 已生成并审核；
- [ ] isolated train dot/class shard 显示 val/test loaded=0；
- [ ] train tiled labeled cache 与 no-GT generator 通过 tensor 一致性测试；
- [ ] B0 stage telemetry 已独立落盘并复算第 1 节数字；
- [ ] G0-G4 的 gate、rank、representative 依赖已逐项审计；
- [ ] F0/F1 使用相同 manifest、seeds、训练步数；
- [ ] early stopping 和 threshold 选择各有唯一主指标；
- [ ] structured group output 与 scalar total 一致；
- [ ] official validation/test annotation 不进入 Phase 0-6 的训练、预测或选择进程；冻结 B0 的独立事后 audit 已与选择进程隔离；
- [ ] 输出目录、JSON schema、命名和 SHA-256 流程已预注册。

在上述检查完成前，本计划状态保持“仅计划，未执行”。
