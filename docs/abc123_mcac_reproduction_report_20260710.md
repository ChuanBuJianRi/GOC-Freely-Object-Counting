# ABC123 在 MCAC 上的官方协议复现报告

**日期**：2026-07-10

**数据集**：MCAC test full 2,115 images

**论文参考值**：Per-class MAE=9.52 / RMSE=17.64
**结论**：成功复现，完整 2,115 张的本地结果为 MAE=9.46 / RMSE=17.52。

## 1. 复现结论

本轮直接使用 ABC123 官方仓库的数据集类、官方测试配置和官方发布 checkpoint，重新评测 MCAC test。主要结果如下：

| 结果 | 图像数 | image-class pairs | Per-class MAE | Per-class RMSE | Bias |
|---|---:|---:|---:|---:|---:|
| ABC123 published | - | - | 9.52 | 17.64 | - |
| 本地 full，torchvision 0.13 resize 语义 | 2,115 | 3,630 | **9.4595** | **17.5186** | -1.5031 |
| 本地官方 `drop_last=True` 口径 | 2,114 | 3,629 | 9.4500 | 17.5057 | -1.5156 |
| 本地 full，当前 torchvision 默认 resize | 2,115 | 3,630 | 9.5635 | 17.7605 | -2.0643 |

相对论文值，主要复现行的差异只有 `-0.0605 MAE / -0.1214 RMSE`。当前 torchvision 默认行为的差异也只有 `+0.0435 / +0.1205`。两种合理环境口径分别落在论文值两侧，因此不能认为存在实现级复现失败。

以图像为重采样单位做 1,000 次 bootstrap，主要复现行的 95% CI 为：

- MAE：`[8.9391, 10.0326]`
- RMSE：`[16.4117, 18.6755]`

论文的 `9.52 / 17.64` 均位于区间内。论文主表仍应引用 published `9.52 / 17.64`；本地 `9.46 / 17.52` 应作为独立 reproduction 行或复现说明，不能替换论文原值。

## 2. 官方配置核对

| 项目 | 本轮配置 |
|---|---|
| 官方仓库 | `ActiveVisionLab/ABC123` |
| 仓库 commit | `1ca1f30614e0809f72dca8e225f8dd5dbe31be5d` |
| checkpoint | `/tmp/ABC123/checkpoints/model_chkpt.ckpt` |
| checkpoint SHA256 | `2c251850da061f183b239aa86f258f44085923f02bef196f247bd84d9d7b919c` |
| checkpoint metadata | epoch 86，global step 102,254 |
| 测试配置 | `configs/ABC123test.yml` |
| 输入 | image-only，无 exemplar/text/class prompt |
| crop | center crop 672x672 |
| GT 过滤 | `occlusion_crop672 < 70` |
| 模型 | DINO ViT backbone，5 个 density/count heads |
| matching | density matching；L2 normalize 后 L1 cost 的平方，Hungarian assignment |
| `gtd_scale` | 400 |
| 官方 eval batch size | 2 |
| 官方 `drop_last` | `True` |

评测脚本直接导入官方 `MCAC_Dataset`，因此中心裁剪、occlusion 过滤、预计算 density map、有效类别压缩及图像顺序均沿用官方实现。模型 forward 复用此前已审计的本地 wrapper；该 wrapper 与官方 `ABC123` forward 在相同输入上的 count/density 输出最大差值为 0。

官方环境文件固定 `Python 3.8.12 / PyTorch 1.12.1 / torchvision 0.13.1 / timm 1.0.3`。当前运行环境为 PyTorch 2.12.0；主要复现行显式设置 Tensor Resize `antialias=False`，用于还原 torchvision 0.13.1 的默认语义。另用 timm 1.0.3 复核后，输出与当前 timm 下的 legacy-resize 结果逐字节一致，timm 版本不是残余差异来源。

## 3. 完整性与 `drop_last` 问题

本地 MCAC test 目录有 2,115 张图。官方默认配置同时满足：

- `eval_batch_size=2`
- `drop_last=True`
- dataset image ID 来自未排序的 `os.listdir`

因此单卡官方 DataLoader 实际只评 2,114 张，且漏掉哪一张依赖文件系统目录顺序。在当前目录顺序下，漏掉的是：

```text
3671918445863908
```

该图有 1 个有效类别，GT=70，主要复现行预测为 114.1371，绝对误差 44.1371。补回后 MAE/RMSE 从 `9.4500/17.5057` 变为 `9.4595/17.5186`，只产生小幅变化。

本轮同时保留此前缺失的零目标样本：

```text
2277443934862561
```

该图在 `crop672 + occlusion<70` 后没有有效 GT 类别，因此不增加 3,630 个 per-class pair 的分母，但应进入完整图像级假阳性审计。

## 4. Resize 版本差异

当前 torchvision 对 Tensor Resize 默认启用 antialias，而 ABC123 原环境 torchvision 0.13.1 默认不启用。该差异对汇总指标不大，但会明显改变部分单图预测：

| 对比项 | 数量/数值 |
|---|---:|
| 预测 count 发生数值变化的图像 | 2,111 / 2,115 |
| density matching head 发生变化的图像 | 264 / 2,115 |
| 配对后单个 count 最大差值 | 185.54 |

因此本报告将 `antialias=False` 的 full-2115 结果作为最接近论文原环境的主要本地复现值，同时完整保存当前默认 resize 结果，避免隐式依赖库版本。

## 5. 难度切片

### 5.1 按每图有效类别数

| 有效类别数 | 图像数 | pairs | Per-class MAE | Per-class RMSE |
|---:|---:|---:|---:|---:|
| 0 | 1 | 0 | - | - |
| 1 | 1,013 | 1,013 | 6.82 | 14.70 |
| 2 | 760 | 1,520 | 8.05 | 16.37 |
| 3 | 267 | 801 | 13.81 | 21.47 |
| 4 | 74 | 296 | 13.95 | 20.02 |

ABC123 在 1-2 类图上最稳定；3-4 类图的 per-class MAE 上升到约 13.8-13.9。该趋势说明多 head density matching 可以处理多类别，但类别数增加仍会加大分离和回归误差。

### 5.2 按单类别 GT count

| GT count bin | pairs | Per-class MAE | Per-class RMSE | Bias |
|---|---:|---:|---:|---:|
| 1-10 | 890 | 2.39 | 4.66 | -0.50 |
| 11-50 | 1,470 | 7.04 | 11.78 | -0.43 |
| 51-100 | 732 | 13.12 | 21.24 | +3.68 |
| 101-200 | 368 | 19.25 | 28.35 | -6.98 |
| 201-300 | 170 | 30.44 | 39.46 | -26.46 |

高密度类别仍是主要误差来源。最差 pair 为图像 `1849326747169697`：GT=60，prediction=228.6510，绝对误差 168.6510。

## 6. 官方指标的协议限制

ABC123 的 published per-class 指标只评估与非零 GT density map 匹配的 heads；没有匹配到 GT 的额外预测 heads 不受惩罚。这一点对比较非常重要：

| 诊断口径 | MAE | RMSE | Bias |
|---|---:|---:|---:|
| 官方 matched per-class | 9.46 | 17.52 | -1.50 |
| 每图 matched-head total | 11.54 | 22.02 | -2.58 |
| 每图全部 5 heads total | 58.70 | 90.20 | +56.77 |

零目标图 `2277443934862561` 的 5 个 head count 总和为 401.20，但官方 per-class 指标中该图贡献 0 个 pair，因而这一假阳性完全不进入 `9.46 / 17.52`。

这不是本地修改，而是官方 matching 指标本身的定义。与 OCCAM、UniCounting 比较时，应对所有方法使用一致的 GT-class matching 口径；同时建议额外报告 total-count 或 unmatched-group penalty，避免模型通过输出多余 group 获得不完整的评价。

## 7. 最终判断与论文使用方式

1. **ABC123 的 MCAC published 结果可以认为已成功复现。** 本地完整集结果与论文只有约 0.6%-0.7% 的差异。
2. **残余差异主要是环境和评测细节，不是模型实现错误。** 最明确的来源是 torchvision 0.13 与当前版本的 resize antialias 行为；官方 `drop_last` 和未排序目录顺序也会造成小幅波动。
3. **论文表中保留 published `9.52 / 17.64`。** 可在脚注或附录加入“official checkpoint local rerun: `9.46 / 17.52`, full 2,115”。
4. **ABC123 仍不是同监督协议。** 它是 prompt-free/image-only，但训练依赖 per-class density map 和 count supervision；我们的核心差异仍是 dense-map-free 和 count-supervision-free。
5. **不要只依据 matched per-class 指标宣称完整的开放集分组能力。** 该指标忽略额外 heads，建议主文同时披露 total-count 诊断。

## 8. 复现实验产物

| 文件 | 内容 |
|---|---|
| `script/eval_abc123_mcac_official.py` | 官方 MCAC 数据协议、density matching、full/drop-last 双口径评测 |
| `result/logs/abc123_mcac_reproduction_summary_20260710.json` | 机器可读汇总、切片、CI 与协议诊断 |
| `result/logs/abc123_mcac_test_full2115_official_20260710.json.gz` | 当前 torchvision 默认 resize 的逐图结果 |
| `result/logs/abc123_mcac_test_full2115_official_legacyresize_20260710.json.gz` | torchvision 0.13 resize 语义的逐图结果 |

主要复现命令：

```bash
/home/czp/ws_yiyang/FreeCounting/venv/bin/python3 \
  script/eval_abc123_mcac_official.py \
  --abc-root /tmp/ABC123 \
  --checkpoint /tmp/ABC123/checkpoints/model_chkpt.ckpt \
  --mcac-root /home/czp/ljs/dataset/MCAC \
  --batch-size 2 --device cuda --legacy-tensor-resize \
  --out result/logs/abc123_mcac_test_full2115_official_legacyresize_20260710.json
```
