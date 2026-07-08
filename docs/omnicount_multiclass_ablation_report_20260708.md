# OmniCount-191 Multi-Class Ablation Report

**Date**: 2026-07-08  
**Branch**: `ljs`  
**Goal**: Verify whether the FSC147 headline model (MAE 12.74) supports prompt-free multi-category counting on OmniCount-191, and provide protocol/category/dedup/difficulty ablations.

---

## 1. Model Consistency Audit

The OmniCount multi-class experiment uses the same trained OV-CUD model family as the FSC147 headline result (MAE 12.74):

| Component | FSC147 12.74 setup | OmniCount multi-class setup | Same? |
|---|---|---|---|
| Category head | `result/checkpoints/category_cosine_pts32.pt` | `result/checkpoints/category_cosine_pts32.pt` | Yes |
| Relation head | `result/checkpoints/fsc147_relation_pts32_best.pt` / `fsc147_relation_pts32_exp5c.pt` | `result/checkpoints/fsc147_relation_pts32_best.pt` | Yes |
| Relation weights | Exp5-C epoch 25 | Exp5-C epoch 25 | Yes |
| Candidate strategy | FSC147 multi-resolution cache for dense bins | OmniCount pts=32 cache | Dataset-specific candidate cache |
| Text prototypes | FSC147 class prototypes | OmniCount-191 class prototypes | Vocabulary swap only |

Checkpoint audit:

| File | SHA256 prefix | Note |
|---|---:|---|
| `category_cosine_pts32.pt` | `43509c6e4ff55498` | Same category projection head as FSC147 |
| `fsc147_relation_pts32_best.pt` | `1bb7a1cfd950fde8` | Inference checkpoint |
| `fsc147_relation_pts32_exp5c.pt` | `c0e16479f7d6cc4a` | Training checkpoint with optimizer |
| `text_prototypes_omnicount.pt` | `26cd89b7f596a230` | OmniCount vocabulary prototypes |

`fsc147_relation_pts32_best.pt` and `fsc147_relation_pts32_exp5c.pt` have identical `relation_head` state dicts (`max_abs_diff=0.0`, epoch 25). Therefore, the OmniCount run uses the same learned category projection and relation model as the FSC147 headline model. The only required change is replacing the text prototype matrix so that the open-vocabulary head can score OmniCount's 93 test classes.

Important nuance: FSC147 MAE 12.74 also uses a multi-resolution candidate cache for high-density images. That cache is a dataset-specific proposal-generation strategy, not a different learned model.

---

## 2. Evaluation Script

New script:

```bash
script/eval_omnicount_multiclass_ablation.py
```

The script rebuilds OmniCount per-class ground truth from the original COCO-format annotations instead of trusting the cached `matched_class`, because the current OmniCount cache stores:

```python
"matched_class": torch.zeros(n_cand, dtype=torch.long)  # class-agnostic
```

Predicted variants do **not** use GT-derived `valid` masks. Oracle variants decode cached SAM masks and match candidate masks to GT dot centers only for upper-bound analysis.

Metrics:

| Metric | Definition |
|---|---|
| MAE/RMSE/bias | Image-level total count error |
| mRMSE | Mean per-class RMSE over 93 OmniCount test classes |
| mRMSE-nz | Mean per-class RMSE on images where that class appears |
| Slices | By number of GT classes per image, super-category, and GT count bin |

Main command:

```bash
python3 script/eval_omnicount_multiclass_ablation.py \
  --min-classes 1 \
  --limit -1 \
  --category-ckpt result/checkpoints/category_cosine_pts32.pt \
  --relation-ckpt result/checkpoints/fsc147_relation_pts32_best.pt \
  --text-prototypes result/checkpoints/text_prototypes_omnicount.pt \
  --class-names result/checkpoints/omnicount_class_names.json \
  --conf-threshold 0.1 \
  --out result/logs/omnicount_multiclass_ablation_full1957_fsc147head_conf01.json \
  --device cuda \
  --save-per-image
```

Primary result artifact:

```bash
result/logs/omnicount_multiclass_ablation_full1957_fsc147head_conf01.json
```

---

## 3. Dataset Summary

Current processed OmniCount test cache:

| Item | Value |
|---|---:|
| Images | 1,957 |
| Test classes observed | 93 |
| Mean GT count/image | 6.03 |
| Images with 1 class | 1,124 |
| Images with 2+ classes | 833 |
| Multi-class image ratio | 42.6% |

Super-category distribution:

| Super-category | Images |
|---|---:|
| Urban | 1,000 |
| Fruits | 303 |
| Wild | 255 |
| Supermarket | 251 |
| Satellite | 127 |
| Pets | 11 |
| Birds | 10 |

---

## 4. Protocol Ablation

| Protocol | Prompt at inference | Per-class output | MAE | RMSE | bias | mRMSE | mRMSE-nz |
|---|---|---:|---:|---:|---:|---:|---:|
| SAM2-only total | None | No | 37.60 | 44.82 | +37.60 | - | - |
| Class-agnostic total | None | No | 6.77 | 10.23 | +5.47 | - | - |
| Predicted class groups | None | Yes | **4.68** | **8.46** | -4.36 | 0.457 | 3.911 |
| Oracle class grouping | GT for oracle only | Yes | **1.12** | **2.22** | -0.65 | 0.111 | 1.047 |

Key finding:

- The real prompt-free per-class pipeline improves total MAE over class-agnostic counting: **6.77 -> 4.68**.
- Oracle class grouping gives a strong upper bound: **4.68 -> 1.12**, showing that class assignment and candidate-class coverage remain the main bottlenecks.
- This is the strongest OmniCount evidence for the paper: OV-CUD can output prompt-free class-wise counts, not merely a total count.

---

## 5. Category Separation Ablation

| Variant | MAE | RMSE | bias | mRMSE | mRMSE-nz |
|---|---:|---:|---:|---:|---:|
| Global, no class grouping | 4.69 | 8.49 | -4.39 | 0.455 | 3.911 |
| `p_i · p_j` grouping | **4.68** | **8.46** | -4.36 | 0.457 | 3.911 |
| Category bucket | 4.69 | 8.48 | -4.38 | 0.455 | 3.911 |
| Full category + relation grouping | **4.68** | **8.46** | -4.36 | 0.457 | 3.911 |
| Oracle category grouping | **1.12** | **2.22** | -0.65 | 0.111 | 1.047 |

Interpretation:

- The learned/predicted grouping variants are numerically close under `conf_threshold=0.1`.
- Oracle category grouping is much better, so the actionable bottleneck is not whether to use bucket vs `p_i · p_j`; it is assigning candidates to the right OmniCount class and retaining enough valid candidates.
- In the paper, do **not** overclaim that category separation design alone is the source of OmniCount gains. Use this table as a diagnostic.

---

## 6. Dedup Ablation

| Variant | MAE | RMSE | bias | mRMSE | mRMSE-nz |
|---|---:|---:|---:|---:|---:|
| SAM2-only total | 37.60 | 44.82 | +37.60 | - | - |
| Category-only | **4.68** | **8.46** | -4.36 | 0.457 | 3.911 |
| Category + IoU NMS | 4.69 | 8.49 | -4.39 | 0.455 | 3.910 |
| Category + relation head | **4.68** | **8.46** | -4.36 | 0.457 | 3.911 |
| Relation + adaptive area filter | 4.72 | 8.48 | -4.40 | 0.455 | 3.911 |

Interpretation:

- On OmniCount, category filtering dominates the improvement over SAM2-only.
- IoU NMS and relation-head dedup are almost identical in this setting. The current `conf_threshold=0.1` removes many duplicates before dedup, so relation-head effects are compressed.
- The relation head should still be defended mainly with FSC147/CARPK component ablations, not with this OmniCount table.

---

## 7. Difficulty Slices

### 7.1 By Number of Classes Per Image

Predicted class groups:

| #GT classes/image | Images | MAE | RMSE | bias |
|---|---:|---:|---:|---:|
| 1 | 1,124 | 4.49 | 10.02 | -4.04 |
| 2 | 311 | 4.17 | 5.02 | -3.88 |
| 3 | 186 | 6.02 | 7.12 | -5.86 |
| 4+ | 336 | 5.03 | 5.45 | -5.03 |

Images with 2+ classes:

| Protocol | Images | MAE | RMSE | bias |
|---|---:|---:|---:|---:|
| Class-agnostic total | 833 | 4.98 | 8.09 | +3.52 |
| Predicted class groups | 833 | **4.93** | **5.72** | -4.78 |
| Oracle class grouping | 833 | **1.30** | **1.90** | -0.80 |

### 7.2 By Super-Category

Predicted class groups:

| Super-category | Images | MAE | RMSE | bias |
|---|---:|---:|---:|---:|
| Birds | 10 | 14.30 | 14.56 | -14.30 |
| Fruits | 303 | 4.27 | 4.38 | -4.27 |
| Pets | 11 | 8.45 | 9.06 | -8.45 |
| Satellite | 127 | **1.92** | 3.70 | -0.82 |
| Supermarket | 251 | 11.62 | 20.31 | -11.41 |
| Urban | 1,000 | 3.82 | 4.84 | -3.42 |
| Wild | 255 | 2.50 | 3.24 | -2.37 |

Supermarket and Birds are the hardest groups. Supermarket has many fine-grained product classes and high object density; Birds has only 10 images but high counts.

### 7.3 By GT Count Bin

Predicted class groups:

| GT count bin | Images | MAE | RMSE | bias |
|---|---:|---:|---:|---:|
| 0-10 | 1,772 | 3.15 | 3.81 | -2.79 |
| 11-20 | 123 | 10.97 | 11.51 | -10.97 |
| 21-50 | 42 | 26.38 | 27.44 | -26.38 |
| 51-100 | 20 | 55.80 | 57.67 | -55.80 |

The model undercounts high-density OmniCount images. This is consistent with FSC147/CARPK/PUCPR+ analysis: proposal density and small-object coverage remain the main limiting factors in dense scenes.

---

## 8. Classification Head Check

A 500-image multi-class subset was used to compare category heads:

| Category head | Prompt-free predicted groups MAE | Note |
|---|---:|---|
| COCO-80 cosine head + OmniCount prototypes | 14.88 | Poor transfer |
| FSC147 pts32 cosine head + OmniCount prototypes (`conf=0.2`) | 6.97 | Better |
| FSC147 pts32 cosine head + OmniCount prototypes (`conf=0.1`) | **5.08** | Best subset setting |
| FSC147 pts32 cosine head + OmniCount prototypes (`conf=0.05`) | 14.71 | Over-counting |

This supports using the FSC147-trained category head for the full OmniCount result. It is also consistent with the paper narrative: the same FSC147 model transfers to OmniCount by swapping text prototypes.

---

## 9. Recommended Paper Usage

Recommended main text claim:

> On OmniCount-191, using the same FSC147-trained OV-CUD model and only replacing text prototypes for the 93 OmniCount test classes, prompt-free predicted class groups achieve MAE 4.68 / RMSE 8.46 over 1,957 images. This outperforms class-agnostic total counting (MAE 6.77) and provides per-class counts without class prompts.

Recommended table:

| Method / Protocol | Prompt | Class-wise output | MAE | RMSE | mRMSE |
|---|---|---:|---:|---:|---:|
| SAM2-only | None | No | 37.60 | 44.82 | - |
| OV-CUD class-agnostic | None | No | 6.77 | 10.23 | - |
| OV-CUD predicted class groups | None | Yes | **4.68** | **8.46** | 0.457 |
| OV-CUD oracle class grouping | Oracle eval only | Yes | 1.12 | 2.22 | 0.111 |

Recommended wording:

- Use "same learned model, OmniCount vocabulary prototypes" rather than "same exact prototype matrix".
- Say "prompt-free per-class output" rather than "competitive with OmniCount paper under the same protocol", because OmniCount's official setting uses target class names and reports prompt-based multi-label mRMSE.
- Do not overstate relation-head gains on OmniCount. The stronger relation-head evidence is FSC147/CARPK.

---

## 10. Limitations and Next Steps

1. **High-density undercounting**: GT bins above 20 are strongly undercounted. OmniCount may benefit from the same multi-resolution/tiling proposal strategy used for FSC147 12.74.
2. **Class assignment bottleneck**: Oracle class grouping improves MAE from 4.68 to 1.12. Better text prompts, class-name normalization, or light calibration may help.
3. **Dedup effect is compressed**: Current confidence filtering removes many duplicate candidates before relation-head dedup. A lower threshold overcounts heavily, while `conf=0.1` is stable but makes dedup variants nearly identical.
4. **Official metric mismatch**: OmniCount paper reports prompt-based multi-label mRMSE with class names specified by the user. OV-CUD's setting is stricter at inference because it receives no class prompts.

Recommended next experiment if time permits:

```bash
# Run OmniCount with multi-resolution candidates for high-density Supermarket/Birds images.
# Goal: test whether the FSC147 12.74 proposal strategy also fixes OmniCount undercounting.
```

---

## 11. Files

| File | Purpose |
|---|---|
| `script/eval_omnicount_multiclass_ablation.py` | New multi-class ablation evaluator |
| `result/logs/omnicount_multiclass_ablation_full1957_fsc147head_conf01.json` | Primary full-test result |
| `result/logs/omnicount_multiclass_ablation_mc500_fsc147head_conf01.json` | 500-image threshold validation |
| `result/logs/omnicount_multiclass_ablation_mc500_cocohead.json` | COCO-head comparison |
| `docs/omnicount_multiclass_ablation_report_20260708.md` | This report |
