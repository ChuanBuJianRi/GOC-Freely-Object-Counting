"""Baseline methods for OmniCount-191 evaluation.

1. Vanilla SAM2: count = number of SAM2 masks (already in eval_omnicount.py)
2. OWLv2 class-agnostic: generic "object" prompt → count all detections
3. OWLv2 class-aware: class name prompts → per-class counts

For comparison with published results from OmniCount paper (Table 1):
  - GroundingDINO (text): mRMSE=1.29, mRMSE-nz=3.27
  - CLIPSeg (text):       mRMSE=1.54, mRMSE-nz=4.28
  - TFOC (text):          mRMSE=0.95, mRMSE-nz=2.89
  - OmniCount (text):     mRMSE=0.70, mRMSE-nz=2.00

Usage:
    # OWLv2 class-agnostic baseline
    python script/run_omnicount_baselines.py --mode class-agnostic

    # OWLv2 class-aware baseline
    python script/run_omnicount_baselines.py --mode class-aware
"""

from __future__ import annotations

import argparse, json, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from script.preprocess_omnicount import load_omnicount_coco

BINS = [
    ("0-10", 0, 10), ("11-20", 11, 20), ("21-50", 21, 50),
    ("51-100", 51, 100), ("100+", 101, 10 ** 9),
]


def bin_of(c):
    for lab, lo, hi in BINS:
        if lo <= c <= hi:
            return lab
    return "100+"


def load_owlv2(device="cuda"):
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    model_id = "google/owlv2-base-patch16-ensemble"
    processor = Owlv2Processor.from_pretrained(model_id)
    model = Owlv2ForObjectDetection.from_pretrained(model_id).to(device).eval()
    return processor, model


@torch.no_grad()
def detect_and_count(image, text_queries, processor, model, device, conf_threshold=0.1):
    """Run OWLv2 detection and return number of detections."""
    inputs = processor(text=text_queries, images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]]).to(device)
    results = processor.post_process_grounded_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=conf_threshold
    )
    boxes = results[0]["boxes"]
    return len(boxes)


@torch.no_grad()
def detect_per_class(image, class_names, processor, model, device, conf_threshold=0.1):
    """Run OWLv2 for each class, return per-class detection counts."""
    counts = {}
    for cname in class_names:
        text_queries = [[f"a photo of a {cname}", cname]]
        counts[cname] = detect_and_count(
            image, text_queries, processor, model, device, conf_threshold
        )
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="class-agnostic",
                    choices=["class-agnostic", "class-aware"])
    ap.add_argument("--omnicount-dir", default="/home/czp/official_code/dataset/omnicount/OmniCount-191")
    ap.add_argument("--out", default="")
    ap.add_argument("--conf-threshold", type=float, default=0.1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()

    device = args.device

    print("=" * 70)
    print(f"OmniCount-191 Baseline: OWLv2 ({args.mode})")
    print("=" * 70)
    print(f"conf_threshold={args.conf_threshold}")

    # Load data
    entries = load_omnicount_coco(args.omnicount_dir)
    if args.limit > 0:
        entries = entries[:args.limit]
    print(f"[data] {len(entries)} images")

    # Load OWLv2
    print("[load] Loading OWLv2...")
    t0 = time.time()
    processor, model = load_owlv2(device)
    print(f"[load] Ready in {time.time() - t0:.0f}s")

    results = []
    results_by_cat = defaultdict(list)
    errors_per_class = defaultdict(list)  # class_name -> [(pred, gt)]

    t_start = time.time()
    for i, entry in enumerate(entries):
        file_name = entry["file_name"]
        gt_total = entry["total_count"]
        category = entry["category"]
        unique_classes = entry["unique_classes"]

        try:
            image = Image.open(entry["img_path"]).convert("RGB")
        except Exception:
            continue

        if args.mode == "class-agnostic":
            # Count all objects with generic prompt
            pred = detect_and_count(
                image, [["object", "objects", "item"]],
                processor, model, device, args.conf_threshold
            )
        else:
            # Per-class counting
            per_class_preds = detect_per_class(
                image, unique_classes, processor, model, device, args.conf_threshold
            )
            pred = sum(per_class_preds.values())
            # Track per-class errors
            for cname in unique_classes:
                gt_c = entry["class_counts"].get(cname, 0)
                pred_c = per_class_preds.get(cname, 0)
                errors_per_class[cname].append({"pred": pred_c, "gt": gt_c})

        results.append({
            "file_name": file_name, "category": category,
            "gt_count": gt_total, "pred_count": pred,
        })
        results_by_cat[category].append({
            "file_name": file_name, "gt_count": gt_total, "pred_count": pred,
        })

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / max(elapsed, 0.01)
            preds = np.array([r["pred_count"] for r in results])
            gts = np.array([r["gt_count"] for r in results])
            mae = float(np.mean(np.abs(preds - gts)))
            print(f"  [{i+1}/{len(entries)}] MAE={mae:.2f}  "
                  f"rate={rate:.1f}/s  ETA={((len(entries)-i-1)/max(rate,0.01)/60):.1f}min")

    elapsed = time.time() - t_start

    # Metrics
    preds = np.array([r["pred_count"] for r in results], float)
    gts = np.array([r["gt_count"] for r in results], float)
    err = preds - gts

    summary = {
        "method": f"OWLv2-{args.mode}",
        "conf_threshold": args.conf_threshold,
        "n_images": len(results),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)),
        "mean_GT": float(np.mean(gts)),
        "nMAE": float(np.mean(np.abs(err)) / max(np.mean(gts), 1)),
        "runtime_s": float(elapsed),
    }

    print(f"\n{'='*70}")
    print(f"OWLv2 {args.mode} Results on OmniCount-191")
    print(f"{'='*70}")
    print(f"Images: {summary['n_images']}")
    print(f"MAE: {summary['MAE']:.2f}  RMSE: {summary['RMSE']:.2f}  bias: {summary['bias']:+.2f}")
    print(f"nMAE: {summary['nMAE']:.3f}  mean_GT: {summary['mean_GT']:.1f}")

    # Per-bin
    by_bin = defaultdict(list)
    for r in results:
        by_bin[bin_of(r["gt_count"])].append(r)
    print(f"\nPer-bin:")
    print(f"  {'Bin':<10} {'#':>4} {'MAE':>8} {'RMSE':>8} {'bias':>8}")
    per_bin = {}
    for lab, _, _ in BINS:
        rs = by_bin.get(lab, [])
        if not rs:
            continue
        bp = np.array([r["pred_count"] for r in rs], float)
        bg = np.array([r["gt_count"] for r in rs], float)
        be = bp - bg
        per_bin[lab] = {
            "n": len(rs), "MAE": float(np.mean(np.abs(be))),
            "RMSE": float(np.sqrt(np.mean(be ** 2))), "bias": float(np.mean(be)),
        }
        print(f"  {lab:<10} {per_bin[lab]['n']:>4} {per_bin[lab]['MAE']:>8.2f} "
              f"{per_bin[lab]['RMSE']:>8.2f} {per_bin[lab]['bias']:>+8.2f}")

    # Per-category
    print(f"\nPer-category:")
    print(f"  {'Category':<20} {'#':>4} {'MAE':>8} {'RMSE':>8} {'bias':>8}")
    per_cat = {}
    for cat in sorted(results_by_cat):
        rs = results_by_cat[cat]
        cp = np.array([r["pred_count"] for r in rs], float)
        cg = np.array([r["gt_count"] for r in rs], float)
        ce = cp - cg
        per_cat[cat] = {
            "n": len(rs), "MAE": float(np.mean(np.abs(ce))),
            "RMSE": float(np.sqrt(np.mean(ce ** 2))), "bias": float(np.mean(ce)),
        }
        print(f"  {cat:<20} {per_cat[cat]['n']:>4} {per_cat[cat]['MAE']:>8.2f} "
              f"{per_cat[cat]['RMSE']:>8.2f} {per_cat[cat]['bias']:>+8.2f}")

    # Per-class (class-aware mode only)
    per_class_summary = {}
    if args.mode == "class-aware" and errors_per_class:
        for cname, items in sorted(errors_per_class.items()):
            if len(items) < 3:
                continue
            cp = np.array([x["pred"] for x in items], float)
            cg = np.array([x["gt"] for x in items], float)
            ce = cp - cg
            per_class_summary[cname] = {
                "n": len(items), "MAE": float(np.mean(np.abs(ce))),
                "bias": float(np.mean(ce)),
            }

    output = {
        "summary": summary,
        "per_bin": per_bin,
        "per_category": per_cat,
        "per_class": per_class_summary,
        "results": results,
    }

    if args.out:
        os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
