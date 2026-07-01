"""P2-3: OWLv2 Open-Vocabulary Detection Baseline for FSC147 Counting.

Runs OWLv2 (via HuggingFace transformers) with class-name prompts on FSC147 test
images and compares detection-based counting against OV-CUD's prompt-free counting.

OWLv2 represents a strong prompt-based baseline: it receives the target class name
as a text prompt, while OV-CUD receives no prompt at all.

Usage:
    python script/run_owlv2_baseline.py \
        --images-file result/logs/sample100_test.json \
        --fsc147-dir /home/czp/official_code/dataset/FSC147 \
        --out result/logs/p2_owlv2_baseline.json \
        --n-images 100

For the full FSC147 test set (1190 images), omit --images-file and use --split test.
"""

from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

BINS = [
    ("0-10", 0, 10), ("11-20", 11, 20), ("21-50", 21, 50),
    ("51-100", 51, 100), ("100+", 101, 10 ** 9),
]


def bin_of(c):
    for lab, lo, hi in BINS:
        if lo <= c <= hi:
            return lab
    return "100+"


def load_fsc147_data(fsc147_dir: str, images_file: str | None, split: str, n_images: int):
    """Load FSC147 image list, class mapping, and GT counts."""
    # Image -> class name
    class_map = {}
    class_file = os.path.join(fsc147_dir, "ImageClasses_FSC147.txt")
    with open(class_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                class_map[parts[0]] = parts[1]

    # Image -> GT count
    ann_file = os.path.join(fsc147_dir, "annotation_FSC147_384.json")
    with open(ann_file) as f:
        ann = json.load(f)

    # Image list
    if images_file and os.path.exists(images_file):
        with open(images_file) as f:
            img_list = json.load(f)
        if isinstance(img_list, dict):
            img_list = list(img_list.keys())
    else:
        split_file = os.path.join(fsc147_dir, "Train_Test_Val_FSC_147.json")
        with open(split_file) as f:
            splits = json.load(f)
        img_list = splits.get(split, splits.get("test", []))
        if isinstance(img_list, dict):
            img_list = list(img_list.keys())

    if n_images > 0:
        img_list = img_list[:n_images]

    # Build data entries
    entries = []
    for img_name in img_list:
        if img_name not in ann:
            continue
        gt_count = len(ann[img_name]["points"])
        class_name = class_map.get(img_name, "unknown")
        img_path = ann[img_name].get("img_path", "")
        if not img_path or not os.path.exists(img_path):
            # Try alternative path
            alt_path = os.path.join(fsc147_dir, "images_384_VarV2", img_name)
            if os.path.exists(alt_path):
                img_path = alt_path
            else:
                continue
        entries.append({
            "img_name": img_name,
            "img_path": img_path,
            "class_name": class_name,
            "gt_count": gt_count,
        })

    print(f"[data] {len(entries)} images, {len(set(e['class_name'] for e in entries))} unique classes")
    return entries


def load_owlv2(device="cuda"):
    """Load OWLv2 model and processor."""
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    model_id = "google/owlv2-base-patch16-ensemble"
    processor = Owlv2Processor.from_pretrained(model_id)
    model = Owlv2ForObjectDetection.from_pretrained(model_id).to(device).eval()
    return processor, model


@torch.no_grad()
def detect_and_count(image, class_name, processor, model, device, conf_threshold=0.1):
    """Run OWLv2 detection and return predicted count."""
    # Prepare text queries: the target class + some variations
    text_queries = [[f"a photo of {class_name}", class_name]]

    inputs = processor(text=text_queries, images=image, return_tensors="pt").to(device)

    outputs = model(**inputs)

    # Post-process
    target_sizes = torch.tensor([image.size[::-1]]).to(device)
    results = processor.post_process_grounded_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=conf_threshold
    )

    boxes = results[0]["boxes"]
    scores = results[0]["scores"]
    labels = results[0]["labels"]

    # Count bounding boxes above threshold
    n_detections = len(boxes)

    return {
        "n_detections": n_detections,
        "mean_score": float(scores.mean()) if len(scores) > 0 else 0.0,
        "max_score": float(scores.max()) if len(scores) > 0 else 0.0,
        "boxes": boxes.cpu().numpy().tolist() if len(boxes) > 0 else [],
        "scores": scores.cpu().numpy().tolist() if len(scores) > 0 else [],
    }


def main():
    ap = argparse.ArgumentParser(description="P2-3: OWLv2 baseline for FSC147 counting")
    ap.add_argument("--images-file", default="result/logs/sample100_test.json",
                    help="JSON file with image list (sample100)")
    ap.add_argument("--fsc147-dir", default="/home/czp/official_code/dataset/FSC147")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--out", default="result/logs/p2_owlv2_baseline.json")
    ap.add_argument("--n-images", type=int, default=100, help="Max images (0=all)")
    ap.add_argument("--conf-threshold", type=float, default=0.1,
                    help="Detection confidence threshold")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device

    print("=" * 70)
    print("P2-3: OWLv2 Open-Vocabulary Detection Baseline")
    print("=" * 70)
    print(f"conf_threshold={args.conf_threshold}")

    # Load data
    entries = load_fsc147_data(args.fsc147_dir, args.images_file, args.split, args.n_images)

    # Load model
    print("\n[load] Loading OWLv2 model...")
    t0 = time.time()
    processor, model = load_owlv2(device)
    print(f"[load] Model ready in {time.time() - t0:.0f}s")

    # Run detection
    print(f"\n[run] Processing {len(entries)} images...")
    results = []
    preds, gts, class_names = [], [], []
    per_class = {}
    per_bin = {lab: {"preds": [], "gts": []} for lab, _, _ in BINS}

    t_start = time.time()
    for i, entry in enumerate(entries):
        img_name = entry["img_name"]
        class_name = entry["class_name"]
        gt = entry["gt_count"]

        try:
            image = Image.open(entry["img_path"]).convert("RGB")
        except Exception as e:
            print(f"  [warn] Cannot load {img_name}: {e}")
            continue

        detection = detect_and_count(image, class_name, processor, model, device, args.conf_threshold)
        pred = detection["n_detections"]

        preds.append(pred)
        gts.append(gt)
        class_names.append(class_name)

        results.append({
            "img_name": img_name,
            "class_name": class_name,
            "gt_count": gt,
            "pred_count": pred,
            "mean_score": detection["mean_score"],
            "max_score": detection["max_score"],
        })

        # Per-class stats
        if class_name not in per_class:
            per_class[class_name] = {"preds": [], "gts": []}
        per_class[class_name]["preds"].append(pred)
        per_class[class_name]["gts"].append(gt)

        # Per-bin stats
        b = bin_of(gt)
        per_bin[b]["preds"].append(pred)
        per_bin[b]["gts"].append(gt)

        if (i + 1) % 20 == 0 or i == 0:
            errs = np.abs(np.array(preds) - np.array(gts))
            elapsed = time.time() - t_start
            rate = (i + 1) / max(elapsed, 0.01)
            print(f"  [{i+1}/{len(entries)}] MAE={np.mean(errs):.2f}  rate={rate:.1f}/s  "
                  f"ETA={((len(entries)-i-1)/max(rate,0.01))/60:.1f}min")

    # ---- Summary ----
    elapsed = time.time() - t_start
    preds = np.array(preds)
    gts = np.array(gts)
    errors = np.abs(preds - gts)
    signed_err = preds - gts

    summary = {
        "method": "OWLv2-base-patch16-ensemble",
        "conf_threshold": args.conf_threshold,
        "n_images": len(results),
        "mae": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "bias": float(np.mean(signed_err)),
        "mean_gt": float(np.mean(gts)),
        "nmae": float(np.mean(errors) / max(np.mean(gts), 1)),
        "mean_pred": float(np.mean(preds)),
        "runtime_total_s": float(elapsed),
        "runtime_per_image_s": float(elapsed / max(len(results), 1)),
    }

    # Per-bin
    bin_summary = {}
    for lab, _, _ in BINS:
        bd = per_bin[lab]
        if not bd["preds"]:
            continue
        bp = np.array(bd["preds"])
        bg = np.array(bd["gts"])
        be = np.abs(bp - bg)
        bin_summary[lab] = {
            "n": len(be),
            "mae": float(np.mean(be)),
            "rmse": float(np.sqrt(np.mean(be ** 2))),
            "bias": float(np.mean(bp - bg)),
        }

    # Per-class (classes with >= 3 images)
    class_summary = {}
    for cname, cd in sorted(per_class.items()):
        if len(cd["preds"]) < 3:
            continue
        cp = np.array(cd["preds"])
        cg = np.array(cd["gts"])
        ce = np.abs(cp - cg)
        class_summary[cname] = {
            "n": len(ce),
            "mae": float(np.mean(ce)),
            "bias": float(np.mean(cp - cg)),
        }

    print(f"\n{'='*70}")
    print("P2-3 Results: OWLv2 Detection Baseline")
    print(f"{'='*70}")
    print(f"n_images={summary['n_images']}")
    print(f"MAE={summary['mae']:.2f}  RMSE={summary['rmse']:.2f}  bias={summary['bias']:+.2f}")
    print(f"nMAE={summary['nmae']:.3f}  mean_GT={summary['mean_gt']:.1f}")
    print(f"runtime={elapsed:.0f}s ({elapsed/len(results):.1f}s/img)")

    print(f"\nPer-bin:")
    print(f"  {'Bin':<10} {'#':>4} {'MAE':>8} {'RMSE':>8} {'bias':>8}")
    for lab, _, _ in BINS:
        if lab in bin_summary:
            b = bin_summary[lab]
            print(f"  {lab:<10} {b['n']:>4} {b['mae']:>8.2f} {b['rmse']:>8.2f} {b['bias']:>+8.2f}")

    print(f"\nPer-class (n>=3):")
    print(f"  {'Class':<25} {'#':>4} {'MAE':>8} {'bias':>8}")
    for cname, cd in sorted(class_summary.items(), key=lambda x: -x[1]['n']):
        print(f"  {cname:<25} {cd['n']:>4} {cd['mae']:>8.2f} {cd['bias']:>+8.2f}")

    # Save
    output = {
        "summary": summary,
        "per_bin": bin_summary,
        "per_class": class_summary,
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.out}")

    # Comparison with OV-CUD
    print(f"\n{'='*70}")
    print("Comparison: OWLv2 vs OV-CUD")
    print(f"{'='*70}")
    print(f"  OWLv2 (prompt-based):  MAE={summary['mae']:.2f}")
    print(f"  OV-CUD (prompt-free):  MAE=61.36 (all-groups-sum, simplified pipeline)")
    print(f"  OV-CUD (full pipeline): MAE=8.73 (best)")
    print(f"\n  OWLv2 uses the GT class name as a text prompt.")
    print(f"  OV-CUD receives no prompt — it discovers and counts all categories.")
    print(f"  The gap quantifies the cost of being prompt-free.")


if __name__ == "__main__":
    main()
