"""Visualize OV-CUD multi-class counting on OmniCount with trained heads."""
import os
import sys
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.matrix.pairwise_features import build_pairwise_features, box_geometry, pairwise_feature_dim
from code.heads.relation_head import PairwiseRelationHead, RelationHeadConfig
from code.clustering.first_neighbor import category_aware_clustering_with_spatial
from code.counting.deduplicate import build_same_instance_components, build_same_instance_components_adaptive
from code.counting.representative import select_representatives

PALETTE = [
    (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60),
    (250, 190, 190), (0, 128, 128), (170, 110, 40), (128, 0, 0),
    (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128),
]


def color(idx):
    return PALETTE[idx % len(PALETTE)]


def decode_masks(d):
    from pycocotools import mask as mask_utils
    out = []
    for rle in d["masks_rle"]:
        counts = rle["counts"]
        if isinstance(counts, str):
            counts = counts.encode("ascii")
        out.append(mask_utils.decode({"size": rle["size"], "counts": counts}).astype(bool))
    return out


def mask_outline(m):
    m = m.astype(bool)
    er = m.copy()
    er[1:, :] &= m[:-1, :]
    er[:-1, :] &= m[1:, :]
    er[:, 1:] &= m[:, :-1]
    er[:, :-1] &= m[:, 1:]
    return m & ~er


def load_category_head(ckpt_path, device):
    from script.train_category_v2 import CosineCategoryHead
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    head = CosineCategoryHead(in_dim=ck["in_dim"], proj_dim=ck["proj_dim"], dropout=0.3, num_layers=2)
    head.load_state_dict(ck["head"])
    head.to(device).eval()
    return head


def load_relation_head(ckpt_path, z_dim, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    feat_dim = pairwise_feature_dim(z_dim)
    head = PairwiseRelationHead(RelationHeadConfig(
        feat_dim=feat_dim, hidden_dim=ck["hidden_dim"], num_layers=ck["num_layers"], dropout=0.1))
    head.load_state_dict(ck["relation_head"])
    head.to(device).eval()
    return head


@torch.no_grad()
def relation_matrices(rel_head, z, cat_probs, bbox, device):
    n = z.shape[0]
    A_sem = np.zeros((n, n), dtype=np.float32)
    A_inst = np.zeros((n, n), dtype=np.float32)
    A_part = np.zeros((n, n), dtype=np.float32)
    if n < 2:
        return A_sem, A_inst, A_part
    ii, jj = torch.triu_indices(n, n, offset=1, device=device)
    z = z.to(device); p = cat_probs.to(device); box = bbox.to(device)
    geom_ij = box_geometry(box[ii], box[jj])
    geom_ji = box_geometry(box[jj], box[ii])
    phi_ij = build_pairwise_features(z[ii], z[jj], p[ii], p[jj], box[ii], box[jj], geom=geom_ij)
    phi_ji = build_pairwise_features(z[jj], z[ii], p[jj], p[ii], box[jj], box[ii], geom=geom_ji)
    out_ij = rel_head(phi_ij)
    out_ji = rel_head(phi_ji)
    sem = 0.5 * (out_ij["sem"] + out_ji["sem"])
    inst = 0.5 * (out_ij["inst"] + out_ji["inst"])
    part = out_ij["part"]
    ii_np = ii.cpu().numpy(); jj_np = jj.cpu().numpy()
    A_sem[ii_np, jj_np] = sem.cpu().numpy(); A_sem[jj_np, ii_np] = sem.cpu().numpy()
    A_inst[ii_np, jj_np] = inst.cpu().numpy(); A_inst[jj_np, ii_np] = inst.cpu().numpy()
    A_part[ii_np, jj_np] = part.cpu().numpy()
    return A_sem, A_inst, A_part


def count_and_group(d, cat_head, rel_head, tp, class_names, device, tau_inst=0.5, tau_aff=0.3):
    n = d["z"].shape[0]
    h, w = int(d["height"]), int(d["width"])
    if n == 0:
        return [], np.zeros((0,), dtype=int)
    z = d["z"].float()
    bbox = d["bbox"].float()
    with torch.no_grad():
        logits = cat_head(z.to(device), tp.to(device))
        cat_probs = torch.softmax(logits, dim=-1).cpu()
    top_class = cat_probs.argmax(dim=-1).numpy()
    A_sem, A_inst, A_part = relation_matrices(rel_head, z, cat_probs, bbox, device)
    bbox_np = bbox.numpy()
    image_area = float(h * w)
    groups = category_aware_clustering_with_spatial(
        cat_probs.numpy(), A_sem, bbox_np, image_area,
        tau_affinity=tau_aff, max_group_size=30, use_bucketing=True)
    cp = cat_probs.numpy()
    result_groups = []
    for group in groups:
        if len(group) == 0:
            continue
        if len(group) > 20:
            comps = build_same_instance_components_adaptive(
                group, A_inst, base_tau=tau_inst, use_greedy=True, max_comp_size=5)
        else:
            comps = build_same_instance_components(group, A_inst, tau_inst=tau_inst)
        reps = select_representatives(comps, A_part, cp, bbox_np, image_area, min_category_conf=0.05)
        if not reps:
            continue
        cls_ids = [int(top_class[r]) for r in reps]
        cls_id = max(set(cls_ids), key=cls_ids.count)
        cname = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
        result_groups.append({"class_name": cname, "count": len(reps), "rep_indices": reps})
    return result_groups, top_class


def visualize(image, masks, bboxes, result_groups, alpha=0.45):
    base = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    overlay = np.array(base).astype(np.float32)
    for gi, g in enumerate(result_groups):
        col = np.array(color(gi), dtype=np.float32)
        for r in g["rep_indices"]:
            if r < len(masks):
                m = masks[r]
                if m.shape == overlay.shape[:2]:
                    overlay[m] = (1 - alpha) * overlay[m] + alpha * col
                    overlay[mask_outline(m)] = col
    out = Image.fromarray(overlay.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(out)
    for gi, g in enumerate(result_groups):
        col = color(gi)
        for r in g["rep_indices"]:
            if r < len(bboxes):
                x, y, bw, bh = [float(v) for v in bboxes[r]]
                draw.rectangle([x, y, x + bw, y + bh], outline=col, width=2)
    return out


def side_by_side(orig, viz, header):
    lines = header.count("\n") + 1
    pad_top = 8 + 12 * lines
    W = orig.width + viz.width + 12
    H = max(orig.height, viz.height) + pad_top
    canvas = Image.new("RGB", (W, H), (20, 20, 20))
    canvas.paste(orig, (0, pad_top))
    canvas.paste(viz, (orig.width + 12, pad_top))
    ImageDraw.Draw(canvas).text((4, 4), header, fill=(255, 255, 255))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/home/czp/ws_yiyang/ovcud_cache/omnicount_test")
    ap.add_argument("--image-root", default="/home/czp/official_code/dataset/omnicount/OmniCount-191")
    ap.add_argument("--category-ckpt", default="result/checkpoints/category_coco80_cosine.pt")
    ap.add_argument("--relation-ckpt", default="result/checkpoints/coco_relation_1152.pt")
    ap.add_argument("--text-prototypes", default="result/checkpoints/text_prototypes_coco80.pt")
    ap.add_argument("--class-names", default="/home/czp/official_code/cache/coco_class_names.json")
    ap.add_argument("--num-images", type=int, default=8)
    ap.add_argument("--min-classes", type=int, default=2)
    ap.add_argument("--min-count", type=int, default=0, help="only keep images with gt_count >= this")
    ap.add_argument("--sort-by-count", action="store_true", help="pick the densest images first")
    ap.add_argument("--tau-inst", type=float, default=0.5)
    ap.add_argument("--tau-aff", type=float, default=0.3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default="result/viz_omnicount")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    class_names = json.load(open(args.class_names))

    files = sorted(Path(args.cache_dir).glob("*.pt"))

    if args.sort_by_count:
        # scan all, keep candidates matching filters, then sort by gt_count desc
        cands = []
        for fp in files:
            d = torch.load(fp, map_location="cpu", weights_only=False)
            uc = d.get("unique_classes") or []
            gt = int(d.get("gt_count", 0))
            if len(uc) >= args.min_classes and d["z"].shape[0] >= 2 and gt >= args.min_count:
                cands.append((gt, fp, d))
        cands.sort(key=lambda t: t[0], reverse=True)
        picked = [(fp, d) for _, fp, d in cands[:args.num_images]]
    else:
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(files))
        picked = []
        for oi in order:
            fp = files[int(oi)]
            d = torch.load(fp, map_location="cpu", weights_only=False)
            uc = d.get("unique_classes") or []
            gt = int(d.get("gt_count", 0))
            if len(uc) >= args.min_classes and d["z"].shape[0] >= 2 and gt >= args.min_count:
                picked.append((fp, d))
            if len(picked) >= args.num_images:
                break
    print("[viz] picked", len(picked), "multi-class images (unique_classes >=", args.min_classes, ")")

    z_dim = picked[0][1]["z"].shape[1]
    cat_head = load_category_head(args.category_ckpt, device)
    rel_head = load_relation_head(args.relation_ckpt, z_dim, device)
    tp = F.normalize(torch.load(args.text_prototypes, map_location=device, weights_only=False).float(), dim=-1)

    summary = []
    for i, (fp, d) in enumerate(picked):
        groups, _ = count_and_group(d, cat_head, rel_head, tp, class_names, device,
                                    tau_inst=args.tau_inst, tau_aff=args.tau_aff)
        masks = decode_masks(d)
        bboxes = d["bbox"].float().numpy()
        category = d.get("category", "unknown")
        img_path = Path(args.image_root) / category / "test" / d["file_name"]
        if not img_path.exists():
            cand = list(Path(args.image_root).rglob(d["file_name"]))
            img_path = cand[0] if cand else None
        if img_path is None or not img_path.exists():
            print("[viz] image not found:", d["file_name"], "-> skip")
            continue
        image = np.array(Image.open(img_path).convert("RGB"))
        gt_cc = d.get("class_counts", {})
        pred_str = "; ".join("%s:%d" % (g["class_name"], g["count"]) for g in groups) or "(empty)"
        pred_total = sum(g["count"] for g in groups)
        header = ("%s  cat=%s\nGT: %s (total=%d)\nPred: %s (total=%d)" % (
            d["file_name"][:40], category, gt_cc, int(d.get("gt_count", 0)), pred_str, pred_total))
        viz = visualize(image, masks, bboxes, groups)
        orig = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        combo = side_by_side(orig, viz, header)
        out_path = out_dir / ("%02d_%s.png" % (i, d["file_name"].split(".")[0][:30]))
        combo.save(out_path)
        print("[viz]", out_path.name, "| GT", gt_cc, "| Pred", pred_str)
        summary.append({"file": d["file_name"], "category": category, "gt_class_counts": gt_cc,
                        "gt_count": int(d.get("gt_count", 0)),
                        "pred_groups": [{"class": g["class_name"], "count": g["count"]} for g in groups],
                        "pred_total": pred_total, "out": str(out_path)})

    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2, ensure_ascii=False)
    print("[viz] done ->", out_dir / "summary.json")


if __name__ == "__main__":
    main()

