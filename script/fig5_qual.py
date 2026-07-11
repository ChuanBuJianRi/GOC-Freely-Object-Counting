"""Generate Figure 5 (fig:qual) qualitative grid for OV-CUD paper.

Modes:
  scan   -- run final-config pipeline over a cache dir, dump per-image json
  render -- render selected images (per-group colored mask overlays) and
            assemble the 2-column grid_qual.png / grid_qual.pdf

Pipeline logic is copied verbatim from script/eval_carpk.py::count_image_carpk,
extended only to also return each group's representative candidate indices.
Final config: tau_inst=0.99, tau_affinity=0.1, conf_threshold=0.1.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/czp/ljs/Freely-Object-Counting")
sys.path.insert(0, str(REPO))

from script.eval_carpk import (  # noqa: E402
    load_category_head, load_relation_head, get_category_probs,
)
from code.clustering.first_neighbor import (  # noqa: E402
    category_aware_clustering_with_spatial,
)
from code.counting.deduplicate import (  # noqa: E402
    build_same_instance_components,
    build_same_instance_components_adaptive,
)
from code.counting.representative import select_representatives  # noqa: E402

TAU_INST = 0.99
TAU_AFF = 0.1
CONF = 0.1
MAX_GROUP_SIZE = 30

CKPT = REPO / "result/checkpoints"
OUT_DIR = REPO / "result_main"

CACHE_CARPK = Path("/home/czp/ws_yiyang/ovcud_cache/carpk_test")
CACHE_FSC = Path("/home/czp/ws_yiyang/ovcud_cache/fsc147_test_pts32_sample")
IMG_CARPK = Path("/home/czp/official_code/datasets/CARPK_devkit/data/Images")
IMG_FSC = Path("/home/czp/official_code/dataset/FSC147/images_384_VarV2")

# 8 bright colors, cycled per group
COLORS = [
    (255, 59, 48), (255, 204, 0), (52, 199, 89), (0, 122, 255),
    (255, 149, 0), (175, 82, 222), (90, 200, 250), (255, 45, 85),
]
ALPHA = 0.45


def load_models(device):
    category_head = load_category_head(CKPT / "category_cosine_pts32.pt", device)
    relation_head = load_relation_head(CKPT / "fsc147_relation_pts32_exp5c.pt", device)
    tp = torch.nn.functional.normalize(
        torch.load(CKPT / "text_prototypes_fsc147.pt", map_location="cpu",
                   weights_only=False).float(), dim=-1).to(device)
    cat_meta = json.load(open(CKPT / "text_prototypes_fsc147_categories.json"))
    cat_list = sorted(cat_meta["categories"], key=lambda c: c["contiguous_id"])
    categories = [c["name"] for c in cat_list]
    return category_head, relation_head, tp, categories


def count_image_viz(d, category_head, relation_head, text_prototypes, device,
                    tau_inst=TAU_INST, tau_affinity=TAU_AFF,
                    conf_threshold=CONF, max_group_size=MAX_GROUP_SIZE):
    """count_image_carpk from script/eval_carpk.py, additionally returning
    per-group representative candidate indices for visualization."""
    h, w = int(d["height"]), int(d["width"])
    file_name = d.get("file_name", "unknown")
    gt_count = int(d.get("gt_count", 0))
    z = d["z"].float()
    bbox_np = d["bbox"].float().numpy()
    n_cand = z.shape[0]

    empty = {"pred_count": 0, "n_candidates": n_cand, "n_groups": 0,
             "gt_count": gt_count, "file_name": file_name, "group_infos": []}
    if n_cand == 0:
        return empty

    category_probs = get_category_probs(category_head, z, text_prototypes, device)
    top_conf = category_probs.max(axis=1)
    top_class = category_probs.argmax(axis=1)

    conf_valid = top_conf >= conf_threshold
    valid_orig = np.asarray(d["valid"]) > 0
    effective_valid = valid_orig & conf_valid
    if effective_valid.sum() == 0:
        return empty

    A_sem = np.eye(n_cand, dtype=np.float32)
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            sem_score = float((category_probs[i] * category_probs[j]).sum())
            A_sem[i, j] = A_sem[j, i] = sem_score * 10.0

    from code.matrix.pairwise_features import build_pairwise_features, box_geometry
    p_t = torch.from_numpy(category_probs).float()
    box_t = d["bbox"].float()
    z_t = z
    max_cand = min(n_cand, 200)
    if n_cand > max_cand:
        keep = np.argsort(-top_conf)[:max_cand]
        z_t = z_t[keep]; p_t = p_t[keep]; box_t = box_t[keep]
        keep_set = set(keep)
    else:
        keep = None
        keep_set = set(range(n_cand))

    ii, jj = np.triu_indices(len(keep_set) if n_cand > max_cand else n_cand, k=1)
    A_inst = np.zeros((n_cand, n_cand), dtype=np.float32)
    if len(ii) > 0:
        zi, zj = z_t[ii], z_t[jj]
        pi, pj = p_t[ii], p_t[jj]
        bi, bj = box_t[ii], box_t[jj]
        geom_ij = box_geometry(bi, bj)
        geom_ji = box_geometry(bj, bi)
        phi_ij = build_pairwise_features(zi, zj, pi, pj, bi, bj, geom=geom_ij)
        phi_ji = build_pairwise_features(zj, zi, pj, pi, bj, bi, geom=geom_ji)
        with torch.no_grad():
            out = relation_head(torch.cat([phi_ij.to(device), phi_ji.to(device)], dim=0))
            P = len(ii)
            inst_logits = 0.5 * (out["inst"][:P] + out["inst"][P:]).cpu().numpy()
        for k in range(P):
            i, j = ((keep[ii[k]], keep[jj[k]]) if n_cand > max_cand
                    else (ii[k], jj[k]))
            A_inst[i, j] = A_inst[j, i] = inst_logits[k]

    valid_indices = [i for i in range(n_cand) if effective_valid[i]]
    if len(valid_indices) <= 1:
        empty["pred_count"] = len(valid_indices)
        empty["n_groups"] = len(valid_indices)
        empty["group_infos"] = [
            {"rep_indices": [gi], "class_id": int(top_class[gi]), "count": 1}
            for gi in valid_indices]
        return empty

    sub_probs = category_probs[valid_indices]
    sub_A_sem = A_sem[valid_indices][:, valid_indices]
    image_area = float(h * w)

    groups = category_aware_clustering_with_spatial(
        sub_probs, sub_A_sem, bbox_np[valid_indices], image_area,
        tau_affinity=tau_affinity, max_group_size=max_group_size,
        use_bucketing=True,
    )

    total_reps = 0
    group_infos = []
    for group in groups:
        group_global = [valid_indices[i] for i in group]
        if len(group_global) == 0:
            continue
        if len(group_global) == 1:
            reps = list(group_global)
        else:
            sub_A_inst = A_inst[group_global][:, group_global]
            if len(group_global) > 20:
                components = build_same_instance_components_adaptive(
                    list(range(len(group_global))), sub_A_inst,
                    base_tau=tau_inst, use_greedy=True)
            else:
                components = build_same_instance_components(
                    list(range(len(group_global))), sub_A_inst, tau_inst=tau_inst)
            reps = select_representatives(
                [[group_global[i] for i in comp] for comp in components],
                np.zeros((n_cand, n_cand)),
                category_probs, bbox_np, image_area, min_category_conf=0.05)
        total_reps += len(reps)
        reps = [int(r) for r in reps]
        cls_ids, cls_cnt = np.unique(top_class[reps], return_counts=True)
        group_infos.append({
            "rep_indices": reps,
            "class_id": int(cls_ids[np.argmax(cls_cnt)]),
            "count": len(reps),
        })

    return {"pred_count": total_reps, "n_candidates": n_cand,
            "n_groups": len(groups), "gt_count": gt_count,
            "file_name": file_name, "group_infos": group_infos}


# ---------------------------------------------------------------------------
def run_scan(args, device):
    category_head, relation_head, tp, _ = load_models(device)
    cache_files = sorted(Path(args.cache_dir).glob("*.pt"))
    print(f"[scan] {len(cache_files)} cache files")
    results = []
    import time
    t0 = time.time()
    for i, cf in enumerate(cache_files):
        d = torch.load(cf, map_location="cpu", weights_only=False)
        r = count_image_viz(d, category_head, relation_head, tp, device)
        results.append({k: r[k] for k in
                        ("file_name", "pred_count", "gt_count", "n_groups", "n_candidates")})
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(cache_files)}] {(i+1)/(time.time()-t0):.1f}/s", flush=True)
    OUT_DIR.mkdir(exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=1)
    print(f"[scan] saved -> {args.out}")


# ---------------------------------------------------------------------------
def render_one(cache_path, img_dir, category_head, relation_head, tp,
               categories, device):
    from pycocotools import mask as mask_utils
    from PIL import Image

    d = torch.load(cache_path, map_location="cpu", weights_only=False)
    r = count_image_viz(d, category_head, relation_head, tp, device)
    h, w = int(d["height"]), int(d["width"])

    img_path = img_dir / r["file_name"]
    img = Image.open(img_path).convert("RGB")
    if img.size != (w, h):
        img = img.resize((w, h), Image.BILINEAR)
    canvas = np.asarray(img).astype(np.float32)

    def boundary_of(m):
        er = m.copy()
        for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
            er &= np.roll(m, sh, axis=ax)
        return m & ~er

    groups_sorted = sorted(r["group_infos"], key=lambda g: -g["count"])
    for gidx, g in enumerate(groups_sorted):
        color = np.array(COLORS[gidx % len(COLORS)], dtype=np.float32)
        union = np.zeros((h, w), dtype=bool)
        edges = np.zeros((h, w), dtype=bool)
        for ridx in g["rep_indices"]:
            m = mask_utils.decode(d["masks_rle"][ridx]).astype(bool)
            if m.shape != (h, w):
                m = np.array(Image.fromarray(m.astype(np.uint8) * 255)
                             .resize((w, h), Image.NEAREST)) > 127
            union |= m
            edges |= boundary_of(m)
        canvas[union] = canvas[union] * (1 - ALPHA) + color * ALPHA
        # per-instance dark outline (same hue) so touching instances stay separable
        canvas[edges] = color * 0.45

    canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    parts = [f"{categories[g['class_id']]} × {g['count']}" for g in groups_sorted[:4]]
    if len(groups_sorted) > 4:
        parts.append("…")
    line1 = " | ".join(parts)
    tick = " ✓" if r["pred_count"] == r["gt_count"] else ""
    line2 = f"pred {r['pred_count']} / GT {r['gt_count']}{tick}"
    return canvas, line1, line2, r


def run_render(args, device):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    category_head, relation_head, tp, categories = load_models(device)
    OUT_DIR.mkdir(exist_ok=True)

    selections = json.load(open(args.select))  # list of {dataset, name}
    panels = []
    report = []
    for sel in selections:
        if sel["dataset"] == "carpk":
            cache, imgd = CACHE_CARPK, IMG_CARPK
        else:
            cache, imgd = CACHE_FSC, IMG_FSC
        stem = Path(sel["name"]).stem
        cpath = cache / f"{stem}.pt"
        canvas, line1, line2, r = render_one(
            cpath, imgd, category_head, relation_head, tp, categories, device)
        panels.append((canvas, line1, line2))
        report.append({"file_name": r["file_name"], "pred": r["pred_count"],
                       "gt": r["gt_count"],
                       "groups": [{"class": categories[g["class_id"]],
                                   "count": g["count"]}
                                  for g in sorted(r["group_infos"],
                                                  key=lambda g: -g["count"])]})
        # single-image figure
        fig, ax = plt.subplots(figsize=(canvas.shape[1] / 100, canvas.shape[0] / 100))
        ax.imshow(canvas); ax.axis("off")
        ax.set_title(f"{line1}\n{line2}", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"grid_{stem}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[render] grid_{stem}.png  {line2}  [{line1}]", flush=True)

    # combined 2-column grid
    ncols = 2
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.4 * nrows))
    axes = np.atleast_2d(axes)
    for k in range(nrows * ncols):
        ax = axes[k // ncols, k % ncols]
        ax.axis("off")
        if k < len(panels):
            canvas, line1, line2 = panels[k]
            ax.imshow(canvas)
            ax.set_title(f"{line1}\n{line2}", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "grid_qual.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / "grid_qual.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    json.dump(report, open(OUT_DIR / "grid_qual_report.json", "w"), indent=2)
    print("[render] grid_qual.png / grid_qual.pdf saved")
    print(json.dumps(report, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["scan", "render"])
    ap.add_argument("--cache-dir", default=str(CACHE_FSC))
    ap.add_argument("--out", default=str(OUT_DIR / "fsc147_pts32_scan.json"))
    ap.add_argument("--select", default=str(OUT_DIR / "selection.json"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.mode == "scan":
        run_scan(args, args.device)
    else:
        run_render(args, args.device)


if __name__ == "__main__":
    main()
