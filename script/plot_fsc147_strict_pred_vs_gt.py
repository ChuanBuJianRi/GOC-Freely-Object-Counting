"""Plot FSC-147 strict no-GT predictions against ground-truth counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, FixedLocator  # noqa: E402


REPO = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = REPO / "result/logs/fsc147_strict_nogt_cp_free_full1190.json"
DEFAULT_OUTPUT = REPO / "docs/figures/fsc147_strict_nogt_pred_vs_gt_seed73"
PRIMARY_SEED = "73"
ZOOM_MAX = 250

GT_BINS = (
    ("0-10", 0, 10, "#0072B2"),
    ("11-20", 11, 20, "#009E73"),
    ("21-50", 21, 50, "#56B4E9"),
    ("51-100", 51, 100, "#E69F00"),
    ("100+", 101, np.inf, "#D55E00"),
)


def comma_tick(value: float, _position: int) -> str:
    return f"{int(value):,}"


def load_primary_rows(path: Path) -> tuple[list[dict], dict]:
    payload = json.loads(path.read_text())
    if str(payload.get("primary_seed")) != PRIMARY_SEED:
        raise RuntimeError(f"expected primary seed {PRIMARY_SEED}: {path}")
    rows = payload.get("runs", {}).get(PRIMARY_SEED, {}).get("rows", [])
    metrics = payload.get("runs", {}).get(PRIMARY_SEED, {}).get("metrics", {})
    if len(rows) != 1190 or int(metrics.get("n", -1)) != len(rows):
        raise RuntimeError(f"expected all 1,190 FSC-147 test rows: {path}")

    gt = np.asarray([row["gt_count"] for row in rows], dtype=np.float64)
    pred = np.asarray([row["pred_count"] for row in rows], dtype=np.float64)
    recomputed = {
        "MAE": float(np.abs(pred - gt).mean()),
        "RMSE": float(np.sqrt(np.square(pred - gt).mean())),
        "bias": float((pred - gt).mean()),
    }
    for key, value in recomputed.items():
        if not np.isclose(value, float(metrics[key]), atol=1e-12, rtol=0):
            raise RuntimeError(f"recorded {key} does not match rows: {path}")
    return rows, metrics


def scatter_by_gt_bin(
    axis: plt.Axes,
    gt: np.ndarray,
    pred: np.ndarray,
    selection: np.ndarray,
    *,
    point_size: float,
    alpha: float,
) -> None:
    for label, low, high, color in GT_BINS:
        selected = selection & (gt >= low) & (gt <= high)
        axis.scatter(
            gt[selected],
            pred[selected],
            s=point_size,
            c=color,
            alpha=alpha,
            edgecolors="none",
            label=f"GT {label}",
            rasterized=True,
        )


def style_axis(axis: plt.Axes) -> None:
    axis.set_axisbelow(True)
    axis.grid(True, which="major", color="#D8DDE3", linewidth=0.7)
    axis.grid(True, which="minor", color="#EEF1F4", linewidth=0.45)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#6B7280")
    axis.spines["bottom"].set_color("#6B7280")
    axis.tick_params(colors="#374151", labelsize=9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    rows, metrics = load_primary_rows(args.result)
    gt = np.asarray([row["gt_count"] for row in rows], dtype=np.float64)
    pred = np.asarray([row["pred_count"] for row in rows], dtype=np.float64)
    correlation = float(np.corrcoef(gt, pred)[0, 1])

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelcolor": "#111827",
        "axes.titlecolor": "#111827",
        "text.color": "#111827",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.75))
    figure.suptitle(
        "FSC-147 strict no-GT: predicted count vs. ground truth",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.93,
        (
            f"Primary relation seed 73 | N={len(rows):,} | "
            f"MAE={metrics['MAE']:.2f} | RMSE={metrics['RMSE']:.2f} | "
            f"bias={metrics['bias']:+.2f} | Pearson r={correlation:.3f}"
        ),
        ha="center",
        fontsize=9.5,
        color="#374151",
    )

    full = np.ones(len(rows), dtype=bool)
    left = axes[0]
    scatter_by_gt_bin(left, gt, pred, full, point_size=18, alpha=0.62)
    full_limit = 5000
    identity = np.geomspace(1, full_limit, 500)
    left.plot(identity, identity, color="#20242A", linewidth=1.25, linestyle="--")
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlim(1, full_limit)
    left.set_ylim(1, full_limit)
    ticks = [1, 10, 100, 1000]
    left.xaxis.set_major_locator(FixedLocator(ticks))
    left.yaxis.set_major_locator(FixedLocator(ticks))
    left.xaxis.set_major_formatter(FuncFormatter(comma_tick))
    left.yaxis.set_major_formatter(FuncFormatter(comma_tick))
    left.set_xlabel("Ground-truth count")
    left.set_ylabel("Predicted count")
    left.set_title("All test images (log-log)", fontsize=11, pad=8)
    style_axis(left)

    row_by_name = {row["file_name"]: row for row in rows}
    callouts = {
        "1123.jpg": (-70, 12),
        "7611.jpg": (-58, 20),
    }
    for file_name, offset in callouts.items():
        row = row_by_name[file_name]
        left.scatter(
            [row["gt_count"]],
            [row["pred_count"]],
            s=52,
            facecolors="none",
            edgecolors="#111827",
            linewidths=1.1,
            zorder=5,
        )
        left.annotate(
            f"{Path(file_name).stem}: ({row['gt_count']:,}, {row['pred_count']:,})",
            xy=(row["gt_count"], row["pred_count"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            color="#111827",
            arrowprops={"arrowstyle": "-", "color": "#4B5563", "lw": 0.8},
        )

    right = axes[1]
    central = (gt <= ZOOM_MAX) & (pred <= ZOOM_MAX)
    scatter_by_gt_bin(right, gt, pred, central, point_size=18, alpha=0.58)
    right.plot(
        [0, ZOOM_MAX],
        [0, ZOOM_MAX],
        color="#20242A",
        linewidth=1.25,
        linestyle="--",
        label="Ideal: prediction = GT",
    )
    right.set_xlim(0, ZOOM_MAX)
    right.set_ylim(0, ZOOM_MAX)
    right.set_aspect("equal", adjustable="box")
    right.set_xlabel("Ground-truth count")
    right.set_ylabel("Predicted count")
    right.set_title(
        f"Central range 0-{ZOOM_MAX} (n={int(central.sum()):,})",
        fontsize=11,
        pad=8,
    )
    style_axis(right)

    handles, labels = left.get_legend_handles_labels()
    ideal_handle = right.get_legend_handles_labels()[0][-1]
    figure.legend(
        handles + [ideal_handle],
        labels + ["Ideal: prediction = GT"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=6,
        frameon=False,
        fontsize=8.5,
        handletextpad=0.45,
        columnspacing=1.0,
    )
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.17, top=0.855, wspace=0.22)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = args.output_prefix.with_suffix(".png")
    pdf_path = args.output_prefix.with_suffix(".pdf")
    figure.savefig(png_path, dpi=args.dpi, facecolor="white")
    figure.savefig(pdf_path, facecolor="white")
    plt.close(figure)
    print(json.dumps({
        "input": str(args.result),
        "primary_seed": int(PRIMARY_SEED),
        "n": len(rows),
        "central_n": int(central.sum()),
        "pearson_r": correlation,
        "png": str(png_path),
        "pdf": str(pdf_path),
    }, indent=2))


if __name__ == "__main__":
    main()
