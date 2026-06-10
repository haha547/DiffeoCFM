"""
plot_aug.py
-----------
Plot TSTR classification metrics vs. augmentation factor.

Reads:  figures/asd_classification_b.csv  (from evaluate_b.py --aug 1 2 3 5 10)

For each dataset, produces one figure showing how ROC-AUC and F1 change
as more generated samples are added (AugFactor = 1× to N×).

Layout per figure:
    rows = Condition (EC | CPT | All)
    cols = Metric    (ROC-AUC | F1)
    lines = Method

Baseline (Real→Val, AugFactor=0) is drawn as a horizontal dashed line.

Usage:
    python plot_aug.py
    python plot_aug.py --csv figures/asd_classification_b.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

PATH_FIGURES = Path("figures")

METHOD_NAMES = {
    "logeuclidean_DiffeoGauss":            "DiffeoGauss (LogEuc)",
    "lower_triangular_DiffeoCFM":          "DiffeoCFM (Triang)",
    "lower_triangular_DiffeoCFM_projected":"DiffeoCFM (Triang†)",
    "logeuclidean_DiffeoCFM":              "DiffeoCFM (LogEuc)",
    "logeuclidean_DiffeoCFM_projected":    "DiffeoCFM (LogEuc†)",
}

CONDITION_ORDER = ["EC", "CPT", "All"]
METRICS         = ["ROC-AUC", "F1"]
MARKERS         = ["o", "s", "^", "D", "*"]


def dataset_title(ds: str) -> str:
    parts = ds.replace("cov_", "").replace("ov", "OV").split("_")
    region = parts[-1].upper() if parts[-1] in ("s", "p") else ""
    rest   = " ".join(parts[:-1]) if region else " ".join(parts)
    return f"{rest}  [{region} region]"


def plot_dataset(df_ds: pd.DataFrame, dataset: str, out_dir: Path):
    mpl.rcParams.update({
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    conditions = [c for c in CONDITION_ORDER if c in df_ds["Condition"].unique()]
    methods    = sorted(df_ds["Method"].unique())
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    method_style = {
        m: {"color": prop_cycle[i % len(prop_cycle)],
            "marker": MARKERS[i % len(MARKERS)]}
        for i, m in enumerate(methods)
    }

    n_rows = len(conditions)
    n_cols = len(METRICS)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 3.2 * n_rows),
                             sharey="col", squeeze=False)
    fig.suptitle(f"TSTR vs Augmentation Factor — {dataset_title(dataset)}",
                 fontsize=11, fontweight="bold", y=1.01)

    # Separate TSTR rows from baseline (AugFactor=0)
    df_tstr     = df_ds[df_ds["Comparison"] == "TSTR (Gen→Val)"]
    df_baseline = df_ds[df_ds["Comparison"] == "Baseline (Real→Val)"]

    for row_i, cond in enumerate(conditions):
        df_c      = df_tstr[df_tstr["Condition"] == cond]
        df_base_c = df_baseline[df_baseline["Condition"] == cond]

        aug_vals = sorted(df_c["AugFactor"].unique())

        for col_i, metric in enumerate(METRICS):
            ax = axes[row_i][col_i]

            # Baseline horizontal line
            if not df_base_c.empty and metric in df_base_c.columns:
                bval = df_base_c[metric].mean()
                ax.axhline(bval, color="gray", linestyle="--", linewidth=1.5,
                           label="Baseline (Real)" if col_i == 0 else None,
                           zorder=0)

            # TSTR lines per method
            for method in methods:
                style = method_style[method]
                df_m  = df_c[df_c["Method"] == method]
                if df_m.empty:
                    continue

                xs, ys = [], []
                for aug in aug_vals:
                    row = df_m[df_m["AugFactor"] == aug]
                    if row.empty or metric not in row.columns:
                        continue
                    xs.append(aug)
                    ys.append(row[metric].iloc[0])

                if xs:
                    label = method if row_i == 0 and col_i == 0 else None
                    ax.plot(xs, ys,
                            color=style["color"], marker=style["marker"],
                            linewidth=1.5, markersize=6,
                            label=label, zorder=2)

            # Axes formatting
            if row_i == 0:
                ax.set_title(metric, fontweight="bold")
            if col_i == 0:
                ax.set_ylabel(cond, fontweight="bold", labelpad=4)
            ax.set_xlabel("Augmentation factor (×real)" if row_i == n_rows - 1 else "")
            ax.set_xticks(aug_vals if aug_vals else [1])
            ax.set_ylim(0, 1.05)
            ax.axhline(0.5, color="red", linestyle=":", linewidth=0.8,
                       alpha=0.5, zorder=1)
            ax.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, zorder=0)
            ax.set_axisbelow(True)

    # Legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels,
                   loc="upper center", ncol=min(4, len(handles)),
                   bbox_to_anchor=(0.5, 1.06), frameon=False, fontsize=8)

    fig.tight_layout()
    stem = f"aug_sweep_{dataset}"
    plt.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.savefig(out_dir / f"{stem}.png", bbox_inches="tight", dpi=150)
    print(f"  Saved → {out_dir / stem}.pdf/png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="figures/asd_classification_b.csv",
                        help="CSV from evaluate_b.py (default: figures/asd_classification_b.csv)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Run evaluate_b.py first.")

    df = pd.read_csv(csv_path)

    # Friendly method names
    df["Method"] = df["Method"].map(METHOD_NAMES).fillna(df["Method"])

    if "AugFactor" not in df.columns:
        raise ValueError("CSV has no 'AugFactor' column — re-run evaluate_b.py with --aug flag.")

    PATH_FIGURES.mkdir(exist_ok=True)
    datasets = sorted(df["Dataset"].unique())
    print(f"Found {len(datasets)} dataset(s): {datasets}")

    for dataset in datasets:
        print(f"\nPlotting: {dataset}")
        plot_dataset(df[df["Dataset"] == dataset], dataset, PATH_FIGURES)

    print("\nDone.")
