"""
plot_asd.py
-----------
Plot ASD/TD classification results for Direction A and Direction B.

Reads:
    figures/asd_classification_a.csv  (from evaluate_a.py)
    figures/asd_classification_b.csv  (from evaluate_b.py)

Produces one figure per dataset (e.g. cov_2s_0ov_s), saved to figures/.

Layout per figure:
    rows = Condition  (EC | CPT | All)
    cols = Direction  (A  | B)
    each subplot = grouped bars (Method × Comparison), metric = F1 or ROC-AUC

Usage:
    python plot_asd.py
    python plot_asd.py --metric roc_auc
    python plot_asd.py --metric f1 --no-trts
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

PATH_FIGURES = Path("figures")

# ── label mappings ─────────────────────────────────────────────────────────────

METHOD_NAMES = {
    "logeuclidean_DiffeoGauss":           "DiffeoGauss\n(LogEuc)",
    "lower_triangular_DiffeoCFM":         "DiffeoCFM\n(Triang)",
    "lower_triangular_DiffeoCFM_projected":"DiffeoCFM\n(Triang†)",
    "logeuclidean_DiffeoCFM":             "DiffeoCFM\n(LogEuc)",
    "logeuclidean_DiffeoCFM_projected":   "DiffeoCFM\n(LogEuc†)",
}

COMPARISON_LABELS = {
    "Baseline (Real→Val)": "Baseline\n(Real→Val)",
    "TSTR (Gen→Val)":      "TSTR\n(Gen→Val)",
}

CONDITION_ORDER = ["EC", "CPT", "All"]
DIRECTION_ORDER = ["A", "B"]

COMPARISON_COLORS = {
    "Baseline (Real→Val)": "#4C72B0",   # blue
    "TSTR (Gen→Val)":      "#DD8452",   # orange
}

METRIC_COLS = {
    "f1":      "F1",
    "roc_auc": "ROC-AUC",
}


# ── helpers ────────────────────────────────────────────────────────────────────

def load_data(show_trts: bool) -> pd.DataFrame:
    """Load Direction A/B CSVs for all regions (s, p, inter_gram)."""
    frames = []

    # Direction A: one CSV per region
    for region in ("s", "p", "inter_gram"):
        path = PATH_FIGURES / f"asd_classification_a_{region}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["Direction"] = "A"
        frames.append(df)
        print(f"  Loaded A/{region}: {len(df)} rows")

    # Direction B: one CSV per region
    for region in ("s", "p", "inter_gram"):
        path = PATH_FIGURES / f"asd_classification_b_{region}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["Direction"] = "B"
        frames.append(df)
        print(f"  Loaded B/{region}: {len(df)} rows")

    if not frames:
        raise FileNotFoundError(
            "No classification CSVs found.\n"
            "Expected: figures/asd_classification_a_s.csv and/or asd_classification_b_s.csv\n"
            "Run evaluate_a.py and/or evaluate_b.py first."
        )

    df = pd.concat(frames, ignore_index=True)

    # keep only wanted comparisons
    wanted = ["Baseline (Real→Val)", "TSTR (Gen→Val)"]
    df = df[df["Comparison"].isin(wanted)].copy()

    # friendly method names
    df["Method"] = df["Method"].map(METHOD_NAMES).fillna(df["Method"])

    return df


def get_datasets(df: pd.DataFrame) -> list[str]:
    return sorted(df["Dataset"].unique())


def dataset_title(ds: str) -> str:
    """Turn cov_2s_0ov_s → 'Cov 2s 0OV [S]', cov_2s_0ov_inter_gram → '... [Inter-brain]'"""
    if ds.endswith("_inter_gram"):
        base = ds[:-len("_inter_gram")].replace("cov_", "").replace("ov", "OV").replace("_", " ")
        return f"Dataset: {base}  [Inter-brain Gram]"
    parts = ds.replace("cov_", "").replace("ov", "OV").split("_")
    region = parts[-1].upper() if parts[-1] in ("S", "P", "s", "p") else ""
    rest   = " ".join(parts[:-1]) if region else " ".join(parts)
    return f"Dataset: {rest}{f'  [{region} region]' if region else ''}"


# ── plotting ───────────────────────────────────────────────────────────────────

def plot_dataset(df_ds: pd.DataFrame, dataset: str, metric_col: str,
                 show_trts: bool, out_dir: Path):
    """One figure per dataset."""

    directions    = [d for d in DIRECTION_ORDER if d in df_ds["Direction"].unique()]
    conditions    = [c for c in CONDITION_ORDER if c in df_ds["Condition"].unique()]
    comparisons   = [c for c in COMPARISON_LABELS if c in df_ds["Comparison"].unique()]

    n_rows = len(conditions)
    n_cols = len(directions)

    mpl.rcParams.update({
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 3.5 * n_rows),
        sharey=True,
        squeeze=False,
    )
    fig.suptitle(dataset_title(dataset), fontsize=11, fontweight="bold", y=1.01)

    for col_i, direction in enumerate(directions):
        df_dir = df_ds[df_ds["Direction"] == direction]

        methods = [m for m in df_dir["Method"].unique()]
        n_methods     = len(methods)
        n_comparisons = len(comparisons)

        bar_width   = 0.7 / n_comparisons
        method_x    = np.arange(n_methods)

        for row_i, condition in enumerate(conditions):
            ax = axes[row_i][col_i]

            df_cell = df_dir[df_dir["Condition"] == condition]

            # aggregate over splits: mean ± std
            agg = (df_cell
                   .groupby(["Method", "Comparison"])[metric_col]
                   .agg(["mean", "std"])
                   .reset_index())

            for cmp_i, cmp in enumerate(comparisons):
                offset = (cmp_i - (n_comparisons - 1) / 2) * bar_width
                agg_cmp = agg[agg["Comparison"] == cmp]
                color   = COMPARISON_COLORS.get(cmp, f"C{cmp_i}")

                for m_i, method in enumerate(methods):
                    row_m = agg_cmp[agg_cmp["Method"] == method]
                    if row_m.empty:
                        continue
                    mean_val = row_m["mean"].iloc[0]
                    std_val  = row_m["std"].iloc[0]
                    label    = COMPARISON_LABELS.get(cmp, cmp) if (row_i == 0 and m_i == 0) else None

                    ax.bar(
                        m_i + offset, mean_val,
                        width=bar_width * 0.9,
                        color=color, alpha=0.85,
                        label=label,
                        zorder=2,
                    )
                    ax.errorbar(
                        m_i + offset, mean_val,
                        yerr=std_val,
                        fmt="none", color="black",
                        capsize=3, linewidth=1, zorder=3,
                    )

            # axes labels
            if row_i == 0:
                ax.set_title(f"Direction {direction}", fontweight="bold", pad=6)
            if col_i == 0:
                ax.set_ylabel(f"{condition}\n{metric_col}", labelpad=4)
            ax.set_xticks(method_x)
            ax.set_xticklabels(methods, rotation=15, ha="right")
            ax.set_ylim(0, 1.05)
            ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
            ax.set_axisbelow(True)

            # chance level line
            ax.axhline(0.5, color="red", linestyle=":", linewidth=1.0,
                       alpha=0.6, zorder=1, label="Chance" if (row_i == 0 and col_i == 0) else None)

    # shared legend from first subplot
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="upper center", ncol=len(comparisons) + 1,
               bbox_to_anchor=(0.5, 1.06), frameon=False, fontsize=8)

    fig.tight_layout()

    out_name = f"asd_clf_{dataset}_{metric_col.lower().replace('-','_')}.pdf"
    out_path = out_dir / out_name
    plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"  Saved → {out_path}")
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="f1", choices=["f1", "roc_auc"],
                        help="Metric to plot (default: f1)")
    args = parser.parse_args()

    metric_col = METRIC_COLS[args.metric]

    PATH_FIGURES.mkdir(exist_ok=True)

    print(f"Loading classification CSVs  [metric={metric_col}]")
    df = load_data(show_trts=False)

    datasets = get_datasets(df)
    print(f"Found {len(datasets)} dataset(s): {datasets}")

    for dataset in datasets:
        print(f"\nPlotting dataset: {dataset}")
        df_ds = df[df["Dataset"] == dataset]
        plot_dataset(df_ds, dataset, metric_col, show_trts, PATH_FIGURES)

    print("\nDone.")
