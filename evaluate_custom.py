"""
evaluate_custom.py
------------------
Evaluate results produced by train_custom.py.

Usage:
    python evaluate_custom.py --data "D:/東元/I.RMT-R/cov_2s_0ov"
    python evaluate_custom.py --data "D:/東元/I.RMT-R/cov_2s_0ov" "D:/東元/I.RMT-R/cov_4s_0ov"
    python evaluate_custom.py --data "D:/東元/I.RMT-R/cov_2s_0ov" --region s
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline

from prdc import compute_prdc

PATH_RESULTS = Path("results")
PATH_FIGURES = Path("figures")

# =============================================================================
# Args
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--data", nargs="+", required=True,
                    help="Folder(s) containing covariance .npy files (same as train_custom.py --data)")
parser.add_argument("--region", type=str, default="s", choices=["p", "s"],
                    help="Which channel group: p=前8ch, s=後8ch (default: s)")
parser.add_argument("--jobs", type=int, default=4)
args = parser.parse_args()

# Derive dataset names the same way train_custom.py does:
#   dataset_name = f"{DATA_DIR.name}_{REGION}"
REGION   = args.region
DATASETS = [f"{Path(d).name}_{REGION}" for d in args.data]
N_JOBS   = args.jobs


# =============================================================================
# Helpers (same logic as evaluate.py)
# =============================================================================

def fraction_covariance_matrix(matrices, tol=1e-12):
    is_sym     = np.array([np.allclose(m, m.T, atol=tol)          for m in matrices])
    is_pd      = np.array([np.all(np.linalg.eigvalsh(m) > tol)    for m in matrices])
    return {"Sym.": np.mean(is_sym), "Pos. def.": np.mean(is_pd), "in M": np.mean(is_sym & is_pd)}


def project_on_SPD(matrices, eps=1e-8):
    orig_shape = matrices.shape
    *_, n, _ = orig_shape
    flat = matrices.reshape(-1, n, n)
    min_eigs = np.linalg.eigvalsh(flat).min(axis=1)
    mask = min_eigs < eps
    alphas = np.zeros_like(min_eigs)
    alphas[mask] = (eps - min_eigs[mask]) / (1 - min_eigs[mask])
    eye = np.eye(n)[None]
    out = (1 - alphas)[:, None, None] * flat + alphas[:, None, None] * eye
    return out.reshape(orig_shape)


def quality_metrics(X_real, X_fake):
    r = X_real.reshape(len(X_real), -1)
    f = X_fake.reshape(len(X_fake), -1)
    n = min(len(r), len(f))
    r, f = r[:n], f[:n]
    prdc = compute_prdc(real_features=r, fake_features=f, nearest_k=10)
    return {
        "Precision": prdc["precision"],
        "Recall":    prdc["recall"],
        "Density":   prdc["density"],
        "Coverage":  prdc["coverage"],
    }


def clf_metrics(X_train, y_train, X_test, y_test):
    clf = LogisticRegressionCV(
        cv=5, solver="liblinear",
        l1_ratios=(0,),
        class_weight="balanced", random_state=42, max_iter=5000,
        use_legacy_attributes=False,
    )
    pipe = make_pipeline(TangentSpace(metric="riemann"), clf)
    pipe.fit(X_train, y_train)
    y_pred  = pipe.predict(X_test)
    y_score = pipe.predict_proba(X_test)[:, 1]
    return {
        "ROC-AUC":   roc_auc_score(y_test, y_score),
        "F1":        f1_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
    }


# =============================================================================
# Per-split evaluation
# =============================================================================

def evaluate_split(dataset, group, method, split, path_method):
    def load(name):
        return np.load(path_method / f"split_{split}_{name}.npy")

    cov_train  = load("covariances_train")
    y_train    = load("conditionals_train")
    cov_val    = load("covariances_val")
    y_val      = load("conditionals_val")
    gen_train  = load("covariances_generated_samples_train")  # (T, N, d, d)
    yg_train   = load("conditionals_generated_samples_train")
    gen_val    = load("covariances_generated_samples_val")
    yg_val     = load("conditionals_generated_samples_val")
    train_time = float(load("training_time").flat[0])
    samp_time  = float(load("sampling_time").flat[0])

    gen_train_last = gen_train[-1]  # final ODE step
    gen_val_last   = gen_val[-1]

    # --- SPD check & projection ---
    frac = fraction_covariance_matrix(gen_train_last)
    in_M = frac["in M"] == 1.0

    if not in_M:
        gen_train_use = project_on_SPD(gen_train_last)
        gen_val_use   = project_on_SPD(gen_val_last)
        method_label  = method + "_projected"
    else:
        gen_train_use = gen_train_last
        gen_val_use   = gen_val_last
        method_label  = method

    base = {"Dataset": dataset, "Group": group, "Method": method, "Split": split}

    # --- Constraint fractions ---
    constraints = [
        {**base, "Subset": "Train", **fraction_covariance_matrix(cov_train)},
        {**base, "Subset": "Val",   **fraction_covariance_matrix(cov_val)},
        {**base, "Subset": "Gen.",  **frac},
    ]

    # --- Quality metrics ---
    quality = [
        {**base, "Comparison": "Train vs Val",  **quality_metrics(cov_train, cov_val)},
        {**base, "Comparison": "Train vs Gen.", **quality_metrics(cov_train, gen_train_use),
         "Train time (s)": train_time, "Sampling time (s)": samp_time},
        {**base, "Comparison": "Val vs Gen.",   **quality_metrics(cov_val,   gen_val_use)},
    ]

    # --- Classification metrics ---
    #   Baseline: train on real, test on real val
    #   TSTR:     train on generated, test on real val  (the key metric)
    #   TRTS:     train on real, test on generated
    base_m = {**base, "Method": method}
    gen_m  = {**base, "Method": method_label}
    clf_rows = [
        {**base_m, "Comparison": "Real→Val (baseline)",
         **clf_metrics(cov_train, y_train, cov_val, y_val),
         "Train time (s)": train_time, "Sampling time (s)": samp_time},
        {**gen_m,  "Comparison": "Gen→Val (TSTR)",
         **clf_metrics(gen_train_use, yg_train, cov_val, y_val),
         "Train time (s)": train_time, "Sampling time (s)": samp_time},
        {**gen_m,  "Comparison": "Real→Gen (TRTS)",
         **clf_metrics(cov_train, y_train, gen_train_use, yg_train),
         "Train time (s)": train_time, "Sampling time (s)": samp_time},
    ]

    print(f"  [{dataset}/{method}] split {split} done  "
          f"F1(TSTR)={clf_rows[1]['F1']:.3f}")
    return constraints, quality, clf_rows


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    PATH_FIGURES.mkdir(exist_ok=True)

    tasks = []
    for dataset in DATASETS:
        dataset_dir = PATH_RESULTS / dataset
        if not dataset_dir.exists():
            print(f"WARNING: {dataset_dir} not found, skipping.")
            continue
        for group_dir in dataset_dir.iterdir():
            for method_dir in group_dir.iterdir():
                split_ids = set()
                for f in method_dir.glob("split_*"):
                    m = re.match(r"split_(\d+)_", f.name)
                    if m:
                        split_ids.add(int(m.group(1)))
                for split in split_ids:
                    tasks.append((dataset, group_dir.name, method_dir.name,
                                  split, method_dir))

    print(f"Found {len(tasks)} split(s) to evaluate across: {DATASETS}")

    results = Parallel(n_jobs=N_JOBS)(
        delayed(evaluate_split)(ds, grp, mth, sp, path)
        for ds, grp, mth, sp, path in tasks
    )

    all_constraints = [r for rows, _, _ in results for r in rows]
    all_quality     = [r for _, rows, _ in results for r in rows]
    all_clf         = [r for _, _, rows in results for r in rows]

    out = PATH_FIGURES
    pd.DataFrame(all_constraints).to_csv(out / "fraction_constraints.csv",  index=False, float_format="%.3f")
    pd.DataFrame(all_quality    ).to_csv(out / "quality_metrics.csv",        index=False, float_format="%.3f")
    pd.DataFrame(all_clf        ).to_csv(out / "classification_metrics.csv", index=False, float_format="%.3f")

    print(f"\nSaved CSVs to {out}/")

    # Quick summary
    df_clf = pd.DataFrame(all_clf)
    tstr = df_clf[df_clf["Comparison"] == "Gen→Val (TSTR)"]
    if not tstr.empty:
        summary = tstr.groupby(["Dataset", "Method"])["F1"].agg(["mean", "std"]).round(3)
        print("\nF1 summary (TSTR — train on generated, test on real):")
        print(summary.to_string())
