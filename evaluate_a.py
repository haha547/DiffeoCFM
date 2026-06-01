"""
evaluate_a.py  —  Direction A
------------------------------
Evaluate ASD/TD classification using generated EC or CPT covariances.

Uses results from train_custom.py (model conditioned on EC/CPT).
Maps subject groups → ASD/TD labels via GroupInfo.mat, then runs
TSTR / baseline / TRTS for ASD vs TD downstream classification.

Key question: does generated EC or CPT data help more for ASD/TD classification?

Usage:
    python evaluate_a.py --data "./cov_2s_0ov" --region p
    python evaluate_a.py --data "./cov_2s_0ov" "./cov_4s_0ov" --region s
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from joblib import Parallel, delayed
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline

PATH_RESULTS = Path("results")
PATH_FIGURES = Path("figures")

# =============================================================================
# Args
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--data", nargs="+", required=True,
                    help="Same folder(s) as train_custom.py --data")
parser.add_argument("--region", type=str, default="p", choices=["p", "s"])
parser.add_argument("--jobs", type=int, default=4)
parser.add_argument("--groupinfo", type=str, default="GroupInfo.mat")
args = parser.parse_args()

REGION     = args.region
DATASETS   = [f"{Path(d).name}_{REGION}" for d in args.data]
N_JOBS     = args.jobs
REGION_ROW = 0 if REGION == "p" else 1

# =============================================================================
# Load subject-level ASD/TD labels
# =============================================================================
g_info = scipy.io.loadmat(args.groupinfo)
# condiction: (2, 43)  row0=p, row1=s  |  0=TD, 1=ASD
subject_diagnosis = g_info["GroupInfo"][0, 0]["condiction"][REGION_ROW, :]  # (43,)
print(f"Region '{REGION}': "
      f"{int(np.sum(subject_diagnosis == 1))} ASD, "
      f"{int(np.sum(subject_diagnosis == 0))} TD subjects loaded.")


# =============================================================================
# Helpers
# =============================================================================
def project_on_SPD(matrices, eps=1e-8):
    orig = matrices.shape
    flat = matrices.reshape(-1, orig[-2], orig[-1])
    min_eigs = np.linalg.eigvalsh(flat).min(axis=1)
    bad = min_eigs < eps
    a = np.zeros_like(min_eigs)
    a[bad] = (eps - min_eigs[bad]) / (1 - min_eigs[bad])
    eye = np.eye(orig[-1])[None]
    out = (1 - a)[:, None, None] * flat + a[:, None, None] * eye
    return out.reshape(orig)


def is_all_spd(matrices, tol=1e-12):
    sym = all(np.allclose(m, m.T, atol=tol) for m in matrices)
    pd  = all(np.all(np.linalg.eigvalsh(m) > tol) for m in matrices)
    return sym and pd


def clf_metrics(X_train, y_train, X_test, y_test):
    """Returns dict of metrics, or None if not enough class diversity."""
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None
    clf = LogisticRegressionCV(
        cv=min(5, int(np.min(np.bincount(y_train)))),
        solver="liblinear",
        class_weight="balanced",
        random_state=42,
        max_iter=5000,
    )
    pipe = make_pipeline(TangentSpace(metric="riemann"), clf)
    try:
        pipe.fit(X_train, y_train)
        y_pred  = pipe.predict(X_test)
        y_score = pipe.predict_proba(X_test)[:, 1]
        return {
            "ROC-AUC":   roc_auc_score(y_test, y_score),
            "F1":        f1_score(y_test, y_pred, zero_division=0),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall":    recall_score(y_test, y_pred, zero_division=0),
        }
    except Exception as e:
        print(f"    clf failed: {e}")
        return None


# =============================================================================
# Per-split evaluation
# =============================================================================
def evaluate_split(dataset, group, method, split, path_method):
    def load(name):
        return np.load(path_method / f"split_{split}_{name}.npy", allow_pickle=False)

    cov_train  = load("covariances_train")          # (N, 8, 8)
    ec_cpt_tr  = load("conditionals_train")         # (N,) 0=EC, 1=CPT
    groups_tr  = load("groups_train")               # (N,) subject index 0-based

    cov_val    = load("covariances_val")
    ec_cpt_va  = load("conditionals_val")
    groups_va  = load("groups_val")

    gen_train  = load("covariances_generated_samples_train")  # (T, N, 8, 8)
    gen_val    = load("covariances_generated_samples_val")

    train_time = float(load("training_time").flat[0])
    samp_time  = float(load("sampling_time").flat[0])

    # Map subject index → ASD/TD
    diag_tr = subject_diagnosis[groups_tr]   # (N,) 0=TD 1=ASD
    diag_va = subject_diagnosis[groups_va]

    # Final ODE step
    gen_tr_last = gen_train[-1]
    gen_va_last = gen_val[-1]

    if not is_all_spd(gen_tr_last):
        gen_tr_last = project_on_SPD(gen_tr_last)
        gen_va_last = project_on_SPD(gen_va_last)

    rows = []
    base = {
        "Dataset": dataset, "Group": group, "Method": method, "Split": split,
        "Train time (s)": train_time, "Sampling time (s)": samp_time,
    }

    # Evaluate for each condition subset and "All"
    for cond_name, cond_val in [("EC", 0), ("CPT", 1), ("All", None)]:
        if cond_val is not None:
            tr_m = ec_cpt_tr == cond_val
            va_m = ec_cpt_va == cond_val
        else:
            tr_m = np.ones(len(cov_train), dtype=bool)
            va_m = np.ones(len(cov_val),   dtype=bool)

        X_real_tr = cov_train[tr_m];   y_tr = diag_tr[tr_m]
        X_gen_tr  = gen_tr_last[tr_m]
        X_real_va = cov_val[va_m];     y_va = diag_va[va_m]
        X_gen_va  = gen_va_last[va_m]

        for comparison, X_tr, y_train_c, X_va, y_val_c in [
            ("Real→Val (baseline)", X_real_tr, y_tr, X_real_va, y_va),
            ("Gen→Val (TSTR)",      X_gen_tr,  y_tr, X_real_va, y_va),
            ("Real→Gen (TRTS)",     X_real_tr, y_tr, X_gen_va,  y_va),
        ]:
            m = clf_metrics(X_tr, y_train_c, X_va, y_val_c)
            if m is not None:
                rows.append({**base, "Condition": cond_name,
                             "Comparison": comparison, **m})

    print(f"  [{dataset}/{method}] split {split} done  "
          f"(ASD in val: {int(np.sum(diag_va == 1))})")
    return rows


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
        for group_dir in sorted(dataset_dir.iterdir()):
            for method_dir in sorted(group_dir.iterdir()):
                split_ids = set()
                for f in method_dir.glob("split_*"):
                    m = re.match(r"split_(\d+)_", f.name)
                    if m:
                        split_ids.add(int(m.group(1)))
                for split in sorted(split_ids):
                    tasks.append((dataset, group_dir.name, method_dir.name,
                                  split, method_dir))

    print(f"Found {len(tasks)} split(s) to evaluate across: {DATASETS}")

    results = Parallel(n_jobs=N_JOBS)(
        delayed(evaluate_split)(ds, grp, mth, sp, path)
        for ds, grp, mth, sp, path in tasks
    )

    all_rows = [r for rows in results for r in rows]
    if not all_rows:
        print("No results collected — check that training has been run.")
        raise SystemExit(1)

    df = pd.DataFrame(all_rows)
    out_csv = PATH_FIGURES / "asd_classification_a.csv"
    df.to_csv(out_csv, index=False, float_format="%.3f")
    print(f"\nSaved → {out_csv}")

    tstr = df[df["Comparison"] == "Gen→Val (TSTR)"]
    if not tstr.empty:
        summary = (tstr.groupby(["Dataset", "Method", "Condition"])["F1"]
                   .agg(["mean", "std"]).round(3))
        print("\nF1 summary — TSTR (train on generated, test on real ASD/TD):")
        print(summary.to_string())
