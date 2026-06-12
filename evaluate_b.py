"""
evaluate_b.py  —  Direction B  (LOSO aggregation)
--------------------------------------------------
Evaluate ASD/TD classification using 4-class generated covariances.

LOSO structure means each val set contains exactly ONE subject (single class).
Per-split classification metrics are therefore undefined.

Correct approach (implemented here):
  - Each split: train classifier, record P(ASD) score for the left-out subject.
  - After ALL splits: aggregate (y_true, y_score) across subjects → compute metrics.

Label encoding:
    0 = TD-EC  |  1 = TD-CPT  |  2 = ASD-EC  |  3 = ASD-CPT
Decode:
    ASD/TD  : label // 2  →  0=TD, 1=ASD
    EC/CPT  : label % 2   →  0=EC, 1=CPT

Usage:
    python evaluate_b.py --data "./cov_2s_0ov" --region s
    python evaluate_b.py --data "./cov_2s_0ov" "./cov_4s_0ov" --region s
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline

PATH_RESULTS = Path("results_b")
PATH_FIGURES = Path("figures")

# =============================================================================
# Args
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--data", nargs="+", required=True,
                    help="Same folder(s) as train_b.py --data")
parser.add_argument("--region", type=str, default="s", choices=["p", "s", "inter_gram"])
parser.add_argument("--aug",    type=int, nargs="+", default=[1],
                    help="Augmentation factor(s) to evaluate. "
                         "E.g. --aug 1 2 3 5 10. Each value k uses k× generated "
                         "samples per real training sample (must be ≤ --max-aug "
                         "used during training, default 1).")
parser.add_argument("--jobs",   type=int, default=1)
args = parser.parse_args()

REGION      = args.region
DATASETS    = [f"{Path(d).name}_{REGION}" for d in args.data]
AUG_FACTORS = sorted(set(args.aug))
N_JOBS      = args.jobs


# =============================================================================
# Label decoders
# =============================================================================
def to_asd_td(y4):
    return (y4 // 2).astype(np.int64)   # 0=TD, 1=ASD


def to_ec_cpt(y4):
    return (y4 % 2).astype(np.int64)    # 0=EC, 1=CPT


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
    return (all(np.allclose(m, m.T, atol=tol) for m in matrices) and
            all(np.all(np.linalg.eigvalsh(m) > tol) for m in matrices))


def score_subject(X_train, y_train, X_val):
    """
    Train TangentSpace + LR on (X_train, y_train).
    Return mean P(ASD) over all val trials, or None on failure.
    """
    if len(np.unique(y_train)) < 2:
        return None
    clf = LogisticRegression(
        C=1.0,
        solver="liblinear",
        class_weight="balanced",
        random_state=42,
        max_iter=1000,
    )
    pipe = make_pipeline(TangentSpace(metric="riemann"), clf)
    try:
        pipe.fit(X_train, y_train)
        scores = pipe.predict_proba(X_val)[:, 1]   # P(ASD) per trial
        return float(scores.mean())
    except Exception as e:
        print(f"    score_subject failed: {e}")
        return None


# =============================================================================
# Per-split: return per-subject prediction rows (NOT computed metrics)
# =============================================================================
def evaluate_split(dataset, group, method, split, path_method, aug_factors):
    """
    Returns per-subject prediction rows for each requested aug_factor.

    Pool layout (saved by train_b.py):
        conditionals_generated_samples_train = np.repeat(y_train, max_aug)
        → indices [i*max_aug : i*max_aug+max_aug] correspond to original sample i.

    For aug_factor=k, we take the first k out of each group of max_aug,
    giving k× as many generated training samples.
    """
    def load(name):
        return np.load(path_method / f"split_{split}_{name}.npy", allow_pickle=False)

    cov_train   = load("covariances_train")
    y4_train    = load("conditionals_train")
    cov_val     = load("covariances_val")
    y4_val      = load("conditionals_val")
    groups_va   = load("groups_val")
    gen_train   = load("covariances_generated_samples_train")   # (T, N*max_aug, 8, 8)
    y4_gen_pool = load("conditionals_generated_samples_train")  # (N*max_aug,)
    train_time  = float(load("training_time").flat[0])
    samp_time   = float(load("sampling_time").flat[0])

    # Determine pool size (backwards compat: if file missing, assume 1)
    aug_max_path = path_method / f"split_{split}_aug_factor_max.npy"
    max_aug = int(np.load(aug_max_path)[0]) if aug_max_path.exists() else 1

    diag_tr = to_asd_td(y4_train)
    diag_va = to_asd_td(y4_val)
    cond_tr = to_ec_cpt(y4_train)
    cond_va = to_ec_cpt(y4_val)

    subject_id  = int(groups_va[0])
    y_true_subj = int(diag_va[0])

    N = len(y4_train)   # original training count

    gen_pool_last = gen_train[-1]   # (N*max_aug, 8, 8)
    if not is_all_spd(gen_pool_last):
        gen_pool_last = project_on_SPD(gen_pool_last)

    rows = []
    base = {
        "Dataset": dataset, "Group": group, "Method": method,
        "Split": split, "Subject": subject_id, "y_true": y_true_subj,
        "Train time (s)": train_time, "Sampling time (s)": samp_time,
    }

    for aug in aug_factors:
        k = min(aug, max_aug)
        if k < aug:
            print(f"  WARN split {split}: aug={aug} > max_aug={max_aug}, capped at {k}")

        # Slice first k out of each group of max_aug
        # Pool order: [y[0]]*max_aug, [y[1]]*max_aug, ..., [y[N-1]]*max_aug
        idx = np.concatenate([np.arange(i * max_aug, i * max_aug + k)
                               for i in range(N)])
        gen_tr_k  = gen_pool_last[idx]   # (N*k, 8, 8)
        y_gen_tr_k = to_asd_td(y4_gen_pool[idx])   # decoded diagnosis labels

        for cond_name, cond_val in [("EC", 0), ("CPT", 1), ("All", None)]:
            if cond_val is not None:
                tr_m = cond_tr == cond_val
                va_m = cond_va == cond_val
                # For the generated pool, replicate the mask k times
                tr_m_gen = np.concatenate([tr_m] * k)
            else:
                tr_m     = np.ones(len(cov_train), dtype=bool)
                va_m     = np.ones(len(cov_val),   dtype=bool)
                tr_m_gen = np.ones(len(gen_tr_k),  dtype=bool)

            if va_m.sum() == 0:
                continue

            X_real_tr = cov_train[tr_m];       y_tr = diag_tr[tr_m]
            X_gen_tr  = gen_tr_k[tr_m_gen];    y_gen = y_gen_tr_k[tr_m_gen]
            X_real_va = cov_val[va_m]

            for comparison, X_tr_use, y_tr_use in [
                ("Baseline (Real→Val)", X_real_tr, y_tr),
                ("TSTR (Gen→Val)",      X_gen_tr,  y_gen),
            ]:
                if aug > 1 and comparison == "Baseline (Real→Val)":
                    # Baseline doesn't depend on aug_factor; record only once (aug=1)
                    continue
                s = score_subject(X_tr_use, y_tr_use, X_real_va)
                if s is not None:
                    rows.append({**base,
                                 "AugFactor":  aug,
                                 "Condition":  cond_name,
                                 "Comparison": comparison,
                                 "y_score":    s})

    # Add baseline (aug-independent) once with AugFactor=0 as marker
    for cond_name, cond_val in [("EC", 0), ("CPT", 1), ("All", None)]:
        if cond_val is not None:
            tr_m = cond_tr == cond_val
            va_m = cond_va == cond_val
        else:
            tr_m = np.ones(len(cov_train), dtype=bool)
            va_m = np.ones(len(cov_val),   dtype=bool)
        if va_m.sum() == 0:
            continue
        X_real_tr = cov_train[tr_m];  y_tr = diag_tr[tr_m]
        X_real_va = cov_val[va_m]
        s = score_subject(X_real_tr, y_tr, X_real_va)
        if s is not None:
            rows.append({**base,
                         "AugFactor":  0,   # 0 = "Real only" baseline
                         "Condition":  cond_name,
                         "Comparison": "Baseline (Real→Val)",
                         "y_score":    s})

    print(f"  [{dataset}/{method}] split {split} — "
          f"subject {subject_id} ({'ASD' if y_true_subj else 'TD '}), "
          f"{len(rows)} rows (aug_factors={aug_factors}, max_aug={max_aug})")
    return rows


# =============================================================================
# Aggregate: compute metrics across all left-out subjects
# =============================================================================
def aggregate_predictions(df_pred: pd.DataFrame) -> pd.DataFrame:
    """
    df_pred has one row per (split, condition, comparison) with y_true / y_score.
    Aggregate over all subjects → compute ROC-AUC, F1, Precision, Recall.
    """
    group_cols = ["Dataset", "Method", "Condition", "Comparison", "AugFactor"]
    agg_rows = []

    for keys, g in df_pred.groupby(group_cols):
        if len(g["y_true"].unique()) < 2:
            print(f"  SKIP {keys}: only one class ({len(g)} subjects)")
            continue

        roc = roc_auc_score(g["y_true"], g["y_score"])
        y_pred = (g["y_score"] >= 0.5).astype(int)
        agg_rows.append({
            **dict(zip(group_cols, keys)),
            "N_subjects":        len(g),
            "ROC-AUC":           roc,
            "F1":                f1_score(g["y_true"], y_pred, zero_division=0),
            "Precision":         precision_score(g["y_true"], y_pred, zero_division=0),
            "Recall":            recall_score(g["y_true"], y_pred, zero_division=0),
            "Train time (s)":    g["Train time (s)"].mean(),
            "Sampling time (s)": g["Sampling time (s)"].mean(),
        })

    return pd.DataFrame(agg_rows)


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

    if not tasks:
        print("No result files found — check that train_b.py has been run.")
        sys.exit(1)

    print(f"Found {len(tasks)} split(s) across: {DATASETS}")

    print(f"AugFactor sweep: {AUG_FACTORS}")

    results = Parallel(n_jobs=N_JOBS)(
        delayed(evaluate_split)(ds, grp, mth, sp, path, AUG_FACTORS)
        for ds, grp, mth, sp, path in tasks
    )

    # Flatten per-subject prediction rows
    all_pred = [r for rows in results for r in rows]
    if not all_pred:
        print("No predictions collected — all splits may have skipped.")
        sys.exit(1)

    df_pred = pd.DataFrame(all_pred)

    # Save raw predictions (useful for debugging)
    raw_csv = PATH_FIGURES / f"asd_predictions_b_{REGION}.csv"
    df_pred.to_csv(raw_csv, index=False, float_format="%.4f")
    print(f"Raw predictions → {raw_csv}  ({len(df_pred)} rows)")

    # Aggregate and compute final metrics
    df_agg = aggregate_predictions(df_pred)
    if df_agg.empty:
        print("Aggregation produced no rows — check class balance across splits.")
        sys.exit(1)

    out_csv = PATH_FIGURES / f"asd_classification_b_{REGION}.csv"
    df_agg.to_csv(out_csv, index=False, float_format="%.3f")
    print(f"\nSaved → {out_csv}")

    # Summary
    baseline = df_agg[df_agg["Comparison"] == "Baseline (Real→Val)"]
    tstr     = df_agg[df_agg["Comparison"] == "TSTR (Gen→Val)"]

    for label, sub in [("Baseline", baseline), ("TSTR", tstr)]:
        if sub.empty:
            continue
        summary = (sub.groupby(["Dataset", "Method", "Condition"])
                   [["ROC-AUC", "F1"]].mean().round(3))
        print(f"\n{label}:")
        print(summary.to_string())
