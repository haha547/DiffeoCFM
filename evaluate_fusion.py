"""
evaluate_fusion.py
------------------
LOSO ASD/TD classification on fused P+S intra-brain covariances.

Data structure: each session records two subjects (Primary + Secondary) simultaneously.
The raw 16×16 covariance is block-structured:
    [ P_intra (8×8) |  inter  ]
    [  inter^T      | S_intra ]
G##_EC_p.npy = P_intra block; G##_EC_s.npy = S_intra block.
The inter-brain block is not used here.

Fusion combines P_intra and S_intra → one 8×8 matrix for classification.
Includes single-block baselines (p_only, s_only) for direct comparison.

Usage:
    python evaluate_fusion.py --data "./cov_2s_0ov"
    python evaluate_fusion.py --data "./cov_2s_0ov" "./cov_4s_0ov"
    python evaluate_fusion.py --data "./cov_2s_0ov" --methods arith_mean matrix_product
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline

from fuse import FUSION_METHODS

PATH_FIGURES = Path("figures")

# =============================================================================
# Args
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--data",          nargs="+", required=True,
                    help="Data folder(s) containing G##_EC_p.npy / G##_EC_s.npy etc.")
parser.add_argument("--groupinfo",     default="GroupInfo.mat")
parser.add_argument("--groupinfo-row", type=int, default=1, dest="groupinfo_row",
                    help="Row of condiction matrix for ASD/TD labels "
                         "(0=P region, 1=S region, default=1)")
parser.add_argument("--methods",       nargs="+", default=None,
                    help=f"Fusion methods to run (default: all). "
                         f"Available: {list(FUSION_METHODS)}")
args = parser.parse_args()

DATASETS = {Path(d).name: Path(d) for d in args.data}
METHODS  = {k: v for k, v in FUSION_METHODS.items()
            if args.methods is None or k in (args.methods or [])}

if not METHODS:
    print(f"No matching methods. Available: {list(FUSION_METHODS)}")
    sys.exit(1)

g_info = scipy.io.loadmat(args.groupinfo)
subject_diagnosis = g_info["GroupInfo"][0, 0]["condiction"][args.groupinfo_row, :]
_avail = g_info["GroupInfo"][0, 0]["availability"]   # (3, 43)
subject_available = np.all(_avail > 0, axis=0)        # (43,) bool
print(f"Labels (groupinfo row {args.groupinfo_row}): "
      f"{int(np.sum(subject_diagnosis == 0))} TD, "
      f"{int(np.sum(subject_diagnosis == 1))} ASD")
print(f"Available subjects (all 3 rows > 0): {int(subject_available.sum())} / {len(subject_available)}")
print(f"Methods: {list(METHODS)}")


# =============================================================================
# Helpers
# =============================================================================
def ensure_spd(matrices, eps=1e-6):
    """Always apply a small regularization to guarantee positive definiteness.
    eps=1e-6 is large enough for Riemannian ops (TangentSpace), small enough
    not to distort the data. Fusion ops (esp. matrix_product) can amplify
    condition numbers, so unconditional regularization is safer than checking.
    """
    orig  = matrices.shape
    flat  = matrices.reshape(-1, orig[-2], orig[-1])
    # symmetrize (guards against tiny numerical asymmetry from fusion)
    flat  = (flat + flat.transpose(0, 2, 1)) * 0.5
    eigs  = np.linalg.eigvalsh(flat).min(axis=1)
    alpha = np.where(eigs < eps, (eps - eigs) / (1 - eigs), 0.0)
    eye   = np.eye(orig[-1])[None]
    out   = (1 - alpha)[:, None, None] * flat + alpha[:, None, None] * eye
    return out.reshape(orig)


def score_subject(X_train, y_train, X_val):
    """Train TangentSpace + LR; return mean P(ASD) over val trials or None."""
    if len(np.unique(y_train)) < 2:
        return None
    clf = LogisticRegression(
        C=1.0, solver="liblinear", class_weight="balanced",
        random_state=42, max_iter=1000,
    )
    pipe = make_pipeline(TangentSpace(metric="riemann"), clf)
    try:
        pipe.fit(X_train, y_train)
        return float(pipe.predict_proba(X_val)[:, 1].mean())
    except Exception as e:
        print(f"    score_subject failed: {e}")
        return None


# =============================================================================
# Data loading: fuse P and S trial-by-trial
# =============================================================================
def load_fused(data_dir: Path, fusion_fn) -> tuple:
    """
    Load paired P+S covariances (and optionally the inter block), fuse trial-by-trial.
    Returns (X_fused, y_cond, groups) or (None, None, None) if no data.
    y_cond: 0=EC, 1=CPT
    groups: 0-indexed subject index

    The inter block (G##_<cond>_inter.npy) is loaded when present and passed to
    fusion_fn as the keyword argument `inter`.  Methods that do not need it
    (arith_mean, s_only, …) declare `inter=None` and ignore it.
    Methods that require it (inter_gram) raise ValueError if inter=None, which
    causes that subject to be skipped with a warning.
    """
    all_X, all_cond, all_groups = [], [], []
    n_skipped_inter = 0

    for cond_name, cond_idx in [("EC", 0), ("CPT", 1)]:
        for fp_p in sorted(data_dir.glob(f"G*_{cond_name}_p.npy")):
            sub_str = fp_p.stem.split("_")[0]          # "G03"
            sub_idx = int(sub_str[1:]) - 1              # 0-indexed

            fp_s     = fp_p.parent / f"{sub_str}_{cond_name}_s.npy"
            fp_inter = fp_p.parent / f"{sub_str}_{cond_name}_inter.npy"

            if not fp_s.exists():
                print(f"    SKIP {sub_str} {cond_name}: missing {fp_s.name}")
                continue

            P = np.load(fp_p)                                # (n_trials, 8, 8)
            S = np.load(fp_s)
            inter = np.load(fp_inter) if fp_inter.exists() else None

            if len(P) != len(S):
                n = min(len(P), len(S))
                print(f"    WARN {sub_str} {cond_name}: trial count mismatch "
                      f"(P={len(P)}, S={len(S)}), truncating to {n}")
                P, S = P[:n], S[:n]
                if inter is not None:
                    inter = inter[:n]

            if sub_idx < len(subject_available) and not subject_available[sub_idx]:
                continue   # skip subjects missing any of the 3 availability entries

            try:
                fused_raw = fusion_fn(P, S, inter=inter)
            except (TypeError, ValueError) as e:
                # inter_gram will raise ValueError if inter=None (file not found)
                n_skipped_inter += 1
                if n_skipped_inter == 1:
                    print(f"    SKIP {sub_str} {cond_name}: {e}")
                continue

            fused = ensure_spd(fused_raw)                    # (n_trials, 8, 8)
            all_X.append(fused)
            all_cond.append(np.full(len(fused), cond_idx, dtype=np.int64))
            all_groups.append(np.full(len(fused), sub_idx, dtype=np.int64))

    if n_skipped_inter > 1:
        print(f"    ... ({n_skipped_inter} subjects skipped — inter block files missing)")
    if not all_X:
        return None, None, None
    return (np.concatenate(all_X),
            np.concatenate(all_cond),
            np.concatenate(all_groups))


# =============================================================================
# LOSO evaluation for one (dataset, fusion method) pair
# =============================================================================
def evaluate_one(dataset: str, data_dir: Path,
                 fusion_name: str, fusion_fn) -> list[dict]:
    print(f"  [{dataset}] {fusion_name} ...", flush=True)

    X, y_cond, groups = load_fused(data_dir, fusion_fn)
    if X is None:
        print(f"    No paired P+S data found in {data_dir}")
        return []

    diag = subject_diagnosis
    loso = LeaveOneGroupOut()
    pred_rows = []

    for split, (tr_idx, va_idx) in enumerate(loso.split(X, groups, groups=groups)):
        subject_id  = int(groups[va_idx[0]])
        y_true_subj = int(diag[subject_id])

        X_tr   = X[tr_idx];      y_tr_d = diag[groups[tr_idx]]
        y_tr_c = y_cond[tr_idx]; X_va   = X[va_idx]
        y_va_c = y_cond[va_idx]

        base = {"Dataset": dataset, "FusionMethod": fusion_name,
                "Split": split, "Subject": subject_id, "y_true": y_true_subj}

        for cond_name, cond_val in [("EC", 0), ("CPT", 1), ("All", None)]:
            if cond_val is not None:
                tr_m = y_tr_c == cond_val
                va_m = y_va_c == cond_val
            else:
                tr_m = np.ones(len(X_tr), dtype=bool)
                va_m = np.ones(len(X_va), dtype=bool)

            if va_m.sum() == 0:
                continue

            s = score_subject(X_tr[tr_m], y_tr_d[tr_m], X_va[va_m])
            if s is not None:
                pred_rows.append({**base, "Condition": cond_name, "y_score": s})

    n_splits = split + 1 if pred_rows else 0
    print(f"    {len(pred_rows)} prediction rows from {n_splits} splits")
    return pred_rows


# =============================================================================
# Aggregate across LOSO subjects
# =============================================================================
def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    gcols = ["Dataset", "FusionMethod", "Condition"]
    rows = []
    for keys, g in df.groupby(gcols):
        if len(g["y_true"].unique()) < 2:
            print(f"  SKIP {keys}: only one class ({len(g)} subjects)")
            continue
        roc    = roc_auc_score(g["y_true"], g["y_score"])
        y_pred = (g["y_score"] >= 0.5).astype(int)
        rows.append({
            **dict(zip(gcols, keys)),
            "N_subjects": len(g),
            "ROC-AUC":    roc,
            "F1":         f1_score(g["y_true"], y_pred, zero_division=0),
            "Precision":  precision_score(g["y_true"], y_pred, zero_division=0),
            "Recall":     recall_score(g["y_true"], y_pred, zero_division=0),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    PATH_FIGURES.mkdir(exist_ok=True)

    all_pred = []
    for ds_name, data_dir in DATASETS.items():
        if not data_dir.exists():
            print(f"WARNING: {data_dir} not found, skipping.")
            continue
        print(f"\nDataset: {ds_name}")
        for fname, ffn in METHODS.items():
            all_pred.extend(evaluate_one(ds_name, data_dir, fname, ffn))

    if not all_pred:
        print("No predictions collected.")
        sys.exit(1)

    df_pred = pd.DataFrame(all_pred)
    raw_csv = PATH_FIGURES / "fusion_predictions.csv"
    df_pred.to_csv(raw_csv, index=False, float_format="%.4f")
    print(f"\nRaw predictions → {raw_csv}  ({len(df_pred)} rows)")

    df_agg = aggregate(df_pred)
    if df_agg.empty:
        print("Aggregation produced no rows.")
        sys.exit(1)

    out_csv = PATH_FIGURES / "fusion_classification.csv"
    df_agg.to_csv(out_csv, index=False, float_format="%.3f")
    print(f"Saved → {out_csv}")
    print("\n" + df_agg.to_string())
