"""
train_b.py  —  Direction B
---------------------------
Train DiffeoCFM with 4-class conditioning:
    0 = TD-EC  |  1 = TD-CPT  |  2 = ASD-EC  |  3 = ASD-CPT

One model learns the joint distribution across all four groups.
Generated samples can be decoded back to ASD/TD and EC/CPT.

Usage:
    python train_b.py --data "./cov_2s_0ov" --region s
    python train_b.py --data "./cov_2s_0ov" --region s --debug
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import time
from pathlib import Path

import numpy as np
import scipy.io
import torch
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import OAS
from sklearn.model_selection import LeaveOneGroupOut

from fm import DiffeoCFM
from fuse import ensure_spd


# =============================================================================
# Args
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--data",      type=str, required=True,
                    help="Folder containing G##_EC_p.npy / G##_CPT_s.npy etc.")
parser.add_argument("--region",    type=str, default="s", choices=["p", "s", "inter_gram"])
parser.add_argument("--groupinfo", type=str, default="GroupInfo.mat")
parser.add_argument("--max-aug",   type=int, default=1, dest="max_aug",
                    help="Pool size: generate this many synthetic samples per real "
                         "training sample and save all of them (default 1). "
                         "Use --aug in evaluate_b.py to test different sub-sizes.")
parser.add_argument("--debug",     action="store_true")
args = parser.parse_args()

DATA_DIR   = Path(args.data)
REGION     = args.region
DEBUG      = args.debug
MAX_AUG    = args.max_aug
REGION_ROW = 0 if REGION == "p" else 1

# =============================================================================
# Settings
# =============================================================================
torch.manual_seed(42)
np.random.seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE}.")

EPOCHS        = 10  if DEBUG else 2000
WARMUP_EPOCHS = 5   if DEBUG else 10

CONFIG_FM = {
    "FM_TYPE":       "classic",
    "WARMUP_EPOCHS": WARMUP_EPOCHS,
    "FACTOR_LR":     0.1,
    "LR":            0.001,
    "BATCH_SIZE":    64,
    "EPOCHS":        EPOCHS,
    "HIDDEN_DIM":    [512],
    "PRINT_EVERY":   100,
    "PATIENCE":      50,
    "MIN_DELTA":     1e-6,
    "T_GRID":        torch.linspace(0, 1, 6, device=DEVICE, dtype=torch.float64),
    "DEVICE":        DEVICE,
    "RNG":           np.random.RandomState(42),
}

def _make_config(diffeo):
    cfg = CONFIG_FM.copy()
    cfg["RNG"] = np.random.RandomState(42)
    cfg["DIFFEO"] = diffeo
    return cfg

METHODS = [
    {"diffeo": "logeuclidean",     "model": DiffeoCFM(_make_config("logeuclidean"))},
    {"diffeo": "lower_triangular", "model": DiffeoCFM(_make_config("lower_triangular"))},
]

PATH_RESULTS = Path("results_b")
PATH_RESULTS.mkdir(exist_ok=True)

# =============================================================================
# Load ASD/TD labels
# =============================================================================
g_info = scipy.io.loadmat(args.groupinfo)
# condiction: (2, 43)  row0=p, row1=s  |  0=TD, 1=ASD
subject_diagnosis = g_info["GroupInfo"][0, 0]["condiction"][REGION_ROW, :]  # (43,)
# availability: (3, 43)  —  include subject only if all 3 entries are non-zero
_avail = g_info["GroupInfo"][0, 0]["availability"]          # (3, 43)
subject_available = np.all(_avail > 0, axis=0)              # (43,) bool

# =============================================================================
# Load data  →  4-class label: asd*2 + cond_idx
#   0=TD-EC  1=TD-CPT  2=ASD-EC  3=ASD-CPT
# =============================================================================
print(f"Loading data from {DATA_DIR} [region={REGION}] ...")

COND_MAP = {"EC": 0, "CPT": 1}
all_X, all_y, all_groups = [], [], []

_glob_suffix = "inter" if REGION == "inter_gram" else REGION

for cond_name, cond_idx in COND_MAP.items():
    for fpath in sorted(DATA_DIR.glob(f"G*_{cond_name}_{_glob_suffix}.npy")):
        arr     = np.load(fpath)                              # (n_trials, 8, 8)
        if REGION == "inter_gram":
            arr = ensure_spd(arr @ arr.transpose(0, 2, 1))   # inter @ inter.T → SPD
        sub_idx = int(fpath.stem.split("_")[0][1:]) - 1      # "G03" → 2
        asd     = int(subject_diagnosis[sub_idx])             # 0=TD, 1=ASD
        label   = asd * 2 + cond_idx                         # 0/1/2/3
        all_X.append(arr)
        all_y.append(np.full(len(arr), label, dtype=np.int64))
        all_groups.append(np.full(len(arr), sub_idx, dtype=np.int64))

X      = np.concatenate(all_X,      axis=0)  # (N, 8, 8)
y      = np.concatenate(all_y,      axis=0)  # (N,) 4-class
groups = np.concatenate(all_groups, axis=0)  # (N,) subject index

counts = {name: int(np.sum(y == i))
          for i, name in enumerate(["TD-EC", "TD-CPT", "ASD-EC", "ASD-CPT"])}
print(f"  X: {X.shape}  |  {counts}  |  subjects: {len(np.unique(groups))}")

# =============================================================================
# Outlier filtering
# =============================================================================
mask_abs = np.max(np.abs(X), axis=(-2, -1)) < 1e4

oas = OAS()
oas.fit(X.reshape(X.shape[0], -1))
cov_inv     = np.linalg.inv(oas.covariance_)
X_flat_mean = np.mean(X, axis=0).flatten()
distances   = np.array([
    mahalanobis(X[i].flatten(), X_flat_mean, cov_inv) for i in range(len(X))
])
mask_maha = distances < np.percentile(distances, 90)

mask   = mask_abs & mask_maha
X      = X[mask]
y      = y[mask]
groups = groups[mask]
print(f"  After outlier filter: {len(X)} samples  (removed {mask.size - mask.sum()})")

# Availability filter: keep only subjects with all 3 availability entries non-zero
mask_avail = subject_available[groups]
n_excl = int((~subject_available).sum())
X      = X[mask_avail]
y      = y[mask_avail]
groups = groups[mask_avail]
print(f"  After availability filter: {len(X)} samples "
      f"({n_excl} subjects excluded, {int(subject_available.sum())} remain)")


# =============================================================================
# run_split
# =============================================================================
def run_split(split, cov_train, cov_val, y_train, y_val,
              groups_train, groups_val, model, path_results,
              aug_factor: int = 1):
    """
    aug_factor: how many synthetic samples to generate per real training sample.
      1 → same size as real data (default, 1:1)
      3 → 3× real data size (e.g. 1 real → 3 generated)
    Each call to model.sample() draws a fresh random prior point, so
    repeating y_train k times naturally yields k diverse samples per label.
    """
    assert set(groups_train).isdisjoint(set(groups_val)), \
        "Groups are not disjoint between train and val sets."

    t0 = time.time()
    train_info = model.fit(cov_train, y_train)
    training_time = time.time() - t0

    t0 = time.time()
    # Generate pool by calling sample(y_train) aug_factor times to avoid GPU OOM.
    # Each call processes only N samples; results are stacked into (T, N*aug, 8, 8)
    # with np.repeat ordering so evaluate_b.py pool-slicing stays correct.
    if aug_factor > 1:
        sol_parts = [model.sample(y_train) for _ in range(aug_factor)]
        T_steps = sol_parts[0].shape[0]
        N_tr    = sol_parts[0].shape[1]
        extra   = sol_parts[0].shape[2:]
        sol_train = (np.stack(sol_parts, axis=2)
                     .reshape(T_steps, N_tr * aug_factor, *extra))
    else:
        sol_train = model.sample(y_train)
    y_train_aug = np.repeat(y_train, aug_factor)   # (N×aug_factor,) labels
    sol_val     = model.sample(y_val)              # val stays 1:1
    sampling_time = time.time() - t0

    def save(name, arr):
        np.save(path_results / f"split_{split}_{name}.npy", arr)

    if train_info is not None:
        save("train_losses", train_info["train_loss"])
        save("val_losses",   train_info["val_loss"])

    save("covariances_train",                    cov_train)
    save("conditionals_train",                   y_train)        # original (N,)
    save("groups_train",                         groups_train)
    save("covariances_val",                      cov_val)
    save("conditionals_val",                     y_val)
    save("groups_val",                           groups_val)
    save("covariances_generated_samples_train",  sol_train)
    save("conditionals_generated_samples_train", y_train_aug)    # (N×aug_factor,)
    save("covariances_generated_samples_val",    sol_val)
    save("conditionals_generated_samples_val",   y_val)
    save("training_time",                        np.array([training_time]))
    save("sampling_time",                        np.array([sampling_time]))
    save("aug_factor_max",                       np.array([aug_factor]))


# =============================================================================
# Train
# =============================================================================
dataset_name = f"{DATA_DIR.name}_{REGION}"

for method in METHODS:
    diffeo_name = method["diffeo"]
    model       = method["model"]
    model_name  = model.__class__.__name__

    out_dir = PATH_RESULTS / dataset_name / "group_None" / f"{diffeo_name}_{model_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining {model_name} [{diffeo_name}] with 4-class conditioning ...")

    all_splits = list(LeaveOneGroupOut().split(X, y, groups=groups))
    splits = all_splits[:2] if DEBUG else all_splits
    print(f"  LOSO: {len(splits)} splits")

    # Sequential (no joblib) — required when using CUDA
    for split, (train_idx, val_idx) in enumerate(splits):
        print(f"  Split {split + 1}/{len(splits)}  "
              f"(val subject: {groups[val_idx[0]]}) ...")
        run_split(
            split        = split,
            cov_train    = X[train_idx],
            cov_val      = X[val_idx],
            y_train      = y[train_idx],
            y_val        = y[val_idx],
            groups_train = groups[train_idx],
            groups_val   = groups[val_idx],
            model        = model,
            path_results = out_dir,
            aug_factor   = MAX_AUG,
        )

    # Free GPU memory before next method
    if hasattr(model, "vf"):
        del model.vf
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print(f"  Done. Results → {out_dir}")

print("\nAll finished.")
