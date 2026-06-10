"""
train_custom.py
---------------
Train DiffeoCFM on custom EEG covariance matrices produced by phase1_cov.py.

Usage:
    python train_custom.py --data "./cov_2s_0ov" --region s
    python train_custom.py --data "./cov_2s_0ov" --region s --debug
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import OAS
from sklearn.model_selection import LeaveOneGroupOut

from fm import DiffeoCFM
from gaussian import DiffeoGauss


# Inlined from train.py to avoid importing constants.py (which runs argparse)
def run_split(split, cov_train, cov_val, y_train, y_val,
              groups_train, groups_val, model, path_results,
              aug_factor: int = 1):
    assert set(groups_train).isdisjoint(set(groups_val)), \
        "Groups are not disjoint between train and val sets."

    training_start = time.time()
    train_info = model.fit(cov_train, y_train)
    if train_info is not None:
        train_losses_epoch = train_info["train_loss"]
        val_losses_epoch   = train_info["val_loss"]
    training_time = time.time() - training_start

    sampling_start = time.time()
    y_train_aug = np.repeat(y_train, aug_factor)
    sol_train   = model.sample(y_train_aug)
    sol_val     = model.sample(y_val)
    sampling_time = time.time() - sampling_start

    def save(name, arr):
        np.save(path_results / f"split_{split}_{name}.npy", arr)

    if isinstance(model, DiffeoCFM):
        save("train_losses", train_losses_epoch)
        save("val_losses",   val_losses_epoch)

    save("covariances_train",  cov_train)
    save("conditionals_train", y_train)
    save("groups_train",       groups_train)
    save("covariances_val",    cov_val)
    save("conditionals_val",   y_val)
    save("groups_val",         groups_val)
    save("covariances_generated_samples_train", sol_train)
    save("conditionals_generated_samples_train", y_train_aug)
    save("covariances_generated_samples_val",   sol_val)
    save("conditionals_generated_samples_val",  y_val)
    save("training_time",   np.array([training_time]))
    save("sampling_time",   np.array([sampling_time]))
    save("aug_factor_max",  np.array([aug_factor]))

# =============================================================================
# Args
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--data",   type=str, required=True,
                    help="Folder containing G##_EC_p.npy / G##_CPT_s.npy etc.")
parser.add_argument("--region", type=str, default="s", choices=["p", "s"],
                    help="Which channel group: p=前8ch, s=後8ch (default: s)")
parser.add_argument("--max-aug", type=int, default=1, dest="max_aug",
                    help="Pool size: generate this many synthetic samples per real "
                         "training sample and save all of them (default 1).")
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

DATA_DIR   = Path(args.data)
REGION     = args.region
DEBUG      = args.debug
MAX_AUG    = args.max_aug

# =============================================================================
# Fixed settings (EEG-style, non-normalized SPD)
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
    "PATIENCE":      50,   # early stopping: epochs without val loss improvement
    "MIN_DELTA":     1e-6, # minimum improvement to count
    "T_GRID":        torch.linspace(0, 1, 6, device=DEVICE, dtype=torch.float64),
    "DEVICE":        DEVICE,
    "RNG":           np.random.RandomState(42),
}

def _make_config(diffeo=None):
    cfg = CONFIG_FM.copy()
    cfg["RNG"] = np.random.RandomState(42)
    if diffeo is not None:
        cfg["DIFFEO"] = diffeo
    return cfg

# Methods to run (SPD, non-normalized → logeuclidean / lower_triangular)
METHODS = [
    {"diffeo": "logeuclidean",      "model": DiffeoGauss({"RNG": np.random.RandomState(42), "DIFFEO": "logeuclidean"})},
    {"diffeo": "lower_triangular",  "model": DiffeoCFM(_make_config("lower_triangular"))},
    {"diffeo": "logeuclidean",      "model": DiffeoCFM(_make_config("logeuclidean"))},
]

PATH_RESULTS = Path("results")
PATH_RESULTS.mkdir(exist_ok=True)

# =============================================================================
# Load data
# =============================================================================
print(f"Loading data from {DATA_DIR} [region={REGION}] ...")
# Scan G##_EC_{region}.npy (label=0) and G##_CPT_{region}.npy (label=1)
COND_LABELS = {"EC": 0, "CPT": 1}
all_X, all_y, all_groups = [], [], []

for cond, label in COND_LABELS.items():
    for fpath in sorted(DATA_DIR.glob(f"G*_{cond}_{REGION}.npy")):
        arr = np.load(fpath)                         # (n_trials, 8, 8)
        sub_idx = int(fpath.stem.split("_")[0][1:]) - 1  # "G03" → 2
        all_X.append(arr)
        all_y.append(np.full(len(arr), label, dtype=np.int64))
        all_groups.append(np.full(len(arr), sub_idx, dtype=np.int64))

X      = np.concatenate(all_X,      axis=0)  # (N, 8, 8)
y      = np.concatenate(all_y,      axis=0)  # (N,)  0=EC, 1=CPT
groups = np.concatenate(all_groups, axis=0)  # (N,)  subject index

print(f"  X: {X.shape}  |  EC={np.sum(y==0)}, CPT={np.sum(y==1)}  |  subjects: {len(np.unique(groups))}")

# =============================================================================
# Outlier filtering (same as EEG pipeline in train.py)
# =============================================================================
mask_abs = np.max(np.abs(X), axis=(-2, -1)) < 1e4

oas = OAS()
oas.fit(X.reshape(X.shape[0], -1))
cov_inv = np.linalg.inv(oas.covariance_)
X_flat_mean = np.mean(X, axis=0).flatten()
distances = np.array([
    mahalanobis(X[i].flatten(), X_flat_mean, cov_inv)
    for i in range(X.shape[0])
])
mask_maha = distances < np.percentile(distances, 90)

mask   = mask_abs & mask_maha
X      = X[mask]
y      = y[mask]
groups = groups[mask]
print(f"  After filtering: {X.shape[0]} samples remain  "
      f"(removed {mask.size - mask.sum()})")

# =============================================================================
# Train
# =============================================================================
dataset_name = f"{DATA_DIR.name}_{REGION}"  # e.g. "cov_2s_0ov_p"

for method in METHODS:
    diffeo_name = method["diffeo"]
    model       = method["model"]
    model_name  = model.__class__.__name__

    out_dir = PATH_RESULTS / dataset_name / "group_None" / f"{diffeo_name}_{model_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining {model_name} with {diffeo_name} ...")

    all_splits = list(LeaveOneGroupOut().split(X, y, groups=groups))
    splits = all_splits[:2] if DEBUG else all_splits
    print(f"  LOSO: {len(splits)} splits")

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

    print(f"  Done. Results → {out_dir}")

print("\nAll finished.")
