import os
import numpy as np
import scipy.io
import warnings
from sklearn.covariance import OAS
from tqdm import tqdm
import csv

# =============================================================================
# Configuration (mirror phase1_preprocessing.py)
# =============================================================================
DATA_ROOT = "D:/東元/A.Data"
GROUP_INFO_PATH = "GroupInfo.mat"
SFREQ = 1000
N_CHANS = 16
N_CHANS_FRONT = 8   # channels 0-7  → Primary subject
N_CHANS_BACK  = 8   # channels 8-15 → Secondary subject

TASK_MAP = {
    0: 'EC1',
    1: 'CPT',
    2: 'EC2'
}

# EC and CPT kept separate; diagnosis label (TD=0, ASD=1) assigned per subject
NEW_TASKS = [
    {"name": "EC",  "indices": [0, 2]},
    {"name": "CPT", "indices": [1]},
]

CONFIGS = [
    {"window": 2.0, "overlap": 0.0, "folder": "cov_2s_0ov"},
    {"window": 2.0, "overlap": 0.5, "folder": "cov_2s_50ov"},
    {"window": 4.0, "overlap": 0.0, "folder": "cov_4s_0ov"},
    {"window": 4.0, "overlap": 0.5, "folder": "cov_4s_50ov"},
]

LOG_FILE = "cov_processing_report.csv"

warnings.filterwarnings("ignore")


# =============================================================================
# Helpers
# =============================================================================

def load_data_from_fdt(fdt_path, n_chans):
    """Loads raw float32 data from .fdt file. Returns (n_chans, n_samples)."""
    try:
        data = np.fromfile(fdt_path, dtype='<f4')
        n_samples = data.size // n_chans
        if data.size % n_chans != 0:
            return None, f"Divisibility error ({data.size}/{n_chans})"
        data = data[:n_samples * n_chans].reshape((n_chans, n_samples), order='F')
        return data, None
    except Exception as e:
        return None, str(e)


def compute_joint_cov_matrices(data_raw, config, normalize=True):
    """
    Slide a window over 16-channel data, compute OAS covariance on all 16 channels
    per window, then extract P (top-left 8×8), S (bottom-right 8×8), and
    inter (bottom-left 8×8, i.e. S-rows × P-cols) blocks.

    Computing the joint covariance first (rather than P and S separately) ensures
    P, S, and inter blocks come from identical time windows — no alignment mismatch.

    Parameters
    ----------
    data_raw  : (16, n_total_samples)
    config    : dict with 'window' (sec) and 'overlap' (0~1)
    normalize : if True, convert to correlation matrix before extracting blocks
                (removes amplitude confounds; recommended for ASD classification)

    Returns
    -------
    covs_p     : (n_valid_trials, 8, 8)  Primary intra-brain block
    covs_s     : (n_valid_trials, 8, 8)  Secondary intra-brain block
    covs_inter : (n_valid_trials, 8, 8)  Cross-brain block (S-rows × P-cols)
                 Not SPD; use inter @ inter.T (Gram matrix) for Riemannian ops.
    status     : str
    """
    win_samples = int(config["window"] * SFREQ)
    step        = int(win_samples * (1 - config["overlap"]))
    n_trials    = (data_raw.shape[1] - win_samples) // step + 1

    if n_trials <= 2:
        return None, None, None, f"Too few trials: {n_trials}"

    p  = N_CHANS_FRONT  # 8
    covs_p, covs_s, covs_inter = [], [], []

    for t in range(n_trials):
        start = t * step
        end   = start + win_samples
        seg   = data_raw[:, start:end]  # (16, win_samples)

        # Skip window if ANY channel has NaN or the whole segment is flat
        if np.isnan(seg).any() or np.all(seg == 0):
            continue

        # Joint 16×16 OAS covariance (both subjects simultaneously)
        cov16 = OAS().fit(seg.T).covariance_  # (16, 16)

        if normalize:
            std = np.sqrt(np.diag(cov16))
            # Avoid division by zero (flat channel)
            std = np.where(std < 1e-10, 1.0, std)
            cov16 = cov16 / (std[:, None] * std[None, :])
            np.fill_diagonal(cov16, 1.0)

        covs_p.append(cov16[:p, :p])          # top-left  (8×8) Primary intra
        covs_s.append(cov16[p:, p:])          # bot-right (8×8) Secondary intra
        covs_inter.append(cov16[p:, :p])      # bot-left  (8×8) cross-brain

    if not covs_p:
        return None, None, None, "All windows failed"

    return (np.stack(covs_p),
            np.stack(covs_s),
            np.stack(covs_inter),
            "Success")


# =============================================================================
# Main
# =============================================================================

def main():
    if not os.path.exists(GROUP_INFO_PATH):
        print(f"Error: {GROUP_INFO_PATH} not found!")
        return

    g_info = scipy.io.loadmat(GROUP_INFO_PATH)
    availability = g_info['GroupInfo'][0, 0]['availability']  # (3, 43)
    diagnosis    = g_info['GroupInfo'][0, 0]['condiction'][1, :]  # (43,) 0=TD, 1=ASD

    for cfg in CONFIGS:
        os.makedirs(cfg["folder"], exist_ok=True)

    report = []
    subjects = [f"G{i+1:02d}" for i in range(43)]

    total_ops = len(CONFIGS) * 43 * len(NEW_TASKS)
    pbar = tqdm(total=total_ops, desc="Phase 1 Covariance")

    for cfg in CONFIGS:
        out_base = cfg["folder"]

        all_X_p     = {"EC": [], "CPT": []}
        all_X_s     = {"EC": [], "CPT": []}
        all_X_inter = {"EC": [], "CPT": []}
        all_y       = {"EC": [], "CPT": []}
        all_grps    = {"EC": [], "CPT": []}

        for sub_idx, sub in enumerate(subjects):
            diag_label = int(diagnosis[sub_idx])  # 0=TD, 1=ASD

            for task_cfg in NEW_TASKS:
                cond_name      = task_cfg["name"]
                target_indices = task_cfg["indices"]

                combined_data = []

                for t_idx in target_indices:
                    orig_cond    = TASK_MAP[t_idx]
                    is_available = availability[t_idx, sub_idx]

                    if is_available == 1:
                        possible_paths = [
                            os.path.join(DATA_ROOT, sub, f"merged_{orig_cond}.fdt"),
                            os.path.join(DATA_ROOT, sub, f"{sub}_{orig_cond}.fdt"),
                        ]
                        fdt_path = next((p for p in possible_paths if os.path.exists(p)), None)

                        if fdt_path:
                            data_raw, err = load_data_from_fdt(fdt_path, N_CHANS)
                            if not err:
                                combined_data.append(data_raw)

                status = "Skipped"
                shape  = "N/A"

                if combined_data:
                    final_raw = np.concatenate(combined_data, axis=1)  # (16, n_samples)

                    covs_p, covs_s, covs_inter, status = compute_joint_cov_matrices(
                        final_raw, cfg, normalize=True
                    )

                    if covs_p is not None:
                        n_trials = len(covs_p)
                        shape    = str(covs_p.shape)

                        np.save(os.path.join(out_base, f"{sub}_{cond_name}_p.npy"),     covs_p)
                        np.save(os.path.join(out_base, f"{sub}_{cond_name}_s.npy"),     covs_s)
                        np.save(os.path.join(out_base, f"{sub}_{cond_name}_inter.npy"), covs_inter)

                        all_X_p[cond_name].append(covs_p)
                        all_X_s[cond_name].append(covs_s)
                        all_X_inter[cond_name].append(covs_inter)
                        all_y[cond_name].append(np.full(n_trials, diag_label,  dtype=np.int64))
                        all_grps[cond_name].append(np.full(n_trials, sub_idx, dtype=np.int64))
                else:
                    status = "No data found"

                report.append({
                    "Config": cfg["folder"], "Subject": sub,
                    "Condition": cond_name, "Status": status, "Shape": shape,
                })
                pbar.set_postfix({"Cfg": cfg["folder"], "Sub": sub, "Task": cond_name})
                pbar.update(1)

        # Save aggregate files (X/y/groups per condition and suffix)
        for cond_name in ("EC", "CPT"):
            if not all_y[cond_name]:
                continue
            y_all   = np.concatenate(all_y[cond_name],    axis=0)
            grp_all = np.concatenate(all_grps[cond_name], axis=0)
            np.save(os.path.join(out_base, f"y_{cond_name}.npy"),      y_all)
            np.save(os.path.join(out_base, f"groups_{cond_name}.npy"), grp_all)

            for suffix, acc in (("p", all_X_p), ("s", all_X_s), ("inter", all_X_inter)):
                if not acc[cond_name]:
                    continue
                X_all = np.concatenate(acc[cond_name], axis=0)
                np.save(os.path.join(out_base, f"X_{cond_name}_{suffix}.npy"), X_all)
                print(f"\n[{cfg['folder']}] {cond_name}_{suffix}: X={X_all.shape}, "
                      f"TD={np.sum(y_all==0)}, ASD={np.sum(y_all==1)}, "
                      f"subjects={len(np.unique(grp_all))}")

    pbar.close()

    with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, fieldnames=["Config", "Subject", "Condition", "Status", "Shape"]
        )
        writer.writeheader()
        writer.writerows(report)

    print(f"\nDone. Report: {LOG_FILE}")


if __name__ == "__main__":
    main()
