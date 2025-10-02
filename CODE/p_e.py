import numpy as np
import os
from scipy.linalg import logm, norm, sqrtm


# import matplotlib.pyplot as plt

def load_connectivity_matrix(file_path):
    try:
        return np.loadtxt(file_path, delimiter=' ')
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def generate_file_paths(base_path, scan_type, num_subjects=428):
    file_paths = []
    subject_ids = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    subject_ids = subject_ids[:num_subjects]
    for subject_id in subject_ids:
        file_path = os.path.join(base_path, subject_id, f'{subject_id}_tfMRI_EMOTION_{scan_type}_900')
        file_paths.append(file_path)
    return file_paths


def make_spd(matrix, tau=1e-6):
    symmetric_matrix = (matrix + matrix.T) / 2
    regularized_matrix = symmetric_matrix + tau * np.eye(matrix.shape[0])
    return regularized_matrix




# Pearson distance
def compute_pearson_distance(X, Y):
    # Flatten the matrices to vectors
    X_vec = X.flatten()
    Y_vec = Y.flatten()

    # Compute Pearson correlation coefficient
    r = np.corrcoef(X_vec, Y_vec)[0, 1]

    # Pearson distance is defined as 1 - correlation coefficient
    distance = 1 - r
    return distance


# Euclidean disatance
def compute_euclidean_distance(X, Y):
    # Flatten the matrices to vectors
    X_vec = X.flatten()
    Y_vec = Y.flatten()

    # Compute Euclidean distance
    distance = np.linalg.norm(X_vec - Y_vec)
    return distance


def distance_matrix(connectivity_matrices_1, connectivity_matrices_2, tau=1e-6):
    num_subjects = len(connectivity_matrices_1)
    distance_matrix = np.zeros((num_subjects, num_subjects))
    for i, matrix_1 in enumerate(connectivity_matrices_1):
        matrix_1 = make_spd(matrix_1, tau=tau)
        for j, matrix_2 in enumerate(connectivity_matrices_2):
            matrix_2 = make_spd(matrix_2, tau=tau)
            distance_matrix[i, j] = compute_pearson_distance(matrix_1, matrix_2)
    return distance_matrix


def compute_id_rate(distance_matrix):
    correct_identifications = sum(np.argmin(distance_matrix[i, :]) == i for i in range(distance_matrix.shape[0]))
    return correct_identifications / distance_matrix.shape[0]


#base_path = "C:/Users/ksrru/Documents/Research AU/b_f/connectomes_100/"
#base_path = 'D:/Research AU/Python/connectomes_100/'
base_path = 'D:/Research AU/connectomes_900/'
# base_path = 'connectomes_100/'
lr_paths = generate_file_paths(base_path, 'LR')
rl_paths = generate_file_paths(base_path, 'RL')

connectivity_matrices_lr = [load_connectivity_matrix(path) for path in lr_paths]
connectivity_matrices_rl = [load_connectivity_matrix(path) for path in rl_paths]

tau = 0.0  # Adjusted regularization parameter

# Regularize matrices and compute distances
distance_matrix_1 = distance_matrix(connectivity_matrices_lr, connectivity_matrices_rl, tau=tau)
id_rate_1 = compute_id_rate(distance_matrix_1)

distance_matrix_2 = distance_matrix(connectivity_matrices_rl, connectivity_matrices_lr, tau=tau)
id_rate_2 = compute_id_rate(distance_matrix_2)

final_id_rate = (id_rate_1 + id_rate_2) / 2
print(f"ID Rate 1: {id_rate_1}")
print(f"ID Rate 2: {id_rate_2}")
print(f"Final ID Rate: {final_id_rate}")

#%%
import numpy as np
import os
from scipy.linalg import logm, norm, sqrtm


# import matplotlib.pyplot as plt

def load_connectivity_matrix(file_path):
    try:
        return np.loadtxt(file_path, delimiter=' ')
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def generate_file_paths(base_path, scan_type, num_subjects=428):
    file_paths = []
    subject_ids = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    subject_ids = subject_ids[:num_subjects]
    for subject_id in subject_ids:
        file_path = os.path.join(base_path, subject_id, f'{subject_id}_tfMRI_EMOTION_{scan_type}_900')
        file_paths.append(file_path)
    return file_paths


def make_spd(matrix, tau=1e-6):
    symmetric_matrix = (matrix + matrix.T) / 2
    regularized_matrix = symmetric_matrix + tau * np.eye(matrix.shape[0])
    return regularized_matrix




# Pearson distance
def compute_pearson_distance(X, Y):
    # Flatten the matrices to vectors
    X_vec = X.flatten()
    Y_vec = Y.flatten()

    # Compute Pearson correlation coefficient
    r = np.corrcoef(X_vec, Y_vec)[0, 1]

    # Pearson distance is defined as 1 - correlation coefficient
    distance = 1 - r
    return distance


# Euclidean disatance
def compute_euclidean_distance(X, Y):
    # Flatten the matrices to vectors
    X_vec = X.flatten()
    Y_vec = Y.flatten()

    # Compute Euclidean distance
    distance = np.linalg.norm(X_vec - Y_vec)
    return distance


def distance_matrix(connectivity_matrices_1, connectivity_matrices_2, tau=1e-6):
    num_subjects = len(connectivity_matrices_1)
    distance_matrix = np.zeros((num_subjects, num_subjects))
    for i, matrix_1 in enumerate(connectivity_matrices_1):
        matrix_1 = make_spd(matrix_1, tau=tau)
        for j, matrix_2 in enumerate(connectivity_matrices_2):
            matrix_2 = make_spd(matrix_2, tau=tau)
            distance_matrix[i, j] = compute_pearson_distance(matrix_1, matrix_2)
    return distance_matrix


def compute_id_rate(distance_matrix):
    correct_identifications = sum(np.argmin(distance_matrix[i, :]) == i for i in range(distance_matrix.shape[0]))
    return correct_identifications / distance_matrix.shape[0]


#base_path = "C:/Users/ksrru/Documents/Research AU/b_f/connectomes_100/"
#base_path = 'D:/Research AU/Python/connectomes_100/'
base_path = 'D:/Research AU/connectomes_900/'
# base_path = 'connectomes_100/'
lr_paths = generate_file_paths(base_path, 'LR')
rl_paths = generate_file_paths(base_path, 'RL')

connectivity_matrices_lr = [load_connectivity_matrix(path) for path in lr_paths]
connectivity_matrices_rl = [load_connectivity_matrix(path) for path in rl_paths]

tau = 0.0  # Adjusted regularization parameter

# Regularize matrices and compute distances
distance_matrix_1 = distance_matrix(connectivity_matrices_lr, connectivity_matrices_rl, tau=tau)
id_rate_1 = compute_id_rate(distance_matrix_1)

distance_matrix_2 = distance_matrix(connectivity_matrices_rl, connectivity_matrices_lr, tau=tau)
id_rate_2 = compute_id_rate(distance_matrix_2)

final_id_rate = (id_rate_1 + id_rate_2) / 2
print(f"ID Rate 1: {id_rate_1}")
print(f"ID Rate 2: {id_rate_2}")
print(f"Final ID Rate: {final_id_rate}")
# %%
import numpy as np
import os
from spd_metrics_id.distance import (
    alpha_z_bw,
    alpha_procrustes,
    bures_wasserstein,
    geodesic_distance,
    log_euclidean_distance,
    pearson_distance,
    euclidean_distance,
)

# --------------------------
# CONFIG: set your paths
# --------------------------
#BASE_PATH  = r"D:/Research AU/Python/connectomes_100/"
BASE_PATH  = r"D:/Research AU/connectomes_400/"
SUBJECT_ID = "100206"
TASK       = "REST1"     # or any from: REST1, EMOTION, GAMBLING, LANGUAGE, MOTOR, RELATIONAL, SOCIAL, WM
SUFFIX     = "400"       # filename suffix for your FCs
P          = 414         # matrix size

def path_for(subject_id: str, task: str, scan: str) -> str:
    prefix = "rfMRI" if task == "REST1" else "tfMRI"
    fname  = f"{subject_id}_{prefix}_{task}_{scan}_{SUFFIX}"
    return os.path.join(BASE_PATH, subject_id, fname)

def load_fc(fp: str) -> np.ndarray:
    X = np.loadtxt(fp, delimiter=' ')
    if X.shape != (P, P):
        raise ValueError(f"Expected {P}x{P}, got {X.shape} at {fp}")
    return 0.5 * (X + X.T)  # enforce symmetry

# --------------------------
# Load two FCs (LR vs RL)
# --------------------------
fp_lr = path_for(SUBJECT_ID, TASK, "LR")
fp_rl = path_for(SUBJECT_ID, TASK, "RL")
A = load_fc(fp_lr)
B = load_fc(fp_rl)

# --------------------------
# Compute distances
# --------------------------
results = {
    "alpha_z_bw": alpha_z_bw(A, B, alpha=0.99, z=1.0),
    "alpha_procrustes": alpha_procrustes(A, B, alpha=0.6),
    "bures_wasserstein": bures_wasserstein(A, B),
    "geodesic_distance": geodesic_distance(A, B),
    "log_euclidean_distance": log_euclidean_distance(A, B),
    "pearson_distance": pearson_distance(A, B),
    "euclidean_distance": euclidean_distance(A, B),
}

# --------------------------
# Print results
# --------------------------
print(f"\nSubject {SUBJECT_ID} | Task {TASK} | LR vs RL")
w = max(len(k) for k in results.keys())
for k, v in results.items():
    print(f"{k:<{w}} : {v:.6f}")

# %%
import os
import numpy as np
import pandas as pd
from spd_metrics_id.distance import (
    alpha_z_bw,
    alpha_procrustes,
    bures_wasserstein,
    geodesic_distance,
    log_euclidean_distance,
    pearson_distance,
    euclidean_distance,
)

# --------------------------
# CONFIG
# --------------------------
BASE_PATH   = r"D:/Research AU/Python/connectomes_100/"
TASK        = "REST1"     # task to use
SCAN        = "LR"        # pick one scan type (LR or RL)
SUFFIX      = "100"       # file suffix
P           = 114         # matrix size
REF_SUBJECT = "100206"    # subject to compare against
N_COMPARE   = 30          # number of other subjects

ALPHA_Z_PARAMS = dict(alpha=0.99, z=1.0)
APROC_PARAMS   = dict(alpha=0.6)

# --------------------------
# Helpers
# --------------------------
def path_for(subject_id: str, task: str, scan: str) -> str:
    prefix = "rfMRI" if task == "REST1" else "tfMRI"
    fname  = f"{subject_id}_{prefix}_{task}_{scan}_{SUFFIX}"
    return os.path.join(BASE_PATH, subject_id, fname)

def load_fc(fp: str) -> np.ndarray:
    X = np.loadtxt(fp, delimiter=" ")
    if X.shape != (P, P):
        raise ValueError(f"Expected {P}x{P}, got {X.shape} at {fp}")
    return 0.5 * (X + X.T)  # enforce symmetry

# --------------------------
# Load reference FC
# --------------------------
fp_ref = path_for(REF_SUBJECT, TASK, SCAN)
A = load_fc(fp_ref)

# --------------------------
# Pick 30 subjects
# --------------------------
all_subjects = sorted(os.listdir(BASE_PATH))
other_subjects = [s for s in all_subjects if s != REF_SUBJECT]
other_subjects = other_subjects[:N_COMPARE]

# --------------------------
# Compute distances
# --------------------------
rows = []
for subj in other_subjects:
    try:
        fp = path_for(subj, TASK, SCAN)
        if not os.path.exists(fp):
            continue
        B = load_fc(fp)

        d = {
            "subject": subj,
            "alpha_z_bw": alpha_z_bw(A, B, **ALPHA_Z_PARAMS),
            "alpha_procrustes": alpha_procrustes(A, B, **APROC_PARAMS),
            "bures_wasserstein": bures_wasserstein(A, B),
            "geodesic_distance": geodesic_distance(A, B),
            "log_euclidean_distance": log_euclidean_distance(A, B),
            "pearson_distance": pearson_distance(A, B),
            "euclidean_distance": euclidean_distance(A, B),
        }
        rows.append(d)
    except Exception as e:
        print(f"Skip {subj}: {e}")

df = pd.DataFrame(rows)
print(df)

# Save to CSV for later use
out_path = f"distances_{REF_SUBJECT}_vs_{N_COMPARE}subjects_{TASK}_{SCAN}.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved distances to {out_path}")

# %%
import os
import numpy as np
import pandas as pd
from spd_metrics_id.distance import (
    alpha_z_bw,
    alpha_procrustes,
    bures_wasserstein,
    geodesic_distance,
    log_euclidean_distance,
    pearson_distance,
    euclidean_distance,
)

# --------------------------
# CONFIG
# --------------------------
BASE_PATH   = r"D:/Research AU/Python/connectomes_100/"
#BASE_PATH   = r"D:/Research AU/connectomes_100/"
TASK        = "REST1"     # REST1, EMOTION, ...
SUFFIX      = "100"       # filename suffix
P           = 114         # matrix size
REF_SUBJECT = "137027"
N_COMPARE   = 30

ALPHA_Z_PARAMS = dict(alpha=0.99, z=1.0)
APROC_PARAMS   = dict(alpha=0.6)

# --------------------------
# Helpers
# --------------------------
def path_for(subject_id: str, task: str, scan: str) -> str:
    prefix = "rfMRI" if task == "REST1" else "tfMRI"
    fname  = f"{subject_id}_{prefix}_{task}_{scan}_{SUFFIX}"
    return os.path.join(BASE_PATH, subject_id, fname)

def load_fc(fp: str) -> np.ndarray:
    X = np.loadtxt(fp, delimiter=" ")
    if X.shape != (P, P):
        raise ValueError(f"Expected {P}x{P}, got {X.shape} at {fp}")
    return 0.5 * (X + X.T)  # enforce symmetry

def compute_all(A: np.ndarray, B: np.ndarray) -> dict:
    return {
        "alpha_z_bw":             alpha_z_bw(A, B, **ALPHA_Z_PARAMS),
        "alpha_procrustes":       alpha_procrustes(A, B, **APROC_PARAMS),
        "bures_wasserstein":      bures_wasserstein(A, B),
        "geodesic_distance":      geodesic_distance(A, B),
        "log_euclidean_distance": log_euclidean_distance(A, B),
        "pearson_distance":       pearson_distance(A, B),
        "euclidean_distance":     euclidean_distance(A, B),
    }

def collect_subjects(ref_subject: str, n_compare: int):
    subs = sorted([s for s in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, s))])
    others = [s for s in subs if s != ref_subject]
    return [ref_subject] + others[:n_compare]

def build_table(ref_subject: str, anchor_scan: str, compare_scan: str, task: str) -> pd.DataFrame:
    """
    Anchor REF on `anchor_scan` and compare to:
      - self on `compare_scan` (guaranteed included as first row)
      - N_COMPARE other subjects on `compare_scan`
    """
    subjects = collect_subjects(ref_subject, N_COMPARE)

    # Load anchor (ref) FC once
    fp_anchor = path_for(ref_subject, task, anchor_scan)
    if not os.path.exists(fp_anchor):
        raise FileNotFoundError(f"Missing FC: {fp_anchor}")
    A_anchor = load_fc(fp_anchor)

    rows = []

    # 1) Self pair first (e.g., 100206-LR vs 100206-RL)
    fp_self = path_for(ref_subject, task, compare_scan)
    if os.path.exists(fp_self):
        B_self = load_fc(fp_self)
        d = compute_all(A_anchor, B_self)
        d.update({
            "subject": ref_subject,
            "anchor_subject": ref_subject,
            "anchor_scan": anchor_scan,
            "compare_scan": compare_scan,
            "pair_type": "within"
        })
        rows.append(d)
    else:
        print(f"[warn] Missing self {compare_scan} for {ref_subject}: {fp_self}")

    # 2) Compare to others on the opposite scan
    for subj in subjects:
        if subj == ref_subject:
            continue
        fp_other = path_for(subj, task, compare_scan)
        if not os.path.exists(fp_other):
            print(f"[skip] Missing {compare_scan} for {subj}: {fp_other}")
            continue
        B = load_fc(fp_other)
        d = compute_all(A_anchor, B)
        d.update({
            "subject": subj,
            "anchor_subject": ref_subject,
            "anchor_scan": anchor_scan,
            "compare_scan": compare_scan,
            "pair_type": "between"
        })
        rows.append(d)

    df = pd.DataFrame(rows, columns=[
        "subject", "anchor_subject", "anchor_scan", "compare_scan", "pair_type",
        "alpha_z_bw", "alpha_procrustes", "bures_wasserstein", "geodesic_distance",
        "log_euclidean_distance", "pearson_distance", "euclidean_distance"
    ])
    return df

# --------------------------
# Build BOTH directions you asked for:
#   A) 100206-LR vs (self RL + others RL)
#   B) 100206-RL vs (self LR + others LR)
# --------------------------
df_LR_vs_RL = build_table(REF_SUBJECT, anchor_scan="LR", compare_scan="RL", task=TASK)
df_RL_vs_LR = build_table(REF_SUBJECT, anchor_scan="RL", compare_scan="LR", task=TASK)

print("\n=== Anchor LR → Compare RL ===")
print(df_LR_vs_RL.head())
print("\n=== Anchor RL → Compare LR ===")
print(df_RL_vs_LR.head())

# Save both
outA = f"distances_{REF_SUBJECT}_{TASK}_anchorLR_vs_RL_{N_COMPARE}subs.csv"
outB = f"distances_{REF_SUBJECT}_{TASK}_anchorRL_vs_LR_{N_COMPARE}subs.csv"
df_LR_vs_RL.to_csv(outA, index=False)
df_RL_vs_LR.to_csv(outB, index=False)
print(f"\nSaved:\n  {outA}\n  {outB}")
#%%

import os, time
from pathlib import Path
import pandas as pd

DST_DIR = r"C:\Users\ksrru\Downloads\bb59818382_25_1\timeseries_700"
IDS_FILE =  r"D:\Research AU\HCP_428_unrelated_subjects_IDs.xlsx"   # or .csv
SHEET_NAME = "Sheet1"  # ignored for CSV

def read_ids(path, sheet="Sheet1"):
    p = Path(path)
    if p.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(p, sheet_name=sheet, header=None)
    else:
        df = pd.read_csv(p, header=None)
    ids = []
    for v in df.values.flatten():
        if pd.isna(v): continue
        s = str(v).strip()
        if s.endswith(".0"): s = s[:-2]
        s = "".join(ch for ch in s if ch.isdigit())
        if s: ids.append(s)
    return ids

def main():
    base = Path(DST_DIR)
    ids = read_ids(IDS_FILE, SHEET_NAME)
    # start from "now", step +1 second per folder so order is preserved when sorting by Date Modified
    t = int(time.time())
    for sid in ids:
        folder = base / sid
        if folder.is_dir():
            t += 1
            os.utime(folder, (t, t))  # access & modified times
            print("Touched:", folder)
        else:
            print("Missing (skip):", folder)

if __name__ == "__main__":
    main()
#%%
import os
import glob
import numpy as np

# =========================
# CONFIG
# =========================
BASE_PATH = r"C:/Users/ksrru/Downloads/bb59818382_25_1/timeseries_700"    # root folder with per-subject subfolders
OUT_PATH  = r"D:/Research AU/truncated_timeseries_sl_700"  # where truncated outputs go
TR = 0.72
DURATIONS_MIN = [1.2, 2.4, 3.6,4.8,6,7.2,8.40,9.60,10.8,12,13.2,14.28]                  # truncations (minutes)
MIN_T = 30                                      # minimum frames to keep a truncated version

# =========================
# HELPERS
# =========================
def list_subjects(base_path: str):
    subs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    subs.sort()
    return subs

def find_ts_files_for_subject(base_path: str, subject_id: str):
    """
    Returns (lr_path, rl_path) by globbing HCP-style names:
    <sub>/<sub>_rfMRI_REST1_LR_100*
    <sub>/<sub>_rfMRI_REST1_RL_100*
    """
    sub_dir = os.path.join(base_path, subject_id)
    lr_candidates = sorted(glob.glob(os.path.join(sub_dir, f"{subject_id}_rfMRI_REST1_LR_700*")))
    rl_candidates = sorted(glob.glob(os.path.join(sub_dir, f"{subject_id}_rfMRI_REST1_RL_700*")))
    lr_path = lr_candidates[0] if lr_candidates else None
    rl_path = rl_candidates[0] if rl_candidates else None
    return lr_path, rl_path

def load_ts(path: str):
    try:
        X = np.loadtxt(path)
        if X.ndim == 1:
            X = X[:, None]
        return np.asarray(X, dtype=np.float64)
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
        return None

def vols_for_minutes(minutes: float, tr: float) -> int:
    return int(np.floor((minutes * 60.0) / tr))

def save_ts(X: np.ndarray, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savetxt(out_path, X, fmt="%.6f")

# =========================
# MAIN
# =========================
def truncate_and_save(base_path=BASE_PATH, out_path=OUT_PATH):
    subs = list_subjects(base_path)
    print(f"[INFO] Found {len(subs)} subjects.")

    for sid in subs:
        lr_path, rl_path = find_ts_files_for_subject(base_path, sid)
        if lr_path is None or rl_path is None:
            print(f"[SKIP] Missing LR/RL for {sid}")
            continue

        ts_lr, ts_rl = load_ts(lr_path), load_ts(rl_path)
        if ts_lr is None or ts_rl is None:
            continue

        Tlr, Trl = ts_lr.shape[0], ts_rl.shape[0]

        for m in DURATIONS_MIN:
            need_vols = vols_for_minutes(m, TR)
            take_lr, take_rl = min(Tlr, need_vols), min(Trl, need_vols)

            if take_lr < MIN_T or take_rl < MIN_T:
                print(f"[SKIP] {sid} {m}min: too few volumes")
                continue

            # Prepare output dirs
            out_subdir = os.path.join(out_path, f"{m}min", sid)
            os.makedirs(out_subdir, exist_ok=True)

            # Save truncated files
            save_ts(ts_lr[:take_lr, :], os.path.join(out_subdir, f"{sid}_LR_{m}min.txt"))
            save_ts(ts_rl[:take_rl, :], os.path.join(out_subdir, f"{sid}_RL_{m}min.txt"))

        print(f"[DONE] {sid}")

if __name__ == "__main__":
    truncate_and_save()
    print("[INFO] Finished all truncations.")
#%%
import os
import numpy as np
import pandas as pd

from spd_metrics_id.distance import (
    alpha_z_bw,
    alpha_procrustes,
    bures_wasserstein,
    geodesic_distance,
    log_euclidean_distance,
    pearson_distance,
    euclidean_distance,
)

# --------------------------
# CONFIG
# --------------------------
BASE_PATH   = r"D:/Research AU/Python/connectomes_100/"
TASK        = "REST1"
SUFFIX      = "100"
P           = 114
ALPHA_Z_PARAMS = dict(alpha=0.99, z=1.0)
APROC_PARAMS   = dict(alpha=0.6)

RNG_SEED   = 12345   # change for different random picks
SAMPLE_K   = 10      # number of random subjects to test

# --------------------------
# Helpers
# --------------------------
def path_for(subject_id: str, task: str, scan: str) -> str:
    prefix = "rfMRI" if task == "REST1" else "tfMRI"
    fname  = f"{subject_id}_{prefix}_{task}_{scan}_{SUFFIX}"
    return os.path.join(BASE_PATH, subject_id, fname)

def load_fc(fp: str) -> np.ndarray:
    X = np.loadtxt(fp, delimiter=" ").astype(np.float64)
    if X.shape != (P, P):
        raise ValueError(f"Expected {P}x{P}, got {X.shape} at {fp}")
    return 0.5 * (X + X.T)

def ensure_spd(A: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    w = np.clip(w, a_min=eps, a_max=None)
    A_spd = (V * w) @ V.T
    return np.real_if_close(0.5 * (A_spd + A_spd.T), tol=1000)

def compute_spd_metrics(A: np.ndarray, B: np.ndarray) -> dict:
    A_spd = ensure_spd(A)
    B_spd = ensure_spd(B)
    return dict(
        alpha_z_bw        = alpha_z_bw(A_spd, B_spd, **ALPHA_Z_PARAMS),
        alpha_procrustes  = alpha_procrustes(A_spd, B_spd, **APROC_PARAMS),
        bures_wasserstein = bures_wasserstein(A_spd, B_spd),
        geodesic_distance = geodesic_distance(A_spd, B_spd),
        log_euclidean_distance = log_euclidean_distance(A_spd, B_spd),
    )

def compute_raw_metrics(A: np.ndarray, B: np.ndarray) -> dict:
    return dict(
        pearson_distance  = pearson_distance(A, B),
        euclidean_distance= euclidean_distance(A, B),
    )

def eligible_subjects():
    subs = sorted([s for s in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, s))])
    keep = []
    for s in subs:
        if os.path.exists(path_for(s, TASK, "LR")) and os.path.exists(path_for(s, TASK, "RL")):
            keep.append(s)
    return keep

def orientation_check(anchor_subj: str, anchor_scan: str, compare_scan: str, pool_subjects: list):
    """
    For one subject and one orientation (e.g., LR->RL), test:
      alpha-z : within < min(between)
      pearson : within > min(between)
    Returns dict with details if condition holds, else None.
    """
    A = load_fc(path_for(anchor_subj, TASK, anchor_scan))

    # within (self)
    B_self = load_fc(path_for(anchor_subj, TASK, compare_scan))
    d_spd_self  = compute_spd_metrics(A, B_self)
    d_raw_self  = compute_raw_metrics(A, B_self)

    # between (vs others)
    az_between = []
    pr_between = []
    pr_argmins = []  # store (other_subj, pearson_dist)
    for o in pool_subjects:
        if o == anchor_subj:
            continue
        B_other = load_fc(path_for(o, TASK, compare_scan))
        d_spd_other = compute_spd_metrics(A, B_other)
        d_raw_other = compute_raw_metrics(A, B_other)
        az_between.append(d_spd_other["alpha_z_bw"])
        pr_between.append(d_raw_other["pearson_distance"])
        pr_argmins.append((o, d_raw_other["pearson_distance"]))

    if len(az_between) == 0:
        return None

    az_within  = float(d_spd_self["alpha_z_bw"])
    pr_within  = float(d_raw_self["pearson_distance"])
    az_min_bet = float(np.min(az_between))
    pr_min_bet = float(np.min(pr_between))
    pr_match   = min(pr_argmins, key=lambda t: t[1])[0] if pr_argmins else None

    cond_alpha_z  = az_within < az_min_bet
    cond_pearson  = pr_within > pr_min_bet   # Pearson fails (self not the closest)

    if cond_alpha_z and cond_pearson:
        return {
            "subject": anchor_subj,
            "orientation": f"{anchor_scan}->{compare_scan}",
            "alpha_z_within": az_within,
            "alpha_z_min_between": az_min_bet,
            "pearson_within": pr_within,
            "pearson_min_between": pr_min_bet,
            "pearson_argmin_subject": pr_match,
        }
    return None

# --------------------------
# Main logic
# --------------------------
subs = eligible_subjects()
if len(subs) < SAMPLE_K:
    raise RuntimeError(f"Found only {len(subs)} eligible subjects; need at least {SAMPLE_K}.")

rng = np.random.default_rng(RNG_SEED)
sample = rng.choice(subs, size=SAMPLE_K, replace=False)
print(f"Random sample ({SAMPLE_K}): {', '.join(sample)}")

hits = []
for s in sample:
    # Try LR->RL
    res = orientation_check(s, "LR", "RL", subs)
    if res is None:
        # Try RL->LR
        res = orientation_check(s, "RL", "LR", subs)
    if res is not None:
        hits.append(res)

df_hits = pd.DataFrame(hits, columns=[
    "subject", "orientation",
    "alpha_z_within", "alpha_z_min_between",
    "pearson_within", "pearson_min_between",
    "pearson_argmin_subject"
])

if df_hits.empty:
    print("\nNo subjects in this random sample satisfied: "
          "within is minimal for Alpha-Z but NOT for Pearson.")
else:
    print("\nSubjects where Alpha-Z gets self as the unique minimum but Pearson does NOT:")
    print(df_hits.to_string(index=False))

    out_csv = f"alphaZ_wins_pearson_fails_{TASK}_{SUFFIX}_{SAMPLE_K}sample.csv"
    df_hits.to_csv(out_csv, index=False)
    print(f"\nSaved details to: {out_csv}")

# %%
