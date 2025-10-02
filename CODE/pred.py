# pred_age_gender_alphaZ_vs_baselines_kernelized.py
import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import SVC
from sklearn.kernel_ridge import KernelRidge

# Your metric implementations
from spd_metrics_id.distance import alpha_z_bw, pearson_distance, euclidean_distance

# =========================
# Config
# =========================
BASE_PATH = r"D:/Research AU/Python/connectomes_100/"
EXCEL     = r"D:/Research AU/100_Subj_Full_v3.xlsx"
USE_SCAN  = "mean"     # "mean", "LR", or "RL"
CV        = 5          # CV folds
SEED      = 42

# =========================
# I/O helpers
# =========================
def load_connectivity_matrix(file_path):
    """Load a whitespace-delimited matrix and basic-check it's square."""
    try:
        A = np.loadtxt(file_path, delimiter=' ')
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"Not a square matrix: {file_path}")
        return A
    except Exception as e:
        print(f"[WARN] Error loading {file_path}: {e}")
        return None

def sid_from_path(p):
    m = re.search(r'([0-9]{6})_rfMRI_REST1_', os.path.basename(p))
    return m.group(1) if m else None

def generate_file_paths_for_subjects(base_path, scan_type, subject_ids):
    """Build expected HCP file paths for each subject id and scan_type (LR/RL)."""
    return [os.path.join(base_path, sid, f"{sid}_rfMRI_REST1_{scan_type}_100") for sid in subject_ids]

def build_fc_dict_for_subjects(subject_ids, use="mean"):
    """
    Return dict {subject_id: symmetric_matrix}.
    If use='mean', average LR/RL when both exist, then symmetrize.
    (No PD enforcement; Alpha-Z assumed robust. Add jitter if needed.)
    """
    lr_paths = generate_file_paths_for_subjects(BASE_PATH, "LR", subject_ids)
    rl_paths = generate_file_paths_for_subjects(BASE_PATH, "RL", subject_ids)

    rl_by_id = {sid_from_path(p): p for p in rl_paths if os.path.exists(p)}
    fc = {}

    for lp in lr_paths:
        sid = sid_from_path(lp)
        if sid is None:
            continue

        L = load_connectivity_matrix(lp) if os.path.exists(lp) else None
        R = load_connectivity_matrix(rl_by_id[sid]) if sid in rl_by_id else None

        # Symmetrize (no PD enforcement)
        L = 0.5 * (L + L.T) if L is not None else None
        R = 0.5 * (R + R.T) if R is not None else None

        if use == "LR" and L is not None:
            fc[sid] = L
        elif use == "RL" and R is not None:
            fc[sid] = R
        else:
            if L is not None and R is not None:
                if L.shape != R.shape:
                    print(f"[WARN] Shape mismatch for {sid}: LR {L.shape} vs RL {R.shape}; using LR only.")
                    fc[sid] = L
                else:
                    M = 0.5 * (L + R)
                    fc[sid] = 0.5 * (M + M.T)  # symmetrize the mean
            elif L is not None:
                fc[sid] = L
            elif R is not None:
                fc[sid] = R
            # else: neither exists -> skip

    return fc

# =========================
# Distances (A,B) -> scalar  ==>  N×N matrix
# =========================
def pairwise_distance_matrix(X, fn, symmetric=True):
    """
    X: list of matrices [A1, A2, ..., AN]
    fn: callable (A, B) -> float
    symmetric: average d(A,B) and d(B,A) to enforce symmetry (use True for Alpha-Z)
    """
    N = len(X)
    D = np.zeros((N, N), dtype=float)
    for i in range(N):
        D[i, i] = 0.0
        Ai = X[i]
        for j in range(i + 1, N):
            Aj = X[j]
            d_ij = float(fn(Ai, Aj))
            if symmetric:
                d_ji = float(fn(Aj, Ai))
                d = 0.5 * (d_ij + d_ji)
            else:
                d = d_ij
            D[i, j] = d
            D[j, i] = d
    return D

# =========================
# Kernels from distances + CV helpers
# =========================
def rbf_kernel_from_distance(D, gamma=None):
    """
    Build an RBF kernel K = exp(-gamma * D^2) from a precomputed distance matrix D.
    If gamma is None, use the median heuristic on the upper-tri distances.
    """
    tri = D[np.triu_indices_from(D, k=1)]
    if gamma is None:
        tri_pos = tri[tri > 0]
        med = np.median(tri_pos) if tri_pos.size else 1.0
        gamma = 1.0 / (2.0 * (med**2 + 1e-12))
    K = np.exp(-gamma * (D**2))
    return K, gamma

def svm_classify_precomputed_kernel(K, y, n_splits=5, seed=0, C=1.0):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(np.zeros(len(y)), y):
        K_tr_tr = K[np.ix_(tr, tr)]
        K_te_tr = K[np.ix_(te, tr)]
        clf = SVC(kernel='precomputed', C=C)
        clf.fit(K_tr_tr, y[tr])
        yhat = clf.predict(K_te_tr)
        accs.append(accuracy_score(y[te], yhat))
    return float(np.mean(accs)), accs

def krr_regress_precomputed_kernel(K, y, n_splits=5, seed=0, alpha=1.0):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.zeros_like(y, dtype=float)
    for tr, te in kf.split(np.zeros(len(y))):
        K_tr_tr = K[np.ix_(tr, tr)]
        K_te_tr = K[np.ix_(te, tr)]
        reg = KernelRidge(alpha=alpha, kernel='precomputed')
        reg.fit(K_tr_tr, y[tr])
        preds[te] = reg.predict(K_te_tr).ravel()
    yc = y - np.nanmean(y)
    pc = preds - np.nanmean(preds)
    r = float((yc @ pc) / (np.linalg.norm(yc) * np.linalg.norm(pc) + 1e-12))
    r2 = float(r2_score(y, preds))
    return r, r2, preds

# =========================
# Main
# =========================
if __name__ == "__main__":
    np.random.seed(SEED)

    # 0) Read the subject list FROM the Excel (preserve its order)
    behav_all = pd.read_excel(EXCEL)
    if "Subject" not in behav_all.columns:
        raise RuntimeError("The Excel file must have a 'Subject' column.")
    subj_list_in = behav_all["Subject"].astype(str).str.zfill(6).tolist()

    # 1) Load FCs for these subjects (skip ones missing FC files)
    fc_mats = build_fc_dict_for_subjects(subj_list_in, use=USE_SCAN)
    have_fc_ids = set(fc_mats.keys())
    if not have_fc_ids:
        raise RuntimeError("No FC matrices found for the listed subjects.")

    # 2) Align behavior to the same subjects AND FC availability (order = Excel order)
    behav_all["Subject"] = behav_all["Subject"].astype(str).str.zfill(6)
    subjects = [sid for sid in subj_list_in if sid in have_fc_ids and sid in set(behav_all["Subject"])]

    dropped = [sid for sid in subj_list_in if sid not in subjects]
    if dropped:
        print(f"[WARN] Dropped {len(dropped)} subjects (missing FC and/or behavior): "
              f"{dropped[:10]}{' ...' if len(dropped)>10 else ''}")

    if len(subjects) < 20:
        print("[WARN] Very few overlapping subjects.")

    # Build X (FC list) & B (behavior table for subjects)
    X = [fc_mats[s] for s in subjects]
    B = behav_all.set_index("Subject").loc[subjects].copy()

    # Targets: Gender (1/0), Age (float)
    g = B["Gender"].astype(str).str.strip().str.upper().str[0].map({"M": 1, "F": 0})
    if g.isna().any():
        g = pd.Categorical(B["Gender"]).codes
    y_gender_full = g.to_numpy().astype(float)  # float for NaN mask first

    y_age_full = pd.to_numeric(B["Age"], errors="coerce").to_numpy().astype(float)

    # Save subjects actually used
    pd.Series(subjects, name="Subject").to_csv("subjects_used.csv", index=False)
    print(f"[INFO] N subjects used: {len(subjects)}")

    # 3) Distance matrices
    print("[INFO] Computing distance matrices ...")
    D_alpha = pairwise_distance_matrix(
        X, lambda A, B: alpha_z_bw(A, B, 0.99, 1.0), symmetric=True
    )
    D_pear = pairwise_distance_matrix(
        X, lambda A, B: pearson_distance(A, B), symmetric=True
    )
    D_eucl = pairwise_distance_matrix(
        X, lambda A, B: euclidean_distance(A, B), symmetric=True
    )

    # 4) Build RBF kernels from distances (median heuristic gamma)
    print("[INFO] Building kernels from distances ...")
    K_alpha, gamma_a = rbf_kernel_from_distance(D_alpha, gamma=None)
    K_pear,  gamma_p = rbf_kernel_from_distance(D_pear,  gamma=None)
    K_eucl,  gamma_e = rbf_kernel_from_distance(D_eucl,  gamma=None)
    print(f"[INFO] RBF gammas -> Alpha-Z: {gamma_a:.3e} | Pearson: {gamma_p:.3e} | Euclid: {gamma_e:.3e}")

    # 5a) Gender (SVM on precomputed kernels)
    mask_g = ~np.isnan(y_gender_full)
    y_gender = y_gender_full[mask_g].astype(int)
    if len(np.unique(y_gender)) < 2:
        raise ValueError("Gender has fewer than 2 classes after cleaning.")

    K_alpha_g = K_alpha[np.ix_(mask_g, mask_g)]
    K_pear_g  = K_pear[np.ix_(mask_g, mask_g)]
    K_eucl_g  = K_eucl[np.ix_(mask_g, mask_g)]

    svm_alpha_acc, _ = svm_classify_precomputed_kernel(K_alpha_g, y_gender, n_splits=CV, seed=SEED, C=1.0)
    svm_pear_acc,  _ = svm_classify_precomputed_kernel(K_pear_g,  y_gender, n_splits=CV, seed=SEED, C=1.0)
    svm_eucl_acc,  _ = svm_classify_precomputed_kernel(K_eucl_g,  y_gender, n_splits=CV, seed=SEED, C=1.0)

    print("\nGender (SVM, kernels from distances):")
    print(f"  Alpha-Z  : {svm_alpha_acc:.3f}")
    print(f"  Pearson  : {svm_pear_acc:.3f}")
    print(f"  Euclidean: {svm_eucl_acc:.3f}")

    # 5b) Age (Kernel Ridge on precomputed kernels)
    mask_a = ~np.isnan(y_age_full)
    y_age  = y_age_full[mask_a]

    K_alpha_a = K_alpha[np.ix_(mask_a, mask_a)]
    K_pear_a  = K_pear[np.ix_(mask_a, mask_a)]
    K_eucl_a  = K_eucl[np.ix_(mask_a, mask_a)]

    kr_alpha_r, kr_alpha_r2, _ = krr_regress_precomputed_kernel(K_alpha_a, y_age, n_splits=CV, seed=SEED, alpha=1.0)
    kr_pear_r,  kr_pear_r2,  _ = krr_regress_precomputed_kernel(K_pear_a,  y_age, n_splits=CV, seed=SEED, alpha=1.0)
    kr_eucl_r,  kr_eucl_r2,  _ = krr_regress_precomputed_kernel(K_eucl_a,  y_age, n_splits=CV, seed=SEED, alpha=1.0)

    print("\nAge (Kernel Ridge on distance kernels; r / R^2):")
    print(f"  Alpha-Z  : r={kr_alpha_r:.3f}, R^2={kr_alpha_r2:.3f}")
    print(f"  Pearson  : r={kr_pear_r:.3f},  R^2={kr_pear_r2:.3f}")
    print(f"  Euclidean: r={kr_eucl_r:.3f},  R^2={kr_eucl_r2:.3f}")

    # 6) Write summary CSV
    summary = pd.DataFrame({
        "Outcome": ["Gender acc (SVM)", "Age r (KRR)", "Age R^2 (KRR)"],
        "Alpha-Z": [svm_alpha_acc, kr_alpha_r, kr_alpha_r2],
        "Pearson": [svm_pear_acc,  kr_pear_r,  kr_pear_r2],
        "Euclid":  [svm_eucl_acc,  kr_eucl_r,  kr_eucl_r2],
    })
    summary.to_csv("hcp_age_gender_summary.csv", index=False)

    print("\n[WROTE] hcp_age_gender_summary.csv and subjects_used.csv")
