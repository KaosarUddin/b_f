# pred_traits_alphaZ_vs_baselines.py
import os
import re
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, r2_score
from spd_metrics_id.distance import alpha_z_bw, pearson_distance, euclidean_distance

# =========================
# Config
# =========================
#BASE_PATH = r"D:/Research AU/Python/connectomes_100/"
BASE_PATH = r"D:/Research AU/connectomes_900/"
EXCEL     = r"D:/Research AU/100_Subj_Full_v3.xlsx"
USE_SCAN  = "mean"     # "mean", "LR", or "RL"
K         = 5          # kNN neighbors
CV        = 5          # CV folds
SEED      = 42
MAX_NAN_FRAC = 0.20    # drop traits with >20% missing before imputation

# =========================
# I/O helpers
# =========================
def load_connectivity_matrix(file_path):
    try:
        A = np.loadtxt(file_path, delimiter=' ')
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"Not a square matrix: {file_path}")
        return A
    except Exception as e:
        print(f"[WARN] Error loading {file_path}: {e}")
        return None

def generate_file_paths(base_path, scan_type, num_subjects=10**9):
    subs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    subs.sort()
    subs = subs[:num_subjects]
    return [os.path.join(base_path, sid, f"{sid}_rfMRI_REST1_{scan_type}_900") for sid in subs]

def ensure_spd(M, tau=1e-6):
    if M is None:
        return None
    M = 0.5*(M + M.T) + tau*np.eye(M.shape[0])
    w, V = eigh(M)
    w = np.maximum(w, 1e-10)
    return (V * w) @ V.T

def build_fc_dict(lr_paths, rl_paths, use="mean"):
    """Return dict {subject_id: SPD_matrix}. If use='mean', average LR/RL when both exist."""
    def sid_from_path(p):
        m = re.search(r'([0-9]{6})_rfMRI_REST1_', os.path.basename(p))
        return m.group(1) if m else None

    rl_by_id = {sid_from_path(p): p for p in rl_paths if os.path.exists(p)}
    fc = {}

    for lp in lr_paths:
        sid = sid_from_path(lp)
        if sid is None:
            continue
        L = load_connectivity_matrix(lp) if os.path.exists(lp) else None
        R = load_connectivity_matrix(rl_by_id[sid]) if sid in rl_by_id else None
        L = ensure_spd(L) if L is not None else None
        R = ensure_spd(R) if R is not None else None

        if use == "LR" and L is not None:
            fc[sid] = L
        elif use == "RL" and R is not None:
            fc[sid] = R
        else:
            if L is not None and R is not None:
                fc[sid] = ensure_spd(0.5*(L + R), tau=0.0)
            elif L is not None:
                fc[sid] = L
            elif R is not None:
                fc[sid] = R
            # else: skip if neither exists
    return fc

# =========================
# Pairwise distances (A,B) -> scalar  ==>  N×N matrix
# =========================
def pairwise_distance_matrix(X, fn, symmetric=True):
    """
    X: list of SPD matrices [A1, A2, ..., AN]
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
# kNN with precomputed distances
# =========================
def knn_classify_precomputed(D, y, n_neighbors=5, n_splits=5, seed=0):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(np.zeros(len(y)), y):
        D_tr_tr = D[np.ix_(tr, tr)]
        D_te_tr = D[np.ix_(te, tr)]
        clf = KNeighborsClassifier(n_neighbors=n_neighbors, metric="precomputed")
        clf.fit(D_tr_tr, y[tr])
        yhat = clf.predict(D_te_tr)
        accs.append(accuracy_score(y[te], yhat))
    return float(np.mean(accs)), accs

def knn_regress_precomputed(D, y, n_neighbors=5, n_splits=5, seed=0):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.zeros_like(y, dtype=float)
    for tr, te in kf.split(np.zeros(len(y))):
        D_tr_tr = D[np.ix_(tr, tr)]
        D_te_tr = D[np.ix_(te, tr)]
        reg = KNeighborsRegressor(n_neighbors=n_neighbors, metric="precomputed")
        reg.fit(D_tr_tr, y[tr])
        preds[te] = reg.predict(D_te_tr).ravel()
    yc = y - np.nanmean(y)
    pc = preds - np.nanmean(preds)
    r = float((yc @ pc) / (np.linalg.norm(yc) * np.linalg.norm(pc) + 1e-12))
    r2 = float(r2_score(y, preds))
    return r, r2, preds

# =========================
# Main
# =========================
if __name__ == "__main__":
    # 1) Load FCs
    lr_paths = generate_file_paths(BASE_PATH, "LR")
    rl_paths = generate_file_paths(BASE_PATH, "RL")
    fc_mats = build_fc_dict(lr_paths, rl_paths, use=USE_SCAN)

    # 2) Load behavioral; align subjects
    behav = pd.read_excel(EXCEL)
    behav["Subject"] = behav["Subject"].astype(str)
    fc_ids = set(fc_mats.keys())
    bh_ids = set(behav["Subject"].astype(str))
    subjects = sorted(list(fc_ids & bh_ids))

    if len(subjects) < 20:
        print("[WARN] Very few overlapping subjects.")

    X = [fc_mats[s] for s in subjects]
    B = behav.set_index("Subject").loc[subjects].copy()

    # Gender to numeric (robust, no FutureWarning)
    g = B["Gender"].astype(str).str.strip().str.upper().str[0].map({"M": 1, "F": 0})
    if g.isna().any():
        g = pd.Categorical(B["Gender"]).codes
    y_gender_full = g.to_numpy().astype(int)

    # Age
    y_age_full = pd.to_numeric(B["Age"], errors="coerce").to_numpy().astype(float)

    # Traits: numeric-only, drop high-NaN columns, impute with numeric means
    # Curated list of 58 HCP traits (present in Excel: 56 of them)
    trait_list = [
        "PicSeq_Unadj", "PicSeq_AgeAdj", "CardSort_Unadj", "CardSort_AgeAdj",
        "Flanker_Unadj", "Flanker_AgeAdj", "ReadEng_Unadj", "ReadEng_AgeAdj",
        "PicVocab_Unadj", "PicVocab_AgeAdj", "ProcSpeed_Unadj", "ProcSpeed_AgeAdj",
        "ListSort_Unadj", "ListSort_AgeAdj", "PMAT24_A_CR", "IWRD_TOT", "VSPLOT_TC",
        "DDisc_AUC_200", "DDisc_AUC_40K", "AngAffect_Unadj", "AngHostil_Unadj",
        "AngAggr_Unadj", "FearAffect_Unadj", "FearSomat_Unadj", "Sadness_Unadj",
        "PosAffect_Unadj", "LifeSatisf_Unadj", "MeanPurp_Unadj", "Friendship_Unadj",
        "Loneliness_Unadj", "PercHostil_Unadj", "PercReject_Unadj", "EmotSupp_Unadj",
        "InstruSupp_Unadj", "PercStress_Unadj", "SelfEff_Unadj",
        "NEOFAC_A", "NEOFAC_C", "NEOFAC_E", "NEOFAC_N", "NEOFAC_O",
        "PSQI_Score", "MMSE_Score", "Endurance_Unadj", "Endurance_AgeAdj",
        "Dexterity_Unadj", "Dexterity_AgeAdj", "Strength_Unadj", "Strength_AgeAdj",
        "GaitSpeed_Comp", "Odor_Unadj", "Odor_AgeAdj", "Taste_Unadj", "Taste_AgeAdj",
        "Mars_Final", "PainInterf_Tscore"
    ]

    # Keep only these traits (intersection with available columns)
    trait_list = [c for c in trait_list if c in B.columns]

    traits_num = B[trait_list].apply(pd.to_numeric, errors="coerce")
    traits_num = traits_num.fillna(traits_num.mean(numeric_only=True))
    Y_mat_full = traits_num.to_numpy()
    trait_cols = list(traits_num.columns)

    print(f"[INFO] N subjects: {len(subjects)}")
    print(f"[INFO] Traits kept: {len(trait_cols)}")

    # 3) Distance matrices via pairwise wrappers
    # Alpha‑Z: enforce symmetry via averaging d(A,B) and d(B,A)
    D_alpha = pairwise_distance_matrix(
        X, lambda A, B: alpha_z_bw(A, B, 0.99, 1.0), symmetric=True
    )
    D_pear = pairwise_distance_matrix(
        X, lambda A, B: pearson_distance(A, B), symmetric=True
    )
    D_eucl = pairwise_distance_matrix(
        X, lambda A, B: euclidean_distance(A, B), symmetric=True
    )

    # (Optional) sanity checks
    # print("Alpha-Z symmetric:", np.allclose(D_alpha, D_alpha.T, atol=1e-8))

    # 4a) Gender (drop NaNs, ensure at least two classes)
    mask_g = ~np.isnan(y_gender_full)
    y_gender = y_gender_full[mask_g]
    D_alpha_g = D_alpha[np.ix_(mask_g, mask_g)]
    D_pear_g  = D_pear[np.ix_(mask_g, mask_g)]
    D_eucl_g  = D_eucl[np.ix_(mask_g, mask_g)]
    if len(np.unique(y_gender)) < 2:
        raise ValueError("Gender has fewer than 2 classes after cleaning.")

    acc_alpha, _ = knn_classify_precomputed(D_alpha_g, y_gender, n_neighbors=K, n_splits=CV, seed=SEED)
    acc_pear,  _ = knn_classify_precomputed(D_pear_g,  y_gender, n_neighbors=K, n_splits=CV, seed=SEED)
    acc_eucl,  _ = knn_classify_precomputed(D_eucl_g,  y_gender, n_neighbors=K, n_splits=CV, seed=SEED)

    print("\nGender (accuracy):")
    print(f"  Alpha-Z  : {acc_alpha:.3f}")
    print(f"  Pearson  : {acc_pear:.3f}")
    print(f"  Euclidean: {acc_eucl:.3f}")

    # 4b) Age (drop NaNs for target only; subset D accordingly)
    mask_a = ~np.isnan(y_age_full)
    y_age  = y_age_full[mask_a]
    D_alpha_a = D_alpha[np.ix_(mask_a, mask_a)]
    D_pear_a  = D_pear[np.ix_(mask_a, mask_a)]
    D_eucl_a  = D_eucl[np.ix_(mask_a, mask_a)]

    r_alpha, r2_alpha, _ = knn_regress_precomputed(D_alpha_a, y_age, n_neighbors=K, n_splits=CV, seed=SEED)
    r_pear,  r2_pear,  _ = knn_regress_precomputed(D_pear_a,  y_age, n_neighbors=K, n_splits=CV, seed=SEED)
    r_eucl,  r2_eucl,  _ = knn_regress_precomputed(D_eucl_a,  y_age, n_neighbors=K, n_splits=CV, seed=SEED)

    print("\nAge (r / R^2):")
    print(f"  Alpha-Z  : r={r_alpha:.3f}, R^2={r2_alpha:.3f}")
    print(f"  Pearson  : r={r_pear:.3f},  R^2={r2_pear:.3f}")
    print(f"  Euclidean: r={r_eucl:.3f},  R^2={r2_eucl:.3f}")

    # 4c) Traits (already numeric & imputed; no NaNs)
    rows = []
    for j, name in enumerate(trait_cols):
        y = Y_mat_full[:, j].astype(float)
        rA, r2A, _ = knn_regress_precomputed(D_alpha, y, n_neighbors=K, n_splits=CV, seed=SEED)
        rP, r2P, _ = knn_regress_precomputed(D_pear,  y, n_neighbors=K, n_splits=CV, seed=SEED)
        rE, r2E, _ = knn_regress_precomputed(D_eucl,  y, n_neighbors=K, n_splits=CV, seed=SEED)
        rows.append({
            "trait": name,
            "AlphaZ_r": rA, "AlphaZ_R2": r2A,
            "Pearson_r": rP, "Pearson_R2": r2P,
            "Euclid_r": rE, "Euclid_R2": r2E
        })

    traits_df = pd.DataFrame(rows).sort_values("AlphaZ_r", ascending=False)
    traits_df.to_csv("D:/Research AU/hcp_traits_prediction_results.csv", index=False)

    summary = pd.DataFrame({
        "Outcome": ["Gender acc", "Age r", "Age R^2",
                    "Traits mean r", "Traits mean R^2"],
        "Alpha-Z": [acc_alpha, r_alpha, r2_alpha,
                    traits_df["AlphaZ_r"].mean(), traits_df["AlphaZ_R2"].mean()],
        "Pearson": [acc_pear,  r_pear,  r2_pear,
                    traits_df["Pearson_r"].mean(), traits_df["Pearson_R2"].mean()],
        "Euclid":  [acc_eucl,  r_eucl,  r2_eucl,
                    traits_df["Euclid_r"].mean(),  traits_df["Euclid_R2"].mean()],
    })
    summary.to_csv("D:/Research AU/hcp_prediction_summary.csv", index=False)

    print("\n[WROTE] hcp_traits_prediction_results.csv (per-trait) and hcp_prediction_summary.csv (summary).")
