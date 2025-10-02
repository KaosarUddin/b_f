# fc_gender_age_traits_svm_svr_rest_lr.py
import os
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict
from scipy.linalg import fractional_matrix_power

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, r2_score

# =========================
# TRAIT LIST (as provided)
# =========================
TRAIT_LIST = [
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

# -------------------------
# Custom distance functions
# -------------------------
def compute_alpha_z_BW_distance(A: np.ndarray, B: np.ndarray, alpha: float, z: float) -> float:
    if not (0 <= alpha <= z <= 1):
        raise ValueError("Alpha and z must satisfy 0 <= alpha <= z <= 1")

    def Q_alpha_z(A, B, alpha, z):
        if z == 0:
            return np.zeros_like(A)
        part1 = fractional_matrix_power(B, (1 - alpha) / (2 * z))
        part2 = fractional_matrix_power(A, alpha / z)
        part3 = fractional_matrix_power(B, (1 - alpha) / (2 * z))
        Q_az = fractional_matrix_power(part1.dot(part2).dot(part3), z)
        return Q_az

    Q_az = Q_alpha_z(A, B, alpha, z)
    divergence = np.trace((1 - alpha) * A + alpha * B) - np.trace(Q_az)
    return float(np.real(divergence))

def compute_pearson_distance(X: np.ndarray, Y: np.ndarray) -> float:
    X_vec = X.ravel()
    Y_vec = Y.ravel()
    r = np.corrcoef(X_vec, Y_vec)[0, 1]
    if np.isnan(r):
        r = 0.0  # fall back if one vector is constant
    return 1.0 - float(r)

def compute_euclidean_distance(X: np.ndarray, Y: np.ndarray) -> float:
    return float(np.linalg.norm(X.ravel() - Y.ravel()))

# -------------------------
# Paths & loading
# -------------------------
def prefix(task: str) -> str:
    # Only REST is allowed; prefix is rfMRI
    return 'rfMRI'

def generate_file_paths_hcp(
    base_path: str,
    rest_run: str = "REST1",      # "REST1" or "REST2" ONLY
    ts_length: int = 400,
    subject_filter: Optional[List[str]] = None,
    num_subjects: Optional[int] = None,
) -> List[str]:
    assert rest_run in {"REST1", "REST2"}, "rest_run must be 'REST1' or 'REST2'"
    scan_type = "LR"  # FORCE LR ONLY
    all_ids = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    if subject_filter is not None:
        subj = [s for s in all_ids if s in set(map(str, subject_filter))]
    else:
        subj = all_ids
    if num_subjects is not None:
        subj = subj[:num_subjects]

    files = []
    for sid in subj:
        fname = f"{sid}_{prefix(rest_run)}_{rest_run}_{scan_type}_{ts_length}"
        files.append(os.path.join(base_path, sid, fname))
    return files

def load_connectivity_matrix(file_path: str) -> Optional[np.ndarray]:
    try:
        return np.loadtxt(file_path, delimiter=' ')
    except Exception as e:
        print(f"[WARN] Could not load {file_path}: {e}")
        return None

def load_fc_matrices(paths: List[str]) -> Tuple[List[np.ndarray], List[str]]:
    mats, keep_ids = [], []
    for p in paths:
        A = load_connectivity_matrix(p)
        if A is not None:
            mats.append(A.astype(np.float64))
            keep_ids.append(os.path.basename(os.path.dirname(p)))
    return mats, keep_ids

# -------------------------
# Behavior I/O
# -------------------------
def load_behavior_labels(excel_path: str, sheet: str = "Behavior",
                         subject_col: str = "Subject") -> pd.DataFrame:
    df = pd.ExcelFile(excel_path).parse(sheet)
    if subject_col not in df.columns:
        raise ValueError(f"'{subject_col}' not found in sheet '{sheet}'.")
    df[subject_col] = df[subject_col].astype(str)
    return df

def extract_gender_labels(df: pd.DataFrame) -> np.ndarray:
    if "Gender" in df.columns:
        y_raw = df["Gender"].astype(str).str.strip()
        y = y_raw.map({"F": 0, "M": 1}).to_numpy()
    elif "Gender 1-2" in df.columns:
        y = df["Gender 1-2"].map({1: 0, 2: 1}).to_numpy()
    else:
        raise ValueError("No gender column found. Expected 'Gender' or 'Gender 1-2'.")
    return y

# -------------------------
# Pairwise distances (FC→FC)
# -------------------------
def pairwise_distance_matrix(
    mats: List[np.ndarray],
    metric: str = "alpha_z",
    alpha: float = 0.99,
    z: float = 1.0,
    symmetric: bool = True,
) -> np.ndarray:
    n = len(mats)
    D = np.zeros((n, n), dtype=float)

    if metric == "alpha_z":
        def fn(A, B):
            d1 = compute_alpha_z_BW_distance(A, B, alpha, z)
            if symmetric:
                d2 = compute_alpha_z_BW_distance(B, A, alpha, z)
                return 0.5 * (d1 + d2)
            return d1
    elif metric == "pearson":
        fn = compute_pearson_distance
    elif metric == "euclidean":
        fn = compute_euclidean_distance
    else:
        raise ValueError("metric must be 'alpha_z', 'pearson', or 'euclidean'")

    for i in range(n):
        for j in range(i + 1, n):
            d = fn(mats[i], mats[j])
            D[i, j] = d
            D[j, i] = d
    return D

# -------------------------
# Distances → RBF kernel
# -------------------------
def kernel_from_distance(D: np.ndarray, gamma: Optional[float] = None, gamma_scale: float = 2.0) -> Tuple[np.ndarray, float]:
    """
    Returns (K, gamma) where K = exp(-gamma * D^2).
    By default, gamma = gamma_scale / median(D^2) with gamma_scale=2.0 (as in your earlier code).
    """
    X = D.copy()
    np.fill_diagonal(X, 0.0)
    sq = X ** 2
    nz = sq[sq > 0]
    med = np.median(nz) if nz.size > 0 else 1.0
    if (med <= 0) or np.isnan(med):
        med = 1.0
    if gamma is None:
        gamma = gamma_scale / med
    K = np.exp(-gamma * sq)
    return K, float(gamma)

# -------------------------
# CV routines
# -------------------------
def svm_cv(K: np.ndarray, y: np.ndarray, C: float = 100, n_splits: int = 5, seed: int = 42) -> Tuple[float, List[float]]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(K, y):
        Ktr = K[np.ix_(tr, tr)]
        Kte = K[np.ix_(te, tr)]
        clf = SVC(kernel="precomputed", C=C)
        clf.fit(Ktr, y[tr])
        yhat = clf.predict(Kte)
        accs.append(accuracy_score(y[te], yhat))
    return float(np.mean(accs)), [float(a) for a in accs]

def svr_cv(K: np.ndarray, y: np.ndarray, C: float = 1.0, n_splits: int = 5, seed: int = 42) -> Tuple[float, List[float]]:
    """K-fold CV for regression with precomputed kernel. Returns (mean R^2, list of R^2)."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    r2s = []
    for tr, te in kf.split(K):
        Ktr = K[np.ix_(tr, tr)]
        Kte = K[np.ix_(te, tr)]
        ytr = y[tr]
        yte = y[te]
        # If y contains NaNs in this fold, skip those (rare if we prefilter)
        tr_mask = np.isfinite(ytr)
        te_mask = np.isfinite(yte)
        if tr_mask.sum() < 3 or te_mask.sum() < 1:
            r2s.append(float('nan'))
            continue
        svr = SVR(kernel="precomputed", C=C)
        svr.fit(Ktr[tr_mask][:, tr_mask], ytr[tr_mask])
        yhat = svr.predict(Kte[te_mask][:, tr_mask])
        r2s.append(r2_score(yte[te_mask], yhat))
    # Filter NaNs in aggregation
    r2s_clean = [r for r in r2s if np.isfinite(r)]
    mean_r2 = float(np.mean(r2s_clean)) if len(r2s_clean) > 0 else float('nan')
    return mean_r2, [float(r) if np.isfinite(r) else float('nan') for r in r2s]

# -------------------------
# Main pipeline
# -------------------------
def run_gender_age_traits_rest_lr(
    excel_path: str,
    fc_base_dir: str,
    rest_run: str = "REST1",  # "REST1" or "REST2"
    ts_length: int = 100,
    subject_col: str = "Subject",
    behavior_sheet: str = "Behavior",
    n_splits: int = 5,
    C_cls: float = 100,        # SVC C
    C_reg: float = 1.0,        # SVR C
    seed: int = 42,
    alpha: float = 0.99,
    z: float = 1.0,
    gamma_scale: float = 2.0,  # gamma = gamma_scale / median(D^2)
    min_samples_trait: int = 30
):
    assert rest_run in {"REST1", "REST2"}, "rest_run must be 'REST1' or 'REST2'"

    # Load behavior table & build RESTx–LR filepaths for available subjects
    beh = load_behavior_labels(excel_path, behavior_sheet, subject_col)
    subj_list = beh[subject_col].astype(str).tolist()
    paths = generate_file_paths_hcp(
        base_path=fc_base_dir,
        rest_run=rest_run,
        ts_length=ts_length,
        subject_filter=subj_list,
        num_subjects=None,
    )
    FCs, keep_ids = load_fc_matrices(paths)
    if not FCs:
        raise RuntimeError("No REST–LR FCs loaded. Check base path, filenames, and ts_length.")

    # Align behavior to the actual FC subject order
    beh_aligned = beh.set_index(subject_col).loc[keep_ids].reset_index()
    y_gender = extract_gender_labels(beh_aligned)

    # =========================
    # ADDED: align and export Age + selected traits
    # =========================
    sel_cols = ["Subject", "Gender", "Age"] + [c for c in TRAIT_LIST if c in beh_aligned.columns]
    missing_cols = [c for c in TRAIT_LIST if c not in beh_aligned.columns]
    if missing_cols:
        print(f"[INFO] Missing {len(missing_cols)} trait columns not found in Behavior sheet (skipped):")
        print("       " + ", ".join(missing_cols[:12]) + (" ..." if len(missing_cols) > 12 else ""))
    df_traits = beh_aligned[sel_cols].copy()
    out_csv = os.path.join(fc_base_dir, f"_aligned_traits_age_{rest_run}_LR_{ts_length}.csv")
    try:
        df_traits.to_csv(out_csv, index=False)
        print(f"[INFO] Saved aligned traits/age to: {out_csv}")
    except Exception as e:
        print(f"[WARN] Could not save traits file: {e}")

    # -------------------------
    # Compute distances & kernels once
    # -------------------------
    Ks: Dict[str, np.ndarray] = {}
    gammas: Dict[str, float] = {}
    for metric in ["alpha_z", "pearson", "euclidean"]:
        print(f"\n>>> Computing pairwise distances: {metric.upper()}")
        D = pairwise_distance_matrix(FCs, metric=metric, alpha=alpha, z=z, symmetric=True)
        K, gamma_val = kernel_from_distance(D, gamma=None, gamma_scale=gamma_scale)
        Ks[metric] = K
        gammas[metric] = gamma_val
        medD2 = (np.median((D[D > 0]) ** 2) if np.any(D > 0) else 1.0)
        print(f"    median(D^2) = {medD2:.4g}, gamma = {gamma_val:.4g}")

    # -------------------------
    # Classification: Gender
    # -------------------------
    print("\n=== Gender Classification (SVC, precomputed kernel) ===")
    gender_results = {}
    for metric, K in Ks.items():
        mean_acc, fold_accs = svm_cv(K, y_gender, C=C_cls, n_splits=n_splits, seed=seed)
        print(f"  {metric:>9}: mean_acc={mean_acc:.3f}, folds={np.round(fold_accs, 3)}")
        gender_results[metric] = (mean_acc, fold_accs)

    # -------------------------
    # Regression: Age
    # -------------------------
    print("\n=== Age Prediction (SVR, precomputed kernel) ===")
    age = pd.to_numeric(beh_aligned["Age"], errors="coerce").to_numpy()
    mask_age = np.isfinite(age)
    if mask_age.sum() >= min_samples_trait:
        age_results = {}
        for metric, K in Ks.items():
            K_sub = K[np.ix_(mask_age, mask_age)]
            mean_r2, fold_r2s = svr_cv(K_sub, age[mask_age], C=C_reg, n_splits=n_splits, seed=seed)
            print(f"  {metric:>9}: R^2_mean={np.nan_to_num(mean_r2):.3f}, folds={np.round(fold_r2s, 3)}")
            age_results[metric] = (mean_r2, fold_r2s)
    else:
        print(f"  [WARN] Not enough samples with finite Age (have {mask_age.sum()}, need {min_samples_trait}). Skipping Age.")

    # -------------------------
    # Regression: Traits
    # -------------------------
    print("\n=== Trait Prediction (SVR, precomputed kernel) ===")
    trait_results: Dict[str, Dict[str, Tuple[float, List[float]]]] = {}

    for trait in [c for c in TRAIT_LIST if c in beh_aligned.columns]:
        y_vec = pd.to_numeric(beh_aligned[trait], errors="coerce").to_numpy()
        mask = np.isfinite(y_vec)
        if mask.sum() < min_samples_trait:
            print(f"  {trait}: [skip] only {mask.sum()} finite samples (need {min_samples_trait})")
            continue

        trait_results[trait] = {}
        for metric, K in Ks.items():
            K_sub = K[np.ix_(mask, mask)]
            mean_r2, fold_r2s = svr_cv(K_sub, y_vec[mask], C=C_reg, n_splits=n_splits, seed=seed)
            print(f"  {trait:>20} | {metric:>9}: R^2_mean={np.nan_to_num(mean_r2):.3f}, folds={np.round(fold_r2s, 3)}")
            trait_results[trait][metric] = (mean_r2, fold_r2s)

    # -------------------------
    # Summary
    # -------------------------
    print("\n=== Summary (Gender / Age / Traits) ===")
    print("Gender (accuracy):")
    for m, (ma, fa) in gender_results.items():
        print(f"  {m:>9}: mean={ma:.3f}  folds={np.round(fa, 3)}")

    if 'age_results' in locals():
        print("\nAge (R^2):")
        for m, (mr, frs) in age_results.items():
            print(f"  {m:>9}: R^2_mean={np.nan_to_num(mr):.3f}  folds={np.round(frs, 3)}")

    # Show top-5 traits by best metric R^2 (if any)
    trait_best_rows = []
    for trait, by_metric in trait_results.items():
        best_metric = max(by_metric.keys(), key=lambda k: (by_metric[k][0] if np.isfinite(by_metric[k][0]) else -np.inf))
        best_mean = by_metric[best_metric][0]
        trait_best_rows.append((trait, best_metric, best_mean))
    if trait_best_rows:
        trait_best_rows.sort(key=lambda x: (x[2] if np.isfinite(x[2]) else -np.inf), reverse=True)
        print("\nTop traits by best R^2 (any metric):")
        for t, m, r in trait_best_rows[:5]:
            print(f"  {t:>20} | {m:>9}: R^2_mean={np.nan_to_num(r):.3f}")

# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    run_gender_age_traits_rest_lr(
        excel_path=r"D:/Research AU/100_Subj_Full_v3.xlsx",
        fc_base_dir=r"D:/Research AU/connectomes_400/",
        rest_run="REST1",   # or "REST2"
        ts_length=100,
        n_splits=5,
        C_cls=100,          # SVC C for gender
        C_reg=1.0,          # SVR C for age/traits
        seed=42,
        alpha=0.99, z=1.0,
        gamma_scale=2.0,    # gamma = 2 / median(D^2), as you used before
        min_samples_trait=30
    )
