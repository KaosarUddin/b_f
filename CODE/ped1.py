# az_landmark_pls_server.py
import os
import argparse
import json
import numpy as np
import pandas as pd

from typing import List, Tuple, Optional

from scipy.linalg import eigh
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.cross_decomposition import PLSRegression

from joblib import Parallel, delayed
from tqdm import tqdm

# ============== CONFIG (defaults; override with CLI) ==============
DEF_EXCEL   = r"/mmfs1/home/mzu0014/100_Subj_Full_v3.xlsx"
DEF_BASE    = r"/mmfs1/home/mzu0014/connectomes_100"
DEF_OUTDIR  = r"/mmfs1/home/mzu0014/project1/alphaZ_gender_results"

DEF_SCAN    = "REST1"
DEF_HEMI    = "LR"
DEF_PARC    = 100
DEF_FILE_TEMPLATE = "{sid}_{prefix}_{scan}_{hemi}_{parc}"  # add extension if needed
DEF_DELIM   = " "
DEF_EXT     = ""        # e.g., ".txt" if your files have one

# Alpha–Z (valid range typically 0 <= alpha <= z <= 1)
DEF_ALPHA   = 0.99
DEF_Z       = 1.0

DEF_SPLITS  = 5
DEF_NPROT   = 50
DEF_USE_RBF = True
DEF_PLS_C   = 10
DEF_THREADS = 8
DEF_SEED    = 42


# ===================== I/O & Utilities ============================
def prefix(task: str) -> str:
    return "rfMRI" if task.startswith("REST") else "tfMRI"

def construct_file_path(base: str, sid: str, scan: str, hemi: str, parc: int,
                        tmpl: str, ext: str) -> str:
    name = tmpl.format(sid=sid, prefix=prefix(scan), scan=scan, hemi=hemi, parc=parc)
    if ext:
        name += ext
    return os.path.join(base, sid, name)

def load_id_gender(excel_path: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path)
    cols = [c.lower() for c in df.columns]

    id_col = None
    for cand in ["subject", "subj", "id", "subject_id", "sid"]:
        if cand in cols:
            id_col = df.columns[cols.index(cand)]
            break
    if id_col is None:
        raise ValueError("Could not find Subject ID column (expected Subject/Subj/ID/Subject_ID/sid).")

    gender_col = None
    for cand in ["gender", "sex"]:
        if cand in cols:
            gender_col = df.columns[cols.index(cand)]
            break
    if gender_col is None:
        raise ValueError("Could not find Gender column (expected Gender or Sex).")

    sids = df[id_col].astype(str).tolist()
    graw = df[gender_col].tolist()

    gnorm = []
    for g in graw:
        if isinstance(g, str):
            gl = g.strip().lower()
            if gl in ("m", "male", "1"):
                gnorm.append(1)
            elif gl in ("f", "female", "2"):
                gnorm.append(2)
            else:
                raise ValueError(f"Unrecognized gender string: {g}")
        else:
            val = int(g)
            if val in (0, 1, 2):
                if val == 0: gnorm.append(1)
                elif val == 1: gnorm.append(2)
                else: gnorm.append(2)
            else:
                raise ValueError(f"Unrecognized gender value: {g}")

    return pd.DataFrame({"sid": sids, "gender": gnorm})

def load_connectivity_matrix(path: str, delim: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        print(f"[WARN] Missing file: {path}")
        return None
    try:
        X = np.loadtxt(path, delimiter=delim)
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] != X.shape[1]:
            raise ValueError(f"Matrix not square at {path}: shape={X.shape}")
        # Symmetrize + gentle eig-floor (keep SPD)
        X = 0.5 * (X + X.T)
        w, V = eigh(X)
        w = np.maximum(w, 1e-10)
        X = (V * w) @ V.T
        return X
    except Exception as e:
        print(f"[ERROR] Failed to load {path}: {e}")
        return None

# ===================== Alpha–Z BW divergence (inline) =====================

def _powm_spd(A: np.ndarray, p: float) -> np.ndarray:
    """Symmetric matrix power A^p for SPD A via eigendecomposition."""
    w, V = eigh(A)
    w = np.maximum(w, 1e-18)
    wp = w**p
    return (V * wp) @ V.T

def _trace_pow_spd(M: np.ndarray, p: float) -> float:
    """Trace of M^p for SPD M using eigenvalues."""
    w, _ = eigh(M)
    w = np.maximum(w, 1e-18)
    return float(np.sum(w**p))

def _phi_alpha_z(A: np.ndarray, B: np.ndarray, alpha: float, z: float,
                 A_pow1: np.ndarray, B_pow2: np.ndarray) -> float:
    """
    Φ_{α,z}(A,B) = tr((1-α)A + αB) - tr( ( A^{(1-α)/2z} B^{α/z} A^{(1-α)/2z} )^z )
    Inputs may include precomputed A^{(1-α)/2z} and B^{α/z}.
    """
    # (1) linear term
    lin = (1.0 - alpha) * np.trace(A) + alpha * np.trace(B)
    # (2) sandwich power term
    M = A_pow1 @ B_pow2 @ A_pow1    # SPD
    trQ = _trace_pow_spd(M, z)
    return lin - trQ

def alpha_z_bw_distance(A: np.ndarray, B: np.ndarray, alpha: float, z: float,
                        A_pow1: Optional[np.ndarray] = None,
                        B_pow2: Optional[np.ndarray] = None,
                        B_pow1: Optional[np.ndarray] = None,
                        A_pow2: Optional[np.ndarray] = None) -> float:
    """
    Symmetrized Alpha–Z BW divergence:
        d_{α,z}(A,B) = Φ_{α,z}(A,B) + Φ_{α,z}(B,A)
    For α=z=1/2, this equals the BW **squared** distance:
        tr(A)+tr(B) - 2 tr( (A^{1/2} B A^{1/2})^{1/2} ).
    """
    # Precompute powers if not provided
    if A_pow1 is None: A_pow1 = _powm_spd(A, (1.0 - alpha) / (2.0 * z))
    if B_pow2 is None: B_pow2 = _powm_spd(B, alpha / z)
    if B_pow1 is None: B_pow1 = _powm_spd(B, (1.0 - alpha) / (2.0 * z))
    if A_pow2 is None: A_pow2 = _powm_spd(A, alpha / z)

    phi_ab = _phi_alpha_z(A, B, alpha, z, A_pow1, B_pow2)
    phi_ba = _phi_alpha_z(B, A, alpha, z, B_pow1, A_pow2)
    return phi_ab + phi_ba


# ===================== Distance Computation =======================
def compute_full_alphaZ_matrix(mats: List[np.ndarray], alpha: float, z: float,
                               n_jobs: int = 8) -> np.ndarray:
    """
    Parallel precompute full NxN symmetrized Alpha–Z BW distances.
    We cache per-subject matrix powers to avoid recomputing inside the loop.
    """
    n = len(mats)
    D = np.zeros((n, n), dtype=float)

    # Precompute per-subject powers used repeatedly
    pow1 = [ _powm_spd(M, (1.0 - alpha) / (2.0 * z)) for M in mats ]
    pow2 = [ _powm_spd(M, alpha / z)                   for M in mats ]

    # Also cache traces to avoid repeated np.trace
    trc  = np.array([float(np.trace(M)) for M in mats], dtype=float)

    def row_job(i: int) -> Tuple[int, np.ndarray]:
        di = np.zeros(n, dtype=float)
        Ai, Ai_pow1, Ai_pow2, tri = mats[i], pow1[i], pow2[i], trc[i]
        for j in range(i + 1, n):
            Bj, Bj_pow1, Bj_pow2, trj = mats[j], pow1[j], pow2[j], trc[j]
            # Compute Φ_ab and Φ_ba using cached powers
            # Φ_ab
            Mab = Ai_pow1 @ Bj_pow2 @ Ai_pow1
            trQ_ab = _trace_pow_spd(Mab, z)
            phi_ab = (1.0 - alpha) * tri + alpha * trj - trQ_ab
            # Φ_ba
            Mba = Bj_pow1 @ Ai_pow2 @ Bj_pow1
            trQ_ba = _trace_pow_spd(Mba, z)
            phi_ba = (1.0 - alpha) * trj + alpha * tri - trQ_ba

            di[j] = phi_ab + phi_ba  # symmetrized divergence
        return i, di

    # Prefer threads to share precomputed arrays without copying
    results = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(row_job)(i) for i in range(n))

    for i, di in results:
        D[i, i+1:] = di[i+1:]
    D = D + D.T
    np.fill_diagonal(D, 0.0)
    return D


# ===================== Prototypes & Features ======================
def farthest_point_sampling(D: np.ndarray, m: int) -> List[int]:
    n = D.shape[0]
    total = D.sum(axis=1)
    start = int(np.argmin(total))
    sel = [start]
    min_to_sel = D[start, :].copy()
    for _ in range(1, min(m, n)):
        idx = int(np.argmax(min_to_sel))
        sel.append(idx)
        min_to_sel = np.minimum(min_to_sel, D[idx, :])
    return sel

def build_features_from_prototypes(D_tt: np.ndarray,
                                   D_trte: np.ndarray,
                                   proto_idx: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    F_train = D_tt[:, proto_idx]
    F_test  = D_trte[:, proto_idx]
    return F_train, F_test

def rbf_featurize(Ftr: np.ndarray, Fte: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    vals = Ftr.reshape(-1)
    med2 = np.median((vals ** 2))
    if med2 <= 0:
        med2 = np.median((Ftr**2).ravel()) + 1e-8
    gamma = 1.0 / med2
    Phi_tr = np.exp(-gamma * (Ftr ** 2))
    Phi_te = np.exp(-gamma * (Fte ** 2))
    return Phi_tr, Phi_te, gamma


# =========================== Main =================================
def main():
    ap = argparse.ArgumentParser(description="Alpha–Z Landmark/Prototype PLS (server-optimized, inline α–z BW)")
    ap.add_argument("--excel", default=DEF_EXCEL)
    ap.add_argument("--base", default=DEF_BASE)
    ap.add_argument("--outdir", default=DEF_OUTDIR)
    ap.add_argument("--scan", default=DEF_SCAN)
    ap.add_argument("--hemi", default=DEF_HEMI)
    ap.add_argument("--parc", type=int, default=DEF_PARC)
    ap.add_argument("--file-template", default=DEF_FILE_TEMPLATE)
    ap.add_argument("--ext", default=DEF_EXT)
    ap.add_argument("--delim", default=DEF_DELIM)

    ap.add_argument("--alpha", type=float, default=DEF_ALPHA)
    ap.add_argument("--z", type=float, default=DEF_Z)

    ap.add_argument("--splits", type=int, default=DEF_SPLITS)
    ap.add_argument("--n-prototypes", type=int, default=DEF_NPROT)
    ap.add_argument("--use-rbf", action="store_true", default=DEF_USE_RBF)
    ap.add_argument("--components", type=int, default=DEF_PLS_C)
    ap.add_argument("--threads", type=int, default=DEF_THREADS)
    ap.add_argument("--seed", type=int, default=DEF_SEED)

    ap.add_argument("--load-D", default="", help="Path to precomputed full distance matrix .npy")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    log_path = os.path.join(args.outdir, "run_config.json")
    with open(log_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    # 1) Load meta
    meta = load_id_gender(args.excel)
    sids = meta["sid"].tolist()
    y    = np.array(meta["gender"].tolist(), dtype=int)

    # 2) Load matrices
    mats, keep_idx, missing = [], [], []
    for k, sid in enumerate(tqdm(sids, desc="Loading FC matrices")):
        p = construct_file_path(args.base, sid, args.scan, args.hemi, args.parc,
                                args.file_template, args.ext)
        A = load_connectivity_matrix(p, args.delim)
        if A is None:
            missing.append(sid)
            continue
        mats.append(A)
        keep_idx.append(k)

    if missing:
        with open(os.path.join(args.outdir, "missing_files.txt"), "w") as f:
            for sid in missing:
                f.write(f"{sid}\n")
        print(f"[WARN] Missing {len(missing)} subjects. Kept {len(keep_idx)}.")

    sids = [sids[i] for i in keep_idx]
    y    = y[keep_idx]
    n = len(mats)
    if n == 0:
        raise RuntimeError("No matrices loaded. Check paths/file names/extensions.")

    # 3) Precompute or load full distance matrix
    D_full_path = args.load_D if args.load_D else os.path.join(args.outdir, "alphaZ_full_D.npy")
    if os.path.exists(D_full_path) and args.load_D:
        print(f"Loading precomputed distance matrix from: {D_full_path}")
        D_full = np.load(D_full_path)
        if D_full.shape != (n, n):
            raise ValueError(f"Loaded D has wrong shape {D_full.shape}, expected {(n,n)}.")
    else:
        print("Computing full Alpha–Z BW distance matrix (parallel)...")
        D_full = compute_full_alphaZ_matrix(mats, args.alpha, args.z, n_jobs=args.threads)
        np.save(D_full_path, D_full)
        print(f"Saved full distance matrix to: {D_full_path}")

    # 4) Stratified CV over subjects
    skf = StratifiedKFold(n_splits=args.splits, shuffle=True, random_state=args.seed)

    fold_rows = []
    all_pred, all_true, all_sid = [], [], []

    for fidx, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(n), y), 1):
        print(f"\n=== Fold {fidx}/{args.splits} ===")
        ytr = y[tr_idx]
        yte = y[te_idx]

        # Slices from the precomputed matrix
        D_trtr = D_full[np.ix_(tr_idx, tr_idx)]   # (n_tr x n_tr)
        D_tetr = D_full[np.ix_(te_idx, tr_idx)]   # (n_te x n_tr)

        # Prototypes from training
        proto_idx = farthest_point_sampling(D_trtr, args.n_prototypes)
        print(f"Selected {len(proto_idx)} prototypes.")

        # Distance-to-prototype features
        Ftr, Fte = build_features_from_prototypes(D_trtr, D_tetr, proto_idx)

        # Optional RBF featurization
        used_gamma = None
        if args.use_rbf:
            Ftr, Fte, used_gamma = rbf_featurize(Ftr, Fte)
            print(f"RBF gamma: {used_gamma:.4g}")

        # PLS (regression style); threshold at 1.5 for gender
        pls = PLSRegression(n_components=args.components)
        pls.fit(Ftr, ytr.astype(float))
        yhat_cont = pls.predict(Fte).ravel()
        yhat = np.where(yhat_cont >= 1.5, 2, 1)

        # Metrics
        acc = accuracy_score(yte, yhat)
        yte01  = (yte == 2).astype(int)
        score  = yhat_cont - 1.0
        auc = roc_auc_score(yte01, score)

        cm = confusion_matrix(yte, yhat, labels=[1,2])
        print(f"Fold Acc={acc:.3f}  AUC={auc:.3f}")
        print("Confusion matrix (rows=true [M,F], cols=pred [M,F]):")
        print(cm)

        fold_rows.append({
            "fold": fidx,
            "n_train": len(tr_idx),
            "n_test": len(te_idx),
            "n_prototypes": len(proto_idx),
            "gamma": used_gamma if used_gamma is not None else np.nan,
            "accuracy": float(acc),
            "auc": float(auc)
        })

        all_pred.extend(yhat.tolist())
        all_true.extend(yte.tolist())
        all_sid.extend([sids[i] for i in te_idx])

        # Save fold predictions
        fold_pred_path = os.path.join(args.outdir, f"predictions_fold{fidx}.csv")
        pd.DataFrame({
            "sid": [sids[i] for i in te_idx],
            "y_true": yte,
            "y_pred": yhat,
            "y_pred_cont": yhat_cont
        }).to_csv(fold_pred_path, index=False)

    # Summary
    df_folds = pd.DataFrame(fold_rows)
    df_folds.to_csv(os.path.join(args.outdir, "cv_summary_by_fold.csv"), index=False)

    mean_acc = df_folds["accuracy"].mean()
    std_acc  = df_folds["accuracy"].std()
    mean_auc = df_folds["auc"].mean()
    std_auc  = df_folds["auc"].std()

    print("\n=== Summary (Gender) ===")
    print(f"Accuracy per fold: {np.round(df_folds['accuracy'].values, 3)}")
    print(f"Mean Acc: {mean_acc:.3f} ± {std_acc:.3f}")
    print(f"AUC per fold: {np.round(df_folds['auc'].values, 3)}")
    print(f"Mean AUC: {mean_auc:.3f} ± {std_auc:.3f}")

    with open(os.path.join(args.outdir, "final_summary.txt"), "w") as f:
        f.write(f"Mean Acc: {mean_acc:.4f} ± {std_acc:.4f}\n")
        f.write(f"Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}\n")

    pd.DataFrame({"sid": all_sid, "y_true": all_true, "y_pred": all_pred}).to_csv(
        os.path.join(args.outdir, "predictions_all.csv"), index=False
    )

if __name__ == "__main__":
    main()
