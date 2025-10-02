# rbf_alphaZ_svm_svr.py
# Controls vs AD (classification) or continuous target (regression) using SVM/SVR with an RBF kernel
# built from Alpha-Z divergence between SPD connectivity matrices.
#
# - Classification (default): uses labels Controls vs AD from the ADNI .mat
# - Regression (optional): pass --task svr and a CSV with a column of targets via --targets
#
# Outputs:
#   - CSV with CV metrics
#   - Confusion matrix PNG (classification)
#
# Requirements:
#   pip install scikit-learn scipy numpy pandas matplotlib
#   pip install spd-metrics-id  # provides alpha_z_bw

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.linalg import eigh
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
import matplotlib.pyplot as plt

# Alpha-Z (your package)
from spd_metrics_id.distance import alpha_z_bw


# ======================
# Utils
# ======================
def sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)

def make_spd(A: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Symmetrize and eigen-floor to ensure SPD (numerical stability)."""
    A = sym(A)
    w, V = eigh(A)
    w = np.maximum(w, eps)
    return (V * w) @ V.T

def sanitize_distance(D: np.ndarray) -> np.ndarray:
    """Ensure distances are real, finite, non-negative; symmetrize if square; zero diagonal."""
    D = np.asarray(np.real(D), dtype=np.float64)
    if not np.isfinite(D).all():
        raise ValueError("Found NaN/Inf in distance matrix; check inputs/SPD step.")
    D[D < 0] = 0.0
    if D.shape[0] == D.shape[1]:
        D = 0.5 * (D + D.T)
        np.fill_diagonal(D, 0.0)
    return D

def rbf_kernel_from_distance(D: np.ndarray, gamma: float) -> np.ndarray:
    """Build RBF kernel K = exp(-gamma * D^2) from a (possibly rectangular) distance matrix."""
    return np.exp(-gamma * (D ** 2))

def save_confusion(cm: np.ndarray, classes: list[str], title: str, path_png: Path) -> None:
    plt.figure(figsize=(5.5, 4.5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45, ha="right")
    plt.yticks(ticks, classes)
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(path_png, bbox_inches="tight")
    plt.close()


# ======================
# Data loading (AD vs Controls)
# ======================
def load_mat_safely(path_like: str | Path):
    p = Path(path_like)
    if p.exists():
        return p, sio.loadmat(str(p))

    # Try common defaults if path not found
    candidates = [

        Path(r"D:\Research AU\Alzheimer FCNs\Alzheimer FCNs\sec_adni_dc.mat"),
        Path(r"D:\Research AU\Alzheimer FCNs\Alzheimer FCNs\sfc_adni_dc.mat"),
    ]
    for c in candidates:
        if c.exists():
            return c, sio.loadmat(str(c))

    tried = [str(p)] + [str(c) for c in candidates]
    raise FileNotFoundError(
        "Could not find the .mat file. Tried:\n  - " + "\n  - ".join(tried)
    )

def detect_prefix(matdict: dict) -> str:
    keys = set(matdict.keys())
    if "sec_adni_dc_controls1" in keys: return "sec"
    if "sfc_adni_dc_controls1" in keys: return "sfc"
    raise KeyError("Unrecognized .mat structure (expected sec_adni_dc_* or sfc_adni_dc_* keys).")

def build_controls_vs_ad(matdict: dict, prefix: str):
    ctl_key = f"{prefix}_adni_dc_controls1"
    ad_key  = f"{prefix}_adni_dc_AD1"
    controls = matdict[ctl_key]   # (P,P,Nc)
    ad       = matdict[ad_key]    # (P,P,Na)
    X, y = [], []
    for i in range(controls.shape[2]):
        X.append(sym(controls[:, :, i])); y.append("Controls")
    for i in range(ad.shape[2]):
        X.append(sym(ad[:, :, i])); y.append("AD")
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    return X, y


# ======================
# Alpha-Z pairwise distances
# ======================
def alphaZ_pairwise_train_test(Xtr: np.ndarray, Xte: np.ndarray | None, alpha: float, z: float):
    """
    Compute Alpha-Z distances:
      - if Xte is None: returns D_train (ntr x ntr)
      - else: returns D_test (nte x ntr)
    Xtr/Xte must already be SPD.
    """
    if Xte is None:
        ntr = Xtr.shape[0]
        D = np.zeros((ntr, ntr), dtype=np.float64)
        for i in range(ntr):
            for j in range(i + 1, ntr):
                d = float(np.real(alpha_z_bw(Xtr[i], Xtr[j], alpha=alpha, z=z)))
                D[i, j] = D[j, i] = d
        return sanitize_distance(D)
    else:
        nte, ntr = Xte.shape[0], Xtr.shape[0]
        D = np.zeros((nte, ntr), dtype=np.float64)
        for i in range(nte):
            for j in range(ntr):
                D[i, j] = float(np.real(alpha_z_bw(Xte[i], Xtr[j], alpha=alpha, z=z)))
        return sanitize_distance(D)


# ======================
# CV runners (SVM/SVR with RBF kernel built from Alpha-Z distances)
# ======================
def run_svm_alphaZ_rbf(X: np.ndarray, y_cls: np.ndarray,
                       alpha: float, z: float,
                       C: float = 100, gamma: float = 2,
                       folds: int = 5, seed: int = 42):
    """
    Classification: SVM (RBF on Alpha-Z distances).
    Returns accuracy/F1 (macro) and aggregated confusion matrix.
    """
    classes = ["Controls", "AD"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx[c] for c in y_cls])

    # SPD once
    X_spd = np.empty_like(X)
    for i in range(X.shape[0]):
        X_spd[i] = make_spd(X[i], eps=1e-5)

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    accs, f1s = [], []
    cm_total = np.zeros((2, 2), dtype=int)

    for tr, te in skf.split(np.zeros(len(y_idx)), y_idx):
        Xtr, Xte = X_spd[tr], X_spd[te]

        # Alpha-Z distances
        D_tr = alphaZ_pairwise_train_test(Xtr, None, alpha=alpha, z=z)
        D_te = alphaZ_pairwise_train_test(Xtr, Xte, alpha=alpha, z=z)

        # RBF kernel from distances
        K_tr = rbf_kernel_from_distance(D_tr, gamma=gamma)
        K_te = rbf_kernel_from_distance(D_te, gamma=gamma)

        # SVM with precomputed kernel
        clf = SVC(C=C, kernel="precomputed", probability=False)
        clf.fit(K_tr, y_idx[tr])
        y_pred = clf.predict(K_te)

        accs.append(accuracy_score(y_idx[te], y_pred))
        f1s.append(f1_score(y_idx[te], y_pred, average="macro"))
        cm_total += confusion_matrix(y_idx[te], y_pred, labels=[0, 1])

    return {
        "classes": classes,
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs, ddof=1)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s, ddof=1)),
        "confusion_matrix": cm_total,
    }

def run_svr_alphaZ_rbf(X: np.ndarray, y_reg: np.ndarray,
                       alpha: float, z: float,
                       C: float = 100, epsilon: float = 0.1, gamma: float = 2,
                       folds: int = 5, seed: int = 42):
    """
    Regression: SVR (RBF on Alpha-Z distances).
    Returns MAE/MSE/R2 averaged across folds.
    """
    # SPD once
    X_spd = np.empty_like(X)
    for i in range(X.shape[0]):
        X_spd[i] = make_spd(X[i], eps=1e-5)

    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    maes, mses, r2s = [], [], []

    for tr, te in kf.split(np.arange(len(y_reg))):
        Xtr, Xte = X_spd[tr], X_spd[te]

        # Alpha-Z distances
        D_tr = alphaZ_pairwise_train_test(Xtr, None, alpha=alpha, z=z)
        D_te = alphaZ_pairwise_train_test(Xtr, Xte, alpha=alpha, z=z)

        # RBF kernel
        K_tr = rbf_kernel_from_distance(D_tr, gamma=gamma)
        K_te = rbf_kernel_from_distance(D_te, gamma=gamma)

        # SVR with precomputed kernel
        svr = SVR(C=C, epsilon=epsilon, kernel="precomputed")
        svr.fit(K_tr, y_reg[tr])
        y_hat = svr.predict(K_te)

        maes.append(mean_absolute_error(y_reg[te], y_hat))
        mses.append(mean_squared_error(y_reg[te], y_hat))
        r2s.append(r2_score(y_reg[te], y_hat))

    return {
        "MAE_mean": float(np.mean(maes)),
        "MAE_std":  float(np.std(maes, ddof=1)),
        "MSE_mean": float(np.mean(mses)),
        "MSE_std":  float(np.std(mses, ddof=1)),
        "R2_mean":  float(np.mean(r2s)),
        "R2_std":   float(np.std(r2s, ddof=1)),
    }


# ======================
# CLI
# ======================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", type=str, default=None,
                    help="Path to sec_adni_dc.mat or sfc_adni_dc.mat (Controls vs AD).")
    ap.add_argument("--task", type=str, choices=["svc", "svr"], default="svc",
                    help="svc = classification (Controls vs AD); svr = regression (requires --targets).")
    ap.add_argument("--targets", type=str, default=None,
                    help="CSV with a single numeric target column (required for --task svr).")
    ap.add_argument("--alpha", type=float, default=0.99)
    ap.add_argument("--z", type=float, default=1.0)
    ap.add_argument("--C", type=float, default=100, help="Regularization for SVC/SVR.")
    ap.add_argument("--gamma", type=float, default=2, help="RBF gamma applied to D^2.")
    ap.add_argument("--epsilon", type=float, default=0.1, help="SVR epsilon-insensitive tube.")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--outdir", type=str, default="adni_alphaZ_rbf_outputs")
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    if args.mat is None:
        mat_path, matdict = load_mat_safely("D:/Research AU/Alzheimer FCNs/Alzheimer FCNs/sec_adni_dc.mat")
    else:
        mat_path, matdict = load_mat_safely(args.mat)
    prefix = detect_prefix(matdict)
    print(f"Loaded: {mat_path}   (detected {prefix.upper()})")

    # Build matrices/labels for Controls vs AD
    X, y_cls = build_controls_vs_ad(matdict, prefix)
    print(f"Subjects: {len(y_cls)} | Shape: {X.shape}  (Controls={np.sum(y_cls=='Controls')}, AD={np.sum(y_cls=='AD')})")

    if args.task == "svc":
        print(f"[SVC] Alpha-Z RBF | C={args.C}, gamma={args.gamma}, folds={args.folds}, alpha={args.alpha}, z={args.z}")
        res = run_svm_alphaZ_rbf(
            X, y_cls,
            alpha=args.alpha, z=args.z,
            C=args.C, gamma=args.gamma,
            folds=args.folds, seed=42
        )
        # Save CSV
        df = pd.DataFrame([{
            "Task": "Controls vs AD",
            "Dataset": prefix.upper(),
            "Model": "SVM (RBF on Alpha-Z)",
            "alpha": args.alpha, "z": args.z,
            "C": args.C, "gamma": args.gamma,
            "CV Acc (mean)": res["accuracy_mean"],
            "CV Acc (std)":  res["accuracy_std"],
            "CV Macro-F1 (mean)": res["macro_f1_mean"],
            "CV Macro-F1 (std)":  res["macro_f1_std"],
        }])
        csv_path = outdir / f"svm_alphaZ_controls_vs_ad_{prefix}.csv"
        df.to_csv(csv_path, index=False)
        # Save confusion matrix
        png_path = outdir / f"svm_alphaZ_controls_vs_ad_confusion_{prefix}.png"
        save_confusion(res["confusion_matrix"], res["classes"],
                       f"Controls vs AD – SVM RBF (Alpha-Z) [{prefix.upper()}]",
                       png_path)
        print("Saved:", csv_path)
        print("Saved:", png_path)

    else:  # svr
        if args.targets is None:
            raise ValueError("For --task svr you must provide --targets CSV with one numeric column.")
        # Load targets
        tpath = Path(args.targets)
        if not tpath.exists():
            raise FileNotFoundError(f"Targets CSV not found: {tpath}")
        tdf = pd.read_csv(tpath)
        if tdf.shape[1] != 1:
            raise ValueError("Targets CSV must have exactly one numeric column.")
        y_reg = tdf.iloc[:, 0].to_numpy(dtype=float)
        if y_reg.shape[0] != X.shape[0]:
            raise ValueError(f"Targets length ({y_reg.shape[0]}) does not match number of subjects ({X.shape[0]}).")

        print(f"[SVR] Alpha-Z RBF | C={args.C}, eps={args.epsilon}, gamma={args.gamma}, folds={args.folds}, alpha={args.alpha}, z={args.z}")
        res = run_svr_alphaZ_rbf(
            X, y_reg,
            alpha=args.alpha, z=args.z,
            C=args.C, epsilon=args.epsilon, gamma=args.gamma,
            folds=args.folds, seed=42
        )
        # Save CSV
        df = pd.DataFrame([{
            "Task": "SVR (continuous)",
            "Dataset": prefix.upper(),
            "Model": "SVR (RBF on Alpha-Z)",
            "alpha": args.alpha, "z": args.z,
            "C": args.C, "epsilon": args.epsilon, "gamma": args.gamma,
            "CV MAE (mean)": res["MAE_mean"], "CV MAE (std)": res["MAE_std"],
            "CV MSE (mean)": res["MSE_mean"], "CV MSE (std)": res["MSE_std"],
            "CV R2  (mean)": res["R2_mean"],  "CV R2  (std)": res["R2_std"],
        }])
        csv_path = outdir / f"svr_alphaZ_{prefix}.csv"
        df.to_csv(csv_path, index=False)
        print("Saved:", csv_path)

    print("Done.")


if __name__ == "__main__":
    main()
