# r2q2.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.linalg import eigh, logm
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Alpha-Z divergence (your package)
from spd_metrics_id.distance import alpha_z_bw  # pip install spd-metrics-id


# ======================
# Utils
# ======================
def sym(A):
    return 0.5 * (A + A.T)

def make_spd(A, eps=1e-5):
    """Symmetrize and eigen-floor to ensure SPD (guard against numerics)."""
    A = sym(A)
    w, V = eigh(A)
    w = np.maximum(w, eps)
    return (V * w) @ V.T

def sanitize_distance(D):
    """
    Ensure distances are real, finite, non-negative; symmetrize if square; zero diagonal.
    Sklearn's precomputed metric requires non-negative numbers.
    """
    D = np.asarray(np.real(D), dtype=np.float64)
    if not np.isfinite(D).all():
        raise ValueError("Found NaN/Inf in distance matrix; check inputs/SPD step.")
    D[D < 0] = 0.0
    if D.shape[0] == D.shape[1]:
        D = 0.5 * (D + D.T)
        np.fill_diagonal(D, 0.0)
    return D


# ======================
# Dataset loading
# ======================
def load_mat_safely(path_like: str | Path):
    p = Path(path_like)
    if p.exists():
        return p, sio.loadmat(str(p))

    # Try common defaults if user passed a folder or wrong file
    candidates = []
    if p.is_dir():
        candidates = [p / "sec_adni_dc.mat", p / "sfc_adni_dc.mat"]
    else:
        candidates = [Path(r"D:\Research AU\Alzheimer FCNs\Alzheimer FCNs\sec_adni_dc.mat"),
                      Path(r"D:\Research AU\Alzheimer FCNs\Alzheimer FCNs\sfc_adni_dc.mat")]

    for c in candidates:
        if c.exists():
            return c, sio.loadmat(str(c))

    tried = [str(p)] + [str(c) for c in candidates]
    raise FileNotFoundError(
        "Could not find the .mat file. Tried:\n  - " + "\n  - ".join(tried) +
        '\nTip: run with --mat "D:\\Research AU\\sec_adni_dc.mat" (or sfc).'
    )

def detect_prefix(matdict: dict) -> str:
    keys = set(matdict.keys())
    if "sec_adni_dc_controls1" in keys:
        return "sec"
    if "sfc_adni_dc_controls1" in keys:
        return "sfc"
    raise KeyError(
        "Unrecognized .mat structure. Expected keys like "
        "'sec_adni_dc_controls1' or 'sfc_adni_dc_controls1'."
    )

def build_controls_vs_ad(matdict: dict, prefix: str):
    ctl_key = f"{prefix}_adni_dc_controls1"
    ad_key  = f"{prefix}_adni_dc_AD1"
    controls = matdict[ctl_key]   # (200,200,Nc)
    ad       = matdict[ad_key]    # (200,200,Na)
    X, y = [], []
    for i in range(controls.shape[2]):
        X.append(sym(controls[:, :, i]))
        y.append("Controls")
    for i in range(ad.shape[2]):
        X.append(sym(ad[:, :, i]))
        y.append("AD")
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    return X, y


# ======================
# Distances
# ======================
def pairwise_alphaZ_train_test(Xtr, Xte, alpha=0.99, z=1.0, alpha_z_func=None):
    """
    Build train-vs-train (square) and test-vs-train (rectangular) Alpha-Z distance matrices.
    Xtr, Xte are SPD matrices (Ntr, 200, 200) / (Nte, 200, 200).
    """
    assert alpha_z_func is not None, "Provide alpha_z_func=alpha_z_bw"
    ntr, nte = Xtr.shape[0], Xte.shape[0]

    D_train = np.zeros((ntr, ntr), dtype=np.float64)
    for i in range(ntr):
        for j in range(i + 1, ntr):
            d = float(np.real(alpha_z_func(Xtr[i], Xtr[j], alpha=alpha, z=z)))
            D_train[i, j] = D_train[j, i] = d

    D_test = np.zeros((nte, ntr), dtype=np.float64)
    for i in range(nte):
        for j in range(ntr):
            D_test[i, j] = float(np.real(alpha_z_func(Xte[i], Xtr[j], alpha=alpha, z=z)))

    return D_train, D_test

def pairwise_corr_distance(train_vecs, test_vecs=None):
    """Pearson distance = 1 - corr between row-vectors."""
    def _corr(M1, M2=None):
        M1c = M1 - M1.mean(axis=1, keepdims=True)
        if M2 is None:
            M2c = M1c
        else:
            M2c = M2 - M2.mean(axis=1, keepdims=True)
        G = M1c @ M2c.T
        n1 = np.sqrt((M1c * M1c).sum(axis=1, keepdims=True))
        n2 = np.sqrt((M2c * M2c).sum(axis=1, keepdims=True)) if M2 is not None else n1
        denom = n1 @ n2.T
        denom[denom == 0] = 1e-12
        return G / denom

    if test_vecs is None:
        C = _corr(train_vecs, None)
        D = 1.0 - C
        np.fill_diagonal(D, 0.0)
        return D
    else:
        C = _corr(test_vecs, train_vecs)
        D = 1.0 - C
        return D


# ======================
# CV runners
# ======================
def run_knn_cv_alphaZ(X, y, alpha=0.5, z=.5, k=5, folds=5, seed=42, alpha_z_func=None):
    """
    X: (N, 200, 200) subject matrices
    y: array of labels ('Controls'/'AD')
    """
    assert alpha_z_func is not None, "Provide alpha_z_func=alpha_z_bw"
    classes = ["Controls", "AD"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx[c] for c in y])

    # SPD-ize once (faster & stable)
    X_spd = np.empty_like(X)
    for i in range(X.shape[0]):
        X_spd[i] = make_spd(X[i], eps=1e-5)

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    accs, f1s = [], []
    cm_total = np.zeros((2, 2), dtype=int)

    for tr, te in skf.split(np.zeros(len(y_idx)), y_idx):
        Xtr, Xte = X_spd[tr], X_spd[te]

        D_tr, D_te = pairwise_alphaZ_train_test(
            Xtr, Xte, alpha=alpha, z=z, alpha_z_func=alpha_z_func
        )

        D_tr = sanitize_distance(D_tr)
        D_te = sanitize_distance(D_te)

        knn = KNeighborsClassifier(n_neighbors=k, metric="precomputed")
        knn.fit(D_tr, y_idx[tr])
        y_pred = knn.predict(D_te)

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

def run_knn_cv_pearson(X, y, k=5, folds=5, seed=42):
    """Optional fast baseline: Pearson distance on vectorized upper triangles."""
    classes = ["Controls", "AD"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx[c] for c in y])

    iu = np.triu_indices(X.shape[1], k=1)
    V = X[:, iu[0], iu[1]]

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    accs, f1s = [], []
    cm_total = np.zeros((2, 2), dtype=int)

    for tr, te in skf.split(np.zeros(len(y_idx)), y_idx):
        D_tr = pairwise_corr_distance(V[tr], None)
        D_te = pairwise_corr_distance(V[tr], V[te])

        D_tr = sanitize_distance(D_tr)
        D_te = sanitize_distance(D_te)

        knn = KNeighborsClassifier(n_neighbors=k, metric="precomputed")
        knn.fit(D_tr, y_idx[tr])
        y_pred = knn.predict(D_te)

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


# ======================
# Plot/save helpers
# ======================
def save_confusion(cm, classes, title, path_png):
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
# Main
# ======================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", type=str, default=None,
                    help="Path to sec_adni_dc.mat or sfc_adni_dc.mat. If omitted, common paths are tried.")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--z", type=float, default=.5)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--outdir", type=str, default="adni_alphaZ_outputs_controls_vs_ad")
    ap.add_argument("--with_pearson", action="store_true",
                    help="Also run Pearson baseline side-by-side.")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load .mat (robust)
    if args.mat is None:
        mat_path, matdict = load_mat_safely("D:/Research AU/Alzheimer FCNs/Alzheimer FCNs/sec_adni_dc.mat")
    else:
        mat_path, matdict = load_mat_safely(args.mat)

    prefix = detect_prefix(matdict)  # 'sec' or 'sfc'
    print(f"Loaded: {mat_path}   (detected dataset type: {prefix.upper()})")

    # Build Controls vs AD dataset
    X, y = build_controls_vs_ad(matdict, prefix)
    print(f"Subjects: {len(y)}  |  Shapes: {X.shape}  (Controls={np.sum(y=='Controls')}, AD={np.sum(y=='AD')})")

    # === Alpha-Z CV ===
    print(f"Running Alpha-Z (alpha={args.alpha}, z={args.z}), k={args.k}, folds={args.folds} ...")
    res_az = run_knn_cv_alphaZ(
        X, y,
        alpha=args.alpha, z=args.z,
        k=args.k, folds=args.folds, seed=42,
        alpha_z_func=alpha_z_bw
    )

    df = pd.DataFrame([{
        "Task": "Controls vs AD",
        "Dataset": prefix.upper(),
        "Metric": f"Alpha-Z (alpha={args.alpha}, z={args.z})",
        "CV Acc (mean)": res_az["accuracy_mean"],
        "CV Acc (std)":  res_az["accuracy_std"],
        "CV Macro-F1 (mean)": res_az["macro_f1_mean"],
        "CV Macro-F1 (std)":  res_az["macro_f1_std"],
    }])

    # Save Alpha-Z outputs
    csv_path_az = outdir / f"alphaZk_controls_vs_ad_summary_{prefix}.csv"
    df.to_csv(csv_path_az, index=False)
    png_path_az = outdir / f"alphaZk_controls_vs_ad_confusion_{prefix}.png"
    save_confusion(res_az["confusion_matrix"], res_az["classes"],
                   f"Controls vs AD – Alpha-Z ({prefix.upper()})",
                   png_path_az)

    print("Alpha-Z saved:", csv_path_az)
    print("Alpha-Z saved:", png_path_az)

    # === Optional Pearson baseline ===
    if args.with_pearson:
        print(f"Running Pearson baseline, k={args.k}, folds={args.folds} ...")
        res_p = run_knn_cv_pearson(X, y, k=args.k, folds=args.folds, seed=42)
        df_p = pd.DataFrame([{
            "Task": "Controls vs AD",
            "Dataset": prefix.upper(),
            "Metric": "Pearson (1 - corr)",
            "CV Acc (mean)": res_p["accuracy_mean"],
            "CV Acc (std)":  res_p["accuracy_std"],
            "CV Macro-F1 (mean)": res_p["macro_f1_mean"],
            "CV Macro-F1 (std)":  res_p["macro_f1_std"],
        }])
        df_all = pd.concat([df, df_p], ignore_index=True)
        csv_path_both = outdir / f"alphaZ_and_pearson_controls_vs_ad_{prefix}.csv"
        df_all.to_csv(csv_path_both, index=False)

        png_path_p = outdir / f"pearson_controls_vs_ad_confusion_{prefix}.png"
        save_confusion(res_p["confusion_matrix"], res_p["classes"],
                       f"Controls vs AD – Pearson ({prefix.upper()})",
                       png_path_p)

        print("Pearson saved:", csv_path_both)
        print("Pearson saved:", png_path_p)

    print("\nDone.")


if __name__ == "__main__":
    main()

# %%
import pandas as pd

demo_path = r"D:\Research AU\Alzheimer FCNs\Alzheimer FCNs\ADNI_subjects_infos_ForRestingStateFMRI.xls"
df = pd.read_excel(demo_path, engine="xlrd")

print(df.shape)
print(df.columns)
print(df.head())
# %%
