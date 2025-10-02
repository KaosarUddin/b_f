# r2q2_peu_spd.py
# Controls vs AD — kNN with Pearson and Euclidean distances from spd_metrics_
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Import from your package
from spd_metrics_id.distance import pearson_distance, euclidean_distance


# ======================
# Utils
# ======================
def sym(A):
    return 0.5 * (A + A.T)

def sanitize_distance(D):
    """Ensure distances are valid for sklearn metric='precomputed'."""
    D = np.asarray(np.real(D), dtype=np.float64)
    if not np.isfinite(D).all():
        raise ValueError("Found NaN/Inf in distance matrix.")
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

    candidates = [
        Path(r"D:\Research AU\Alzheimer FCNs\Alzheimer FCNs\sec_adni_dc.mat"),
        Path(r"D:\Research AU\Alzheimer FCNs\Alzheimer FCNs\sfc_adni_dc.mat")
    ]
    for c in candidates:
        if c.exists():
            return c, sio.loadmat(str(c))

    tried = [str(p)] + [str(c) for c in candidates]
    raise FileNotFoundError("Could not find .mat. Tried:\n" + "\n".join(tried))

def detect_prefix(matdict):
    keys = set(matdict.keys())
    if "sec_adni_dc_controls1" in keys: return "sec"
    if "sfc_adni_dc_controls1" in keys: return "sfc"
    raise KeyError("Expected keys like sec_adni_dc_controls1 or sfc_adni_dc_controls1")

def build_controls_vs_ad(matdict, prefix):
    ctl_key = f"{prefix}_adni_dc_controls1"
    ad_key  = f"{prefix}_adni_dc_AD1"
    controls = matdict[ctl_key]
    ad       = matdict[ad_key]
    X, y = [], []
    for i in range(controls.shape[2]):
        X.append(sym(controls[:,:,i]))
        y.append("Controls")
    for i in range(ad.shape[2]):
        X.append(sym(ad[:,:,i]))
        y.append("AD")
    return np.asarray(X, dtype=np.float64), np.asarray(y)


# ======================
# Pairwise builders using spd_metrics_id
# ======================
def pairwise_distance_matrix(Xtr, Xte=None, metric="pearson"):
    """Compute pairwise distances using provided spd_metrics_id functions."""
    ntr = Xtr.shape[0]
    if metric == "pearson":
        fn = pearson_distance
    elif metric == "euclidean":
        fn = euclidean_distance
    else:
        raise ValueError("Unknown metric")

    if Xte is None:
        D = np.zeros((ntr, ntr))
        for i in range(ntr):
            for j in range(i+1, ntr):
                d = fn(Xtr[i], Xtr[j])
                D[i,j] = D[j,i] = float(d)
        return sanitize_distance(D)
    else:
        nte = Xte.shape[0]
        D = np.zeros((nte, ntr))
        for i in range(nte):
            for j in range(ntr):
                D[i,j] = float(fn(Xte[i], Xtr[j]))
        return sanitize_distance(D)


# ======================
# CV runner
# ======================
def run_knn_cv(X, y, metric="pearson", k=5, folds=5, seed=42):
    classes = ["Controls", "AD"]
    class_to_idx = {c:i for i,c in enumerate(classes)}
    y_idx = np.array([class_to_idx[c] for c in y])

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    accs, f1s = [], []
    cm_total = np.zeros((2,2), dtype=int)

    for tr, te in skf.split(np.zeros(len(y_idx)), y_idx):
        D_tr = pairwise_distance_matrix(X[tr], None, metric=metric)
        D_te = pairwise_distance_matrix(X[tr], X[te], metric=metric)

        knn = KNeighborsClassifier(n_neighbors=k, metric="precomputed")
        knn.fit(D_tr, y_idx[tr])
        y_pred = knn.predict(D_te)

        accs.append(accuracy_score(y_idx[te], y_pred))
        f1s.append(f1_score(y_idx[te], y_pred, average="macro"))
        cm_total += confusion_matrix(y_idx[te], y_pred, labels=[0,1])

    return {
        "classes": classes,
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs, ddof=1)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s, ddof=1)),
        "confusion_matrix": cm_total,
    }


# ======================
# Plot/save
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
                    help="Path to sec_adni_dc.mat or sfc_adni_dc.mat.")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--outdir", type=str, default="adni_peu_outputs_controls_vs_ad")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    if args.mat is None:
        mat_path, matdict = load_mat_safely("D:/Research AU/sec_adni_dc.mat")
    else:
        mat_path, matdict = load_mat_safely(args.mat)
    prefix = detect_prefix(matdict)
    print(f"Loaded: {mat_path} (detected {prefix.upper()})")

    X, y = build_controls_vs_ad(matdict, prefix)
    print(f"Subjects: {len(y)} | Shape: {X.shape}")

    # Pearson
    print("Running Pearson...")
    res_p = run_knn_cv(X, y, metric="pearson", k=args.k, folds=args.folds)
    # Euclidean
    print("Running Euclidean...")
    res_e = run_knn_cv(X, y, metric="euclidean", k=args.k, folds=args.folds)

    # Save CSV
    df = pd.DataFrame([
        {
            "Task": "Controls vs AD", "Dataset": prefix.upper(),
            "Metric": "Pearson (spd_metrics_id)",
            "CV Acc (mean)": res_p["accuracy_mean"],
            "CV Acc (std)": res_p["accuracy_std"],
            "CV Macro-F1 (mean)": res_p["macro_f1_mean"],
            "CV Macro-F1 (std)": res_p["macro_f1_std"],
        },
        {
            "Task": "Controls vs AD", "Dataset": prefix.upper(),
            "Metric": "Euclidean (spd_metrics_id)",
            "CV Acc (mean)": res_e["accuracy_mean"],
            "CV Acc (std)": res_e["accuracy_std"],
            "CV Macro-F1 (mean)": res_e["macro_f1_mean"],
            "CV Macro-F1 (std)": res_e["macro_f1_std"],
        }
    ])
    csv_path = outdir / f"pearson_euclidean_controls_vs_ad_{prefix}.csv"
    df.to_csv(csv_path, index=False)

    # Save confusion matrices
    png_p = outdir / f"pearson_controls_vs_ad_confusion_{prefix}.png"
    save_confusion(res_p["confusion_matrix"], res_p["classes"],
                   f"Controls vs AD – Pearson ({prefix.upper()})", png_p)
    png_e = outdir / f"euclidean_controls_vs_ad_confusion_{prefix}.png"
    save_confusion(res_e["confusion_matrix"], res_e["classes"],
                   f"Controls vs AD – Euclidean ({prefix.upper()})", png_e)

    print("Saved:", csv_path)
    print("Saved:", png_p)
    print("Saved:", png_e)


if __name__ == "__main__":
    main()
