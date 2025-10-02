#!/usr/bin/env python3
"""
CPM-style behavioural prediction with Alpha–Z edges (works for regression & classification)
-----------------------------------------------------------------------------
- Loads per-subject Alpha–Z edge vectors (e.g., flattened upper triangle of an SPD distance/affinity matrix)
- Runs CPM feature selection (edge-wise correlation with target, p<thr)
- Builds summary strengths (positive & negative sets)
- Predicts with Linear/Logistic Regression or SVR/SVC (RBF) using CV
- Optional permutation test

USAGE (examples):

  # Regression (e.g., age)
  python cpm_alphaZ_prediction.py \
      --features_dir D:/Research_AU/alphaZ_edges/ \
      --excel D:/Research_AU/100_Subj_Full_v3.xlsx \
      --id_col Subject \
      --target Age \
      --task regression \
      --p_thresh 0.01 \
      --model svr \
      --cv_folds 10 --cv_repeats 5 \
      --as_similarity rbf --gamma 0.1

  # Classification (e.g., Gender as {M,F} or {0,1})
  python cpm_alphaZ_prediction.py \
      --features_dir D:/Research_AU/alphaZ_edges/ \
      --excel D:/Research_AU/100_Subj_Full_v3.xlsx \
      --id_col Subject \
      --target Gender \
      --task classification \
      --p_thresh 0.01 \
      --model svc \
      --cv_folds 10 --cv_repeats 5 \
      --as_similarity rbf --gamma 0.1

NOTES
- This script treats your provided Alpha–Z values as *distances*. If you prefer to
  align with similarity-style CPM, set --as_similarity rbf (recommended) or --as_similarity negate.
- Expected feature files: one .npy per subject, containing a 1D edge vector of identical length across subjects.
  Filenames should start with the subject ID followed by an underscore, e.g., "100307_REST1_hemi_LR_D.npy".
  You can also provide an explicit manifest CSV (subject_id,file_path) with --manifest.

Author: ChatGPT (GPT-5 Thinking)
"""

import argparse
import os
import re
import sys
import json
import math
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict

from scipy.stats import pearsonr
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# I/O helpers
# -----------------------------

def discover_feature_files(features_dir: str, pattern: str = r"^(?P<sid>[^_]+)_.+\.npy$") -> Dict[str, str]:
    """Recursively collect .npy files and infer subject ID from filename prefix before first underscore.
    Returns dict mapping subject_id -> file_path.
    """
    rx = re.compile(pattern)
    mapping: Dict[str, str] = {}
    for root, _, files in os.walk(features_dir):
        for f in files:
            if f.lower().endswith('.npy'):
                m = rx.match(f)
                if m:
                    sid = str(m.group('sid'))
                    mapping[sid] = os.path.join(root, f)
    if not mapping:
        raise FileNotFoundError(f"No .npy features found in {features_dir} matching pattern {pattern}")
    return mapping


def load_manifest(manifest_csv: str, sid_col: str = 'subject_id', path_col: str = 'file_path') -> Dict[str, str]:
    df = pd.read_csv(manifest_csv)
    if sid_col not in df.columns or path_col not in df.columns:
        raise ValueError(f"Manifest must have columns: {sid_col},{path_col}")
    mapping = {str(r[sid_col]): str(r[path_col]) for _, r in df.iterrows()}
    return mapping


def load_edge_vector(path: str) -> np.ndarray:
    v = np.load(path)
    v = np.asarray(v).ravel()
    if not np.isfinite(v).all():
        v = np.nan_to_num(v, nan=0.0, posinf=np.nanmax(np.where(np.isfinite(v), v, 0.0)), neginf=np.nanmin(np.where(np.isfinite(v), v, 0.0)))
    return v


def align_subjects(feature_map: Dict[str, str], excel: str, id_col: str, target_col: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    meta = pd.read_excel(excel)
    if id_col not in meta.columns:
        raise ValueError(f"id_col '{id_col}' not in Excel columns: {list(meta.columns)}")
    if target_col not in meta.columns:
        raise ValueError(f"target '{target_col}' not in Excel columns: {list(meta.columns)}")

    # Keep only subjects present in both sources
    meta[id_col] = meta[id_col].astype(str)
    common_ids = [sid for sid in meta[id_col].tolist() if sid in feature_map]
    if len(common_ids) < 10:
        raise ValueError(f"Too few matched subjects ({len(common_ids)}). Check IDs/filenames.")

    # Build X matrix
    X_list = []
    y_list = []
    kept_ids = []
    for sid in common_ids:
        vec = load_edge_vector(feature_map[sid])
        yval = meta.loc[meta[id_col] == sid, target_col].values[0]
        if pd.isna(yval):
            continue
        X_list.append(vec)
        y_list.append(yval)
        kept_ids.append(sid)

    X = np.vstack(X_list)
    y = np.array(y_list)
    return X, y, kept_ids

# -----------------------------
# Feature transforms
# -----------------------------

def rbf_similarity_from_distance(D_vec: np.ndarray, gamma: float) -> np.ndarray:
    # elementwise RBF: exp(-gamma * d)
    return np.exp(-gamma * D_vec)


def prepare_features(X: np.ndarray, as_similarity: str = 'none', gamma: float = 0.1) -> np.ndarray:
    """Transform distance features to similarity if requested.
    as_similarity in {'none','negate','rbf'}
    - 'none': use distances as-is
    - 'negate': similarity = -distance
    - 'rbf': similarity = exp(-gamma * distance)
    """
    if as_similarity == 'none':
        return X
    if as_similarity == 'negate':
        return -X
    if as_similarity == 'rbf':
        if gamma <= 0:
            raise ValueError("gamma must be > 0 for RBF")
        return np.exp(-gamma * X)
    raise ValueError("as_similarity must be one of {'none','negate','rbf'}")

# -----------------------------
# CPM feature selection
# -----------------------------

def cpm_select_edges(X_tr: np.ndarray, y_tr: np.ndarray, p_thresh: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Edge-wise Pearson correlation with y (continuous or binary 0/1).
    Returns: (pos_idx, neg_idx, r_values)
    """
    n_edges = X_tr.shape[1]
    rvals = np.zeros(n_edges, dtype=float)
    pvals = np.ones(n_edges, dtype=float)

    # Compute correlation per edge (vectorized loops for clarity)
    for j in range(n_edges):
        r, p = pearsonr(X_tr[:, j], y_tr)
        rvals[j] = r
        pvals[j] = p
    # Significant edges
    sig = pvals < p_thresh
    pos_idx = np.where((rvals > 0) & sig)[0]
    neg_idx = np.where((rvals < 0) & sig)[0]
    return pos_idx, neg_idx, rvals


def summarize_strengths(X: np.ndarray, pos_idx: np.ndarray, neg_idx: np.ndarray) -> np.ndarray:
    """Return 2-column matrix: [sum_pos, sum_neg] per subject."""
    if pos_idx.size == 0 and neg_idx.size == 0:
        # No edges selected -> zeros
        return np.zeros((X.shape[0], 2), dtype=float)
    sum_pos = X[:, pos_idx].sum(axis=1) if pos_idx.size else np.zeros(X.shape[0])
    sum_neg = X[:, neg_idx].sum(axis=1) if neg_idx.size else np.zeros(X.shape[0])
    return np.column_stack([sum_pos, sum_neg])

# -----------------------------
# Modeling
# -----------------------------

def fit_predict_regression(Z_tr: np.ndarray, y_tr: np.ndarray, Z_te: np.ndarray, model: str = 'linear') -> np.ndarray:
    if model == 'linear':
        reg = LinearRegression()
    elif model == 'svr':
        reg = SVR(kernel='rbf', C=1.0, gamma='scale')
    else:
        raise ValueError("Regression model must be 'linear' or 'svr'")
    reg.fit(Z_tr, y_tr)
    return reg.predict(Z_te)


def fit_predict_classification(Z_tr: np.ndarray, y_tr: np.ndarray, Z_te: np.ndarray, model: str = 'logreg') -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if model == 'logreg':
        clf = LogisticRegression(max_iter=200, solver='lbfgs')
    elif model == 'svc':
        clf = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    else:
        raise ValueError("Classification model must be 'logreg' or 'svc'")
    clf.fit(Z_tr, y_tr)
    y_pred = clf.predict(Z_te)
    y_proba = None
    if hasattr(clf, 'predict_proba'):
        y_proba = clf.predict_proba(Z_te)[:, 1]
    return y_pred, y_proba

# -----------------------------
# Cross-validation pipeline
# -----------------------------

def run_cpm(X: np.ndarray, y: np.ndarray, task: str, p_thresh: float, model: str,
            cv_folds: int, cv_repeats: int, random_state: int = 42) -> Dict:
    n, p = X.shape

    if task == 'classification':
        # Encode labels to 0/1
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        splitter = RepeatedStratifiedKFold(n_splits=cv_folds, n_repeats=cv_repeats, random_state=random_state)
        accs = []
        aucs = []
        n_pos_edges = []
        n_neg_edges = []
        y_true_all = []
        y_pred_all = []
        for tr_idx, te_idx in splitter.split(X, y_enc):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y_enc[tr_idx], y_enc[te_idx]

            pos_idx, neg_idx, _ = cpm_select_edges(X_tr, y_tr, p_thresh=p_thresh)
            Z_tr = summarize_strengths(X_tr, pos_idx, neg_idx)
            Z_te = summarize_strengths(X_te, pos_idx, neg_idx)

            y_hat, y_proba = fit_predict_classification(Z_tr, y_tr, Z_te, model=model)
            accs.append(accuracy_score(y_te, y_hat))
            if y_proba is not None and len(np.unique(y_enc)) == 2:
                aucs.append(roc_auc_score(y_te, y_proba))
            n_pos_edges.append(len(pos_idx))
            n_neg_edges.append(len(neg_idx))
            y_true_all.extend(y_te.tolist())
            y_pred_all.extend(y_hat.tolist())

        out = {
            'task': task,
            'metric_mean_acc': float(np.mean(accs)),
            'metric_std_acc': float(np.std(accs)),
            'metric_mean_auc': float(np.mean(aucs)) if aucs else None,
            'edges_pos_mean': float(np.mean(n_pos_edges)),
            'edges_neg_mean': float(np.mean(n_neg_edges)),
            'n_subjects': int(n),
            'n_edges': int(p),
        }
        return out

    elif task == 'regression':
        splitter = RepeatedKFold(n_splits=cv_folds, n_repeats=cv_repeats, random_state=random_state)
        r2s = []
        corrs = []
        n_pos_edges = []
        n_neg_edges = []
        for tr_idx, te_idx in splitter.split(X):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            pos_idx, neg_idx, _ = cpm_select_edges(X_tr, y_tr, p_thresh=p_thresh)
            Z_tr = summarize_strengths(X_tr, pos_idx, neg_idx)
            Z_te = summarize_strengths(X_te, pos_idx, neg_idx)

            y_hat = fit_predict_regression(Z_tr, y_tr, Z_te, model=model)
            r2s.append(r2_score(y_te, y_hat))
            # Pearson correlation between y_te and y_hat
            r, _ = pearsonr(y_te, y_hat)
            corrs.append(r)
            n_pos_edges.append(len(pos_idx))
            n_neg_edges.append(len(neg_idx))

        out = {
            'task': task,
            'metric_mean_r': float(np.mean(corrs)),
            'metric_std_r': float(np.std(corrs)),
            'metric_mean_r2': float(np.mean(r2s)),
            'metric_std_r2': float(np.std(r2s)),
            'edges_pos_mean': float(np.mean(n_pos_edges)),
            'edges_neg_mean': float(np.mean(n_neg_edges)),
            'n_subjects': int(n),
            'n_edges': int(p),
        }
        return out

    else:
        raise ValueError("task must be 'classification' or 'regression'")

# -----------------------------
# Permutation testing
# -----------------------------

def permutation_test(X: np.ndarray, y: np.ndarray, task: str, p_thresh: float, model: str,
                      cv_folds: int, cv_repeats: int, n_perm: int, random_state: int = 42) -> Dict:
    rng = np.random.default_rng(random_state)

    if task == 'classification':
        base = run_cpm(X, y, task, p_thresh, model, cv_folds, cv_repeats, random_state)
        base_metric = base['metric_mean_acc']
        null = []
        for b in range(n_perm):
            y_perm = rng.permutation(y)
            res = run_cpm(X, y_perm, task, p_thresh, model, cv_folds, cv_repeats, random_state + b + 1)
            null.append(res['metric_mean_acc'])
        pval = (np.sum(np.array(null) >= base_metric) + 1) / (n_perm + 1)
        base['perm_p_acc'] = float(pval)
        base['perm_null_acc'] = null
        return base

    else:  # regression
        base = run_cpm(X, y, task, p_thresh, model, cv_folds, cv_repeats, random_state)
        base_metric = base['metric_mean_r']
        null = []
        for b in range(n_perm):
            y_perm = np.random.default_rng(random_state + b + 1).permutation(y)
            res = run_cpm(X, y_perm, task, p_thresh, model, cv_folds, cv_repeats, random_state + b + 1)
            null.append(res['metric_mean_r'])
        pval = (np.sum(np.array(null) >= base_metric) + 1) / (n_perm + 1)
        base['perm_p_r'] = float(pval)
        base['perm_null_r'] = null
        return base

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description='CPM with Alpha–Z edges (regression & classification)')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--features_dir', type=str, help='Directory containing per-subject .npy edge vectors')
    src.add_argument('--manifest', type=str, help='CSV with columns subject_id,file_path')

    ap.add_argument('--excel', type=str, required=True, help='Excel file with subject metadata (traits/labels)')
    ap.add_argument('--id_col', type=str, required=True, help='Column name for subject ID in Excel')
    ap.add_argument('--target', type=str, required=True, help='Target column (trait/age/gender) in Excel')

    ap.add_argument('--task', type=str, choices=['regression','classification'], required=True)
    ap.add_argument('--p_thresh', type=float, default=0.01, help='CPM edge selection p-value threshold')
    ap.add_argument('--model', type=str, help='Model type', default=None)

    ap.add_argument('--cv_folds', type=int, default=10)
    ap.add_argument('--cv_repeats', type=int, default=5)
    ap.add_argument('--random_state', type=int, default=42)

    ap.add_argument('--as_similarity', type=str, choices=['none','negate','rbf'], default='rbf', help='Transform distances to similarity')
    ap.add_argument('--gamma', type=float, default=0.1, help='RBF gamma if --as_similarity rbf')

    ap.add_argument('--n_perm', type=int, default=0, help='Permutation test iterations (0 to skip)')
    ap.add_argument('--save_json', type=str, default=None, help='Path to save results JSON')

    args = ap.parse_args()

    # Feature files mapping
    if args.features_dir:
        fmap = discover_feature_files(args.features_dir)
    else:
        fmap = load_manifest(args.manifest)

    # Align with metadata/labels
    X, y, kept_ids = align_subjects(fmap, args.excel, args.id_col, args.target)

    # Choose default model if not provided
    if args.model is None:
        args.model = 'svr' if args.task == 'regression' else 'svc'

    # Transform features (distances -> similarity optional)
    Xp = prepare_features(X, as_similarity=args.as_similarity, gamma=args.gamma)

    # Run CV (and permutation if requested)
    if args.n_perm and args.n_perm > 0:
        results = permutation_test(Xp, y, args.task, args.p_thresh, args.model, args.cv_folds, args.cv_repeats, args.n_perm, args.random_state)
    else:
        results = run_cpm(Xp, y, args.task, args.p_thresh, args.model, args.cv_folds, args.cv_repeats, args.random_state)

    results['as_similarity'] = args.as_similarity
    results['gamma'] = args.gamma
    results['p_thresh'] = args.p_thresh
    results['model'] = args.model
    results['target'] = args.target
    results['n_perm'] = args.n_perm

    print(json.dumps(results, indent=2))

    if args.save_json:
        with open(args.save_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {args.save_json}")


if __name__ == '__main__':
    main()
