#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dahan-style spatio–temporal GCN for fMRI with Alpha–Z graphs
------------------------------------------------------------
Inputs per subject:
  - Time series matrix (T_total x N) from REST1 (per subject file)
  - Alpha–Z adjacency Ahat.npy of shape (N x N)

Pipeline:
  - Build sliding windows from time series
  - Lightweight MS-G3D-inspired model: spatial graph op + temporal depthwise conv blocks
  - 5-fold subject-level CV; mini-batch training; GPU if available
  - Task: classification (sex) by default; can switch to regression

Outputs (saved under outdir):
  - Per-fold predictions CSV
  - CV summary-by-fold CSV
  - Final summary text

Defaults are set for your server. Override with CLI flags if needed.
"""

from __future__ import print_function

import os
import glob
import math
import random
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# -----------------------------
# Defaults (edit here or use CLI)
# -----------------------------
DEF_TS_DIR     = r"/mmfs1/home/mzu0014/timeseries_100"           # per-subject time series
DEF_GRAPH_DIR  = r"/mmfs1/home/mzu0014/alphaZ_graphs"            # per-subject Ahat.npy
DEF_EXCEL      = r"/mmfs1/home/mzu0014/100_Subj_Full_v3.xlsx"    # metadata with Subject, Gender, Age, etc.
DEF_OUTDIR     = r"/mmfs1/home/mzu0014/project1/alphaZ_gcn_results"

DEF_TASK       = "sex"       # "sex" for classification, "age" (or any numeric col) for regression
DEF_ID_COL     = "Subject"
DEF_LABEL_COL  = "Gender"    # "Gender" for sex, or e.g. "Age" for regression

DEF_SCAN       = "REST1"
DEF_HEMI       = "LR"
DEF_PARC       = 100

DEF_T          = 100         # window length (timepoints)
DEF_WINS_TRAIN = 16          # windows per subject for training
DEF_WINS_VAL   = 64          # windows per subject for validation (averaged)
DEF_BATCH      = 16
DEF_EPOCHS     = 30
DEF_LR         = 1e-3
DEF_WD         = 1e-3
DEF_SPLITS     = 5
DEF_SEED       = 42
DEF_NUM_WORKERS= 2

# -----------------------------
# Robust Excel/CSV reader
# -----------------------------
def read_table(path, sheet_name=0):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xlsm", ".xltx", ".xltm"):
        # Prefer xlrd if available (works on many HPCs with older pandas when xlrd==1.2.0), else openpyxl
        try:
            import xlrd  # noqa: F401
            return pd.read_excel(path, sheet_name=sheet_name, engine="xlrd")
        except Exception:
            try:
                import openpyxl  # noqa: F401
                return pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
            except Exception as e:
                raise ImportError("Install xlrd==1.2.0 or openpyxl to read .xlsx. Last error: %s" % e)
    elif ext == ".xls":
        import xlrd  # noqa: F401
        return pd.read_excel(path, sheet_name=sheet_name, engine="xlrd")
    elif ext == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError("Unsupported table extension: %s" % ext)

# -----------------------------
# File utilities
# -----------------------------
def glob_one(patterns: List[str]) -> str:
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError("No files match any of: %r" % patterns)

def find_timeseries(ts_dir: str, sid: str, hemi: str, parc: int) -> str:
    # Check in subject subfolder then root; allow both space and underscore around tokens, any extension
    pats = [
        os.path.join(ts_dir, sid, "%s_rfMRI_REST1_%s_%d*" % (sid, hemi, parc)),
        os.path.join(ts_dir, sid, "%s_rfMRI_REST1_%s_*"   % (sid, hemi)),
        os.path.join(ts_dir, sid, "%s*REST1*%s*%d*"       % (sid, hemi, parc)),
        os.path.join(ts_dir, "%s_rfMRI_REST1_%s_%d*"      % (sid, hemi, parc)),
        os.path.join(ts_dir, "%s*REST1*%s*%d*"            % (sid, hemi, parc)),
    ]
    return glob_one(pats)

def load_timeseries(ts_path: str) -> np.ndarray:
    # Try text first; if that fails, try CSV via pandas
    try:
        X = np.loadtxt(ts_path)
        if X.ndim == 1:
            X = X[:, None]
        return np.asarray(X, dtype=np.float32)
    except Exception:
        df = pd.read_csv(ts_path, sep=None, engine="python", header=None)
        X = df.values
        if X.ndim == 1:
            X = X[:, None]
        return np.asarray(X, dtype=np.float32)

def find_graph(graph_dir: str, sid: str, hemi: str) -> str:
    # Typical name: {sid}_REST1_hemi_{hemi}_Ahat.npy
    pats = [
        os.path.join(graph_dir, "%s_REST1_hemi_%s_Ahat.npy" % (sid, hemi)),
        os.path.join(graph_dir, "%s_*hemi*%s*_Ahat.npy"     % (sid, hemi)),
        os.path.join(graph_dir, sid, "%s*REST1*%s*Ahat.npy" % (sid, hemi)),
        os.path.join(graph_dir, "%s_Ahat.npy"               % sid),
    ]
    return glob_one(pats)

def zscore_time_series(X: np.ndarray) -> np.ndarray:
    # Z-score per node (column)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    return (X - mu) / sd

# -----------------------------
# Dataset
# -----------------------------
class HCPAlphaZWindows(torch.utils.data.Dataset):
    def __init__(self,
                 subj_ids: List[str],
                 ts_dir: str,
                 graph_dir: str,
                 labels: Dict[str, float],
                 hemi: str = "LR",
                 T: int = 100,
                 windows_per_subj: int = 16,
                 mode: str = "train"):
        self.items = []  # list of (sid, t0)
        self.labels = labels
        self.hemi = hemi
        self.T = T
        self.mode = mode
        self.cache_ts: Dict[str, np.ndarray] = {}
        self.cache_A: Dict[str, np.ndarray] = {}

        rng = random.Random(0 if mode != "train" else None)

        for sid in subj_ids:
            ts_path = find_timeseries(ts_dir, sid, hemi, parc=DEF_PARC)
            X = load_timeseries(ts_path)
            X = zscore_time_series(X)
            self.cache_ts[sid] = X

            A_path = find_graph(graph_dir, sid, hemi)
            A = np.load(A_path)
            A = np.asarray(A, dtype=np.float32)
            if A.ndim != 2 or A.shape[0] != A.shape[1]:
                raise ValueError("Ahat must be square (N x N). Got %r for %s" % (A.shape, sid))
            # Symmetrize gently
            A = 0.5 * (A + A.T)
            self.cache_A[sid] = A

            Ttot = X.shape[0]
            if mode == "train":
                for _ in range(windows_per_subj):
                    t0 = rng.randrange(0, max(1, Ttot - T + 1)) if Ttot > T else 0
                    self.items.append((sid, t0))
            else:
                # deterministic set of starts
                K = max(1, windows_per_subj)
                if Ttot <= T:
                    starts = [0]
                else:
                    gap = (Ttot - T) / max(1, K - 1)
                    starts = [int(round(i * gap)) for i in range(K)]
                for t0 in starts:
                    self.items.append((sid, t0))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        sid, t0 = self.items[i]
        X = self.cache_ts[sid]
        A = self.cache_A[sid]
        T = self.T

        if X.shape[0] >= T:
            xw = X[t0:t0+T]
        else:
            pad = np.zeros((T - X.shape[0], X.shape[1]), dtype=X.dtype)
            xw = np.vstack([X, pad])

        # shapes:
        # x: (B, C=1, T, N); A: (B, K=1, N, N)
        x = torch.from_numpy(xw.T[None, ...])      # (1, T, N) -> add channel dim later
        x = x.unsqueeze(0).squeeze(0)              # (1, T, N) already; we'll add C=1 in model
        x = x.unsqueeze(0)                         # (1, 1, T, N)
        A = torch.from_numpy(A[None, None, ...])   # (1, 1, N, N)

        y = self.labels[sid]
        y = torch.tensor(y)
        return x.float(), A.float(), y.long(), sid

# -----------------------------
# Model (MS-G3D-inspired lite)
# -----------------------------
class SpatialGraph(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, K: int = 1):
        super(SpatialGraph, self).__init__()
        self.K = K
        self.edge_importance = nn.Parameter(torch.ones(1, K, 1, 1))
        self.proj = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(1,1), bias=True)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x, A):
        # x: (B, C, T, N) ; A: (B, K, N, N)
        B, C, T, N = x.shape
        K = A.shape[1]
        Aeff = A * self.edge_importance  # (B, K, N, N)

        # Graph message passing for each k, sum over K
        x_ = x.permute(0,2,1,3).contiguous().view(B*T, C, N)  # (B*T, C, N)
        yk = []
        for k in range(K):
            Ak = Aeff[:, k]                                   # (B, N, N)
            Ak_rep = Ak.repeat_interleave(T, dim=0)           # (B*T, N, N)
            # simple first-order: aggregate neighbors (no learnable weights yet)
            agg = torch.bmm(x_.transpose(1,2), Ak_rep).transpose(1,2)  # (B*T, C, N)
            yk.append(agg)
        y = torch.stack(yk, dim=1).sum(1)  # (B*T, C, N)
        y = y.view(B, T, C, N).permute(0,2,1,3)  # (B, C, T, N)
        y = self.proj(y)
        y = self.bn(y)
        return F.relu(y, inplace=True)

class TemporalBlock(nn.Module):
    def __init__(self, C: int, dilation: int = 1, dropout: float = 0.1):
        super(TemporalBlock, self).__init__()
        pad = (3-1) * dilation
        self.dw = nn.Conv2d(C, C, kernel_size=(3,1), padding=(pad,0), dilation=(dilation,1), groups=C, bias=False)
        self.pw = nn.Conv2d(C, C, kernel_size=(1,1), bias=True)
        self.bn = nn.BatchNorm2d(C)
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        y = self.dw(x)
        y = self.pw(y)
        y = self.bn(y)
        y = F.relu(y, inplace=True)
        return self.do(y + x)

class MSG3D_Lite(nn.Module):
    def __init__(self, N: int, num_classes: int = 2):
        super(MSG3D_Lite, self).__init__()
        self.s1 = SpatialGraph(1, 96, K=1)
        self.t1 = TemporalBlock(96, dilation=1)
        self.s2 = SpatialGraph(96, 192, K=1)
        self.t2 = TemporalBlock(192, dilation=2)
        self.s3 = SpatialGraph(192, 384, K=1)
        self.t3 = TemporalBlock(384, dilation=3)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc   = nn.Linear(384, num_classes)

    def forward(self, x, A):
        # x: (B, 1, T, N) ; A: (B, 1, N, N)
        y = self.s1(x, A)
        y = self.t1(y)
        y = self.s2(y, A)
        y = self.t2(y)
        y = self.s3(y, A)
        y = self.t3(y)
        y = self.pool(y).squeeze(-1).squeeze(-1)  # (B, 384)
        return self.fc(y)

# -----------------------------
# Label utilities
# -----------------------------
def load_subjects_and_labels(excel_path: str, id_col: str, label_col: str, task: str) -> Tuple[List[str], Dict[str, float], bool]:
    meta = read_table(excel_path)
    cols_lower = [c.lower() for c in meta.columns]
    if id_col not in meta.columns:
        # try to guess
        for cand in ["subject", "subj", "id", "subject_id", "sid"]:
            if cand in cols_lower:
                id_col = meta.columns[cols_lower.index(cand)]
                break
    if id_col not in meta.columns:
        raise ValueError("Could not find subject id column (got columns: %r)" % list(meta.columns))

    if label_col not in meta.columns:
        raise ValueError("Label column %r not found in %r" % (label_col, excel_path))

    S_raw = [str(s).split('.')[0] for s in meta[id_col].tolist()]

    is_classification = (task == "sex") or (label_col.strip().lower() in ("gender", "sex"))
    labels = {}

    if is_classification:
        g = meta[label_col].astype(str).str.upper().map({'F':0,'M':1,'0':0,'1':1})
        if g.isna().any():
            raise ValueError("Unrecognized gender values; expected F/M or 0/1.")
        for i, sid in enumerate(S_raw):
            labels[str(sid)] = int(g.iloc[i])
    else:
        y = pd.to_numeric(meta[label_col], errors="coerce")
        if y.isna().any():
            raise ValueError("Non-numeric values found for regression label column %r." % label_col)
        for i, sid in enumerate(S_raw):
            labels[str(sid)] = float(y.iloc[i])

    return [str(s) for s in S_raw], labels, is_classification

# -----------------------------
# CV splitting
# -----------------------------
def build_splits(ids: List[str], labels: Dict[str, float], is_classification: bool, seed: int, kfold: int):
    X = np.zeros((len(ids), 1))
    y = np.array([labels[sid] for sid in ids])
    if is_classification:
        skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=seed)
        return list(skf.split(X, y))
    else:
        kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)
        return list(kf.split(X))

# -----------------------------
# Evaluation helpers
# -----------------------------
def evaluate(model, loader, device, task_is_classification=True):
    model.eval()
    with torch.no_grad():
        agg = {}   # sid -> list of logits
        gts = {}
        for x, A, y, sid in loader:
            x = x.to(device, non_blocking=True)
            A = A.to(device, non_blocking=True)
            logits = model(x, A)
            for i, s in enumerate(sid):
                agg.setdefault(s, []).append(logits[i].detach().cpu())
                gts[s] = y[i].item()

        ys, ps = [], []
        for s, chunks in agg.items():
            m = torch.stack(chunks, 0).mean(0)
            ps.append(m)
            ys.append(gts[s])

        P = torch.stack(ps, 0)          # (num_subjects, C or 1)
        Y = torch.tensor(ys)

        if task_is_classification:
            pred = (P[:,1] > P[:,0]).long()
            acc = (pred == Y.long()).float().mean().item()
            # AUC (optional): need probabilities; softmax on logits
            prob1 = torch.softmax(P, dim=1)[:,1].numpy()
            auc = roc_auc_score(Y.numpy(), prob1)
            return {"acc": acc, "auc": float(auc)}
        else:
            # Pearson r between predicted scalar and ground truth
            yhat = P.squeeze(-1).numpy()
            y = Y.numpy().astype(np.float64)
            ymu = y.mean(); yhatmu = yhat.mean()
            num = ((y-ymu)*(yhat-yhatmu)).sum()
            den = math.sqrt(((y-ymu)**2).sum() * ((yhat-yhatmu)**2).sum() + 1e-12)
            r = float(num/den) if den>0 else 0.0
            return {"r": r}

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Dahan-style spatio–temporal GCN for fMRI with Alpha–Z graphs")
    ap.add_argument("--ts-dir", default=DEF_TS_DIR)
    ap.add_argument("--graph-dir", default=DEF_GRAPH_DIR)
    ap.add_argument("--excel", default=DEF_EXCEL)
    ap.add_argument("--outdir", default=DEF_OUTDIR)

    ap.add_argument("--task", default=DEF_TASK, help='Use "sex" for classification; anything else with numeric labels treated as regression')
    ap.add_argument("--id-col", default=DEF_ID_COL)
    ap.add_argument("--label-col", default=DEF_LABEL_COL)

    ap.add_argument("--hemi", default=DEF_HEMI)
    ap.add_argument("--parc", type=int, default=DEF_PARC)

    ap.add_argument("--T", type=int, default=DEF_T)
    ap.add_argument("--wins-train", type=int, default=DEF_WINS_TRAIN)
    ap.add_argument("--wins-val", type=int, default=DEF_WINS_VAL)

    ap.add_argument("--batch-size", type=int, default=DEF_BATCH)
    ap.add_argument("--epochs", type=int, default=DEF_EPOCHS)
    ap.add_argument("--lr", type=float, default=DEF_LR)
    ap.add_argument("--weight-decay", type=float, default=DEF_WD)

    ap.add_argument("--splits", type=int, default=DEF_SPLITS)
    ap.add_argument("--seed", type=int, default=DEF_SEED)
    ap.add_argument("--num-workers", type=int, default=DEF_NUM_WORKERS)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load subjects + labels
    all_ids, labels, is_cls = load_subjects_and_labels(args.excel, args.id_col, args.label_col, args.task)

    # Check one graph to infer N
    probe_sid = all_ids[0]
    A0 = np.load(find_graph(args.graph_dir, probe_sid, args.hemi)).astype(np.float32)
    N = A0.shape[0]

    # Build CV splits
    splits = build_splits(all_ids, labels, is_cls, args.seed, args.splits)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_rows = []
    all_pred, all_true, all_sid = [], [], []

    for fidx, (tr_idx, va_idx) in enumerate(splits, 1):
        tr_ids = [all_ids[i] for i in tr_idx]
        va_ids = [all_ids[i] for i in va_idx]

        print("\n=== Fold %d/%d: train=%d  val=%d ===" % (fidx, args.splits, len(tr_ids), len(va_ids)))

        train_ds = HCPAlphaZWindows(tr_ids, args.ts_dir, args.graph_dir, labels,
                                    hemi=args.hemi, T=args.T, windows_per_subj=args.wins_train, mode="train")
        val_ds   = HCPAlphaZWindows(va_ids, args.ts_dir, args.graph_dir, labels,
                                    hemi=args.hemi, T=args.T, windows_per_subj=args.wins_val, mode="val")

        train_ld = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True, drop_last=False)
        val_ld   = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.num_workers, pin_memory=True, drop_last=False)

        num_classes = 2 if is_cls else 1
        model = MSG3D_Lite(N, num_classes=num_classes).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        crit = nn.CrossEntropyLoss() if is_cls else nn.MSELoss()

        best_score = -1e9
        for epoch in range(1, args.epochs+1):
            model.train()
            tot = 0.0
            nsamp = 0
            for x, A, y, sid in train_ld:
                x = x.to(device, non_blocking=True)          # (B, 1, T, N)
                A = A.to(device, non_blocking=True)          # (B, 1, N, N)
                if is_cls:
                    y = y.to(device, non_blocking=True).long()
                    logits = model(x, A)
                    loss = crit(logits, y)
                else:
                    y = y.to(device, non_blocking=True).float()
                    logits = model(x, A).squeeze(-1)
                    loss = crit(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                bs = x.size(0)
                tot += float(loss.item()) * bs
                nsamp += bs
            tr_loss = tot / max(1, nsamp)

            metrics = evaluate(model, val_ld, device, task_is_classification=is_cls)
            if is_cls:
                score = metrics["acc"]
                msg = "epoch %02d: train_loss=%.4f  val_acc=%.3f  val_auc=%.3f" % (epoch, tr_loss, metrics["acc"], metrics["auc"])
            else:
                score = metrics["r"]
                msg = "epoch %02d: train_loss=%.4f  val_r=%.3f" % (epoch, tr_loss, metrics["r"])
            print(msg)

            if score > best_score:
                best_score = score

        # Final eval on val set for fold predictions
        model.eval()
        fold_sid, fold_true, fold_prob, fold_pred = [], [], [], []
        with torch.no_grad():
            for x, A, y, sid in val_ld:
                x = x.to(device)
                A = A.to(device)
                logits = model(x, A).cpu()
                if is_cls:
                    probs = torch.softmax(logits, dim=1)[:,1]
                    preds = (probs >= 0.5).long()
                    fold_prob.extend(probs.numpy().tolist())
                    fold_pred.extend(preds.numpy().tolist())
                    fold_true.extend(y.numpy().tolist())
                    fold_sid.extend(list(sid))
                else:
                    yhat = logits.squeeze(-1)
                    fold_prob.extend(yhat.numpy().tolist())  # store predictions
                    fold_true.extend(y.numpy().tolist())
                    fold_sid.extend(list(sid))

        # Metrics per fold
        if is_cls:
            acc = accuracy_score(fold_true, fold_pred)
            try:
                auc = roc_auc_score(fold_true, fold_prob)
            except Exception:
                auc = float('nan')
            print("Fold %d summary: best=%.3f | final Acc=%.3f AUC=%.3f" % (fidx, best_score, acc, auc))
            fold_rows.append({"fold": fidx, "n_train": len(tr_ids), "n_val": len(va_ids),
                              "accuracy": float(acc), "auc": float(auc), "best": float(best_score)})
        else:
            # compute Pearson r on subject-averaged predictions:
            y = np.array(fold_true, dtype=np.float64)
            yhat = np.array(fold_prob, dtype=np.float64)
            ymu, yhatmu = y.mean(), yhat.mean()
            num = ((y-ymu)*(yhat-yhatmu)).sum()
            den = math.sqrt(((y-ymu)**2).sum() * ((yhat-yhatmu)**2).sum() + 1e-12)
            r = float(num/den) if den>0 else 0.0
            print("Fold %d summary: best=%.3f | final r=%.3f" % (fidx, best_score, r))
            fold_rows.append({"fold": fidx, "n_train": len(tr_ids), "n_val": len(va_ids),
                              "r": float(r), "best": float(best_score)})

        # Save fold predictions
        pred_df = pd.DataFrame({
            "sid": fold_sid,
            "y_true": fold_true,
            "y_pred": fold_pred if is_cls else [None]*len(fold_true),
            "y_score": fold_prob
        })
        pred_df.to_csv(os.path.join(args.outdir, "predictions_fold%d.csv" % fidx), index=False)

        all_sid.extend(fold_sid)
        all_true.extend(fold_true)
        all_pred.extend(fold_pred if is_cls else fold_prob)

    # Save summary
    df_folds = pd.DataFrame(fold_rows)
    df_folds.to_csv(os.path.join(args.outdir, "cv_summary_by_fold.csv"), index=False)

    if is_cls:
        mean_acc = df_folds["accuracy"].mean()
        std_acc  = df_folds["accuracy"].std()
        mean_auc = df_folds["auc"].mean()
        std_auc  = df_folds["auc"].std()
        with open(os.path.join(args.outdir, "final_summary.txt"), "w") as f:
            f.write("Mean Acc: %.4f ± %.4f\n" % (mean_acc, std_acc))
            f.write("Mean AUC: %.4f ± %.4f\n" % (mean_auc, std_auc))
        print("\n=== Summary (Sex) ===")
        print("Mean Acc: %.3f ± %.3f | Mean AUC: %.3f ± %.3f" % (mean_acc, std_acc, mean_auc, std_auc))
    else:
        mean_r = df_folds["r"].mean()
        std_r  = df_folds["r"].std()
        with open(os.path.join(args.outdir, "final_summary.txt"), "w") as f:
            f.write("Mean r: %.4f ± %.4f\n" % (mean_r, std_r))
        print("\n=== Summary (Regression) ===")
        print("Mean r: %.3f ± %.3f" % (mean_r, std_r))

    # Save all predictions
    pd.DataFrame({"sid": all_sid, "y_true": all_true, "y_out": all_pred}).to_csv(
        os.path.join(args.outdir, "predictions_all.csv"), index=False
    )

if __name__ == "__main__":
    main()
