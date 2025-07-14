
import numpy as np
import os
from scipy.linalg import fractional_matrix_power
import matplotlib.pyplot as plt
import random
def load_connectivity_matrix(file_path):
    try:
        return np.loadtxt(file_path, delimiter=' ')
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def generate_file_paths(base_path, scan_type, num_subjects=30):
    file_paths = []
    subject_ids = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    subject_ids.sort()
    subject_ids = subject_ids[:num_subjects]
    for subject_id in subject_ids:
        file_path = os.path.join(base_path, subject_id, f'{subject_id}_rfMRI_REST1_{scan_type}_100')
        file_paths.append(file_path)
    return file_paths

def compute_alpha_z_BW_distance(A, B, alpha, z):
    if not (0 <= alpha <= z <= 1):
        raise ValueError("Alpha and z must satisfy 0 <= alpha <= z <= 1")
    
    def Q_alpha_z(A, B, alpha, z):
        if z == 0:
            return np.zeros_like(A)
        part1 = fractional_matrix_power(B, (1-alpha)/(2*z))
        part2 = fractional_matrix_power(A, alpha/z)
        part3 = fractional_matrix_power(B, (1-alpha)/(2*z))
        Q_az = fractional_matrix_power(part1.dot(part2).dot(part3), z)
        return Q_az

    Q_az = Q_alpha_z(A, B, alpha, z)
    divergence = np.trace((1-alpha) * A + alpha * B) - np.trace(Q_az)    
    return np.real(divergence)

def distance_matrix(connectivity_matrices_1, connectivity_matrices_2, alpha, z):
    num_subjects = len(connectivity_matrices_1)
    distance_matrix = np.zeros((num_subjects, num_subjects))
    for i, matrix_1 in enumerate(connectivity_matrices_1):
        if matrix_1 is None:
            continue
        for j, matrix_2 in enumerate(connectivity_matrices_2):
            if matrix_2 is None:
                continue
            distance_matrix[i, j] = compute_alpha_z_BW_distance(matrix_1, matrix_2, alpha, z)
    return distance_matrix

def compute_id_rate(distance_matrix):
    correct_identifications = sum(np.argmin(distance_matrix[i, :]) == i for i in range(distance_matrix.shape[0]))
    return correct_identifications / distance_matrix.shape[0]



#base_path='/mmfs1/home/mzu0014/connectomes_200/'
base_path = 'D:/Research AU/Python/connectomes_100/'


lr_paths = generate_file_paths(base_path, 'LR')
rl_paths = generate_file_paths(base_path, 'RL')

connectivity_matrices_lr = [load_connectivity_matrix(path) for path in lr_paths]
connectivity_matrices_rl = [load_connectivity_matrix(path) for path in rl_paths]

alpha=.99
z=1
distance_matrix_1 = distance_matrix(connectivity_matrices_lr, connectivity_matrices_rl, alpha, z)
id_rate_1 = compute_id_rate(distance_matrix_1)
distance_matrix_2 = distance_matrix(connectivity_matrices_rl, connectivity_matrices_lr, alpha, z)
id_rate_2 = compute_id_rate(distance_matrix_2)
current_id_rate = (id_rate_1 + id_rate_2) / 2
print(id_rate_1)
print(id_rate_2)
print(current_id_rate)

#%%
import numpy as np
import scipy.io as sio
from scipy.linalg import fractional_matrix_power

# -------------------------
# YOUR Alpha–Z + distance + ID (unchanged)
# -------------------------
def compute_alpha_z_BW_distance(A, B, alpha, z):
    if not (0 <= alpha <= z <= 1):
        raise ValueError("Alpha and z must satisfy 0 <= alpha <= z <= 1")
    def Q_alpha_z(A, B, alpha, z):
        if z == 0:
            return np.zeros_like(A)
        part1 = fractional_matrix_power(B, (1-alpha)/(2*z))
        part2 = fractional_matrix_power(A, alpha/z)
        part3 = fractional_matrix_power(B, (1-alpha)/(2*z))
        Q_az = fractional_matrix_power(part1.dot(part2).dot(part3), z)
        return Q_az
    Q_az = Q_alpha_z(A, B, alpha, z)
    divergence = np.trace((1-alpha) * A + alpha * B) - np.trace(Q_az)
    return np.real(divergence)

def distance_matrix(connectivity_matrices_1, connectivity_matrices_2, alpha, z):
    num_subjects = len(connectivity_matrices_1)
    D = np.zeros((num_subjects, num_subjects), dtype=float)
    for i, A in enumerate(connectivity_matrices_1):
        if A is None: continue
        for j, B in enumerate(connectivity_matrices_2):
            if B is None: continue
            D[i, j] = compute_alpha_z_BW_distance(A, B, alpha, z)
    return D

def compute_id_rate(D):
    hits = sum(np.argmin(D[i, :]) == i for i in range(D.shape[0]))
    return hits / D.shape[0]

# -------------------------
# 1) Robust loader for group-key .mat (returns list of T×R arrays)
# -------------------------
def load_ts_adni_by_groups(mat_path):
    """
    Loads per-subject time series from group-specific keys like:
    ts_adni_filt_controls1, ts_adni_filt_EMCI1, ts_adni_filt_LMCI1, ts_adni_filt_AD1
    Returns:
        subjects_ts: list of arrays [T_i x R] (one per subject, all groups concatenated)
        group_ids:   list of strings with group label for each subject (optional metadata)
    """
    M = sio.loadmat(mat_path)
    subjects_ts, group_ids = [], []

    # pick all keys that look like ts_adni_* (edit the prefix if needed)
    keys = [k for k in M.keys() if k.startswith('ts_adni_filt_')]
    if not keys:
        raise KeyError(f"No 'ts_adni_filt_*' keys found. Keys present: {list(M.keys())}")

    for k in keys:
        arr = M[k]
        # Possibility A: MATLAB cell array (S x 1), each cell is T×R
        if arr.ndim == 2 and arr.size > 0 and isinstance(arr.flat[0], np.ndarray):
            for i in range(arr.shape[0]):
                Xi = np.asarray(arr[i,0], dtype=float)
                if Xi.ndim != 2:
                    continue
                subjects_ts.append(Xi)
                group_ids.append(k)
        # Possibility B: 3D numeric array [T,R,S]
        elif arr.ndim == 3:
            T, R, S = arr.shape
            for s in range(S):
                Xi = np.asarray(arr[:,:,s], dtype=float)
                subjects_ts.append(Xi)
                group_ids.append(k)
        else:
            # Sometimes stored as list-like; try to coerce any 2D matrices
            if isinstance(arr, np.ndarray) and arr.ndim == 2:
                # Single subject case (rare)
                subjects_ts.append(np.asarray(arr, dtype=float))
                group_ids.append(k)

    if not subjects_ts:
        raise ValueError("No per-subject T×R matrices could be extracted from the group keys.")

    return subjects_ts, group_ids

# -------------------------
# 2) Split per subject (half or even/odd)
# -------------------------
def split_subject_ts_list(subjects_ts, mode="evenodd"):
    A_list, B_list = [], []
    for X in subjects_ts:
        T = X.shape[0]
        if mode == "half":
            mid = T // 2
            A_list.append(X[:mid, :])
            B_list.append(X[mid:, :])
        elif mode == "evenodd":
            A_list.append(X[::2, :])   # even TRs (0-based)
            B_list.append(X[1::2, :])  # odd TRs
        else:
            raise ValueError("mode must be 'half' or 'evenodd'")
    return A_list, B_list

# -------------------------
# 3) FC from time series + SPD ridge
# -------------------------
def fc_from_ts(X):
    X = X - np.nanmean(X, axis=0, keepdims=True)
    C = np.corrcoef(X, rowvar=False)
    C = 0.5*(C + C.T)
    eps = 1e-3 * (np.mean(np.diag(C)**2) + 1e-12)
    C = C + eps*np.eye(C.shape[0])
    return C

# -------------------------
# 4) Run the pipeline
# -------------------------
if __name__ == "__main__":
    mat_path = r"D:\Research AU\Alzheimer FCNs\Alzheimer FCNs\ts_adni_filt.mat"  # your path
    subjects_ts, group_ids = load_ts_adni_by_groups(mat_path)
    print(f"Loaded {len(subjects_ts)} subjects from keys: {set(group_ids)}")

    # Split halves (even/odd is robust for short scans)
    A_ts, B_ts = split_subject_ts_list(subjects_ts, mode="evenodd")

    # Build FCs
    A_FC = [fc_from_ts(X) for X in A_ts]
    B_FC = [fc_from_ts(X) for X in B_ts]

    # Alpha–Z (same as you use elsewhere)
    alpha, z = 0.99, 1.0

    # Distances with YOUR functions
    D_AB = distance_matrix(A_FC, B_FC, alpha, z)
    D_BA = distance_matrix(B_FC, A_FC, alpha, z)

    # ID rates
    id_AB = compute_id_rate(D_AB)
    id_BA = compute_id_rate(D_BA)
    id_mean = 0.5*(id_AB + id_BA)

    print(f"Split=even/odd | alpha={alpha}, z={z}")
    print(f"ID(A→B) = {id_AB:.3f}")
    print(f"ID(B→A) = {id_BA:.3f}")
    print(f"ID(mean)= {id_mean:.3f}")

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


# %%
import numpy as np
import scipy.io as sio

# -------------------------
# 0) Distance functions
# -------------------------

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

# -------------------------
# 1) Distance matrix + ID rate
# -------------------------
def distance_matrix(connectivity_mats_A, connectivity_mats_B, metric="pearson"):
    """
    Build N x N distance matrix between two equal-length lists of matrices.
    metric: "pearson" or "euclidean"
    """
    N = len(connectivity_mats_A)
    D = np.zeros((N, N), dtype=float)

    if metric not in {"pearson", "euclidean"}:
      raise ValueError("metric must be 'pearson' or 'euclidean'")

    fn = compute_pearson_distance if metric == "pearson" else compute_euclidean_distance

    for i, A in enumerate(connectivity_mats_A):
        if A is None: continue
        for j, B in enumerate(connectivity_mats_B):
            if B is None: continue
            D[i, j] = fn(A, B)
    return D

def compute_id_rate(D):
    """
    Identification rate from A->B given a distance matrix D (rows=A, cols=B).
    """
    hits = sum(np.argmin(D[i, :]) == i for i in range(D.shape[0]))
    return hits / D.shape[0]

# -------------------------
# 2) Loader for group-key ADNI .mat  (returns list of T×R arrays)
# -------------------------
def load_ts_adni_by_groups(mat_path):
    """
    Reads keys like:
      ts_adni_filt_controls1, ts_adni_filt_EMCI1, ts_adni_filt_LMCI1, ts_adni_filt_AD1
    Returns:
      subjects_ts: [list of T_i x R arrays]
      group_ids:   [list of key names per subject] (metadata)
    """
    M = sio.loadmat(mat_path)
    subjects_ts, group_ids = [], []

    keys = [k for k in M.keys() if k.startswith('ts_adni_filt_')]
    if not keys:
        raise KeyError(f"No 'ts_adni_filt_*' keys found. Keys present: {list(M.keys())}")

    for k in keys:
        arr = M[k]
        # MATLAB cell: (S x 1), each cell is T x R
        if arr.ndim == 2 and arr.size > 0 and isinstance(arr.flat[0], np.ndarray):
            for i in range(arr.shape[0]):
                Xi = np.asarray(arr[i, 0], dtype=float)
                if Xi.ndim == 2:
                    subjects_ts.append(Xi)
                    group_ids.append(k)
        # 3D numeric array: [T, R, S]
        elif arr.ndim == 3:
            T, R, S = arr.shape
            for s in range(S):
                Xi = np.asarray(arr[:, :, s], dtype=float)
                subjects_ts.append(Xi)
                group_ids.append(k)
        # single 2D numeric: one subject
        elif isinstance(arr, np.ndarray) and arr.ndim == 2:
            subjects_ts.append(np.asarray(arr, dtype=float))
            group_ids.append(k)

    if not subjects_ts:
        raise ValueError("No per-subject T×R matrices extracted from the group keys.")
    return subjects_ts, group_ids

# -------------------------
# 3) Split per subject (half or even/odd)
# -------------------------
def split_subject_ts_list(subjects_ts, mode="evenodd"):
    """
    Return two lists A_list, B_list of [T_i x R] per subject.
    mode='half'    -> first half vs second half
    mode='evenodd' -> even vs odd TRs
    """
    A_list, B_list = [], []
    for X in subjects_ts:
        T = X.shape[0]
        if mode == "half":
            mid = T // 2
            A_list.append(X[:mid, :])
            B_list.append(X[mid:, :])
        elif mode == "evenodd":
            A_list.append(X[::2, :])   # even TRs (0-based)
            B_list.append(X[1::2, :])  # odd TRs
        else:
            raise ValueError("mode must be 'half' or 'evenodd'")
    return A_list, B_list

def two_windows_100(X, L=100, min_gap=10, seed=42):
    """
    Return two T_win x R windows per subject aiming for L=100 TR each.
    Strategy:
      1) If T >= 2L + min_gap -> choose two NON-OVERLAPPING windows of length L.
      2) Else if T >= L + min_gap -> choose two windows with a gap (may overlap less strictly).
      3) Else -> fallback to even/odd split, then truncate/pad to L when possible.
    """
    rng = np.random.default_rng(seed)
    T, R = X.shape

    # Case 1: plenty of data for two non-overlapping 100-TR windows
    if T >= 2*L + min_gap:
        # sample a start for A in [0 .. T-2L-min_gap]
        a_start = rng.integers(0, T - 2*L - min_gap + 1)
        b_start_min = a_start + L + min_gap
        b_start = rng.integers(b_start_min, b_start_min + (T - (b_start_min + L)) + 1)
        A = X[a_start:a_start+L, :]
        B = X[b_start:b_start+L, :]
        return A, B

    # Case 2: enough for at least one 100-TR window and a second nearby with a gap
    if T >= L + min_gap:
        a_start = 0
        b_start = min(a_start + L + min_gap, max(T - L, 0))
        # If b_start runs past, shift a_start left if we can
        if b_start + L > T:
            shift = (b_start + L) - T
            a_start = max(0, a_start - shift)
            b_start = min(a_start + L + min_gap, max(T - L, 0))
        A = X[a_start:a_start+L, :]
        B = X[b_start:b_start+L, :]
        return A, B

    # Case 3: fallback (short scans) — even/odd split, then truncate to L if possible
    A = X[::2, :]
    B = X[1::2, :]
    # If either is longer than L, take a centered 100-TR crop; if shorter, just use what we have
    def crop_center(Y, L):
        if Y.shape[0] <= L: 
            return Y
        start = (Y.shape[0] - L) // 2
        return Y[start:start+L, :]

    A = crop_center(A, L)
    B = crop_center(B, L)
    return A, B
# -------------------------
# 4) FC from time series (Pearson) – with tiny ridge to stabilize
# -------------------------
def fc_from_ts(X):
    """
    Pearson FC (R x R). Adds a tiny ridge to stabilize numerics.
    """
    X = X - np.nanmean(X, axis=0, keepdims=True)
    C = np.corrcoef(X, rowvar=False)     # R x R
    C = 0.5 * (C + C.T)                  # symmetrize
    eps = 1e-8
    C = C + eps * np.eye(C.shape[0])     # ridge (small; fine for similarity/distance)
    return C

# -------------------------
# 5) Run the split-half ID with Pearson/Euclidean
# -------------------------
if __name__ == "__main__":
    mat_path = r"D:\Research AU\Alzheimer FCNs\Alzheimer FCNs\ts_adni_filt.mat"  # <-- your path
    subjects_ts, group_ids = load_ts_adni_by_groups(mat_path)
    print(f"Loaded {len(subjects_ts)} subjects from keys: {sorted(set(group_ids))}")

    # Choose how to split: "evenodd" (recommended for short scans) or "half"
    A_ts, B_ts = split_subject_ts_list(subjects_ts, mode="evenodd")

    # Build FCs per subject for each split
    A_FC = [fc_from_ts(X) for X in A_ts]
    B_FC = [fc_from_ts(X) for X in B_ts]

    # --- Pearson ---
    D_AB_p = distance_matrix(A_FC, B_FC, metric="pearson")
    D_BA_p = distance_matrix(B_FC, A_FC, metric="pearson")
    id_AB_p = compute_id_rate(D_AB_p)
    id_BA_p = compute_id_rate(D_BA_p)
    id_mean_p = 0.5 * (id_AB_p + id_BA_p)

    print("\n[Split-half ID] Metric = Pearson")
    print(f"ID(A→B) = {id_AB_p:.3f}")
    print(f"ID(B→A) = {id_BA_p:.3f}")
    print(f"ID(mean)= {id_mean_p:.3f}")

    # --- Euclidean ---
    D_AB_e = distance_matrix(A_FC, B_FC, metric="euclidean")
    D_BA_e = distance_matrix(B_FC, A_FC, metric="euclidean")
    id_AB_e = compute_id_rate(D_AB_e)
    id_BA_e = compute_id_rate(D_BA_e)
    id_mean_e = 0.5 * (id_AB_e + id_BA_e)

    print("\n[Split-half ID] Metric = Euclidean")
    print(f"ID(A→B) = {id_AB_e:.3f}")
    print(f"ID(B→A) = {id_BA_e:.3f}")
    print(f"ID(mean)= {id_mean_e:.3f}")
    A_ts_100, B_ts_100 = [], []
for X in subjects_ts:
    A100, B100 = two_windows_100(X, L=100, min_gap=10, seed=123)  # tweak seed if you want
    A_ts_100.append(A100)
    B_ts_100.append(B100)

# Build FCs and run your existing Pearson/Euclidean ID code
A_FC = [fc_from_ts(X) for X in A_ts_100]
B_FC = [fc_from_ts(X) for X in B_ts_100]

# Distance matrices
D_AB_p = distance_matrix(A_FC, B_FC, metric="pearson")
D_BA_p = distance_matrix(B_FC, A_FC, metric="pearson")
D_AB_e = distance_matrix(A_FC, B_FC, metric="euclidean")
D_BA_e = distance_matrix(B_FC, A_FC, metric="euclidean")

# ID rates
id_AB_p = compute_id_rate(D_AB_p); id_BA_p = compute_id_rate(D_BA_p)
id_AB_e = compute_id_rate(D_AB_e); id_BA_e = compute_id_rate(D_BA_e)
print(f"Pearson ID mean = {(id_AB_p+id_BA_p)/2:.3f}")
print(f"Euclid  ID mean = {(id_AB_e+id_BA_e)/2:.3f}")

# %%
import numpy as np
import scipy.io as sio

# -------------------------
# Distance helpers
# -------------------------
def _upper_tri_vec(M):
    """Upper-triangular (k=1) vector (exclude diagonal)."""
    iu = np.triu_indices_from(M, k=1)
    return M[iu]

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

def distance_matrix(mats_A, mats_B, metric="pearson"):
    if metric not in {"pearson", "euclidean"}:
        raise ValueError("metric must be 'pearson' or 'euclidean'")
    fn = compute_pearson_distance if metric == "pearson" else compute_euclidean_distance
    N = len(mats_A)
    D = np.zeros((N, N), dtype=float)
    for i, A in enumerate(mats_A):
        for j, B in enumerate(mats_B):
            D[i, j] = fn(A, B)
    return D

def compute_id_rate(D):
    """Identification rate from A->B given distance matrix D (rows=A, cols=B)."""
    return np.mean([np.argmin(D[i, :]) == i for i in range(D.shape[0])])

# -------------------------
# Load ADNI time series organized by group keys
# -------------------------
def load_ts_adni_by_groups(mat_path):
    """
    Expects keys like:
      ts_adni_filt_controls1, ts_adni_filt_EMCI1, ts_adni_filt_LMCI1, ts_adni_filt_AD1
    Returns: subjects_ts = [T_i x R arrays], group_ids = [key per subject]
    """
    M = sio.loadmat(mat_path)
    subjects_ts, group_ids = [], []
    keys = sorted([k for k in M.keys() if k.startswith('ts_adni_filt_')])
    if not keys:
        raise KeyError(f"No 'ts_adni_filt_*' keys found. Keys present: {list(M.keys())}")

    for k in keys:
        arr = M[k]
        # MATLAB cell: (S x 1), each cell is T x R
        if arr.ndim == 2 and arr.size > 0 and isinstance(arr.flat[0], np.ndarray):
            for i in range(arr.shape[0]):
                Xi = np.asarray(arr[i, 0], dtype=float)
                if Xi.ndim == 2:
                    subjects_ts.append(Xi); group_ids.append(k)
        # 3D numeric array: [T, R, S]
        elif arr.ndim == 3:
            T, R, S = arr.shape
            for s in range(S):
                Xi = np.asarray(arr[:, :, s], dtype=float)
                subjects_ts.append(Xi); group_ids.append(k)
        # Single 2D numeric: one subject
        elif isinstance(arr, np.ndarray) and arr.ndim == 2:
            subjects_ts.append(np.asarray(arr, dtype=float)); group_ids.append(k)

    if not subjects_ts:
        raise ValueError("No per-subject T×R matrices extracted.")
    return subjects_ts, group_ids

# -------------------------
# Even/Odd split and FC
# -------------------------
def zscore_per_split(X):
    """Z-score time series per ROI within the split to avoid leakage and scale effects."""
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True) + 1e-8
    return (X - mu) / sd

def fc_from_ts(X):
    """Pearson FC with tiny ridge; zero diagonal for fair upper-tri features."""
    X = zscore_per_split(X)
    C = np.corrcoef(X, rowvar=False)   # R x R
    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 0.0)           # diagonals add no info; avoid trivial self-sim
    C += 1e-8 * np.eye(C.shape[0])     # tiny ridge
    return C

def build_even_odd_FCs(subjects_ts):
    """
    For each subject, build FC_even from X[::2,:] and FC_odd from X[1::2,:].
    Returns: A_FC (even list), B_FC (odd list)
    """
    A_FC, B_FC = [], []
    for X in subjects_ts:
        even = X[::2, :]
        odd  = X[1::2, :]
        # guard: if one split is empty (very short T), skip subject
        if even.shape[0] < 2 or odd.shape[0] < 2:
            continue
        A_FC.append(fc_from_ts(even))
        B_FC.append(fc_from_ts(odd))
    return A_FC, B_FC

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    mat_path = r"D:\Research AU\Alzheimer FCNs\Alzheimer FCNs\ts_adni_filt.mat"  # <-- your file
    subjects_ts, group_ids = load_ts_adni_by_groups(mat_path)
    print(f"Loaded {len(subjects_ts)} subjects from keys: {sorted(set(group_ids))}")

    # Build even/odd FCs
    A_FC, B_FC = build_even_odd_FCs(subjects_ts)
    N = min(len(A_FC), len(B_FC))
    A_FC = A_FC[:N]; B_FC = B_FC[:N]
    print(f"Using {N} subjects with valid even/odd splits")

    # Pearson
    D_AB_p = distance_matrix(A_FC, B_FC, metric="pearson")
    D_BA_p = distance_matrix(B_FC, A_FC, metric="pearson")
    print("\n[Even/Odd] Pearson")
    print(f"ID(A→B) = {compute_id_rate(D_AB_p):.3f}")
    print(f"ID(B→A) = {compute_id_rate(D_BA_p):.3f}")
    print(f"ID(mean)= {0.5*(compute_id_rate(D_AB_p)+compute_id_rate(D_BA_p)):.3f}")

    # Euclidean
    D_AB_e = distance_matrix(A_FC, B_FC, metric="euclidean")
    D_BA_e = distance_matrix(B_FC, A_FC, metric="euclidean")
    print("\n[Even/Odd] Euclidean")
    print(f"ID(A→B) = {compute_id_rate(D_AB_e):.3f}")
    print(f"ID(B→A) = {compute_id_rate(D_BA_e):.3f}")
    print(f"ID(mean)= {0.5*(compute_id_rate(D_AB_e)+compute_id_rate(D_BA_e)):.3f}")

# %%
import numpy as np
import scipy.io as sio
from scipy.linalg import fractional_matrix_power

# -------------------------
# Distance helpers
# -------------------------
def _upper_tri_vec(M):
    """Upper-triangular (k=1) vector (exclude diagonal)."""
    iu = np.triu_indices_from(M, k=1)
    return M[iu]

def compute_pearson_distance(X, Y):
    """Pearson distance = 1 - r, on upper-tri entries."""
    xv = _upper_tri_vec(X); yv = _upper_tri_vec(Y)
    mask = ~(np.isnan(xv) | np.isnan(yv))
    xv = xv[mask]; yv = yv[mask]
    if xv.size == 0: return 1.0
    sx, sy = np.std(xv), np.std(yv)
    if sx == 0 or sy == 0: return 1.0
    r = np.corrcoef(xv, yv)[0, 1]
    return float(1.0 - np.clip(r, -1.0, 1.0))

def compute_ai_distance(A, B, eps=1e-6):
    """
    Affine-Invariant (AIRM) distance for SPD matrices:
      d(A,B) = || log( A^{-1/2} B A^{-1/2} ) ||_F
             = sqrt( sum_i (log λ_i)^2 ),  λ_i eigvals of A^{-1/2} B A^{-1/2}
    Assumes A,B are SPD; adds tiny ridge if needed.
    """
    # symmetrize and ridge
    A = 0.5*(A + A.T); B = 0.5*(B + B.T)
    # ensure SPD (strictly positive eigs)
    A = A + eps*np.eye(A.shape[0])
    B = B + eps*np.eye(B.shape[0])

    # A^{-1/2} B A^{-1/2}
    A_inv_sqrt = fractional_matrix_power(A, -0.5)
    M = A_inv_sqrt @ B @ A_inv_sqrt

    # eigenvalues should be positive for SPD; clip for numerical safety
    w = np.linalg.eigvalsh(0.5*(M + M.T))
    w = np.clip(w, 1e-12, None)
    return float(np.linalg.norm(np.log(w)))

def distance_matrix(mats_A, mats_B, metric="pearson"):
    """
    Build N x N distance matrix.
    metric ∈ {'pearson','ai'}
    """
    if metric not in {"pearson", "ai"}:
        raise ValueError("metric must be 'pearson' or 'ai'")
    fn = compute_pearson_distance if metric == "pearson" else compute_ai_distance

    N = len(mats_A)
    D = np.zeros((N, N), dtype=float)
    for i, A in enumerate(mats_A):
        for j, B in enumerate(mats_B):
            D[i, j] = fn(A, B)
    return D

def compute_id_rate(D):
    """Identification rate from A->B given distance matrix D (rows=A, cols=B)."""
    return np.mean([np.argmin(D[i, :]) == i for i in range(D.shape[0])])

# -------------------------
# Load ADNI time series organized by group keys
# -------------------------
def load_ts_adni_by_groups(mat_path):
    """
    Expects keys like:
      ts_adni_filt_controls1, ts_adni_filt_EMCI1, ts_adni_filt_LMCI1, ts_adni_filt_AD1
    Returns: subjects_ts = [T_i x R], group_ids = [key per subject]
    """
    M = sio.loadmat(mat_path)
    subjects_ts, group_ids = [], []
    keys = sorted([k for k in M.keys() if k.startswith('ts_adni_filt_')])
    if not keys:
        raise KeyError(f"No 'ts_adni_filt_*' keys found. Keys present: {list(M.keys())}")

    for k in keys:
        arr = M[k]
        if arr.ndim == 2 and arr.size > 0 and isinstance(arr.flat[0], np.ndarray):
            for i in range(arr.shape[0]):
                Xi = np.asarray(arr[i, 0], dtype=float)
                if Xi.ndim == 2:
                    subjects_ts.append(Xi); group_ids.append(k)
        elif arr.ndim == 3:
            T, R, S = arr.shape
            for s in range(S):
                Xi = np.asarray(arr[:, :, s], dtype=float)
                subjects_ts.append(Xi); group_ids.append(k)
        elif isinstance(arr, np.ndarray) and arr.ndim == 2:
            subjects_ts.append(np.asarray(arr, dtype=float)); group_ids.append(k)

    if not subjects_ts:
        raise ValueError("No per-subject T×R matrices extracted.")
    return subjects_ts, group_ids

# -------------------------
# Even/Odd split and FC (SPD-friendly)
# -------------------------
def zscore_per_split(X):
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True) + 1e-8
    return (X - mu) / sd

def fc_from_ts_spd(X, ridge=1e-6):
    """
    Pearson FC (R x R) with symmetrization + *no zeroing of diagonal* +
    tiny ridge to maintain SPD for AIRM.
    """
    X = zscore_per_split(X)
    C = np.corrcoef(X, rowvar=False)   # R x R
    C = 0.5*(C + C.T)
    # No diagonal zeroing here — AIRM needs SPD
    C += ridge * np.eye(C.shape[0])
    return C

def build_even_odd_FCs(subjects_ts, ridge=1e-6):
    A_FC, B_FC = [], []
    for X in subjects_ts:
        even = X[::2, :]
        odd  = X[1::2, :]
        if even.shape[0] < 2 or odd.shape[0] < 2:
            continue
        A_FC.append(fc_from_ts_spd(even, ridge=ridge))
        B_FC.append(fc_from_ts_spd(odd,  ridge=ridge))
    return A_FC, B_FC

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    mat_path = r"D:\Research AU\Alzheimer FCNs\Alzheimer FCNs\ts_adni_filt.mat"  # <-- your file
    subjects_ts, group_ids = load_ts_adni_by_groups(mat_path)
    print(f"Loaded {len(subjects_ts)} subjects from keys: {sorted(set(group_ids))}")

    # Build even/odd FCs (SPD-ready)
    A_FC, B_FC = build_even_odd_FCs(subjects_ts, ridge=1e-6)
    N = min(len(A_FC), len(B_FC))
    A_FC = A_FC[:N]; B_FC = B_FC[:N]
    print(f"Using {N} subjects with valid even/odd splits")

    # Pearson (optional, still works with SPD FCs)
    D_AB_p = distance_matrix(A_FC, B_FC, metric="pearson")
    D_BA_p = distance_matrix(B_FC, A_FC, metric="pearson")
    print("\n[Even/Odd] Pearson")
    print(f"ID(A→B) = {compute_id_rate(D_AB_p):.3f}")
    print(f"ID(B→A) = {compute_id_rate(D_BA_p):.3f}")
    print(f"ID(mean)= {0.5*(compute_id_rate(D_AB_p)+compute_id_rate(D_BA_p)):.3f}")

    # AI (AIRM) distance
    D_AB_ai = distance_matrix(A_FC, B_FC, metric="ai")
    D_BA_ai = distance_matrix(B_FC, A_FC, metric="ai")
    print("\n[Even/Odd] AI (AIRM)")
    print(f"ID(A→B) = {compute_id_rate(D_AB_ai):.3f}")
    print(f"ID(B→A) = {compute_id_rate(D_BA_ai):.3f}")
    print(f"ID(mean)= {0.5*(compute_id_rate(D_AB_ai)+compute_id_rate(D_BA_ai)):.3f}")

# %%
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross-preprocessing identification (ADNI).
Gallery = ts_adni_filt.mat
Query   = ts_adni_dc.mat

Distance metrics: Pearson, Euclidean.
"""

import os
import numpy as np
import scipy.io as sio

# -----------------------------
# CONFIG
# -----------------------------
MAT_PATH_FILT = r"D:\Research AU\Alzheimer FCNs\Alzheimer FCNs\ts_adni_filt.mat"
MAT_PATH_DC   = r"D:\Research AU\Alzheimer FCNs\Alzheimer FCNs\ts_adni_dc.mat"
TR = 0.72             # seconds per frame (adjust if needed)
DURATIONS_MIN = [0.4, 0.6, 0.8, 1.0, 1.2]  # test durations
MIN_T = 30            # min frames for FC computation
RIDGE_EPS = 1e-3      # SPD regularization

# -----------------------------
# Distance functions
# -----------------------------
def compute_pearson_distance(X, Y):
    X_vec = X.flatten()
    Y_vec = Y.flatten()
    r = np.corrcoef(X_vec, Y_vec)[0, 1]
    return 1 - r

def compute_euclidean_distance(X, Y):
    X_vec = X.flatten()
    Y_vec = Y.flatten()
    return np.linalg.norm(X_vec - Y_vec)

# -----------------------------
# FC computation
# -----------------------------
def compute_fc(ts, ridge_eps=RIDGE_EPS):
    """Pearson correlation FC with SPD regularization."""
    X = ts - ts.mean(axis=0, keepdims=True)
    std = X.std(axis=0, ddof=1, keepdims=True)
    std[std == 0] = 1.0
    X = X / std
    C = np.corrcoef(X, rowvar=False)
    C = 0.5 * (C + C.T)
    C = C + ridge_eps * np.eye(C.shape[0])
    return C

# -----------------------------
# Loader
# -----------------------------
def load_timeseries_from_mat(path):
    """Return ndarray shaped (N,T,P)."""
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    # guess key
    key = None
    for k, v in mat.items():
        if k.startswith("__"): continue
        if isinstance(v, np.ndarray) and v.ndim >= 2:
            key = k; break
    if key is None:
        raise ValueError(f"No time-series found in {path}")
    arr = np.asarray(mat[key])
    if arr.ndim == 3:
        # assume (N,T,P) or (N,P,T)
        if arr.shape[1] >= 30:  # likely (N,T,P)
            return arr.astype(np.float64)
        else:                   # maybe (N,P,T)
            return np.transpose(arr, (0,2,1)).astype(np.float64)
    elif arr.ndim == 2:
        return arr[np.newaxis,:,:].astype(np.float64)  # single subject
    else:
        raise ValueError(f"Unsupported shape {arr.shape} in {path}")

# -----------------------------
# Identification
# -----------------------------
def identification_rate(FC_gallery, FC_query, metric="pearson"):
    N = FC_gallery.shape[0]
    D = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if metric == "pearson":
                d = compute_pearson_distance(FC_query[i], FC_gallery[j])
            elif metric == "euclidean":
                d = compute_euclidean_distance(FC_query[i], FC_gallery[j])
            else:
                raise ValueError("Unknown metric")
            D[i,j] = d
    nn = np.argmin(D, axis=1)
    correct = np.sum(nn == np.arange(N))
    return correct / N

# -----------------------------
# Main
# -----------------------------
def main():
    print("Loading Filt:", MAT_PATH_FILT)
    ts_filt = load_timeseries_from_mat(MAT_PATH_FILT)
    print("Loading DC  :", MAT_PATH_DC)
    ts_dc   = load_timeseries_from_mat(MAT_PATH_DC)

    # Align on min time length
    N = ts_filt.shape[0]
    T = min(ts_filt.shape[1], ts_dc.shape[1])
    P = ts_filt.shape[2]
    print(f"Data shape: N={N}, T={T}, P={P}")

    for dmin in DURATIONS_MIN:
        frames = int(round((dmin*60)/TR))
        if frames < MIN_T or frames > T:
            continue
        FC_gallery = np.array([compute_fc(ts_filt[i,:frames,:]) for i in range(N)])
        FC_query   = np.array([compute_fc(ts_dc[i,:frames,:])   for i in range(N)])
        for m in ["pearson","euclidean"]:
            rate = identification_rate(FC_gallery, FC_query, metric=m)
            print(f"[{m:9s}] duration={dmin:>4.1f} min -> ID = {rate:.3f}")

if __name__ == "__main__":
    main()

# %%
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross-preprocessing identification (ADNI)
Gallery = ts_adni_filt.mat
Query   = ts_adni_dc.mat

Distances: Pearson, Euclidean (as provided by user).
"""

import numpy as np
import scipy.io as sio

# -----------------------------
# CONFIG
# -----------------------------
MAT_PATH_FILT = r"D:\Research AU\Alzheimer FCNs\Alzheimer FCNs\ts_adni_filt.mat"
MAT_PATH_DC   = r"D:\Research AU\Alzheimer FCNs\Alzheimer FCNs\ts_adni_dc.mat"
TR = 0.72  # seconds per frame (change if needed)
DURATIONS_MIN = [0.4, 0.6, 0.8, 1.0, 1.2]
MIN_T = 30
RIDGE_EPS = 1e-3

# -----------------------------
# Distance functions (exactly as requested, with a small NaN guard)
# -----------------------------
def compute_pearson_distance(X, Y):
    X_vec = X.flatten()
    Y_vec = Y.flatten()
    # Guard: if lengths differ, raise a clear error
    if X_vec.size != Y_vec.size:
        raise ValueError(f"Cannot correlate arrays of different lengths: {X_vec.size} vs {Y_vec.size}")
    r = np.corrcoef(X_vec, Y_vec)[0, 1]
    if not np.isfinite(r):  # catch NaN from constant vectors
        return 1.0
    return 1 - r

def compute_euclidean_distance(X, Y):
    X_vec = X.flatten()
    Y_vec = Y.flatten()
    if X_vec.size != Y_vec.size:
        raise ValueError(f"Cannot subtract arrays of different lengths: {X_vec.size} vs {Y_vec.size}")
    return np.linalg.norm(X_vec - Y_vec)

# -----------------------------
# FC computation
# -----------------------------
def compute_fc(ts, ridge_eps=RIDGE_EPS):
    """Pearson correlation FC (SPD-regularized). ts shape: (T, P)."""
    X = ts - ts.mean(axis=0, keepdims=True)
    std = X.std(axis=0, ddof=1, keepdims=True)
    std[std == 0] = 1.0
    X = X / std
    C = np.corrcoef(X, rowvar=False)
    C = 0.5 * (C + C.T)
    C = C + ridge_eps * np.eye(C.shape[0])
    return C

# -----------------------------
# Robust .mat loader → (N,T,P)
# -----------------------------
def load_timeseries_from_mat(path):
    # Try several arg styles for SciPy versions
    try:
        mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    except TypeError:
        try:
            mat = sio.loadmat(path, squeeze_me=True, simplify_cells=True)
        except TypeError:
            mat = sio.loadmat(path, squeeze_me=True)

    # Find a plausible key
    key = None
    for k, v in mat.items():
        if k.startswith("__"): 
            continue
        if isinstance(v, np.ndarray) and v.ndim >= 2:
            key = k
            break
    if key is None:
        raise ValueError(f"No time-series array found in {path}")

    data = np.asarray(mat[key])

    # Handle common shapes
    if data.ndim == 3:
        # try (N,T,P) or (N,P,T)
        if data.shape[1] >= 30:  # likely T in axis=1
            return data.astype(np.float64)
        else:
            return np.transpose(data, (0, 2, 1)).astype(np.float64)
    elif data.ndim == 2:
        # Single subject -> make N=1
        ts = data.astype(np.float64)
        if ts.shape[0] < ts.shape[1]:
            ts = ts.T
        return ts[np.newaxis, :, :]
    else:
        raise ValueError(f"Unsupported array shape in {path}: {data.shape}")

# -----------------------------
# 1-NN identification
# -----------------------------
def identification_rate(FC_gallery, FC_query, metric="pearson"):
    N = FC_gallery.shape[0]
    D = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            if metric == "pearson":
                d = compute_pearson_distance(FC_query[i], FC_gallery[j])
            elif metric == "euclidean":
                d = compute_euclidean_distance(FC_query[i], FC_gallery[j])
            else:
                raise ValueError("Unknown metric")
            D[i, j] = d
    nn = np.argmin(D, axis=1)
    return float(np.sum(nn == np.arange(N))) / N

# -----------------------------
# Main
# -----------------------------
def main():
    print("Loading:", MAT_PATH_FILT)
    ts_filt = load_timeseries_from_mat(MAT_PATH_FILT)
    print("Loading:", MAT_PATH_DC)
    ts_dc   = load_timeseries_from_mat(MAT_PATH_DC)

    # Align subjects (assumed same ordering), time, and ROIs
    N = min(ts_filt.shape[0], ts_dc.shape[0])
    T = min(ts_filt.shape[1], ts_dc.shape[1])
    P = min(ts_filt.shape[2], ts_dc.shape[2])

    # Slice both to the same (N,T,P)
    ts_filt = ts_filt[:N, :T, :P]
    ts_dc   = ts_dc[:N, :T, :P]

    total_min = T * TR / 60.0
    print(f"Aligned shapes: filt={ts_filt.shape}, dc={ts_dc.shape}  -> total_len={total_min:.3f} min")

    for dmin in sorted(DURATIONS_MIN):
        frames = int(round((dmin * 60.0) / TR))
        if frames < MIN_T or frames > T:
            continue

        # Build FCs for the same number of frames from start
        FC_gallery = np.array([compute_fc(ts_filt[i, :frames, :]) for i in range(N)])
        FC_query   = np.array([compute_fc(ts_dc[i,   :frames, :]) for i in range(N)])

        for m in ("pearson", "euclidean"):
            rate = identification_rate(FC_gallery, FC_query, metric=m)
            print(f"[{m:9s}] duration={dmin:>4.1f} min -> ID = {rate:.3f}")

if __name__ == "__main__":
    main()

# %%
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross-preprocessing identification (ADNI) with Alpha-Z BW divergence.
Gallery = ts_adni_filt.mat
Query   = ts_adni_dc.mat

Requires:
    pip install spd-metrics-id
"""

import numpy as np
import scipy.io as sio

# -----------------------------
# CONFIG
# -----------------------------
MAT_PATH_FILT = r"D:\Research AU\Alzheimer FCNs\Alzheimer FCNs\ts_adni_filt.mat"
MAT_PATH_DC   = r"D:\Research AU\Alzheimer FCNs\Alzheimer FCNs\ts_adni_dc.mat"

TR = 0.72                                  # seconds per frame (set correctly for your data)
DURATIONS_MIN = [0.4, 0.6, 0.8, 1.0, 1.2]  # try shorter windows to avoid trivial 1.0 ID
MIN_T = 30                                  # minimum frames to compute FC
RIDGE_EPS = 1e-3                            # small SPD regularization for FCs

# Alpha-Z parameters
ALPHA = 0.99
Z = 1.0

# -----------------------------
# Alpha-Z distance
# -----------------------------
try:
    from spd_metrics_id.distance import alpha_z_bw
except Exception as e:
    raise RuntimeError(
        "Alpha-Z requires 'spd-metrics-id'. Install it with:\n"
        "    pip install spd-metrics-id\n"
        f"Import error: {e}"
    )

def alpha_z_distance(A: np.ndarray, B: np.ndarray, alpha: float = ALPHA, z: float = Z) -> float:
    # A, B must be SPD; our FC builder adds a tiny ridge and symmetrizes
    return float(alpha_z_bw(A, B, alpha=alpha, z=z))

# -----------------------------
# FC computation
# -----------------------------
def compute_fc(ts, ridge_eps=RIDGE_EPS):
    """Pearson correlation FC (SPD-regularized). ts shape: (T, P)."""
    X = ts - ts.mean(axis=0, keepdims=True)
    std = X.std(axis=0, ddof=1, keepdims=True)
    std[std == 0] = 1.0
    X = X / std
    C = np.corrcoef(X, rowvar=False)
    C = 0.5 * (C + C.T)
    C = C + ridge_eps * np.eye(C.shape[0])
    return C

# -----------------------------
# Robust .mat loader → (N,T,P)
# -----------------------------
def load_timeseries_from_mat(path):
    # handle different SciPy versions
    try:
        mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    except TypeError:
        try:
            mat = sio.loadmat(path, squeeze_me=True, simplify_cells=True)
        except TypeError:
            mat = sio.loadmat(path, squeeze_me=True)

    key = None
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.ndim >= 2:
            key = k
            break
    if key is None:
        raise ValueError(f"No time-series array found in {path}")

    data = np.asarray(mat[key])
    if data.ndim == 3:
        # guess (N,T,P) vs (N,P,T)
        if data.shape[1] >= 30:
            return data.astype(np.float64)
        else:
            return np.transpose(data, (0, 2, 1)).astype(np.float64)
    elif data.ndim == 2:
        ts = data.astype(np.float64)
        if ts.shape[0] < ts.shape[1]:
            ts = ts.T
        return ts[np.newaxis, :, :]
    else:
        raise ValueError(f"Unsupported array shape in {path}: {data.shape}")

# -----------------------------
# 1-NN identification (Alpha-Z)
# -----------------------------
def identification_rate_alphaZ(FC_gallery, FC_query, alpha=ALPHA, z=Z):
    N = FC_gallery.shape[0]
    D = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        Ai = FC_query[i]
        for j in range(N):
            Bj = FC_gallery[j]
            D[i, j] = alpha_z_distance(Ai, Bj, alpha=alpha, z=z)
    nn = np.argmin(D, axis=1)
    return float(np.sum(nn == np.arange(N))) / N

# -----------------------------
# Main
# -----------------------------
def main():
    print("Loading:", MAT_PATH_FILT)
    ts_filt = load_timeseries_from_mat(MAT_PATH_FILT)
    print("Loading:", MAT_PATH_DC)
    ts_dc   = load_timeseries_from_mat(MAT_PATH_DC)

    # Align N, T, P (take minima)
    N = min(ts_filt.shape[0], ts_dc.shape[0])
    T = min(ts_filt.shape[1], ts_dc.shape[1])
    P = min(ts_filt.shape[2], ts_dc.shape[2])

    ts_filt = ts_filt[:N, :T, :P]
    ts_dc   = ts_dc[:N, :T, :P]

    total_min = T * TR / 60.0
    print(f"Aligned shapes: filt={ts_filt.shape}, dc={ts_dc.shape}  |  total_len={total_min:.3f} min")
    print(f"Alpha-Z params: alpha={ALPHA}, z={Z}")

    for dmin in sorted(DURATIONS_MIN):
        frames = int(round((dmin * 60.0) / TR))
        if frames < MIN_T or frames > T:
            continue

        # Build FCs from the same number of frames for both preprocs
        FC_gallery = np.array([compute_fc(ts_filt[i, :frames, :]) for i in range(N)])
        FC_query   = np.array([compute_fc(ts_dc[i,   :frames, :]) for i in range(N)])

        rate = identification_rate_alphaZ(FC_gallery, FC_query, alpha=ALPHA, z=Z)
        print(f"[alphaZ] duration={dmin:>4.1f} min -> ID = {rate:.3f}")

if __name__ == "__main__":
    main()

# %%
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full-length (all frames) cross-preprocessing identification with Alpha-Z BW.
Pairs each matching key: filt:key  <->  dc:key
Reports 1-NN ID accuracy per pair (percentage). High ~= good alignment.

Requirements:
    pip install numpy scipy spd-metrics-id
"""

import numpy as np
import scipy.io as sio

# -----------------------------
# CONFIG
# -----------------------------
MAT_PATH_FILT = r"D:\Research AU\Alzheimer FCNs\Alzheimer FCNs\ts_adni_filt.mat"
MAT_PATH_DC   = r"D:\Research AU\Alzheimer FCNs\Alzheimer FCNs\ts_adni_dc.mat"

# matching key pairs (left from *_filt.mat, right from *_dc.mat)
KEY_PAIRS = [
    ("ts_adni_filt_controls1", "ts_adni_dc_controls1"),
    ("ts_adni_filt_EMCI1",     "ts_adni_dc_EMCI1"),
    ("ts_adni_filt_LMCI1",     "ts_adni_dc_LMCI1"),
    ("ts_adni_filt_AD1",       "ts_adni_dc_AD1"),
]

# Alpha-Z parameters
ALPHA = 0.99
Z     = 1.0

# tiny ridge to keep FC SPD
RIDGE_EPS = 1e-3

# -----------------------------
# Alpha-Z distance
# -----------------------------
try:
    from spd_metrics_id.distance import alpha_z_bw
except Exception as e:
    raise RuntimeError(
        "Alpha-Z requires 'spd-metrics-id'. Install it with:\n"
        "    pip install spd-metrics-id\n"
        f"Import error: {e}"
    )

def alpha_z_distance(A: np.ndarray, B: np.ndarray, alpha: float = ALPHA, z: float = Z) -> float:
    return float(alpha_z_bw(A, B, alpha=alpha, z=z))

# -----------------------------
# Helpers
# -----------------------------
def compute_fc(ts: np.ndarray, ridge_eps: float = RIDGE_EPS) -> np.ndarray:
    """Pearson correlation FC (SPD-regularized). ts: (T, P)."""
    X = ts - ts.mean(axis=0, keepdims=True)
    std = X.std(axis=0, ddof=1, keepdims=True)
    std[std == 0] = 1.0
    X = X / std
    C = np.corrcoef(X, rowvar=False)
    C = 0.5 * (C + C.T) + ridge_eps * np.eye(C.shape[0])
    return C

def load_array_by_key(path: str, key: str) -> np.ndarray:
    """Load a specific array by key and coerce to (N, T, P)."""
    # robust loadmat (handles SciPy arg changes)
    try:
        mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    except TypeError:
        try:
            mat = sio.loadmat(path, squeeze_me=True, simplify_cells=True)
        except TypeError:
            mat = sio.loadmat(path, squeeze_me=True)

    if key not in mat:
        # helpful error listing available keys
        user_keys = [k for k in mat.keys() if not k.startswith("__")]
        raise KeyError(f"Key '{key}' not found in {path}. Available: {user_keys}")

    arr = np.asarray(mat[key])
    if arr.ndim == 3:
        # (N,T,P) or (N,P,T)
        if arr.shape[1] >= 30:
            return arr.astype(np.float64)
        return np.transpose(arr, (0, 2, 1)).astype(np.float64)
    if arr.ndim == 2:
        # single subject -> (1,T,P)
        if arr.shape[0] < arr.shape[1]:
            arr = arr.T
        return arr[np.newaxis, :, :].astype(np.float64)
    raise ValueError(f"Unexpected shape for {key} in {path}: {arr.shape}")

def identification_rate_alphaZ(FC_gallery: np.ndarray, FC_query: np.ndarray) -> float:
    """1-NN ID rate using Alpha-Z distance."""
    N = FC_gallery.shape[0]
    D = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        Ai = FC_query[i]
        for j in range(N):
            D[i, j] = alpha_z_distance(Ai, FC_gallery[j])
    pred = np.argmin(D, axis=1)
    return float(np.sum(pred == np.arange(N))) / N

# -----------------------------
# Main
# -----------------------------
def main():
    print("Alpha-Z params:", dict(alpha=ALPHA, z=Z))
    for kf, kd in KEY_PAIRS:
        print(f"\nPair: {kf}  <->  {kd}")
        ts_f = load_array_by_key(MAT_PATH_FILT, kf)   # (N,T,P)
        ts_d = load_array_by_key(MAT_PATH_DC,   kd)   # (N,T,P)

        # Align shapes conservatively
        N = min(ts_f.shape[0], ts_d.shape[0])
        T = min(ts_f.shape[1], ts_d.shape[1])
        P = min(ts_f.shape[2], ts_d.shape[2])
        ts_f = ts_f[:N, :T, :P]
        ts_d = ts_d[:N, :T, :P]

        # Build full-length FCs (all frames)
        FC_f = np.array([compute_fc(ts_f[i]) for i in range(N)])
        FC_d = np.array([compute_fc(ts_d[i]) for i in range(N)])

        # 1-NN with Alpha-Z (gallery=filt, query=dc)
        acc = identification_rate_alphaZ(FC_gallery=FC_f, FC_query=FC_d)
        print(f"Subjects: N={N}, Frames={T}, ROIs={P}")
        print(f"Cross-preproc 1-NN ID (Alpha-Z): {acc*100:.2f}%  (chance ≈ {100.0/N:.2f}%)")

if __name__ == "__main__":
    main()

# %%
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full-length cross-preprocessing identification (gallery=filt, query=dc)
using Euclidean (and Pearson for comparison). Includes optional Hungarian
reorder if a pair's ID looks low.

Requirements:
    pip install numpy scipy matplotlib
"""

import numpy as np
import scipy.io as sio
from scipy.optimize import linear_sum_assignment

# -----------------------------
# CONFIG
# -----------------------------
MAT_PATH_FILT = r"D:\Research AU\Alzheimer FCNs\Alzheimer FCNs\ts_adni_filt.mat"
MAT_PATH_DC   = r"D:\Research AU\Alzheimer FCNs\Alzheimer FCNs\ts_adni_dc.mat"

# matching key pairs (same cohort on both sides)
KEY_PAIRS = [
    ("ts_adni_filt_controls1", "ts_adni_dc_controls1"),  # P ≈ 35
    ("ts_adni_filt_EMCI1",     "ts_adni_dc_EMCI1"),      # P ≈ 34
    ("ts_adni_filt_LMCI1",     "ts_adni_dc_LMCI1"),      # P ≈ 34
    ("ts_adni_filt_AD1",       "ts_adni_dc_AD1"),        # P ≈ 29
]

RIDGE_EPS = 1e-3              # tiny SPD jitter for FCs
RUN_HUNGARIAN_IF_BELOW = 0.95 # try reordering if ID < 95%

np.random.seed(1337)

# -----------------------------
# Helpers
# -----------------------------
def load_array_by_key(path: str, key: str) -> np.ndarray:
    """Load specific array by key and coerce to (N,T,P)."""
    try:
        mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    except TypeError:
        try:
            mat = sio.loadmat(path, squeeze_me=True, simplify_cells=True)
        except TypeError:
            mat = sio.loadmat(path, squeeze_me=True)
    if key not in mat:
        user_keys = [k for k in mat.keys() if not k.startswith("__")]
        raise KeyError(f"Key '{key}' not found in {path}. Available keys: {user_keys}")
    arr = np.asarray(mat[key])
    if arr.ndim == 3:
        # (N,T,P) or (N,P,T)
        if arr.shape[1] >= 30:
            return arr.astype(np.float64)
        return np.transpose(arr, (0, 2, 1)).astype(np.float64)
    if arr.ndim == 2:
        # single subject -> (1,T,P)
        if arr.shape[0] < arr.shape[1]:
            arr = arr.T
        return arr[np.newaxis, :, :].astype(np.float64)
    raise ValueError(f"Unexpected shape for {key} in {path}: {arr.shape}")

def compute_fc(ts: np.ndarray, ridge_eps: float = RIDGE_EPS) -> np.ndarray:
    """Pearson correlation FC (SPD-regularized). ts: (T,P)."""
    X = ts - ts.mean(axis=0, keepdims=True)
    std = X.std(axis=0, ddof=1, keepdims=True); std[std == 0] = 1.0
    X = X / std
    C = np.corrcoef(X, rowvar=False)
    C = 0.5 * (C + C.T) + ridge_eps * np.eye(C.shape[0])
    return C

def upper_tri_vec(M: np.ndarray, k: int = 0) -> np.ndarray:
    iu = np.triu_indices_from(M, k=k)
    return M[iu].astype(float)
# -----------------------------
def pearson_distance(X, Y):
    X_vec = X.flatten()
    Y_vec = Y.flatten()
    r = np.corrcoef(X_vec, Y_vec)[0, 1]
    return 1 - r

def euclidean_distance(X, Y):
    X_vec = X.flatten()
    Y_vec = Y.flatten()
    return np.linalg.norm(X_vec - Y_vec)

def build_D(FC_gallery: np.ndarray, FC_query: np.ndarray, metric: str) -> np.ndarray:
    N = FC_gallery.shape[0]
    D = np.zeros((N, N), dtype=float)
    for i in range(N):
        Ai = FC_query[i]
        for j in range(N):
            Bj = FC_gallery[j]
            if metric == "euclidean":
                D[i, j] = euclidean_distance(Ai, Bj)
            elif metric == "pearson":
                D[i, j] = pearson_distance(Ai, Bj)
            else:
                raise ValueError(metric)
    return D

def id_rate_from_D(D: np.ndarray) -> float:
    pred = np.argmin(D, axis=1)
    return float(np.mean(pred == np.arange(D.shape[0])))

# -----------------------------
# Main
# -----------------------------
def main():
    for kf, kd in KEY_PAIRS:
        print(f"\nPair: {kf}  <->  {kd}")
        try:
            ts_f = load_array_by_key(MAT_PATH_FILT, kf)  # (N,T,P)
            ts_d = load_array_by_key(MAT_PATH_DC,   kd)  # (N,T,P)

            # Align by shape (use full length)
            N = min(ts_f.shape[0], ts_d.shape[0])
            T = min(ts_f.shape[1], ts_d.shape[1])
            P = min(ts_f.shape[2], ts_d.shape[2])
            ts_f = ts_f[:N, :T, :P]
            ts_d = ts_d[:N, :T, :P]

            # Full-length FCs
            FC_f = np.array([compute_fc(ts_f[i]) for i in range(N)])
            FC_d = np.array([compute_fc(ts_d[i]) for i in range(N)])

            # Baseline Euclidean & Pearson
            D_eu = build_D(FC_f, FC_d, metric="euclidean")
            D_pe = build_D(FC_f, FC_d, metric="pearson")
            acc_eu = id_rate_from_D(D_eu)
            acc_pe = id_rate_from_D(D_pe)

            print(f"Subjects: N={N}, Frames={T}, ROIs={P}")
            print(f"Euclidean 1-NN ID: {acc_eu*100:.2f}%   (chance ≈ {100.0/N:.2f}%)")
            print(f"Pearson   1-NN ID: {acc_pe*100:.2f}%   (sanity check)")

            # Optional: if a pair is <95%, try Hungarian reorder (on Pearson cost)
            if acc_eu < RUN_HUNGARIAN_IF_BELOW or acc_pe < RUN_HUNGARIAN_IF_BELOW:
                row_ind, col_ind = linear_sum_assignment(D_pe)  # minimize Pearson distance sum
                FC_d_re = FC_d[col_ind]

                D_eu_re = build_D(FC_f, FC_d_re, metric="euclidean")
                D_pe_re = build_D(FC_f, FC_d_re, metric="pearson")

                acc_eu_re = id_rate_from_D(D_eu_re)
                acc_pe_re = id_rate_from_D(D_pe_re)

                agree = int(np.sum(col_ind == np.arange(N)))
                print(f"Hungarian reorder: fixed={N - agree} subjects (diag agree before={agree})")
                print(f" → Euclidean after align: {acc_eu_re*100:.2f}%")
                print(f" → Pearson  after align: {acc_pe_re*100:.2f}%")

        except Exception as e:
            print(f"[ERROR] {kf} vs {kd}: {e}")

if __name__ == "__main__":
    main()

# %%
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full-length (all frames) cross-preprocessing identification with
Affine-Invariant (AI) Riemannian distance on Pearson FCs.

Pairs each matching key: filt:key  <->  dc:key
Reports 1-NN ID accuracy per pair (percentage).

Requirements:
    pip install numpy scipy
"""

import numpy as np
import scipy.io as sio
from scipy.linalg import eigh

# -----------------------------
# CONFIG
# -----------------------------
MAT_PATH_FILT = r"D:\Research AU\Alzheimer FCNs\Alzheimer FCNs\ts_adni_filt.mat"
MAT_PATH_DC   = r"D:\Research AU\Alzheimer FCNs\Alzheimer FCNs\ts_adni_dc.mat"

# matching key pairs (left from *_filt.mat, right from *_dc.mat)
KEY_PAIRS = [
    ("ts_adni_filt_controls1", "ts_adni_dc_controls1"),
    ("ts_adni_filt_EMCI1",     "ts_adni_dc_EMCI1"),
    ("ts_adni_filt_LMCI1",     "ts_adni_dc_LMCI1"),
    ("ts_adni_filt_AD1",       "ts_adni_dc_AD1"),
]

# Numerical tolerance used **inside the distance only** (no FC regularization)
EIG_FLOOR = 1e-12  # set to 0.0 if you want to be strict (may raise errors)

# -----------------------------
# Helpers
# -----------------------------
def load_array_by_key(path: str, key: str) -> np.ndarray:
    """Load a specific array by key and coerce to (N, T, P)."""
    try:
        mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    except TypeError:
        try:
            mat = sio.loadmat(path, squeeze_me=True, simplify_cells=True)
        except TypeError:
            mat = sio.loadmat(path, squeeze_me=True)

    if key not in mat:
        user_keys = [k for k in mat.keys() if not k.startswith("__")]
        raise KeyError(f"Key '{key}' not found in {path}. Available: {user_keys}")

    arr = np.asarray(mat[key])
    if arr.ndim == 3:
        # (N,T,P) or (N,P,T)
        if arr.shape[1] >= 30:
            return arr.astype(np.float64)
        return np.transpose(arr, (0, 2, 1)).astype(np.float64)
    if arr.ndim == 2:
        # single subject -> (1,T,P)
        if arr.shape[0] < arr.shape[1]:
            arr = arr.T
        return arr[np.newaxis, :, :].astype(np.float64)
    raise ValueError(f"Unexpected shape for {key} in {path}: {arr.shape}")

def compute_fc_no_ridge(ts: np.ndarray) -> np.ndarray:
    """Pearson correlation FC (NO ridge/jitter). ts: (T, P)."""
    X = ts - ts.mean(axis=0, keepdims=True)
    std = X.std(axis=0, ddof=1, keepdims=True)
    std[std == 0] = 1.0
    X = X / std
    C = np.corrcoef(X, rowvar=False)
    # Symmetrize numerically; NO ridge added
    return 0.5 * (C + C.T)

def ai_distance(A: np.ndarray, B: np.ndarray, eig_floor: float = EIG_FLOOR) -> float:
    """
    Affine-Invariant Riemannian metric:
        d(A,B) = || log( A^{-1/2} B A^{-1/2} ) ||_F
    Assumes A,B are SPD. Uses a tiny eigenvalue floor only inside the computation
    for numerical safety (does not modify A or B externally).
    """
    # Force exact symmetry numerically
    A = 0.5 * (A + A.T)
    B = 0.5 * (B + B.T)

    # Eigendecomp of A
    wa, Va = eigh(A)
    if np.any(wa <= 0):
        if eig_floor > 0:
            wa = np.clip(wa, eig_floor, None)
        else:
            raise ValueError("A is not SPD (non-positive eigenvalues) and eig_floor=0.")
    A_inv_sqrt = Va @ np.diag(wa**-0.5) @ Va.T

    # Congruence transform
    C = A_inv_sqrt @ B @ A_inv_sqrt
    C = 0.5 * (C + C.T)

    # Eigendecomp of C, then log of eigenvalues
    wc, _ = eigh(C)
    if np.any(wc <= 0):
        if eig_floor > 0:
            wc = np.clip(wc, eig_floor, None)
        else:
            raise ValueError("A^{-1/2} B A^{-1/2} not SPD and eig_floor=0.")
    return float(np.linalg.norm(np.log(wc)))

def identification_rate_AI(FC_gallery: np.ndarray, FC_query: np.ndarray) -> float:
    """1-NN ID rate using AI distance."""
    N = FC_gallery.shape[0]
    D = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        Ai = FC_query[i]
        for j in range(N):
            D[i, j] = ai_distance(Ai, FC_gallery[j])
    pred = np.argmin(D, axis=1)
    return float(np.mean(pred == np.arange(N)))

# -----------------------------
# Main
# -----------------------------
def main():
    for kf, kd in KEY_PAIRS:
        print(f"\nPair: {kf}  <->  {kd}")
        ts_f = load_array_by_key(MAT_PATH_FILT, kf)   # (N,T,P)
        ts_d = load_array_by_key(MAT_PATH_DC,   kd)   # (N,T,P)

        # Align shapes conservatively
        N = min(ts_f.shape[0], ts_d.shape[0])
        T = min(ts_f.shape[1], ts_d.shape[1])
        P = min(ts_f.shape[2], ts_d.shape[2])
        ts_f = ts_f[:N, :T, :P]
        ts_d = ts_d[:N, :T, :P]

        # Build full-length FCs (NO ridge)
        FC_f = np.array([compute_fc_no_ridge(ts_f[i]) for i in range(N)])
        FC_d = np.array([compute_fc_no_ridge(ts_d[i]) for i in range(N)])

        # 1-NN with AI (gallery=filt, query=dc)
        acc = identification_rate_AI(FC_gallery=FC_f, FC_query=FC_d)
        print(f"Subjects: N={N}, Frames={T}, ROIs={P}")
        print(f"Cross-preproc 1-NN ID (AI): {acc*100:.2f}%  (chance ≈ {100.0/N:.2f}%)")

if __name__ == "__main__":
    main()

# %%
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full-length cross-preprocessing identification (gallery=filt, query=dc)
with Pearson & Euclidean distances on *raw* Pearson FCs (NO ridge).

Distances use the vectorized upper triangle (diagonal excluded).

Requirements:
    pip install numpy scipy
"""

import numpy as np
import scipy.io as sio
# from scipy.optimize import linear_sum_assignment  # enable if you want Hungarian

# -----------------------------
# CONFIG
# -----------------------------
MAT_PATH_FILT = r"D:\Research AU\Alzheimer FCNs\Alzheimer FCNs\ts_adni_filt.mat"
MAT_PATH_DC   = r"D:\Research AU\Alzheimer FCNs\Alzheimer FCNs\ts_adni_dc.mat"

KEY_PAIRS = [
    ("ts_adni_filt_controls1", "ts_adni_dc_controls1"),
    ("ts_adni_filt_EMCI1",     "ts_adni_dc_EMCI1"),
    ("ts_adni_filt_LMCI1",     "ts_adni_dc_LMCI1"),
    ("ts_adni_filt_AD1",       "ts_adni_dc_AD1"),
]

RUN_HUNGARIAN = False  # set True to try subject reordering if needed

# -----------------------------
# Loading & FC (NO RIDGE)
# -----------------------------
def load_array_by_key(path: str, key: str) -> np.ndarray:
    """Load a specific array by key and coerce to (N, T, P)."""
    try:
        mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    except TypeError:
        try:
            mat = sio.loadmat(path, squeeze_me=True, simplify_cells=True)
        except TypeError:
            mat = sio.loadmat(path, squeeze_me=True)

    if key not in mat:
        user_keys = [k for k in mat if not k.startswith("__")]
        raise KeyError(f"Key '{key}' not found in {path}. Available: {user_keys}")

    arr = np.asarray(mat[key])
    if arr.ndim == 3:
        # (N,T,P) or (N,P,T)
        if arr.shape[1] >= 30:
            return arr.astype(np.float64)
        return np.transpose(arr, (0, 2, 1)).astype(np.float64)
    if arr.ndim == 2:
        # single subject -> (1,T,P)
        if arr.shape[0] < arr.shape[1]:
            arr = arr.T
        return arr[np.newaxis, :, :].astype(np.float64)
    raise ValueError(f"Unexpected shape for {key} in {path}: {arr.shape}")

def compute_fc_raw(ts: np.ndarray) -> np.ndarray:
    """Pearson correlation FC with NO regularization. ts: (T, P)."""
    X = ts - ts.mean(axis=0, keepdims=True)
    std = X.std(axis=0, ddof=1, keepdims=True)
    std[std == 0] = 1.0
    X = X / std
    C = np.corrcoef(X, rowvar=False)
    # Symmetrize numerically; DO NOT add ridge/jitter
    return 0.5 * (C + C.T)

# -----------------------------
# Distances (upper-tri only)
# -----------------------------
# -----------------------------
def pearson_distance(X, Y):
    X_vec = X.flatten()
    Y_vec = Y.flatten()
    r = np.corrcoef(X_vec, Y_vec)[0, 1]
    return 1 - r

def euclidean_distance(X, Y):
    X_vec = X.flatten()
    Y_vec = Y.flatten()
    return np.linalg.norm(X_vec - Y_vec)

def build_D(FC_gallery: np.ndarray, FC_query: np.ndarray, metric: str) -> np.ndarray:
    N = FC_gallery.shape[0]
    D = np.zeros((N, N), dtype=float)
    for i in range(N):
        Ai = FC_query[i]
        for j in range(N):
            Bj = FC_gallery[j]
            if metric == "pearson":
                D[i, j] = pearson_distance(Ai, Bj)
            elif metric == "euclidean":
                D[i, j] = euclidean_distance(Ai, Bj)
            else:
                raise ValueError(metric)
    return D

def id_rate_from_D(D: np.ndarray) -> float:
    pred = np.argmin(D, axis=1)
    return float(np.mean(pred == np.arange(D.shape[0])))

# -----------------------------
# Main
# -----------------------------
def main():
    for kf, kd in KEY_PAIRS:
        print(f"\nPair: {kf}  <->  {kd}")
        ts_f = load_array_by_key(MAT_PATH_FILT, kf)  # (N,T,P)
        ts_d = load_array_by_key(MAT_PATH_DC,   kd)  # (N,T,P)

        # Align shapes (use full length)
        N = min(ts_f.shape[0], ts_d.shape[0])
        T = min(ts_f.shape[1], ts_d.shape[1])
        P = min(ts_f.shape[2], ts_d.shape[2])
        ts_f = ts_f[:N, :T, :P]
        ts_d = ts_d[:N, :T, :P]

        # Build FCs with NO ridge
        FC_f = np.array([compute_fc_raw(ts_f[i]) for i in range(N)])
        FC_d = np.array([compute_fc_raw(ts_d[i]) for i in range(N)])

        # Baseline Pearson & Euclidean
        D_pe = build_D(FC_f, FC_d, metric="pearson")
        D_eu = build_D(FC_f, FC_d, metric="euclidean")
        acc_pe = id_rate_from_D(D_pe)
        acc_eu = id_rate_from_D(D_eu)

        print(f"Subjects: N={N}, Frames={T}, ROIs={P}")
        print(f"Pearson   1-NN ID: {acc_pe*100:.2f}%   (chance ≈ {100.0/N:.2f}%)")
        print(f"Euclidean 1-NN ID: {acc_eu*100:.2f}%   (chance ≈ {100.0/N:.2f}%)")

        # Optional: Hungarian reorder (based on Pearson costs)
        if RUN_HUNGARIAN:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(D_pe)  # minimize sum of Pearson distances
            FC_d_re = FC_d[col_ind]
            D_pe_re = build_D(FC_f, FC_d_re, metric="pearson")
            D_eu_re = build_D(FC_f, FC_d_re, metric="euclidean")
            acc_pe_re = id_rate_from_D(D_pe_re)
            acc_eu_re = id_rate_from_D(D_eu_re)
            agree = int(np.sum(col_ind == np.arange(N)))
            print(f"Hungarian reorder: fixed={N - agree} subjects (diag agree before={agree})")
            print(f" → Pearson   after align: {acc_pe_re*100:.2f}%")
            print(f" → Euclidean after align: {acc_eu_re*100:.2f}%")

if __name__ == "__main__":
    main()

# %%
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Combined-cohorts cross-preprocessing identification (gallery=filt, query=dc)
across controls1 + EMCI1 + LMCI1 + AD1 together.

- No ridge/jitter is added to FCs.
- Distances use the vectorized upper triangle (diag excluded).
- Reports overall ID (N_total) and per-cohort IDs (each cohort's queries vs all galleries).

Requirements:
    pip install numpy scipy
"""

import numpy as np
import scipy.io as sio

# -----------------------------
# CONFIG
# -----------------------------
MAT_PATH_FILT = r"D:\Research AU\Alzheimer FCNs\Alzheimer FCNs\ts_adni_filt.mat"
MAT_PATH_DC   = r"D:\Research AU\Alzheimer FCNs\Alzheimer FCNs\ts_adni_dc.mat"

COHORT_KEYS = [
    ("ts_adni_filt_controls1", "ts_adni_dc_controls1"),  # ~35 ROIs
    ("ts_adni_filt_EMCI1",     "ts_adni_dc_EMCI1"),      # ~34 ROIs
    ("ts_adni_filt_LMCI1",     "ts_adni_dc_LMCI1"),      # ~34 ROIs
    ("ts_adni_filt_AD1",       "ts_adni_dc_AD1"),        # ~29 ROIs
]

np.random.seed(1337)

# -----------------------------
# Loading & FC (NO RIDGE)
# -----------------------------
def load_array_by_key(path: str, key: str) -> np.ndarray:
    """Load a specific array by key and coerce to (N, T, P)."""
    try:
        mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    except TypeError:
        try:
            mat = sio.loadmat(path, squeeze_me=True, simplify_cells=True)
        except TypeError:
            mat = sio.loadmat(path, squeeze_me=True)

    if key not in mat:
        user_keys = [k for k in mat if not k.startswith("__")]
        raise KeyError(f"Key '{key}' not found in {path}. Available: {user_keys}")

    arr = np.asarray(mat[key])
    if arr.ndim == 3:
        # (N,T,P) or (N,P,T)
        if arr.shape[1] >= 30:
            return arr.astype(np.float64)
        return np.transpose(arr, (0, 2, 1)).astype(np.float64)
    if arr.ndim == 2:
        # single subject -> (1,T,P)
        if arr.shape[0] < arr.shape[1]:
            arr = arr.T
        return arr[np.newaxis, :, :].astype(np.float64)
    raise ValueError(f"Unexpected shape for {key} in {path}: {arr.shape}")

def compute_fc_raw(ts: np.ndarray) -> np.ndarray:
    """Pearson correlation FC with NO regularization. ts: (T, P)."""
    X = ts - ts.mean(axis=0, keepdims=True)
    std = X.std(axis=0, ddof=1, keepdims=True)
    std[std == 0] = 1.0
    X = X / std
    C = np.corrcoef(X, rowvar=False)
    return 0.5 * (C + C.T)  # symmetrize only (no ridge)

# -----------------------------
# Distances (upper-tri only)
# -----------------------------
def _upper_tri_vec(M: np.ndarray, k: int = 1) -> np.ndarray:
    iu = np.triu_indices_from(M, k=k)
    return M[iu].astype(float)

def pearson_distance(A: np.ndarray, B: np.ndarray) -> float:
    a = _upper_tri_vec(A, 1); b = _upper_tri_vec(B, 1)
    sa, sb = np.std(a), np.std(b)
    if sa == 0 or sb == 0:
        return 1.0
    r = np.corrcoef(a, b)[0, 1]
    return float(1.0 - r)

def euclidean_distance(A: np.ndarray, B: np.ndarray) -> float:
    a = _upper_tri_vec(A, 1); b = _upper_tri_vec(B, 1)
    return float(np.linalg.norm(a - b))

def build_D(FC_gallery: np.ndarray, FC_query: np.ndarray, metric: str) -> np.ndarray:
    N = FC_gallery.shape[0]
    D = np.zeros((N, N), dtype=float)
    for i in range(N):
        Ai = FC_query[i]
        # vectorize loop over gallery if you like; this is clear & fine for N≈520
        for j in range(N):
            Bj = FC_gallery[j]
            if metric == "pearson":
                D[i, j] = pearson_distance(Ai, Bj)
            elif metric == "euclidean":
                D[i, j] = euclidean_distance(Ai, Bj)
            else:
                raise ValueError(metric)
    return D

def id_rate_from_D(D: np.ndarray, idx_start: int = 0, count: int = None) -> float:
    """
    Rank-1 accuracy. If idx_start/count provided, evaluate only that subset of queries,
    but queries still compete against ALL galleries (global identification).
    """
    N = D.shape[0]
    if count is None:
        rows = range(N)
    else:
        rows = range(idx_start, idx_start + count)
    pred = np.argmin(D, axis=1)
    correct = 0
    for i in rows:
        if pred[i] == i:  # ground truth: gallery index equals query index (stacked same order)
            correct += 1
    return float(correct) / (len(list(rows)))

# -----------------------------
# Main
# -----------------------------
def main():
    # Pass 1: load each cohort pair and record min T/P per pair
    pair_data = []
    T_list, P_list = [], []
    for kf, kd in COHORT_KEYS:
        ts_f = load_array_by_key(MAT_PATH_FILT, kf)  # (N,T,P)
        ts_d = load_array_by_key(MAT_PATH_DC,   kd)  # (N,T,P)
        # align within pair
        Np = min(ts_f.shape[0], ts_d.shape[0])
        Tp = min(ts_f.shape[1], ts_d.shape[1])
        Pp = min(ts_f.shape[2], ts_d.shape[2])
        ts_f = ts_f[:Np, :Tp, :Pp]
        ts_d = ts_d[:Np, :Tp, :Pp]
        pair_data.append((kf, kd, ts_f, ts_d))
        T_list.append(Tp); P_list.append(Pp)

    # Global crop so all cohorts share the same T and P (so we can concatenate)
    T_all = int(min(T_list))
    P_all = int(min(P_list))  # expect 29 across all
    # Build combined arrays & keep index ranges per cohort
    ts_f_all, ts_d_all = [], []
    idx_ranges = []  # list of (label, start, count, P_used)
    cursor = 0
    for (kf, kd, ts_f, ts_d) in pair_data:
        Np = ts_f.shape[0]
        ts_f_all.append(ts_f[:, :T_all, :P_all])
        ts_d_all.append(ts_d[:, :T_all, :P_all])
        idx_ranges.append((kf.replace("ts_adni_filt_", ""), cursor, Np, P_all))
        cursor += Np

    ts_f_all = np.concatenate(ts_f_all, axis=0)  # (N_total, T_all, P_all)
    ts_d_all = np.concatenate(ts_d_all, axis=0)

    N_total, T_used, P_used = ts_f_all.shape
    chance = 100.0 / N_total
    print(f"Combined dataset: N_total={N_total}, T={T_used}, P={P_used}  | chance ≈ {chance:.2f}%")
    for label, start, cnt, _ in idx_ranges:
        print(f"  - {label}: N={cnt}")

    # Build FCs (NO ridge)
    FC_f = np.array([compute_fc_raw(ts_f_all[i]) for i in range(N_total)])
    FC_d = np.array([compute_fc_raw(ts_d_all[i]) for i in range(N_total)])

    # Distances & overall ID
    for metric in ["pearson", "euclidean"]:
        D = build_D(FC_f, FC_d, metric=metric)
        overall = id_rate_from_D(D) * 100.0
        print(f"\n{metric.capitalize()} 1-NN ID (overall): {overall:.2f}%  (chance ≈ {chance:.2f}%)")
        # Per-cohort accuracy (queries from that cohort against ALL galleries)
        for label, start, cnt, _ in idx_ranges:
            acc = id_rate_from_D(D, idx_start=start, count=cnt) * 100.0
            print(f"  • {label:<10s} : {acc:.2f}%")

if __name__ == "__main__":
    main()

# %%
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Combined-cohorts cross-preprocessing identification (gallery=filt, query=dc)
across controls1 + EMCI1 + LMCI1 + AD1 together.

- FCs are raw Pearson correlations (NO ridge/jitter).
- Distances computed on:
    * Pearson: 1 - r (upper triangle, diag excluded)
    * Euclidean: ℓ2 on upper triangle (diag excluded)
    * Alpha-Z BW: SPD divergence (uses an eigenvalue floor INSIDE the distance only)

Outputs overall ID-rate and per-cohort ID-rate (queries from each cohort vs ALL galleries).

Requirements:
    pip install numpy scipy
    # for Alpha-Z:
    pip install spd-metrics-id
"""

import numpy as np
import scipy.io as sio

# -----------------------------
# CONFIG
# -----------------------------
MAT_PATH_FILT = r"D:\Research AU\Alzheimer FCNs\Alzheimer FCNs\ts_adni_filt.mat"
MAT_PATH_DC   = r"D:\Research AU\Alzheimer FCNs\Alzheimer FCNs\ts_adni_dc.mat"

COHORT_KEYS = [
    ("ts_adni_filt_controls1", "ts_adni_dc_controls1"),  # ~35 ROIs
    ("ts_adni_filt_EMCI1",     "ts_adni_dc_EMCI1"),      # ~34 ROIs
    ("ts_adni_filt_LMCI1",     "ts_adni_dc_LMCI1"),      # ~34 ROIs
    ("ts_adni_filt_AD1",       "ts_adni_dc_AD1"),        # ~29 ROIs
]

# Alpha-Z params
ALPHA = 0.99
Z     = 1.0
# Numerical safety INSIDE the alpha-z distance (does NOT modify stored FCs)
ALPHAZ_EIG_FLOOR = 1e-12   # set to 0.0 to be strict (may error if not SPD)

np.random.seed(1337)

# -----------------------------
# Loading & FC (NO RIDGE)
# -----------------------------
def load_array_by_key(path: str, key: str) -> np.ndarray:
    """Load a specific array by key and coerce to (N, T, P)."""
    try:
        mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    except TypeError:
        try:
            mat = sio.loadmat(path, squeeze_me=True, simplify_cells=True)
        except TypeError:
            mat = sio.loadmat(path, squeeze_me=True)

    if key not in mat:
        user_keys = [k for k in mat if not k.startswith("__")]
        raise KeyError(f"Key '{key}' not found in {path}. Available: {user_keys}")

    arr = np.asarray(mat[key])
    if arr.ndim == 3:
        # (N,T,P) or (N,P,T)
        if arr.shape[1] >= 30:
            return arr.astype(np.float64)
        return np.transpose(arr, (0, 2, 1)).astype(np.float64)
    if arr.ndim == 2:
        # single subject -> (1,T,P)
        if arr.shape[0] < arr.shape[1]:
            arr = arr.T
        return arr[np.newaxis, :, :].astype(np.float64)
    raise ValueError(f"Unexpected shape for {key} in {path}: {arr.shape}")

def compute_fc_raw(ts: np.ndarray) -> np.ndarray:
    """Pearson correlation FC with NO regularization. ts: (T, P)."""
    X = ts - ts.mean(axis=0, keepdims=True)
    std = X.std(axis=0, ddof=1, keepdims=True)
    std[std == 0] = 1.0
    X = X / std
    C = np.corrcoef(X, rowvar=False)
    return 0.5 * (C + C.T)  # symmetrize only (no ridge)

# -----------------------------
# Distances (upper-tri only)
# -----------------------------
def _upper_tri_vec(M: np.ndarray, k: int = 1) -> np.ndarray:
    iu = np.triu_indices_from(M, k=k)
    return M[iu].astype(float)

def pearson_distance(A: np.ndarray, B: np.ndarray) -> float:
    a = _upper_tri_vec(A, 1); b = _upper_tri_vec(B, 1)
    sa, sb = np.std(a), np.std(b)
    if sa == 0 or sb == 0:
        return 1.0
    r = np.corrcoef(a, b)[0, 1]
    return float(1.0 - r)

def euclidean_distance(A: np.ndarray, B: np.ndarray) -> float:
    a = _upper_tri_vec(A, 1); b = _upper_tri_vec(B, 1)
    return float(np.linalg.norm(a - b))

# -----------------------------
# Alpha-Z (raw FCs; SPD safety INSIDE distance)
# -----------------------------
_has_alpha_z = False
try:
    from spd_metrics_id.distance import alpha_z_bw as _alpha_z_bw
    _has_alpha_z = True
except Exception:
    _has_alpha_z = False

def _spd_clip(M: np.ndarray, floor: float) -> np.ndarray:
    """Return a symmetrized copy with eigenvalues clipped to >= floor (no external modification)."""
    M = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(M)
    if floor > 0.0:
        w = np.clip(w, floor, None)
    elif np.any(w <= 0):
        raise ValueError("Matrix not SPD and floor=0.")
    return (V @ np.diag(w) @ V.T).astype(np.float64)

def alpha_z_distance(A: np.ndarray, B: np.ndarray, alpha: float = ALPHA, z: float = Z,
                     eig_floor: float = ALPHAZ_EIG_FLOOR) -> float:
    if not _has_alpha_z:
        raise RuntimeError("Alpha-Z requires 'spd-metrics-id' (pip install spd-metrics-id).")
    # work on local SPD-projected copies ONLY for the divergence call
    A_spd = _spd_clip(A, eig_floor)
    B_spd = _spd_clip(B, eig_floor)
    return float(_alpha_z_bw(A_spd, B_spd, alpha=alpha, z=z))

# -----------------------------
# ID helpers
# -----------------------------
def build_D(FC_gallery: np.ndarray, FC_query: np.ndarray, metric: str) -> np.ndarray:
    N = FC_gallery.shape[0]
    D = np.zeros((N, N), dtype=float)
    for i in range(N):
        Ai = FC_query[i]
        for j in range(N):
            Bj = FC_gallery[j]
            if metric == "pearson":
                D[i, j] = pearson_distance(Ai, Bj)
            elif metric == "euclidean":
                D[i, j] = euclidean_distance(Ai, Bj)
            elif metric == "alpha_z":
                D[i, j] = alpha_z_distance(Ai, Bj)
            else:
                raise ValueError(metric)
    return D

def id_rate_from_D(D: np.ndarray, idx_start: int = 0, count: int = None) -> float:
    """
    Rank-1 accuracy. If idx_start/count provided, evaluate only that subset of queries,
    but against ALL galleries (global ID).
    """
    N = D.shape[0]
    if count is None:
        rows = range(N)
    else:
        rows = range(idx_start, idx_start + count)
    pred = np.argmin(D, axis=1)
    correct = sum(1 for i in rows if pred[i] == i)
    return float(correct) / (len(list(rows)))

# -----------------------------
# Main
# -----------------------------
def main():
    # Load all cohort pairs; record per-pair min T/P
    pair_data = []
    T_list, P_list = [], []
    for kf, kd in COHORT_KEYS:
        ts_f = load_array_by_key(MAT_PATH_FILT, kf)
        ts_d = load_array_by_key(MAT_PATH_DC,   kd)
        Np = min(ts_f.shape[0], ts_d.shape[0])
        Tp = min(ts_f.shape[1], ts_d.shape[1])
        Pp = min(ts_f.shape[2], ts_d.shape[2])
        ts_f = ts_f[:Np, :Tp, :Pp]
        ts_d = ts_d[:Np, :Tp, :Pp]
        pair_data.append((kf, kd, ts_f, ts_d))
        T_list.append(Tp); P_list.append(Pp)

    # Global crop so everyone shares the same T,P → concatenate
    T_all = int(min(T_list))
    P_all = int(min(P_list))  # expect 29 across all cohorts
    ts_f_all, ts_d_all = [], []
    idx_ranges = []  # (label, start, count)
    cursor = 0
    for (kf, kd, ts_f, ts_d) in pair_data:
        Np = ts_f.shape[0]
        ts_f_all.append(ts_f[:, :T_all, :P_all])
        ts_d_all.append(ts_d[:, :T_all, :P_all])
        label = kf.replace("ts_adni_filt_", "")
        idx_ranges.append((label, cursor, Np))
        cursor += Np

    ts_f_all = np.concatenate(ts_f_all, axis=0)  # (N_total, T_all, P_all)
    ts_d_all = np.concatenate(ts_d_all, axis=0)

    N_total, T_used, P_used = ts_f_all.shape
    chance = 100.0 / N_total
    print(f"Combined dataset: N_total={N_total}, T={T_used}, P={P_used}  | chance ≈ {chance:.2f}%")
    for label, start, cnt in idx_ranges:
        print(f"  - {label}: N={cnt}")

    # Build FCs (NO ridge)
    FC_f = np.array([compute_fc_raw(ts_f_all[i]) for i in range(N_total)])
    FC_d = np.array([compute_fc_raw(ts_d_all[i]) for i in range(N_total)])

    # Run metrics
    METRICS = ["pearson", "euclidean", "alpha_z"]
    for metric in METRICS:
        if metric == "alpha_z" and not _has_alpha_z:
            print("\n[alpha_z] spd-metrics-id not installed → skipping Alpha-Z.")
            continue

        D = build_D(FC_f, FC_d, metric=metric)
        overall = id_rate_from_D(D) * 100.0
        print(f"\n{metric.capitalize()} 1-NN ID (overall): {overall:.2f}%  (chance ≈ {chance:.2f}%)")
        for label, start, cnt in idx_ranges:
            acc = id_rate_from_D(D, idx_start=start, count=cnt) * 100.0
            print(f"  • {label:<10s} : {acc:.2f}%")

if __name__ == "__main__":
    main()

# %%
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bidirectional cross-preprocessing ID (filt<->dc) per cohort, with mean of both directions.
Metrics: Pearson (1-r) and Euclidean on upper-tri of raw Pearson FCs (NO ridge/jitter).
"""

import numpy as np
import scipy.io as sio

# -----------------------------
# CONFIG
# -----------------------------
MAT_PATH_FILT = r"D:\Research AU\Alzheimer FCNs\Alzheimer FCNs\ts_adni_filt.mat"
MAT_PATH_DC   = r"D:\Research AU\Alzheimer FCNs\Alzheimer FCNs\ts_adni_dc.mat"

COHORT_KEYS = [
    ("ts_adni_filt_controls1", "ts_adni_dc_controls1"),  # ~35 ROIs
    ("ts_adni_filt_EMCI1",     "ts_adni_dc_EMCI1"),      # ~34 ROIs
    ("ts_adni_filt_LMCI1",     "ts_adni_dc_LMCI1"),      # ~34 ROIs
    ("ts_adni_filt_AD1",       "ts_adni_dc_AD1"),        # ~29 ROIs
]

np.random.seed(1337)

# -----------------------------
# Loading & FC (NO RIDGE)
# -----------------------------

def load_array_by_key(path: str, key: str) -> np.ndarray:
    """Load cohort array and return (N, T, P). Handles (N,T,P), (N,P,T), and (T,P,N)."""
    try:
        mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    except TypeError:
        try:
            mat = sio.loadmat(path, squeeze_me=True, simplify_cells=True)
        except TypeError:
            mat = sio.loadmat(path, squeeze_me=True)

    if key not in mat:
        user_keys = [k for k in mat if not k.startswith("__")]
        raise KeyError(f"Key '{key}' not found in {path}. Available: {user_keys}")

    arr = np.asarray(mat[key])
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array for {key}, got shape {arr.shape}")

    a, b, c = arr.shape

    # Case 1: already (N,T,P)
    if a >= 20 and b >= 50 and c >= 20 and b >= c:
        return arr.astype(np.float64)

    # Case 2: (N,P,T)
    if a >= 20 and c >= 50 and b >= 20 and c >= b:
        return np.transpose(arr, (0, 2, 1)).astype(np.float64)

    # Case 3: (T,P,N) — ADNI variant: (130, 200, N)
    if a >= 50 and b >= 50 and c >= 20:
        return np.transpose(arr, (2, 0, 1)).astype(np.float64)

    # Fallback: choose the smallest dim as N, largest as P, remaining as T
    dims = np.array([a, b, c])
    idxN = int(np.argmin(dims))
    idxP = int(np.argmax(dims))
    idxT = int({0, 1, 2} - {idxN, idxP}).pop()
    return np.transpose(arr, (idxN, idxT, idxP)).astype(np.float64)


def compute_fc_raw(ts: np.ndarray) -> np.ndarray:
    """Pearson correlation FC with NO regularization. ts: (T, P)."""
    X = ts - ts.mean(axis=0, keepdims=True)
    std = X.std(axis=0, ddof=1, keepdims=True)
    std[std == 0] = 1.0
    X = X / std
    C = np.corrcoef(X, rowvar=False)
    return 0.5 * (C + C.T)  # symmetrize only

# -----------------------------
# Distances (upper-tri only)
# -----------------------------
def _upper_tri_vec(M: np.ndarray, k: int = 0) -> np.ndarray:
    iu = np.triu_indices_from(M, k=k)
    return M[iu].astype(float)

def pearson_distance(A: np.ndarray, B: np.ndarray) -> float:
    a = _upper_tri_vec(A, 0); b = _upper_tri_vec(B, 0)
    sa, sb = np.std(a), np.std(b)
    if sa == 0 or sb == 0:
        return 1.0
    r = np.corrcoef(a, b)[0, 1]
    return float(1.0 - r)

def euclidean_distance(A: np.ndarray, B: np.ndarray) -> float:
    a = _upper_tri_vec(A, 0); b = _upper_tri_vec(B, 0)
    return float(np.linalg.norm(a - b))

def build_D(FC_gallery: np.ndarray, FC_query: np.ndarray, metric: str) -> np.ndarray:
    N = FC_gallery.shape[0]
    D = np.zeros((N, N), dtype=float)
    for i in range(N):
        Ai = FC_query[i]
        for j in range(N):
            Bj = FC_gallery[j]
            if metric == "pearson":
                D[i, j] = pearson_distance(Ai, Bj)
            elif metric == "euclidean":
                D[i, j] = euclidean_distance(Ai, Bj)
            else:
                raise ValueError(metric)
    return D

def id_rate_from_D(D: np.ndarray) -> float:
    pred = np.argmin(D, axis=1)
    return float(np.mean(pred == np.arange(D.shape[0])))

# -----------------------------
# Main
# -----------------------------
def main():
    results = []  # list of dict rows

    for kf, kd in COHORT_KEYS:
        label = kf.replace("ts_adni_filt_", "")

        # Load & align shapes per cohort
        ts_f = load_array_by_key(MAT_PATH_FILT, kf)  # (N,T,P)
        ts_d = load_array_by_key(MAT_PATH_DC,   kd)  # (N,T,P)
        N = min(ts_f.shape[0], ts_d.shape[0])
        T = min(ts_f.shape[1], ts_d.shape[1])
        P = min(ts_f.shape[2], ts_d.shape[2])
        ts_f = ts_f[:N, :T, :P]
        ts_d = ts_d[:N, :T, :P]
        chance = 100.0 / N

        # FCs (raw)
        FC_f = np.array([compute_fc_raw(ts_f[i]) for i in range(N)])
        FC_d = np.array([compute_fc_raw(ts_d[i]) for i in range(N)])

        for metric in ["pearson", "euclidean"]:
            # Direction 1: gallery=filt, query=dc
            D12 = build_D(FC_f, FC_d, metric)
            acc12 = id_rate_from_D(D12) * 100.0

            # Direction 2: gallery=dc, query=filt
            D21 = build_D(FC_d, FC_f, metric)
            acc21 = id_rate_from_D(D21) * 100.0

            mean_acc = 0.5 * (acc12 + acc21)

            print(f"\n[{label}] N={N}, T={T}, P={P}, chance≈{chance:.2f}%")
            print(f"  {metric.capitalize()}  →  filt←dc: {acc12:.2f}% | dc←filt: {acc21:.2f}% | mean: {mean_acc:.2f}%")

            results.append(dict(
                cohort=label, N=N, T=T, P=P, metric=metric,
                acc_filt_as_gallery=acc12, acc_dc_as_gallery=acc21, acc_mean=mean_acc
            ))

    # (Optional) summarize macro means
    for metric in ["pearson", "euclidean"]:
        means = [r["acc_mean"] for r in results if r["metric"] == metric]
        macro = float(np.mean(means)) if means else float("nan")
        print(f"\nMacro-average (unweighted across cohorts) {metric}: {macro:.2f}%")

if __name__ == "__main__":
    main()

# %%
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bidirectional cross-preprocessing ID (filt <-> dc) per cohort using Alpha-Z BW.
- FCs are raw Pearson correlations (NO ridge/jitter).
- For each cohort, compute:
    * filt as gallery, dc as query  (filt←dc)
    * dc as gallery, filt as query  (dc←filt)
    * mean of the two
- Distances are Alpha-Z BW on SPD matrices; a tiny eigenvalue floor is applied
  INSIDE the distance only for numerical safety (FCs themselves remain unmodified).

Requirements:
    pip install numpy scipy spd-metrics-id
"""

import numpy as np
import scipy.io as sio

# -----------------------------
# CONFIG
# -----------------------------
MAT_PATH_FILT = r"D:\Research AU\Alzheimer FCNs\Alzheimer FCNs\ts_adni_filt.mat"
MAT_PATH_DC   = r"D:\Research AU\Alzheimer FCNs\Alzheimer FCNs\ts_adni_dc.mat"

COHORT_KEYS = [
    ("ts_adni_filt_controls1", "ts_adni_dc_controls1"),
    ("ts_adni_filt_EMCI1",     "ts_adni_dc_EMCI1"),
    ("ts_adni_filt_LMCI1",     "ts_adni_dc_LMCI1"),
    ("ts_adni_filt_AD1",       "ts_adni_dc_AD1"),
]

# Alpha-Z parameters
ALPHA = 0.99
Z     = 1.0

# Numerical safety inside the distance (does NOT change stored FCs)
ALPHAZ_EIG_FLOOR = 1e-12   # set to 0.0 to be strict (may error if FC not SPD)

np.random.seed(1337)

# -----------------------------
# Alpha-Z distance
# -----------------------------
try:
    from spd_metrics_id.distance import alpha_z_bw as _alpha_z_bw
except Exception as e:
    raise RuntimeError(
        "Alpha-Z requires 'spd-metrics-id'. Install it with:\n"
        "    pip install spd-metrics-id\n"
        f"Import error: {e}"
    )

def _spd_clip(M: np.ndarray, floor: float) -> np.ndarray:
    """Return a symmetrized copy with eigenvalues clipped to >= floor (no external modification)."""
    M = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(M)
    if floor > 0.0:
        w = np.clip(w, floor, None)
    elif np.any(w <= 0):
        raise ValueError("Matrix not SPD and eig_floor=0.")
    return (V @ np.diag(w) @ V.T).astype(np.float64)

def alpha_z_distance(A: np.ndarray, B: np.ndarray,
                     alpha: float = ALPHA, z: float = Z,
                     eig_floor: float = ALPHAZ_EIG_FLOOR) -> float:
    A_spd = _spd_clip(A, eig_floor)
    B_spd = _spd_clip(B, eig_floor)
    return float(_alpha_z_bw(A_spd, B_spd, alpha=alpha, z=z))

# -----------------------------
# Loading & FC (NO RIDGE)
# -----------------------------
def load_array_by_key(path: str, key: str) -> np.ndarray:
    """Load cohort array and return (N, T, P). Handles (N,T,P), (N,P,T), and (T,P,N)."""
    try:
        mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    except TypeError:
        try:
            mat = sio.loadmat(path, squeeze_me=True, simplify_cells=True)
        except TypeError:
            mat = sio.loadmat(path, squeeze_me=True)

    if key not in mat:
        user_keys = [k for k in mat if not k.startswith("__")]
        raise KeyError(f"Key '{key}' not found in {path}. Available: {user_keys}")

    arr = np.asarray(mat[key])
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array for {key}, got shape {arr.shape}")

    a, b, c = arr.shape

    # Case 1: already (N,T,P)
    if a >= 20 and b >= 50 and c >= 20 and b >= c:
        return arr.astype(np.float64)

    # Case 2: (N,P,T)
    if a >= 20 and c >= 50 and b >= 20 and c >= b:
        return np.transpose(arr, (0, 2, 1)).astype(np.float64)

    # Case 3: (T,P,N) — ADNI variant: (130, 200, N)
    if a >= 50 and b >= 50 and c >= 20:
        return np.transpose(arr, (2, 0, 1)).astype(np.float64)

    # Fallback: choose the smallest dim as N, largest as P, remaining as T
    dims = np.array([a, b, c])
    idxN = int(np.argmin(dims))
    idxP = int(np.argmax(dims))
    idxT = int({0, 1, 2} - {idxN, idxP}).pop()
    return np.transpose(arr, (idxN, idxT, idxP)).astype(np.float64)

def compute_fc_raw(ts: np.ndarray) -> np.ndarray:
    """Pearson correlation FC with NO regularization. ts: (T, P)."""
    X = ts - ts.mean(axis=0, keepdims=True)
    std = X.std(axis=0, ddof=1, keepdims=True)
    std[std == 0] = 1.0
    X = X / std
    C = np.corrcoef(X, rowvar=False)
    return 0.5 * (C + C.T)  # symmetrize only

# -----------------------------
# ID helpers
# -----------------------------
def build_D_alphaZ(FC_gallery: np.ndarray, FC_query: np.ndarray) -> np.ndarray:
    N = FC_gallery.shape[0]
    D = np.zeros((N, N), dtype=float)
    for i in range(N):
        Ai = FC_query[i]
        for j in range(N):
            Bj = FC_gallery[j]
            D[i, j] = alpha_z_distance(Ai, Bj)
    return D

def id_rate_from_D(D: np.ndarray) -> float:
    pred = np.argmin(D, axis=1)
    return float(np.mean(pred == np.arange(D.shape[0])))

# -----------------------------
# Main
# -----------------------------
def main():
    rows = []
    for kf, kd in COHORT_KEYS:
        label = kf.replace("ts_adni_filt_", "")

        # Load & align shapes per cohort
        ts_f = load_array_by_key(MAT_PATH_FILT, kf)  # (N,T,P)
        ts_d = load_array_by_key(MAT_PATH_DC,   kd)  # (N,T,P)
        N = min(ts_f.shape[0], ts_d.shape[0])
        T = min(ts_f.shape[1], ts_d.shape[1])
        P = min(ts_f.shape[2], ts_d.shape[2])
        ts_f = ts_f[:N, :T, :P]
        ts_d = ts_d[:N, :T, :P]
        chance = 100.0 / N

        # FCs (raw, no ridge)
        FC_f = np.array([compute_fc_raw(ts_f[i]) for i in range(N)])
        FC_d = np.array([compute_fc_raw(ts_d[i]) for i in range(N)])

        # Direction 1: gallery=filt, query=dc
        D12 = build_D_alphaZ(FC_f, FC_d)
        acc12 = id_rate_from_D(D12) * 100.0

        # Direction 2: gallery=dc, query=filt
        D21 = build_D_alphaZ(FC_d, FC_f)
        acc21 = id_rate_from_D(D21) * 100.0

        mean_acc = 0.5 * (acc12 + acc21)

        print(f"\n[{label}] N={N}, T={T}, P={P}, chance≈{chance:.2f}%")
        print(f"  Alpha-Z  →  filt←dc: {acc12:.2f}% | dc←filt: {acc21:.2f}% | mean: {mean_acc:.2f}%")

        rows.append(dict(
            cohort=label, N=N, T=T, P=P,
            alphaZ_filt_as_gallery=acc12,
            alphaZ_dc_as_gallery=acc21,
            alphaZ_mean=mean_acc
        ))

    # Macro-average of the per-cohort means (unweighted)
    means = [r["alphaZ_mean"] for r in rows]
    macro = float(np.mean(means)) if means else float("nan")
    print(f"\nMacro-average Alpha-Z (mean of cohort means): {macro:.2f}%")

if __name__ == "__main__":
    main()

# %%
import numpy as np, scipy.io as sio

MAT_F = r"D:\Research AU\Alzheimer FCNs\Alzheimer FCNs\ts_adni_filt.mat"
MAT_D = r"D:\Research AU\Alzheimer FCNs\Alzheimer FCNs\ts_adni_dc.mat"
keys = [
    ("ts_adni_filt_controls1","ts_adni_dc_controls1"),
    ("ts_adni_filt_EMCI1",    "ts_adni_dc_EMCI1"),
    ("ts_adni_filt_LMCI1",    "ts_adni_dc_LMCI1"),
    ("ts_adni_filt_AD1",      "ts_adni_dc_AD1"),
]

def shape(path, key):
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    A = np.asarray(m[key])
    if A.ndim==3 and A.shape[1] < 30:  # (N,P,T) -> (N,T,P)
        A = np.transpose(A, (0,2,1))
    return A.shape  # (N,T,P)

for kf,kd in keys:
    print(kf, "->", shape(MAT_F, kf))
    print(kd, "->", shape(MAT_D, kd))

# %%
