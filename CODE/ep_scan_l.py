import numpy as np
import os
from scipy.linalg import fractional_matrix_power, eigh
from scipy.linalg import sqrtm, logm, norm
import random
from spd_metrics_id.distance import alpha_z_bw, pearson_distance, euclidean_distance


def load_connectivity_matrix(file_path):
    try:
        X = np.loadtxt(file_path, delimiter=' ')
        if X.ndim == 1:
            X = X[:, None]
        X = np.asarray(X, dtype=np.float64)

        # If not square, treat as time series [T x P] -> build P x P correlation
        if X.shape[0] != X.shape[1]:
            X = X - np.nanmean(X, axis=0, keepdims=True)
            C = np.corrcoef(X, rowvar=False)
            C = np.nan_to_num(C, nan=0.0)
            C = 0.5 * (C + C.T)  # symmetrize
            # light eigenvalue floor so fractional powers are defined
            w, V = eigh(C)
            w = np.maximum(w, 1e-10)
            C = (V * w) @ V.T
            return C
        else:
            # already a square matrix
            return X
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def generate_file_paths(base_path, scan_type, num_subjects=100):
    """
    Looks for files saved by your truncation step:
      <base>/<subject>/<subject>_LR_2min.txt
      <base>/<subject>/<subject>_RL_2min.txt
    """
    file_paths = []
    subject_ids = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    subject_ids.sort()
    subject_ids = subject_ids[:num_subjects]
    for subject_id in subject_ids:
        p = os.path.join(base_path, subject_id, f'{subject_id}_{scan_type}_13.2min.txt')
        # fallback if files were saved directly under base_path
        if not os.path.isfile(p):
            alt = os.path.join(base_path, f'{subject_id}_{scan_type}_13.2min.txt')
            p = alt
        file_paths.append(p)
    return file_paths


def compute_alpha_z_BW_distance(A, B, alpha, z):
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
    return np.real(divergence)


def compute_geodesic_distance(A, B):
    C = np.dot(np.linalg.inv(sqrtm(A)), B)
    C = np.dot(C, np.linalg.inv(sqrtm(A)))
    logC = logm(C)
    distance = norm(logC, 'fro')
    return distance


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


def compute_euclidean_distance(X, Y):
    # Flatten the matrices to vectors
    X_vec = X.flatten()
    Y_vec = Y.flatten()

    # Compute Euclidean distance
    distance = np.linalg.norm(X_vec - Y_vec)
    return distance


def distance_matrix(connectivity_matrices_1, connectivity_matrices_2):
    num_subjects = len(connectivity_matrices_1)
    D = np.zeros((num_subjects, num_subjects))
    for i, A in enumerate(connectivity_matrices_1):
        if A is None:
            D[i, :] = np.inf
            continue
        for j, B in enumerate(connectivity_matrices_2):
            if B is None:
                D[i, j] = np.inf
                continue
            D[i, j] = compute_pearson_distance(A, B)
    return D


def compute_id_rate(D):
    correct_identifications = sum(np.argmin(D[i, :]) == i for i in range(D.shape[0]))
    return correct_identifications / D.shape[0]


# ----------------------------
# MAIN
# ----------------------------
base_path = r'D:/Research AU/truncated_timeseries_sl_400/13.2min'

lr_paths = generate_file_paths(base_path, 'LR')
rl_paths = generate_file_paths(base_path, 'RL')

connectivity_matrices_lr = [load_connectivity_matrix(p) for p in lr_paths]
connectivity_matrices_rl = [load_connectivity_matrix(p) for p in rl_paths]

distance_matrix_1 = distance_matrix(connectivity_matrices_lr, connectivity_matrices_rl)
id_rate_1 = compute_id_rate(distance_matrix_1)

distance_matrix_2 = distance_matrix(connectivity_matrices_rl, connectivity_matrices_lr)
id_rate_2 = compute_id_rate(distance_matrix_2)

current_id_rate = 0.5 * (id_rate_1 + id_rate_2)

print(id_rate_1)
print(id_rate_2)
print(current_id_rate)