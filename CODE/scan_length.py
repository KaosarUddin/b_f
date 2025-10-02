import numpy as np
import os
from scipy.linalg import fractional_matrix_power, eigh
import random

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
        p = os.path.join(base_path, subject_id, f'{subject_id}_{scan_type}_2.4min.txt')
        # fallback if files were saved directly under base_path
        if not os.path.isfile(p):
            alt = os.path.join(base_path, f'{subject_id}_{scan_type}_2.4min.txt')
            p = alt
        file_paths.append(p)
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
    D = np.zeros((num_subjects, num_subjects))
    for i, A in enumerate(connectivity_matrices_1):
        if A is None:
            D[i, :] = np.inf
            continue
        for j, B in enumerate(connectivity_matrices_2):
            if B is None:
                D[i, j] = np.inf
                continue
            D[i, j] = compute_alpha_z_BW_distance(A, B, alpha, z)
    return D

def compute_id_rate(D):
    correct_identifications = sum(np.argmin(D[i, :]) == i for i in range(D.shape[0]))
    return correct_identifications / D.shape[0]

# ----------------------------
# MAIN
# ----------------------------
#base_path = r'D:/Research AU/truncated_timeseries_400/2min'
base_path='/mmfs1/home/mzu0014/truncated_timeseries_sl_400/2.4min'
lr_paths = generate_file_paths(base_path, 'LR')
rl_paths = generate_file_paths(base_path, 'RL')

connectivity_matrices_lr = [load_connectivity_matrix(p) for p in lr_paths]
connectivity_matrices_rl = [load_connectivity_matrix(p) for p in rl_paths]

alpha = 0.99
z = 1.0

distance_matrix_1 = distance_matrix(connectivity_matrices_lr, connectivity_matrices_rl, alpha, z)
id_rate_1 = compute_id_rate(distance_matrix_1)

distance_matrix_2 = distance_matrix(connectivity_matrices_rl, connectivity_matrices_lr, alpha, z)
id_rate_2 = compute_id_rate(distance_matrix_2)

current_id_rate = 0.5 * (id_rate_1 + id_rate_2)

results_path = "/mmfs1/home/mzu0014/project1/identification_rates_az(400)_2.4min.txt"
with open(results_path, 'w') as f:
    f.write(f"ID Rate 1: {id_rate_1}\n")
    f.write(f"ID Rate 2: {id_rate_2}\n")
    f.write(f"Average ID Rate: {current_id_rate}\n")

