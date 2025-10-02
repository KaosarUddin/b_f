import numpy as np
import os
from scipy.linalg import fractional_matrix_power, eigh
import random

# ----------------------------
# UNCHANGED HELPERS
# ----------------------------
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

def generate_file_paths(base_path, scan_type, num_subjects=10, duration_tag="1.2min"):
    """
    Looks for files saved by your truncation step:
      <base>/<subject>/<subject>_LR_<duration>.txt
      <base>/<subject>/<subject>_RL_<duration>.txt
    """
    file_paths = []
    subject_ids = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    subject_ids.sort()
    subject_ids = subject_ids[:num_subjects]
    for subject_id in subject_ids:
        p = os.path.join(base_path, subject_id, f'{subject_id}_{scan_type}_{duration_tag}.txt')
        # fallback if files were saved directly under base_path
        if not os.path.isfile(p):
            alt = os.path.join(base_path, f'{subject_id}_{scan_type}_{duration_tag}.txt')
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

if __name__ == "__main__":
    # Root that contains the <m>min subfolders created by your truncation step
    ROOT_BASE = '/mmfs1/home/mzu0014/truncated_timeseries_sl_100'
    RESULTS_BASE = r"/mmfs1/home/mzu0014/project1"   # where results will be saved

    # Your durations in minutes
    DURATIONS_MIN = [1.2, 2.4, 3.6, 4.8, 6, 7.2, 8.40, 9.60, 10.8, 12, 13.2, 14.28]

    alpha = 0.99
    z = 1.0
    num_subjects = 30  # same as before

    for m in DURATIONS_MIN:
        duration_tag = f"{m:g}min"   # matches how files/dirs were named in truncation code
        base_path = os.path.join(ROOT_BASE, duration_tag)

        lr_paths = generate_file_paths(base_path, 'LR', num_subjects=num_subjects, duration_tag=duration_tag)
        rl_paths = generate_file_paths(base_path, 'RL', num_subjects=num_subjects, duration_tag=duration_tag)

        connectivity_matrices_lr = [load_connectivity_matrix(p) for p in lr_paths]
        connectivity_matrices_rl = [load_connectivity_matrix(p) for p in rl_paths]

        distance_matrix_1 = distance_matrix(connectivity_matrices_lr, connectivity_matrices_rl, alpha, z)
        id_rate_1 = compute_id_rate(distance_matrix_1)

        distance_matrix_2 = distance_matrix(connectivity_matrices_rl, connectivity_matrices_lr, alpha, z)
        id_rate_2 = compute_id_rate(distance_matrix_2)

        current_id_rate = 0.5 * (id_rate_1 + id_rate_2)

        print(f"{duration_tag}: id_rate_LR→RL={id_rate_1:.3f}, RL→LR={id_rate_2:.3f}, mean={current_id_rate:.3f}")

        # ----------------------------
        # SAVE RESULTS FOR THIS DURATION
        # ----------------------------
        results_path = os.path.join(
            RESULTS_BASE, f"identification_rates_az(100)_{duration_tag}.txt"
        )
        with open(results_path, 'w') as f:
            f.write(f"ID Rate 1: {id_rate_1}\n")
            f.write(f"ID Rate 2: {id_rate_2}\n")
            f.write(f"Average ID Rate: {current_id_rate}\n")

    print("\n[INFO] Finished all durations. Results saved in", RESULTS_BASE)