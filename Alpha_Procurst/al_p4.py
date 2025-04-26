import numpy as np
import os
from scipy.linalg import fractional_matrix_power, sqrtm

def load_connectivity_matrix(file_path):
    try:
        return np.loadtxt(file_path, delimiter=' ')
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def generate_file_paths(base_path, scan_type, num_subjects=428):
    file_paths = []
    subject_ids = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    subject_ids.sort()
    subject_ids = subject_ids[:num_subjects]
    for subject_id in subject_ids:
        file_path = os.path.join(base_path, subject_id, f'{subject_id}_tfMRI_LANGUAGE_{scan_type}_100')
        file_paths.append(file_path)
    return file_paths

def compute_proE_distance(A, B, alpha):
    if alpha == 0:
        raise ValueError("Alpha cannot be zero.")
    if alpha < 0:
        raise ValueError("Alpha must be positive.")
    # Compute the matrix powers
    A_2alpha = fractional_matrix_power(A, 2 * alpha)
    B_2alpha = fractional_matrix_power(B, 2 * alpha)
    A_alpha = fractional_matrix_power(A, alpha)
    B_2alpha_A_alpha = np.dot(A_alpha, np.dot(B_2alpha, A_alpha))

    # Compute traces
    trace_A2alpha_B2alpha = np.trace(A_2alpha) + np.trace(B_2alpha)
    trace_sqrt = np.trace(sqrtm(B_2alpha_A_alpha))

    # Calculate the distance
    distance = np.sqrt(1 / alpha**2 * (trace_A2alpha_B2alpha - 2 * trace_sqrt))
    return distance
    
def BW_distance_matrix(connectivity_matrices_1, connectivity_matrices_2, alpha):
    num_subjects = len(connectivity_matrices_1)
    distance_matrix = np.zeros((num_subjects, num_subjects))
    for i, matrix_1 in enumerate(connectivity_matrices_1):
        if matrix_1 is None:
            continue
        for j, matrix_2 in enumerate(connectivity_matrices_2):
            if matrix_2 is None:
                continue
            distance_matrix[i, j] = compute_proE_distance(matrix_1, matrix_2, alpha)
    return distance_matrix

def compute_id_rate_single_matrix(distance_matrix):
    id_rate_1 = sum(np.argmin(distance_matrix[i, :]) == i for i in range(distance_matrix.shape[0])) / distance_matrix.shape[0]
    id_rate_2 = sum(np.argmin(distance_matrix[:, j]) == j for j in range(distance_matrix.shape[1])) / distance_matrix.shape[1]
    return id_rate_1, id_rate_2


base_path='/mmfs1/home/mzu0014/connectomes_100/'
lr_paths = generate_file_paths(base_path, 'LR')
rl_paths = generate_file_paths(base_path, 'RL')

connectivity_matrices_lr = [load_connectivity_matrix(path) for path in lr_paths]
connectivity_matrices_rl = [load_connectivity_matrix(path) for path in rl_paths]
alpha=.2

bw_distance_matrix_1 = BW_distance_matrix(connectivity_matrices_lr, connectivity_matrices_rl, alpha)
id_rate_1, id_rate_2 = compute_id_rate_single_matrix(bw_distance_matrix_1)
current_id_rate = (id_rate_1 + id_rate_2) / 2

#print(f"Id_rate_1: {id_rate_1}, Id_rate_2: {id_rate_2}, Current_Id_rate: {current_id_rate}")

results_path = "/mmfs1/home/mzu0014/project1/identification_rates_alpha_pro(100)_language(.2).txt"
with open(results_path, 'w') as f:
    f.write(f"ID Rate 1: {id_rate_1}\n")
    f.write(f"ID Rate 2: {id_rate_2}\n")
    f.write(f"Average ID Rate: {current_id_rate}\n")
