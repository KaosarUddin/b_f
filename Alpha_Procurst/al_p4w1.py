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
        file_path = os.path.join(base_path, subject_id, f'{subject_id}_tfMRI_WM_{scan_type}_400')
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
    return (id_rate_1 + id_rate_2) / 2 


base_path='/mmfs1/home/mzu0014/connectomes_400/'
lr_paths = generate_file_paths(base_path, 'LR')
rl_paths = generate_file_paths(base_path, 'RL')

connectivity_matrices_lr = [load_connectivity_matrix(path) for path in lr_paths]
connectivity_matrices_rl = [load_connectivity_matrix(path) for path in rl_paths]

def compute_id_rate_for_alphas( alpha_values):
    # Dictionary to hold ID rates for each tau
    id_rates_for_alphas = {}

    for alpha in alpha_values:
        bw_distance_matrix = BW_distance_matrix(connectivity_matrices_lr,connectivity_matrices_rl, alpha=alpha)  
        average_id_rate = compute_id_rate_single_matrix(bw_distance_matrix)

        # Store the results in the dictionary
        id_rates_for_alphas[alpha] = average_id_rate
        
    return id_rates_for_alphas
# Assuming connectivity_matrices_lr and connectivity_matrices_rl are already loaded
alpha_values = [.001,.01,.1,.2,.3,.5] # Different regularization strengths to test

# Compute the ID rates for different values of tau
id_rates = compute_id_rate_for_alphas(alpha_values)

# Output the results to a file
output_file = '/mmfs1/home/mzu0014/project1/id_rates_alpha_pro1(400)_wm_tao.txt'
with open(output_file, 'w') as f:
    for alpha, id_rate in id_rates.items():
        f.write(f"Alpha: {alpha}, ID Rate: {id_rate}\n")
