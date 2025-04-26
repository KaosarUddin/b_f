import numpy as np
import os
from scipy.linalg import sqrtm

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
        file_path = os.path.join(base_path, subject_id, f'{subject_id}_rfMRI_REST1_{scan_type}_100')
        file_paths.append(file_path)
    return file_paths

def make_spd(matrix):
    symmetric_matrix = (matrix + matrix.T) / 2
    regularized_matrix = symmetric_matrix + 1e-6 * np.eye(matrix.shape[0])  # Small regularization
    return regularized_matrix

def compute_BW_distance(X, Y):
    term = sqrtm(np.dot(sqrtm(X), np.dot(Y, sqrtm(X))))
    distance = np.trace(X) + np.trace(Y) - 2 * np.trace(term)
    return np.real(np.sqrt(distance))

def distance_matrix(connectivity_matrices_1, connectivity_matrices_2):
    num_subjects = len(connectivity_matrices_1)
    distance_matrix = np.zeros((num_subjects, num_subjects))
    for i, matrix_1 in enumerate(connectivity_matrices_1):
        matrix_1 = make_spd(matrix_1)
        for j, matrix_2 in enumerate(connectivity_matrices_2):
            matrix_2 = make_spd(matrix_2)
            distance_matrix[i, j] = compute_BW_distance(matrix_1, matrix_2)
    return distance_matrix

def compute_id_rate_single_matrix(distance_matrix):
    id_rate_1 = sum(np.argmin(distance_matrix[i, :]) == i for i in range(distance_matrix.shape[0])) / distance_matrix.shape[0]
    id_rate_2 = sum(np.argmin(distance_matrix[:, j]) == j for j in range(distance_matrix.shape[1])) / distance_matrix.shape[1]
    return (id_rate_1 + id_rate_2) / 2

base_path = '/mmfs1/home/mzu0014/connectomes_100/'
lr_paths = generate_file_paths(base_path, 'LR')
rl_paths = generate_file_paths(base_path, 'RL')

connectivity_matrices_lr = [load_connectivity_matrix(path) for path in lr_paths]
connectivity_matrices_rl = [load_connectivity_matrix(path) for path in rl_paths]

bw_distance_matrix = distance_matrix(connectivity_matrices_lr, connectivity_matrices_rl)
id_rate = compute_id_rate_single_matrix(bw_distance_matrix)

output_file = '/mmfs1/home/mzu0014/project1/id_rates_bw_100_rest.txt'
with open(output_file, 'w') as f:
    f.write(f"ID Rate: {id_rate}\n")
