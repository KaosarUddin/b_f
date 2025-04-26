import numpy as np
import os
from scipy.linalg import sqrtm

def generate_file_paths(base_path, task, scan_type, num_subjects=428):
    file_paths = []
    subject_ids = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    subject_ids.sort()
    subject_ids = subject_ids[:num_subjects]
    for subject_id in subject_ids:
        file_name = f'{subject_id}_{prefix(task)}_{task}_{scan_type}_100'
        file_path = os.path.join(base_path, subject_id, file_name)
        file_paths.append(file_path)
    return file_paths

def prefix(task):
    return 'rfMRI' if task == 'REST1' else 'tfMRI'

def load_connectivity_matrix(file_path):
    try:
        return np.loadtxt(file_path, delimiter=' ')
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def make_spd(matrix):
    symmetric_matrix = (matrix + matrix.T) / 2
    regularized_matrix = symmetric_matrix + 1e-6 * np.eye(matrix.shape[0])  # Small regularization
    return regularized_matrix

def compute_BW_distance(X, Y):
    term = sqrtm(np.dot(sqrtm(X), np.dot(Y, sqrtm(X))))
    distance = np.trace(X) + np.trace(Y) - 2 * np.trace(term)
    return np.real(np.sqrt(distance))

def compute_distance_matrix(connectivity_matrices_1, connectivity_matrices_2):
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

def process_task(base_path, task, scan_type):
    task_paths = generate_file_paths(base_path, task, scan_type)
    connectivity_matrices = [load_connectivity_matrix(path) for path in task_paths if load_connectivity_matrix(path) is not None]
    
    if len(connectivity_matrices) == 0:
        print(f"Error: No valid matrices found for task {task} {scan_type}")
        return None

    bw_distance_matrix = compute_distance_matrix(connectivity_matrices, connectivity_matrices)
    average_id_rate = compute_id_rate_single_matrix(bw_distance_matrix)
    
    return average_id_rate

base_path = '/mmfs1/home/mzu0014/connectomes_100/'
tasks = ['REST1', 'EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM']
scan_types = ['LR', 'RL']

id_rates = {}
for task in tasks:
    for scan_type in scan_types:
        id_rate = process_task(base_path, task, scan_type)
        if id_rate is not None:
            id_rates[f"{task}_{scan_type}"] = id_rate

output_file = '/mmfs1/home/mzu0014/project1/id_rates_bw_at(100)_tasks.txt'
with open(output_file, 'w') as f:
    for task_scan_type, id_rate in id_rates.items():
        f.write(f"Task_ScanType: {task_scan_type}, ID Rate: {id_rate}\n")
