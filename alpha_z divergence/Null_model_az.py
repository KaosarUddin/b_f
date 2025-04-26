import numpy as np
from scipy.io import loadmat
from scipy.linalg import fractional_matrix_power
import os

# Load connectivity matrices
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
        file_path = os.path.join(base_path, subject_id, f'{subject_id}_rfMRI_REST1_{scan_type}_400')
        file_paths.append(file_path)
    return file_paths

# Distance computation
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

# Randomize intra- and inter-network connections
def randomize_labels_yeo(matrix, yeo_labels):
    randomized_matrix = np.zeros_like(matrix)
    unique_labels = np.unique(yeo_labels)
    
    # Randomize intra-network connections
    for label in unique_labels:
        indices = np.where(yeo_labels == label)[0]
        np.random.shuffle(indices)
        randomized_matrix[np.ix_(indices, indices)] = matrix[np.ix_(indices, indices)]

    # Randomize inter-network connections
    all_indices = np.arange(len(yeo_labels))
    np.random.shuffle(all_indices)  # Shuffle all nodes globally
    randomized_matrix = randomized_matrix[all_indices][:, all_indices]

    return randomized_matrix

# Main Code
#base_path = '/mmfs1/home/mzu0014/connectomes_800/'
base_path = 'D:/Research AU/connectomes_400/'
lr_paths = generate_file_paths(base_path, 'LR')
rl_paths = generate_file_paths(base_path, 'RL')

connectivity_matrices_lr = [load_connectivity_matrix(path) for path in lr_paths]
connectivity_matrices_rl = [load_connectivity_matrix(path) for path in rl_paths]

# Load Yeo's labels
#yeo_rois_path = '/mmfs1/home/mzu0014/updated_LUT/updated_LUT/YeoROIs_N800.mat'
yeo_rois_path = 'C:/Users/ksrru/Documents/updated_LUT/updated_LUT/YeoROIs_N400.mat'
yeo_data = loadmat(yeo_rois_path)
yeo_labels = yeo_data['YeoROIs'].flatten()

alpha = 0.99
z = 1

# Compute BW distance and ID rate for original data
distance_matrix_1 = distance_matrix(connectivity_matrices_lr, connectivity_matrices_rl, alpha, z)
id_rate_1 = compute_id_rate(distance_matrix_1)

distance_matrix_2 = distance_matrix(connectivity_matrices_rl, connectivity_matrices_lr, alpha, z)
id_rate_2 = compute_id_rate(distance_matrix_2)

current_id_rate = (id_rate_1 + id_rate_2) / 2

# Null model analysis (Single Run)
randomized_yeo_labels = yeo_labels.copy()
np.random.shuffle(randomized_yeo_labels)

randomized_lr = [randomize_labels_yeo(mat, yeo_labels) for mat in connectivity_matrices_lr]
randomized_rl = [randomize_labels_yeo(mat, yeo_labels) for mat in connectivity_matrices_rl]

distance_matrix_null_lr_rl = distance_matrix(randomized_lr, randomized_rl, alpha, z)
distance_matrix_null_rl_lr = distance_matrix(randomized_rl, randomized_lr, alpha, z)

# Compute ID rates for both directions
id_rate_null_lr_rl = compute_id_rate(distance_matrix_null_lr_rl)
id_rate_null_rl_lr = compute_id_rate(distance_matrix_null_rl_lr)

# Average the ID rates
id_rate_null = (id_rate_null_lr_rl + id_rate_null_rl_lr) / 2
print(current_id_rate)
print(id_rate_null)

# Save results
results_path = "/mmfs1/home/mzu0014/null_model_rest(800)_results.txt"
with open(results_path, 'w') as f:
    f.write(f"Original ID Rate: {current_id_rate}\n")
    f.write(f"Null Model ID Rate: {id_rate_null}\n")
   