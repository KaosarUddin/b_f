import numpy as np
import os
from scipy.linalg import fractional_matrix_power
import random
import matplotlib.pyplot as plt

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

def BW_distance_matrix(connectivity_matrices_1, connectivity_matrices_2, alpha, z):
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

def remove_node(connectivity_matrices, node_index):
    """
    Remove a specific node (ROI) from the connectivity matrices.
    """
    residual_matrices = [
        np.delete(np.delete(mat, node_index, axis=0), node_index, axis=1)
        for mat in connectivity_matrices
    ]
    return residual_matrices

# Paths to data
base_path = 'D:/Research AU/Python/connectomes_100/'
lr_paths = generate_file_paths(base_path, 'LR')
rl_paths = generate_file_paths(base_path, 'RL')

# Load connectivity matrices
connectivity_matrices_lr = [load_connectivity_matrix(path) for path in lr_paths]
connectivity_matrices_rl = [load_connectivity_matrix(path) for path in rl_paths]

# Parameters
alpha = 0.99
z = 1

# Original ID rate
bw_distance_matrix_1 = BW_distance_matrix(connectivity_matrices_lr, connectivity_matrices_rl, alpha, z)
id_rate_1 = compute_id_rate(bw_distance_matrix_1)

bw_distance_matrix_2 = BW_distance_matrix(connectivity_matrices_rl, connectivity_matrices_lr, alpha, z)
id_rate_2 = compute_id_rate(bw_distance_matrix_2)

current_id_rate = (id_rate_1 + id_rate_2) / 2
print(f"Original ID Rate: {current_id_rate}")

# Node-level analysis
num_nodes = connectivity_matrices_lr[0].shape[0]  # Assuming all matrices have the same shape
node_id_deltas = []

for node in range(num_nodes):
    # Remove the node from both LR and RL matrices
    residual_lr = remove_node(connectivity_matrices_lr, node)
    residual_rl = remove_node(connectivity_matrices_rl, node)

    # Compute ID rate for the residual network
    bw_distance_matrix_residual = BW_distance_matrix(residual_lr, residual_rl, alpha, z)
    id_rate_residual_1 = compute_id_rate(bw_distance_matrix_residual)

    bw_distance_matrix_residual_reverse = BW_distance_matrix(residual_rl, residual_lr, alpha, z)
    id_rate_residual_2 = compute_id_rate(bw_distance_matrix_residual_reverse)

    residual_id_rate = (id_rate_residual_1 + id_rate_residual_2) / 2
    delta_id = current_id_rate - residual_id_rate
    node_id_deltas.append(delta_id)

# Visualize node-level contributions
node_coordinates = np.random.rand(num_nodes, 3)  # Replace with actual node coordinates if available
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    node_coordinates[:, 0], node_coordinates[:, 1], c=node_id_deltas, cmap='viridis', s=100
)
plt.colorbar(scatter, label="Delta ID")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Node-Level Contribution to ID Rate")
plt.show()
