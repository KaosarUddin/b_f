import numpy as np
import os
from scipy.linalg import fractional_matrix_power

def load_connectivity_matrix(file_path):
    try:
        return np.loadtxt(file_path, delimiter=' ')
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def generate_file_paths(base_path, scan_type, num_subjects=30):
    file_paths = []
    subject_ids = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
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
    divergence = np.trace((1-alpha) * A + alpha * B - Q_az)
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

def optimize_alpha_z(connectivity_matrices_lr, connectivity_matrices_rl, alpha_range, z_range, step=0.01):
    best_id_rate = 0
    best_alpha = None
    best_z = None
    for alpha in np.arange(alpha_range[0], alpha_range[1], step):
        for z in np.arange(max(alpha, z_range[0]), z_range[1], step):
            bw_distance_matrix_1 = BW_distance_matrix(connectivity_matrices_lr, connectivity_matrices_rl, alpha, z)
            id_rate_1 = compute_id_rate(bw_distance_matrix_1)
            bw_distance_matrix_2 = BW_distance_matrix(connectivity_matrices_rl, connectivity_matrices_lr, alpha, z)
            id_rate_2 = compute_id_rate(bw_distance_matrix_2)
            current_id_rate = (id_rate_1 + id_rate_2) / 2
            
            if current_id_rate > best_id_rate:
                best_id_rate = current_id_rate
                best_alpha = alpha
                best_z = z
                
    return best_alpha, best_z, best_id_rate

# Example usage
base_path = 'F:/Research AU/Python/connectomes_100/'
lr_paths = generate_file_paths(base_path, 'LR')
rl_paths = generate_file_paths(base_path, 'RL')

connectivity_matrices_lr = [load_connectivity_matrix(path) for path in lr_paths]
connectivity_matrices_rl = [load_connectivity_matrix(path) for path in rl_paths]

alpha_range = (0.8,.83 )
z_range = (0.8, 0.83)

best_alpha, best_z, best_id_rate = optimize_alpha_z(connectivity_matrices_lr, connectivity_matrices_rl, alpha_range, z_range, step=0.01)

print(f"Best Alpha: {best_alpha}, Best Z: {best_z}, Highest ID Rate: {best_id_rate}")

# Save the results to a file
output_path = '/mmfs1/home/mzu0014/project1/optimize_az.txt'  # Specify the path to your output file
with open(output_path, 'w') as file:
    file.write(f"Best Alpha: {best_alpha}, Best Z: {best_z}, Highest ID Rate: {best_id_rate}\n")

print(f"Results saved to {output_path}")
