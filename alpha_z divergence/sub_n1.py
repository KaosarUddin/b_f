import os
import numpy as np
from scipy.io import loadmat
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
    subject_ids.sort()
    subject_ids = subject_ids[:num_subjects]
    for subject_id in subject_ids:
        file_path = os.path.join(base_path, subject_id, f'{subject_id}_rfMRI_REST1_{scan_type}_100')
        file_paths.append(file_path)
    return file_paths

def compute_alpha_z_BW_distance(A, B, alpha, z):
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

def divide_regions(data, rois):
    """
    Extracts a specific region's submatrix from the connectivity matrix.
    """
    unique_regions = np.unique(rois)
    region_indices = np.where(rois == unique_regions[0])[0]  # Example for one region
    return data[region_indices][:, region_indices]

# Load the ROI assignments
yeo_rois_path = 'C:/Users/ksrru/Documents/updated_LUT/updated_LUT/YeoROIs_N100.mat' # Replace with actual path
yeo_data = loadmat(yeo_rois_path)
yeo_rois = yeo_data['YeoROIs'].flatten()

# Base path for connectivity matrices
base_path = 'D:/Research AU/Python/connectomes_100/'
lr_paths = generate_file_paths(base_path, 'LR')
rl_paths = generate_file_paths(base_path, 'RL')

# Load connectivity matrices
connectivity_matrices_lr = [load_connectivity_matrix(path) for path in lr_paths]
connectivity_matrices_rl = [load_connectivity_matrix(path) for path in rl_paths]

# Compute ID rate for a specific region
region_id = 3  # Replace with desired region ID
region_lr = [divide_regions(matrix, yeo_rois) for matrix in connectivity_matrices_lr if matrix is not None]
region_rl = [divide_regions(matrix, yeo_rois) for matrix in connectivity_matrices_rl if matrix is not None]

# Remove None entries
region_lr = [mat for mat in region_lr if mat is not None]
region_rl = [mat for mat in region_rl if mat is not None]

# Check if there is sufficient data
if region_lr and region_rl:
    alpha = 0.99
    z = 1
    bw_distance_matrix_1 = BW_distance_matrix(region_lr, region_rl, alpha, z)
    id_rate_1 = compute_id_rate(bw_distance_matrix_1)
    
    bw_distance_matrix_2 = BW_distance_matrix(region_rl, region_lr, alpha, z)
    id_rate_2 = compute_id_rate(bw_distance_matrix_2)
    
    id_rate = (id_rate_1 + id_rate_2) / 2
    print(f"Region {region_id}: ID Rate = {id_rate}")
else:
    print(f"Region {region_id} has insufficient data.")
