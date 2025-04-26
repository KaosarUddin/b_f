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

def generate_file_paths(base_path, scan_type, num_subjects=428):
    file_paths = []
    subject_ids = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    subject_ids.sort()
    subject_ids = subject_ids[:num_subjects]
    for subject_id in subject_ids:
        file_path = os.path.join(base_path, subject_id, f'{subject_id}_tfMRI_EMOTION_{scan_type}_800')
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

def divide_regions(data, rois):
    """
    Divides data into regions based on ROI assignments.
    """
    regions = {}
    unique_regions = np.unique(rois)
    for region in unique_regions:
        region_indices = np.where(rois == region)[0]
        regions[region] = data[region_indices][:, region_indices]
    return regions

# Load the ROI assignments
yeo_rois_path = '/mmfs1/home/mzu0014/updated_LUT/updated_LUT/YeoROIs_N800.mat' # Replace with actual path
yeo_data = loadmat(yeo_rois_path)
yeo_rois = yeo_data['YeoROIs'].flatten()

# Base path for connectivity matrices
base_path = '/mmfs1/home/mzu0014/connectomes_800/'
lr_paths = generate_file_paths(base_path, 'LR')
rl_paths = generate_file_paths(base_path, 'RL')

# Load connectivity matrices
connectivity_matrices_lr = [load_connectivity_matrix(path) for path in lr_paths]
connectivity_matrices_rl = [load_connectivity_matrix(path) for path in rl_paths]

# Compute ID rates for each region
alpha = 0.99
z = 1
id_rates = {}

for region in range(1, 8):  # Regions 1 to 7
    print(f"Processing region {region}...")
    region_lr = [divide_regions(matrix, yeo_rois).get(region) for matrix in connectivity_matrices_lr if matrix is not None]
    region_rl = [divide_regions(matrix, yeo_rois).get(region) for matrix in connectivity_matrices_rl if matrix is not None]
    
    # Remove None entries
    region_lr = [mat for mat in region_lr if mat is not None]
    region_rl = [mat for mat in region_rl if mat is not None]
    
    # Compute distances and ID rate
    if region_lr and region_rl:
        bw_distance_matrix_1 = BW_distance_matrix(region_lr, region_rl, alpha, z)
        id_rate_1 = compute_id_rate(bw_distance_matrix_1)
        
        bw_distance_matrix_2 = BW_distance_matrix(region_rl, region_lr, alpha, z)
        id_rate_2 = compute_id_rate(bw_distance_matrix_2)
        
        id_rates[region] = (id_rate_1 + id_rate_2) / 2
    else:
        print(f"Region {region} has insufficient data.")
        id_rates[region] = None

# Save ID rates to a file
output_file = '/mmfs1/home/mzu0014/project1/id_rates_az(800)_emotion(7N).txt'  # Change this to your preferred path on the server
with open(output_file, 'w') as f:
    for region, id_rate in id_rates.items():
        if id_rate is not None:
            f.write(f"Region: {region}, ID Rate: {id_rate}\n")
        else:
            f.write(f"Region: {region}, Insufficient Data\n")

print(f"ID rates saved to {output_file}")

