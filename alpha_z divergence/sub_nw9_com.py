import os
import numpy as np
from scipy.io import loadmat
from scipy.linalg import fractional_matrix_power
from joblib import Parallel, delayed

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
        file_path = os.path.join(base_path, subject_id, f'{subject_id}_tfMRI_WM_{scan_type}_900')
        file_paths.append(file_path)
    return file_paths

def compute_alpha_z_BW_distance(A, B, alpha, z):
    def Q_alpha_z(A, B, alpha, z):
        part1 = fractional_matrix_power(B, (1-alpha)/(2*z))
        part2 = fractional_matrix_power(A, alpha/z)
        part3 = fractional_matrix_power(B, (1-alpha)/(2*z))
        return fractional_matrix_power(part1 @ part2 @ part3, z)

    Q_az = Q_alpha_z(A, B, alpha, z)
    return np.real(np.trace((1-alpha) * A + alpha * B) - np.trace(Q_az))

def compute_distance_row(matrix_1, connectivity_matrices, alpha, z):
    """
    Computes one row of the distance matrix for a single subject.
    """
    return [
        compute_alpha_z_BW_distance(matrix_1, matrix_2, alpha, z) 
        if matrix_2 is not None else np.inf
        for matrix_2 in connectivity_matrices
    ]

def BW_distance_matrix(connectivity_matrices_1, connectivity_matrices_2, alpha, z, n_jobs=-1):
    """
    Compute the distance matrix in parallel.
    """
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(compute_distance_row)(matrix_1, connectivity_matrices_2, alpha, z)
        for matrix_1 in connectivity_matrices_1
    ))

def compute_id_rate(distance_matrix):
    """
    Compute the ID rate by checking how many subjects are correctly matched.
    """
    correct_identifications = sum(np.argmin(distance_matrix[i, :]) == i for i in range(distance_matrix.shape[0]))
    return correct_identifications / distance_matrix.shape[0]

def divide_combined_regions(data, combined_indices):
    """
    Extracts the submatrix for combined regions.
    """
    return data[combined_indices][:, combined_indices]

# Load the ROI assignments
yeo_rois_path = '/mmfs1/home/mzu0014/updated_LUT/updated_LUT/YeoROIs_N900.mat' # Replace with actual path
yeo_data = loadmat(yeo_rois_path)
yeo_rois = yeo_data['YeoROIs'].flatten()

# Base path for connectivity matrices
base_path = '/mmfs1/home/mzu0014/connectomes_900/'
lr_paths = generate_file_paths(base_path, 'LR')
rl_paths = generate_file_paths(base_path, 'RL')

# Load connectivity matrices
connectivity_matrices_lr = [load_connectivity_matrix(path) for path in lr_paths]
connectivity_matrices_rl = [load_connectivity_matrix(path) for path in rl_paths]

# Define combined regions to process
regions_to_combine = {
    "Regions 6+7": [6, 7],
    "Regions 3+6+7": [3, 6, 7]
}

alpha = 0.99
z = 1
results = {}

for label, combined_regions in regions_to_combine.items():
    print(f"Processing {label}...")
    combined_indices = np.where(np.isin(yeo_rois, combined_regions))[0]

    # Extract combined regions for LR and RL
    region_lr = [
        divide_combined_regions(matrix, combined_indices) 
        for matrix in connectivity_matrices_lr if matrix is not None
    ]
    region_rl = [
        divide_combined_regions(matrix, combined_indices) 
        for matrix in connectivity_matrices_rl if matrix is not None
    ]

    # Remove None entries
    region_lr = [mat for mat in region_lr if mat is not None]
    region_rl = [mat for mat in region_rl if mat is not None]

    # Check if there is sufficient data
    if region_lr and region_rl:
        bw_distance_matrix_1 = BW_distance_matrix(region_lr, region_rl, alpha, z, n_jobs=-1)
        id_rate_1 = compute_id_rate(bw_distance_matrix_1)

        bw_distance_matrix_2 = BW_distance_matrix(region_rl, region_lr, alpha, z, n_jobs=-1)
        id_rate_2 = compute_id_rate(bw_distance_matrix_2)

        average_id_rate = (id_rate_1 + id_rate_2) / 2
        results[label] = average_id_rate
        print(f"{label}: ID Rate = {average_id_rate}")
    else:
        print(f"{label} has insufficient data.")
        results[label] = None

# Save results to a file
output_file = '/mmfs1/home/mzu0014/project1/id_rates_az(900)_(7N_combined_region)_wm.txt'
with open(output_file, 'w') as f:
    for label, id_rate in results.items():
        if id_rate is not None:
            f.write(f"{label}: ID Rate = {id_rate}\n")
        else:
            f.write(f"{label}: Insufficient Data\n")

print(f"ID rates saved to {output_file}")


