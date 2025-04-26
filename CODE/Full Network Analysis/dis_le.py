import numpy as np
import os
from scipy.linalg import sqrtm, logm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_connectivity_matrix(file_path):
    try:
        return np.loadtxt(file_path, delimiter=' ')
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def generate_file_paths(base_path, scan_type, num_subjects=428):
    file_paths = []
    subject_ids = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    subject_ids = subject_ids[:num_subjects]
    for subject_id in subject_ids:
        file_path = os.path.join(base_path, subject_id, f'{subject_id}_rfMRI_REST1_{scan_type}_100')
        file_paths.append(file_path)
    return file_paths
def compute_log_euclidean_distance(X, Y):
    log_X = logm(X)
    log_Y = logm(Y)
    distance = np.linalg.norm(log_X - log_Y, 'fro')
    return distance
def BW_distance_matrix(connectivity_matrices_1, connectivity_matrices_2):
    num_subjects = len(connectivity_matrices_1)
    distance_matrix = np.zeros((num_subjects, num_subjects))
    for i, matrix_1 in enumerate(connectivity_matrices_1):
        if matrix_1 is None:
            continue
        for j, matrix_2 in enumerate(connectivity_matrices_2):
            if matrix_2 is None:
                continue
            distance_matrix[i, j] = compute_log_euclidean_distance(matrix_1, matrix_2)
    return distance_matrix

def compute_id_rate(distance_matrix):
    correct_identifications = sum(np.argmin(distance_matrix[i, :]) == i for i in range(distance_matrix.shape[0]))
    return correct_identifications / distance_matrix.shape[0]

base_path='/mmfs1/home/mzu0014/connectomes_100/'
lr_paths = generate_file_paths(base_path, 'LR')
rl_paths = generate_file_paths(base_path, 'RL')

connectivity_matrices_lr = [load_connectivity_matrix(path) for path in lr_paths]
connectivity_matrices_rl = [load_connectivity_matrix(path) for path in rl_paths]

# Compute the BW distance matrix for LR as rows and RL as columns
bw_distance_matrix_1 = BW_distance_matrix(connectivity_matrices_lr, connectivity_matrices_rl)
id_rate_1 = compute_id_rate(bw_distance_matrix_1)

# Compute the BW distance matrix for RL as rows and LR as columns
bw_distance_matrix_2 = BW_distance_matrix(connectivity_matrices_rl, connectivity_matrices_lr)
id_rate_2 = compute_id_rate(bw_distance_matrix_2)

# Compute the final ID rate as the mean of the two ID rates
final_id_rate_1 = (id_rate_1 + id_rate_2) / 2




#ID rate for rank_3
num_subjects = bw_distance_matrix_1.shape[0]  # Assuming square matrix and same subjects in test and retest
successful_identifications = 0

for i in range(num_subjects):
    # Find the indices of the three minimum distances in the row
    three_closest_indices_1 = np.argsort(bw_distance_matrix_1[i, :])[:3]
    
    # Check if the actual retest index is among the three closest matches
    if i in three_closest_indices_1:
        successful_identifications += 1

# Calculate ID Rate considering the best three matches
id_rate_three_closest_1 = successful_identifications / num_subjects

num_subjects = bw_distance_matrix_2.shape[0]  # Assuming square matrix and same subjects in test and retest
successful_identifications = 0

for i in range(num_subjects):
    # Find the indices of the three minimum distances in the row
    three_closest_indices_2 = np.argsort(bw_distance_matrix_2[i, :])[:3]
    
    # Check if the actual retest index is among the three closest matches
    if i in three_closest_indices_2:
        successful_identifications += 1

# Calculate ID Rate considering the best three matches
id_rate_three_closest_2 = successful_identifications / num_subjects

# Compute the final ID rate as the mean of the two ID rates
final_id_rate_3 = (id_rate_three_closest_1 + id_rate_three_closest_2) / 2




# ID rate for rank_5
num_subjects = bw_distance_matrix_1.shape[0]  # Assuming square matrix and same subjects in test and retest
successful_identifications = 0

for i in range(num_subjects):
    # Find the indices of the three minimum distances in the row
    three_closest_indices_1 = np.argsort(bw_distance_matrix_1[i, :])[:5]
    
    # Check if the actual retest index is among the three closest matches
    if i in three_closest_indices_1:
        successful_identifications += 1

# Calculate ID Rate considering the best three matches
id_rate_five_closest_1 = successful_identifications / num_subjects

num_subjects = bw_distance_matrix_2.shape[0]  # Assuming square matrix and same subjects in test and retest
successful_identifications = 0

for i in range(num_subjects):
    # Find the indices of the three minimum distances in the row
    three_closest_indices_2 = np.argsort(bw_distance_matrix_2[i, :])[:5]
    
    # Check if the actual retest index is among the three closest matches
    if i in three_closest_indices_2:
        successful_identifications += 1

# Calculate ID Rate considering the best three matches
id_rate_five_closest_2 = successful_identifications / num_subjects

# Compute the final ID rate as the mean of the two ID rates
final_id_rate_5 = (id_rate_five_closest_1 + id_rate_five_closest_2) / 2





# After calculating final_id_rates for ranks 1, 3, and 5
output_file_path = "/mmfs1/home/mzu0014/project1/id_rate_le_results.txt"
with open(output_file_path, "a") as file:  # Using "a" to append to the file
    file.write(f"ID Rate 1: {id_rate_1}\n")
    file.write(f"ID Rate 1: {id_rate_2}\n")
    file.write(f"ID Rate for Rank 1: {final_id_rate_1}\n")
    file.write(f"ID Rate 1: {id_rate_three_closest_1}\n")
    file.write(f"ID Rate 1: {id_rate_three_closest_2}\n")
    file.write(f"ID Rate for Rank 3: {final_id_rate_3}\n")  # Assuming you calculate this
    file.write(f"ID Rate 1: {id_rate_five_closest_1}\n")
    file.write(f"ID Rate 1: {id_rate_five_closest_2}\n")
    file.write(f"ID Rate for Rank 5: {final_id_rate_5}\n")  # Assuming you calculate this



fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns of subplots

# Heatmap for the first distance matrix
heatmap1 = axes[0].imshow(bw_distance_matrix_1, cmap='hot', interpolation='nearest')
axes[0].set_title(" Distance Matrix of Retest Vs Test FC'sfor Identification Rate")
fig.colorbar(heatmap1, ax=axes[0])

# Heatmap for the second distance matrix
heatmap2 = axes[1].imshow(bw_distance_matrix_2, cmap='hot', interpolation='nearest')
axes[1].set_title("Distance Matrix of Retest Vs Test FC's for Identification Rate")
fig.colorbar(heatmap2, ax=axes[1])



plt.tight_layout()
plot_file_path = '/mmfs1/home/mzu0014/project1/ID_rate_le.png'
plt.savefig(plot_file_path)
plt.close()