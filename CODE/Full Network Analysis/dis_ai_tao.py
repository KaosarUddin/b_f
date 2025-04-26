import numpy as np
import os
from scipy.linalg import logm,norm,sqrtm
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

def make_spd(matrix, tau=1e-6):
    symmetric_matrix = (matrix + matrix.T) / 2
    regularized_matrix = symmetric_matrix + tau * np.eye(matrix.shape[0])
    return regularized_matrix

def compute_geodesic_distance(A, B):
    C = np.dot(np.linalg.inv(sqrtm(A)), B)
    C = np.dot(C, np.linalg.inv(sqrtm(A)))
    logC = logm(C)
    distance = norm(logC, 'fro')
    return distance

def distance_matrix(connectivity_matrices_1, connectivity_matrices_2, tau=1e-6):
    num_subjects = len(connectivity_matrices_1)
    distance_matrix = np.zeros((num_subjects, num_subjects))
    for i, matrix_1 in enumerate(connectivity_matrices_1):
        matrix_1 = make_spd(matrix_1, tau=tau)
        for j, matrix_2 in enumerate(connectivity_matrices_2):
            matrix_2 = make_spd(matrix_2, tau=tau)
            distance_matrix[i, j] = compute_geodesic_distance(matrix_1, matrix_2)
    return distance_matrix

def compute_id_rate(distance_matrix):
    correct_identifications = sum(np.argmin(distance_matrix[i, :]) == i for i in range(distance_matrix.shape[0]))
    return correct_identifications / distance_matrix.shape[0]

base_path = '/mmfs1/home/mzu0014/connectomes_100/'
lr_paths = generate_file_paths(base_path, 'LR')
rl_paths = generate_file_paths(base_path, 'RL')

connectivity_matrices_lr = [load_connectivity_matrix(path) for path in lr_paths]
connectivity_matrices_rl = [load_connectivity_matrix(path) for path in rl_paths]

# Function to compute the ID rate considering the top N matches
def compute_id_rate_for_top_n(distance_matrix, top_n=1):
    num_subjects = distance_matrix.shape[0]
    successful_identifications = 0
    for i in range(num_subjects):
        closest_indices = np.argsort(distance_matrix[i, :])[:top_n]
        if i in closest_indices:
            successful_identifications += 1
    return successful_identifications / num_subjects

# Assuming connectivity_matrices_lr and connectivity_matrices_rl are already loaded
tau_values = [0, .01, .1]  # Different regularization strengths to test

# Dictionary to hold ID rates for different tau values
id_rate_results = {}

for tau in tau_values:
    # Compute distance matrices for regularized matrices
    distance_matrix_1 = distance_matrix(connectivity_matrices_lr, connectivity_matrices_rl, tau=tau)
    distance_matrix_2 = distance_matrix(connectivity_matrices_rl, connectivity_matrices_lr, tau=tau)

    # Calculate ID rates for different ranks
    ranks = [1, 3, 5]
    id_rates_1 = [compute_id_rate_for_top_n(distance_matrix_1, top_n=rank) for rank in ranks]
    id_rates_2 = [compute_id_rate_for_top_n(distance_matrix_2, top_n=rank) for rank in ranks]
    final_id_rates = [(r1 + r2) / 2 for r1, r2 in zip(id_rates_1, id_rates_2)]

    # Store the results in the dictionary
    id_rate_results[tau] = final_id_rates

    # Print ID rates for different ranks
    for rank, id_rate in zip(ranks, final_id_rates):
        print(f"Final ID Rate for tau={tau}, rank_{rank}: {id_rate}")

# Write the results to a file on the server
output_file_path = '/mmfs1/home/mzu0014/project1/id_rate_results_dis_ai_tao.txt'
with open(output_file_path, 'w') as file:
    for tau, rates in id_rate_results.items():
        for rank, id_rate in zip(ranks, rates):
            file.write(f"Tau: {tau}, Rank: {rank}, ID Rate: {id_rate}\n")

# Plotting
plt.figure(figsize=(8, 6))
for tau, rates in id_rate_results.items():
    plt.plot(ranks, rates, label=f'ID Rate Tau={tau}', marker='o', linestyle='--')
plt.xlabel('Rank')
plt.ylabel('Identification Rate')
plt.title('Identification Rates for Different Ranks and Tau Values')
plt.xticks(ranks)
plt.legend()
plt.grid(True)
plot_file_path = '/mmfs1/home/mzu0014/project1/id_rates_ai_tao_values.png'
plt.savefig(plot_file_path)
plt.close()