import numpy as np
import os
from scipy.linalg import logm,norm,sqrtm

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
        file_path = os.path.join(base_path, subject_id, f'{subject_id}_tfMRI_MOTOR_{scan_type}_100')
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

def compute_id_rate_single_matrix(distance_matrix):
    # Using rows for id_rate_1
    id_rate_1 = sum(np.argmin(distance_matrix[i, :]) == i for i in range(distance_matrix.shape[0])) / distance_matrix.shape[0]
    # Using columns for id_rate_2
    id_rate_2 = sum(np.argmin(distance_matrix[:, j]) == j for j in range(distance_matrix.shape[1])) / distance_matrix.shape[1]
    return (id_rate_1 + id_rate_2) / 2  # Return the average of the two rates

base_path = '/mmfs1/home/mzu0014/connectomes_100/'
lr_paths = generate_file_paths(base_path, 'LR')
rl_paths = generate_file_paths(base_path, 'RL')

connectivity_matrices_lr = [load_connectivity_matrix(path) for path in lr_paths]
connectivity_matrices_rl = [load_connectivity_matrix(path) for path in rl_paths]

# Function to compute the ID rate considering the top N matches
def compute_id_rate_for_taus( tau_values):
    # Dictionary to hold ID rates for each tau
    id_rates_for_taus = {}

    for tau in tau_values:
        bw_distance_matrix = distance_matrix(connectivity_matrices_lr,connectivity_matrices_rl, tau=tau)  # Assuming matrices are comparable to themselves
        average_id_rate = compute_id_rate_single_matrix(bw_distance_matrix)

        # Store the results in the dictionary
        id_rates_for_taus[tau] = average_id_rate
        
    return id_rates_for_taus
# Assuming connectivity_matrices_lr and connectivity_matrices_rl are already loaded
tau_values = [0,.001, .01, .1,.2,.3,.4,.5,.6,.7,.8,.9,1] # Different regularization strengths to test
# Compute the ID rates for different values of tau
id_rates = compute_id_rate_for_taus(tau_values)
#print(f"Tau: {tau}, ID Rate: {id_rates}")
output_file = '/mmfs1/home/mzu0014/project1/id_rates_ai_motor_tao.txt'  # Change this to your preferred path on the server
with open(output_file, 'w') as f:
    for tau, id_rate in id_rates.items():
        f.write(f"Tau: {tau}, ID Rate: {id_rate}\n")
