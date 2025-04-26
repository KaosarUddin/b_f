import numpy as np
import os
from scipy.linalg import logm,norm,sqrtm
#import matplotlib.pyplot as plt

def load_connectivity_matrix(file_path):
    try:
        return np.loadtxt(file_path, delimiter=' ')
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def generate_file_paths(base_path, scan_type, num_subjects=15):
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

# AI distance
def compute_geodesic_distance(A, B):
    C = np.dot(np.linalg.inv(sqrtm(A)), B)
    C = np.dot(C, np.linalg.inv(sqrtm(A)))
    logC = logm(C)
    distance = norm(logC, 'fro')
    return distance

# Log _eucildean distance
def compute_log_euclidean_distance(X, Y):
    log_X = logm(X)
    log_Y = logm(Y)
    distance = np.linalg.norm(log_X - log_Y, 'fro')
    return distance

# Pearson distance
def compute_pearson_distance(X, Y):
    # Flatten the matrices to vectors
    X_vec = X.flatten()
    Y_vec = Y.flatten()
    
    # Compute Pearson correlation coefficient
    r = np.corrcoef(X_vec, Y_vec)[0, 1]
    
    # Pearson distance is defined as 1 - correlation coefficient
    distance = 1 - r
    return distance

# Euclidean disatance
def compute_euclidean_distance(X, Y):
    # Flatten the matrices to vectors
    X_vec = X.flatten()
    Y_vec = Y.flatten()
    
    # Compute Euclidean distance
    distance = np.linalg.norm(X_vec - Y_vec)
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

base_path = 'D:/Research AU/Python/connectomes_100/'
#base_path = 'connectomes_100'
lr_paths = generate_file_paths(base_path, 'LR')
rl_paths = generate_file_paths(base_path, 'RL')

connectivity_matrices_lr = [load_connectivity_matrix(path) for path in lr_paths]
connectivity_matrices_rl = [load_connectivity_matrix(path) for path in rl_paths]

tau = 0.0  # Adjusted regularization parameter

# Regularize matrices and compute distances
distance_matrix_1 = distance_matrix(connectivity_matrices_lr, connectivity_matrices_rl, tau=tau)
id_rate_1 = compute_id_rate(distance_matrix_1)

distance_matrix_2 = distance_matrix(connectivity_matrices_rl, connectivity_matrices_lr, tau=tau)
id_rate_2 = compute_id_rate(distance_matrix_2)

final_id_rate = (id_rate_1 + id_rate_2) / 2
print(f"ID Rate 1: {id_rate_1}")
print(f"ID Rate 2: {id_rate_2}")
print(f"Final ID Rate: {final_id_rate}")


# Anaother way to estimate the id rate
def compute_id_rate_single_matrix(distance_matrix):
    # Using rows for id_rate_1
    id_rate_1 = sum(np.argmin(distance_matrix[i, :]) == i for i in range(distance_matrix.shape[0])) / distance_matrix.shape[0]
    # Using columns for id_rate_2
    id_rate_2 = sum(np.argmin(distance_matrix[:, j]) == j for j in range(distance_matrix.shape[1])) / distance_matrix.shape[1]
    return (id_rate_1 + id_rate_2) / 2  # Return the average of the two rates

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
tau_values = [0.01] # Different regularization strengths to test
# Compute the ID rates for different values of tau
id_rates = compute_id_rate_for_taus(tau_values)
print(f"Tau: {tau}, ID Rate: {id_rates}")