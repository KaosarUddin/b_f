import numpy as np
import os
from scipy.linalg import logm, norm, sqrtm
import matplotlib
matplotlib.use('Agg')  # Ensure compatibility with non-GUI environments
import matplotlib.pyplot as plt

def generate_file_paths(base_path, task, scan_type, num_subjects=428):
    file_paths = []
    subject_ids = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    subject_ids = subject_ids[:num_subjects]
    for subject_id in subject_ids:
        file_name = f'{subject_id}_{prefix(task)}_{task}_{scan_type}_100'
        file_path = os.path.join(base_path, subject_id, file_name)
        if os.path.exists(file_path):
            file_paths.append(file_path)
        else:
            print(f"File does not exist: {file_path}")
    return file_paths

def prefix(task):
    return 'rfMRI' if task == 'REST1' else 'tfMRI'

def load_connectivity_matrix(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    try:
        return np.loadtxt(file_path, delimiter=' ')
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
    
def make_spd(matrix, tau=1e-6):
    symmetric_matrix = (matrix + matrix.T) / 2
    regularized_matrix = symmetric_matrix + tau * np.eye(matrix.shape[0])
    return regularized_matrix

def compute_log_euclidean_distance(X, Y):
    log_X = logm(X)
    log_Y = logm(Y)
    distance = np.linalg.norm(log_X - log_Y, 'fro')
    return distance

def distance_matrix(connectivity_matrices_1, connectivity_matrices_2, tau=1e-6):
    num_subjects = len(connectivity_matrices_1)
    dist_matrix = np.zeros((num_subjects, num_subjects))
    for i, matrix_1 in enumerate(connectivity_matrices_1):
        if matrix_1 is None: continue
        matrix_1 = make_spd(matrix_1, tau=tau)
        for j, matrix_2 in enumerate(connectivity_matrices_2):
            if matrix_2 is None: continue
            matrix_2 = make_spd(matrix_2, tau=tau)
            dist_matrix[i, j] = compute_log_euclidean_distance(matrix_1, matrix_2)
    return dist_matrix

def compute_id_rate_for_top_n(distance_matrix, top_n=1):
    num_subjects = len([m for m in distance_matrix if m is not None])
    successful_identifications = 0
    for i in range(num_subjects):
        closest_indices = np.argsort(distance_matrix[i, :])[:top_n]
        if i in closest_indices:
            successful_identifications += 1
    return successful_identifications / num_subjects

# Main analysis
tasks = ['REST1', 'EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM']
base_path = '/mmfs1/home/mzu0014/connectomes_100/'
tau_values = [0, .01, .1]
ranks = [1, 3, 5]

id_rate_results_per_task = {task: {} for task in tasks}

for task in tasks:
    
    lr_paths = generate_file_paths(base_path, task, 'LR')
    rl_paths = generate_file_paths(base_path, task, 'RL')

    connectivity_matrices_lr = [load_connectivity_matrix(path) for path in lr_paths]
    connectivity_matrices_rl = [load_connectivity_matrix(path) for path in rl_paths]

    for tau in tau_values:
        
        dist_matrix_1 = distance_matrix(connectivity_matrices_lr, connectivity_matrices_rl, tau=tau)
        dist_matrix_2 = distance_matrix(connectivity_matrices_rl, connectivity_matrices_lr, tau=tau)

        for rank in ranks:
            id_rate_1 = compute_id_rate_for_top_n(dist_matrix_1, top_n=rank)
            id_rate_2 = compute_id_rate_for_top_n(dist_matrix_2, top_n=rank)
            final_id_rate = (id_rate_1 + id_rate_2) / 2
            id_rate_results_per_task[task].setdefault(tau, []).append(final_id_rate)
            

# Plotting and saving results
plt.figure(figsize=(10, 8))

# Create a color map for different tau values
tau_colors = {0: 'red', .01: 'green', .1: 'blue'}

# Plot ID rates
for task in tasks:
    for rank in ranks:
        rates = [id_rate_results_per_task[task][tau][rank-1] for tau in tau_values]  # rank-1 for zero-based indexing
        plt.plot(tau_values, rates, marker='o', linestyle='-', label=f'{task} Rank {rank}')

plt.title('ID Rates Across Tasks, Tau Values, and Ranks')
plt.xlabel('Tau Value')
plt.ylabel('Identification Rate')
plt.legend(bbox_to_anchor=(.9, .8), loc='upper left')
plt.grid(True)

plt.tight_layout()
plot_file_path = '/mmfs1/home/mzu0014/project1/id_rates_le_dis_comprehensive.png'
plt.savefig(plot_file_path)
plt.close()

output_file_path = '/mmfs1/home/mzu0014/project1/id_rate_le_results_summary.txt'
with open(output_file_path, 'w') as outfile:
    for task, tau_data in id_rate_results_per_task.items():
        outfile.write(f"Task: {task}\n")
        for tau, rates in tau_data.items():
            outfile.write(f"  Tau={tau}:\n")
            for i, rate in enumerate(rates):
                rank = ranks[i]
                outfile.write(f"    Rank {rank}: {rate}\n")
        outfile.write("\n")
