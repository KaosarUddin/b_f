
import numpy as np
import os
from scipy.linalg import fractional_matrix_power
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy.linalg import logm, sqrtm, norm

def generate_file_paths(base_path, task, scan_type, num_subjects=428):
    file_paths = []
    subject_ids = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    subject_ids = subject_ids[:num_subjects]  
    for subject_id in subject_ids:
        file_name = f'{subject_id}_{prefix(task)}_{task}_{scan_type}_100'
        file_path = os.path.join(base_path, subject_id, file_name)
        file_paths.append(file_path)
    return file_paths

def prefix(task):
    return 'rfMRI' if task == 'REST1' else 'tfMRI'

def load_connectivity_matrix(file_path):
    try:
        return np.loadtxt(file_path, delimiter=' ')
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None



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

#base_path = 'F:/Research AU/Python/connectomes_100/'
base_path = '/mmfs1/home/mzu0014/connectomes_100/'  # Adjusted to an example server path
tasks = ['REST1', 'EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM']
alpha_z_values = [(0.5, 0.5), (0.8, 0.8)]

id_rate_results = {}

for task in tasks:
    id_rate_results[task] = {}
    for alpha, z in alpha_z_values:
        lr_paths = generate_file_paths(base_path, task, 'LR', 428)  
        rl_paths = generate_file_paths(base_path, task, 'RL', 428)
        
        connectivity_matrices_lr = [load_connectivity_matrix(path) for path in lr_paths if path is not None]
        connectivity_matrices_rl = [load_connectivity_matrix(path) for path in rl_paths if path is not None]

        bw_distance_matrix_1 = BW_distance_matrix(connectivity_matrices_lr, connectivity_matrices_rl, alpha, z)
        id_rate_1 = compute_id_rate(bw_distance_matrix_1)

        bw_distance_matrix_2 = BW_distance_matrix(connectivity_matrices_rl, connectivity_matrices_lr, alpha, z)
        id_rate_2 = compute_id_rate(bw_distance_matrix_2)
        
        current_id_rate = (id_rate_1 + id_rate_2) / 2
        id_rate_results[task][(alpha, z)] = current_id_rate
        print(f"Task: {task}, Alpha: {alpha}, Z: {z}, Combined ID Rate: {current_id_rate}")

# Save the results to a file on the server
output_file_path = '/mmfs1/home/mzu0014/project1/id_rate_az_at_results_summary.txt'
with open(output_file_path, 'w') as file:
    for task, alpha_z_rates in id_rate_results.items():
        file.write(f"Task: {task}\n")
        for (alpha, z), id_rate in alpha_z_rates.items():
            file.write(f"Alpha: {alpha}, Z: {z}, ID Rate: {id_rate}\n")
        file.write("\n")

# Plotting
plt.figure(figsize=(12, 8))

# To keep track of task and alpha-z configurations on the x-axis
task_alpha_z_labels = []
id_rates = []
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Example colors for differentiation

# Generate indices for each task and alpha-z configuration
for task_index, (task, data) in enumerate(id_rate_results.items()):
    for alpha_z_index, ((alpha, z), id_rate) in enumerate(data.items()):
        label = f"{task} (α={alpha}, Z={z})"
        if label not in task_alpha_z_labels:
            task_alpha_z_labels.append(label)
        id_rates.append((task_alpha_z_labels.index(label), id_rate))

# Split the generated indices and ID rates for plotting
x_indices, y_values = zip(*id_rates)

# Plotting the line
plt.plot(x_indices, y_values, 'o-', label=f"ID Rate by Task and α-Z")

# Adjusting the x-axis to show our custom labels
plt.xticks(range(len(task_alpha_z_labels)), task_alpha_z_labels, rotation=45, ha="right")

plt.title('Identification Rates Across Tasks and Alpha-Z Configurations')
plt.xlabel('Task and Alpha-Z Configuration')
plt.ylabel('Identification Rate')
plt.legend(loc='best')
plt.tight_layout()

# Saving the plot
plot_file_path = '/mmfs1/home/mzu0014/project1/id_rates_az1_at_plot.png'
plt.savefig(plot_file_path)
plt.close()
