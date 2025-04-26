import numpy as np
import os
from scipy.linalg import logm,norm,sqrtm
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for matplotlib
import matplotlib.pyplot as plt
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

def make_spd(matrix, tau=.01):
    symmetric_matrix = (matrix + matrix.T) / 2
    regularized_matrix = symmetric_matrix + tau * np.eye(matrix.shape[0])
    return regularized_matrix


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



def BW_distance_matrix(connectivity_matrices_1, connectivity_matrices_2, alpha, z,tau=.01):
    num_subjects = len(connectivity_matrices_1)
    distance_matrix = np.zeros((num_subjects, num_subjects))
    for i, matrix_1 in enumerate(connectivity_matrices_1):
        matrix_1 = make_spd(matrix_1, tau=tau)
        for j, matrix_2 in enumerate(connectivity_matrices_2):
            matrix_2 = make_spd(matrix_2, tau=tau)
            distance_matrix[i, j] = compute_alpha_z_BW_distance(matrix_1, matrix_2, alpha, z)
    return distance_matrix

def compute_id_rate(distance_matrix):
    correct_identifications = sum(np.argmin(distance_matrix[i, :]) == i for i in range(distance_matrix.shape[0]))
    return correct_identifications / distance_matrix.shape[0]

base_path = '/mmfs1/home/mzu0014/connectomes_100/'
lr_paths = generate_file_paths(base_path, 'LR')
rl_paths = generate_file_paths(base_path, 'RL')

connectivity_matrices_lr = [load_connectivity_matrix(path) for path in lr_paths]
connectivity_matrices_rl = [load_connectivity_matrix(path) for path in rl_paths]

tau =0  # Adjusted regularization parameter

#alpha=0.99
#z=1
#bw_distance_matrix_1 = BW_distance_matrix(connectivity_matrices_lr, connectivity_matrices_rl, alpha, z,tau=tau)
#id_rate_1 = compute_id_rate(bw_distance_matrix_1)
#print("Id_rate:",id_rate_1)
#np.savetxt("distance_matrix_1_(.95,1).csv",bw_distance_matrix_1,delimiter="," )
#bw_distance_matrix_2 = BW_distance_matrix(connectivity_matrices_rl, connectivity_matrices_lr, alpha, z,tau=tau)
#id_rate_2 = compute_id_rate(bw_distance_matrix_2)
#current_id_rate = (id_rate_1 + id_rate_2) / 2
#print("Id_rate:",id_rate_2)
#print("Id_rate:",current_id_rate)


plt.figure(figsize=(12, 8))
alpha_values = np.linspace(0.1, 0.95, 10)
#alpha_values = [0.1 , 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
id_rate_results = []

for alpha in alpha_values:
    z_values = np.arange(start=alpha, stop=1.0, step=0.05)
    id_rates_for_alpha = []
    for z in z_values:
        bw_distance_matrix_1 = BW_distance_matrix(connectivity_matrices_lr, connectivity_matrices_rl, alpha, z,tau=tau)
        id_rate_1 = compute_id_rate(bw_distance_matrix_1)
        
        bw_distance_matrix_2 = BW_distance_matrix(connectivity_matrices_rl, connectivity_matrices_lr, alpha, z,tau=tau)
        id_rate_2 = compute_id_rate(bw_distance_matrix_2)
        
        current_id_rate = (id_rate_1 + id_rate_2) / 2
        id_rates_for_alpha.append(current_id_rate)
        id_rate_results.append((alpha, z, current_id_rate))

    plt.plot(z_values, id_rates_for_alpha, '-o', label=f'Alpha {alpha}')

plt.title('Identification Rates Across Z Values for Different Alphas')
plt.xlabel('Z Value')
plt.ylabel('Identification Rate')
plt.legend()
plt.grid(True)
plot_file_path = '/mmfs1/home/mzu0014/project1/id_rates_rest0_across_z_values.png'
plt.savefig(plot_file_path)
plt.close()


# Print results to the console
#for alpha, z, id_rate in id_rate_results:
    #print(f"Alpha: {alpha:.2f}, Z: {z:.2f}, ID Rate: {id_rate:.4f}")


output_file_path = "/mmfs1/home/mzu0014/project1/id_rate_az_div_rest0_results.txt"
with open(output_file_path, 'w') as file:
    for alpha, z, id_rate in id_rate_results:
        file.write(f"Alpha: {alpha}, Z: {z}, ID Rate: {id_rate}\n")
