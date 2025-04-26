import numpy as np
import os

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
        file_path = os.path.join(base_path, subject_id, f'{subject_id}_tfMRI_WM_{scan_type}_200')
        file_paths.append(file_path)
    return file_paths

def compute_ed_divergence(P, Q, eps=0):
    reg_I = eps * np.eye(P.shape[0])  # Regularization term
    average_matrix = (P + Q) / 2 + reg_I
    product_matrix = P @ Q + reg_I
    sign1, logdet1 = np.linalg.slogdet(average_matrix)
    sign2, logdet2 = np.linalg.slogdet(product_matrix)
    if sign1 <= 0 or sign2 <= 0:
        raise ValueError("Regularization failed to make matrices positive definite")
    ed_divergence =4*(logdet1 - 0.5 * logdet2)
    return np.real(ed_divergence)

#def compute_ed_divergence1(P, Q):
    #reg_I = eps * np.eye(P.shape[0])  # Regularization term
    average_matrix = (P + Q) / 2 
    product_matrix = P @ Q 
    sign1,logdet1 = np.linalg.slogdet(average_matrix)
    sign2,logdet2 = np.linalg.slogdet(product_matrix)
    ed_divergence =4*(logdet1 - 0.5 * logdet2)
    return np.real(ed_divergence)

def ed_distance_matrix(connectivity_matrices_1, connectivity_matrices_2):
    num_subjects = len(connectivity_matrices_1)
    distance_matrix = np.zeros((num_subjects, num_subjects))
    for i, matrix_1 in enumerate(connectivity_matrices_1):
        if matrix_1 is None:
            continue
        for j, matrix_2 in enumerate(connectivity_matrices_2):
            if matrix_2 is None:
                continue
            distance_matrix[i, j] = compute_ed_divergence(matrix_1, matrix_2)
    return distance_matrix

def compute_id_rate(distance_matrix):
    correct_identifications = sum(np.argmin(distance_matrix[i, :]) == i for i in range(distance_matrix.shape[0]))
    return correct_identifications / distance_matrix.shape[0]

base_path = 'D:/Research AU/connectomes_200/'
lr_paths = generate_file_paths(base_path, 'LR')
rl_paths = generate_file_paths(base_path, 'RL')

connectivity_matrices_lr = [load_connectivity_matrix(path) for path in lr_paths]
connectivity_matrices_rl = [load_connectivity_matrix(path) for path in rl_paths]

ed_distance_matrix_1 = ed_distance_matrix(connectivity_matrices_lr, connectivity_matrices_rl)
id_rate_1 = compute_id_rate(ed_distance_matrix_1)

ed_distance_matrix_2 = ed_distance_matrix(connectivity_matrices_rl, connectivity_matrices_lr)
id_rate_2 = compute_id_rate(ed_distance_matrix_2)
current_id_rate = (id_rate_1 + id_rate_2) / 2

results_path = "D:/Research AU/Python/identification_rates_ed_divergence.txt"
with open(results_path, 'w') as f:
    f.write(f"ID Rate 1: {id_rate_1}\n")
    f.write(f"ID Rate 2: {id_rate_2}\n")
    f.write(f"Average ID Rate: {current_id_rate}\n")

print("ID Rate 1:", id_rate_1)
print("ID Rate 2:", id_rate_2)
print("Average ID Rate:", current_id_rate)



import numpy as np
import os

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
        file_path = os.path.join(base_path, subject_id, f'{subject_id}_tfMRI_EMOTION_{scan_type}_100')
        file_paths.append(file_path)
    return file_paths

def make_spd(matrix, tau=10):
    symmetric_matrix = (matrix + matrix.T) / 2
    regularized_matrix = symmetric_matrix + tau * np.eye(matrix.shape[0])
    return regularized_matrix

def compute_ed_divergence(P, Q):
    average_matrix = (P + Q) / 2
    product_matrix = P @ Q
    sign1, logdet1 = np.linalg.slogdet(average_matrix)
    sign2, logdet2 = np.linalg.slogdet(product_matrix)
    if sign1 <= 0 or sign2 <= 0:
        raise ValueError("Regularization failed to make matrices positive definite")
    ed_divergence = 4 * (logdet1 - 0.5 * logdet2)
    return np.real(ed_divergence)



def distance_matrix(connectivity_matrices_1, connectivity_matrices_2, tau=30):
    num_subjects = len(connectivity_matrices_1)
    distances = np.zeros((num_subjects, num_subjects))
    for i, matrix_1 in enumerate(connectivity_matrices_1):
        if matrix_1 is None:
            continue
        matrix_1 = make_spd(matrix_1, tau)
        for j, matrix_2 in enumerate(connectivity_matrices_2):
            if matrix_2 is None:
                continue
            matrix_2 = make_spd(matrix_2, tau)
            distances[i, j] = compute_ed_divergence(matrix_1, matrix_2)
    return distances

def compute_id_rate(distance_matrix):
    num_subjects = distance_matrix.shape[0]
    correct_identifications = 0
    for i in range(num_subjects):
        # Assuming the matrix's smallest value on the diagonal would mean correct identification
        if np.argmin(distance_matrix[i, :]) == i:
            correct_identifications += 1
    return correct_identifications / num_subjects if num_subjects > 0 else 0

# Assuming the rest of your script is already loaded and running as you have described
base_path = 'D:/Research AU/Python/connectomes_100/'
lr_paths = generate_file_paths(base_path, 'LR')
rl_paths = generate_file_paths(base_path, 'RL')

connectivity_matrices_lr = [load_connectivity_matrix(path) for path in lr_paths]
connectivity_matrices_rl = [load_connectivity_matrix(path) for path in rl_paths]
tau_values = [0]
# Compute distances
distance_matrix_1 = distance_matrix(connectivity_matrices_lr, connectivity_matrices_rl)
distance_matrix_2 = distance_matrix(connectivity_matrices_rl, connectivity_matrices_lr)

# Compute identification rates
id_rate_1 = compute_id_rate(distance_matrix_1)
id_rate_2 = compute_id_rate(distance_matrix_2)
current_id_rate = (id_rate_1 + id_rate_2) / 2

# Output the results
print("ID Rate 1:", id_rate_1)
print("ID Rate 2:", id_rate_2)
print("Average ID Rate:", current_id_rate)



import numpy as np
import os

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
        file_path = os.path.join(base_path, subject_id, f'{subject_id}_tfMRI_LANGUAGE_{scan_type}_400')
        file_paths.append(file_path)
    return file_paths

def make_spd(matrix, tau):
    symmetric_matrix = (matrix + matrix.T) / 2
    regularized_matrix = symmetric_matrix + tau * np.eye(matrix.shape[0])
    return regularized_matrix

def compute_ed_divergence(P, Q, tau):
    P = make_spd(P, tau)
    Q = make_spd(Q, tau)
    average_matrix = (P + Q) / 2
    product_matrix = P @ Q
    sign1, logdet1 = np.linalg.slogdet(average_matrix)
    sign2, logdet2 = np.linalg.slogdet(product_matrix)
    if sign1 <= 0 or sign2 <= 0:
        raise ValueError("Regularization failed to make matrices positive definite")
    ed_divergence = 4 * (logdet1 - 0.5 * logdet2)
    return np.real(ed_divergence)

def distance_matrix(connectivity_matrices_1, connectivity_matrices_2, tau):
    num_subjects = len(connectivity_matrices_1)
    distances = np.zeros((num_subjects, num_subjects))
    for i, matrix_1 in enumerate(connectivity_matrices_1):
        if matrix_1 is None:
            continue
        for j, matrix_2 in enumerate(connectivity_matrices_2):
            if matrix_2 is None:
                continue
            distances[i, j] = compute_ed_divergence(matrix_1, matrix_2, tau)
    return distances

def compute_id_rate(distance_matrix):
    num_subjects = distance_matrix.shape[0]
    correct_identifications = 0
    for i in range(num_subjects):
        if np.argmin(distance_matrix[i, :]) == i:
            correct_identifications += 1
    return correct_identifications / num_subjects if num_subjects > 0 else 0

def main():
    base_path = 'D:/Research AU/connectomes_400/'
    lr_paths = generate_file_paths(base_path, 'LR')
    rl_paths = generate_file_paths(base_path, 'RL')
    connectivity_matrices_lr = [load_connectivity_matrix(path) for path in lr_paths]
    connectivity_matrices_rl = [load_connectivity_matrix(path) for path in rl_paths]
    tau_values = [ 50]

    for tau in tau_values:
        print(f"Running for tau = {tau}")
        distance_matrix_1 = distance_matrix(connectivity_matrices_lr, connectivity_matrices_rl, tau)
        distance_matrix_2 = distance_matrix(connectivity_matrices_rl, connectivity_matrices_lr, tau)
        id_rate_1 = compute_id_rate(distance_matrix_1)
        id_rate_2 = compute_id_rate(distance_matrix_2)
        current_id_rate = (id_rate_1 + id_rate_2) / 2

        print(f"ID Rate 1 for tau = {tau}: {id_rate_1}")
        print(f"ID Rate 2 for tau = {tau}: {id_rate_2}")
        print(f"Average ID Rate for tau = {tau}: {current_id_rate}")

if __name__ == "__main__":
    main()



