### Final code for 3D figure.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D
import os
from matplotlib.patches import FancyBboxPatch

# Function to generate file paths for subjects' connectivity matrices
def generate_file_paths(base_path, scan_type, num_subjects=5):
    file_paths = []
    subject_ids = sorted(os.listdir(base_path))[:num_subjects]
    for subject_id in subject_ids:
        if os.path.isdir(os.path.join(base_path, subject_id)):
            file_path = os.path.join(base_path, subject_id, f'{subject_id}_rfMRI_REST1_{scan_type}_400')
            if os.path.exists(file_path):
                file_paths.append(file_path)
    return file_paths

# Function to load connectivity matrix from a file
def load_connectivity_matrix(file_path):
    try:
        matrix = np.loadtxt(file_path, delimiter=' ')
        if matrix.shape[0] != matrix.shape[1] or not np.allclose(matrix, matrix.T):
            print(f"Matrix at {file_path} is not square or symmetric. Skipping.")
            return None
        return matrix
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Function to compute the geodesic (AI) distance
def compute_geodesic_distance(A, B):
    from scipy.linalg import sqrtm, logm
    C = np.dot(np.linalg.inv(sqrtm(A)), B)
    C = np.dot(C, np.linalg.inv(sqrtm(A)))
    logC = logm(C)
    distance = np.linalg.norm(logC, 'fro')
    return distance

# Function to compute the Alpha-Z BW divergence distance
def compute_alpha_z_BW_distance(A, B, alpha=0.99, z=1):
    from scipy.linalg import fractional_matrix_power
    part1 = fractional_matrix_power(B, (1-alpha)/(2*z))
    part2 = fractional_matrix_power(A, alpha/z)
    part3 = fractional_matrix_power(B, (1-alpha)/(2*z))
    Q_az = fractional_matrix_power(np.dot(np.dot(part1, part2), part3), z)
    divergence = np.trace((1-alpha) * A + alpha * B) - np.trace(Q_az)
    return np.real(divergence)

# Function to compute the distance matrix for a set of connectivity matrices
def compute_distance_matrix(matrices, distance_func):
    num_matrices = len(matrices)
    distance_matrix = np.zeros((num_matrices, num_matrices))
    for i in range(num_matrices):
        for j in range(i+1, num_matrices):
            distance = distance_func(matrices[i], matrices[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  # Symmetry
    np.fill_diagonal(distance_matrix, 0)
    return distance_matrix

# Function to compute match percentages based on normalized distances
def compute_match_percentages(distance_matrix):
    max_distance = np.max(distance_matrix)
    if max_distance == 0:
        match_percentages = np.ones_like(distance_matrix) * 100
    else:
        match_percentages = 100 * (1 - distance_matrix / max_distance)
    return np.round(match_percentages, 2)

# Function to convert subject IDs to Subject 1, Subject 2 format
def convert_to_generic_subject_ids(subject_ids):
    subject_base_names = sorted(set([sid.split('_')[0] for sid in subject_ids]))
    base_to_subj = {orig: f'Subject {i+1}' for i, orig in enumerate(subject_base_names)}
    new_subject_ids = [f"{base_to_subj[sid.split('_')[0]]}_{sid.split('_')[1]}" for sid in subject_ids]
    return new_subject_ids

# Function to visualize the 3D embeddings with match percentages and highlighted box
def visualize_combined_3d_with_same_subject_percentages(matrices, subject_ids, distance_matrix, ax, title):
    mds = MDS(n_components=3, dissimilarity="precomputed", random_state=42)
    try:
        embeddings = mds.fit_transform(distance_matrix)
    except ValueError as e:
        print(f"Error in MDS fit_transform: {e}")
        return

    match_percentages = compute_match_percentages(distance_matrix)
    box_lines = []

    for i, embedding in enumerate(embeddings):
        color = 'red' if 'LR' in subject_ids[i] else 'blue'
        ax.scatter(*embedding, color=color, s=50)
        ax.text(*embedding, f'{subject_ids[i]}', color='black', fontsize=8, weight='bold')

    for i, subject_id_lr in enumerate(subject_ids):
        if '_LR' in subject_id_lr:
            base_id = subject_id_lr.replace('_LR', '')
            for j, subject_id_rl in enumerate(subject_ids):
                if subject_id_rl == base_id + '_RL':
                    match_percentage = match_percentages[i, j]
                    subject_num = base_id.split()[1] if ' ' in base_id else base_id
                    box_lines.append(f"{base_id}: {match_percentage}%")

    # Draw single annotation box with all subjects
    ax.text2D(0.02, 0.98, '\n'.join(box_lines), transform=ax.transAxes,
              bbox=dict(boxstyle="round,pad=0.8", edgecolor="black", facecolor="#d2f8d2"),
              fontsize=12, verticalalignment='top', color='black')

    ax.set_facecolor('lightyellow')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Component 1', fontsize=10)
    ax.set_ylabel('Component 2', fontsize=10)
    ax.set_zlabel('Component 3', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='z', which='major', labelsize=8)
#f5f5ff
# Example usage
base_path = 'D:/Research AU/connectomes_400/'
lr_paths = generate_file_paths(base_path, 'LR', 5)
rl_paths = generate_file_paths(base_path, 'RL', 5)
subject_ids = [os.path.basename(path).split('_')[0] + '_LR' for path in lr_paths] + \
               [os.path.basename(path).split('_')[0] + '_RL' for path in rl_paths]
subject_ids = convert_to_generic_subject_ids(subject_ids)
matrices = [matrix for path in lr_paths + rl_paths if (matrix := load_connectivity_matrix(path)) is not None]

if matrices:
    distance_matrix_ai = compute_distance_matrix(matrices, compute_geodesic_distance)
    distance_matrix_alpha_z = compute_distance_matrix(matrices, compute_alpha_z_BW_distance)

    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    visualize_combined_3d_with_same_subject_percentages(matrices, subject_ids, distance_matrix_ai, ax1, "AI Distance")

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    visualize_combined_3d_with_same_subject_percentages(matrices, subject_ids, distance_matrix_alpha_z, ax2, "Alpha Z Divergence")

    plt.tight_layout()
    plt.show()
else:
    print("No valid matrices were loaded. Please check file paths and matrix validity.")

