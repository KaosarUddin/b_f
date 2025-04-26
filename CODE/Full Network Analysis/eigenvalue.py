import numpy as np

# Path to your file
file_path = 'D:/Research AU/Python/connectomes_100/100206/100206_rfMRI_REST1_LR_100'

# Load the matrix from the file
matrix = np.loadtxt(file_path)

# Calculate the eigenvalues of the matrix
eigenvalues, _ = np.linalg.eig(matrix+np.eye(matrix.shape[0])*.01)

# Display the eigenvalues
print(eigenvalues)
total_eigenvalues = len(eigenvalues)

# Display the total number of eigenvalues
print("Total number of eigenvalues:", total_eigenvalues)

smallest_eigenvalue = min(eigenvalues)
lrei=max(eigenvalues)
# Display the smallest eigenvalue
print("Smallest eigenvalue:", smallest_eigenvalue)
print("Smallest eigenvalue:", lrei)