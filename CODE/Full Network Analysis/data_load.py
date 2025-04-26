import os
import numpy as np
import matplotlib.pyplot as plt

# Base path where all subject folders are located
#base_path = 'D:/Research AU/connectomes_200'
base_path = 'connectomes_100'

# Generate file paths
file_paths = []
for subject_id in os.listdir(base_path):
    # Check if the path is a directory (to avoid reading non-directory files)
    if os.path.isdir(os.path.join(base_path, subject_id)):
        # Assuming each subject has both LR and RL files
        lr_file = f'{base_path}/{subject_id}/{subject_id}_rfMRI_REST1_LR_100'
        rl_file = f'{base_path}/{subject_id}/{subject_id}_rfMRI_REST1_RL_100'
        file_paths.extend([lr_file, rl_file])

# Now file_paths contains the paths for all subjects
print(f"Total files: {len(file_paths)}")
for i, file_path in enumerate(file_paths, start=1):
    try:
        # Load the data
        connectivity_data = np.loadtxt(file_path, delimiter=' ')
        print(f"Data from Task {i} loaded successfully.")
        print(f"Data shape: {connectivity_data.shape}")

        #Plot the data
        #plt.figure(figsize=(10, 4))  # Adjust the size as needed
        #plt.imshow(connectivity_data, aspect='auto', interpolation='none')
        #plt.colorbar()
        #plt.title(f'Connectivity Data Visualization - Task {i}')
        #plt.show()
        

    except Exception as e:
        print(f"An error occurred for Task {i}: {e}")



