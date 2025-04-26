import numpy as np
import os
from scipy.linalg import fractional_matrix_power
import networkx as nx
from scipy.io import loadmat

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

def compute_network_metrics(connectivity_matrix):
    """
    Compute node-level metrics (degree, betweenness centrality, etc.) for the given matrix.
    """
    G = nx.from_numpy_array(connectivity_matrix)
    degree = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    clustering = nx.clustering(G)
    return degree, betweenness, clustering

def group_metrics_by_network(metrics, yeo_labels, num_networks=7):
    """
    Group node-level metrics by Yeo networks and calculate mean and std.
    """
    grouped_metrics = {}
    for network_id in range(1, num_networks + 1):
        indices = [i for i, label in enumerate(yeo_labels) if label == network_id]
        network_metrics = {key: [metrics[key][i] for i in indices] for key in metrics.keys()}
        grouped_metrics[network_id] = {
            'mean': {key: np.mean(values) for key, values in network_metrics.items()},
            'std': {key: np.std(values) for key, values in network_metrics.items()}
        }
    return grouped_metrics

# Load Yeo's network labels
yeo_rois_path = 'C:/Users/ksrru/Documents/updated_LUT/updated_LUT/YeoROIs_N100.mat' # Replace with actual path
yeo_data = loadmat(yeo_rois_path)
yeo_labels = yeo_data['YeoROIs'].flatten()# Assuming 'YeoLabels' contains the labels for ROIs

# Paths to data
base_path = 'D:/Research AU/Python/connectomes_100/'
lr_paths = generate_file_paths(base_path, 'LR')
rl_paths = generate_file_paths(base_path, 'RL')

# Load connectivity matrices
connectivity_matrices_lr = [load_connectivity_matrix(path) for path in lr_paths]
connectivity_matrices_rl = [load_connectivity_matrix(path) for path in rl_paths]

alpha = 0.99
z = 1

# Compute ID rate
bw_distance_matrix_1 = BW_distance_matrix(connectivity_matrices_lr, connectivity_matrices_rl, alpha, z)
id_rate_1 = compute_id_rate(bw_distance_matrix_1)

bw_distance_matrix_2 = BW_distance_matrix(connectivity_matrices_rl, connectivity_matrices_lr, alpha, z)
id_rate_2 = compute_id_rate(bw_distance_matrix_2)

current_id_rate = (id_rate_1 + id_rate_2) / 2
print(f"Current ID Rate: {current_id_rate}")

# Compute node-level metrics and group by network
network_metrics = []
for matrix in connectivity_matrices_lr:
    if matrix is not None:
        degree, betweenness, clustering = compute_network_metrics(matrix)
        metrics = {
            'degree': degree,
            'betweenness': betweenness,
            'clustering': clustering
        }
        grouped_metrics = group_metrics_by_network(metrics, yeo_labels)
        network_metrics.append(grouped_metrics)

# Print metrics for each network
for network_id, stats in grouped_metrics.items():
    print(f"Network {network_id} - Mean Metrics: {stats['mean']}")
    print(f"Network {network_id} - Std Metrics: {stats['std']}")

import numpy as np
import os
from scipy.linalg import fractional_matrix_power
import networkx as nx
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

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

def compute_alpha_z_BW_distance(A, B, alpha, z):
    if not (0 <= alpha <= z <= 1):
        raise ValueError("Alpha and z must satisfy 0 <= alpha <= z <= 1")

    def Q_alpha_z(A, B, alpha, z):
        part1 = fractional_matrix_power(B, (1-alpha)/(2*z))
        part2 = fractional_matrix_power(A, alpha/z)
        part3 = fractional_matrix_power(B, (1-alpha)/(2*z))
        return fractional_matrix_power(part1.dot(part2).dot(part3), z)

    Q_az = Q_alpha_z(A, B, alpha, z)
    return np.real(np.trace((1-alpha) * A + alpha * B) - np.trace(Q_az))

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

def compute_node_metrics(adjacency_matrix):
    G = nx.from_numpy_array(adjacency_matrix)
    degree = dict(G.degree())
    betweenness = nx.betweenness_centrality(G)
    return {'degree': degree, 'betweenness': betweenness}

def aggregate_metrics(node_metrics, network_labels):
    metrics_by_network = {}
    for network in set(network_labels):
        nodes_in_network = [i for i, label in enumerate(network_labels) if label == network]
        metrics_by_network[network] = {
            key: {
                'mean': np.mean([node_metrics[key][node] for node in nodes_in_network]),
                'std': np.std([node_metrics[key][node] for node in nodes_in_network])
            } for key in node_metrics.keys()
        }
    return metrics_by_network

def compare_metrics_with_id_rates(network_metrics, id_rates):
    correlations = {}
    for metric in list(network_metrics.values())[0].keys():
        metric_means = [network_metrics[network][metric]['mean'] for network in network_metrics.keys()]
        if len(set(metric_means)) == 1 or len(set(id_rates)) == 1:
            correlations[metric] = {'correlation': np.nan, 'p_value': np.nan}
            continue
        corr, p_value = pearsonr(metric_means, id_rates)
        correlations[metric] = {'correlation': corr, 'p_value': p_value}
    return correlations

# Load Yeo's network labels
yeo_rois_path = 'C:/Users/ksrru/Documents/updated_LUT/updated_LUT/YeoROIs_N100.mat' # Replace with actual path
yeo_data = loadmat(yeo_rois_path)
yeo_labels = yeo_data['YeoROIs'].flatten()# Assuming 'YeoLabels' contains the labels for ROIs
# Paths to data
base_path = 'D:/Research AU/Python/connectomes_100/'
lr_paths = generate_file_paths(base_path, 'LR')
rl_paths = generate_file_paths(base_path, 'RL')

# Load connectivity matrices
connectivity_matrices_lr = [load_connectivity_matrix(path) for path in lr_paths]
connectivity_matrices_rl = [load_connectivity_matrix(path) for path in rl_paths]

# Compute node-level metrics and aggregate by network
id_rates = []
network_metrics = {}
alpha = 0.99
z = 1

for i, matrix in enumerate(connectivity_matrices_lr):
    if matrix is not None and connectivity_matrices_rl[i] is not None:
        bw_distance = BW_distance_matrix([matrix], [connectivity_matrices_rl[i]], alpha, z)
        subject_id_rate = compute_id_rate(bw_distance)
        id_rates.append(subject_id_rate)
        
        node_metrics = compute_node_metrics(matrix)
        grouped_metrics = aggregate_metrics(node_metrics, yeo_labels)
        network_metrics[i] = grouped_metrics

# Compare and plot
correlations = compare_metrics_with_id_rates(network_metrics, id_rates)
for metric in correlations.keys():
    metric_means = [network_metrics[network][metric]['mean'] for network in network_metrics.keys()]
    plt.scatter(metric_means, id_rates, label=f'{metric} (r={correlations[metric]["correlation"]:.2f})')

plt.xlabel('Network Metric Mean')
plt.ylabel('ID Rate')
plt.legend()
plt.title('Relationship Between Network Metrics and ID Rates')
plt.show()



import networkx as nx
import community as community_louvain
import numpy as np
import networkx as nx
from sklearn.metrics import modularity_score

# Example FC dataset
fc_matrix = np.random.rand(114, 114)  # Replace with your actual FC matrix
np.fill_diagonal(fc_matrix, 0)  # Remove self-connections
threshold = 0.5  # Example threshold

# Threshold the matrix to create a binary adjacency matrix
adj_matrix = (fc_matrix > threshold).astype(int)

# Convert adjacency matrix to a graph
G = nx.from_numpy_matrix(adj_matrix)

# Compute node-level metrics
node_modularity = nx.community.greedy_modularity_communities(G)  # Modularity
node_betweenness = nx.betweenness_centrality(G)  # Betweenness centrality
node_degree = dict(G.degree())  # Degree
node_clustering = nx.clustering(G)  # Clustering coefficient
# Print results
print("Modularity:", node_modularity)
print("Betweenness Centrality:", node_betweenness)
print("Degree:", node_degree)
print("Clustering Coefficient:", node_clustering)


import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from community import community_louvain  # Ensure this library is installed: pip install python-louvain

# Function to load connectivity matrix from a file
def load_connectivity_matrix(file_path):
    try:
        return np.loadtxt(file_path, delimiter=' ')
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Function to generate file paths for a specific scan type and number of subjects
def generate_file_paths(base_path, scan_type, num_subjects=30):
    file_paths = []
    subject_ids = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    subject_ids.sort()
    subject_ids = subject_ids[:num_subjects]
    for subject_id in subject_ids:
        file_path = os.path.join(base_path, subject_id, f'{subject_id}_rfMRI_REST1_{scan_type}_100')
        file_paths.append(file_path)
    return file_paths  # Corrected return type to list

# Base path for FC dataset
base_path = 'D:/Research AU/Python/connectomes_100/'

# Generate file paths for LR and RL scans
lr_paths = generate_file_paths(base_path, 'LR')
rl_paths = generate_file_paths(base_path, 'RL')

# Load connectivity matrices
connectivity_matrices_lr = [load_connectivity_matrix(path) for path in lr_paths if os.path.exists(path)]
connectivity_matrices_rl = [load_connectivity_matrix(path) for path in rl_paths if os.path.exists(path)]

# Filter out None values in case of loading errors
connectivity_matrices_lr = [mat for mat in connectivity_matrices_lr if mat is not None]
connectivity_matrices_rl = [mat for mat in connectivity_matrices_rl if mat is not None]

# Process and visualize the first matrix (example with LR scan)
if connectivity_matrices_lr:
    fc_matrix = connectivity_matrices_lr[1]  # Use the first subject's matrix
    np.fill_diagonal(fc_matrix, 0)  # Remove self-connections
    threshold = 0.5  # Threshold for adjacency matrix creation

    # Threshold the FC matrix to create a binary adjacency matrix
    adj_matrix = (fc_matrix > threshold).astype(int)

    # Convert adjacency matrix to a graph
    G = nx.from_numpy_array(adj_matrix)

    # Compute node-level metrics
    node_modularity = list(nx.community.greedy_modularity_communities(G))  # Modularity
    node_betweenness = nx.betweenness_centrality(G)  # Betweenness centrality
    node_degree = dict(G.degree())  # Degree
    node_clustering = nx.clustering(G)  # Clustering coefficient

    # Print results
    print(f"Number of Communities Detected: {len(node_modularity)}")
    print("Betweenness Centrality:", node_betweenness)
    print("Degree:", node_degree)
    print("Clustering Coefficient:", node_clustering)

    # Visualize the graph
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G, pos, with_labels=True, node_size=50, node_color="skyblue", edge_color="gray"
    )
    plt.title("Graph Representation of Functional Connectivity Matrix")
    plt.show()
else:
    print("No valid connectivity matrices were loaded.")


# Load Yeo's network labels
yeo_rois_path = 'C:/Users/ksrru/Documents/updated_LUT/updated_LUT/YeoROIs_N100.mat' # Replace with actual path
yeo_data = loadmat(yeo_rois_path)
yeo_labels = yeo_data['YeoROIs'].flatten()# Assuming 'YeoLabels' contains the labels for ROIs
# Example Yeo network assignment (200 nodes, 7 networks)
yeo_assignment = np.random.randint(1, 8, size=114)  # Replace with actual Yeo labels

# Aggregate metrics for each Yeo network
yeo_metrics = {}
for network in np.unique(yeo_assignment):
    nodes_in_network = np.where(yeo_assignment == network)[0]
    yeo_metrics[network] = {
        "avg_modularity": np.mean([node_modularity[node] for node in nodes_in_network]),
        "avg_betweenness": np.mean([node_betweenness[node] for node in nodes_in_network]),
        "avg_degree": np.mean([node_degree[node] for node in nodes_in_network]),
        "avg_clustering": np.mean([node_clustering[node] for node in nodes_in_network]),
    }

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Example: Create a DataFrame for statistical analysis
yeo_df = pd.DataFrame.from_dict(yeo_metrics, orient="index")
yeo_df["ID_rate"] = np.random.rand(len(yeo_df))  # Replace with actual ID rate

# Plot relationships
sns.pairplot(yeo_df)
plt.show()

# Regression example
import statsmodels.api as sm
X = yeo_df[["avg_modularity", "avg_betweenness", "avg_degree", "avg_clustering"]]
y = yeo_df["ID_rate"]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

import os
import numpy as np
import networkx as nx
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from community import community_louvain  # Ensure this library is installed: pip install python-louvain
from scipy.io import loadmat
import statsmodels.api as sm

# Function to load connectivity matrix from a file
def load_connectivity_matrix(file_path):
    try:
        return np.loadtxt(file_path, delimiter=' ')
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Function to generate file paths for a specific scan type and number of subjects
def generate_file_paths(base_path, scan_type, num_subjects=30):
    file_paths = []
    subject_ids = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    subject_ids.sort()
    subject_ids = subject_ids[:num_subjects]
    for subject_id in subject_ids:
        file_path = os.path.join(base_path, subject_id, f'{subject_id}_rfMRI_REST1_{scan_type}_100')
        file_paths.append(file_path)
    return file_paths  # Corrected return type to list

# Base path for FC dataset
base_path = 'D:/Research AU/Python/connectomes_100/'

# Generate file paths for LR and RL scans
lr_paths = generate_file_paths(base_path, 'LR')
rl_paths = generate_file_paths(base_path, 'RL')

# Load connectivity matrices
connectivity_matrices_lr = [load_connectivity_matrix(path) for path in lr_paths if os.path.exists(path)]
connectivity_matrices_rl = [load_connectivity_matrix(path) for path in rl_paths if os.path.exists(path)]

# Filter out None values in case of loading errors
connectivity_matrices_lr = [mat for mat in connectivity_matrices_lr if mat is not None]
connectivity_matrices_rl = [mat for mat in connectivity_matrices_rl if mat is not None]

# Process and visualize the first matrix (example with LR scan)
if connectivity_matrices_lr:
    fc_matrix = connectivity_matrices_lr[0]  # Use the first subject's matrix
    np.fill_diagonal(fc_matrix, 0)  # Remove self-connections
    threshold = 0.5  # Threshold for adjacency matrix creation

    # Threshold the FC matrix to create a binary adjacency matrix
    adj_matrix = (fc_matrix > threshold).astype(int)

    # Convert adjacency matrix to a graph
    G = nx.from_numpy_array(adj_matrix)

    # Compute node-level metrics
    node_modularity = list(nx.community.greedy_modularity_communities(G))  # Modularity
    node_betweenness = nx.betweenness_centrality(G)  # Betweenness centrality
    node_degree = dict(G.degree())  # Degree
    node_clustering = nx.clustering(G)  # Clustering coefficient

    # Print results
    print(f"Number of Communities Detected: {len(node_modularity)}")
    print("Betweenness Centrality:", node_betweenness)
    print("Degree:", node_degree)
    print("Clustering Coefficient:", node_clustering)

    # Visualize the graph
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G, pos, with_labels=True, node_size=50, node_color="skyblue", edge_color="gray"
    )
    plt.title("Graph Representation of Functional Connectivity Matrix")
    plt.show()

    # Simulate Yeo network assignments (replace with actual data)
    yeo_assignment = np.random.randint(1, 8, size=len(fc_matrix))  # 7 Yeo networks, 114 nodes

    # Aggregate metrics for each Yeo network
    yeo_metrics = {}
    for network in np.unique(yeo_assignment):
        nodes_in_network = np.where(yeo_assignment == network)[0]
        yeo_metrics[network] = {
            "avg_modularity": np.mean([node_degree.get(node, 0) for node in nodes_in_network]),
            "avg_betweenness": np.mean([node_betweenness.get(node, 0) for node in nodes_in_network]),
            "avg_degree": np.mean([node_degree.get(node, 0) for node in nodes_in_network]),
            "avg_clustering": np.mean([node_clustering.get(node, 0) for node in nodes_in_network]),
        }

    # Create a DataFrame for analysis
    yeo_df = pd.DataFrame.from_dict(yeo_metrics, orient="index")
    yeo_df["ID_rate"] = np.random.rand(len(yeo_df))  # Random ID rates for demonstration

    # Visualize relationships
    sns.pairplot(yeo_df)
    plt.show()

    # Perform regression analysis
    X = yeo_df[["avg_modularity", "avg_betweenness", "avg_degree", "avg_clustering"]]
    y = yeo_df["ID_rate"]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print(model.summary())
else:
    print("No valid connectivity matrices were loaded.")



import os
import numpy as np
import networkx as nx
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from community import community_louvain  # Ensure this library is installed: pip install python-louvain
from scipy.io import loadmat
import statsmodels.api as sm

# Function to load connectivity matrix from a file
def load_connectivity_matrix(file_path):
    try:
        return np.loadtxt(file_path, delimiter=' ')
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Function to generate file paths for a specific scan type and number of subjects
def generate_file_paths(base_path, scan_type, num_subjects=30):
    file_paths = []
    subject_ids = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    subject_ids.sort()
    subject_ids = subject_ids[:num_subjects]
    for subject_id in subject_ids:
        file_path = os.path.join(base_path, subject_id, f'{subject_id}_rfMRI_REST1_{scan_type}_100')
        file_paths.append(file_path)
    return file_paths  # Corrected return type to list

# Base path for FC dataset
base_path = 'D:/Research AU/Python/connectomes_100/'

# Generate file paths for LR and RL scans
lr_paths = generate_file_paths(base_path, 'LR', num_subjects=30)
rl_paths = generate_file_paths(base_path, 'RL', num_subjects=30)

# Load connectivity matrices
connectivity_matrices_lr = [load_connectivity_matrix(path) for path in lr_paths if os.path.exists(path)]
connectivity_matrices_rl = [load_connectivity_matrix(path) for path in rl_paths if os.path.exists(path)]

# Filter out None values in case of loading errors
connectivity_matrices_lr = [mat for mat in connectivity_matrices_lr if mat is not None]
connectivity_matrices_rl = [mat for mat in connectivity_matrices_rl if mat is not None]

# Load Yeo ROIs labels
yeo_rois_path = 'C:/Users/ksrru/Documents/updated_LUT/updated_LUT/YeoROIs_N100.mat' # Replace with actual path
yeo_data = loadmat(yeo_rois_path)
yeo_labels = yeo_data['YeoROIs'].flatten()  # Assuming 'YeoROIs' contains the labels for ROIs

# Process all FC matrices
all_metrics_lr = []
all_metrics_rl = []

for fc_matrices, all_metrics in zip([connectivity_matrices_lr, connectivity_matrices_rl], [all_metrics_lr, all_metrics_rl]):
    for idx, fc_matrix in enumerate(fc_matrices):
        if fc_matrix is not None:
            np.fill_diagonal(fc_matrix, 0)  # Remove self-connections
            threshold = 0.5  # Threshold for adjacency matrix creation

            # Threshold the FC matrix to create a binary adjacency matrix
            adj_matrix = (fc_matrix > threshold).astype(int)

            # Convert adjacency matrix to a graph
            G = nx.from_numpy_array(adj_matrix)

            # Compute node-level metrics
            node_modularity = list(nx.community.greedy_modularity_communities(G))  # Modularity
            node_betweenness = nx.betweenness_centrality(G)  # Betweenness centrality
            eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6) # Eigenvector centrality
            node_clustering = nx.clustering(G)  # Clustering coefficient

            # Aggregate metrics for each Yeo network
            for network in np.unique(yeo_labels):
                nodes_in_network = np.where(yeo_labels == network)[0]
                avg_modularity = len(node_modularity)  # Number of communities detected
                avg_betweenness = np.mean([node_betweenness.get(node, 0) for node in nodes_in_network])
                avg_eigenvector = np.mean([eigenvector_centrality.get(node, 0) for node in nodes_in_network])
                avg_clustering = np.mean([node_clustering.get(node, 0) for node in nodes_in_network])

                # Append the metrics as a row
                all_metrics.append({
                    "avg_modularity": avg_modularity,
                    "avg_betweenness": avg_betweenness,
                    "avg_eigenvector": avg_eigenvector,
                    "avg_clustering": avg_clustering,
                    "network_id": network
                })

# Create DataFrames for LR and RL metrics
df_metrics_lr = pd.DataFrame(all_metrics_lr)
df_metrics_rl = pd.DataFrame(all_metrics_rl)

# Save or inspect the DataFrame
print(df_metrics_lr.head())
print(df_metrics_rl.head())

# Save to CSV for further inspection
df_metrics_lr.to_csv("D:/Research AU/Python/metrics_lr.csv", index=False)
df_metrics_rl.to_csv("D:/Research AU/Python/metrics_rl.csv", index=False)

all_subjects_df_lr.to_csv("D:/Research AU/Python/cleaned_lr_metrics.csv", index=False)
all_subjects_df_rl.to_csv("D:/Research AU/Python/cleaned_rl_metrics.csv", index=False)
# Combine metrics for LR and RL
all_subjects_df_lr = pd.DataFrame.from_records(all_metrics_lr)
all_subjects_df_rl = pd.DataFrame.from_records(all_metrics_rl)

print(all_subjects_df_lr.columns)
print(all_subjects_df_rl.columns)
print(yeo_metrics)



# Check RL matrices
print(f"Number of RL matrices: {len(connectivity_matrices_rl)}")

# Inspect the shape of the DataFrame
print("LR DataFrame shape:", all_subjects_df_lr.shape)
print("RL DataFrame shape:", all_subjects_df_rl.shape)

# Filter and rename columns
metrics_columns = ["avg_modularity", "avg_betweenness", "avg_eigenvector", "avg_clustering"]

if all_subjects_df_lr.shape[1] >= len(metrics_columns):
    all_subjects_df_lr = all_subjects_df_lr.iloc[:, :len(metrics_columns)]
    all_subjects_df_lr.columns = metrics_columns

if all_subjects_df_rl.shape[1] >= len(metrics_columns):
    all_subjects_df_rl = all_subjects_df_rl.iloc[:, :len(metrics_columns)]
    all_subjects_df_rl.columns = metrics_columns

# Add network IDs
all_subjects_df_lr["network_id"] = range(1, len(all_subjects_df_lr) + 1)
all_subjects_df_rl["network_id"] = range(1, len(all_subjects_df_rl) + 1)

# Validate changes
print("Updated LR DataFrame:")
print(all_subjects_df_lr.head())

print("Updated RL DataFrame:")
print(all_subjects_df_rl.head())




# Summary statistics for LR and RL
summary_lr = all_subjects_df_lr.describe().T
summary_lr["Dataset"] = "LR"

summary_rl = all_subjects_df_rl.describe().T
summary_rl["Dataset"] = "RL"

# Combine summaries
summary_combined = pd.concat([summary_lr, summary_rl])

print(summary_combined)

# Extract numeric values from dictionary columns
for df in [all_subjects_df_lr, all_subjects_df_rl]:
    for metric in metrics_columns:
        df[metric] = df[metric].apply(lambda x: x.get(metric) if isinstance(x, dict) else None)
        
print(all_subjects_df_lr.dtypes)
print(all_subjects_df_lr.head())

print(all_subjects_df_rl.dtypes)
print(all_subjects_df_rl.head())

for metric in metrics_columns:
    plt.figure(figsize=(10, 6))
    plt.bar(all_subjects_df_lr["network_id"], all_subjects_df_lr[metric], alpha=0.6, label="LR", color="blue")
    plt.bar(all_subjects_df_rl["network_id"], all_subjects_df_rl[metric], alpha=0.6, label="RL", color="orange")
    plt.xlabel("Yeo Network ID")
    plt.ylabel(metric.replace("avg_", "").capitalize())
    plt.title(f"{metric.replace('avg_', '').capitalize()} by Network (LR vs RL)")
    plt.legend()
    plt.show()





all_subjects_df_lr.to_csv("D:/Research AU/Python/cleaned_lr_metrics.csv", index=False)
all_subjects_df_rl.to_csv("D:/Research AU/Python/cleaned_rl_metrics.csv", index=False)






# Plot the effects of network science metrics
for metric in ["avg_modularity", "avg_betweenness", "avg_eigenvector", "avg_clustering"]:
    plt.figure(figsize=(10, 6))
    plt.plot(all_subjects_df_lr[metric], all_subjects_df_lr["ID_rate"], 'o', label=f"LR {metric}")
    plt.plot(all_subjects_df_rl[metric], all_subjects_df_rl["ID_rate"], 'x', label=f"RL {metric}")
    plt.xlabel(metric)
    plt.ylabel("ID_rate")
    plt.title(f"Effect of {metric} on ID Rate")
    plt.legend()
    plt.show()



# Perform regression analysis for LR and RL
for all_subjects_df, label in zip([all_subjects_df_lr, all_subjects_df_rl], ["LR", "RL"]):
    X = all_subjects_df[["avg_modularity", "avg_betweenness", "avg_eigenvector", "avg_clustering"]]
    y = all_subjects_df["ID_rate"]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print(f"Regression Results for {label}:")
    print(model.summary())





import os
import numpy as np
import networkx as nx
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from community import community_louvain  # Ensure this library is installed: pip install python-louvain
from scipy.io import loadmat
import statsmodels.api as sm

# Function to load connectivity matrix from a file
def load_connectivity_matrix(file_path):
    try:
        return np.loadtxt(file_path, delimiter=' ')
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Function to generate file paths for a specific scan type and number of subjects
def generate_file_paths(base_path, scan_type, num_subjects=428):
    file_paths = []
    subject_ids = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    subject_ids.sort()
    subject_ids = subject_ids[:num_subjects]
    for subject_id in subject_ids:
        file_path = os.path.join(base_path, subject_id, f'{subject_id}_rfMRI_REST1_{scan_type}_100')
        file_paths.append(file_path)
    return file_paths  # Corrected return type to list

# Base path for FC dataset
base_path = 'D:/Research AU/Python/connectomes_100/'

# Generate file paths for LR and RL scans
lr_paths = generate_file_paths(base_path, 'LR', num_subjects=428)
rl_paths = generate_file_paths(base_path, 'RL', num_subjects=428)

# Load connectivity matrices
connectivity_matrices_lr = [load_connectivity_matrix(path) for path in lr_paths if os.path.exists(path)]
connectivity_matrices_rl = [load_connectivity_matrix(path) for path in rl_paths if os.path.exists(path)]

# Filter out None values in case of loading errors
connectivity_matrices_lr = [mat for mat in connectivity_matrices_lr if mat is not None]
connectivity_matrices_rl = [mat for mat in connectivity_matrices_rl if mat is not None]

# Load Yeo ROIs labels
yeo_rois_path = 'C:/Users/ksrru/Documents/updated_LUT/updated_LUT/YeoROIs_N100.mat' # Replace with actual path
yeo_data = loadmat(yeo_rois_path)
yeo_labels = yeo_data['YeoROIs'].flatten()  # Assuming 'YeoROIs' contains the labels for ROIs

# Given ID rates for 7 networks
id_rate = {1: 0.3388, 2: 0.2734, 3: 0.2886, 4: 0.2862, 5: 0.0210, 6: 0.3516, 7: 0.4416}

# Process all FC matrices
all_metrics_lr = []
all_metrics_rl = []

for fc_matrices, all_metrics in zip([connectivity_matrices_lr, connectivity_matrices_rl], [all_metrics_lr, all_metrics_rl]):
    for idx, fc_matrix in enumerate(fc_matrices):
        if fc_matrix is not None:
            np.fill_diagonal(fc_matrix, 0)  # Remove self-connections
            threshold = 0.5  # Threshold for adjacency matrix creation

            # Threshold the FC matrix to create a binary adjacency matrix
            adj_matrix = (fc_matrix > threshold).astype(int)

            # Convert adjacency matrix to a graph
            G = nx.from_numpy_array(adj_matrix)

            # Compute node-level metrics
            node_modularity = list(nx.community.greedy_modularity_communities(G))  # Modularity
            node_betweenness = nx.betweenness_centrality(G)  # Betweenness centrality
            eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6) # Eigenvector centrality
            node_clustering = nx.clustering(G)  # Clustering coefficient

            # Aggregate metrics for each Yeo network
            for network in np.unique(yeo_labels):
                nodes_in_network = np.where(yeo_labels == network)[0]
                avg_modularity = len(node_modularity)  # Number of communities detected
                avg_betweenness = np.mean([node_betweenness.get(node, 0) for node in nodes_in_network])
                avg_eigenvector = np.mean([eigenvector_centrality.get(node, 0) for node in nodes_in_network])
                avg_clustering = np.mean([node_clustering.get(node, 0) for node in nodes_in_network])
                std_betweenness = np.std([node_betweenness.get(node, 0) for node in nodes_in_network])
                std_eigenvector = np.std([eigenvector_centrality.get(node, 0) for node in nodes_in_network])

                # Append the metrics as a row
                all_metrics.append({
                    "network_id": network,
                    "avg_modularity": avg_modularity,
                    "avg_betweenness": avg_betweenness,
                    "std_betweenness": std_betweenness,
                    "avg_eigenvector": avg_eigenvector,
                    "std_eigenvector": std_eigenvector,
                    "avg_clustering": avg_clustering,
                    "id_rate": id_rate[network]  # Map the network to its ID rate
                })

# Create DataFrames for LR and RL metrics
df_metrics_lr = pd.DataFrame(all_metrics_lr)
df_metrics_rl = pd.DataFrame(all_metrics_rl)

# Combine LR and RL metrics into a single DataFrame
df_combined = pd.concat([df_metrics_lr, df_metrics_rl])

# Group by network ID to aggregate metrics across all subjects
aggregated_metrics = df_combined.groupby("network_id").agg({
    "avg_modularity": ["mean", "std"],
    "avg_betweenness": ["mean", "std"],
    "avg_eigenvector": ["mean", "std"],
    "avg_clustering": ["mean", "std"],
}).reset_index()

# Flatten the MultiIndex columns for better readability
aggregated_metrics.columns = [
    "network_id", 
    "modularity_mean", "modularity_std",
    "betweenness_mean", "betweenness_std",
    "eigenvector_mean", "eigenvector_std",
    "clustering_mean", "clustering_std"
]

# Add the ID rate to the aggregated DataFrame
aggregated_metrics["id_rate"] = aggregated_metrics["network_id"].map(id_rate)
# Correlation analysis
correlation_matrix = aggregated_metrics.corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Visualize pairwise relationships
sns.pairplot(aggregated_metrics, vars=[
    "betweenness_mean", "eigenvector_mean", "clustering_mean", "id_rate"
])
plt.show()

# Regression analysis
X = df_combined[["avg_betweenness", "avg_eigenvector", "avg_clustering"]]
y = df_combined["id_rate"]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
# Regression analysis
X = aggregated_metrics[["betweenness_mean", "eigenvector_mean", "clustering_mean"]]
y = aggregated_metrics["id_rate"]
X = sm.add_constant(X)  # Add intercept
model = sm.OLS(y, X).fit()
print(model.summary())

# Save the combined DataFrame
df_combined.to_csv("D:/Research AU/Python/combined_metrics.csv", index=False)


import os
import numpy as np
import networkx as nx
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from community import community_louvain  # Ensure this library is installed: pip install python-louvain
from scipy.io import loadmat
import statsmodels.api as sm

# Function to load connectivity matrix from a file
def load_connectivity_matrix(file_path):
    try:
        return np.loadtxt(file_path, delimiter=' ')
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Function to generate file paths for a specific scan type and number of subjects
def generate_file_paths(base_path, scan_type, num_subjects=428):
    file_paths = []
    subject_ids = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    subject_ids.sort()
    subject_ids = subject_ids[:num_subjects]
    for subject_id in subject_ids:
        file_path = os.path.join(base_path, subject_id, f'{subject_id}_rfMRI_REST1_{scan_type}_100')
        file_paths.append(file_path)
    return file_paths

# Base path for FC dataset
base_path = 'D:/Research AU/Python/connectomes_100/'

# Generate file paths for LR and RL scans
lr_paths = generate_file_paths(base_path, 'LR', num_subjects=428)
rl_paths = generate_file_paths(base_path, 'RL', num_subjects=428)

# Load connectivity matrices
connectivity_matrices_lr = [load_connectivity_matrix(path) for path in lr_paths if os.path.exists(path)]
connectivity_matrices_rl = [load_connectivity_matrix(path) for path in rl_paths if os.path.exists(path)]

# Filter out None values
connectivity_matrices_lr = [mat for mat in connectivity_matrices_lr if mat is not None]
connectivity_matrices_rl = [mat for mat in connectivity_matrices_rl if mat is not None]

# Load Yeo ROIs labels
yeo_rois_path = 'C:/Users/ksrru/Documents/updated_LUT/updated_LUT/YeoROIs_N100.mat'
yeo_data = loadmat(yeo_rois_path)
yeo_labels = yeo_data['YeoROIs'].flatten()  # Assuming 'YeoROIs' contains the labels for ROIs

# Given ID rates for 7 networks
id_rate = {1: 0.3388, 2: 0.2734, 3: 0.2886, 4: 0.2862, 5: 0.0210, 6: 0.3516, 7: 0.4416}

# Initialize a dictionary to store metrics aggregated across all subjects
network_metrics = {network: {"modularity": [], "betweenness": [], "eigenvector": [], "clustering": []} for network in id_rate.keys()}

# Process all FC matrices and compute metrics
for fc_matrices in [connectivity_matrices_lr, connectivity_matrices_rl]:
    for idx, fc_matrix in enumerate(fc_matrices):
        if fc_matrix is not None:
            np.fill_diagonal(fc_matrix, 0)  # Remove self-connections
            threshold = 0.5  # Threshold for adjacency matrix creation

            # Threshold the FC matrix to create a binary adjacency matrix
            adj_matrix = (fc_matrix > threshold).astype(int)

            # Convert adjacency matrix to a graph
            G = nx.from_numpy_array(adj_matrix)

            # Compute node-level metrics
            node_modularity = list(nx.community.greedy_modularity_communities(G))  # Modularity
            node_betweenness = nx.betweenness_centrality(G)  # Betweenness centrality
            eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6)  # Eigenvector centrality
            node_clustering = nx.clustering(G)  # Clustering coefficient

            # Aggregate metrics for each Yeo network
            for network in np.unique(yeo_labels):
                nodes_in_network = np.where(yeo_labels == network)[0]
                avg_betweenness = np.mean([node_betweenness.get(node, 0) for node in nodes_in_network])
                avg_eigenvector = np.mean([eigenvector_centrality.get(node, 0) for node in nodes_in_network])
                avg_clustering = np.mean([node_clustering.get(node, 0) for node in nodes_in_network])

                # Append metrics for aggregation later
                network_metrics[network]["modularity"].append(len(node_modularity))  # Modularity is constant for all nodes
                network_metrics[network]["betweenness"].append(avg_betweenness)
                network_metrics[network]["eigenvector"].append(avg_eigenvector)
                network_metrics[network]["clustering"].append(avg_clustering)

# Compute mean and standard deviation for each network
aggregated_metrics = []
for network, metrics in network_metrics.items():
    aggregated_metrics.append({
        "network_id": network,
        "modularity_mean": np.mean(metrics["modularity"]),
        "modularity_std": np.std(metrics["modularity"]),
        "betweenness_mean": np.mean(metrics["betweenness"]),
        "betweenness_std": np.std(metrics["betweenness"]),
        "eigenvector_mean": np.mean(metrics["eigenvector"]),
        "eigenvector_std": np.std(metrics["eigenvector"]),
        "clustering_mean": np.mean(metrics["clustering"]),
        "clustering_std": np.std(metrics["clustering"]),
        "id_rate": id_rate[network],
    })

# Create a DataFrame for aggregated metrics
df_aggregated_metrics = pd.DataFrame(aggregated_metrics)

# Correlation analysis
correlation_matrix = df_aggregated_metrics.corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Visualize pairwise relationships
sns.pairplot(df_aggregated_metrics, vars=[
    "betweenness_mean", "eigenvector_mean", "clustering_mean", "id_rate"
])
plt.show()

# Regression analysis
X = df_aggregated_metrics[["betweenness_mean", "eigenvector_mean", "clustering_mean"]]
y = df_aggregated_metrics["id_rate"]
X = sm.add_constant(X)  # Add intercept
model = sm.OLS(y, X).fit()
print(model.summary())

# Save the aggregated metrics to CSV
df_aggregated_metrics.to_csv("D:/Research AU/Python/aggregated_metrics.csv", index=False)

###
import os
import numpy as np
import networkx as nx
import pandas as pd
from scipy.io import loadmat

# Function to load connectivity matrix from a file
def load_connectivity_matrix(file_path):
    try:
        return np.loadtxt(file_path, delimiter=' ')
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Function to generate file paths for a specific scan type and number of subjects
def generate_file_paths(base_path, scan_type, num_subjects=428):
    file_paths = []
    subject_ids = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    subject_ids.sort()
    subject_ids = subject_ids[:num_subjects]
    for subject_id in subject_ids:
        file_path = os.path.join(base_path, subject_id, f'{subject_id}_rfMRI_REST1_{scan_type}_100')
        file_paths.append(file_path)
    return file_paths

# Base path for FC dataset
base_path = 'D:/Research AU/Python/connectomes_100/'

# Generate file paths for LR and RL scans
lr_paths = generate_file_paths(base_path, 'LR', num_subjects=428)
rl_paths = generate_file_paths(base_path, 'RL', num_subjects=428)

# Load connectivity matrices
connectivity_matrices_lr = [load_connectivity_matrix(path) for path in lr_paths if os.path.exists(path)]
connectivity_matrices_rl = [load_connectivity_matrix(path) for path in rl_paths if os.path.exists(path)]

# Filter out None values
connectivity_matrices_lr = [mat for mat in connectivity_matrices_lr if mat is not None]
connectivity_matrices_rl = [mat for mat in connectivity_matrices_rl if mat is not None]

# Load Yeo ROIs labels
yeo_rois_path = 'C:/Users/ksrru/Documents/updated_LUT/updated_LUT/YeoROIs_N100.mat'
yeo_data = loadmat(yeo_rois_path)
yeo_labels = yeo_data['YeoROIs'].flatten()  # Assuming 'YeoROIs' contains the labels for ROIs

# Initialize dictionary to store aggregated metrics for each subject and scan type
all_subject_metrics = []

# Process all FC matrices for LR and RL
for fc_matrices, scan_type in zip([connectivity_matrices_lr, connectivity_matrices_rl], ["LR", "RL"]):
    for subject_id, fc_matrix in enumerate(fc_matrices):
        if fc_matrix is not None:
            np.fill_diagonal(fc_matrix, 0)  # Remove self-connections
            threshold = 0.5  # Threshold for adjacency matrix creation

            # Threshold the FC matrix to create a binary adjacency matrix
            adj_matrix = (fc_matrix > threshold).astype(int)

            # Convert adjacency matrix to a graph
            G = nx.from_numpy_array(adj_matrix)

            # Compute node-level metrics
            node_betweenness = nx.betweenness_centrality(G)  # Betweenness centrality
            eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6)  # Eigenvector centrality
            node_clustering = nx.clustering(G)  # Clustering coefficient

            # Aggregate metrics for each Yeo network
            for network in np.unique(yeo_labels):
                nodes_in_network = np.where(yeo_labels == network)[0]

                # Calculate averages of node-level metrics for this network
                avg_betweenness = np.mean([node_betweenness.get(node, 0) for node in nodes_in_network])
                avg_eigenvector = np.mean([eigenvector_centrality.get(node, 0) for node in nodes_in_network])
                avg_clustering = np.mean([node_clustering.get(node, 0) for node in nodes_in_network])

                # Append metrics for this subject, scan type, and Yeo network
                all_subject_metrics.append({
                    "scan_type": scan_type,
                    "subject_id": subject_id,
                    "network_id": network,
                    "avg_betweenness": avg_betweenness,
                    "avg_eigenvector": avg_eigenvector,
                    "avg_clustering": avg_clustering
                })

# Create a DataFrame for all subject metrics
df_subject_metrics = pd.DataFrame(all_subject_metrics)

# Aggregate the averages across all subjects for each network
df_aggregated_metrics = df_subject_metrics.groupby("network_id").agg({
    "avg_betweenness": ["mean", "std"],
    "avg_eigenvector": ["mean", "std"],
    "avg_clustering": ["mean", "std"]
}).reset_index()

# Flatten the MultiIndex columns for better readability
df_aggregated_metrics.columns = [
    "network_id",
    "betweenness_mean", "betweenness_std",
    "eigenvector_mean", "eigenvector_std",
    "clustering_mean", "clustering_std"
]

# Save results to CSV for further inspection
df_subject_metrics.to_csv("D:/Research AU/Python/subject_metrics.csv", index=False)
df_aggregated_metrics.to_csv("D:/Research AU/Python/aggregated_metrics.csv", index=False)

# Display results
print("Aggregated Metrics Across All Subjects:")
print(df_aggregated_metrics)


###
import os
import numpy as np
import networkx as nx
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Function to load connectivity matrix from a file
def load_connectivity_matrix(file_path):
    try:
        return np.loadtxt(file_path, delimiter=' ')
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Function to generate file paths for a specific scan type and number of subjects
def generate_file_paths(base_path, scan_type, num_subjects=428):
    file_paths = []
    subject_ids = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    subject_ids.sort()
    subject_ids = subject_ids[:num_subjects]
    for subject_id in subject_ids:
        file_path = os.path.join(base_path, subject_id, f'{subject_id}_rfMRI_REST1_{scan_type}_100')
        file_paths.append(file_path)
    return file_paths

# Base path for FC dataset
base_path = 'D:/Research AU/Python/connectomes_100/'

# Generate file paths for LR and RL scans
lr_paths = generate_file_paths(base_path, 'LR', num_subjects=428)
rl_paths = generate_file_paths(base_path, 'RL', num_subjects=428)

# Load connectivity matrices
connectivity_matrices_lr = [load_connectivity_matrix(path) for path in lr_paths if os.path.exists(path)]
connectivity_matrices_rl = [load_connectivity_matrix(path) for path in rl_paths if os.path.exists(path)]

# Filter out None values
connectivity_matrices_lr = [mat for mat in connectivity_matrices_lr if mat is not None]
connectivity_matrices_rl = [mat for mat in connectivity_matrices_rl if mat is not None]

# Load Yeo ROIs labels
yeo_rois_path = 'C:/Users/ksrru/Documents/updated_LUT/updated_LUT/YeoROIs_N100.mat'
yeo_data = loadmat(yeo_rois_path)
yeo_labels = yeo_data['YeoROIs'].flatten()  # Assuming 'YeoROIs' contains the labels for ROIs

# Initialize dictionary to store aggregated metrics for each subject and scan type
all_subject_metrics = []

# Process all FC matrices for LR and RL
for fc_matrices, scan_type in zip([connectivity_matrices_lr, connectivity_matrices_rl], ["LR", "RL"]):
    for subject_id, fc_matrix in enumerate(fc_matrices):
        if fc_matrix is not None:
            np.fill_diagonal(fc_matrix, 0)  # Remove self-connections
            threshold = 0.5  # Threshold for adjacency matrix creation

            # Threshold the FC matrix to create a binary adjacency matrix
            adj_matrix = (fc_matrix > threshold).astype(int)

            # Convert adjacency matrix to a graph
            G = nx.from_numpy_array(adj_matrix)

            # Compute node-level metrics
            node_betweenness = nx.betweenness_centrality(G)  # Betweenness centrality
            eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6)  # Eigenvector centrality
            node_clustering = nx.clustering(G)  # Clustering coefficient

            # Aggregate metrics for each Yeo network
            for network in np.unique(yeo_labels):
                nodes_in_network = np.where(yeo_labels == network)[0]

                # Calculate averages of node-level metrics for this network
                avg_betweenness = np.mean([node_betweenness.get(node, 0) for node in nodes_in_network])
                avg_eigenvector = np.mean([eigenvector_centrality.get(node, 0) for node in nodes_in_network])
                avg_clustering = np.mean([node_clustering.get(node, 0) for node in nodes_in_network])

                # Append metrics for this subject, scan type, and Yeo network
                all_subject_metrics.append({
                    "scan_type": scan_type,
                    "subject_id": subject_id,
                    "network_id": network,
                    "avg_betweenness": avg_betweenness,
                    "avg_eigenvector": avg_eigenvector,
                    "avg_clustering": avg_clustering
                })

# Create a DataFrame for all subject metrics
df_subject_metrics = pd.DataFrame(all_subject_metrics)

# Aggregate the averages across all subjects for each network
df_aggregated_metrics = df_subject_metrics.groupby("network_id").agg({
    "avg_betweenness": ["mean", "std"],
    "avg_eigenvector": ["mean", "std"],
    "avg_clustering": ["mean", "std"]
}).reset_index()

# Flatten the MultiIndex columns for better readability
df_aggregated_metrics.columns = [
    "network_id",
    "betweenness_mean", "betweenness_std",
    "eigenvector_mean", "eigenvector_std",
    "clustering_mean", "clustering_std"
]

# Save results to CSV for further inspection
df_subject_metrics.to_csv("D:/Research AU/Python/subject_metrics.csv", index=False)
df_aggregated_metrics.to_csv("D:/Research AU/Python/aggregated_metrics.csv", index=False)

# Display results
print("Aggregated Metrics Across All Subjects:")
print(df_aggregated_metrics)

# Add the ID rate to the aggregated DataFrame
id_rate = {1: 0.3388, 2: 0.2734, 3: 0.2886, 4: 0.2862, 5: 0.0210, 6: 0.3516, 7: 0.4416}
df_aggregated_metrics["id_rate"] = df_aggregated_metrics["network_id"].map(id_rate)

# Pairplot: Relationships between metrics and ID rate
sns.pairplot(df_aggregated_metrics, vars=[
    "betweenness_mean", "eigenvector_mean", "clustering_mean", "id_rate"
], diag_kind="kde")
plt.suptitle("Pairplot of Metrics and ID Rate", y=1.02)
plt.savefig("pairplot_metrics_id_rate.png")
plt.show()

# Calculate the correlation matrix
correlation_matrix = df_aggregated_metrics.corr()

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Heatmap of Metrics and ID Rate")
plt.savefig("correlation_heatmap.png")
plt.show()

for metric in ["betweenness", "eigenvector", "clustering"]:
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        df_aggregated_metrics["network_id"],
        df_aggregated_metrics[f"{metric}_mean"],
        yerr=df_aggregated_metrics[f"{metric}_std"],
        fmt="o-", capsize=5, label=f"{metric.capitalize()} Mean ± Std"
    )
    plt.title(f"{metric.capitalize()} Across Networks")
    plt.xlabel("Network ID")
    plt.ylabel(metric.capitalize())
    plt.grid(True)
    plt.legend()
    plt.savefig(f"lineplot_{metric}_errorbars.png")
    plt.show()

for metric in ["betweenness_mean", "eigenvector_mean", "clustering_mean"]:
    sns.jointplot(
        x=metric, y="id_rate", data=df_aggregated_metrics, kind="reg", height=7
    )
    plt.suptitle(f"{metric.replace('_mean', '').capitalize()} vs ID Rate", y=1.02)
    plt.savefig(f"jointplot_{metric}_id_rate.png")
    plt.show()

from math import pi

metrics = ["betweenness_mean", "eigenvector_mean", "clustering_mean"]
categories = [f"Network {i}" for i in df_aggregated_metrics["network_id"]]
values = df_aggregated_metrics[metrics].values.T

# Radar chart setup
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
for i, (network, value) in enumerate(zip(categories, values.T)):
    angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
    angles += angles[:1]
    ax.plot(angles, np.append(value, value[0]), label=network)
    ax.fill(angles, np.append(value, value[0]), alpha=0.1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics)
plt.title("Radar Chart of Metrics by Network")
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
plt.savefig("radar_chart_metrics.png")
plt.show()

# Scatterplots for individual metrics vs ID rate
for metric in ["betweenness_mean", "eigenvector_mean", "clustering_mean"]:
    plt.figure(figsize=(8, 5))
    plt.scatter(df_aggregated_metrics[metric], df_aggregated_metrics["id_rate"], s=100, alpha=0.7)
    plt.title(f"{metric.replace('_mean', '').capitalize()} vs ID Rate")
    plt.xlabel(metric.replace('_mean', '').capitalize())
    plt.ylabel("ID Rate")
    plt.grid(True)
    plt.savefig(f"scatter_{metric}_id_rate.png")
    plt.show()

# Visualization: Aggregation and Metrics Analysis

# Pairplot of metrics and ID rate (example visualization)
sns.pairplot(df_aggregated_metrics, vars=[
    "betweenness_mean", "eigenvector_mean", "clustering_mean"
], diag_kind="kde")
plt.suptitle("Pairplot of Metrics Across Networks", y=1.02)
plt.savefig("pairplot_metrics_across_networks.png")
plt.show()

# Bar plots for mean and standard deviation of metrics across networks
for metric in ["betweenness", "eigenvector", "clustering"]:
    plt.figure(figsize=(10, 6))
    plt.bar(
        df_aggregated_metrics["network_id"],
        df_aggregated_metrics[f"{metric}_mean"],
        yerr=df_aggregated_metrics[f"{metric}_std"],
        capsize=5,
        label=f"{metric.capitalize()} (Mean ± Std)"
    )
    plt.xlabel("Network ID")
    plt.ylabel(f"{metric.capitalize()} Mean ± Std")
    plt.title(f"{metric.capitalize()} Across Networks")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{metric}_metrics_across_networks.png")
    plt.show()


# Regression analysis
X = df_aggregated_metrics[["betweenness_mean", "eigenvector_mean", "clustering_mean"]]
y = df_aggregated_metrics["id_rate"]
X = sm.add_constant(X)  # Add intercept
model = sm.OLS(y, X).fit()
print(model.summary())


###
import os
import numpy as np
import networkx as nx
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io import loadmat
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# Function to load connectivity matrix from a file
def load_connectivity_matrix(file_path):
    try:
        return np.loadtxt(file_path, delimiter=' ')
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Function to generate file paths for a specific scan type and number of subjects
def generate_file_paths(base_path, scan_type, num_subjects=428):
    file_paths = []
    subject_ids = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    subject_ids.sort()
    subject_ids = subject_ids[:num_subjects]
    for subject_id in subject_ids:
        file_path = os.path.join(base_path, subject_id, f'{subject_id}_rfMRI_REST1_{scan_type}_200')
        file_paths.append(file_path)
    return file_paths


# Base path for FC dataset
#base_path = 'D:/Research AU/Python/connectomes_100/'
base_path = 'D:/Research AU/connectomes_200/'

# Generate file paths for LR and RL scans
lr_paths = generate_file_paths(base_path, 'LR', num_subjects=428)
rl_paths = generate_file_paths(base_path, 'RL', num_subjects=428)

# Load connectivity matrices
connectivity_matrices_lr = [load_connectivity_matrix(path) for path in lr_paths if os.path.exists(path)]
connectivity_matrices_rl = [load_connectivity_matrix(path) for path in rl_paths if os.path.exists(path)]

# Filter out None values
connectivity_matrices_lr = [mat for mat in connectivity_matrices_lr if mat is not None]
connectivity_matrices_rl = [mat for mat in connectivity_matrices_rl if mat is not None]

# Load Yeo ROIs labels
yeo_rois_path = 'C:/Users/ksrru/Documents/updated_LUT/updated_LUT/YeoROIs_N200.mat'
yeo_data = loadmat(yeo_rois_path)
yeo_labels = yeo_data['YeoROIs'].flatten()  # Assuming 'YeoROIs' contains the labels for ROIs

# Initialize dictionary to store aggregated metrics for each subject and scan type
all_subject_metrics = []

# Process all FC matrices for LR and RL
for fc_matrices, scan_type in zip([connectivity_matrices_lr, connectivity_matrices_rl], ["LR", "RL"]):
    for subject_id, fc_matrix in enumerate(fc_matrices):
        if fc_matrix is not None:
            np.fill_diagonal(fc_matrix, 0)  # Remove self-connections
            threshold = 0.5  # Threshold for adjacency matrix creation

            # Threshold the FC matrix to create a binary adjacency matrix
            adj_matrix = (fc_matrix > threshold).astype(int)

            # Convert adjacency matrix to a graph
            G = nx.from_numpy_array(adj_matrix)

            # Compute node-level metrics
            node_betweenness = nx.betweenness_centrality(G)  # Betweenness centrality
            eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6)  # Eigenvector centrality
            node_clustering = nx.clustering(G)  # Clustering coefficient

            # Aggregate metrics for each Yeo network
            for network in np.unique(yeo_labels):
                nodes_in_network = np.where(yeo_labels == network)[0]

                # Calculate averages of node-level metrics for this network
                avg_betweenness = np.mean([node_betweenness.get(node, 0) for node in nodes_in_network])
                avg_eigenvector = np.mean([eigenvector_centrality.get(node, 0) for node in nodes_in_network])
                avg_clustering = np.mean([node_clustering.get(node, 0) for node in nodes_in_network])

                # Append metrics for this subject, scan type, and Yeo network
                all_subject_metrics.append({
                    "scan_type": scan_type,
                    "subject_id": subject_id,
                    "network_id": network,
                    "avg_betweenness": avg_betweenness,
                    "avg_eigenvector": avg_eigenvector,
                    "avg_clustering": avg_clustering
                })

# Create a DataFrame for all subject metrics
df_subject_metrics = pd.DataFrame(all_subject_metrics)

# Aggregate the averages across all subjects for each network
df_aggregated_metrics = df_subject_metrics.groupby("network_id").agg({
    "avg_betweenness": ["mean", "std"],
    "avg_eigenvector": ["mean", "std"],
    "avg_clustering": ["mean", "std"]
}).reset_index()

# Flatten the MultiIndex columns for better readability
df_aggregated_metrics.columns = [
    "network_id",
    "betweenness_mean", "betweenness_std",
    "eigenvector_mean", "eigenvector_std",
    "clustering_mean", "clustering_std"
]

# Add the ID rate to the aggregated DataFrame


#id_rate = {1: 0.3388, 2: 0.2734, 3: 0.2886, 4: 0.2862, 5: 0.0210, 6: 0.3516, 7: 0.4416}
id_rate = {1: .5070, 2: 0.4813, 3: 0.7465, 4: 0.3189, 5: 0.0736, 6: 0.8703, 7: 0.8318}
#id_rate = {1: 0.6402, 2: 0.6051, 3: 0.8972, 4: 0.5023, 5: 0.1928, 6: 0.9720, 7: 0.9614}
#id_rate = {1: 0.6846, 2: 0.6717, 3: 0.9603, 4: 0.6180, 5: 0.2185, 6: 0.9895, 7: 0.9883}
df_aggregated_metrics["id_rate"] = df_aggregated_metrics["network_id"].map(id_rate)

# Pairplot: Relationships between metrics and ID rate
sns.pairplot(
    df_aggregated_metrics, 
    vars=["betweenness_mean", "eigenvector_mean", "clustering_mean", "id_rate"],
    diag_kind="kde",
    kind="reg"
)
plt.suptitle("Pairplot of Metrics and ID Rate", y=1.02)
plt.savefig("pairplot_metrics_id_rate.png")
plt.show()

### Scatter Plots with Regression Lines
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


metrics = ["betweenness_mean", "eigenvector_mean", "clustering_mean"]

plt.figure(figsize=(18, 5))
for i, metric in enumerate(metrics):
    plt.subplot(1, len(metrics), i + 1)
    sns.regplot(
        x=df_aggregated_metrics[metric],
        y=df_aggregated_metrics["id_rate"],
        ci=95,
        line_kws={"color": "red"},
    )
    plt.xlabel(metric.replace("_mean", "").capitalize())
    plt.ylabel("ID Rate")
    plt.title(f"{metric.replace('_mean', '').capitalize()} vs ID Rate")

plt.tight_layout()
plt.savefig("scatter_plots_with_regression_lines.png")
plt.show()

# Combined Bar Plot for Metrics and ID Rate
plt.figure(figsize=(12, 6))
bar_width = 0.2
network_ids = df_aggregated_metrics["network_id"]

# Plot each metric as a bar
for i, metric in enumerate(metrics):
    plt.bar(
        network_ids + i * bar_width,
        df_aggregated_metrics[metric],
        width=bar_width,
        label=metric.replace("_mean", "").capitalize(),
        alpha=0.8,
    )

# Plot ID rate as a bar
plt.bar(
    network_ids + len(metrics) * bar_width,
    df_aggregated_metrics["id_rate"],
    width=bar_width,
    label="ID Rate",
    color="black",
    alpha=0.7,
)

# Adjust plot details
plt.xticks(network_ids + bar_width, network_ids)
plt.xlabel("Network ID")
plt.ylabel("Value")
plt.title("Comparison of Metrics and ID Rate Across Yeo Networks")
plt.legend()
plt.tight_layout()
plt.savefig("bar_plot_metrics_vs_id_rate.png")
plt.show()
### Important one
# Enhanced Scatter Plots with Color-Coded Networks
plt.figure(figsize=(18, 5))
for i, metric in enumerate(metrics):
    plt.subplot(1, len(metrics), i + 1)
    sns.scatterplot(
        x=df_aggregated_metrics[metric],
        y=df_aggregated_metrics["id_rate"],
        hue=df_aggregated_metrics["network_id"],
        palette="Set2",
        s=100,
    )
    sns.regplot(
        x=df_aggregated_metrics[metric],
        y=df_aggregated_metrics["id_rate"],
        scatter=False,
        ci=None,
        line_kws={"color": "black"},
    )
    plt.xlabel(metric.replace("_mean", "").capitalize())
    plt.ylabel("ID Rate")
    plt.title(f"{metric.replace('_mean', '').capitalize()} vs ID Rate")
    plt.legend(title="Network ID", bbox_to_anchor=(1.05, 1), loc="upper left")

plt.tight_layout()
plt.savefig("scatter_colored_networks.png")
plt.show()





# Regression analysis
X = df_aggregated_metrics[["betweenness_mean", "eigenvector_mean", "clustering_mean"]]
y = df_aggregated_metrics["id_rate"]
X = sm.add_constant(X)  # Add intercept

# VIF Analysis
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("Variance Inflation Factor (VIF):")
print(vif_data)

# Run OLS Regression
# Filter rows with valid `id_rate`
df_filtered = df_aggregated_metrics[df_aggregated_metrics["id_rate"].notna()]

# Independent variables and dependent variable
X = df_filtered[["betweenness_mean", "eigenvector_mean", "clustering_mean"]]
y = df_filtered["id_rate"]

# Check for constant columns or insufficient variation
print("Variation in X:\n", X.var())
print("Variation in y:\n", y.var())

# Standardize predictors
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_scaled = sm.add_constant(X_scaled)

# Run regression
model = sm.OLS(y, X_scaled).fit()
print(model.summary())



### Final one 
###
import os
import numpy as np
import networkx as nx
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io import loadmat
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# Function to load connectivity matrix from a file
def load_connectivity_matrix(file_path):
    try:
        return np.loadtxt(file_path, delimiter=' ')
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Function to generate file paths for a specific scan type and number of subjects
def generate_file_paths(base_path, scan_type, num_subjects=428):
    file_paths = []
    subject_ids = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    subject_ids.sort()
    subject_ids = subject_ids[:num_subjects]
    for subject_id in subject_ids:
        file_path = os.path.join(base_path, subject_id, f'{subject_id}_tfMRI_GAMBLING_{scan_type}_100')
        file_paths.append(file_path)
    return file_paths


# Base path for FC dataset
base_path = 'D:/Research AU/Python/connectomes_100/'
#base_path = 'D:/Research AU/connectomes_200/'

# Generate file paths for LR and RL scans
lr_paths = generate_file_paths(base_path, 'LR', num_subjects=428)
rl_paths = generate_file_paths(base_path, 'RL', num_subjects=428)

# Load connectivity matrices
connectivity_matrices_lr = [load_connectivity_matrix(path) for path in lr_paths if os.path.exists(path)]
connectivity_matrices_rl = [load_connectivity_matrix(path) for path in rl_paths if os.path.exists(path)]

# Filter out None values
connectivity_matrices_lr = [mat for mat in connectivity_matrices_lr if mat is not None]
connectivity_matrices_rl = [mat for mat in connectivity_matrices_rl if mat is not None]

# Load Yeo ROIs labels
yeo_rois_path = 'C:/Users/ksrru/Documents/updated_LUT/updated_LUT/YeoROIs_N100.mat'
yeo_data = loadmat(yeo_rois_path)
yeo_labels = yeo_data['YeoROIs'].flatten()  # Assuming 'YeoROIs' contains the labels for ROIs

# Initialize dictionary to store aggregated metrics for each subject and scan type
all_subject_metrics = []

# Process all FC matrices for LR and RL
for fc_matrices, scan_type in zip([connectivity_matrices_lr, connectivity_matrices_rl], ["LR", "RL"]):
    for subject_id, fc_matrix in enumerate(fc_matrices):
        if fc_matrix is not None:
            np.fill_diagonal(fc_matrix, 0)  # Remove self-connections
            threshold = 0.5  # Threshold for adjacency matrix creation

            # Threshold the FC matrix to create a binary adjacency matrix
            adj_matrix = (fc_matrix > threshold).astype(int)

            # Convert adjacency matrix to a graph
            G = nx.from_numpy_array(adj_matrix)

            # Compute node-level metrics
            node_betweenness = nx.betweenness_centrality(G)  # Betweenness centrality
            eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6)  # Eigenvector centrality
            node_clustering = nx.clustering(G)  # Clustering coefficient

            # Aggregate metrics for each Yeo network
            for network in np.unique(yeo_labels):
                nodes_in_network = np.where(yeo_labels == network)[0]

                # Calculate averages of node-level metrics for this network
                avg_betweenness = np.mean([node_betweenness.get(node, 0) for node in nodes_in_network])
                avg_eigenvector = np.mean([eigenvector_centrality.get(node, 0) for node in nodes_in_network])
                avg_clustering = np.mean([node_clustering.get(node, 0) for node in nodes_in_network])

                # Append metrics for this subject, scan type, and Yeo network
                all_subject_metrics.append({
                    "scan_type": scan_type,
                    "subject_id": subject_id,
                    "network_id": network,
                    "avg_betweenness": avg_betweenness,
                    "avg_eigenvector": avg_eigenvector,
                    "avg_clustering": avg_clustering
                })

# Create a DataFrame for all subject metrics
df_subject_metrics = pd.DataFrame(all_subject_metrics)

# Aggregate the averages across all subjects for each network
df_aggregated_metrics = df_subject_metrics.groupby("network_id").agg({
    "avg_betweenness": ["mean", "std"],
    "avg_eigenvector": ["mean", "std"],
    "avg_clustering": ["mean", "std"]
}).reset_index()

# Flatten the MultiIndex columns for better readability
df_aggregated_metrics.columns = [
    "network_id",
    "betweenness_mean", "betweenness_std",
    "eigenvector_mean", "eigenvector_std",
    "clustering_mean", "clustering_std"
]

# Function to read ID rates from the file
def load_task_id_rates(file_path):
    task_id_rates = {}
    with open(file_path, 'r') as file:
        current_task = None
        for line in file:
            line = line.strip()
            if line.endswith(":"):
                current_task = line[:-1]  # Extract the task name
                task_id_rates[current_task] = {}
            elif "Region" in line and current_task:
                region, id_rate = line.split(", ID Rate:")
                region_id = int(region.split(":")[1].strip())
                task_id_rates[current_task][region_id] = float(id_rate.strip())
    return task_id_rates

# Path to the ID rate file
id_rate_file_path = "D:/Research AU/Multi scale analysis of ID rate/ID_rate_100_Yeo's.txt"

# Load the ID rates
task_id_rates = load_task_id_rates(id_rate_file_path)
print("Loaded Task ID Rates:", task_id_rates)

# Example: Accessing ID rates for a specific task and region
rest_id_rates = task_id_rates.get("Gambling", {})
print("REST Task ID Rates:", rest_id_rates)
# Function to analyze and plot metrics with ID rates for a given task
def analyze_and_plot_with_enhanced_visualizations(df_aggregated_metrics, task_id_rates, task_name):
    if task_name not in task_id_rates:
        print(f"Task {task_name} not found in ID rates.")
        return

    # Map ID rates for the task to the aggregated metrics
    df_aggregated_metrics["id_rate"] = df_aggregated_metrics["network_id"].map(task_id_rates[task_name])
    
    # Pairplot: Relationships between metrics and ID rate
    sns.pairplot(
        df_aggregated_metrics, 
        vars=["betweenness_mean", "eigenvector_mean", "clustering_mean", "id_rate"],
        diag_kind="kde",
        kind="reg"
    )
    plt.suptitle(f"Pairplot of Metrics and ID Rate for {task_name}", y=1.02)
    plt.savefig(f"pairplot_{task_name.lower()}_metrics_id_rate.png")
    plt.show()

    # Scatter Plots with Regression Lines
    metrics = ["betweenness_mean", "eigenvector_mean", "clustering_mean"]
    plt.figure(figsize=(18, 5))
    for i, metric in enumerate(metrics):
        plt.subplot(1, len(metrics), i + 1)
        sns.regplot(
            x=df_aggregated_metrics[metric],
            y=df_aggregated_metrics["id_rate"],
            ci=95,
            line_kws={"color": "red"}
        )
        plt.xlabel(metric.replace("_mean", "").capitalize())
        plt.ylabel("ID Rate")
        plt.title(f"{metric.replace('_mean', '').capitalize()} vs ID Rate")
    plt.tight_layout()
    plt.savefig(f"scatter_plots_{task_name.lower()}_with_regression.png")
    plt.show()

    # Enhanced Scatter Plots with Color-Coded Networks
    plt.figure(figsize=(18, 5))
    for i, metric in enumerate(metrics):
        plt.subplot(1, len(metrics), i + 1)
        sns.scatterplot(
            x=df_aggregated_metrics[metric],
            y=df_aggregated_metrics["id_rate"],
            hue=df_aggregated_metrics["network_id"],
            palette="Set2",
            s=100,
        )
        sns.regplot(
            x=df_aggregated_metrics[metric],
            y=df_aggregated_metrics["id_rate"],
            scatter=False,
            ci=None,
            line_kws={"color": "black"},
        )
        plt.xlabel(metric.replace("_mean", "").capitalize())
        plt.ylabel("ID Rate")
        plt.title(f"{metric.replace('_mean', '').capitalize()} vs ID Rate")
        plt.legend(title="Network ID", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(f"scatter_colored_networks_{task_name.lower()}.png")
    plt.show()

# Example: Analyze and plot for REST task
analyze_and_plot_with_enhanced_visualizations(df_aggregated_metrics, task_id_rates, "Gambling")







 
