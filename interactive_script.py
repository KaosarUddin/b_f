import numpy as np
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from spd_metrics_id.io import find_subject_paths, load_matrix
from spd_metrics_id.distance import (
    alpha_z_bw,
    alpha_procrustes,
    bures_wasserstein,
    geodesic_distance,
    log_euclidean_distance,
    pearson_distance,
    euclidean_distance,
)
from spd_metrics_id.id_rate import compute_id_rate

def verbose_print(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}")
# Start timing the whole process
start_time = time.time()
verbose_print("Starting connectome identification process...")

# Get user configuration
print("\n                                      ===== CONNECTOME ANALYSIS CONFIGURATION =====")
print("This script is designed to analyze connectome data, which involves examining the neural connectivity matrices that map the connections between different regions of the brain. \nBy applying various distance and divergence metrics, the script computes identification rates, which measure the accuracy of identifying between subjects based on their unique connectome profiles. \nThis process helps in understanding the effectiveness of different metrics in capturing the distinctiveness of individual brain connectivity patterns.")
print("To test this script, ensure you have the required connectome data files and run the script with different configurations to verify the accuracy of identification rates.")
print("To proceed, follow these steps: first, select the tasks you wish to analyze; next, choose the distance metrics to apply; then, specify any necessary tuning parameters; \nafter that, select the directory containing the connectome datasets; and finally, enter the number of subjects you want to include in the analysis.")

def multi_choice(prompt, options):
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        print(f"{i}. {opt}")
    choices = input("Enter choices (comma-separated numbers): ")
    idxs = [int(c.strip())-1 for c in choices.split(',')]
    return [options[i] for i in idxs]

# --- User Selections ---
# 1) Task selection
TASKS = ['REST', 'EMOTION', 'LANGUAGE', 'WM', 'MOTOR', 'RELATIONAL', 'GAMBLING', 'SOCIAL']
selected_tasks = multi_choice("Select tasks to process:", TASKS)

# 2) Distance metrics selection
METRIC_FUNCS = {
    'Alpha Z': alpha_z_bw,
    'Alpha Procrustes': alpha_procrustes,
    'Bures-Wasserstein': bures_wasserstein,
    'AI': geodesic_distance,
    'Log-Euclidean': log_euclidean_distance,
    'Pearson': pearson_distance,
    'Euclidean': euclidean_distance
}
metric_names = list(METRIC_FUNCS.keys())
selected_metrics = multi_choice("Select distance metrics:", metric_names)

# 3) Parameter prompts for metrics
# Prompt for tau for Geodesic/Log-Euclidean
tau_geo_log = None
if any(m in ['AI', 'Log-Euclidean'] for m in selected_metrics):
    tau_input = input("Enter tau values for AI/Log-Euclidean (comma-separated, e.g. 0.01,0.1): ")
    tau_geo_log = [float(t.strip()) for t in tau_input.split(',')]

# Prompt for tau and z for Alpha Z
tau_alpha_z = None
z_alpha_z = None
if 'Alpha Z' in selected_metrics:
    alpha_input = input("Enter alpha value for Alpha Z distance (comma-separated): ")
    alpha_alpha_z = [float(t.strip()) for t in alpha_input.split(',')]
    z_input = input("Enter z exponent values for Alpha Z (comma-separated): ")
    z_alpha_z = [float(z.strip()) for z in z_input.split(',')]

# Prompt for Alpha Procrustes
alpha_pro = None
if 'Alpha Procrustes' in selected_metrics:
    alpha_input = input("Enter alpha values for Alpha Procrustes distance (comma-separated): ")
    alpha_pro = [float(a.strip()) for a in alpha_input.split(',')]

# 4) Base directory and subjects
base_dir = input("Enter base directory for connectome files [connectomes_100/]: ") or "connectomes_100/"
num_subjects = int(input("Enter number of subjects to process (e.g. 30): ") or 30)

# --- Analysis Loop ---
start_time = time.time()
verbose_print("Starting interactive connectome analysis...")
results = []

for task in selected_tasks:
    verbose_print(f"Loading data for task: {task}")
    lr_paths = find_subject_paths(base_dir, task, 'LR', [100], n=num_subjects)
    rl_paths = find_subject_paths(base_dir, task, 'RL', [100], n=num_subjects)
    mats_lr = [load_matrix(p) for p in lr_paths]
    mats_rl = [load_matrix(p) for p in rl_paths]

    for metric in selected_metrics:
        fn = METRIC_FUNCS[metric]
        # Geodesic or Log-Euclidean with tau
        if metric in ['AI', 'Log-Euclidean']:
            for tau in tau_geo_log:
                verbose_print(f"Computing {metric} (tau={tau}) for {task}")
                D12 = np.array([[fn(A, B, tau) for B in mats_rl] for A in mats_lr])
                D21 = np.array([[fn(A, B, tau) for B in mats_lr] for A in mats_rl])
                id12 = compute_id_rate(D12)
                id21 = compute_id_rate(D21)
                accuracy=(id12 + id21) / 2
                results.append({'task': task, 'metric': metric, 'tau': tau, 'param': None, 'id12': id12, 'id21': id21,'accuracy':accuracy})
        # Alpha Z with tau and z
        elif metric == 'Alpha Z':
            for alpha in alpha_alpha_z:
                for z in z_alpha_z:
                    verbose_print(f"Computing Alpha Z (tau={alpha}, z={z}) for {task}")
                    D12 = np.array([[fn(A, B, alpha, z=z) for B in mats_rl] for A in mats_lr])
                    D21 = np.array([[fn(A, B, alpha, z=z) for B in mats_lr] for A in mats_rl])
                    id12 = compute_id_rate(D12)
                    id21 = compute_id_rate(D21)
                    accuracy=(id12 + id21) / 2
                    results.append({'task': task, 'metric': metric, 'alpha': alpha, 'param': z, 'id12': id12, 'id21': id21,'accuracy':accuracy})
        # Alpha Procrustes with alpha
        elif metric == 'Alpha Procrustes':
            for alpha in alpha_pro:
                verbose_print(f"Computing Alpha Procrustes (alpha={alpha}) for {task}")
                D12 = np.array([[fn(A, B, alpha) for B in mats_rl] for A in mats_lr])
                D21 = np.array([[fn(A, B, alpha) for B in mats_lr] for A in mats_rl])
                id12 = compute_id_rate(D12)
                id21 = compute_id_rate(D21)
                accuracy = (id12 + id21) / 2
                results.append({'task': task, 'metric': metric, 'tau': None, 'param': alpha, 'id12': id12, 'id21': id21,'accuracy':accuracy})
        # Metrics without extra params
        else:
            verbose_print(f"Computing {metric} for {task}")
            D12 = np.array([[fn(A, B) for B in mats_rl] for A in mats_lr])
            D21 = np.array([[fn(A, B) for B in mats_lr] for A in mats_rl])
            id12 = compute_id_rate(D12)
            id21 = compute_id_rate(D21)
            accuracy = (id12 + id21) / 2
            results.append({'task': task, 'metric': metric, 'tau': None, 'param': None, 'id12': id12, 'id21': id21,'accuracy':accuracy})

# Summarize results
df = pd.DataFrame(results)
print("\nIdentification Rates Summary:")
print(df.to_string(index=False))
verbose_print(f"Total runtime: {time.time() - start_time:.2f}s")
