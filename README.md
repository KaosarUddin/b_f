# SPD Metrics ID

![Build Status](https://img.shields.io/badge/build-passing-brightgreen) ![License: MIT](https://img.shields.io/badge/license-MIT-blue)

**SPD Metrics ID** is a Python package for computing identification rates (ID rates) between symmetric positive-definite (SPD) connectivity matrices using a wide variety of distance and divergence metrics.

It provides both an easy-to-use **command-line interface (CLI)** and a **Python API** for flexible, customizable analysis of brain connectomes across different tasks, scan directions, parcellation resolutions, and regularization settings.

## 📚 Table of Contents
- [Features](#-features)
- [Installation](#-installation)
- [Command-Line Usage](#-command-line-usage)
- [Python API Example](#-python-api-example)
- [Interactive Analysis Script](#-interactive-analysis-script)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

- **Alpha-Z Bures–Wasserstein divergence**
- **Alpha-Procrustes** (“ProE”) distance
- **Bures–Wasserstein** distance
- **Affine-invariant Riemannian** distance
- **Log-Euclidean** distance
- **Pearson correlation–based** distance
- **Euclidean** distance (flattened matrices)
- CLI with customizable tasks, scan directions, resolutions, and SPD regularization (`τ`)
- Python API for programmatic integration
- Interactive configuration script (full example below)
- Unit tests using **pytest**

---

## 📦 Installation

Install from [PyPI](https://pypi.org/project/spd-metrics-id/):

```bash
pip install spd-metrics-id

```

Or clone from GitHub for development:

```bash
git clone https://github.com/yourusername/spd-metrics-id.git
cd spd-metrics-id

# (Recommended) Create a virtual environment
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate

# Install in editable mode
pip install -e .
```

---

## 🖥 Command-Line Usage

After installation, the `spd-id` console script is available:

```bash
spd-id \
  --base-path PATH/TO/DATA \
  --tasks REST LANGUAGE EMOTION \
  --scan-types LR RL \
  --resolutions 100 200 \
  --metric alpha_z \
  --alpha 0.99 \
  --z 1.0 \
  --tau 0.00 \
  --num-subjects 30
```

### 🔑 Key Arguments

| Argument        | Description                                                                      |
|-----------------|----------------------------------------------------------------------------------|
| `--base-path`   | Path to root folder containing subject subfolders.                               |
| `--tasks`       | List of tasks (`REST`, `EMOTION`, `GAMBLING`, `LANGUAGE`, etc.).                 |
| `--scan-types`  | Two scan directions to compare (e.g., `LR RL`).                                  |
| `--resolutions` | Parcellation sizes (e.g., `100 200 300`).                                        |
| `--metric`      | Distance metric: `alpha_z`, `alpha_pro`, `bw`, `AI`, `log`, `pearson`, `euclid`. |
| `--alpha`, `--z`| Parameters for Alpha-based metrics.                                              |
| `--tau`         | SPD regularization (default: `1e-6`).                                            |
| `--num-subjects`| Maximum number of subjects to include.                                           |

---

## 🚀 Python API Example

```python
import numpy as np
from spd_metrics_id.io import find_subject_paths, load_matrix
from spd_metrics_id.distance import alpha_z_bw
from spd_metrics_id.id_rate import compute_id_rate

# Base directory
base = "connectomes_100/"

# Find subject paths
lr_paths = find_subject_paths(base, "REST", "LR", [100], n=30)
rl_paths = find_subject_paths(base, "REST", "RL", [100], n=30)

# Load matrices
mats_lr = [load_matrix(p) for p in lr_paths]
mats_rl = [load_matrix(p) for p in rl_paths]

# Compute distance matrices
D12 = np.array([[alpha_z_bw(A, B, alpha=0.99, z=1.0) for B in mats_rl] for A in mats_lr])
D21 = np.array([[alpha_z_bw(A, B, alpha=0.99, z=1.0) for B in mats_lr] for A in mats_rl])

# Compute ID rates
id1 = compute_id_rate(D12)
id2 = compute_id_rate(D21)
print("Average ID rate:", (id1 + id2) / 2)
```


---
## 🎛 Interactive Analysis Script Example

### 🧠 Connectome Analysis Configuration

This interactive script is designed to **analyze connectome data**, which involves examining the neural connectivity matrices that map the connections between different regions of the brain.  
By applying various distance and divergence metrics, the script computes **identification rates**, which measure the accuracy of identifying or distinguishing between subjects based on their unique connectome profiles.
This process helps in understanding the **effectiveness** of different metrics in capturing the distinctiveness of individual brain connectivity patterns.

🔑 Key Steps
1.Task Selection: Choose the tasks you wish to analyze (e.g., REST, EMOTION, LANGUAGE, etc.).
2.Metric Selection: Select the distance or divergence metrics to apply (e.g., Alpha Z, Alpha Procrustes, Bures-Wasserstein, etc.).
3.Parameter Specification: Enter any necessary tuning parameters such as τ, α, and z for the selected metrics.
4.Base Directory: Specify the directory containing the connectome datasets.
5.Subject Count: Enter the number of subjects to include in the analysis.

Ensure you have the required connectome data files prepared. Running the script across different configurations allows you to verify the robustness and accuracy of computed identification rates.
---

<details>
<summary>▶️ Click to expand full interactive script</summary>

```python
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
```

</details>

---

## 🧪 Testing

Run the full test suite with `pytest`:

```bash
python -m pytest
```

✅ All distance functions and ID-rate calculations are covered by unit tests.

---

## 🤝 Contributing

We welcome contributions!

1. Fork the repository.
2. Create a new feature branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Write your code and add corresponding unit tests.
4. Run `pytest` to ensure everything passes:
   ```bash
   python -m pytest
   ```
5. Submit a pull request.

Please follow [PEP 8](https://pep8.org/) coding standards.

---

## 📜 License

Distributed under the **MIT License**.  
See the [LICENSE](LICENSE) file for complete details.

---