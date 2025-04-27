# SPD Metrics ID

**SPD Metrics ID** is a Python package for computing identification rates (ID rates) between symmetric positive-definite (SPD) connectivity matrices using a wide variety of distance and divergence metrics.

It provides both an easy-to-use **command-line interface (CLI)** and a **Python API** for flexible, customizable analysis of brain connectomes across different tasks, scan directions, parcellation resolutions, and regularization settings.

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
  --tau 1e-6 \
  --num-subjects 30
```

### 🔑 Key Arguments

| Argument        | Description |
|-----------------|-------------|
| `--base-path`   | Path to root folder containing subject subfolders. |
| `--tasks`       | List of tasks (`REST`, `EMOTION`, `GAMBLING`, `LANGUAGE`, etc.). |
| `--scan-types`  | Two scan directions to compare (e.g., `LR RL`). |
| `--resolutions` | Parcellation sizes (e.g., `100 200 300`). |
| `--metric`      | Distance metric: `alpha_z`, `alpha_pro`, `bw`, `geo`, `log`, `pearson`, `euclid`. |
| `--alpha`, `--z`| Parameters for Alpha-based metrics. |
| `--tau`         | SPD regularization (default: `1e-6`). |
| `--num-subjects`| Maximum number of subjects to include. |

---

## 🚀 Python API Example

```python
import numpy as np
from spd_metrics_id.io import find_subject_paths, load_matrix
from spd_metrics_id.distance import compute_alpha_z_bw
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
D12 = np.array([[compute_alpha_z_bw(A, B, alpha=0.99, z=1.0) for B in mats_rl] for A in mats_lr])
D21 = np.array([[compute_alpha_z_bw(A, B, alpha=0.99, z=1.0) for B in mats_lr] for A in mats_rl])

# Compute ID rates
id1 = compute_id_rate(D12)
id2 = compute_id_rate(D21)
print("Average ID rate:", (id1 + id2) / 2)
```

---

## 🎛 Interactive Analysis Script Example

You can also create a **fully interactive analysis script** using SPD Metrics ID!  
Here’s a preview:

```python
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from spd_metrics_id.io import find_subject_paths, load_matrix
from spd_metrics_id.distance import (
    compute_alpha_z_bw,
    compute_alpha_procrustes,
    compute_bw,
    compute_geodesic_distance,
    compute_log_euclidean_distance,
    compute_pearson_distance,
    compute_euclidean_distance
)
from spd_metrics_id.id_rate import compute_id_rate

def verbose_print(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}")

start_time = time.time()
verbose_print("Starting connectome identification process...")

# Interactive configuration
def multi_choice(prompt, options):
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        print(f"{i}. {opt}")
    choices = input("Enter choices (comma-separated numbers): ")
    idxs = [int(c.strip())-1 for c in choices.split(',')]
    return [options[i] for i in idxs]

# 1) Select tasks
TASKS = ['REST', 'EMOTION', 'LANGUAGE', 'WM', 'MOTOR', 'RELATIONAL', 'GAMBLING', 'SOCIAL']
selected_tasks = multi_choice("Select tasks to process:", TASKS)

# 2) Select metrics
METRIC_FUNCS = {
    'Alpha Z': compute_alpha_z_bw,
    'Alpha Procrustes': compute_alpha_procrustes,
    'Bures-Wasserstein': compute_bw,
    'AI': compute_geodesic_distance,
    'Log-Euclidean': compute_log_euclidean_distance,
    'Pearson': compute_pearson_distance,
    'Euclidean': compute_euclidean_distance
}
selected_metrics = multi_choice("Select distance metrics:", list(METRIC_FUNCS.keys()))

# 3) User parameters (tau, alpha, z)
tau_geo_log = None
if any(m in ['AI', 'Log-Euclidean'] for m in selected_metrics):
    tau_geo_log = [float(x.strip()) for x in input("Enter tau values (comma-separated): ").split(',')]

tau_alpha_z = None
z_alpha_z = None
if 'Alpha Z' in selected_metrics:
    alpha_input = input("Enter alpha values (comma-separated): ")
    tau_alpha_z = [float(t.strip()) for t in alpha_input.split(',')]
    z_input = input("Enter z exponents (comma-separated): ")
    z_alpha_z = [float(z.strip()) for z in z_input.split(',')]

alpha_pro = None
if 'Alpha Procrustes' in selected_metrics:
    alpha_input = input("Enter alpha values for Alpha Procrustes (comma-separated): ")
    alpha_pro = [float(a.strip()) for a in alpha_input.split(',')]

# 4) Directory and subjects
base_dir = input("Enter base directory [connectomes_100/]: ") or "connectomes_100/"
num_subjects = int(input("Number of subjects: ") or 30)

# --- Main Analysis ---
verbose_print("Starting analysis...")
results = []

for task in selected_tasks:
    lr_paths = find_subject_paths(base_dir, task, 'LR', [100], n=num_subjects)
    rl_paths = find_subject_paths(base_dir, task, 'RL', [100], n=num_subjects)
    mats_lr = [load_matrix(p) for p in lr_paths]
    mats_rl = [load_matrix(p) for p in rl_paths]

    for metric in selected_metrics:
        fn = METRIC_FUNCS[metric]
        if metric in ['AI', 'Log-Euclidean']:
            for tau in tau_geo_log:
                D12 = np.array([[fn(A, B, tau) for B in mats_rl] for A in mats_lr])
                D21 = np.array([[fn(A, B, tau) for B in mats_lr] for A in mats_rl])
                id12 = compute_id_rate(D12)
                id21 = compute_id_rate(D21)
                results.append({'task': task, 'metric': metric, 'param': tau, 'accuracy': (id12 + id21) / 2})
        elif metric == 'Alpha Z':
            for alpha in tau_alpha_z:
                for z in z_alpha_z:
                    D12 = np.array([[fn(A, B, alpha, z=z) for B in mats_rl] for A in mats_lr])
                    D21 = np.array([[fn(A, B, alpha, z=z) for B in mats_lr] for A in mats_rl])
                    id12 = compute_id_rate(D12)
                    id21 = compute_id_rate(D21)
                    results.append({'task': task, 'metric': metric, 'param': (alpha, z), 'accuracy': (id12 + id21) / 2})
        elif metric == 'Alpha Procrustes':
            for alpha in alpha_pro:
                D12 = np.array([[fn(A, B, alpha) for B in mats_rl] for A in mats_lr])
                D21 = np.array([[fn(A, B, alpha) for B in mats_lr] for A in mats_rl])
                id12 = compute_id_rate(D12)
                id21 = compute_id_rate(D21)
                results.append({'task': task, 'metric': metric, 'param': alpha, 'accuracy': (id12 + id21) / 2})
        else:
            D12 = np.array([[fn(A, B) for B in mats_rl] for A in mats_lr])
            D21 = np.array([[fn(A, B) for B in mats_lr] for A in mats_rl])
            id12 = compute_id_rate(D12)
            id21 = compute_id_rate(D21)
            results.append({'task': task, 'metric': metric, 'param': None, 'accuracy': (id12 + id21) / 2})

# Summary
df = pd.DataFrame(results)
print("\nIdentification Rates Summary:")
print(df.to_string(index=False))
verbose_print(f"Total runtime: {time.time() - start_time:.2f}s")
```

```
## Testing

Run the full test suite with `pytest`:

```bash
python -m pytest
```

All divergence functions and the ID-rate calculation are covered by unit tests.

---

## Contributing

1. Fork the repository.  
2. Create a new branch:  
   ```bash
   git checkout -b feature/your-feature
   ```  
3. Write code & tests.  
4. Run `pytest` to verify.  
5. Submit a pull request.

Please adhere to PEP 8 and include new tests for any added functionality.

---

## License

SPDX license identifier: **MIT**  
See the [LICENSE](LICENSE) file for full terms.