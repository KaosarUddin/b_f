# SPD Metrics ID

Compute identification rates (ID-rates) for symmetric positive-definite (SPD) connectivity matrices using a variety of distance and divergence metrics.

---

## Features

- **Alpha-Z Bures–Wasserstein divergence**  
- **Alpha-Procrustes** (“ProE”) distance  
- **Bures–Wasserstein** distance  
- **Affine-invariant Riemannian** distance  
- **Log-Euclidean** distance  
- **Pearson** correlation–based distance  
- **Euclidean** distance on flattened matrices  
- CLI interface with customizable tasks, scan directions, parcellation resolutions, and SPD regularization (`τ`)  
- Python API for programmatic integration  
- Unit tests with **pytest**  

---

## Installation

Clone the repo and install:

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
pip install --upgrade pip
pip install -e .
```

---

## Command-Line Usage

After installation, the `spd-id` console script is available:

```bash
spd-id \
  --base-path PATH/TO/DATA \
  --tasks REST1 LANGUAGE EMOTION \
  --scan-types LR RL \
  --resolutions 100 200 \
  --metric alpha_z \
  --alpha 0.99 \
  --z 1.0 \
  --tau 1e-6 \
  --num-subjects 30
```

### Key Arguments

- `--base-path`  
  Path to root folder containing one subfolder per subject.

- `--tasks`  
  List of tasks. Use `REST1` for resting-state (`rfMRI_REST1`) or any of  
  `EMOTION`, `GAMBLING`, `LANGUAGE`, `MOTOR`, `RELATIONAL`, `SOCIAL`, `WM`  
  for task fMRI (`tfMRI_<TASK>`).

- `--scan-types`  
  Two scan directions to compare, e.g. `LR RL`.

- `--resolutions`  
  One or more parcellation sizes, e.g. `100 200 300`.

- `--metric`  
  Choice of SPD metric:  
  `alpha_z`, `alpha_pro`, `bw`, `geo`, `log`, `pearson`, `euclid`.

- `--alpha`, `--z`  
  Parameters for `alpha_z` and `alpha_pro` metrics.

- `--tau`  
  SPD regularization for `geo` and `log` metrics (default `1e-6`).

- `--num-subjects`  
  Maximum number of subjects to include (default: all).

### Example Output

```
INFO: Task=REST, Scans=LR→RL, Metric=Alpha-Z BW, Res=[100], Subjects=30, ID1=0.9000, ID2=0.8333, AvgID=0.8667
```

---

## Python API

```python
import numpy as np
from spd_metrics_id.io import find_subject_paths, load_matrix
from spd_metrics_id.distance import compute_alpha_z_bw
from spd_metrics_id.id_rate import compute_id_rate

base = "connectomes_100/"
# find the 30 REST‐LR and REST‐RL files at resolution 100
lr_paths = find_subject_paths(base, "REST", "LR", [100], n=30)
rl_paths = find_subject_paths(base, "REST", "RL", [100], n=30)

# load them
mats_lr = [load_matrix(p) for p in lr_paths]
mats_rl = [load_matrix(p) for p in rl_paths]

# build distance matrices
D12 = np.array([[compute_compute_alpha_z_bw(A, B,0.99,1)
                 for B in mats_rl] for A in mats_lr])
D21 = np.array([[compute_compute_alpha_z_bw(A, B,0.99,1)
                 for B in mats_lr] for A in mats_rl])

# compute ID rates
id1 = compute_id_rate(D12)
id2 = compute_id_rate(D21)
print("Average ID rate:", (id1 + id2) / 2)
```

---

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