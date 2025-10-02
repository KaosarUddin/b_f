import numpy as np
from spd_metrics_id.io import find_subject_paths, load_matrix
from spd_metrics_id.distance import alpha_z_bw
from spd_metrics_id.id_rate import compute_id_rate

base = "connectomes_100/"
# find the 30 REST‐LR and REST‐RL files at resolution 100
lr_paths = find_subject_paths(base, "REST1", "LR", [100], n=30)
rl_paths = find_subject_paths(base, "REST1", "RL", [100], n=30)

# load them
mats_lr = [load_matrix(p) for p in lr_paths]
mats_rl = [load_matrix(p) for p in rl_paths]

# build distance matrices
D12 = np.array([[alpha_z_bw(A, B,0.99,1)
                 for B in mats_rl] for A in mats_lr])
D21 = np.array([[alpha_z_bw(A, B,0.99,1)
                 for B in mats_lr] for A in mats_rl])

# compute ID rates
id1 = compute_id_rate(D12)
id2 = compute_id_rate(D21)
print("Average ID rate:", (id1 + id2) / 2)