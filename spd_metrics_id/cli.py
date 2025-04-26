import argparse
import logging
import numpy as np
from .io import load_matrix, find_subject_paths
from .distance import (
    compute_alpha_z_bw,
    compute_alpha_procrustes,
    compute_bw,
    compute_geodesic_distance,
    compute_log_euclidean_distance,
    compute_pearson_distance,
    compute_euclidean_distance,
)
from .id_rate import compute_id_rate

# Define for each metric: (Pretty Name, function, list of kwargs it needs)
METRICS = {
    'alpha_z': ('Alpha-Z BW', compute_alpha_z_bw, ['alpha', 'z']),
    'alpha_pro': ('Alpha-Procrustes', compute_alpha_procrustes, ['alpha']),
    'bw': ('Bures-Wasserstein', compute_bw, []),
    'geo': ('Affine-invariant', compute_geodesic_distance, ['tau']),
    'log': ('Log-Euclidean', compute_log_euclidean_distance, ['tau']),
    'pearson': ('Pearson', compute_pearson_distance, []),
    'euclid': ('Euclidean', compute_euclidean_distance, []),
}

def main():
    parser = argparse.ArgumentParser("SPD-ID: compute ID rates for SPD metrics")
    parser.add_argument("--base-path", required=True,
                        help="Root directory containing subject subfolders")
    parser.add_argument("--tasks", nargs='+',
                        default=["REST","EMOTION","GAMBLING","LANGUAGE","MOTOR","RELATIONAL","SOCIAL","WM"],
                        help="List of tasks (REST uses rfMRI_REST1, others tfMRI_<TASK>)")
    parser.add_argument("--scan-types", nargs=2, default=["LR","RL"],
                        help="Pair of scan directions to compare (e.g. LR RL)")
    parser.add_argument("--resolutions", nargs='+', type=int, default=[100],
                        help="Parcellation resolutions (e.g. 100 200 ...)")
    parser.add_argument("--metric", choices=METRICS.keys(), default='alpha_z',
                        help="Which SPD metric to use for distances")
    parser.add_argument("--alpha", type=float, default=0.99,
                        help="Alpha parameter (for alpha_z or alpha_pro)")
    parser.add_argument("--z", type=float, default=1.0,
                        help="Z parameter (for alpha_z only)")
    parser.add_argument("--tau", type=float, default=1e-6,
                        help="Regularization tau (for geo/log metrics)")
    parser.add_argument("--num-subjects", type=int, default=None,
                        help="Max number of subjects to include (None = all)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    name, fn, params = METRICS[args.metric]

    for task in args.tasks:
        # Load LR and RL matrices for this task
        paths_LR = find_subject_paths(args.base_path, task, args.scan_types[0], args.resolutions, args.num_subjects)
        paths_RL = find_subject_paths(args.base_path, task, args.scan_types[1], args.resolutions, args.num_subjects)
        mats_LR = [load_matrix(p) for p in paths_LR]
        mats_RL = [load_matrix(p) for p in paths_RL]

        # Compute cross‐scan distances
        D12 = np.zeros((len(mats_LR), len(mats_RL)))
        D21 = np.zeros((len(mats_RL), len(mats_LR)))
        for i, A in enumerate(mats_LR):
            for j, B in enumerate(mats_RL):
                kwargs = {k: getattr(args, k) for k in params}
                D12[i, j] = fn(A, B, **kwargs)
        for i, A in enumerate(mats_RL):
            for j, B in enumerate(mats_LR):
                kwargs = {k: getattr(args, k) for k in params}
                D21[i, j] = fn(A, B, **kwargs)

        # Identification rates
        id1 = compute_id_rate(D12)
        id2 = compute_id_rate(D21)
        avg_id = (id1 + id2) / 2

        logging.info(
            f"Task={task}, Scans={args.scan_types[0]}→{args.scan_types[1]}, "
            f"Metric={name}, Res={args.resolutions}, Subjects={args.num_subjects}, "
            f"{'Tau='+str(args.tau) if 'tau' in params else ''} "
            f"ID1={id1:.4f}, ID2={id2:.4f}, AvgID={avg_id:.4f}"
        )

if __name__ == "__main__":
    main()
