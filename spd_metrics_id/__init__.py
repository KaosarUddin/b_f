__version__ = "0.1.0"

from .io import load_matrix, find_subject_paths
from .distance import (
    alpha_z_bw,
    alpha_procrustes,
    bures_wasserstein,
    geodesic_distance,
    log_euclidean_distance,
    pearson_distance,
    euclidean_distance,
)
from .id_rate import compute_id_rate