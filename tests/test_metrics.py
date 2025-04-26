import numpy as np
import pytest
from spd_metrics_id.distance import (
    compute_alpha_z_bw,
    compute_alpha_procrustes,
    compute_bw,
    compute_geodesic_distance,
    compute_log_euclidean_distance,
    compute_pearson_distance,
    compute_euclidean_distance,
)

@pytest.mark.parametrize("fn,args", [
    (compute_alpha_z_bw, (np.eye(4), np.eye(4), 0.99, 1.0)),
    (compute_alpha_procrustes, (np.eye(4), np.eye(4), 0.6)),
    (compute_bw, (np.eye(4), np.eye(4))),
    (compute_geodesic_distance, (np.eye(4), np.eye(4))),
    (compute_log_euclidean_distance, (np.eye(4), np.eye(4))),
    (compute_pearson_distance, (np.eye(4), np.eye(4))),
    (compute_euclidean_distance, (np.eye(4), np.eye(4))),
])
def test_zero_on_identity(fn, args):
    assert pytest.approx(0.0, abs=1e-6) == fn(*args)