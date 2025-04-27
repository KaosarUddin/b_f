import numpy as np
import pytest
from spd_metrics_id.distance import (
    alpha_z_bw,
    alpha_procrustes,
    bures_wasserstein,
    geodesic_distance,
    log_euclidean_distance,
    pearson_distance,
    euclidean_distance,
)

@pytest.mark.parametrize("fn,args", [
    (alpha_z_bw, (np.eye(4), np.eye(4), 0.99, 1.0)),
    (alpha_procrustes, (np.eye(4), np.eye(4), 0.6)),
    (bures_wasserstein, (np.eye(4), np.eye(4))),
    (geodesic_distance, (np.eye(4), np.eye(4))),
    (log_euclidean_distance, (np.eye(4), np.eye(4))),
    (pearson_distance, (np.eye(4), np.eye(4))),
    (euclidean_distance, (np.eye(4), np.eye(4))),
])
def test_zero_on_identity(fn, args):
    assert pytest.approx(0.0, abs=1e-6) == fn(*args)