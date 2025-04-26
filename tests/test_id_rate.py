import numpy as np
import pytest  
from spd_metrics_id.id_rate import compute_id_rate

def test_id_rate_perfect():
     D = np.array([[0,1],[1,0]])
     assert compute_id_rate(D) == pytest.approx(1.0)

def test_id_rate_type():
     D = np.array([[0,2],[0.5,0]])
     assert isinstance(compute_id_rate(D), float)
     
