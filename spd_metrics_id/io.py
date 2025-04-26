import os
import numpy as np
from typing import List, Union

def load_matrix(path: str) -> np.ndarray:
    """Load a whitespace-delimited SPD/FC matrix from a text file."""
    return np.loadtxt(path, delimiter=' ')


def find_subject_paths(
    base_dir: str,
    task: str,
    scan: str,
    resolutions: Union[int, List[int]],
    n: int = None
) -> List[str]:
    """
    Generate file paths for each subject, task, scan direction, and resolution.

    Uses 'rfMRI_REST1' for REST (task 'REST' or 'REST1') and 'tfMRI_<TASK>' for others.

    Args:
      base_dir: root directory containing subject subfolders
      task: fMRI task label (e.g., 'REST', 'EMOTION', ...)
      scan: scan direction (e.g., 'LR' or 'RL')
      resolutions: single resolution or list (e.g., [100,200,...])
      n: max number of subjects (None = all)

    Returns:
      List of filepath strings
    """
    if isinstance(resolutions, int):
        resolutions = [resolutions]
    subs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    subs.sort()
    if n is not None:
        subs = subs[:n]
    paths = []
    for sid in subs:
        for res in resolutions:
            # REST uses REST1 in filename
            if task.upper() in ("REST", "REST1"):
                fname = f"{sid}_rfMRI_REST1_{scan}_{res}"
            else:
                fname = f"{sid}_tfMRI_{task}_{scan}_{res}"
            full_path = os.path.join(base_dir, sid, fname)
            paths.append(full_path)
    return paths