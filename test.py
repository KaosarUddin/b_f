#!/usr/bin/env python3
"""
alpha_z_bw_test.py

Test script for computing Alpha-Z Bures-Wasserstein divergence-based identification rates
for functional connectivity matrices.

Usage:
    python alpha_z_bw_test.py \
        --base-path connectomes_100/ \
        --scan-types LR RL \
        --num-subjects 30 \
        --alpha 0.99 \
        --z 1.0

If --base-path is not provided, defaults to './connectomes_100/'.
"""

import argparse
import logging
import numpy as np
import os
from scipy.linalg import fractional_matrix_power


def load_connectivity_matrix(file_path: str) -> np.ndarray:
    """Load a connectivity matrix from a plain-text file.

    Each file should contain whitespace-delimited numeric values.

    Args:
        file_path (str): Path to the matrix text file.

    Returns:
        np.ndarray: Loaded square matrix, or None if loading fails.
    """
    try:
        matrix = np.loadtxt(file_path, delimiter=' ')
        logging.debug(f"Loaded matrix from {file_path} with shape {matrix.shape}")
        return matrix
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        return None


def generate_file_paths(base_path: str, scan_type: str, num_subjects: int = 30) -> list[str]:
    """Generate file paths for a given scan type across multiple subjects.

    Args:
        base_path (str): Directory containing subject subdirectories.
        scan_type (str): Scan type identifier, e.g., 'LR' or 'RL'.
        num_subjects (int): Maximum number of subjects to include.

    Returns:
        list[str]: Fully qualified file paths for connectivity matrices.
    """
    subject_ids = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    subject_ids.sort()
    subject_ids = subject_ids[:num_subjects]

    paths = []
    for sid in subject_ids:
        file_name = f"{sid}_rfMRI_REST1_{scan_type}_100"
        full_path = os.path.join(base_path, sid, file_name)
        paths.append(full_path)
        logging.debug(f"Generated file path: {full_path}")
    return paths


def compute_alpha_z_BW_distance(A: np.ndarray, B: np.ndarray, alpha: float, z: float) -> float:
    """Compute the Alpha-Z Bures-Wasserstein divergence between two PSD matrices.

    D_{α,z}(A, B) = trace((1-α)A + αB) - trace([B^{(1-α)/(2z)} A^{α/z} B^{(1-α)/(2z)}]^{z}).

    Args:
        A (np.ndarray): First positive semi-definite matrix.
        B (np.ndarray): Second positive semi-definite matrix.
        alpha (float): Weight parameter in [0, 1].
        z (float): Exponent parameter, with alpha <= z <= 1.

    Returns:
        float: Real-valued divergence.
    """
    if not (0 <= alpha <= z <= 1):
        raise ValueError("Alpha and z must satisfy 0 <= alpha <= z <= 1")

    P = fractional_matrix_power(B, (1 - alpha) / (2 * z)) if z != 0 else np.zeros_like(A)
    Q = fractional_matrix_power(A, alpha / z) if z != 0 else np.zeros_like(A)
    M = P.dot(Q).dot(P)
    Qaz = fractional_matrix_power(M, z) if z != 0 else np.zeros_like(A)

    divergence = np.trace((1 - alpha) * A + alpha * B) - np.trace(Qaz)
    logging.debug(f"Computed divergence: {divergence}")
    return float(np.real(divergence))


def compute_distance_matrix(mats1: list[np.ndarray], mats2: list[np.ndarray], alpha: float, z: float) -> np.ndarray:
    """Compute pairwise Alpha-Z BW divergence matrix between two lists of matrices.

    Args:
        mats1 (list[np.ndarray]): List of PSD matrices.
        mats2 (list[np.ndarray]): List of PSD matrices.
        alpha (float): Alpha parameter.
        z (float): Z parameter.

    Returns:
        np.ndarray: Distance matrix of shape (n_subjects, n_subjects).
    """
    n = len(mats1)
    D = np.full((n, n), np.nan)
    for i, A in enumerate(mats1):
        if A is None:
            continue
        for j, B in enumerate(mats2):
            if B is None:
                continue
            D[i, j] = compute_alpha_z_BW_distance(A, B, alpha, z)
    return D


def compute_id_rate(dist_mat: np.ndarray) -> float:
    """Compute identification rate: proportion of correct nearest matches on the diagonal.

    Args:
        dist_mat (np.ndarray): Pairwise distance matrix.

    Returns:
        float: Identification rate in [0, 1].
    """
    correct = 0
    n = dist_mat.shape[0]
    for i in range(n):
        if np.isnan(dist_mat[i]).all():
            continue
        idx_min = int(np.nanargmin(dist_mat[i]))
        if idx_min == i:
            correct += 1
    return correct / n


def main():
    parser = argparse.ArgumentParser(
        description="Compute ID rates using Alpha-Z Bures-Wasserstein divergence."
    )
    parser.add_argument(
        "--base-path", type=str, default="connectomes_100/",
        help="Base directory containing connectivity subdirectories. (default: './connectomes_100/')"
    )
    parser.add_argument(
        "--scan-types", nargs=2, default=["LR", "RL"], metavar=('TYPE1','TYPE2'),
        help="Two scan types to compare (e.g., LR RL)."
    )
    parser.add_argument(
        "--num-subjects", type=int, default=30,
        help="Number of subjects to include (default: 30)."
    )
    parser.add_argument(
        "--alpha", type=float, default=0.99,
        help="Alpha parameter (default: 0.99)."
    )
    parser.add_argument(
        "--z", type=float, default=1.0,
        help="Z parameter (default: 1.0)."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Verify base path exists
    if not os.path.isdir(args.base_path):
        logging.error(f"Base path '{args.base_path}' does not exist.")
        return

    # Generate file paths
    paths1 = generate_file_paths(args.base_path, args.scan_types[0], args.num_subjects)
    paths2 = generate_file_paths(args.base_path, args.scan_types[1], args.num_subjects)

    # Load matrices
    mats1 = [load_connectivity_matrix(p) for p in paths1]
    mats2 = [load_connectivity_matrix(p) for p in paths2]

    # Compute distance matrices and ID rates
    D12 = compute_distance_matrix(mats1, mats2, args.alpha, args.z)
    id1 = compute_id_rate(D12)
    D21 = compute_distance_matrix(mats2, mats1, args.alpha, args.z)
    id2 = compute_id_rate(D21)
    average_id = (id1 + id2) / 2

    # Print results
    logging.info(f"ID rate ({args.scan_types[0]}→{args.scan_types[1]}): {id1:.4f}")
    logging.info(f"ID rate ({args.scan_types[1]}→{args.scan_types[0]}): {id2:.4f}")
    logging.info(f"Average ID rate: {average_id:.4f}")


if __name__ == "__main__":
    main()
