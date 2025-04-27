import numpy as np
from scipy.linalg import fractional_matrix_power, sqrtm, logm
from numpy.linalg import norm


def make_spd(matrix: np.ndarray, tau: float = 1e-6) -> np.ndarray:
    """Symmetrize matrix and add tau*I for SPD regularization."""
    sym = (matrix + matrix.T) / 2
    return sym + tau * np.eye(sym.shape[0])




def alpha_z_bw(A: np.ndarray, B: np.ndarray, alpha: float, z: float) -> float:
    """Alpha-Z Bures–Wasserstein divergence."""
    if not (0 <= alpha <= z <= 1):
        raise ValueError("Alpha and z must satisfy 0 <= alpha <= z <= 1")
    
    def Q_alpha_z(A, B, alpha, z):
        if z == 0:
            return np.zeros_like(A)
        part1 = fractional_matrix_power(B, (1-alpha)/(2*z))
        part2 = fractional_matrix_power(A, alpha/z)
        part3 = fractional_matrix_power(B, (1-alpha)/(2*z))
        Q_az = fractional_matrix_power(part1.dot(part2).dot(part3), z)
        return Q_az

    Q_az = Q_alpha_z(A, B, alpha, z)
    divergence = np.trace((1-alpha) * A + alpha * B) - np.trace(Q_az)    
    return float(np.real(divergence))

def alpha_procrustes(A: np.ndarray, B: np.ndarray, alpha: float) -> float:
    """Alpha-procrustes distance."""
    if alpha <= 0:
        raise ValueError("alpha > 0 required")
    A2 = fractional_matrix_power(A, 2 * alpha)
    B2 = fractional_matrix_power(B, 2 * alpha)
    S = fractional_matrix_power(A, alpha).dot(B2).dot(fractional_matrix_power(A, alpha))
    return float(np.sqrt((np.trace(A2) + np.trace(B2) - 2 * np.trace(sqrtm(S))) / alpha**2))


def bures_wasserstein(X: np.ndarray, Y: np.ndarray) -> float:
    """Bures–Wasserstein distance (no tau regularization)."""
    # Ensure symmetry but no additional regularization
    Xsp = (X + X.T) / 2
    Ysp = (Y + Y.T) / 2
    root = sqrtm(Xsp)
    term = sqrtm(root.dot(Ysp).dot(root))
    return float(np.real(np.trace(Xsp) + np.trace(Ysp) - 2 * np.trace(term)))


def geodesic_distance(
    A: np.ndarray,
    B: np.ndarray,
    tau: float = 1e-6
) -> float:
    """Affine‐invariant Riemannian distance with SPD regularization."""
    A_sp = make_spd(A, tau)
    B_sp = make_spd(B, tau)
    root = sqrtm(A_sp)
    invr = np.linalg.inv(root)
    C = invr.dot(B_sp).dot(invr)
    return float(norm(logm(C), 'fro'))

def log_euclidean_distance(X: np.ndarray, Y: np.ndarray, tau: float = 1e-6) -> float:
    """Log-Euclidean distance with SPD regularization."""
    X_sp = make_spd(X, tau)
    Y_sp = make_spd(Y, tau)
    return float(norm(logm(X_sp) - logm(Y_sp), 'fro'))


def pearson_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Pearson-based distance (no SPD regularization)."""
    v1, v2 = X.flatten(), Y.flatten()
    r = np.corrcoef(v1, v2)[0,1]
    return float(1 - r)


def euclidean_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Euclidean distance of flattened matrices (no SPD regularization)."""
    return float(norm(X.flatten() - Y.flatten()))