"""
This module provides the functionality for:
- The randomized (generalized) spectral decomposition of a hermitian matrix.
- The randomized (generalized) singular value decomposition.
- Means to check the accuracy of the randomized algorithms.

All the functions in this module are based on theory from the below paper.

- Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure with
    randomness: Probabilistic algorithms for constructing approximate matrix
    decompositions. SIAM review, 53(2), 217-288.
"""

__docformat__ = "google"

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

type ExplicitLinearOperator = np.ndarray | sps.spmatrix
type LinearOperator = ExplicitLinearOperator | spsla.LinearOperator


def single_pass_hep(
    A: LinearOperator, k: int, Omega: np.ndarray | None = None, s: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Single pass randomized (generalized) spectral decomposition of a hermitian matrix.

    Args:
        A (LinearOperator): n x m hermitian matrix to decompose.
        k (int): The number of eigenpairs to approximate.
        Omega (np.ndarray | None): The random Gaussian matrix to use in the
            decomposition. If None, a random Gaussian matrix will be generated.
            Should have m rows and atleast k columns. Defaults to None.
        s (int): The number of power iterations to use in the decomposition. Defaults to
            1.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple whose first element is the approximation to
            the k largest eigenvalues and the second element is the corresponding
            approximate eigenvectors.
    """
    n, m = A.shape
    if Omega is None:
        Omega = np.random.randn(m, k)
    else:
        if Omega.shape[0] != m:
            raise ValueError("Omega's number of rows must match A's number of columns.")
        elif Omega.shape[1] < k:
            raise ValueError("Omega must have at least k columns.")
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    Y = Omega.copy()
    Y_pr = np.zeros_like(Y)
    for _ in range(s):
        Y_pr, Y = Y, Y_pr
        Y = A @ Y_pr

    Q = np.linalg.qr(Y, mode="reduced")[0]

    B = np.linalg.solve(Omega.T @ Q, Y.T @ Q).T

    d, V = np.linalg.eigh(B)

    desc_idx = np.argsort(-d)

    d = d[desc_idx[0:k]]
    U = Q @ V[:, desc_idx[0:k]]

    return d, U


def double_pass_hep(
    A: LinearOperator, k: int, Omega: ExplicitLinearOperator | None = None, s: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Double pass randomized (generalized) spectral decomposition of a hermitian matrix.

    Args:
        A (LinearOperator): The hermitian matrix to decompose.
        k (int): The number of eigenpairs to approximate.
        Omega (np.ndarray | None): The random Gaussian matrix to use in the
            decomposition. If None, a random Gaussian matrix will be generated. Defaults
            to None.
        s (int): The number of power iterations to use in the decomposition. Defaults to
            1.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple whose first element is the approximate
            eigenvalues and the second element is the approximate eigenvectors.
    """
    n, m = A.shape
    if Omega is None:
        Omega = np.random.randn(m, k)
    else:
        if Omega.shape[0] != m:
            raise ValueError("Omega's number of rows must match A's number of columns.")
        elif Omega.shape[1] < k:
            raise ValueError("Omega must have at least k columns.")
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    Y = Omega.copy()  # type:ignore
    Y_pr = np.zeros_like(Y)
    for _ in range(s):
        Y_pr, Y = Y, Y_pr
        Y = A @ Y_pr

    Q = np.linalg.qr(Y, mode="reduced")[0]  # type:ignore

    B = Q.T @ (A @ Q)

    d, V = np.linalg.eigh(B)

    desc_idx = np.argsort(-d)

    d = d[desc_idx[0:k]]
    U = Q @ V[:, desc_idx[0:k]]

    return d, U


def chol_qr(
    Y: np.ndarray, W: ExplicitLinearOperator | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Algorithm to compute the QR factorization of Y using an internal Cholesky
    decomposition in the W-inner product.

    Args:
        Y (LinearOperator): The matrix to decompose.
        W (np.ndarray | sps.spmatrix | None): The inner product to perform the
            decomposition within. If None, then it's assumed to be the identity.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple of Q, WQ, and R.
    """
    m, _ = Y.shape
    if W is None:
        W = np.eye(m, m)
    if W.shape != (m, m):
        raise ValueError("Y and W are not conformable!")
    Z = W @ Y
    C = Y.T @ Z
    R = np.linalg.cholesky(C).T
    Q = np.linalg.solve(R.T, Y.T).T
    # TODO: Check WQ
    WQ = np.linalg.solve(R.T, Z.T).T
    return Q, WQ, R
