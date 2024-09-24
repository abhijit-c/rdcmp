"""
Tests for the randomized hermition eigenvalue problem algorithms
"""

from itertools import product

import numpy as np
import scipy.sparse as sps

import pytest

from rdcmp import single_pass_hep

RANDOM_SEED = 42


@pytest.mark.parametrize(
    "n,k,type",
    product([100, 1000], [10, 50], ["dense", "sparse", "linear_operator"]),
)
def test_single_pass_hep(n, k, type, n_trials=10):
    """
    Test the single pass hep algorithm.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    Omega = [rng.standard_normal((n, k)) for _ in range(n_trials)]
    for i in range(n_trials):
        if type == "dense":
            A = rng.standard_normal((n, k))
            A = A.T @ A
        elif type == "sparse":
            A = sps.random(n, k, density=0.1, format="csr", random_state=rng)
            A = A.T @ A
        elif type == "linear_operator":
            A = rng.standard_normal((n, k))
            A = A.T @ A
            A = sps.linalg.aslinearoperator(A)

        d, U = single_pass_hep(A, k, Omega=Omega[i])
        d_true = sps.linalg.eigsh(A, k=k, which="LM")

        assert d.shape == (k,)
        assert U.shape == (n, k)
        assert np.allclose(U.T @ U, np.eye(k))
        assert np.allclose(d, d_true)
