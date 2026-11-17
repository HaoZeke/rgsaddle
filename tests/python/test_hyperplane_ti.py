"""Fixed-plane thermodynamic integration through the callback binding."""

import numpy as np

from rgsaddle import hyperplane_thermodynamic_integration


def test_constant_derivative_integrates_exactly() -> None:
    value = 1.75

    def sampler(
        point: np.ndarray,
        normal: np.ndarray,
        coupling: float,
        thermal: float,
        seed: int,
        thermalize: int,
        stride: int,
        samples: int,
    ) -> np.ndarray:
        assert point.shape == normal.shape == (2,)
        assert 0.0 < coupling < 1.0
        assert thermal > 0.0 and seed >= 0
        assert thermalize >= 0 and stride > 0
        return np.full(samples, value, dtype=np.float64)

    free_energy, error, evaluations = hyperplane_thermodynamic_integration(
        sampler,
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        nodes=4,
        samples=100,
        blocks=4,
    )
    assert abs(free_energy - value) < 2.0e-14
    assert error == 0.0
    assert evaluations == 400
