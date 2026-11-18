"""Basin density from an external sampler through the nanobind surface."""

from __future__ import annotations

import math

import numpy as np

from rgsaddle import ohtst_basin_density


def test_harmonic_engine_matches_the_gaussian_density() -> None:
    thermal = 0.25
    curvature = 2.0
    spread = math.sqrt(thermal / curvature)

    def engine(point, normal, thermal, seed, thermalize, stride, samples, start):
        del thermal, seed, thermalize, stride, start
        xi = float(np.dot(normal, point))
        force = -curvature * xi
        count = int(samples)
        width = int(point.shape[0])
        forces = np.full(count, force, dtype=np.float64)
        positions = np.repeat(np.asarray(point, dtype=np.float64).reshape(1, width), count, axis=0)
        projected = np.zeros((count, width), dtype=np.float64)
        return forces, positions, projected

    minimum = np.array([0.0], dtype=np.float64)
    normal = np.array([1.0], dtype=np.float64)
    result = ohtst_basin_density(engine, minimum, normal, thermal, spread, 0, 1, 80, 4, 1, 0)
    gap = abs(result.density - result.harmonic_density)
    allowance = (
        4.0 * result.density_error
        + result.quadrature_error * result.harmonic_density
        + result.tail * result.harmonic_density
    )
    assert gap <= allowance
