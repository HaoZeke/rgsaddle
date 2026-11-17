"""Fixed-plane thermodynamic integration through the callback binding."""

from pathlib import Path

import numpy as np

import rgsaddle
from rgsaddle import THERMODYNAMIC_DERIVATIVE, hyperplane_thermodynamic_integration
from rgsaddle._core import hyperplane_thermodynamic_integration as core_ti
from rgsaddle._core import THERMODYNAMIC_DERIVATIVE as CORE_DERIVATIVE

STUB = Path(rgsaddle.__file__).resolve().parent / "_core.pyi"
DOC = (
    "Fixed-plane lambda TI through a Python sampler. "
    "The callback receives point, normal, lambda, thermal, seed, thermalize, "
    "stride and samples, and returns float64 C-contiguous "
    "V_full - V_harmonic observations."
)


def test_thermodynamic_derivative_name_is_stable() -> None:
    assert THERMODYNAMIC_DERIVATIVE == CORE_DERIVATIVE == "potential.full_minus_harmonic"


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


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_binding_docstring_names_the_sampler_contract() -> None:
    for fn in (core_ti, hyperplane_thermodynamic_integration):
        text = _flat(fn.__doc__ or "")
        assert text.startswith("hyperplane_thermodynamic_integration(")
        assert "Callable[..., numpy.typing.NDArray[numpy.float64]]" in text
        assert "point: numpy.typing.NDArray[numpy.float64]" in text
        assert "normal: numpy.typing.NDArray[numpy.float64]" in text
        assert _flat(DOC) in text


def test_checked_in_stub_matches_stubgen_contract() -> None:
    text = STUB.read_text(encoding="utf-8")
    start = text.index("def hyperplane_thermodynamic_integration")
    assert start < text.index("class StencilReport")
    window = text[start : text.index("\nclass ", start)]
    assert "Callable[..., numpy.typing.NDArray[numpy.float64]]" in window
    assert "point: numpy.typing.NDArray[numpy.float64]" in window
    assert "normal: numpy.typing.NDArray[numpy.float64]" in window
    assert _flat(DOC) in _flat(window)
    assert "Fixed-plane lambda TI. sampler(" not in window
