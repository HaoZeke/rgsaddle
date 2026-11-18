"""rgsaddle nanobind surface: status mapping and the ASE image callback."""

from collections.abc import Callable
from typing import Annotated

import numpy
import numpy.typing
from numpy.typing import NDArray


def version() -> str:
    """Package version string."""

RGSADDLE_FORCE_L2: int = 0

RGSADDLE_FORCE_LINF: int = 1

RGSADDLE_FORCE_MAX_ATOM: int = 2

RGSADDLE_MINMODE_DIMER: int = 0

RGSADDLE_MINMODE_LANCZOS: int = 1

RGSADDLE_IRC_GS2: int = 0

RGSADDLE_IRC_MOROKUMA: int = 1

RGSADDLE_IRC_SELLA_SPHERE: int = 2

class NumericalError(ArithmeticError):
    pass

class ClaimSlot:
    def __init__(self) -> None: ...

    def __enter__(self) -> ClaimSlot:
        """Claim SurfaceAdapter until __exit__."""

    def __exit__(self, exc_type: object | None, exc_value: object | None, traceback: object | None) -> bool:
        """Release the claim. Exceptions propagate."""

def last_error() -> str:
    """
    Thread-local slot reserved for the C ABI last-error text. Returns an empty string unless an error text was set. raise_status reads it and then clears it.
    """

class StencilReport:
    @property
    def status(self) -> int:
        """0 while running, 1 when converged."""

    @property
    def max_force(self) -> float:
        """Maximum absolute force component at the step."""

    @property
    def curvature(self) -> float:
        """Curvature along the mode. NaN for a minimiser."""

    @property
    def iteration(self) -> int:
        """Steps taken."""

    @property
    def evaluations(self) -> int:
        """Surface evaluations in this call."""

    def converged(self) -> bool:
        """True when status is converged."""

class Stencil:
    @staticmethod
    def point(x: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', writable=False)], preset: str = 'dimer', mode: object | None = None, dr: float = 0.001, force_tol: float = 0.001, max_move: float = 0.2, rotation_tol: float = 0.0001, max_rotations: int = 20, krylov_dim: int = 12, threads: int = 1, art_threshold: float = 0.0, art_push: int = 20) -> Stencil:
        """
        Point search. dimer and lanczos follow MinModeSession. The dimer keeps its Polak-Ribiere rotation history across translations. rotation_tol 1e-4 also stops the dimer after one rotation, and lanczos at two Krylov vectors, once the rotational force is under 1. rotation_angle_tol is the Rust default of five degrees (0 would disable that angle test) and is not a Python argument. A curvature that is not negative steps by max_move along the mode. Band preset names raise ValueError.
        """

    @staticmethod
    def band(x: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', writable=False)], preset: str = 'ci-neb') -> None:
        """Band presets stay on feat/band-methods-zoom and raise ValueError."""

    def step(self, surface: Callable[..., StencilReport]) -> StencilReport:
        """
        One step. surface(x) returns (energy, gradient) with x shaped (rows, dof).
        """

    def run(self, surface: Callable[..., StencilReport], max_steps: int) -> StencilReport:
        """Loop of step until convergence or max_steps."""

    def positions(self) -> NDArray[numpy.float64]:
        """Copy of the current position."""

    def mode(self) -> numpy.typing.NDArray[numpy.float64] | None:
        """Copy of the mode, or None for a minimiser."""

    def curvature(self) -> float:
        """Curvature along the mode."""

class ConnectReport:
    @property
    def position(self) -> NDArray[numpy.float64]:
        """Copy of the returned geometry."""

    @property
    def energy(self) -> float:
        """Energy at that geometry."""

    @property
    def force_norm(self) -> float:
        """Euclidean norm of the Cartesian gradient."""

    @property
    def kind(self) -> str:
        """
        s for an accepted saddle, u for an upper bound, f when a segment could not be split.
        """

    @property
    def hessian_index(self) -> int:
        """
        Negative eigenvalues of the rigid-projected Hessian, or -1 when it was not computed.
        """

    @property
    def evaluations(self) -> int:
        """Surface evaluations, including a Hessian that decided acceptance."""

def connect(surface: Callable[..., tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]], left: numpy.typing.NDArray[numpy.float64], right: numpy.typing.NDArray[numpy.float64], force_tol: float = 0.001, max_calls: int = 250000, quench_nodes: bool = True) -> ConnectReport:
    """
    Double-ended connect. surface(x) returns (energy, gradient) for one image shaped (1, 3N).
    """

class SessionReport:
    @property
    def status(self) -> int:
        """0 while running, 1 when converged."""

    @property
    def max_force(self) -> float:
        """Largest force component, or NaN for an estimate."""

    @property
    def curvature(self) -> float:
        """Curvature along the mode. 0 for a band projection."""

    @property
    def iteration(self) -> int:
        """Steps taken."""

    @property
    def evaluations(self) -> int:
        """Surface evaluations in this call."""

    @property
    def ci_index(self) -> int:
        """Climbing image, or -1."""

    @property
    def rotations(self) -> int:
        """Dimer rotations or Lanczos actions."""

    @property
    def saddle_index(self) -> int:
        """Saddle index certified at the session's current position, or -1."""

class SaddleIndex:
    @property
    def certified(self) -> bool:
        """
        True when a certificate holds for the current point; otherwise the counts are the session's.
        """

    @property
    def index(self) -> int:
        """Curvatures under -negative_tol; 1 for a first-order saddle."""

    @property
    def converged(self) -> bool:
        """Every resolved pair met residual_tol, or the space filled."""

    @property
    def curvatures(self) -> NDArray[numpy.float64]:
        """Lowest curvatures resolved, ascending, energy per length squared."""

    @property
    def evaluations(self) -> int:
        """Surface evaluations of this certificate."""

    @property
    def total_evaluations(self) -> int:
        """
        Surface evaluations of every certificate and escape push of the session.
        """

    @property
    def escapes(self) -> int:
        """Escape pushes of the session."""

def saddle_index(surface: Callable[..., tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]], x: numpy.typing.NDArray[numpy.float64], gradient: numpy.typing.NDArray[numpy.float64] | None = None, mode: numpy.typing.NDArray[numpy.float64] | None = None, fixed: numpy.typing.NDArray[numpy.uint8] | None = None, lowest: int = 4, max_actions: int = 100, dr: float = 0.001, residual_tol: float = 0.01, negative_tol: float = 0.001) -> SaddleIndex:
    """
    Saddle index of x (ABI minor 40): the lowest curvatures on the free coordinates by block Lanczos on forward-difference Hessian actions, each one surface call. surface(x) is one image shaped (1, 3N). gradient at x saves one call; mode seeds the space; fixed is one uint8 per atom, 1 for a fixed atom.
    """

def project_band(positions: numpy.typing.NDArray[numpy.float64], energies: numpy.typing.NDArray[numpy.float64], gradients: numpy.typing.NDArray[numpy.float64], k: float = 0.1, climb: bool = False, projection: str = 'neb', tangent: str = 'improved', remove_translation: bool = False, force_tol: float = 0.001, fixed: numpy.typing.NDArray[numpy.uint8] | None = None) -> tuple[numpy.typing.NDArray[numpy.float64], int]:
    """
    Project an evaluated band. Returns (forces, ci_index). forces is flat, one row per image including zero endpoints, in the ASE force convention. climb selects the interior image by the endpoint-slope rule on this call; a higher endpoint that slopes into the band does not climb. Units are the caller's. k is energy per length squared. fixed is one uint8 per atom, 1 for a fixed atom (ABI minor 39): its force rows are zero, tangents and springs use the free atoms, and the mean force stays. The endpoints must agree on every fixed atom.
    """

def reparametrize(positions: numpy.typing.NDArray[numpy.float64], scheme: str = 'linear', anchor: int | None = None, cell: numpy.typing.NDArray[numpy.float64] | None = None) -> numpy.typing.NDArray[numpy.float64]:
    """
    Place the interior images of a band at equal arc length and return the new band. scheme 'linear' places them on the polyline; 'not-a-knot' (or 'spline') on the not-a-knot cubic spline through the images in their arc length (E, Ren and Vanden-Eijnden 2007). The first and last rows, and an interior anchor, do not move. cell is the 3x3 row-major cell; the chords are then minimum images. No surface is evaluated.
    """

def halve_raised_energy(previous: float, trial: float, step: float) -> tuple[bool, float]:
    """
    Compare a trial energy with the energy before the step. Returns (raised, new_step). A trial more than 1e-4 above previous halves step. No surface is evaluated.
    """

def trust_update(radius: float, predicted: float, actual: float, step_norm: float, sigma_inc: float, sigma_dec: float, rho_inc: float, rho_dec: float, delta_min: float) -> float:
    """
    Trust radius from a predicted energy change and the change that occurred. actual is the trial energy minus the energy before the step. A prediction below 1e-14 in absolute value returns radius. The remaining arguments are the Sella schedule. No surface is evaluated.
    """

class MinMode:
    def __init__(self, x: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu', writable=False)], mode: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu', writable=False)], kind: str = 'dimer', dr: float = 0.001, force_tol: float = 0.001, max_move: float = 0.2, rotation_tol: float = 0.0001, max_rotations: int = 20, krylov_dim: int = 12, rotation_stop_length: float = 0.0) -> None:
        """
        Dimer or Lanczos minimum-mode session. Lengths are the caller's. A positive rotation_stop_length stops a rotation once |C| * rotation_stop_length * |sin w| is at most force_tol (Kastner and Sherwood 2008); 0 turns that stop off.
        """

    def estimate(self, surface: Callable[..., tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]]) -> SessionReport:
        """Refresh the lowest mode. surface(x) is one image shaped (1, 3N)."""

    def step(self, surface: Callable[..., tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]]) -> SessionReport:
        """
        Estimate the mode, invert the force, and take one translation. A converged step certifies the saddle index; with escape_attempts set, a step that pushes off a saddle of index two or more reports status 0.
        """

    def set_position(self, x: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu', writable=False)], gradient: object | None = None) -> None:
        """
        Move to a new point. gradient, when given, skips the next centre evaluation.
        """

    def position(self) -> NDArray[numpy.float64]:
        """Copy of the current position."""

    def mode(self) -> NDArray[numpy.float64]:
        """Copy of the current mode."""

    def saddle_index(self) -> SaddleIndex:
        """
        Certificate of the current position (ABI minor 40). On by default; a push or a move drops it.
        """

    def set_saddle_check(self, enabled: bool = True, escape_attempts: int = 0, escape_push: float = 0.1, lowest: int = 4, max_actions: int = 100, dr: float = 0.001, residual_tol: float = 0.01, negative_tol: float = 0.001) -> None:
        """
        Replace the saddle check. enabled=False turns it off. escape_attempts above 0 pushes a converged point of index two or more by escape_push, the largest per-atom displacement, doubled on each later attempt, along its extra negative modes and searches again.
        """

class SellaSaddle:
    def __init__(self, x: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu', writable=False)], masses: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu', writable=False)], order: int = 1, delta: float = 0.1, force_tol: float = 0.001) -> None:
        """
        Order-1 Sella saddle search. masses has one entry per atom, in the caller's mass unit. delta is the initial trust radius per free coordinate.
        """

    def step(self, surface: Callable[..., tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]]) -> SessionReport:
        """
        One partitioned-RFO trust step. surface(x) is one image shaped (1, 3N).
        """

    def position(self) -> NDArray[numpy.float64]:
        """Copy of the current Cartesian position."""

class OhtstBasin:
    """Basin density from an external hyperplane sampler."""

    @property
    def density(self) -> float:
        """Z_H / Z_R on the plane through the minimum."""

    @property
    def density_error(self) -> float:
        """Block standard error of the density."""

    @property
    def quadrature_error(self) -> float:
        """Leave-one-out estimate of the interpolation error of ln(density)."""

    @property
    def tail(self) -> float:
        """Fraction of the reciprocal density in the extrapolated tails."""

    @property
    def log_density(self) -> float:
        """Natural log of the density."""

    @property
    def log_density_error(self) -> float:
        """Standard error of the log density."""

    @property
    def spread(self) -> float:
        """Harmonic width used for the plane spacing."""

    @property
    def harmonic_density(self) -> float:
        """Harmonic density 1 / sqrt(2 pi sigma^2)."""

    @property
    def evaluations(self) -> int:
        """Records the engine returned, summed over planes."""

    @property
    def positive_end(self) -> int:
        """How the positive side ended: 0 cap, 1 turn, 2 width."""

    @property
    def negative_end(self) -> int:
        """How the negative side ended: 0 cap, 1 turn, 2 width."""

class OhtstSample:
    """Langevin sample of one hyperplane: centroid, normal force, virial."""

    @property
    def centroid(self) -> NDArray[numpy.float64]:
        """Centroid of the retained records."""

    @property
    def grad_normal(self) -> NDArray[numpy.float64]:
        """Mean gradient of the normal force in the plane."""

    @property
    def normal_force(self) -> float:
        """Mean force along the normal."""

    @property
    def normal_force_error(self) -> float:
        """Block standard error of the normal force."""

    @property
    def virial(self) -> float:
        """In-plane virial."""

    @property
    def virial_error(self) -> float:
        """Block standard error of the virial."""

    @property
    def acceptance(self) -> float:
        """Accepted fraction of the proposals."""

    @property
    def records(self) -> int:
        """Records inside the blocks."""

    @property
    def evaluations(self) -> int:
        """Surface evaluations."""

class OhtstOptimum:
    """Last accepted plane of a free-energy climb."""

    @property
    def point(self) -> NDArray[numpy.float64]:
        """A point on the plane."""

    @property
    def normal(self) -> NDArray[numpy.float64]:
        """Unit normal, pointing toward the product."""

    @property
    def iterations(self) -> int:
        """Outer trust-region iterations."""

    @property
    def evaluations(self) -> int:
        """Surface evaluations."""

    @property
    def normal_force(self) -> float:
        """Mean force along the last normal."""

    @property
    def normal_force_error(self) -> float:
        """Block standard error of that force."""

class OhtstMbar:
    """MBAR free energies, covariance and overlap."""

    @property
    def free_energies(self) -> NDArray[numpy.float64]:
        """Reduced free energy of every state, with the first at zero."""

    @property
    def covariance(self) -> NDArray[numpy.float64]:
        """Asymptotic covariance, one row and column per state."""

    @property
    def overlap(self) -> NDArray[numpy.float64]:
        """Overlap matrix. Each row sums to one."""

class OhtstWork:
    """Reversible work from the first plane to the last."""

    @property
    def work(self) -> float:
        """Work by the requested estimate."""

    @property
    def error(self) -> float:
        """Standard error of that work."""

    @property
    def integrated_work(self) -> float:
        """Work by thermodynamic integration."""

    @property
    def integrated_error(self) -> float:
        """Standard error of the integral."""

    @property
    def bar_work(self) -> float:
        """Work by BAR, or NaN when no configurations were kept."""

    @property
    def bar_error(self) -> float:
        """Standard error of the BAR work, or NaN."""

    @property
    def mbar_work(self) -> float:
        """Work by MBAR, or NaN when MBAR is off."""

    @property
    def mbar_error(self) -> float:
        """Standard error of the MBAR work, or NaN."""

    @property
    def evaluations(self) -> int:
        """Surface evaluations over the nodes."""

class OhtstTransmission:
    """Transmission coefficient counted at the first crossing."""

    @property
    def kappa(self) -> float:
        """Transmission coefficient."""

    @property
    def error(self) -> float:
        """Block standard error of kappa."""

    @property
    def shots(self) -> int:
        """Shots whose halves both reached a state."""

    @property
    def unassigned(self) -> int:
        """Shots left out."""

    @property
    def reactive(self) -> int:
        """Shots from the reactant into the product."""

    @property
    def recrossed(self) -> int:
        """Backward halves that crossed the plane again."""

    @property
    def evaluations(self) -> int:
        """Surface evaluations."""

class OhtstRate:
    """OH-TST rate from a reactant density and a reversible work."""

    @property
    def rate(self) -> float:
        """The rate, in the time unit of the mass-weighted surface."""

    @property
    def error(self) -> float:
        """First-order standard error of the rate."""

    @property
    def log_rate(self) -> float:
        """Natural log of the rate."""

    @property
    def log_rate_error(self) -> float:
        """Standard error of the log rate."""

    @property
    def flux(self) -> float:
        """One-way flux factor, square root of kT over 2 pi."""

def ohtst_basin_density(engine: Callable[..., tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]], minimum: numpy.typing.NDArray[numpy.float64], normal: numpy.typing.NDArray[numpy.float64], thermal: float, spread: float, burn_in: int, stride: int, records: int, blocks: int, seed: int, refinements: int = 8) -> OhtstBasin:
    """
    Basin density from an engine. engine(point, normal, thermal, seed, thermalize, stride, samples, start) returns (constraint_force, positions, projected_force). spread is the harmonic width.
    """

def ohtst_sample(surface: Callable[..., tuple[float, numpy.typing.NDArray[numpy.float64]]], point: numpy.typing.NDArray[numpy.float64], normal: numpy.typing.NDArray[numpy.float64], thermal: float, time_step: float, friction: float, burn_in: int, stride: int, records: int, blocks: int, seed: int) -> OhtstSample:
    """
    Sample one hyperplane. surface(q) returns (energy, gradient) for a point of length n.
    """

def ohtst_optimize(surface: Callable[..., tuple[float, numpy.typing.NDArray[numpy.float64]]], point: numpy.typing.NDArray[numpy.float64], normal: numpy.typing.NDArray[numpy.float64], thermal: float, time_step: float, friction: float, burn_in: int, stride: int, records: int, blocks: int, seed: int, max_iterations: int, max_translation: float, max_rotation: float, patience: int = 3, translate: bool = True, rotate: bool = True) -> OhtstOptimum:
    """Climb the plane free energy. surface(q) returns (energy, gradient)."""

def ohtst_mbar(reduced: numpy.typing.NDArray[numpy.float64], counts: numpy.typing.NDArray[numpy.int64], initial: numpy.typing.NDArray[numpy.float64] | None = None) -> OhtstMbar:
    """
    Solve MBAR. reduced is (samples, states); counts has one positive entry per state.
    """

def ohtst_work(surface: Callable[..., tuple[float, numpy.typing.NDArray[numpy.float64]]], points: numpy.typing.NDArray[numpy.float64], normals: numpy.typing.NDArray[numpy.float64], thermal: float, time_step: float, friction: float, burn_in: int, stride: int, records: int, blocks: int, seed: int, nodes: int, bar_samples: int, start: numpy.typing.NDArray[numpy.float64] | None = None, mbar: bool = False, estimate: int = 0) -> OhtstWork:
    """
    Reversible work along planes. estimate is 0 for integration, 1 for BAR, 2 for MBAR. nodes must be odd when Simpson weights are used.
    """

def ohtst_transmission(surface: Callable[..., tuple[float, numpy.typing.NDArray[numpy.float64]]], point: numpy.typing.NDArray[numpy.float64], normal: numpy.typing.NDArray[numpy.float64], starts: numpy.typing.NDArray[numpy.float64], thermal: float, time_step: float, friction: float, steps: int, check_every: int, shots: int, blocks: int, seed: int) -> OhtstTransmission:
    """
    Shoot from starts on the plane. Friction zero is Newtonian. surface(q) returns (energy, gradient).
    """

def ohtst_rate(thermal: float, density: float, density_error: float, work: float, work_error: float) -> OhtstRate:
    """
    k = sqrt(kT / 2 pi) * density * exp(-work / kT). density is the reactant density on the first plane.
    """

def raise_status(code: int) -> bool:
    """
    Map an rgsaddle_status_t value. The message is the stored text, or the status name when that text is empty. The stored text is cleared. True on OK. False is not used: convergence is not a status code.
    """
