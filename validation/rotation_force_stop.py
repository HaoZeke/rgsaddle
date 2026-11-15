"""The curvature in the dimer's rotation force-error stop.

Run: uv run --no-project --with sympy python validation/rotation_force_stop.py

Two modes with curvatures l1 < l2 span the rotation plane. A dimer at
angle w from the lower mode reads the curvature and rotational force

    C = l1 cos^2 w + l2 sin^2 w,    F = (l2 - l1) sin w cos w,

so the curvature it reaches by rotating back to the mode is

    l1 = C - F tan w.

rotation_force_error in src/minmode.rs takes the larger magnitude of C
and C - F tan w, with w the Kastner-Sherwood estimate atan(F/|C|)/2.
Halfway between l1 = -1 and l2 = 1 the dimer reads C = 0 and F = 1: the
estimate is w = pi/4, and C - F tan w = -1 is the lower mode, where the
current curvature alone would give a force error of zero.
"""

if not __debug__:
    raise SystemExit("Symbolic validation requires Python assertions; disable optimization.")

import sympy as sp

w, l1, l2 = sp.symbols("w l1 l2", real=True)
C = l1 * sp.cos(w) ** 2 + l2 * sp.sin(w) ** 2
F = (l2 - l1) * sp.sin(w) * sp.cos(w)
assert sp.simplify(C - F * sp.tan(w) - l1) == 0

half = {l1: -1, l2: 1, w: sp.pi / 4}
c_half, f_half = C.subs(half), F.subs(half)
assert sp.simplify(c_half) == 0 and sp.simplify(f_half) == 1
estimate = sp.atan2(f_half, sp.Abs(c_half)) / 2
assert sp.simplify(estimate - sp.pi / 4) == 0
assert sp.simplify(c_half - f_half * sp.tan(estimate)) == -1
print("rotation force-error curvature ok")
