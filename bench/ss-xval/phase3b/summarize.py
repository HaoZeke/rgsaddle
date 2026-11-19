#!/usr/bin/env python3
"""Recompute the phase-3b report from runs-3b.csv and relax.csv."""

import csv
import hashlib
import math
import sys
from collections import defaultdict
from pathlib import Path


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def as_float(text):
    try:
        value = float(text)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def median_iqr(texts):
    finite = [value for value in (as_float(text) for text in texts) if math.isfinite(value)]
    if not finite:
        return math.nan, math.nan
    return percentile(finite, 0.5), percentile(finite, 0.75) - percentile(finite, 0.25)


def num_text(value):
    if not math.isfinite(value):
        return ""
    nearest = round(value)
    if value == 0:
        return "0"
    if nearest != 0 and abs(value - nearest) <= 1e-9 * abs(nearest):
        return str(int(nearest))
    return f"{value:.10g}"


def pair(median, iqr):
    if not math.isfinite(median):
        return ""
    return f"{num_text(median)} ({num_text(iqr)})"


def machine_hash():
    path = Path("/etc/machine-id")
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    root = Path(__file__).resolve().parent
    rows = load(root / "runs-3b.csv")
    relax = load(root / "relax.csv")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["case"], row["code"], row["optimizer"], row["tolerance"])].append(row)

    fields = [
        "iterations",
        "force_calls",
        "wall_s",
        "native_max_force",
        "native_barrier",
        "common_barrier",
        "common_saddle_volume",
        "common_saddle_max_atomic",
        "common_saddle_max_stress",
        "min_perp_width",
        "saddle_index",
    ]
    facts = {}
    for key, items in groups.items():
        returned = [item for item in items if math.isfinite(as_float(item.get("common_barrier", "")))]
        pool = returned or items
        stats = {name: median_iqr(item.get(name, "") for item in pool) for name in fields}
        facts[key] = {
            "n": len(items),
            "conv": sum(item.get("converged") == "1" for item in items),
            "force_pass": sum(item.get("pass_force") == "1" for item in items),
            "stress_pass": sum(item.get("pass_stress") == "1" for item in items),
            "stats": stats,
            "statuses": sorted({item.get("status", "") for item in items}),
            "items": items,
        }

    def stat(case, code, optimizer, tolerance, name):
        return facts[(case, code, optimizer, tolerance)]["stats"][name]

    lines = ["# Phase 3b solid-state band", ""]
    ident = machine_hash()
    if ident:
        lines.append(f"Machine id {ident}.")
        lines.append("")
    lines.append(
        "Six repeats, one core, one thread. The table is the median and, in parentheses, "
        "the linear-percentile interquartile range. Force calls and wall time use every repeat. "
        "A barrier, force, stress, width, or saddle index uses the repeats that returned a band. "
        "A blank cell has no finite value."
    )
    lines.append("")
    lines.append(
        "eOn force calls are the sum of forceCallCounter on the distinct potentials held by "
        "the images, minus the parent's count from before the band. A potential that keeps a "
        "neighbor table is cloned for each interior image, so the parent counter alone counts "
        "only the endpoints."
    )
    lines.append("")
    lines.append(
        "rgsaddle checks the perpendicular width after every accepted band step. eOn exposes "
        "only compute(), so its width is the initial path and the cells compute() returns. "
        "CP2K is checked on the final replica cells. A width below 2 times the cutoff is "
        "fail_cell_width and is not rerun. No returned row recorded that status."
    )
    lines.append("")
    lines.append(
        "Tolerance 1e-3 is the rgsaddle maximum component. Tolerance 0.01 is the eOn force "
        "norm. Each of those two codes is run at both values, as separate rows. CP2K keeps "
        "its default band convergence and is labelled native. The spring is 5 eV/A^2. "
        "rgsaddle climbing starts at a force factor of 0.5. eOn climbing is 0.5 except on "
        "the Bain path, where it is off. The iteration cap is 2000. FIRE uses the library "
        "time step. The common frame is one potential evaluation of the returned images: "
        "enthalpy is energy plus pressure times volume, the saddle is the highest interior "
        "enthalpy, and the atomic limit is 1e-3 eV/A. The stress limit is 1e-3 times the "
        "cube root of the saddle volume, in eV/A^3."
    )
    lines.append("")
    header = (
        "| case | code | optimizer | tolerance | converged | iterations | force calls | wall s "
        "| native force | native barrier | common barrier | volume | final force | final stress "
        "| min width | saddle | force pass | stress pass |"
    )
    lines.append(header)
    lines.append("|" + " --- |" * 18)
    for key in sorted(facts):
        fact = facts[key]
        cells = [f"{fact['conv']}/{fact['n']}"]
        for name in fields:
            median, iqr = fact["stats"][name]
            cells.append(pair(median, iqr))
        cells.append(str(fact["force_pass"]))
        cells.append(str(fact["stress_pass"]))
        lines.append("| " + " | ".join([*key, *cells]) + " |")

    lines.append("")
    lines.append("## Argon vacancy")
    lines.append("")
    rg_calls = stat("ar-vacancy", "rgsaddle", "fire", "1e-3", "force_calls")[0]
    eon_calls = stat("ar-vacancy", "eon", "fire", "0.01", "force_calls")[0]
    rg_barrier = stat("ar-vacancy", "rgsaddle", "fire", "1e-3", "common_barrier")[0]
    lines.append(
        "The path is a nearest-neighbor hop into a vacancy in a 2x2x2 fcc cell at the "
        "shifted-LJ minimum (cutoff 4.8 A, guard 9.6 A, pressure 0). Every returned "
        "argon band has a positive common barrier."
    )
    lines.append("")
    if rg_barrier > 0 and eon_calls:
        lines.append(
            f"rgsaddle FIRE at 1e-3 over eOn FIRE at 0.01 force calls: "
            f"{rg_calls / eon_calls:.10g} ({num_text(rg_calls)} / {num_text(eon_calls)})."
        )
    matched_rg = stat("ar-vacancy", "rgsaddle", "fire", "1e-3", "force_calls")[0]
    matched_eon = stat("ar-vacancy", "eon", "fire", "1e-3", "force_calls")[0]
    loose_rg = stat("ar-vacancy", "rgsaddle", "fire", "0.01", "force_calls")[0]
    loose_eon = stat("ar-vacancy", "eon", "fire", "0.01", "force_calls")[0]
    lines.append(
        f"At tolerance 1e-3 the same ratio is {matched_rg / matched_eon:.10g} "
        f"({num_text(matched_rg)} / {num_text(matched_eon)}). "
        f"At tolerance 0.01 it is {loose_rg / loose_eon:.10g} "
        f"({num_text(loose_rg)} / {num_text(loose_eon)})."
    )
    lines.append("")
    tight = [
        stat("ar-vacancy", "rgsaddle", "fire", "1e-3", "common_barrier")[0],
        stat("ar-vacancy", "eon", "fire", "1e-3", "common_barrier")[0],
        stat("ar-vacancy", "eon", "lbfgs", "1e-3", "common_barrier")[0],
    ]
    volumes = [
        stat("ar-vacancy", "rgsaddle", "fire", "1e-3", "common_saddle_volume")[0],
        stat("ar-vacancy", "eon", "fire", "1e-3", "common_saddle_volume")[0],
        stat("ar-vacancy", "eon", "lbfgs", "1e-3", "common_saddle_volume")[0],
    ]
    lines.append(
        f"At tolerance 1e-3 the three returned barriers span {max(tight) - min(tight):.10g} eV "
        f"and the volumes span {max(volumes) - min(volumes):.10g} A^3. The saddle index is "
        f"{num_text(stat('ar-vacancy', 'rgsaddle', 'fire', '1e-3', 'saddle_index')[0])}. "
        f"The smallest returned width is "
        f"{num_text(min(stat('ar-vacancy', code, opt, tol, 'min_perp_width')[0] for code, opt, tol in (('rgsaddle','fire','1e-3'),('rgsaddle','fire','0.01'),('eon','fire','0.01'),('eon','fire','1e-3'),('eon','lbfgs','0.01'),('eon','lbfgs','1e-3'),('cp2k','diis','native'))))} A."
    )
    lines.append("")
    lines.append(
        "Rows at 0.01, and the CP2K row, miss the common-frame atomic limit. Rows at 1e-3 meet "
        "it. Every argon row meets the stress limit."
    )

    lines.append("")
    lines.append("## Tersoff")
    lines.append("")
    lines.append(
        "Both endpoints were relaxed with the Sella minimum session at maximum-component "
        "tolerance 1e-3 and trust delta 0.2 before the band was built. The session residual "
        "and the common-frame residual are both recorded. The cell is fixed, so the stress "
        "cannot relax."
    )
    lines.append("")
    for row in relax:
        wall = f"{row['wall_s']} s" if row["wall_s"] else "not retained"
        lines.append(
            f"{row['endpoint']}: status {row['status']}, iterations {row['iterations']}, "
            f"force calls {row['force_calls']}, session max component {row['session_max_force']} eV/A, "
            f"common-frame max component {row['max_force']} eV/A, "
            f"max enthalpy stress {row['max_stress']} eV/A^3, limit {row['stress_limit']}, "
            f"wall {wall}."
        )
    lines.append("")
    t_force = facts[("tersoff", "rgsaddle", "fire", "1e-3")]
    t_loose = facts[("tersoff", "rgsaddle", "fire", "0.01")]
    lines.append(
        f"After that relaxation, rgsaddle FIRE at 1e-3 returns {t_force['conv']}/{t_force['n']} "
        f"and passes force on {t_force['force_pass']} and stress on {t_force['stress_pass']}. "
        f"At 0.01 it passes force on {t_loose['force_pass']} and stress on {t_loose['stress_pass']}. "
        "The 0.01 frames that miss the atomic limit are stored with the logs as "
        "tersoff-rgsaddle-miss/0.01/rep-0 through rep-5, each with final.geom, common.txt, "
        "result.txt, and config.txt."
    )
    lines.append("")
    lines.append(
        "eOn FIRE and L-BFGS reach the iteration cap at both tolerances. Their force-call "
        f"medians are {num_text(stat('tersoff','eon','fire','0.01','force_calls')[0])} and "
        f"{num_text(stat('tersoff','eon','lbfgs','0.01','force_calls')[0])}. "
        "The two tolerances match because the cap is reached first. "
        "CP2K returns a barrier in line with rgsaddle and misses the atomic limit."
    )

    lines.append("")
    lines.append("## Vacancy")
    lines.append("")
    diis = [item for item in facts[("vacancy-vc", "cp2k", "diis", "native")]["items"]]
    diis_calls = [item["force_calls"] for item in sorted(diis, key=lambda item: int(item["repeat"]))]
    lines.append(
        "CP2K DIIS does not return. The harness stops a line search that keeps rewriting one "
        "energy, so the force calls are the counts at the stop, in repeat order: "
        + ", ".join(diis_calls)
        + f". The median is {pair(*stat('vacancy-vc','cp2k','diis','native','force_calls'))}."
    )
    lines.append("")
    lines.append(
        "CG and BFGS were submitted as band optimizer rows. Every repeat exits with status "
        "exit -6. The band output on each repeat says invalid value for enumeration, for CG "
        "and for BFGS. This build's band optimizer accepts MD and DIIS. The DIIS row is kept."
    )
    lines.append("")
    lines.append(
        "eOn at 0.01 misses the atomic limit and meets the stress limit. eOn at 1e-3 meets both. "
        "rgsaddle meets both at both tolerances. The returned barriers agree."
    )

    lines.append("")
    lines.append("## Bain")
    lines.append("")
    bain_force = facts[("bain-tet", "eon", "lbfgs", "0.01")]
    bain_tight = facts[("bain-tet", "eon", "lbfgs", "1e-3")]
    lines.append(
        f"eOn L-BFGS at 0.01 passes the atomic test on {bain_force['force_pass']} of "
        f"{bain_force['n']}. The median final atomic force is "
        f"{num_text(stat('bain-tet','eon','lbfgs','0.01','common_saddle_max_atomic')[0])} eV/A. "
        f"The same optimizer at 1e-3 passes on {bain_tight['force_pass']} of {bain_tight['n']}, "
        f"with median final atomic force "
        f"{num_text(stat('bain-tet','eon','lbfgs','1e-3','common_saddle_max_atomic')[0])} eV/A. "
        "The 0/6 result is the 0.01 setting. Stress passes on every Bain row."
    )
    lines.append("")
    lines.append(
        "At 0.01, CP2K and eOn FIRE place the saddle on image "
        f"{num_text(stat('bain-tet','cp2k','diis','native','saddle_index')[0])}, while rgsaddle "
        f"and eOn L-BFGS place it on image "
        f"{num_text(stat('bain-tet','rgsaddle','fire','0.01','saddle_index')[0])}. "
        "The enthalpies agree and the volumes do not, because the winning image differs. "
        "At 1e-3, eOn FIRE also places the saddle on image "
        f"{num_text(stat('bain-tet','eon','fire','1e-3','saddle_index')[0])}."
    )
    lines.append("")
    text = "\n".join(lines) + "\n"
    if len(sys.argv) > 1 and sys.argv[1] == "--write":
        (root / "REPORT.md").write_text(text)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
