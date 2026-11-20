# Gas-phase ammonia borane

Section III A uses ORCA with B3LYP/def2-SV(P) and the geometrical
counterpoise correction. The inputs reconstruct this protocol. The staggered
reactant starts from gas-phase structural parameters; the eclipsed saddle
and separated product are coordinate guesses. All files have the same atom
order, with atoms 3 and 6 (one-based) forming H₂.

From the `orca/` directory, run the reactant and product optimizations
(`01_` and `03_`) using the full path to the ORCA executable. The frequency
inputs read those optimized XYZ files. The NEB-TS input reads both minima.
The alternative OptTS input starts directly from the eclipsed geometry.
Check for six rigid-body modes and one unstable internal mode at the saddle,
and verify that displacements along it lead to the indicated reactant and
products. The reference zero-point-corrected activation energy is 1.50 eV.

Start the ten-image LI-NEB energy ladder 0.05 eV below the electronic saddle
and lower it over roughly 1 eV. The paper uses 240 ring-polymer replicas.
Compare the reactant-normalized HQTST rate in inverse seconds with
`../figure4_hqtst.csv`. The harmonic crossover is reported as 330 K in the
text and 333 K in the caption. Both can be compared with the saddle's
imaginary frequency.
