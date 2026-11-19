# Solid-state saddle cross-validation, phase 3

Builds reused from the earlier scout and phase 2, release, one thread, affinity on CPU 0.

- CP2K a942a44f18d07e1b47cddea61c678c9ee74c0ee5
- rgsaddle library f6a2f1e4d56d6c8bd5147aaf0f3b0c9231b3fdcc
- eOn 4ee85b89e55b2ddea83f3e60b202055ef31d5bf5
- rgpot e40fd175f5230098d56333bf2aff38cd839dc810
- readcon-core 6183e294d14547a140b1342c42514efcfe6faece

Six repeats. Median and the linear-percentile interquartile range, with the quartile position `(n-1)*fraction`. A blank cell was not finite. Counts are how many of the six repeats passed. Pass and fail below are the common frame, plus the width guard.

## Optimizer settings

rgsaddle FIRE: method 0, max move 0.2, force tolerance 1e-3, time step 1 as set by the C parser, flags CI_AFTER and NO_MIN_MODE, improved tangent, uniform spring, NEB projection, cell weight 1.

eOn FIRE: OptType FIRE on the band and the optimizer, max move 0.2, force tolerance 0.01, convergence metric norm. Parameters() resolves time_step_input 1 through the time unit. That resolved step is left in place.

eOn L-BFGS: OptType LBFGS on the band and the optimizer, max move 0.2, force tolerance 0.01. L-BFGS does not use the FIRE time step.

CP2K: DIIS, MAX_STEPS 2000, default line search, default MAX_DR 0.0002, MAX_FORCE 0.00045, RMS_DR 0.0001, RMS_FORCE 0.00030 in internal units. ALIGN_FRAMES and ROTATE_FRAMES are off.

Climbing image: rgsaddle trigger force 0.5 eV/A on every case. eOn trigger is infinity on bain-tet and ar-replica, and 0.5 eV/A on vacancy-vc and tersoff, plus the requirement that the highest interior enthalpy exceed both endpoints. CP2K NSTEPS_IT is 5 on bain-tet, vacancy-vc, and tersoff, and 0 on ar-replica. NSTEPS_IT is not a force threshold.

Springs: bain-tet uses 5 eV/A^2 in rgsaddle and eOn and 0.02 Ha/Bohr^2 in CP2K. vacancy-vc and tersoff use 5 eV/A^2, which is 5.1454291598313009e-02 Ha/Bohr^2. ar-replica uses 0.02 Ha/Bohr^2, which is 1.94347249 eV/A^2.

Iteration cap 2000. Order is case, then repeat, then CP2K, rgsaddle, eOn L-BFGS, eOn FIRE.

## Cases

bain-tet: 3x3x3 shifted Lennard-Jones, cutoff 1.2, epsilon = sigma = 1, pressure 0, 54 atoms, 9 images. Endpoints are tetragonal, a0 = 1.13125, c0 = 1.48125 and a1 = 1.35, c1 = 1.1375. The ideal lattice has one interior energy maximum, 10.398 eV above the higher endpoint. Guard 2.4 A. The cell is free in all three codes.

vacancy-vc: the 31-atom hop on the cubic cell of side 2*2^(2/3), cutoff 1.5, pressure 0, with the cell free. The ideal path rises 10.359 eV. Guard 3.0 A. The initial perpendicular width is 3.1748 A. All three codes accept a cell degree of freedom for this potential.

ar-replica: 3x3x3 copies of the 10 A cube, side 30 A, 27 dimers, 54 atoms. Each dimer sits at 2^(1/6)*3.405 A and rotates from x to y, so the ideal path compresses every dimer and rises 13.658 eV. This is not a tiling of the earlier monotonic shear. Pressure 50 bar, cutoff 4.8 A, guard 9.6 A. The cell is free.

tersoff: diamond conventional cell 5.431 A, 2x2x2, the origin atom removed and the (0.25, 0.25, 0.25) neighbor moved to the origin. 63 atoms, fixed cell, pressure 0, outer cutoff 3.0 A, guard 6.0 A. The ideal path rises 6.358 eV. rgpot Tersoff publishes energy and forces and does not publish stress, so eOn refuses a solid-state band and the cell is not a degree of freedom in any of the three codes. The common-frame cell derivative is a central difference of the energy at fixed fractional coordinates, step 1e-6. CP2K lambda3 is 0. The remaining parameters are the rgpot Si(C) numbers: A 1830.8, B 471.18, lambda1 2.4799, lambda2 1.7322, alpha 0, beta 1.1e-6, n 0.78734, c 100390, d 16.217, h -0.59825, bigR 2.85, bigD 0.15.

Initial-image parity, one rgpot build against one CP2K energy and force evaluation, before any band: bain-tet energy span 2.4e-6 eV, vacancy-vc 1.1e-5 eV, ar-replica 1.8e-7 eV, tersoff 2.5e-5 eV. Force spans are below 3e-6 eV/A and the tersoff force span is 1e-7 eV/A. The tersoff finite-difference stress matches the CP2K stress to 3e-8 eV/A^3. The analytic cell derivative on the Lennard-Jones cases disagrees with the finite difference by at most 3.4e-8 eV/A.

## Cell-width guard

Perpendicular widths are the volume divided by each face area. rgsaddle checks all three widths of every image after every optimizer step and stops that row at the first image below 2*rc. It does not restart. eOn runs the band inside one compute() call and does not return the cell between optimizer steps, so the eOn guard is the initial path plus the band compute() returns. CP2K writes one lattice per replica in the final band. Its per-step trajectory takes the force-environment cell once, before the replica loop, so that file is not a per-image record and is not used as the guard. No completed row fell below its guard.

## Common frame

One rgpot build re-evaluates every returned image. H = E + pV. The barrier is H at the highest interior image minus H at the initial image. The atomic test is the maximum component at that image, limit 1e-3 eV/A. The stress test is max |sigma + p I| against 1e-3 * V^(1/3). The product is applied as written; the two factors do not share units. Barrier agreement is 1e-4 eV and saddle volume agreement is 1e-3 A^3 among rows that converged.

## Rows

Each numeric cell is the median, with the interquartile range in parentheses. Wall time is seconds. Barriers are eV. Volume is A^3. Force is eV/A. Stress is eV/A^3. Width is the minimum perpendicular width in A.

| case | code | optimizer | converged | iterations | force calls | wall | native barrier | common barrier | volume | saddle force | saddle stress | min width | force pass | stress pass | width pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bain-tet | cp2k | diis | 6/6 | 39 (0) | 756 (0) | 0.5701 (0.0134) | 3.61512 (0) | 3.61512 (0) | 57.4958 (0) | 3.81e-08 (0) | 8.15e-05 (0) | 3.394 (0) | 6 | 6 | 6 |
| bain-tet | rgsaddle | fire | 6/6 | 121 (0) | 975 (0) | 0.3459 (0.0104) | 3.61512 (0) | 3.61512 (0) | 58.1392 (0) | 1.80e-10 (0) | 2.15e-05 (0) | 3.394 (0) | 6 | 6 | 6 |
| bain-tet | eon | lbfgs | 6/6 | 110 (0) | 772 (0) | 0.2949 (0.0038) | 3.61512 (0) | 3.61512 (0) | 58.2066 (0) | 2.22e-03 (0) | 2.75e-05 (0) | 3.394 (0) | 0 | 6 | 6 |
| bain-tet | eon | fire | 6/6 | 78 (0) | 548 (0) | 0.2048 (0.0067) | 3.61512 (0) | 3.61512 (0) | 57.3910 (0) | 1.90e-04 (0) | 2.37e-04 (0) | 3.394 (0) | 6 | 6 | 6 |
| vacancy-vc | cp2k | diis | 0/6 | 1 (0) | 1.386e+05 (7311) | 20.00 (0.003) | 5.62096 (0) | | | | | | 0 | 0 | 0 |
| vacancy-vc | rgsaddle | fire | 6/6 | 136 (0) | 1087 (0) | 0.1663 (0.0097) | 4.74763 (0) | 4.74763 (0) | 32.3390 (0) | 9.24e-06 (0) | 1.97e-05 (0) | 3.158 (0) | 6 | 6 | 6 |
| vacancy-vc | eon | lbfgs | 6/6 | 42 (0) | 296 (0) | 0.0542 (0.0049) | 4.74763 (0) | 4.74763 (0) | 32.3390 (0) | 4.11e-03 (0) | 3.61e-04 (0) | 3.160 (0) | 0 | 6 | 6 |
| vacancy-vc | eon | fire | 6/6 | 160 (0) | 1122 (0) | 0.1861 (0.0159) | 4.74763 (0) | 4.74763 (0) | 32.3390 (0) | 6.68e-03 (0) | 2.77e-05 (0) | 3.160 (0) | 0 | 6 | 6 |
| ar-replica | cp2k | diis | 6/6 | 200 (0) | 4995 (0) | 10.00 (0.0002) | -0.0208499 (0) | -0.0208498 (0) | 26330.1 (0) | 3.28e-04 (0) | 3.12e-05 (0) | 29.61 (0) | 6 | 6 | 6 |
| ar-replica | rgsaddle | fire | 6/6 | 1540 (0) | 11780 (0) | 3.501 (0.110) | -0.0601354 (0) | -0.0587148 (0) | 25117.4 (0) | 3.83e-04 (0) | 3.12e-05 (0) | 20.93 (0) | 6 | 6 | 6 |
| ar-replica | eon | lbfgs | 6/6 | 749 (0) | 5245 (0) | 1.648 (0.065) | -0.245775 (0) | -0.222655 (0) | 19864.5 (0) | 3.14e-04 (0) | 3.12e-05 (0) | 21.36 (0) | 6 | 6 | 6 |
| ar-replica | eon | fire | 6/6 | 321 (0) | 2249 (0) | 0.7088 (0.059) | -0.240048 (0) | -0.217975 (0) | 20014.4 (0) | 3.27e-04 (0) | 3.12e-05 (0) | 21.97 (0) | 6 | 6 | 6 |
| tersoff | cp2k | diis | 0/6 | | | 10.00 (0.0001) | | | | | | | 0 | 0 | 0 |
| tersoff | rgsaddle | fire | 6/6 | 443 (0) | 3483 (0) | 0.4171 (0.021) | 1.59642 (0) | 1.59642 (0) | 1281.53 (0) | 1.48e-03 (0) | 1.09e-02 (0) | 10.86 (0) | 0 | 0 | 6 |
| tersoff | eon | lbfgs | 0/6 | 2000 (0) | 2 (0) | 0.8779 (0.019) | 62.1693 (0) | 24771.1 (0) | 1281.53 (0) | 1.14e+03 (0) | 27.0 (0) | 10.86 (0) | 0 | 0 | 6 |
| tersoff | eon | fire | 0/6 | 2000 (0) | 2 (0) | 0.7800 (0.040) | 82.2081 (0) | 26835.2 (0) | 1281.53 (0) | 1.20e+03 (0) | 28.5 (0) | 10.86 (0) | 0 | 0 | 6 |

The vacancy-vc CP2K native barrier is the last completed band step, not a converged saddle. Its six force-call counts are 601438, 138411, 67845, 145350, 135755, and 138869. Those counts are how far the repeated evaluation had been written when the run was stopped. They are not a converged total.

## Agreement

bain-tet: common-barrier spread 5.17e-7 eV over 24 converged repeats, inside 1e-4 eV. Saddle-volume spread 0.816 A^3, outside 1e-3 A^3. CP2K and eOn FIRE place the highest image at index 2. rgsaddle and eOn L-BFGS place it at index 3. Those images differ in the cell, and their enthalpies differ by less than the barrier spread. eOn L-BFGS fails the atomic test at 2.22e-3 eV/A. The other three pass. Stress passes for all four. Every image stayed at or above 3.394 A, against 2.4 A.

vacancy-vc: rgsaddle and both eOn optimizers agree to 1.30e-6 eV and 5.51e-5 A^3. The saddle is image 4 and the volume is 32.339 A^3. Stress passes, so the free cell meets the stress test. eOn L-BFGS fails the atomic test at 4.11e-3 eV/A and eOn FIRE fails it at 6.68e-3 eV/A. rgsaddle passes at 9.24e-6 eV/A. The minimum width is 3.158 A, against 3.0 A. CP2K did not return a band. Inside optimizer step 2 the line search rewrote one geometry, and the energy file never advanced past step 1. The same input was left in place. All three codes can free this cell. CP2K accepts the solid-state band and its line search does not finish.

ar-replica: all 24 repeats converged and passed the atomic test, the stress test, and the width guard. The common barriers are negative. The highest interior image lies below both endpoints, so the optimized band is not a saddle. The spread is 0.202 eV and 6466 A^3. Saddle volumes are 26330 A^3 for CP2K, 25117 A^3 for rgsaddle, 19865 A^3 for eOn L-BFGS, and 20014 A^3 for eOn FIRE, against an initial cube of 27000 A^3. CP2K's highest image is index 4. The other three pick index 1. eOn does not climb when no image is above both endpoints, and its trigger on this case is infinity. The minimum width is 20.93 A, against 9.6 A. The ideal path's interior maximum was not the converged result. The settings were not changed.

tersoff: rgsaddle is the only code that converged. Its barrier is 1.59642 eV and the cell volume is 1281.53 A^3, the fixed cube of side 10.862 A, on all nine images. The saddle force is 1.477e-3 eV/A, which fails 1e-3. The saddle stress is 1.091e-2 eV/A^3 against a limit of 1.086e-2. The cell is not a degree of freedom, so that stress cannot be relaxed. The reactant endpoint itself has an atomic force of 0.502 eV/A. eOn FIRE and L-BFGS reached the iteration cap. The common-frame barriers on the bands they returned are 26835 eV and 24771 eV, with saddle forces near 10^3 eV/A. The runner's force counter stayed at 2. The band builds a separate potential instance for each interior image when the potential keeps a neighbor table, and those instances are not the counter the runner reads. The two calls are the shared endpoint evaluations. CP2K aborted with signal 6 in the neighbor-list distance check during a DIIS line search. One pair reached 0.049 bohr, inside the spline table, and the run stopped with the message that the geometry is wrong or EMAX_SPLINE is too small. All six repeats did this at about 10 s, after the band log had reached optimizer step 46. The input was not changed. Every tersoff image that was returned has perpendicular width 10.862 A, against 6.0 A.

## Minimum width on the first repeat

The nine numbers are the minimum perpendicular width of images 0 through 8, in angstrom, next to 2*rc.

bain-tet, guard 2.4. CP2K: 3.394, 3.523, 3.636, 3.724, 3.806, 3.884, 3.744, 3.576, 3.412. rgsaddle: 3.394, 3.448, 3.558, 3.636, 3.700, 3.783, 3.664, 3.541, 3.412. eOn L-BFGS and eOn FIRE: 3.394, 3.476, 3.558, 3.640, 3.722, 3.799, 3.670, 3.541, 3.412.

vacancy-vc, guard 3.0. CP2K returned no band. rgsaddle: 3.175, 3.163, 3.159, 3.159, 3.158, 3.159, 3.159, 3.163, 3.175. eOn L-BFGS: 3.175, 3.164, 3.161, 3.161, 3.160, 3.161, 3.161, 3.165, 3.175. eOn FIRE: 3.175, 3.165, 3.161, 3.161, 3.160, 3.161, 3.160, 3.164, 3.175.

ar-replica, guard 9.6. CP2K: 30.00, 29.62, 29.61, 29.61, 29.61, 29.61, 29.61, 29.62, 30.00. rgsaddle: 30.00, 29.20, 26.00, 22.51, 20.93, 21.51, 22.68, 26.65, 30.00. eOn L-BFGS: 30.00, 27.03, 23.28, 22.48, 21.36, 22.55, 23.28, 27.03, 30.00. eOn FIRE: 30.00, 27.10, 23.39, 22.72, 21.97, 22.72, 23.39, 27.10, 30.00.

tersoff, guard 6.0. CP2K returned no band. rgsaddle, eOn L-BFGS, and eOn FIRE: 10.862 on every image.

## Held back

EAM, CdSe, and Mo A15 were not run. The variable-cell dimer remains a capability of the rgsaddle library and was not timed.
