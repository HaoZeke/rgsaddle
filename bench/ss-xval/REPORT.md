# Solid-state saddle cross-validation

One thread on logical CPU 0. OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS,
BLIS_NUM_THREADS, RAYON_NUM_THREADS, VECLIB_MAXIMUM_THREADS, NUMEXPR_NUM_THREADS,
and GOTO_NUM_THREADS are 1, and OMP_DYNAMIC is FALSE. Repeats are interleaved by code.
The iteration cap is 2000. Six repeats. The median and the interquartile range use the
linear percentile (for six samples, the quartile positions are 1.25 and 3.75).
Pass and fail are decided only after re-evaluating each finished band with one shifted-LJ build:
exact H = E + pV, and the full 3N+9 gradient. The atomic test is the maximum absolute
Cartesian force on the highest-enthalpy interior image. The stress test is the maximum
absolute entry of sigma + p I, compared with 1e-3 times the cube root of that image's volume.
The written product mixes eV/A^3 with 1/A; the numeric comparison is applied as written.
The same re-evaluation also records the maximum absolute entry of dH/dh.

## Revisions and builds

The libraries from the scout were reused. No library was rebuilt.

| component | revision |
|---|---|
| CP2K | `a942a44f18d07e1b47cddea61c678c9ee74c0ee5` |
| rgsaddle | `f6a2f1e4d56d6c8bd5147aaf0f3b0c9231b3fdcc` |
| eOn develop | `4ee85b89e55b2ddea83f3e60b202055ef31d5bf5` |
| rgpot | `e40fd175f5230098d56333bf2aff38cd839dc810` |
| readcon-core | `6183e294d14547a140b1342c42514efcfe6faece` |

CP2K serial ssmp: GCC, G++, and GFortran 13.3.0, CMake 3.28.3, Ninja 1.11.1,
OpenBLAS 0.3.26, FFTW 3.3.10, DBCSR 8e8c7d4, libwignernj 0.8.0.

```sh
taskset -c 0 tools/toolchain/install_cp2k_toolchain.sh -j 1 \
  --mpi-mode=no --math-mode=openblas --target-cpu=native \
  --with-gcc=system --with-cmake=system --with-ninja=system \
  --with-openmpi=no --with-mpich=no --with-intelmpi=no \
  --with-openblas=system --with-fftw=system --with-libxc=no \
  --with-eigen=no --with-libint=no --with-cosma=no --with-libxsmm=no \
  --with-libxs=no --with-libxstream=no --with-scalapack=no \
  --with-elpa=no --with-gsl=no --with-libvori=no --with-spglib=no \
  --with-tblite=no --with-libwignernj=install --with-dbcsr=install
taskset -c 0 ./build_cp2k.sh -j 1 --preset=native-gnu-x86_64
```

rgsaddle: Rust and Cargo 1.88.0.

```sh
taskset -c 0 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  CARGO_BUILD_JOBS=1 cargo +1.88.0 build --release --locked \
  --features capi -j 1
```

eOn: GCC and G++ 13.3.0, Rust 1.88.0, Meson 1.12.1, Python 3.12.3, Eigen 3.4.0,
Cap'n Proto 1.0.1. The reconfigure pins inih-r62 and Highway 1.4.0.

```sh
taskset -c 0 env RUSTUP_TOOLCHAIN=1.88.0 OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
  RAYON_NUM_THREADS=1 CARGO_BUILD_JOBS=1 MESON_NUM_PROCESSES=1 \
  CC=gcc-13 CXX=g++-13 meson setup --wipe build-gcc --buildtype=release
taskset -c 0 env RUSTUP_TOOLCHAIN=1.88.0 OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
  RAYON_NUM_THREADS=1 CARGO_BUILD_JOBS=1 MESON_NUM_PROCESSES=1 \
  CC=gcc-13 CXX=g++-13 meson setup --reconfigure build-gcc \
  --force-fallback-for=INIReader,inireader,hwy,highway
taskset -c 0 env RUSTUP_TOOLCHAIN=1.88.0 OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
  RAYON_NUM_THREADS=1 CARGO_BUILD_JOBS=1 MESON_NUM_PROCESSES=1 \
  CC=gcc-13 CXX=g++-13 meson compile -C build-gcc eonclient -j 1
```

Hardware from the processor identity: Intel Xeon, 8 logical CPUs, one socket,
8 cores per socket, one thread per core. Affinity is logical CPU 0.

## Shared setup

Every code reads one initial path. Endpoints are oriented into the lower-triangular
right-strain chart, then interior images are the linear interpolation of the cell
and of the minimum-image fractional coordinates. Case 1 uses the argon regtest
endpoints, one interior image, shifted Lennard-Jones with epsilon 119.8 K, sigma
3.405 A, cutoff 4.8 A, and 50 bar. Its spring is the regtest value 0.02 Ha/Bohr^2,
which is 1.94347249 eV/A^2 in rgsaddle and eOn.
Case 2 is a two-atom Bain cell tiled on each side. Epsilon and sigma are 1, the
cutoff is 1.2 A, and the shift is the value at the cutoff. The body-centered cell
edge is the nearest-neighbor distance times 2/sqrt(3). The face-centered cell is
tetragonal with basal edge equal to the nearest-neighbor distance and long edge
equal to that distance times sqrt(2). Seven interior images. rgsaddle and eOn use
their default spring 5 eV/A^2. CP2K uses its default 0.02 Ha/Bohr^2, and only the
3x3x3 cell is run there: the 2x2x2 cell is narrower than twice the cutoff, so a
minimum-image potential and a multiple-image list are different potentials.
Case 3 is the 31-atom periodic vacancy in a cubic cell of side 2^(2/3)*2, cutoff
1.5 A, pressure 0, and seven interior images. The spring is 5 eV/A^2 in rgsaddle and
eOn, and 0.051454292 Ha/Bohr^2 in CP2K, the same constant.

Climbing triggers stay at each code's own rule, except case 3, where rgsaddle and
eOn both arm at 0.5 eV/A. rgsaddle case 1 uses its default trigger of 0.5 eV/A.
eOn cases 1 and 2 keep the default trigger at infinity, plus the endpoint-energy test.
CP2K has no force threshold. Its case 1 uses NSTEPS_IT 0, and its cases 2 and 3 use
the default NSTEPS_IT 5.
eOn also requires the climbing image to lie above both endpoints. rgsaddle runs
plain CI-NEB: RGSADDLE_BAND_NO_MIN_MODE is set, so the atomic min-mode refinement
stays off and the method under test is the band. eOn FIRE uses the Parameters
constructor. That constructor resolves the default time_step_input of 1 before FIRE
is built. The phase-2 FIRE rows use that resolved step, max_move 0.2, and the native
force norm 0.01. The scout driver, which leaves the optimizer time step untouched
before FIRE is constructed, still aborts with the message
[FIRE] m_dt is too small: 0.0000. That configuration was left as recorded.
Native tolerances are untouched: rgsaddle maximum component 1e-3, eOn force norm
0.01, and the CP2K default maximum and RMS force and displacement tests.

## Finite-difference check

dE/dh = V h^{-T} sigma, at fixed fractional coordinates, checked by a central
difference of 1e-6 A on the first image of each path.

| case | size | max abs error |
|---|---|---|
| ar-cineb | 1 | 1.885e-11 |
| bain | 2 | 4.440e-10 |
| bain | 3 | 8.463e-14 |
| bain | 4 | 1.377e-13 |
| bain | 5 | 7.105e-09 |
| vacancy | 1 | 7.106e-09 |

## Initial-image parity

CP2K FIST against the same shifted-LJ evaluation. Spans are absolute differences.

| case | size | image | energy span eV | force span eV/A | stress span eV/A^3 |
|---|---|---|---|---|---|
| ar-cineb | 1 | 0 | 2.151770e-09 | 6.463666e-08 | 1.822573e-10 |
| ar-cineb | 1 | 1 | 2.918096e-09 | 3.572514e-08 | 1.032758e-10 |
| ar-cineb | 1 | 2 | 5.588469e-09 | 1.670691e-08 | 4.931582e-11 |
| bain | 2 | 0 | 6.127577e-07 | 6.472004e-28 | 6.214630e-12 |
| bain | 3 | 0 | 2.068057e-06 | 3.288056e-14 | 6.219740e-12 |
| bain | 3 | 8 | 3.102086e-06 | 2.371738e-14 | 1.015437e-11 |
| vacancy | 1 | 0 | 1.074262e-05 | 5.750776e-12 | 1.715511e-11 |
| vacancy | 1 | 8 | 1.074262e-05 | 5.804586e-12 | 1.715715e-11 |

## Native frame

Medians with the interquartile range in parentheses. Force calls, iterations, and wall
time use every repeat. The native barrier is the rise of that code's own objective above
the initial image, so a descent is recorded as 0 for rgsaddle and eOn. CP2K's native
barrier is the exact enthalpy difference from its band output. FIRE is its own row.

| case | size | code | optimizer | converged | iterations | force calls | wall s | native barrier eV | status |
|---|---|---|---|---|---|---|---|---|---|
| ar-cineb | 1 | cp2k | diis | 6/6 | 7 (0) | 108 (0) | 0.08703 (0.004889) | -0.0124184 (0) | converged |
| ar-cineb | 1 | eon | fire | 6/6 | 19 (0) | 21 (0) | 0.004936 (0.0001335) | 0 (0) | converged |
| ar-cineb | 1 | eon | lbfgs | 6/6 | 6 (0) | 8 (0) | 0.005196 (0.000317) | 0 (0) | converged |
| ar-cineb | 1 | rgsaddle | fire | 6/6 | 45 (0) | 49 (0) | 0.003186 (0.0003233) | 0 (0) | converged |
| bain | 2 | eon | fire | 6/6 | 50 (0) | 352 (0) | 0.02047 (0.001341) | 1.94185e-08 (0) | converged |
| bain | 2 | eon | lbfgs | 6/6 | 40 (0) | 282 (0) | 0.01803 (0.001465) | 4.11734e-08 (0) | converged |
| bain | 2 | rgsaddle | fire | 6/6 | 279 (0) | 2179 (0) | 0.09593 (0.002045) | 7.5655e-12 (0) | converged |
| bain | 3 | cp2k | diis | 6/6 | 22 (0) | 450 (0) | 0.3602 (0.0051) | 0 (0) | converged |
| bain | 3 | eon | fire | 6/6 | 85 (0) | 597 (0) | 0.2269 (0.008106) | 2.51884e-08 (0) | converged |
| bain | 3 | eon | lbfgs | 6/6 | 45 (0) | 317 (0) | 0.1256 (0.005765) | 3.50354e-07 (0) | converged |
| bain | 3 | rgsaddle | fire | 6/6 | 1393 (0) | 10677 (0) | 3.884 (0.05746) | 4.92761e-12 (0) | converged |
| bain | 4 | eon | fire | 6/6 | 79 (0) | 555 (0) | 1.11 (0.05094) | 3.32128e-07 (0) | converged |
| bain | 4 | eon | lbfgs | 6/6 | 849 (0) | 5945 (0) | 12.41 (0.6019) | 5.86466e-06 (0) | converged |
| bain | 4 | rgsaddle | fire | 6/6 | 492 (0) | 3817 (0) | 7.258 (0.2296) | 8.66862e-13 (0) | converged |
| bain | 5 | eon | fire | 6/6 | 102 (0) | 716 (0) | 4.898 (0.2276) | 4.40775e-09 (0) | converged |
| bain | 5 | eon | lbfgs | 6/6 | 59 (0) | 415 (0) | 2.884 (0.1268) | 3.7173e-07 (0) | converged |
| bain | 5 | rgsaddle | fire | 6/6 | 498 (0) | 3866 (0) | 26.16 (0.683) | 6.39488e-13 (0) | converged |
| vacancy | 1 | cp2k | diis | 6/6 | 23 (0) | 639 (0) | 0.2991 (0.006426) | 5.05026 (0) | converged |
| vacancy | 1 | eon | fire | 6/6 | 100 (0) | 702 (0) | 0.01275 (0.0002513) | 5.05026 (0) | converged |
| vacancy | 1 | eon | lbfgs | 6/6 | 32 (0) | 226 (0) | 0.008893 (0.0002118) | 5.05026 (0) | converged |
| vacancy | 1 | rgsaddle | fire | 6/6 | 155 (0) | 1324 (0) | 0.02322 (0.000205) | 5.05026 (0) | converged |

## Common frame

The saddle is the interior image of highest exact enthalpy. A row passes when that
image's maximum atomic force is at most 1e-3 eV/A and its maximum |sigma + p I|
is at most 1e-3 V^(1/3).

The last force column is the worst raw atomic force on any interior image.

| case | size | code | optimizer | barrier eV | volume A^3 | saddle atomic | saddle |sigma+pI| | worst interior atomic | pass force | pass stress |
|---|---|---|---|---|---|---|---|---|---|---|
| ar-cineb | 1 | cp2k | diis | -0.0124184 (0) | 839.281 (0) | 0.004066 (0) | 3.121e-05 (0) | 0.004066 (0) | 0/6 | 6/6 |
| ar-cineb | 1 | eon | fire | -0.00681135 (0) | 1018.48 (0) | 0.004297 (0) | 3.121e-05 (0) | 0.004297 (0) | 0/6 | 6/6 |
| ar-cineb | 1 | eon | lbfgs | -0.00693512 (0) | 1017.9 (0) | 0.002142 (0) | 3.121e-05 (0) | 0.002142 (0) | 0/6 | 6/6 |
| ar-cineb | 1 | rgsaddle | fire | -0.0256515 (0) | 419.328 (0) | 0.0004996 (0) | 3.454e-05 (0) | 0.0004996 (0) | 6/6 | 6/6 |
| bain | 2 | eon | fire | 1.94185e-08 (0) | 17.205 (0) | 0.0005047 (0) | 0.0001919 (0) | 0.3791 (0) | 6/6 | 6/6 |
| bain | 2 | eon | lbfgs | 4.11734e-08 (0) | 17.2047 (0) | 0.001794 (0) | 0.0001004 (0) | 0.4184 (0) | 0/6 | 6/6 |
| bain | 2 | rgsaddle | fire | 7.5655e-12 (0) | 17.1388 (0) | 1.807e-10 (0) | 6.234e-06 (0) | 1.442 (0) | 6/6 | 6/6 |
| bain | 3 | cp2k | diis | 1.58647e-09 (0) | 57.5246 (0) | 2.704e-07 (0) | 5.082e-05 (0) | 0.002735 (0) | 6/6 | 6/6 |
| bain | 3 | eon | fire | 2.51884e-08 (0) | 57.4701 (0) | 7.456e-05 (0) | 0.0002008 (0) | 0.01251 (0) | 6/6 | 6/6 |
| bain | 3 | eon | lbfgs | 3.50354e-07 (0) | 58.0649 (0) | 0.002506 (0) | 3.291e-05 (0) | 0.005046 (0) | 0/6 | 6/6 |
| bain | 3 | rgsaddle | fire | 4.92761e-12 (0) | 57.5219 (0) | 2.956e-12 (0) | 2.81e-06 (0) | 3.306e-05 (0) | 6/6 | 6/6 |
| bain | 4 | eon | fire | 3.32128e-07 (0) | 137.639 (0) | 0.00243 (0) | 1.091e-05 (0) | 0.01468 (0) | 0/6 | 6/6 |
| bain | 4 | eon | lbfgs | 5.86466e-06 (0) | 138.285 (0) | 3.617e-05 (0) | 4.255e-07 (0) | 1.208 (0) | 6/6 | 6/6 |
| bain | 4 | rgsaddle | fire | 8.66862e-13 (0) | 136.351 (0) | 1.501e-13 (0) | 1.012e-06 (0) | 2.88e-05 (0) | 6/6 | 6/6 |
| bain | 5 | eon | fire | 4.40775e-09 (0) | 270.695 (0) | 0.0001602 (0) | 9.733e-07 (0) | 0.005459 (0) | 6/6 | 6/6 |
| bain | 5 | eon | lbfgs | 3.7173e-07 (0) | 266.215 (0) | 0.0015 (0) | 1.274e-05 (0) | 0.01243 (0) | 0/6 | 6/6 |
| bain | 5 | rgsaddle | fire | 6.39488e-13 (0) | 266.313 (0) | 2.834e-13 (0) | 7.191e-07 (0) | 3.334e-05 (0) | 6/6 | 6/6 |
| vacancy | 1 | cp2k | diis | 5.05026 (0) | 32 (0) | 0.0004369 (0) | 1.052 (0) | 8.159 (0) | 6/6 | 0/6 |
| vacancy | 1 | eon | fire | 5.05026 (0) | 32 (0) | 0.001239 (0) | 1.052 (0) | 7.712 (0) | 0/6 | 0/6 |
| vacancy | 1 | eon | lbfgs | 5.05026 (0) | 32 (0) | 0.002644 (0) | 1.052 (0) | 7.853 (0) | 0/6 | 0/6 |
| vacancy | 1 | rgsaddle | fire | 5.05026 (0) | 32 (0) | 5.17e-06 (0) | 1.052 (0) | 7.858 (0) | 6/6 | 0/6 |

## Agreement and causes

Bain 2x2x2 image 0: energy span 6.128e-07 eV, force span 6.472e-28 eV/A, stress span 6.215e-12 eV/A^3. The cutoff 1.2 A is longer than half the face-centered edge, so a 2x2x2 CP2K band was left out. The body-centered end still has one neighbor shell inside the cutoff, and the force span there is numerical noise.
bain size 3 image 0 initial span is 2.068e-06 eV, 3.288e-14 eV/A, 6.220e-12 eV/A^3. The force and stress match. The energy offset is the CP2K spline against the analytic shift, and it is common to both endpoints.
bain size 3 image 8 initial span is 3.102e-06 eV, 2.372e-14 eV/A, 1.015e-11 eV/A^3. The force and stress match. The energy offset is the CP2K spline against the analytic shift, and it is common to both endpoints.
vacancy size 1 image 0 initial span is 1.074e-05 eV, 5.751e-12 eV/A, 1.716e-11 eV/A^3. The force and stress match. The energy offset is the CP2K spline against the analytic shift, and it is common to both endpoints.
vacancy size 1 image 8 initial span is 1.074e-05 eV, 5.805e-12 eV/A, 1.716e-11 eV/A^3. The force and stress match. The energy offset is the CP2K spline against the analytic shift, and it is common to both endpoints.
Argon, one interior image, across cp2k diis, eon fire, eon lbfgs, rgsaddle fire: common-frame barrier spread 1.884e-02 eV and saddle-volume spread 5.992e+02 A^3. The single interior image is the saddle, and on every finished band it lies below both endpoints, so the common barrier is negative. The initial path also lies below both endpoints. eOn arms a solid climbing image only when that image is above both endpoints, so the finished eOn cells stay near the interpolated volume, 1018 A^3. rgsaddle arms climbing from a force of 0.5 eV/A and CP2K arms it from the first step. rgsaddle's image converges at 419.328 A^3 with a common atomic force of 5.00e-4 eV/A, inside the 1e-3 test. CP2K stops at 839.281 A^3 with a common atomic force of 4.07e-3 eV/A. Its native maximum force is 0.00045 in internal units, about 0.023 eV/A. eOn's native test is a force norm of 0.01, and both eOn optimizers stop with a saddle component above 1e-3 eV/A. The native column records the rise above the reactant. The interior linearized enthalpy, E + p V0 Tr(eps) on the final reactant cell, lies below the reactant by 3.049e-2 eV (rgsaddle), 6.925e-3 eV (eOn L-BFGS), and 6.801e-3 eV (eOn FIRE), so that column is 0. CP2K's native column is the exact enthalpy difference, -1.242e-2 eV.
Bain 2x2x2 across eon fire, eon lbfgs, rgsaddle fire: common-frame barrier spread 4.117e-08 eV and saddle-volume spread 6.615e-02 A^3. The first half of every band stays at the body-centered enthalpy to better than 1e-6 eV, then the enthalpy falls toward the face-centered endpoint. The highest interior image is whichever plateau image is higher by 1e-8 eV or less, and those images have different volumes, so the 1e-3 A^3 test fails while the barriers stay inside 1e-4 eV. Images past the plateau carry raw stresses of order 1 eV/A^3. The pass test uses the highest-enthalpy image. The worst interior atomic force is the downhill images.
At 2x2x2 the face-centered energy is lower by 1.744555 eV, 0.109035 eV/atom. At 3x3x3 and above it is lower by 0.218069 eV/atom. The body-centered energy is -0.436139 eV/atom at every size. The cutoff exceeds half the face-centered edge only at 2x2x2.
Bain 2x2x2 eon lbfgs saddle atomic force 1.794e-03 eV/A. eOn's native test is a force norm of 0.01.
Bain 3x3x3 across cp2k diis, eon fire, eon lbfgs, rgsaddle fire: common-frame barrier spread 3.503e-07 eV and saddle-volume spread 5.948e-01 A^3. The first half of every band stays at the body-centered enthalpy to better than 1e-6 eV, then the enthalpy falls toward the face-centered endpoint. The highest interior image is whichever plateau image is higher by 1e-8 eV or less, and those images have different volumes, so the 1e-3 A^3 test fails while the barriers stay inside 1e-4 eV. Images past the plateau carry raw stresses of order 1 eV/A^3. The pass test uses the highest-enthalpy image. The worst interior atomic force is the downhill images.
Bain 3x3x3 eon lbfgs saddle atomic force 2.506e-03 eV/A. eOn's native test is a force norm of 0.01.
Bain 4x4x4 across eon fire, eon lbfgs, rgsaddle fire: common-frame barrier spread 5.865e-06 eV and saddle-volume spread 1.934e+00 A^3. The first half of every band stays at the body-centered enthalpy to better than 1e-6 eV, then the enthalpy falls toward the face-centered endpoint. The highest interior image is whichever plateau image is higher by 1e-8 eV or less, and those images have different volumes, so the 1e-3 A^3 test fails while the barriers stay inside 1e-4 eV. Images past the plateau carry raw stresses of order 1 eV/A^3. The pass test uses the highest-enthalpy image. The worst interior atomic force is the downhill images.
Bain 4x4x4 eon fire saddle atomic force 2.430e-03 eV/A. eOn's native test is a force norm of 0.01.
Bain 5x5x5 across eon fire, eon lbfgs, rgsaddle fire: common-frame barrier spread 3.717e-07 eV and saddle-volume spread 4.481e+00 A^3. The first half of every band stays at the body-centered enthalpy to better than 1e-6 eV, then the enthalpy falls toward the face-centered endpoint. The highest interior image is whichever plateau image is higher by 1e-8 eV or less, and those images have different volumes, so the 1e-3 A^3 test fails while the barriers stay inside 1e-4 eV. Images past the plateau carry raw stresses of order 1 eV/A^3. The pass test uses the highest-enthalpy image. The worst interior atomic force is the downhill images.
Bain 5x5x5 eon lbfgs saddle atomic force 1.500e-03 eV/A. eOn's native test is a force norm of 0.01.
Vacancy across cp2k diis, eon fire, eon lbfgs, rgsaddle fire: common-frame barrier spread 1.414e-07 eV and saddle-volume spread 1.101e-09 A^3. The barrier and the volume meet the 1e-4 eV and 1e-3 A^3 tests. The cell is fixed, so |sigma+pI| stays at 1.052 eV/A^3 against a limit of 3.17e-3, and dH/dh on the saddle is about 10.6 eV/A. Every code fails the stress test for that reason. The saddle atomic force passes for rgsaddle (5.17e-6 eV/A) and CP2K (4.37e-4 eV/A). eOn L-BFGS stops at 2.64e-3 eV/A and eOn FIRE at 1.24e-3 eV/A, above 1e-3, because the native test is a force norm of 0.01. CP2K has no force-threshold climb; this case uses its default of five non-climbing steps, while rgsaddle and eOn arm at 0.5 eV/A. The worst raw atomic force on the non-climbing images is 7.7 to 8.2 eV/A.
Every phase-2 eOn FIRE row converged. The scout driver still aborts with [FIRE] m_dt is too small: 0.0000. Those FIRE rows stay separate from the L-BFGS medians.

## Variable-cell dimer

rgsaddle has a solid-state dimer on the 3N+6 lower-triangular chart in src/solid_dimer.rs.
The cell block is six strains, the Jacobian is fixed from the starting cell, and the
generalized cell force uses the Cauchy stress. It is not in the C API. CP2K and eOn
keep dimer vectors at 3N. No dimer timing was run.

## Exclusions

EAM and Tersoff are not parameter-identical across the three codes, and the shared
potential does not publish their stress. The CdSe generalized solid-state band needs
a Coulomb sum. The Mo A15-to-BCC path needs an embedded-atom stress. None of those
were run.
