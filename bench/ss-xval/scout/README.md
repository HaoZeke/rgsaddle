# Solid-state saddle cross-validation scout

## Scope and revisions

This is a one-core functional scout, not a timing campaign. Every build and
scientific run was pinned with `taskset -c 0`. `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `BLIS_NUM_THREADS`,
`RAYON_NUM_THREADS`, `CARGO_BUILD_JOBS`, and `MESON_NUM_PROCESSES` were set to
1.

Hardware:

- CPU: Intel Xeon Processor
- 8 logical CPUs; one socket, 8 cores/socket, one thread/core
- scientific-run affinity: logical CPU 0 only

Revisions:

| component | revision |
|---|---|
| CP2K | `a942a44f18d07e1b47cddea61c678c9ee74c0ee5` |
| rgsaddle | `f6a2f1e4d56d6c8bd5147aaf0f3b0c9231b3fdcc` |
| eOn `develop` | `4ee85b89e55b2ddea83f3e60b202055ef31d5bf5` |
| eOn rgpot base | `e40fd175f5230098d56333bf2aff38cd839dc810` plus eOn wrap patches |
| eOn readcon-core | `6183e294d14547a140b1342c42514efcfe6faece` |
| rgsaddle rgpot-core lock entry | `97ebba0b062128b7dc4637501c4b70ed4944ae9d` |
| rgmin | `0.6.0` |

## Exact build record

CP2K serial `ssmp` used GCC/G++/GFortran 13.3.0, CMake 3.28.3, Ninja
1.11.1, OpenBLAS 0.3.26, FFTW 3.3.10, DBCSR `8e8c7d4`, and libwignernj
0.8.0:

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

rgsaddle used Rust/Cargo 1.88.0:

```sh
taskset -c 0 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  CARGO_BUILD_JOBS=1 cargo +1.88.0 build --release --locked \
  --features capi -j 1
```

eOn used GCC/G++ 13.3.0, Rust/Cargo 1.88.0, Meson 1.12.1, Python
3.12.3, Eigen 3.4.0, and Cap'n Proto 1.0.1:

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

The eOn reconfigure selects its pinned `inih-r62` and Highway 1.4.0 because
the system headers lacked APIs used by `develop`.

## Feature inventory

| concern | CP2K | rgsaddle | eOn `develop` |
|---|---|---|---|
| solid-state band | Native solid-state IT-NEB and CI-NEB; fixed endpoints, DIIS and one rank/replica are enforced (`src/input_cp2k_neb.F:93-165`, `src/motion/neb_methods.F:185-218`). | Native NEB, CI-NEB and DNEB in an absolute-cell `SolidBand` (`src/band.rs:238-289,720-735,1252-1627`). | Native solid-state NEB and CI-NEB; rejects DNEB, elastic band, OCINEB/minmode, zoom and OM (`client/NudgedElasticBand.cpp:1017-1054,1079-1188`). |
| cell chart and scaling | Six coordinates. The first endpoint supplies the Sheppard Jacobian; diagonal deformation is logarithmic and shear linear (`src/motion/solid_state_coordinates.F:20-104`). | Six right-strain coordinates. Absolute-cell NEB uses endpoint-mean volume; joint solid-state APIs use a reference volume (`src/solid_state.rs:22-36,176-257`, `src/band.rs:1289-1296`). | Six coordinates with the Sheppard dimensional Jacobian, endpoint-mean volume and averaged finite right strain (`client/SolidStateNEB.cpp:40-253`, `client/NudgedElasticBand.cpp:1079-1099`). |
| pressure and stress | Exact current `E+pV`; virial stress is pulled back to generalized cell forces (`src/motion/neb_utils.F:494-570`, `src/motion/solid_state_coordinates.F:130-169`). | Absolute-cell NEB uses linearized `E+p V0 tr(epsilon)` while its cell force uses current volume; the joint chart consistently supports general external stress (`src/band.rs:1327-1368`, `src/solid_state.rs:259-337`). | Linearized `E+p V0 tr(epsilon)` and current-volume stress cell force (`client/SolidStateNEB.cpp:146-228`). Potential stress is explicit (`include/eon/Potential.h:198-214`). |
| springs and projection | One scalar `K_SPRING`; IT and climbing projections are selected by band type (`src/input_cp2k_neb.F:167-187`, `src/motion/neb_utils.F:601-627`). | Scalar or per-segment springs; solid NEB, CI and DNEB projection (`src/band.rs:1430-1525`). | Scalar or adaptive springs for solid NEB/CI; unsupported projection variants are rejected (`client/NudgedElasticBand.cpp:1017-1054,1103-1188`). |
| climbing criterion | Highest interior image climbs after integer `NSTEPS_IT`, even when it is below an endpoint (`src/motion/neb_utils.F:601-627,933-952`). | CI arms when projected maximum force crosses a threshold (`src/band.rs:1430-1499`). | Highest interior image climbs only if its energy exceeds both endpoints (`client/NudgedElasticBand.cpp:1155-1188`). |
| convergence | All enabled max/RMS force and max/RMS displacement tests must pass; cell components are included (`src/input_cp2k_neb.F:203-233`, `src/motion/neb_methods.F:1018-1062`). | Maximum generalized force component, combining atomic and scaled-cell components (`src/band.rs:1588-1605`). | Selectable norm, maximum-atom or maximum-component tests include the cell block; optional CI-only convergence (`client/NudgedElasticBand.cpp:659-707`). |
| variable-cell dimer | None; dimer arrays are atomic `3N` (`src/motion/dimer_types.F:146-159`). | `SolidDimer` supports `3N+6`, stress or finite-difference cell gradients, dimer or Lanczos mode, and external stress (`src/solid_dimer.rs:60-143,430-603`, `src/minmode.rs:47-100`). It is not exposed by the C ABI. | None; dimer vectors remain atomic `3N` (`client/ImprovedDimer.cpp:47-69`, `include/eon/Dimer.h:20-39`). |

The names are not drop-in equivalents. All three use the Sheppard dimensional
Jacobian formula, but CP2K uses the first cell, a log/linear cell chart and
exact current volume. rgsaddle's absolute band and eOn use endpoint-mean
volume, averaged finite strain and linearized pressure work. CP2K switches CI
by iteration, rgsaddle by force, and eOn by an endpoint-relative energy test.
Only rgsaddle has a variable-cell dimer; a C/C++ caller would first need a C
ABI for it. CP2K and eOn need new `3N+6` dimer implementations.

## CP2K additions and one-run results

Commit `a942a44f` adds exactly one solid-state regtest:

- `tests/NEB/regtest-solid-state/argon.inp`
- `tests/NEB/regtest-solid-state/TEST_FILES.toml`

Byte-identical copies are in `cp2k-regtests/`. No separate IT-NEB input and no
variable-cell dimer input were added. This is a three-image, two-step
solid-state CI-NEB at 50 bar with FIST shifted-LJ argon. `NSTEPS_IT 0` makes
it climb immediately; `MAX_STEPS 2` deliberately prevents convergence.

`TEST_FILES.toml` contains no energy or barrier reference. Its two references
are initial shear force `4.24019199e-5` with tolerance `2e-4`, and final
middle volume `6878.05002324` bohr3 with tolerance `1e-7`. The one CP2K run
gave `4.2401970326e-5` and `6878.050024508261` bohr3. The shear matches; the
volume differs by `1.27e-6` bohr3 on this toolchain.

Initial image results:

| image | energy (eV) | `E+pV` (eV) | volume (A3) |
|---:|---:|---:|---:|
| 0 | 0.001802832599 | 0.033010383571 | 1000.0000131 |
| 1 | -0.002413456960 | 0.029394830763 | 1019.2497374 |
| 2 | -0.004682103477 | 0.027738485002 | 1038.8700136 |

After two steps, image 1 was `-0.002994604779` eV,
`0.028812812899` eV including `pV`, and `1019.2218584` A3. The band
is monotone: its forward barrier is zero and reverse endpoint difference is
`0.00527189857` eV.

The exact physical images, shifted-LJ parameters, pressure and spring were
also attempted once in rgsaddle and eOn:

| implementation | expressibility and band outcome |
|---|---|
| CP2K | Exact source regtest; two optimizer steps by design, not converged. |
| rgsaddle | Same physical problem is expressible, but its chart and pressure work are not algorithmically identical. It converged in 194 iterations to maximum generalized force `9.17e-7`, middle volume `936.8655` A3 and zero forward barrier. |
| eOn | Same physical problem is expressible through the stress-capable rgpot LJ adapter. The one smoke attempt aborted before its first optimizer step with `[FIRE] m_dt is too small: 0.0000`; no band-convergence claim is possible. Its endpoint-relative CI rule would leave this monotone band unclimbed. |

Initial endpoint agreement:

| quantity | CP2K | rgpot/rgsaddle | rgpot/eOn | three-code span |
|---|---:|---:|---:|---:|
| energy (eV) | 0.001802832599 | 0.001802830447 | 0.001802830447* | `2.15e-9` eV |
| maximum force component (eV/A) | 0.05446990484 | 0.05446983403 | 0.05446983403* | `7.08e-8` eV/A |
| maximum stress component (eV/A3) | 0.0001797506593 | 0.0001797504523 | 0.0001797504523* | `2.07e-10` eV/A3 |

CP2K uses an LJ spline; rgpot evaluates the analytic shifted form. rgsaddle is
potential-agnostic, so its smoke callback uses the same rgpot LJ equations.
The starred eOn values are the deterministic output of that same rgpot
adapter: its initial energy, force and stress calls completed before FIRE
threw, but buffered output was not emitted during the abort. The final
rgsaddle/eOn states need not match CP2K because their cell chart and pressure
work differ.

## Common-potential audit and candidates

| potential | identical operational status |
|---|---|
| Lennard-Jones | Yes, modulo CP2K spline roundoff: shifted energy, analytic forces and stress are available to all three. This is the only current three-code solid-state choice. |
| EAM | No. CP2K reads tabulated EAM, while rgpot/eOn expose a fixed analytic double-exponential Al model. Its rgpot capability does not publish stress, so eOn rejects solid-state NEB. |
| Tersoff | No. CP2K's configurable form includes a `lambda3` radial-coordination term; rgpot/eOn use a fixed Si(C) form with different defaults. Its rgpot capability does not publish stress. |

Candidate systems, in priority order:

1. The exact Ar LJ regtest is the immediate CI-NEB expression and agreement
   check.
2. A small unswitched, energy-shifted LJ Bain bcc-to-fcc path is a clean
   phase-transition case after matching cutoff and image conventions.
3. A shifted-LJ vacancy hop in a periodic cell is useful for energy/force
   parity; fixed cell alone does not test solid-state degrees of freedom.
4. Diamond-to-beta-Sn Si becomes useful after setting CP2K `lambda3=0`,
   matching all Tersoff parameters and implementing rgpot stress.

The G-SSNEB paper is Sheppard, Xiao, Chemelewski, Johnson and Henkelman,
J. Chem. Phys. 136, 074103 (2012), DOI `10.1063/1.3684549`. Its CdSe
rocksalt-to-wurtzite case uses an empirical Rabani LJ plus long-range
Coulomb/Ewald model. It is empirical but is not a canonical common potential.

The solid-state dimer paper is Xiao, Sheppard, Rogal and Henkelman,
J. Chem. Phys. 140, 174104 (2014), DOI `10.1063/1.4873437`. Its Mo
A15-to-BCC example uses a Mo EAM. It is a suitable dimer target after a
common tabulated EAM gains a stress capability in rgpot/eOn.

## Blockers before a cross-code campaign

1. Select one cell chart, Jacobian reference volume and pressure-work
   convention before comparing optimized paths.
2. State equivalent climbing and convergence rules explicitly; similarly
   named controls have materially different semantics.
3. LJ is ready for functional checks. EAM and Tersoff need parameter-identical
   implementations and rgpot stress before they are three-code solid-state
   candidates.
4. Expose rgsaddle's solid dimer through its C ABI. CP2K and eOn still need
   variable-cell dimer implementations.
5. The direct eOn smoke harness must initialize FIRE's time-step fields before
   its band can be assessed; the single allowed attempt stopped at that guard.
