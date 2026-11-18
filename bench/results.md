# Saddle-search coverage

Stop when the maximum per-atom force norm is under 0.01 eV/Angstrom.
Spring 5.0 eV/Angstrom^2, linear path, 7 and 11 images, 1000 iterations.
Success: code converged, reference index is 1, energy within 0.01 eV,
free-atom RMSD within 0.05 Angstrom. A free cluster is centered and
Kabsch-aligned, and identical species are matched. Medians and IQRs use n=12.
A/A is a Mann-Whitney test of even rounds against odd rounds.
Method p-values are Mann-Whitney, and success p-values are Fisher exact.
ASE ships no Lanczos min-mode search.

## pt_hop

Reference usable: True; index 1; barrier 0.09056 eV.

| method | images | success | wall s (IQR) | outside s (IQR) | calls (IQR) | barrier eV (IQR) | rmsd A | A/A wall p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rgsaddle-neb | 7 | 12/12 | 0.3075 (0.03911) | 0.07284 (0.01104) | 212 (0) | 0.09057 (0) | 0.0004977 | 1 |
| eon-neb | 7 | 12/12 | 0.1004 (0.01214) | 0.01024 (0.001541) | 57 (0) | 0.09057 (0) | 0.0006239 | 0.3785 |
| ase-neb | 7 | 12/12 | 0.4428 (0.04532) | 0.2478 (0.0297) | 87 (0) | 0.09058 (0) | 0.0005976 | 0.5752 |
| rgsaddle-ci | 7 | 12/12 | 0.3659 (0.0479) | 0.08954 (0.01425) | 250 (0) | 0.09057 (0) | 0.0005525 | 0.6889 |
| eon-ci | 7 | 12/12 | 0.09978 (0.009493) | 0.01003 (0.001866) | 57 (0) | 0.09057 (0) | 0.0005987 | 0.4712 |
| ase-ci | 7 | 12/12 | 0.4476 (0.05489) | 0.2504 (0.03353) | 87 (0) | 0.09058 (0) | 0.0005767 | 0.6889 |
| rgsaddle-neb | 11 | 12/12 | 0.6919 (0.08033) | 0.1654 (0.02438) | 479 (0) | 0.09056 (0) | 0.0002017 | 0.4712 |
| eon-neb | 11 | 12/12 | 0.2115 (0.02132) | 0.01873 (0.002064) | 101 (0) | 0.09058 (0) | 0.0007332 | 0.5752 |
| ase-neb | 11 | 12/12 | 1.162 (0.09262) | 0.8935 (0.07234) | 155 (0) | 0.09058 (0) | 0.0006189 | 0.6889 |
| rgsaddle-ci | 11 | 12/12 | 0.7464 (0.04611) | 0.1793 (0.009864) | 517 (0) | 0.09056 (0) | 0.0006027 | 0.8102 |
| eon-ci | 11 | 12/12 | 0.2174 (0.02283) | 0.01924 (0.001642) | 101 (0) | 0.09058 (0) | 0.0007137 | 0.8102 |
| ase-ci | 11 | 12/12 | 1.191 (0.1029) | 0.9142 (0.1082) | 155 (0) | 0.09058 (0) | 0.0006007 | 0.2298 |
| rgsaddle-dimer |  | 0/12 | 0.3909 (0.07472) | 0.1013 (0.01989) | 250.5 (49) |  () | 0.4594 | 0.8102 |
| eon-dimer |  | 1/12 | 0.2015 (0.04729) | 0.02179 (0.00542) | 132 (59) | 0.09057 (0) | 0.4608 | 0.6889 |
| ase-dimer |  | 0/12 | 0.5101 (0.2191) | 0.1308 (0.0549) | 331 (78.25) |  () | 0.7206 | 0.9362 |
| rgsaddle-lanczos |  | 1/12 | 0.392 (0.137) | 0.09629 (0.03557) | 239.5 (80.25) | 0.09058 (0) | 0.6214 | 0.1735 |
| eon-lanczos |  | 3/12 | 0.3049 (0.1648) | 0.03423 (0.01919) | 187.5 (141.8) | 0.09058 (1.587e-05) | 0.3416 | 0.1282 |

| comparison | images | wall p | outside p | calls p | barrier p | success p |
| --- | --- | --- | --- | --- | --- | --- |
| rgsaddle-neb vs eon-neb | 7 | 3.658e-05 | 3.658e-05 | 1.912e-06 | 1.912e-06 | 1 |
| rgsaddle-neb vs ase-neb | 7 | 3.658e-05 | 3.658e-05 | 1.912e-06 | 1.912e-06 | 1 |
| rgsaddle-ci vs eon-ci | 7 | 3.658e-05 | 3.658e-05 | 1.912e-06 | 1.912e-06 | 1 |
| rgsaddle-ci vs ase-ci | 7 | 0.008616 | 3.658e-05 | 1.912e-06 | 1.912e-06 | 1 |
| rgsaddle-neb vs eon-neb | 11 | 3.658e-05 | 3.658e-05 | 1.912e-06 | 1.912e-06 | 1 |
| rgsaddle-neb vs ase-neb | 11 | 3.658e-05 | 3.658e-05 | 1.912e-06 | 1.912e-06 | 1 |
| rgsaddle-ci vs eon-ci | 11 | 3.658e-05 | 3.658e-05 | 1.912e-06 | 1.912e-06 | 1 |
| rgsaddle-ci vs ase-ci | 11 | 3.658e-05 | 3.658e-05 | 1.912e-06 | 1.912e-06 | 1 |
| rgsaddle-dimer vs eon-dimer |  | 0.0007315 | 6.006e-05 | 0.001652 |  | 1 |
| rgsaddle-dimer vs ase-dimer |  | 0.0404 | 0.04639 | 0.02428 |  | 1 |
| rgsaddle-lanczos vs eon-lanczos |  | 0.1124 | 3.658e-05 | 0.2039 | 1 | 0.5901 |

## pt_exchange

Reference usable: True; index 1; barrier 4.569 eV.

| method | images | success | wall s (IQR) | outside s (IQR) | calls (IQR) | barrier eV (IQR) | rmsd A | A/A wall p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rgsaddle-neb | 7 | 0/12 | 11.4 (1.311) | 1.166 (0.1266) | 2677 (0) |  () | 0.8748 | 0.5752 |
| eon-neb | 7 | 0/12 | 21.5 (1.882) | 1.139 (0.09207) | 5002 (0) |  () | 0.7607 | 0.9362 |
| ase-neb | 7 | 0/12 | 40.8 (4.912) | 19.36 (2.544) | 5007 (0) |  () | 7.601 | 0.4712 |
| rgsaddle-ci | 7 | 0/12 | 11.74 (1.123) | 1.189 (0.1374) | 2677 (0) |  () | 0.8748 | 0.8102 |
| eon-ci | 7 | 0/12 | 1.159 (0.1707) | 0.06307 (0.009553) | 297 (0) |  () | 0.7651 | 0.6889 |
| ase-ci | 7 | 0/12 | 39.54 (5.43) | 18.71 (2.842) | 5007 (0) |  () | 4.048 | 0.2298 |
| rgsaddle-neb | 11 | 0/12 | 4.482 (0.8566) | 0.5416 (0.1083) | 1316 (0) |  () | 0.09887 | 0.0927 |
| eon-neb | 11 | 0/12 | 4.069 (0.6636) | 0.2342 (0.0416) | 1100 (0) |  () | 0.7636 | 0.2298 |
| ase-neb | 11 | 0/12 | 96.57 (13.55) | 59.98 (9.341) | 9011 (0) |  () | 3.999 | 1 |
| rgsaddle-ci | 11 | 12/12 | 3.579 (0.2414) | 0.4709 (0.03809) | 1134 (0) | 4.569 (0) | 0.0002717 | 0.4712 |
| eon-ci | 11 | 0/12 | 2.255 (0.1549) | 0.1384 (0.01728) | 659 (0) |  () | 0.7649 | 0.6889 |
| ase-ci | 11 | 0/12 | 99.39 (11.32) | 61.22 (7.571) | 9011 (0) |  () | 3.902 | 0.6889 |
| rgsaddle-dimer |  | 0/12 | 0.383 (0.08667) | 0.09867 (0.01878) | 250.5 (49) |  () | 0.9235 | 0.9362 |
| eon-dimer |  | 0/12 | 0.2036 (0.07886) | 0.02237 (0.005297) | 132 (59) |  () | 0.9242 | 0.298 |
| ase-dimer |  | 0/12 | 0.5077 (0.1556) | 0.1298 (0.039) | 331 (78.25) |  () | 1.083 | 0.9362 |
| rgsaddle-lanczos |  | 0/12 | 0.3757 (0.139) | 0.09547 (0.03373) | 239.5 (80.25) |  () | 0.8591 | 0.298 |
| eon-lanczos |  | 0/12 | 0.2957 (0.1636) | 0.03359 (0.02128) | 187.5 (141.8) |  () | 0.919 | 0.06555 |

| comparison | images | wall p | outside p | calls p | barrier p | success p |
| --- | --- | --- | --- | --- | --- | --- |
| rgsaddle-neb vs eon-neb | 7 | 3.658e-05 | 0.2366 | 1.912e-06 |  | 1 |
| rgsaddle-neb vs ase-neb | 7 | 3.658e-05 | 3.658e-05 | 1.912e-06 |  | 1 |
| rgsaddle-ci vs eon-ci | 7 | 3.658e-05 | 3.658e-05 | 1.912e-06 |  | 1 |
| rgsaddle-ci vs ase-ci | 7 | 3.658e-05 | 3.658e-05 | 1.912e-06 |  | 1 |
| rgsaddle-neb vs eon-neb | 11 | 0.008616 | 3.658e-05 | 1.912e-06 |  | 1 |
| rgsaddle-neb vs ase-neb | 11 | 3.658e-05 | 3.658e-05 | 1.912e-06 |  | 1 |
| rgsaddle-ci vs eon-ci | 11 | 3.658e-05 | 3.658e-05 | 1.912e-06 |  | 7.396e-07 |
| rgsaddle-ci vs ase-ci | 11 | 3.658e-05 | 3.658e-05 | 1.912e-06 |  | 7.396e-07 |
| rgsaddle-dimer vs eon-dimer |  | 0.000592 | 3.658e-05 | 0.001652 |  | 1 |
| rgsaddle-dimer vs ase-dimer |  | 0.03509 | 0.03038 | 0.02428 |  | 1 |
| rgsaddle-lanczos vs eon-lanczos |  | 0.126 | 3.658e-05 | 0.2039 |  | 1 |

## al_vacancy

Reference usable: True; index 1; barrier 0.2207 eV.

| method | images | success | wall s (IQR) | outside s (IQR) | calls (IQR) | barrier eV (IQR) | rmsd A | A/A wall p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rgsaddle-neb | 7 | 12/12 | 1.444 (0.1397) | 0.07788 (0.008379) | 157 (0) | 0.2209 (0) | 0.00711 | 0.8102 |
| eon-neb | 7 | 12/12 | 0.6726 (0.05011) | 0.01985 (0.001565) | 72 (0) | 0.2208 (0) | 0.005604 | 0.4712 |
| ase-neb | 7 | 12/12 | 1.937 (0.201) | 0.5378 (0.05107) | 152 (0) | 0.2208 (0) | 0.00616 | 0.9362 |
| rgsaddle-ci | 7 | 12/12 | 1.554 (0.1869) | 0.1087 (0.008509) | 211 (0) | 0.2209 (0) | 0.007109 | 0.6889 |
| eon-ci | 7 | 12/12 | 0.5036 (0.08042) | 0.01465 (0.00171) | 52 (0) | 0.2209 (0) | 0.007145 | 0.5752 |
| ase-ci | 7 | 12/12 | 1.94 (0.2648) | 0.5439 (0.06113) | 152 (0) | 0.2208 (0) | 0.006158 | 0.9362 |
| rgsaddle-neb | 11 | 12/12 | 2.476 (0.2857) | 0.1899 (0.01497) | 398 (0) | 0.2208 (0) | 0.005824 | 0.9362 |
| eon-neb | 11 | 12/12 | 1.408 (0.1271) | 0.04172 (0.002828) | 155 (0) | 0.2208 (0) | 0.005093 | 0.5752 |
| ase-neb | 11 | 12/12 | 5.119 (0.6685) | 3.07 (0.3415) | 362 (0) | 0.2208 (0) | 0.005416 | 0.9362 |
| rgsaddle-ci | 11 | 12/12 | 2.541 (0.2656) | 0.2215 (0.008891) | 452 (0) | 0.2208 (0) | 0.005821 | 0.9362 |
| eon-ci | 11 | 12/12 | 0.8119 (0.07196) | 0.02514 (0.002095) | 92 (0) | 0.221 (0) | 0.007438 | 0.9362 |
| ase-ci | 11 | 12/12 | 5.054 (0.8773) | 3.046 (0.5474) | 362 (0) | 0.2208 (0) | 0.005413 | 0.9362 |
| rgsaddle-dimer |  | 0/12 | 7.212 (5.443) | 1.76 (1.293) | 3620 (2309) |  () | 3.295 | 0.298 |
| eon-dimer |  | 2/12 | 0.9573 (0.485) | 0.1092 (0.05862) | 418.5 (275) | 0.221 (0.0001537) | 0.3748 | 0.2298 |
| ase-dimer |  | 0/12 | 5.431 (3.572) | 1.566 (1.076) | 3114 (2230) |  () | 2.176 | 0.3785 |
| rgsaddle-lanczos |  | 0/12 | 7.299 (0.8102) | 1.733 (0.1899) | 3448 (44.75) |  () | 3.964 | 0.9362 |
| eon-lanczos |  | 0/12 | 27.71 (3.567) | 3.429 (0.1659) | 1.317e+04 (326.8) |  () | 0.5494 | 0.1735 |

| comparison | images | wall p | outside p | calls p | barrier p | success p |
| --- | --- | --- | --- | --- | --- | --- |
| rgsaddle-neb vs eon-neb | 7 | 3.658e-05 | 3.658e-05 | 1.912e-06 | 1.912e-06 | 1 |
| rgsaddle-neb vs ase-neb | 7 | 3.658e-05 | 3.658e-05 | 1.912e-06 | 1.912e-06 | 1 |
| rgsaddle-ci vs eon-ci | 7 | 3.658e-05 | 3.658e-05 | 1.912e-06 | 1.912e-06 | 1 |
| rgsaddle-ci vs ase-ci | 7 | 7.658e-05 | 3.658e-05 | 1.912e-06 | 1.912e-06 | 1 |
| rgsaddle-neb vs eon-neb | 11 | 3.658e-05 | 3.658e-05 | 1.912e-06 | 1.912e-06 | 1 |
| rgsaddle-neb vs ase-neb | 11 | 3.658e-05 | 3.658e-05 | 1.912e-06 | 1.912e-06 | 1 |
| rgsaddle-ci vs eon-ci | 11 | 3.658e-05 | 3.658e-05 | 1.912e-06 | 1.912e-06 | 1 |
| rgsaddle-ci vs ase-ci | 11 | 3.658e-05 | 3.658e-05 | 1.912e-06 | 1.912e-06 | 1 |
| rgsaddle-dimer vs eon-dimer |  | 0.002946 | 9.735e-05 | 0.003228 |  | 0.4783 |
| rgsaddle-dimer vs ase-dimer |  | 0.07825 | 0.2145 | 0.06896 |  | 1 |
| rgsaddle-lanczos vs eon-lanczos |  | 3.658e-05 | 3.658e-05 | 3.658e-05 |  | 1 |

## lj7

Reference usable: True; index 1; barrier 1.061 eV.

| method | images | success | wall s (IQR) | outside s (IQR) | calls (IQR) | barrier eV (IQR) | rmsd A | A/A wall p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rgsaddle-neb | 7 | 0/12 | 0.9101 (0.06253) | 0.8945 (0.06167) | 3872 (0) |  () | 0.1366 | 1 |
| eon-neb | 7 | 0/12 | 0.09784 (0.006215) | 0.09323 (0.005921) | 1432 (0) |  () | 0.1367 | 0.9362 |
| ase-neb | 7 | 0/12 | 4.804 (0.4459) | 4.776 (0.4415) | 4627 (0) |  () | 0.2826 | 0.9362 |
| rgsaddle-ci | 7 | 12/12 | 0.1417 (0.01168) | 0.1393 (0.01146) | 602 (0) | 1.061 (0) | 0.0002671 | 0.9362 |
| eon-ci | 7 | 12/12 | 0.01861 (0.002313) | 0.01775 (0.002192) | 272 (0) | 1.061 (0) | 0.0006788 | 0.9362 |
| ase-ci | 7 | 0/12 | 5.08 (0.5506) | 4.98 (0.5381) | 5007 (0) |  () | 11.54 | 0.6889 |
| rgsaddle-neb | 11 | 0/12 | 0.3155 (0.03564) | 0.3101 (0.03505) | 1397 (0) |  () | 0.02341 | 0.9362 |
| eon-neb | 11 | 0/12 | 0.4507 (0.04901) | 0.4295 (0.04678) | 6806 (0) |  () | 0.1366 | 0.5752 |
| ase-neb | 11 | 0/12 | 9.181 (1.372) | 9.133 (1.363) | 9011 (0) |  () | 0.1448 | 0.9362 |
| rgsaddle-ci | 11 | 12/12 | 0.2076 (0.01959) | 0.2041 (0.01918) | 917 (0) | 1.061 (0) | 8.82e-06 | 0.5752 |
| eon-ci | 11 | 0/12 | 0.1468 (0.01886) | 0.14 (0.01801) | 2243 (0) |  () | 0.1367 | 0.8102 |
| ase-ci | 11 | 0/12 | 6.441 (0.7633) | 6.408 (0.7584) | 6275 (0) |  () | 0.1367 | 0.6889 |
| rgsaddle-dimer |  | 0/12 | 0.7653 (0.5734) | 0.7525 (0.5634) | 3086 (2267) |  () | 0.3162 | 0.5752 |
| eon-dimer |  | 2/12 | 0.009673 (0.008082) | 0.009213 (0.007698) | 138.5 (101.2) | 1.061 (2.209e-07) | 0.187 | 0.5752 |
| ase-dimer |  | 2/12 | 0.04062 (0.1736) | 0.03984 (0.1703) | 173.5 (806.5) | 1.061 (3.017e-06) | 0.2825 | 0.4712 |
| rgsaddle-lanczos |  | 0/12 | 1.677 (0.1143) | 1.65 (0.1122) | 6746 (488) |  () | 0.5204 | 0.9362 |
| eon-lanczos |  | 1/12 | 0.01663 (0.0226) | 0.01585 (0.02147) | 229.5 (284.2) | 1.061 (0) | 0.1963 | 0.9362 |

| comparison | images | wall p | outside p | calls p | barrier p | success p |
| --- | --- | --- | --- | --- | --- | --- |
| rgsaddle-neb vs eon-neb | 7 | 3.658e-05 | 3.658e-05 | 1.912e-06 |  | 1 |
| rgsaddle-neb vs ase-neb | 7 | 3.658e-05 | 3.658e-05 | 1.912e-06 |  | 1 |
| rgsaddle-ci vs eon-ci | 7 | 3.658e-05 | 3.658e-05 | 1.912e-06 | 1.912e-06 | 1 |
| rgsaddle-ci vs ase-ci | 7 | 3.658e-05 | 3.658e-05 | 1.912e-06 |  | 7.396e-07 |
| rgsaddle-neb vs eon-neb | 11 | 3.658e-05 | 3.658e-05 | 1.912e-06 |  | 1 |
| rgsaddle-neb vs ase-neb | 11 | 3.658e-05 | 3.658e-05 | 1.912e-06 |  | 1 |
| rgsaddle-ci vs eon-ci | 11 | 3.658e-05 | 3.658e-05 | 1.912e-06 |  | 7.396e-07 |
| rgsaddle-ci vs ase-ci | 11 | 3.658e-05 | 3.658e-05 | 1.912e-06 |  | 7.396e-07 |
| rgsaddle-dimer vs eon-dimer |  | 3.658e-05 | 3.658e-05 | 9.735e-05 |  | 0.4783 |
| rgsaddle-dimer vs ase-dimer |  | 0.001354 | 0.001354 | 0.001493 |  | 0.4783 |
| rgsaddle-lanczos vs eon-lanczos |  | 3.658e-05 | 3.658e-05 | 3.644e-05 |  | 1 |

## cu4_h

Reference usable: True; index 1; barrier 0.2365 eV.

| method | images | success | wall s (IQR) | outside s (IQR) | calls (IQR) | barrier eV (IQR) | rmsd A | A/A wall p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rgsaddle-neb | 7 | 0/12 | 0.06332 (0.007198) | 0.06059 (0.006763) | 267 (0) |  () | 0.08286 | 0.4712 |
| eon-neb | 7 | 0/12 | 0.009479 (0.001314) | 0.008314 (0.001066) | 127 (0) |  () | 0.1086 | 1 |
| ase-neb | 7 | 0/12 | 0.3606 (0.05208) | 0.3558 (0.05121) | 402 (0) |  () | 0.1349 | 0.298 |
| rgsaddle-ci | 7 | 0/12 | 0.7579 (0.1161) | 0.7251 (0.1102) | 3196 (0) |  () | 0.5242 | 0.3785 |
| eon-ci | 7 | 12/12 | 0.009505 (0.0009125) | 0.008327 (0.0007694) | 127 (0) | 0.2365 (0) | 0.006643 | 0.4712 |
| ase-ci | 7 | 0/12 | 4.662 (0.6059) | 4.599 (0.5962) | 5007 (0) |  () | 11.96 | 0.2298 |
| rgsaddle-neb | 11 | 0/12 | 0.2255 (0.0188) | 0.2163 (0.01785) | 929 (0) |  () | 0.06062 | 0.6889 |
| eon-neb | 11 | 0/12 | 0.0168 (0.002227) | 0.01484 (0.001871) | 227 (0) |  () | 0.06253 | 0.06555 |
| ase-neb | 11 | 0/12 | 3.623 (0.3901) | 3.575 (0.3838) | 4097 (0) |  () | 0.1561 | 0.298 |
| rgsaddle-ci | 11 | 12/12 | 0.2883 (0.02399) | 0.2766 (0.0228) | 1226 (0) | 0.2365 (0) | 1.647e-06 | 0.3785 |
| eon-ci | 11 | 12/12 | 0.02782 (0.001962) | 0.02464 (0.001715) | 389 (0) | 0.2365 (0) | 0.007429 | 0.6889 |
| ase-ci | 11 | 0/12 | 7.393 (0.6845) | 7.3 (0.6752) | 8327 (0) |  () | 12.91 | 0.8102 |
| rgsaddle-dimer |  | 3/12 | 0.114 (0.1716) | 0.1109 (0.1664) | 445 (657) | 0.2365 (3.297e-05) | 0.5842 | 0.1282 |
| eon-dimer |  | 3/12 | 0.0437 (0.03005) | 0.04018 (0.02766) | 612 (384) | 0.2365 (1.245e-05) | 0.8569 | 0.1282 |
| ase-dimer |  | 5/12 | 0.03109 (0.09911) | 0.03013 (0.09607) | 136 (409.5) | 0.2365 (8.757e-06) | 0.4848 | 0.3785 |
| rgsaddle-lanczos |  | 2/12 | 0.1765 (0.4325) | 0.1717 (0.4186) | 704 (1754) | 0.2364 (2.624e-05) | 0.534 | 1 |
| eon-lanczos |  | 1/12 | 0.0258 (0.03737) | 0.02376 (0.03438) | 363.5 (511) | 0.2365 (0) | 0.5131 | 0.4712 |

| comparison | images | wall p | outside p | calls p | barrier p | success p |
| --- | --- | --- | --- | --- | --- | --- |
| rgsaddle-neb vs eon-neb | 7 | 3.658e-05 | 3.658e-05 | 1.912e-06 |  | 1 |
| rgsaddle-neb vs ase-neb | 7 | 3.658e-05 | 3.658e-05 | 1.912e-06 |  | 1 |
| rgsaddle-ci vs eon-ci | 7 | 3.658e-05 | 3.658e-05 | 1.912e-06 |  | 7.396e-07 |
| rgsaddle-ci vs ase-ci | 7 | 3.658e-05 | 3.658e-05 | 1.912e-06 |  | 1 |
| rgsaddle-neb vs eon-neb | 11 | 3.658e-05 | 3.658e-05 | 1.912e-06 |  | 1 |
| rgsaddle-neb vs ase-neb | 11 | 3.658e-05 | 3.658e-05 | 1.912e-06 |  | 1 |
| rgsaddle-ci vs eon-ci | 11 | 3.658e-05 | 3.658e-05 | 1.912e-06 | 1.912e-06 | 1 |
| rgsaddle-ci vs ase-ci | 11 | 3.658e-05 | 3.658e-05 | 1.912e-06 |  | 7.396e-07 |
| rgsaddle-dimer vs eon-dimer |  | 0.00201 | 0.001354 | 0.5444 | 0.08086 | 1 |
| rgsaddle-dimer vs ase-dimer |  | 0.0531 | 0.0531 | 0.04639 | 0.233 | 0.6668 |
| rgsaddle-lanczos vs eon-lanczos |  | 0.0004777 | 0.0004777 | 0.09988 | 0.5403 | 1 |

## Failures

| system | method | images | round | status |
| --- | --- | --- | --- | --- |
| pt_hop | eon-dimer |  | 0 | GOOD |
| pt_hop | eon-lanczos |  | 0 | GOOD |
| pt_exchange | rgsaddle-neb | 7 | 0 | stalled |
| pt_exchange | eon-neb | 7 | 0 | BAD_MAX_ITERATIONS |
| pt_exchange | ase-neb | 7 | 0 | max_iterations |
| pt_exchange | rgsaddle-ci | 7 | 0 | stalled |
| pt_exchange | eon-ci | 7 | 0 | GOOD |
| pt_exchange | ase-ci | 7 | 0 | max_iterations |
| pt_exchange | eon-neb | 11 | 0 | GOOD |
| pt_exchange | ase-neb | 11 | 0 | max_iterations |
| pt_exchange | eon-ci | 11 | 0 | GOOD |
| pt_exchange | ase-ci | 11 | 0 | max_iterations |
| pt_exchange | eon-dimer |  | 0 | GOOD |
| pt_exchange | eon-lanczos |  | 0 | GOOD |
| al_vacancy | rgsaddle-dimer |  | 0 | max_iterations |
| al_vacancy | eon-dimer |  | 0 | GOOD |
| al_vacancy | ase-dimer |  | 0 | max_iterations |
| al_vacancy | rgsaddle-lanczos |  | 0 | max_iterations |
| al_vacancy | eon-lanczos |  | 0 | BAD_MAX_ITERATIONS |
| lj7 | eon-neb | 7 | 0 | GOOD |
| lj7 | ase-ci | 7 | 0 | max_iterations |
| lj7 | eon-neb | 11 | 0 | GOOD |
| lj7 | ase-neb | 11 | 0 | max_iterations |
| lj7 | eon-ci | 11 | 0 | GOOD |
| lj7 | eon-dimer |  | 0 | BAD_HIGH_ENERGY |
| lj7 | rgsaddle-lanczos |  | 0 | max_iterations |
| lj7 | eon-lanczos |  | 0 | BAD_HIGH_ENERGY |
| cu4_h | eon-neb | 7 | 0 | GOOD |
| cu4_h | ase-ci | 7 | 0 | max_iterations |
| cu4_h | eon-neb | 11 | 0 | GOOD |
| cu4_h | eon-lanczos |  | 0 | GOOD |
| cu4_h | eon-lanczos |  | 1 | GOOD |
| cu4_h | rgsaddle-lanczos |  | 1 | max_iterations |
| cu4_h | eon-neb | 11 | 1 | GOOD |
| cu4_h | ase-ci | 7 | 1 | max_iterations |
| cu4_h | eon-neb | 7 | 1 | GOOD |
| lj7 | eon-lanczos |  | 1 | GOOD |
| lj7 | rgsaddle-lanczos |  | 1 | max_iterations |
| lj7 | rgsaddle-dimer |  | 1 | max_iterations |
| lj7 | eon-ci | 11 | 1 | GOOD |

336 further failures are in the JSON.

## Build

| component | revision or version |
| --- | --- |
| rgsaddle | 7c1d43280dbb30d84e921592efa41f6c9e6836bf, ABI 1.42, release, C API |
| rgpot | 97ebba0b062128b7dc4637501c4b70ed4944ae9d, 3.7.0, scalar kernels, fast neighbour list on, Fortran pots on |
| eOn | 16434cb979c979a152f4e6750bdcfdfb0bf173be, pyeonclient 0.4.2, project version 3.6.0 |
| ASE | 3.29.0 |
| numpy | 2.5.5 |
| scipy | 1.18.1 |
| rustc | 1.100.0 |
| gcc | 13.3.0 |
| g++ | 13.3.0 |
| linkcell | 0.3.5 with rgsaddle, 0.3.11 with eOn |
| minimage | 0.1.2 with rgsaddle, 0.1.4 with eOn |
| rgmin | 0.6.0 |
| readcon-core | 0.20.0 |

eOn was built with native architecture off and fast math off. Its bundled vector library is not on the timed potential path. The eOn rgpot engine calls capnp directly, so that client was linked with capnp-rpc.
One core, one thread, 12 rounds.
