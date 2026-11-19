# Phase 3b solid-state band

Machine id 9eb296774121518f.

Six repeats, one core, one thread. The table is the median and, in parentheses, the linear-percentile interquartile range. Force calls and wall time use every repeat. A barrier, force, stress, width, or saddle index uses the repeats that returned a band. A blank cell has no finite value.

eOn force calls are the sum of forceCallCounter on the distinct potentials held by the images, minus the parent's count from before the band. A potential that keeps a neighbor table is cloned for each interior image, so the parent counter alone counts only the endpoints.

rgsaddle checks the perpendicular width after every accepted band step. eOn exposes only compute(), so its width is the initial path and the cells compute() returns. CP2K is checked on the final replica cells. A width below 2 times the cutoff is fail_cell_width and is not rerun. No returned row recorded that status.

Tolerance 1e-3 is the rgsaddle maximum component. Tolerance 0.01 is the eOn force norm. Each of those two codes is run at both values, as separate rows. CP2K keeps its default band convergence and is labelled native. The spring is 5 eV/A^2. rgsaddle climbing starts at a force factor of 0.5. eOn climbing is 0.5 except on the Bain path, where it is off. The iteration cap is 2000. FIRE uses the library time step. The common frame is one potential evaluation of the returned images: enthalpy is energy plus pressure times volume, the saddle is the highest interior enthalpy, and the atomic limit is 1e-3 eV/A. The stress limit is 1e-3 times the cube root of the saddle volume, in eV/A^3.

| case | code | optimizer | tolerance | converged | iterations | force calls | wall s | native force | native barrier | common barrier | volume | final force | final stress | min width | saddle | force pass | stress pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ar-vacancy | cp2k | diis | native | 6/6 | 70 (0) | 1746 (0) | 10.00084825 (0.0002055855002) | 0.02720849577 (0) | 0.0486202437 (0) | 0.04862017496 (0) | 1280.809096 (0) | 0.00265553756 (0) | 0.0001439267576 (0) | 10.80951372 (0) | 4 (0) | 0 | 6 |
| ar-vacancy | eon | fire | 0.01 | 6/6 | 68 (0) | 478 (0) | 0.0733491275 (0.000158705001) | 0.009852571014 (0) | 0.04697251562 (0) | 0.04697251562 (0) | 1279.62891 (0) | 0.00351783536 (0) | 4.944100901e-05 (0) | 10.75387475 (0) | 4 (0) | 0 | 6 |
| ar-vacancy | eon | fire | 1e-3 | 6/6 | 103 (0) | 723 (0) | 0.107939007 (0.0004639739991) | 0.0009787611877 (0) | 0.04670211265 (0) | 0.04670211265 (0) | 1283.119974 (0) | 0.0002424258654 (0) | 6.285268408e-06 (0) | 10.75422338 (0) | 4 (0) | 6 | 6 |
| ar-vacancy | eon | lbfgs | 0.01 | 6/6 | 23 (0) | 163 (0) | 0.0296714365 (0.00017263875) | 0.00906714788 (0) | 0.04698553839 (0) | 0.04698553839 (0) | 1283.721174 (0) | 0.002827285545 (0) | 2.573284346e-05 (0) | 10.77206151 (0) | 4 (0) | 0 | 6 |
| ar-vacancy | eon | lbfgs | 1e-3 | 6/6 | 40 (0) | 282 (0) | 0.046265108 (0.0002321364982) | 0.0009891849064 (0) | 0.04670039 (0) | 0.04670039 (0) | 1282.960917 (0) | 0.0004906814344 (0) | 5.965817357e-06 (0) | 10.75297353 (0) | 4 (0) | 6 | 6 |
| ar-vacancy | rgsaddle | fire | 0.01 | 6/6 | 32 (0) | 275 (0) | 0.041534092 (0.0002094617521) | 0.009642835448 (0) | 0.04863215552 (0) | 0.04863215552 (0) | 1280.907426 (0) | 0.002655101353 (0) | 0.0001442384578 (0) | 10.81020116 (0) | 4 (0) | 0 | 6 |
| ar-vacancy | rgsaddle | fire | 1e-3 | 6/6 | 154 (0) | 1206 (0) | 0.169872118 (0.0003603292516) | 0.0009720277858 (0) | 0.04672971759 (0) | 0.04672971759 (0) | 1283.020631 (0) | 8.521850347e-05 (0) | 1.451573722e-05 (0) | 10.76123192 (0) | 4 (0) | 6 | 6 |
| bain-tet | cp2k | diis | native | 6/6 | 39 (0) | 756 (0) | 10.00097694 (4.326274939e-05) | 0.1682485773 (0) | 3.615121181 (0) | 3.615120867 (0) | 57.4958219 (0) | 3.81439005e-08 (0) | 8.148856307e-05 (0) | 3.39375 (0) | 2 (0) | 6 | 6 |
| bain-tet | eon | fire | 0.01 | 6/6 | 78 (0) | 548 (0) | 0.200075798 (0.0004401169999) | 0.002923082761 (0) | 3.615120907 (0) | 3.615120907 (0) | 57.39100472 (0) | 0.0001896363647 (0) | 0.0002365630553 (0) | 3.39375 (0) | 2 (0) | 6 | 6 |
| bain-tet | eon | fire | 1e-3 | 6/6 | 145 (0) | 1017 (0) | 0.3681170225 (0.0003086707493) | 0.0002426889304 (0) | 3.615120863 (0) | 3.615120863 (0) | 58.13611777 (0) | 2.853452043e-05 (0) | 1.080058615e-05 (0) | 3.39375 (0) | 3 (0) | 6 | 6 |
| bain-tet | eon | lbfgs | 0.01 | 6/6 | 110 (0) | 772 (0) | 0.2890103295 (0.002166045249) | 0.009068976957 (0) | 3.61512138 (0) | 3.61512138 (0) | 58.20662082 (0) | 0.002219771183 (0) | 2.750331871e-05 (0) | 3.39375 (0) | 3 (0) | 0 | 6 |
| bain-tet | eon | lbfgs | 1e-3 | 6/6 | 120 (0) | 842 (0) | 0.314198873 (0.004096385251) | 0.0007967242561 (0) | 3.615120867 (0) | 3.615120867 (0) | 58.20669322 (0) | 0.0002146138469 (0) | 4.700316473e-06 (0) | 3.39375 (0) | 3 (0) | 6 | 6 |
| bain-tet | rgsaddle | fire | 0.01 | 6/6 | 87 (0) | 716 (0) | 0.2493344585 (0.0008376689984) | 0.008363014232 (0) | 3.615120868 (0) | 3.615120868 (0) | 58.14190493 (0) | 5.501948289e-12 (0) | 8.017828124e-05 (0) | 3.39375 (0) | 3 (0) | 6 | 6 |
| bain-tet | rgsaddle | fire | 1e-3 | 6/6 | 121 (0) | 975 (0) | 0.3376801525 (0.001120119001) | 0.0008502321691 (0) | 3.615120863 (0) | 3.615120863 (0) | 58.1392023 (0) | 1.801281678e-10 (0) | 2.151536144e-05 (0) | 3.39375 (0) | 3 (0) | 6 | 6 |
| tersoff | cp2k | diis | native | 6/6 | 38 (0) | 1188 (0) | 10.00090375 (9.057000079e-05) | 0.1298054294 (0) | 6.500822052 (0) | 6.50082147 (0) | 1281.531824 (0) | 0.002548300524 (0) | 0.006131049675 (0) | 10.862 (0) | 4 (0) | 0 | 6 |
| tersoff | eon | fire | 0.01 | 0/6 | 2000 (0) | 14002 (0) | 0.7558402825 (0.002565766752) | 76.70569799 (0) | 83.44629143 (0) | 26783.36632 (0) | 1281.531824 (0) | 1090.177586 (0) | 28.60322929 (0) | 10.862 (0) | 5 (0) | 0 | 0 |
| tersoff | eon | fire | 1e-3 | 0/6 | 2000 (0) | 14002 (0) | 0.7562122735 (0.005636822251) | 76.70569799 (0) | 83.44629143 (0) | 26783.36632 (0) | 1281.531824 (0) | 1090.177586 (0) | 28.60322929 (0) | 10.862 (0) | 5 (0) | 0 | 0 |
| tersoff | eon | lbfgs | 0.01 | 0/6 | 2000 (0) | 14002 (0) | 0.8526003855 (0.002007157752) | 13.63681912 (0) | 41.49963248 (0) | 28691.64462 (0) | 1281.531824 (0) | 1234.729331 (0) | 30.90579153 (0) | 10.862 (0) | 2 (0) | 0 | 0 |
| tersoff | eon | lbfgs | 1e-3 | 0/6 | 2000 (0) | 14002 (0) | 0.850807317 (0.003609442751) | 13.63681912 (0) | 41.49963248 (0) | 28691.64462 (0) | 1281.531824 (0) | 1234.729331 (0) | 30.90579153 (0) | 10.862 (0) | 2 (0) | 0 | 0 |
| tersoff | rgsaddle | fire | 0.01 | 6/6 | 53 (0) | 558 (0) | 0.091780009 (0.0005821202494) | 0.007784650667 (0) | 6.50081507 (0) | 6.50081507 (0) | 1281.531824 (0) | 0.002352044308 (0) | 0.006143750946 (0) | 10.862 (0) | 4 (0) | 0 | 6 |
| tersoff | rgsaddle | fire | 1e-3 | 6/6 | 87 (0) | 810 (0) | 0.117195415 (0.0002234572521) | 0.0008159116402 (0) | 6.500794668 (0) | 6.500794668 (0) | 1281.531824 (0) | 0.0001167961507 (0) | 0.006177217741 (0) | 10.862 (0) | 4 (0) | 6 | 6 |
| vacancy-vc | cp2k | bfgs | native | 0/6 |  |  | 10.00104038 (0.0001072412488) |  |  |  |  |  |  |  |  | 0 | 0 |
| vacancy-vc | cp2k | cg | native | 0/6 |  |  | 10.00100959 (0.0002222102485) |  |  |  |  |  |  |  |  | 0 | 0 |
| vacancy-vc | cp2k | diis | native | 0/6 | 1 (0) | 150425.5 (603) | 20.00419483 (6.286699863e-05) |  | 5.62095562 (0) |  |  |  |  |  |  | 0 | 0 |
| vacancy-vc | eon | fire | 0.01 | 6/6 | 160 (0) | 1122 (0) | 0.1685193315 (0.0001115807472) | 0.009362755133 (0) | 4.747630471 (0) | 4.747630471 (0) | 32.3390232 (0) | 0.006678181313 (0) | 2.773878971e-05 (0) | 3.160006306 (0) | 4 (0) | 0 | 6 |
| vacancy-vc | eon | fire | 1e-3 | 6/6 | 185 (0) | 1297 (0) | 0.1939220875 (0.001802461501) | 0.0003116669728 (0) | 4.747631628 (0) | 4.747631628 (0) | 32.33901938 (0) | 0.0002185166537 (0) | 1.152470638e-06 (0) | 3.160004383 (0) | 4 (0) | 6 | 6 |
| vacancy-vc | eon | lbfgs | 0.01 | 6/6 | 42 (0) | 296 (0) | 0.0500646485 (0.000381786248) | 0.007989839558 (0) | 4.747631771 (0) | 4.747631771 (0) | 32.33896813 (0) | 0.004113275198 (0) | 0.0003607133814 (0) | 3.159996606 (0) | 4 (0) | 0 | 6 |
| vacancy-vc | eon | lbfgs | 1e-3 | 6/6 | 59 (0) | 415 (0) | 0.0670294455 (0.00163815575) | 0.0008135558844 (0) | 4.747631628 (0) | 4.747631628 (0) | 32.3390179 (0) | 0.0004059582107 (0) | 1.365410453e-05 (0) | 3.160003623 (0) | 4 (0) | 6 | 6 |
| vacancy-vc | rgsaddle | fire | 0.01 | 6/6 | 96 (0) | 786 (0) | 0.112209381 (0.0002675030009) | 0.009380810795 (0) | 4.747632296 (0) | 4.747632296 (0) | 32.33906402 (0) | 0.0005840921176 (0) | 0.0008100156303 (0) | 3.15809479 (0) | 4 (0) | 6 | 6 |
| vacancy-vc | rgsaddle | fire | 1e-3 | 6/6 | 136 (0) | 1087 (0) | 0.155328555 (0.0008200455004) | 0.000961123621 (0) | 4.74763163 (0) | 4.74763163 (0) | 32.33901856 (0) | 9.241597303e-06 (0) | 1.968434773e-05 (0) | 3.15809479 (0) | 4 (0) | 6 | 6 |

## Argon vacancy

The path is a nearest-neighbor hop into a vacancy in a 2x2x2 fcc cell at the shifted-LJ minimum (cutoff 4.8 A, guard 9.6 A, pressure 0). Every returned argon band has a positive common barrier.

rgsaddle FIRE at 1e-3 over eOn FIRE at 0.01 force calls: 2.523012552 (1206 / 478).
At tolerance 1e-3 the same ratio is 1.668049793 (1206 / 723). At tolerance 0.01 it is 0.5753138075 (275 / 478).

At tolerance 1e-3 the three returned barriers span 2.932759375e-05 eV and the volumes span 0.1590567264 A^3. The saddle index is 4. The smallest returned width is 10.75297353 A.

Rows at 0.01, and the CP2K row, miss the common-frame atomic limit. Rows at 1e-3 meet it. Every argon row meets the stress limit.

## Tersoff

Both endpoints were relaxed with the Sella minimum session at maximum-component tolerance 1e-3 and trust delta 0.2 before the band was built. The session residual and the common-frame residual are both recorded. The cell is fixed, so the stress cannot relax.

reactant: status converged, iterations 169, force calls 507, session max component 6.7105478099927548e-04 eV/A, common-frame max component 0.0010409767899663136 eV/A, max enthalpy stress 0.00429345698994875 eV/A^3, limit 0.010862000000000002, wall not retained.
product: status converged, iterations 174, force calls 522, session max component 8.3569130887795799e-04 eV/A, common-frame max component 0.0009999560346469205 eV/A, max enthalpy stress 0.004293402788222764 eV/A^3, limit 0.010862000000000002, wall 1.932813 s.

After that relaxation, rgsaddle FIRE at 1e-3 returns 6/6 and passes force on 6 and stress on 6. At 0.01 it passes force on 0 and stress on 6. The 0.01 frames that miss the atomic limit are stored with the logs as tersoff-rgsaddle-miss/0.01/rep-0 through rep-5, each with final.geom, common.txt, result.txt, and config.txt.

eOn FIRE and L-BFGS reach the iteration cap at both tolerances. Their force-call medians are 14002 and 14002. The two tolerances match because the cap is reached first. CP2K returns a barrier in line with rgsaddle and misses the atomic limit.

## Vacancy

CP2K DIIS does not return. The harness stops a line search that keeps rewriting one energy, so the force calls are the counts at the stop, in repeat order: 150601, 142693, 150761, 150250, 150074, 151071. The median is 150425.5 (603).

CG and BFGS were submitted as band optimizer rows. Every repeat exits with status exit -6. The band output on each repeat says invalid value for enumeration, for CG and for BFGS. This build's band optimizer accepts MD and DIIS. The DIIS row is kept.

eOn at 0.01 misses the atomic limit and meets the stress limit. eOn at 1e-3 meets both. rgsaddle meets both at both tolerances. The returned barriers agree.

## Bain

eOn L-BFGS at 0.01 passes the atomic test on 0 of 6. The median final atomic force is 0.002219771183 eV/A. The same optimizer at 1e-3 passes on 6 of 6, with median final atomic force 0.0002146138469 eV/A. The 0/6 result is the 0.01 setting. Stress passes on every Bain row.

At 0.01, CP2K and eOn FIRE place the saddle on image 2, while rgsaddle and eOn L-BFGS place it on image 3. The enthalpies agree and the volumes do not, because the winning image differs. At 1e-3, eOn FIRE also places the saddle on image 3.

