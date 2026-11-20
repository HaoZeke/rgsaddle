# Published LI-NEB comparison data

The Figure 4 and 5 CSV files give the calculated HQTST markers in Figures 4 and 5 of
[Ásgeirsson, Arnaldsson and Jónsson, JCP 148, 102334 (2018)](https://doi.org/10.1063/1.5007180).
They are digitizations of the figures. Use the first column as
`1000 / temperature_in_kelvin`. The ordinates are base-ten logarithms.

- **Figure 4** gives the gas-phase ammonia borane dissociation rate in inverse
  seconds. The 22 open diamonds were located in the 2969 by 1787 pixel image
  embedded on PDF page 9. The lower axis places 1 and 5 at x = 457.5 and
  2934.5 pixels. The left axis places 10 and −30 at y = 99 and 1678.5.
  One pixel corresponds to 0.0016 in inverse temperature and 0.025 in the
  logarithm. Use the lower axis consistently, including for the lowest
  temperature point.
- **Figure 5** gives H diffusion in Ta in square centimetres per second.
  Its nine open squares were read at the centres of the vector paths on PDF
  page 10. The plot frame spans 0 to 12 horizontally and −20 to −8 vertically.
  Equation (29) gives the conversion from a single-hop rate in inverse seconds
  to diffusivity, `D = 9.13e-17 * k`.

Compare a calculation at the marker temperatures using the reactant-normalized
rate. The rgsaddle observable `log_rate_over_reactant` contains the natural
logarithm of the rate **times** the reactant partition function, in the
surface's time units. Subtract the matching log partition function, convert
the rate to inverse seconds, and divide the natural logarithm by `ln(10)`.

## Modified Müller–Brown surface (Figures 2b and 3)

Run `cargo run --example lineb_muller_brown` for the ten-image fixed-energy
path. The initial band is straight. Its two ends are the inward crossings
between the minima and the barrier. The potential uses the three additional
terms in Ásgeirsson's
[KNARR implementation](https://github.com/RagnarB83/ash/blob/55235ddbc752ddb40aa6213ad81ce7cb44d08262/ash/knarr/KNARRcalculator/mb.py).
The inferred turning energy, −0.145, comes from evaluating that surface at
the two endpoints in Figure 2b, with the initial and final minima as axis
landmarks. The shape of the band between the turning points is optimized.

`figure3_profile.csv` gives the printed relative-energy curve at the eight
interior-image displacements calculated by the example. The 1594 by 1245
pixel image on PDF page 8 places displacement 0 and 2 at x = 215.5 and
1576.5, and energy −0.02 and 0.16 at y = 1088 and 22. At each displacement
the ordinate is the median of the pixels on the curve. The RMS difference
between the calculated image energies and these ordinates is 0.00021,
and the largest absolute difference is 0.00037 (about two vertical pixels).

For Figure 2a, pass `left` or `right` to the example after `--`.
The two paths connect the intermediate to its neighbouring minima at an
inferred turning energy of −0.079. Both use ten images, a spring constant
of 1 and an endpoint scale of 1. Registering the plotted left and right
minima to the two calculated minima gives RMS image-centre differences
of 0.49% and 0.22% of the panel width for the left and right paths.
This comparison uses the panel coordinates because Figure 2 has no
labelled coordinate axes.

Pass `direct ring` to write the 50 Cartesian beads around the closed orbit.
The return half retraces the outward path, giving 26 distinct sites.
In the supplied Figure 3 raster, `figure3_ring_centres.csv` records 38
separate red markers, including the extremal marker at displacement 2.
These are eight-connected components with red channel at least 210 and
green and blue at most 15, excluding the legend. Their areas fall between
307 and 416 pixels around the 362-pixel legend marker. The overlapping
1793-pixel cluster near displacement zero is excluded. The published
marker locations and the calculated bead locations therefore differ even
though the underlying ten-image energy profile agrees closely.
