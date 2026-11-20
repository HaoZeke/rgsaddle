# Published LI-NEB comparison data

The two CSV files give the calculated HQTST markers in Figures 4 and 5 of
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
