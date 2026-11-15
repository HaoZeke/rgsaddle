//! Minimum-image convention via [linkcell](https://github.com/d-SEAMS/linkcell).
//!
//! Band and IRC differences go through this module. vesin builds cutoff
//! neighbour lists when a surface needs them; it is not a second MIC.

use ndarray::Array1;

/// Re-export: the periodic cell is [linkcell::Cell], not a private 3x3.
pub use linkcell::Cell;

/// Nearest integer, ties to even, without a libm call.
///
/// `f64::round` lowers to a branchy sequence on baseline x86-64 (no
/// SSE4.1 `roundsd`), which dominates per-pair minimum-image loops. Adding
/// and removing 2^52 rounds in the FPU's default mode instead. It agrees
/// with `round` everywhere except exact half-integers, which go to even.
#[inline(always)]
pub fn nearest_integer(x: f64) -> f64 {
    const SHIFT: f64 = 4_503_599_627_370_496.0;
    if x.abs() < SHIFT {
        let shift = SHIFT.copysign(x);
        (x + shift) - shift
    } else {
        x
    }
}

/// Wrap every per-atom 3-vector in `diff` to its minimum image.
pub fn wrap_difference(cell: &Cell, diff: &mut Array1<f64>) {
    let origin = cell.origin();
    for a in 0..diff.len() / 3 {
        let q = [
            origin[0] + diff[3 * a],
            origin[1] + diff[3 * a + 1],
            origin[2] + diff[3 * a + 2],
        ];
        let fp = cell.fractional(origin);
        let fq = cell.fractional(q);
        let ds = [
            (fq[0] - fp[0]) - (fq[0] - fp[0]).round(),
            (fq[1] - fp[1]) - (fq[1] - fp[1]).round(),
            (fq[2] - fp[2]) - (fq[2] - fp[2]).round(),
        ];
        let cart = cell.cartesian([fp[0] + ds[0], fp[1] + ds[1], fp[2] + ds[2]]);
        diff[3 * a] = cart[0] - origin[0];
        diff[3 * a + 1] = cart[1] - origin[1];
        diff[3 * a + 2] = cart[2] - origin[2];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ortho_wraps_across_the_box() {
        let cell = Cell::ortho(10.0, 10.0, 10.0).unwrap();
        let mut d = Array1::from(vec![9.0, 0.0, 0.0]);
        wrap_difference(&cell, &mut d);
        assert!((d[0] + 1.0).abs() < 1e-12, "{}", d[0]);
    }

    #[test]
    fn nearest_integer_matches_round_off_ties() {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        for _ in 0..200_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
            let exponent = (state % 60) as i32 - 8;
            let x = (2.0 * unit - 1.0) * 2f64.powi(exponent);
            if (x.abs().fract() - 0.5).abs() > 0.0 {
                assert_eq!(nearest_integer(x), x.round(), "{x:e}");
            }
        }
        for x in [
            0.0,
            0.49999999999999994,
            -0.49999999999999994,
            1.0e300,
            -1.0e300,
        ] {
            assert_eq!(nearest_integer(x), x.round(), "{x:e}");
        }
        assert_eq!(nearest_integer(0.5), 0.0);
        assert_eq!(nearest_integer(1.5), 2.0);
        assert_eq!(nearest_integer(-2.5), -2.0);
        assert!(nearest_integer(f64::NAN).is_nan());
        assert_eq!(nearest_integer(f64::INFINITY), f64::INFINITY);
    }
}
