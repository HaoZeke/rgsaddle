//! Equal arc-length reparametrization on the polyline and on the
//! not-a-knot spline: spacing, exactness, endpoints, and edge cases.

use ndarray::{Array2, array};
use rgsaddle::reparam::segment_lengths;
use rgsaddle::{Cell, Reparametrization, reparametrize};

const SCHEMES: [Reparametrization; 2] = [Reparametrization::Linear, Reparametrization::NotAKnot];

fn spread(lengths: &[f64]) -> f64 {
    let max = lengths.iter().copied().fold(f64::MIN, f64::max);
    let min = lengths.iter().copied().fold(f64::MAX, f64::min);
    (max - min) / (lengths.iter().sum::<f64>() / lengths.len() as f64)
}

/// Images on the cubic curve `(t, t^3 - t, 0.5 t^2)` at
/// `t = -1 + 2 u^power`, `u` uniform.
fn cubic_band(n: usize, power: f64) -> Array2<f64> {
    Array2::from_shape_fn((n, 3), |(i, c)| {
        let u = i as f64 / (n - 1) as f64;
        let t = -1.0 + 2.0 * u.powf(power);
        match c {
            0 => t,
            1 => t * t * t - t,
            _ => 0.5 * t * t,
        }
    })
}

/// Distance from `p` to the curve of [`cubic_band`], by a dense scan
/// refined with Newton on the squared distance.
fn distance_to_cubic(p: &[f64]) -> f64 {
    let curve = |t: f64| [t, t * t * t - t, 0.5 * t * t];
    let d2 = |t: f64| {
        let q = curve(t);
        (0..3).map(|c| (q[c] - p[c]).powi(2)).sum::<f64>()
    };
    let mut best = -1.0;
    for k in 0..=4000 {
        let t = -1.0 + 2.0 * k as f64 / 4000.0;
        if d2(t) < d2(best) {
            best = t;
        }
    }
    let mut t = best;
    for _ in 0..50 {
        let h = 1e-6;
        let g = (d2(t + h) - d2(t - h)) / (2.0 * h);
        let c = (d2(t + h) - 2.0 * d2(t) + d2(t - h)) / (h * h);
        if c <= 0.0 {
            break;
        }
        t -= g / c;
    }
    d2(t).sqrt()
}

#[test]
fn a_straight_band_is_spaced_exactly_by_both_schemes() {
    for scheme in SCHEMES {
        let mut band = array![
            [0.0, 0.0, 0.0],
            [0.05, 0.1, -0.05],
            [0.1, 0.2, -0.1],
            [0.6, 1.2, -0.6],
            [0.65, 1.3, -0.65],
            [1.0, 2.0, -1.0]
        ];
        let ends = (band.row(0).to_owned(), band.row(5).to_owned());
        reparametrize(&mut band, None, None, scheme);
        assert_eq!(band.row(0), ends.0);
        assert_eq!(band.row(5), ends.1);
        for (j, row) in band.outer_iter().enumerate() {
            let u = j as f64 / 5.0;
            for (c, scale) in [1.0, 2.0, -1.0].into_iter().enumerate() {
                assert!((row[c] - scale * u).abs() < 1e-12, "{scheme:?} {j} {c}");
            }
        }
    }
}

fn off_curve(band: &Array2<f64>) -> f64 {
    (1..band.nrows() - 1)
        .map(|j| distance_to_cubic(band.row(j).as_slice().unwrap()))
        .fold(0.0, f64::max)
}

#[test]
fn the_spline_keeps_a_cubic_band_on_its_curve() {
    let mut spline = cubic_band(17, 1.0);
    let mut linear = spline.clone();
    reparametrize(&mut spline, None, None, Reparametrization::NotAKnot);
    reparametrize(&mut linear, None, None, Reparametrization::Linear);
    let (on_spline, on_linear) = (off_curve(&spline), off_curve(&linear));
    assert!(on_spline < 0.2 * on_linear, "{on_spline} vs {on_linear}");
    assert!(on_spline < 2e-3, "{on_spline}");
}

#[test]
fn the_spline_off_curve_error_falls_as_the_fourth_power() {
    let error = |n: usize, scheme| {
        let mut band = cubic_band(n, 1.0);
        reparametrize(&mut band, None, None, scheme);
        off_curve(&band)
    };
    // Four times the images: a fourth-order placement gains about 256,
    // the polyline's second order about 16.
    let spline = error(9, Reparametrization::NotAKnot) / error(33, Reparametrization::NotAKnot);
    let linear = error(9, Reparametrization::Linear) / error(33, Reparametrization::Linear);
    assert!(spline > 100.0, "spline gain {spline}");
    assert!(linear < 30.0, "linear gain {linear}");
}

#[test]
fn one_spline_pass_spaces_an_uneven_band_at_equal_arc_length() {
    for n in [17, 33] {
        let mut band = cubic_band(n, 1.5);
        let start = spread(&segment_lengths(&band, None, Reparametrization::NotAKnot));
        reparametrize(&mut band, None, None, Reparametrization::NotAKnot);
        let arcs = segment_lengths(&band, None, Reparametrization::NotAKnot);
        assert!(start > 1.0, "{start}");
        assert!(spread(&arcs) < 2e-3, "{n}: {arcs:?}");
        // The arc of the new band is the arc of the old spline.
        let total: f64 = arcs.iter().sum();
        let before: f64 = segment_lengths(&cubic_band(n, 1.5), None, Reparametrization::NotAKnot)
            .iter()
            .sum();
        assert!(
            (total - before).abs() < 1e-4 * before,
            "{total} vs {before}"
        );
    }
}

#[test]
fn endpoints_and_the_anchor_keep_their_bits() {
    for scheme in SCHEMES {
        let mut band = cubic_band(17, 1.5);
        let before = band.clone();
        reparametrize(&mut band, Some(6), None, scheme);
        for row in [0, 6, 16] {
            assert!(
                band.row(row)
                    .iter()
                    .zip(before.row(row))
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{scheme:?} row {row}"
            );
        }
        let lengths = segment_lengths(&band, None, scheme);
        assert!(spread(&lengths[..6]) < 0.05, "{scheme:?} {lengths:?}");
        assert!(spread(&lengths[6..]) < 0.05, "{scheme:?} {lengths:?}");
    }
}

#[test]
fn two_and_three_images_fall_back_to_lower_order() {
    let mut two = array![[0.0, 0.0], [1.0, 3.0]];
    let copy = two.clone();
    reparametrize(&mut two, None, None, Reparametrization::NotAKnot);
    assert_eq!(two, copy);
    // Three images: the parabola through them, at half the arc length.
    let mut three = array![[0.0, 0.0], [0.2, 0.04], [1.0, 1.0]];
    let mut linear = three.clone();
    reparametrize(&mut three, None, None, Reparametrization::NotAKnot);
    reparametrize(&mut linear, None, None, Reparametrization::Linear);
    assert_eq!(three.row(0), array![0.0, 0.0]);
    assert_eq!(three.row(2), array![1.0, 1.0]);
    assert!(three.iter().all(|v| v.is_finite()));
    // The parabola through the three images in their chord arc `t`,
    // and the middle image at half its arc length.
    let (s1, s2) = (0.2_f64.hypot(0.04), 0.8_f64.hypot(0.96));
    let knots = [0.0, s1, s1 + s2];
    let lagrange = |t: f64, y: [f64; 3]| {
        (0..3)
            .map(|i| {
                let mut w = y[i];
                for k in (0..3).filter(|&k| k != i) {
                    w *= (t - knots[k]) / (knots[i] - knots[k]);
                }
                w
            })
            .sum::<f64>()
    };
    let point = |t: f64| [lagrange(t, [0.0, 0.2, 1.0]), lagrange(t, [0.0, 0.04, 1.0])];
    let steps = 200_000;
    let mut arc = vec![0.0; steps + 1];
    for k in 1..=steps {
        let (a, b) = (
            point(knots[2] * (k - 1) as f64 / steps as f64),
            point(knots[2] * k as f64 / steps as f64),
        );
        arc[k] = arc[k - 1] + (a[0] - b[0]).hypot(a[1] - b[1]);
    }
    let half = (0..=steps)
        .min_by(|&i, &j| {
            (arc[i] - 0.5 * arc[steps])
                .abs()
                .total_cmp(&(arc[j] - 0.5 * arc[steps]).abs())
        })
        .unwrap();
    let expected = point(knots[2] * half as f64 / steps as f64);
    assert!(
        (three[(1, 0)] - expected[0]).abs() < 1e-5,
        "{three} vs {expected:?}"
    );
    assert!(
        (three[(1, 1)] - expected[1]).abs() < 1e-5,
        "{three} vs {expected:?}"
    );
    assert!((three[(1, 0)] - linear[(1, 0)]).abs() > 1e-3);
}

#[test]
fn coincident_images_do_not_break_the_fit() {
    let mut band = array![
        [0.0, 0.0],
        [0.0, 0.0],
        [0.5, 0.3],
        [0.5, 0.3 + 1e-15],
        [1.0, 0.4],
        [1.5, 0.3],
        [2.0, 0.0],
        [2.0, 0.0]
    ];
    reparametrize(&mut band, None, None, Reparametrization::NotAKnot);
    assert!(band.iter().all(|v| v.is_finite()), "{band}");
    assert_eq!(band.row(0), array![0.0, 0.0]);
    assert_eq!(band.row(7), array![2.0, 0.0]);
    let lengths = segment_lengths(&band, None, Reparametrization::NotAKnot);
    let distinct: Vec<f64> = lengths.iter().copied().filter(|&l| l > 1e-9).collect();
    assert_eq!(distinct.len(), 7, "{lengths:?}");
    assert!(spread(&distinct) < 0.05, "{lengths:?}");
    // Every image coincident: nothing to place.
    let mut point = Array2::from_elem((5, 3), 0.25);
    let copy = point.clone();
    reparametrize(&mut point, None, None, Reparametrization::NotAKnot);
    assert_eq!(point, copy);
}

#[test]
fn a_periodic_band_is_unwrapped_by_minimum_images() {
    let l = 4.0;
    let cell = Cell([[l, 0.0, 0.0], [0.0, l, 0.0], [0.0, 0.0, l]]);
    for scheme in SCHEMES {
        // One atom crossing the x face, the images stored wrapped.
        let unwrapped = cubic_band(9, 1.5).mapv(|v| 0.5 * v) + &array![[3.6, 1.0, 1.0]];
        let mut wrapped = unwrapped.mapv(|v| v.rem_euclid(l));
        let mut reference = unwrapped.clone();
        reparametrize(&mut wrapped, None, Some(&cell), scheme);
        reparametrize(&mut reference, None, None, scheme);
        for (got, want) in wrapped.outer_iter().zip(reference.outer_iter()) {
            for c in 0..3 {
                let mut d = got[c] - want[c];
                d -= l * (d / l).round();
                assert!(d.abs() < 1e-12, "{scheme:?} {got} vs {want}");
            }
        }
        let wrapped_chords = segment_lengths(&wrapped, Some(&cell), scheme);
        let plain_chords = segment_lengths(&reference, None, scheme);
        for (a, b) in wrapped_chords.iter().zip(&plain_chords) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}
