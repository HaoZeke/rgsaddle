//! Shared sampling statistics: the noise stream and block standard errors.

use crate::SaddleError;

/// Automated equilibration and correlation report for one scalar series.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct EquilibrationReport {
    /// Records discarded after the engine's minimum warm-up floor.
    pub discard: usize,
    /// Engine steps discarded: floor plus `discard * stride`.
    pub equilibration_steps: usize,
    /// Integrated autocorrelation time of the retained tail, in records.
    pub tau: f64,
    /// `(N - discard) / g`, with `g = max(1, 2 tau)`.
    pub n_eff: f64,
    /// Absolute difference between the retained tail's half means.
    pub split_half_drift: f64,
    /// Root-sum-square block standard error of the half means.
    pub split_half_error: f64,
    /// Multiple of [`Self::split_half_error`] used by the drift test.
    pub sigma_multiple: f64,
    /// Whether the split-half drift test passed.
    pub drift_passed: bool,
}

/// Chodera-style automated equilibration followed by a split-half drift test.
///
/// Candidate discards maximize `(N-t0)/g(t0)`. For long series a
/// coarse-to-fine scan bounds the cost of repeated autocorrelation estimates.
/// The drift threshold is four block standard errors: a conservative
/// four-sigma check that rejects visible nonstationarity without turning an
/// ordinary finite-sample fluctuation into a failure.
pub(crate) fn equilibrate(
    values: &[f64],
    warmup_floor: usize,
    stride: usize,
    blocks: usize,
) -> Result<EquilibrationReport, SaddleError> {
    const K: f64 = 4.0;
    if values.len() < 4 {
        return Err(SaddleError::Invalid(
            "equilibration detection needs at least four records".into(),
        ));
    }
    // Do not let the optimizer manufacture stationarity from a tiny suffix.
    // A quarter of the trajectory also leaves enough data for the
    // independent split-half diagnostic.
    let minimum_tail = (2 * blocks)
        .max(8)
        .max(values.len().div_ceil(4))
        .min(values.len());
    let last = values.len().saturating_sub(minimum_tail);
    let step = (last / 128).max(1);
    let mut best = (0usize, 0.0, 0.5);
    let evaluate = |t0: usize| {
        let tau = autocorr_time(&values[t0..]);
        let g = (2.0 * tau).max(1.0);
        ((values.len() - t0) as f64 / g, tau)
    };
    for t0 in (0..=last).step_by(step) {
        let (n_eff, tau) = evaluate(t0);
        if n_eff > best.1 {
            best = (t0, n_eff, tau);
        }
    }
    let lo = best.0.saturating_sub(step);
    let hi = (best.0 + step).min(last);
    for t0 in lo..=hi {
        let (n_eff, tau) = evaluate(t0);
        if n_eff > best.1 {
            best = (t0, n_eff, tau);
        }
    }
    let tail = &values[best.0..];
    let middle = tail.len() / 2;
    let halves = [&tail[..middle], &tail[middle..]];
    let half_blocks = blocks.max(2).min(middle.max(2));
    let (mean_a, error_a, _, _) = block_mean(halves[0], half_blocks);
    let (mean_b, error_b, _, _) = block_mean(halves[1], half_blocks);
    let drift = (mean_a - mean_b).abs();
    let error = error_a.hypot(error_b);
    let roundoff = 32.0 * f64::EPSILON * mean_a.abs().max(mean_b.abs()).max(1.0);
    let passed = drift <= (K * error).max(roundoff);
    let report = EquilibrationReport {
        discard: best.0,
        equilibration_steps: warmup_floor.saturating_add(best.0.saturating_mul(stride)),
        tau: best.2,
        n_eff: best.1,
        split_half_drift: drift,
        split_half_error: error,
        sigma_multiple: K,
        drift_passed: passed,
    };
    let correlation_width = ((2.0 * report.tau).ceil() as usize).max(1);
    let block_width = tail.len().div_ceil(blocks).max(correlation_width);
    let available_blocks = tail.len() / block_width;
    if available_blocks < 2 {
        return Err(SaddleError::InsufficientEffectiveSamples {
            equilibration_steps: report.equilibration_steps,
            n_eff: report.n_eff,
            required: 2.0 * correlation_width as f64,
        });
    }
    if !passed {
        return Err(SaddleError::NotEquilibrated {
            equilibration_steps: report.equilibration_steps,
            drift,
            standard_error: error,
            sigma_multiple: K,
        });
    }
    Ok(report)
}

/// Block mean of a correlated series: `(mean, standard error, τ, n_eff)`.
///
/// The block is at least twice the integrated autocorrelation time, so a
/// correlated series is not treated as independent.
pub(crate) fn block_mean(values: &[f64], blocks: usize) -> (f64, f64, f64, f64) {
    let tau = autocorr_time(values);
    let correlated = (2.0 * tau).ceil() as usize;
    let width = (values.len() / blocks).max(correlated).max(1);
    let n_blocks = (values.len() / width).max(2).min(values.len());
    let width = values.len() / n_blocks;
    let mut means = Vec::with_capacity(n_blocks);
    for block in 0..n_blocks {
        let slice = &values[block * width..(block + 1) * width];
        let mean = slice.iter().sum::<f64>() / slice.len() as f64;
        means.push(mean);
    }
    let mean = means.iter().sum::<f64>() / n_blocks as f64;
    let variance = if n_blocks < 2 {
        0.0
    } else {
        means
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f64>()
            / (n_blocks - 1) as f64
    };
    let n_eff = values.len() as f64 / (2.0 * tau).max(1.0);
    (mean, (variance / n_blocks as f64).sqrt(), tau, n_eff)
}

/// Sokal's integrated autocorrelation time, in samples. The sum stops at
/// the first non-positive autocorrelation.
pub(crate) fn autocorr_time(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 4 {
        return 0.5;
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    let mut variance = 0.0;
    for value in values {
        let d = value - mean;
        variance += d * d;
    }
    variance /= n as f64;
    if variance < 1e-30 {
        return 0.5;
    }
    let mut tau = 0.5;
    let mut lag = 1usize;
    // A 6-point half has `n / 4 == 1`, so `lag < n / 4` never runs. Keep lag 1
    // there; longer series stay on the original cutoff.
    while lag < n / 4 || (n / 4 == 1 && lag == 1) {
        let mut cov = 0.0;
        for i in 0..(n - lag) {
            cov += (values[i] - mean) * (values[i + lag] - mean);
        }
        cov /= (n - lag) as f64;
        let rho = cov / variance;
        if rho <= 0.0 {
            break;
        }
        tau += rho;
        lag += 1;
    }
    tau.max(0.5)
}

/// xoshiro256++. The stream is deterministic for a seed.
pub(crate) struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut s = [0; 4];
        for slot in &mut s {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut mixed = z;
            mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            *slot = mixed ^ (mixed >> 31);
        }
        Self { s }
    }

    pub(crate) fn u64(&mut self) -> u64 {
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform on `[0, 1)` with 53 random bits.
    pub(crate) fn uniform(&mut self) -> f64 {
        (self.u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))
    }

    pub(crate) fn normal(&mut self) -> f64 {
        let u1 = ((self.u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))).max(1.0e-16);
        let u2 = (self.u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64));
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn harmonic_ou_relaxation_is_discarded() {
        // Overdamped harmonic relaxation with unit sampling interval:
        // x(t+1) = rho x(t) + noise and exact relaxation time -1/ln(rho).
        let relaxation = 20.0;
        let rho = (-1.0_f64 / relaxation).exp();
        let sigma = (1.0 - rho * rho).sqrt();
        let mut rng = Rng::new(17);
        let mut x = 100.0;
        let mut values = Vec::with_capacity(2_000);
        for _ in 0..2_000 {
            x = rho * x + sigma * rng.normal();
            values.push(x);
        }
        let report = equilibrate(&values, relaxation as usize, 1, 10).unwrap();
        assert_eq!(report.equilibration_steps, relaxation as usize);
        assert!(report.tau > 5.0 && report.tau < 60.0, "tau={}", report.tau);
        assert!(report.drift_passed);
    }

    #[test]
    fn a_deliberately_drifting_series_is_rejected() {
        let values: Vec<f64> = (0..400)
            .map(|i| 0.01 * i as f64 + if i % 2 == 0 { 0.001 } else { -0.001 })
            .collect();
        let result = equilibrate(&values, 25, 2, 8);
        assert!(
            matches!(
                result,
                Err(SaddleError::NotEquilibrated { .. }
                    | SaddleError::InsufficientEffectiveSamples { .. })
            ),
            "{result:?}"
        );
    }
}
