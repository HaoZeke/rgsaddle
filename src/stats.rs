//! Shared sampling statistics: the noise stream and block standard errors.

/// Block mean of a correlated series: `(mean, standard error, τ, n_eff)`.
///
/// The block is at least twice the integrated autocorrelation time, so a
/// correlated series is not treated as independent.
pub(crate) fn block_mean(values: &[f64], blocks: usize) -> (f64, f64, f64, f64) {
    let tau = autocorr_time(values);
    let (mean, error) = blocked(values, blocks, tau);
    let n_eff = values.len() as f64 / (2.0 * tau).max(1.0);
    (mean, error, tau, n_eff)
}

/// [`block_mean`] with `τ` from Sokal's window, [`windowed_autocorr_time`],
/// and a standard error no smaller than `sqrt(2 τ var / n)`, which a few
/// blocks that agree by chance would otherwise undercut.
pub(crate) fn windowed_block_mean(values: &[f64], blocks: usize) -> (f64, f64, f64, f64) {
    let tau = windowed_autocorr_time(values);
    let (mean, error) = blocked(values, blocks, tau);
    let count = values.len() as f64;
    let spread = values
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f64>()
        / count;
    let error = error.max((2.0 * tau * spread / count).sqrt());
    let n_eff = count / (2.0 * tau).max(1.0);
    (mean, error, tau, n_eff)
}

/// Mean of the block means and its standard error, the blocks at least
/// `2 τ` samples long.
fn blocked(values: &[f64], blocks: usize, tau: f64) -> (f64, f64) {
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
    (mean, (variance / n_blocks as f64).sqrt())
}

/// Sokal's integrated autocorrelation time, in samples. The sum stops at
/// the first non-positive autocorrelation.
pub(crate) fn autocorr_time(values: &[f64]) -> f64 {
    let Some((mean, variance)) = moments(values) else {
        return 0.5;
    };
    let n = values.len();
    let mut tau = 0.5;
    let mut lag = 1usize;
    while lag < n / 4 {
        let rho = covariance(values, mean, lag) / variance;
        if rho <= 0.0 {
            break;
        }
        tau += rho;
        lag += 1;
    }
    tau.max(0.5)
}

/// Integrated autocorrelation time in samples, `1/2 + Σ ρ(t)` over the
/// lags up to the first `M ≥ 5 τ(M)` (Sokal's window), and below a
/// quarter of the series.
pub(crate) fn windowed_autocorr_time(values: &[f64]) -> f64 {
    let Some((mean, variance)) = moments(values) else {
        return 0.5;
    };
    let mut tau = 0.5;
    for lag in 1..values.len() / 4 {
        tau += covariance(values, mean, lag) / variance;
        if lag as f64 >= 5.0 * tau {
            break;
        }
    }
    tau.max(0.5)
}

/// Mean and variance of a series of at least four values that is not
/// constant.
fn moments(values: &[f64]) -> Option<(f64, f64)> {
    let n = values.len();
    if n < 4 {
        return None;
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    let mut variance = 0.0;
    for value in values {
        let d = value - mean;
        variance += d * d;
    }
    variance /= n as f64;
    if variance < 1e-30 {
        return None;
    }
    Some((mean, variance))
}

/// Autocovariance at `lag` about `mean`.
fn covariance(values: &[f64], mean: f64, lag: usize) -> f64 {
    let n = values.len();
    let mut cov = 0.0;
    for i in 0..(n - lag) {
        cov += (values[i] - mean) * (values[i + lag] - mean);
    }
    cov / (n - lag) as f64
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
