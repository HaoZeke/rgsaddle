//! Wigner-Seitz occupancy of a reference crystal in a periodic orthorhombic cell.
//!
//! Every atom belongs to the reference site nearest to it, after the change of
//! the cell edges since the reference and the drift of the centroid are taken
//! out. A defect-free configuration has every site singly occupied; a vacancy
//! is an empty site and an interstitial a site holding two atoms. Nothing here
//! depends on the material or the lattice.
//!
//! Atoms vibrate across Wigner-Seitz boundaries rarely, but a configuration can
//! still read perfect for one sample on its way between two defected states.
//! [`Persistence`] accepts a condition only after it has held for a run of
//! consecutive samples and reports the sample at which that run began.

use crate::error::SaddleError;

/// Site counts of one configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Occupancy {
    /// Sites with no atom.
    pub vacant: usize,
    /// Sites with more than one atom.
    pub multiple: usize,
}

impl Occupancy {
    /// Every site holds exactly one atom.
    pub fn is_perfect(&self) -> bool {
        self.vacant == 0 && self.multiple == 0
    }

    /// Sites whose count is not one.
    pub fn defect_sites(&self) -> usize {
        self.vacant + self.multiple
    }
}

/// Reference sites, binned for nearest-site lookup, and the counts of the
/// last configuration.
#[derive(Clone, Debug)]
pub struct SiteOccupancy {
    sites: Vec<[f64; 3]>,
    lengths: [f64; 3],
    origin: [f64; 3],
    nside: [usize; 3],
    start: Vec<usize>,
    members: Vec<usize>,
    reach: f64,
    counts: Vec<u32>,
}

fn orthorhombic_lengths(cell: [[f64; 3]; 3]) -> Option<[f64; 3]> {
    let off = [
        cell[0][1], cell[0][2], cell[1][0], cell[1][2], cell[2][0], cell[2][1],
    ];
    let lengths = [cell[0][0], cell[1][1], cell[2][2]];
    let flat = off.iter().all(|value| *value == 0.0);
    let positive = lengths
        .iter()
        .all(|length| length.is_finite() && *length > 0.0);
    (flat && positive).then_some(lengths)
}

/// Unweighted mean position of flat `[x0, y0, z0, x1, ...]` coordinates.
pub fn centroid(positions: &[f64]) -> [f64; 3] {
    let n = (positions.len() / 3).max(1) as f64;
    let mut sum = [0.0; 3];
    for atom in positions.chunks_exact(3) {
        for axis in 0..3 {
            sum[axis] += atom[axis];
        }
    }
    sum.map(|value| value / n)
}

impl SiteOccupancy {
    /// Sites are flat `[x0, y0, z0, x1, ...]` coordinates in `cell`, which must
    /// be orthorhombic. `reference_atoms` are the atoms at the reference time;
    /// their centroid is the origin from which drift is measured, so the atoms
    /// need not sit on the sites.
    pub fn new(
        sites: &[f64],
        cell: [[f64; 3]; 3],
        reference_atoms: &[f64],
    ) -> Result<Self, SaddleError> {
        let lengths = orthorhombic_lengths(cell)
            .ok_or_else(|| SaddleError::Invalid("cell is not orthorhombic".into()))?;
        if sites.is_empty() || sites.len() % 3 != 0 {
            return Err(SaddleError::Shape(format!(
                "{} site coordinates",
                sites.len()
            )));
        }
        if sites
            .iter()
            .chain(reference_atoms)
            .any(|value| !value.is_finite())
        {
            return Err(SaddleError::NonFinite("site occupancy reference"));
        }
        let sites: Vec<[f64; 3]> = sites
            .chunks_exact(3)
            .map(|site| [0, 1, 2].map(|axis| site[axis].rem_euclid(lengths[axis])))
            .collect();
        let volume = lengths[0] * lengths[1] * lengths[2];
        let spacing = (volume / sites.len() as f64).cbrt();
        let nside = lengths.map(|length| ((length / (2.0 * spacing)).floor() as usize).max(1));
        let bin_count = nside[0] * nside[1] * nside[2];
        let bin_of = |site: &[f64; 3]| {
            let index = [0, 1, 2].map(|axis| {
                ((site[axis] / lengths[axis] * nside[axis] as f64) as usize).min(nside[axis] - 1)
            });
            (index[0] * nside[1] + index[1]) * nside[2] + index[2]
        };
        let mut start = vec![0usize; bin_count + 1];
        for site in &sites {
            start[bin_of(site) + 1] += 1;
        }
        for bin in 0..bin_count {
            start[bin + 1] += start[bin];
        }
        let mut fill = start.clone();
        let mut members = vec![0usize; sites.len()];
        for (index, site) in sites.iter().enumerate() {
            let bin = bin_of(site);
            members[fill[bin]] = index;
            fill[bin] += 1;
        }
        let reach = [0, 1, 2]
            .map(|axis| lengths[axis] / nside[axis] as f64)
            .into_iter()
            .fold(f64::INFINITY, f64::min);
        let counts = vec![0; sites.len()];
        Ok(Self {
            sites,
            lengths,
            origin: centroid(reference_atoms),
            nside,
            start,
            members,
            reach,
            counts,
        })
    }

    pub fn site_count(&self) -> usize {
        self.sites.len()
    }

    /// Atoms per site from the last call to [`SiteOccupancy::occupancy`].
    pub fn counts(&self) -> &[u32] {
        &self.counts
    }

    /// Nearest site to a point in the reference cell, minimum image.
    ///
    /// The 27 bins around the point cover every site within `reach`; a point
    /// whose nearest site lies farther falls back to a scan of all sites.
    fn nearest(&self, point: [f64; 3]) -> usize {
        let home = [0, 1, 2].map(|axis| {
            ((point[axis] / self.lengths[axis] * self.nside[axis] as f64) as usize)
                .min(self.nside[axis] - 1)
        });
        let mut best = (f64::INFINITY, 0usize);
        let mut seen = [usize::MAX; 27];
        let mut count = 0;
        for ox in [-1i64, 0, 1] {
            for oy in [-1i64, 0, 1] {
                for oz in [-1i64, 0, 1] {
                    let offset = [ox, oy, oz];
                    let index = [0, 1, 2].map(|axis| {
                        (home[axis] as i64 + offset[axis]).rem_euclid(self.nside[axis] as i64)
                            as usize
                    });
                    let bin = (index[0] * self.nside[1] + index[1]) * self.nside[2] + index[2];
                    if seen[..count].contains(&bin) {
                        continue;
                    }
                    seen[count] = bin;
                    count += 1;
                    for &site in &self.members[self.start[bin]..self.start[bin + 1]] {
                        let distance = self.distance2(point, site);
                        if distance < best.0 {
                            best = (distance, site);
                        }
                    }
                }
            }
        }
        if best.0 > self.reach * self.reach {
            for site in 0..self.sites.len() {
                let distance = self.distance2(point, site);
                if distance < best.0 {
                    best = (distance, site);
                }
            }
        }
        best.1
    }

    fn distance2(&self, point: [f64; 3], site: usize) -> f64 {
        let mut sum = 0.0;
        for axis in 0..3 {
            let mut d = point[axis] - self.sites[site][axis];
            d -= self.lengths[axis] * (d / self.lengths[axis]).round();
            sum += d * d;
        }
        sum
    }

    /// Counts atoms per site for `positions` in `cell`.
    ///
    /// Each axis of `cell` is a uniform stretch of the reference edge; the
    /// atoms are mapped back by that stretch after the centroid drift since
    /// the reference is removed.
    pub fn occupancy(
        &mut self,
        positions: &[f64],
        cell: [[f64; 3]; 3],
    ) -> Result<Occupancy, SaddleError> {
        let lengths = orthorhombic_lengths(cell)
            .ok_or_else(|| SaddleError::Invalid("cell is not orthorhombic".into()))?;
        if positions.len() % 3 != 0 {
            return Err(SaddleError::Shape(format!(
                "{} coordinates",
                positions.len()
            )));
        }
        let scale = [0, 1, 2].map(|axis| lengths[axis] / self.lengths[axis]);
        let now = centroid(positions);
        let shift = [0, 1, 2].map(|axis| now[axis] - self.origin[axis] * scale[axis]);
        self.counts.iter_mut().for_each(|count| *count = 0);
        for atom in positions.chunks_exact(3) {
            let point = [0, 1, 2].map(|axis| {
                ((atom[axis] - shift[axis]) / scale[axis]).rem_euclid(self.lengths[axis])
            });
            if point.iter().any(|value| !value.is_finite()) {
                return Err(SaddleError::NonFinite("site occupancy positions"));
            }
            let site = self.nearest(point);
            self.counts[site] = self.counts[site].saturating_add(1);
        }
        let vacant = self.counts.iter().filter(|count| **count == 0).count();
        let multiple = self.counts.iter().filter(|count| **count > 1).count();
        Ok(Occupancy { vacant, multiple })
    }
}

/// First sample at which a condition began to hold for `confirm` consecutive
/// samples.
#[derive(Clone, Debug)]
pub struct Persistence {
    confirm: usize,
    run: usize,
    began: usize,
}

impl Persistence {
    /// `confirm` of zero is treated as one: a single sample confirms.
    pub fn new(confirm: usize) -> Self {
        Self {
            confirm: confirm.max(1),
            run: 0,
            began: 0,
        }
    }

    /// Records whether the condition holds at sample `index`. Returns the
    /// index at which the current run began once it is `confirm` long.
    pub fn observe(&mut self, index: usize, holds: bool) -> Option<usize> {
        if !holds {
            self.run = 0;
            return None;
        }
        if self.run == 0 {
            self.began = index;
        }
        self.run += 1;
        (self.run >= self.confirm).then_some(self.began)
    }

    /// A run has started but is not yet confirmed.
    pub fn pending(&self) -> bool {
        self.run > 0 && self.run < self.confirm
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_cubic(n: usize, a: f64) -> (Vec<f64>, [[f64; 3]; 3]) {
        let mut values = Vec::new();
        for ix in 0..n {
            for iy in 0..n {
                for iz in 0..n {
                    values.extend([ix as f64 * a, iy as f64 * a, iz as f64 * a]);
                }
            }
        }
        let length = n as f64 * a;
        (
            values,
            [[length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]],
        )
    }

    fn jitter(state: &mut u64) -> f64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (*state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }

    #[test]
    fn a_shaken_stretched_and_drifted_crystal_is_perfect() {
        let (sites, cell) = simple_cubic(6, 2.0);
        let mut detector = SiteOccupancy::new(&sites, cell, &sites).unwrap();
        let stretch = [1.01, 0.995, 1.02];
        let drift = [0.7, -1.3, 13.1];
        let mut state = 7u64;
        let mut atoms = sites.clone();
        for atom in atoms.chunks_exact_mut(3) {
            for axis in 0..3 {
                atom[axis] = atom[axis] * stretch[axis] + drift[axis] + 1.2 * jitter(&mut state);
            }
        }
        let mut moved = cell;
        for axis in 0..3 {
            moved[axis][axis] *= stretch[axis];
        }
        let occupancy = detector.occupancy(&atoms, moved).unwrap();
        assert!(occupancy.is_perfect(), "{occupancy:?}");
        assert!(detector.counts().iter().all(|count| *count == 1));
    }

    #[test]
    fn a_frenkel_pair_leaves_one_empty_and_one_double_site() {
        let (sites, cell) = simple_cubic(5, 2.0);
        let mut atoms = sites.clone();
        atoms[0] = 5.0;
        atoms[1] = 5.0;
        atoms[2] = 5.4;
        let mut detector = SiteOccupancy::new(&sites, cell, &sites).unwrap();
        let occupancy = detector.occupancy(&atoms, cell).unwrap();
        assert_eq!(
            occupancy,
            Occupancy {
                vacant: 1,
                multiple: 1
            }
        );
        assert_eq!(occupancy.defect_sites(), 2);
        assert_eq!(detector.counts()[0], 0);
    }

    #[test]
    fn a_cell_of_one_bin_still_finds_the_nearest_site() {
        let (sites, cell) = simple_cubic(2, 1.5);
        let mut detector = SiteOccupancy::new(&sites, cell, &sites).unwrap();
        let atoms: Vec<f64> = sites.iter().map(|value| value + 0.3).collect();
        assert!(detector.occupancy(&atoms, cell).unwrap().is_perfect());
    }

    #[test]
    fn a_sheared_cell_is_refused() {
        let (sites, mut cell) = simple_cubic(3, 2.0);
        cell[1][0] = 0.5;
        assert!(SiteOccupancy::new(&sites, cell, &sites).is_err());
    }

    #[test]
    fn persistence_reports_the_start_of_the_first_long_enough_run() {
        let mut watch = Persistence::new(3);
        let pattern = [false, true, true, false, true, true, true, true];
        let mut confirmed = None;
        for (index, holds) in pattern.into_iter().enumerate() {
            if let Some(began) = watch.observe(index, holds) {
                confirmed = Some((index, began));
                break;
            }
        }
        assert_eq!(confirmed, Some((6, 4)));
        let mut single = Persistence::new(0);
        assert_eq!(single.observe(9, true), Some(9));
        assert!(!single.pending());
    }
}
