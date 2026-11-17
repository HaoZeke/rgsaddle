//! Saddle index of the converged climbing image, and the escape from a
//! saddle of higher index; see [`mod@crate::saddle_index`].

use ndarray::Array1;

use super::min_mode::{ImageSurface, Point, rigid_motions};
use super::{BandReport, BandSession, BandStatus, BandSurface, SaddleIndexCheck};
use crate::error::SaddleError;
use crate::minmode::PointSurface;
use crate::saddle_index::{Certificate, Probe, SaddleIndex, certify, escape_push};

/// The last certificate and the escapes taken.
#[derive(Default)]
pub(super) struct SaddleState {
    /// The climbing image, its position, and the certificate there.
    certified: Option<(usize, Array1<f64>, Certificate)>,
    /// Pushes since the host last moved, restarted or reset the band.
    attempts: usize,
    /// Pushes over the session.
    escapes: usize,
    /// Surface rows the certificates and pushes spent over the session.
    evaluations: usize,
}

impl SaddleState {
    /// The band moved by the host's hand or its surface changed.
    pub(super) fn forget(&mut self) {
        self.certified = None;
        self.attempts = 0;
    }
}

impl BandSession {
    /// The saddle index certified at the climbing image's current
    /// position, if any ([`super::BandConfig::saddle_check`]).
    #[must_use]
    pub fn saddle_index(&self) -> Option<&SaddleIndex> {
        let (image, x, certificate) = self.saddle.certified.as_ref()?;
        (self.state.ci() == Some(*image) && self.positions.row(*image) == x)
            .then_some(&certificate.index)
    }

    /// Pushes off a saddle of higher index over the session.
    #[must_use]
    pub fn saddle_escapes(&self) -> usize {
        self.saddle.escapes
    }

    /// Surface rows the saddle-index certificates and the escape pushes
    /// spent over the session. Each also counts in the
    /// [`BandReport::surface_rows`] of the step that spent it.
    #[must_use]
    pub fn saddle_evaluations(&self) -> usize {
        self.saddle.evaluations
    }

    /// Replace [`super::BandConfig::saddle_check`] between steps; `None`
    /// turns the check off. The last certificate and the escape count
    /// are dropped. On an error the session is unchanged.
    pub fn set_saddle_check(&mut self, check: Option<SaddleIndexCheck>) -> Result<(), SaddleError> {
        if let Some(check) = &check {
            check.validate()?;
        }
        self.config.saddle_check = check;
        self.saddle.forget();
        Ok(())
    }

    /// Certify the climbing image of a converged step, and push off a
    /// saddle of higher index when the check carries an escape.
    pub(super) fn saddle_check<S: BandSurface>(
        &mut self,
        surface: &S,
        report: &mut BandReport,
    ) -> Result<(), SaddleError> {
        let Some(check) = self.config.saddle_check else {
            return Ok(());
        };
        if self.solid.is_some() || self.config.climbing.is_none() {
            return Ok(());
        }
        let Some(image) = report.ci_index else {
            return Ok(());
        };
        let x = self.positions.row(image).to_owned();
        let (energies, gradient) = {
            let interior = self.interior_flat();
            let slot = self.state.last.lock().unwrap_or_else(|e| e.into_inner());
            let Some(last) = slot.as_ref().filter(|l| l.x == interior) else {
                return Ok(());
            };
            (last.energies.clone(), last.gradients.row(image).to_owned())
        };
        let tangent = self.image_tangent(image, &energies)?;
        let fixed = self.state.fixed.clone();
        let image_surface = ImageSurface {
            surface,
            image,
            rows: &self.state.rows,
            fixed: &fixed,
        };
        let mut spent = 0;
        let cached = self
            .saddle
            .certified
            .as_ref()
            .filter(|(at, y, _)| *at == image && *y == x)
            .map(|(_, _, certificate)| certificate.clone());
        let certificate = match cached {
            Some(certificate) => certificate,
            None => {
                let held = if fixed.iter().any(|&f| f) {
                    (0..x.len()).map(|k| fixed[k / 3]).collect()
                } else {
                    Vec::new()
                };
                let excluded = rigid_motions(
                    &x,
                    self.config.cell.as_ref(),
                    self.config.drops_translation(),
                    &[],
                )
                .map(|q| q.columns().into_iter().map(|c| c.to_owned()).collect())
                .unwrap_or_default();
                let hint = self.refine_mode(image).unwrap_or_else(|| tangent.clone());
                let certificate = certify(
                    &check,
                    &image_surface,
                    Probe {
                        x: x.view(),
                        gradient: gradient.view(),
                        held,
                        excluded,
                        hints: vec![hint],
                    },
                )?;
                spent += certificate.index.evaluations;
                self.saddle.certified = Some((image, x.clone(), certificate.clone()));
                certificate
            }
        };
        report.saddle_index = Some(certificate.index.clone());
        if let Some(escape) = check.escape
            && certificate.index.index >= 2
            && self.saddle.attempts < escape.max_attempts
        {
            let push = escape_push(
                &image_surface,
                x.view(),
                &certificate,
                tangent.view(),
                escape.push_at(self.saddle.attempts),
            )?;
            self.saddle.attempts += 1;
            if let Some(push) = push {
                let (energy, pushed) = image_surface.eval(push.start.view())?;
                spent += push.evaluations + 1;
                let start = Point {
                    x: push.start,
                    energy,
                    gradient: pushed,
                };
                let saved = Point {
                    x,
                    energy: energies[image],
                    gradient,
                };
                self.refine_escape(image, start, saved, push.reaction, tangent)?;
                self.saddle.escapes += 1;
                report.status = BandStatus::Running;
            }
        }
        report.surface_rows += spent;
        self.saddle.evaluations += spent;
        Ok(())
    }
}
