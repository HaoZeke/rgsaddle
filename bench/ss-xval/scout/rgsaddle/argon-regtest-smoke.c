/* CP2K solid-state Ar regtest expressed through the rgsaddle C ABI.
 * Units are eV and Angstrom. */
#include "rgsaddle.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define NIMG 3
#define NATOM 2

static const double epsilon = 119.8 * 8.617333262145e-5;
static const double sigma_lj = 3.405;
static const double cutoff = 4.8;
static const double pressure = 50.0 / 1.602176634e6;

static double det3(const double *h) {
  return h[0] * (h[4] * h[8] - h[5] * h[7]) -
         h[1] * (h[3] * h[8] - h[5] * h[6]) +
         h[2] * (h[3] * h[7] - h[4] * h[6]);
}

static int inv3(const double *h, double *out) {
  const double d = det3(h);
  if (!(fabs(d) > 1.0e-14)) {
    return 1;
  }
  out[0] = (h[4] * h[8] - h[5] * h[7]) / d;
  out[1] = (h[2] * h[7] - h[1] * h[8]) / d;
  out[2] = (h[1] * h[5] - h[2] * h[4]) / d;
  out[3] = (h[5] * h[6] - h[3] * h[8]) / d;
  out[4] = (h[0] * h[8] - h[2] * h[6]) / d;
  out[5] = (h[2] * h[3] - h[0] * h[5]) / d;
  out[6] = (h[3] * h[7] - h[4] * h[6]) / d;
  out[7] = (h[1] * h[6] - h[0] * h[7]) / d;
  out[8] = (h[0] * h[4] - h[1] * h[3]) / d;
  return 0;
}

static int lj(const double *x, const double *h, double *energy,
              double *gradient, double *stress) {
  double hinv[9];
  if (inv3(h, hinv) != 0) {
    return 1;
  }
  double d[3] = {x[0] - x[3], x[1] - x[4], x[2] - x[5]};
  double frac[3];
  for (int c = 0; c < 3; ++c) {
    frac[c] = d[0] * hinv[c] + d[1] * hinv[3 + c] +
              d[2] * hinv[6 + c];
    frac[c] -= nearbyint(frac[c]);
  }
  for (int c = 0; c < 3; ++c) {
    d[c] = frac[0] * h[c] + frac[1] * h[3 + c] +
           frac[2] * h[6 + c];
  }
  const double r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
  if (!(r2 < cutoff * cutoff)) {
    *energy = 0.0;
    memset(gradient, 0, 6 * sizeof(*gradient));
    memset(stress, 0, 9 * sizeof(*stress));
    return 0;
  }
  const double a = pow(sigma_lj * sigma_lj / r2, 3);
  const double ac = pow(sigma_lj / cutoff, 6);
  *energy = 4.0 * epsilon * (a * a - a - ac * ac + ac);
  const double scale = 24.0 * epsilon * a * (1.0 - 2.0 * a) / r2;
  for (int c = 0; c < 3; ++c) {
    gradient[c] = scale * d[c];
    gradient[3 + c] = -gradient[c];
  }
  const double volume = fabs(det3(h));
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      stress[3 * i + j] = scale * d[i] * d[j] / volume;
    }
  }
  return 0;
}

static rgsaddle_status_t surface(void *user,
                                 rgsaddle_solid_surface_request_t *req) {
  (void)user;
  const int64_t rows =
      (req->flags & RGSADDLE_REQ_ONE_IMAGE) ? 1 : req->n_images;
  for (int64_t row = 0; row < rows; ++row) {
    if (lj(req->positions + 6 * row, req->cells + 9 * row,
           req->energies + row, req->gradients + 6 * row,
           req->stresses + 9 * row) != 0) {
      return RGSADDLE_SURFACE_FAILED;
    }
  }
  return RGSADDLE_OK;
}

static double linear_enthalpy(double energy, const double *cell) {
  const double trace = (cell[0] + cell[4] + cell[8] - 30.0) / 10.0;
  return energy + pressure * 1000.0 * trace;
}

int main(void) {
  double cells[27] = {
      10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0,
      10.246950765959598, 0.0, 0.0, 0.4, 9.848857801796104, 0.0,
      -0.15, 0.2, 10.099504938362077,
      10.5, 0.0, 0.0, 0.8, 9.7, 0.0, -0.3, 0.4, 10.2,
  };
  double band[18] = {
      2.0, 2.0, 2.0, 5.3, 3.0, 2.0,
      2.050676919316477, 2.000022291778578, 2.000098036812882,
      5.506269079841432, 2.797069031408468, 2.000098036812882,
      2.1, 2.0, 2.0, 5.7, 2.6, 2.0,
  };
  double energy;
  double gradient[6];
  double stress[9];
  if (lj(band, cells, &energy, gradient, stress) != 0) {
    return 2;
  }
  double max_force = 0.0;
  double max_stress = 0.0;
  for (int i = 0; i < 6; ++i) {
    max_force = fmax(max_force, fabs(gradient[i]));
  }
  for (int i = 0; i < 9; ++i) {
    max_stress = fmax(max_stress, fabs(stress[i]));
  }
  printf("initial energy_eV %.15g max_force_eV_per_A %.15g "
         "max_stress_eV_per_A3 %.15g\n",
         energy, max_force, max_stress);
  printf("initial stress_eV_per_A3 %.15g %.15g %.15g %.15g %.15g %.15g\n",
         stress[0], stress[4], stress[8], stress[1], stress[2], stress[5]);

  rgsaddle_band_config_t cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.version.major = RGSADDLE_ABI_MAJOR;
  cfg.version.minor = RGSADDLE_ABI_MINOR;
  cfg.flags = RGSADDLE_BAND_CI_AFTER | RGSADDLE_BAND_NO_MIN_MODE;
  cfg.tangent = RGSADDLE_TANGENT_IMPROVED;
  cfg.spring = RGSADDLE_SPRING_UNIFORM;
  cfg.projection = RGSADDLE_PROJECTION_NEB;
  cfg.method = RGSADDLE_METHOD_FIRE;
  cfg.spring_k = 0.02 * 27.211386245988 /
                 (0.529177210903 * 0.529177210903);
  cfg.ci_trigger_force = INFINITY;
  cfg.force_tol = 1.0e-6;
  cfg.max_move = 0.05;

  RgsaddleBand *session =
      rgsaddle_band_create_solid(&cfg, NIMG, NATOM, band, cells, pressure, 1.0);
  if (session == NULL) {
    return 3;
  }
  rgsaddle_report_t report;
  memset(&report, 0, sizeof(report));
  for (int step = 0;
       step < 10000 && report.status != RGSADDLE_STATUS_CONVERGED; ++step) {
    const rgsaddle_status_t status =
        rgsaddle_band_step_solid(session, surface, NULL, &report);
    if (status != RGSADDLE_OK) {
      fprintf(stderr, "band step: %s\n", rgsaddle_status_name(status));
      rgsaddle_band_free(session);
      return 4;
    }
  }
  rgsaddle_band_positions(session, band);
  rgsaddle_band_cells(session, cells);
  double enthalpy[NIMG];
  for (int image = 0; image < NIMG; ++image) {
    lj(band + 6 * image, cells + 9 * image, &energy, gradient, stress);
    enthalpy[image] = linear_enthalpy(energy, cells + 9 * image);
  }
  double barrier = enthalpy[0];
  for (int image = 1; image < NIMG; ++image) {
    barrier = fmax(barrier, enthalpy[image]);
  }
  printf("band converged %d iterations %lld max_force %.15g "
         "middle_volume_A3 %.15g middle_enthalpy_eV %.15g "
         "barrier_from_first_eV %.15g\n",
         report.status == RGSADDLE_STATUS_CONVERGED,
         (long long)report.iteration, report.max_force, fabs(det3(cells + 9)),
         enthalpy[1], barrier - enthalpy[0]);
  rgsaddle_band_free(session);
  return report.status == RGSADDLE_STATUS_CONVERGED ? 0 : 5;
}
