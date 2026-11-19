/* Shifted Lennard-Jones smoke inputs for the fixed-cell vacancy hop and
 * variable-cell fcc-variant path. Units are eV and Angstrom. */
#include "rgsaddle.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NIMG 7
#define RCUT 1.5
#define SPRING_K 5.0
#define FORCE_TOL 1.0e-3

typedef struct {
  double cell[9];
} FixedSurface;

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

static int lj(int64_t n, const double *x, const double *h, double *energy,
              double *gradient, double *stress) {
  double hinv[9];
  if (inv3(h, hinv) != 0) {
    return 1;
  }
  const double volume = fabs(det3(h));
  const double rc2 = RCUT * RCUT;
  const double qc = pow(1.0 / RCUT, 6);
  const double cutoff_energy = 4.0 * (qc * qc - qc);
  *energy = 0.0;
  memset(gradient, 0, (size_t)(3 * n) * sizeof(*gradient));
  memset(stress, 0, 9 * sizeof(*stress));

  for (int64_t i = 0; i < n; ++i) {
    for (int64_t j = 0; j < i; ++j) {
      double d[3] = {x[3 * i] - x[3 * j], x[3 * i + 1] - x[3 * j + 1],
                     x[3 * i + 2] - x[3 * j + 2]};
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
      if (!(r2 < rc2)) {
        continue;
      }
      const double q = 1.0 / (r2 * r2 * r2);
      *energy += 4.0 * (q * q - q) - cutoff_energy;
      const double scale = 24.0 * q * (1.0 - 2.0 * q) / r2;
      for (int a = 0; a < 3; ++a) {
        const double component = scale * d[a];
        gradient[3 * i + a] += component;
        gradient[3 * j + a] -= component;
        for (int b = 0; b < 3; ++b) {
          stress[3 * a + b] += scale * d[a] * d[b] / volume;
        }
      }
    }
  }
  return 0;
}

static rgsaddle_status_t fixed_surface(void *user,
                                       rgsaddle_surface_request_t *req) {
  FixedSurface *surface = (FixedSurface *)user;
  const int64_t rows =
      (req->flags & RGSADDLE_REQ_ONE_IMAGE) ? 1 : req->n_images;
  const int64_t dof = 3 * req->n_atoms;
  double stress[9];
  for (int64_t row = 0; row < rows; ++row) {
    if (lj(req->n_atoms, req->positions + row * dof, surface->cell,
           req->energies + row, req->gradients + row * dof, stress) != 0) {
      return RGSADDLE_SURFACE_FAILED;
    }
  }
  return RGSADDLE_OK;
}

static rgsaddle_status_t solid_surface(
    void *user, rgsaddle_solid_surface_request_t *req) {
  (void)user;
  const int64_t rows =
      (req->flags & RGSADDLE_REQ_ONE_IMAGE) ? 1 : req->n_images;
  const int64_t dof = 3 * req->n_atoms;
  for (int64_t row = 0; row < rows; ++row) {
    if (lj(req->n_atoms, req->positions + row * dof, req->cells + 9 * row,
           req->energies + row, req->gradients + row * dof,
           req->stresses + 9 * row) != 0) {
      return RGSADDLE_SURFACE_FAILED;
    }
  }
  return RGSADDLE_OK;
}

static rgsaddle_band_config_t config(const double *cell) {
  rgsaddle_band_config_t cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.version.major = RGSADDLE_ABI_MAJOR;
  cfg.version.minor = RGSADDLE_ABI_MINOR;
  cfg.flags = RGSADDLE_BAND_CI_AFTER | RGSADDLE_BAND_NO_MIN_MODE;
  cfg.tangent = RGSADDLE_TANGENT_IMPROVED;
  cfg.spring = RGSADDLE_SPRING_UNIFORM;
  cfg.projection = RGSADDLE_PROJECTION_NEB;
  cfg.method = RGSADDLE_METHOD_FIRE;
  cfg.spring_k = SPRING_K;
  cfg.ci_trigger_force = INFINITY;
  cfg.cell = cell;
  cfg.force_tol = FORCE_TOL;
  cfg.max_move = 0.05;
  return cfg;
}

static double max_abs(const double *values, int64_t n) {
  double out = 0.0;
  for (int64_t i = 0; i < n; ++i) {
    out = fmax(out, fabs(values[i]));
  }
  return out;
}

static void print_initial(const char *name, int64_t n, const double *x,
                          const double *cell) {
  double *gradient = calloc((size_t)(3 * n), sizeof(*gradient));
  double energy;
  double stress[9];
  if (gradient == NULL || lj(n, x, cell, &energy, gradient, stress) != 0) {
    fprintf(stderr, "%s initial evaluation failed\n", name);
    exit(2);
  }
  printf("%s initial_energy_eV %.15g max_force_eV_per_A %.15g "
         "stress_eV_per_A3 %.15g %.15g %.15g %.15g %.15g %.15g\n",
         name, energy, max_abs(gradient, 3 * n), stress[0], stress[4],
         stress[8], stress[1], stress[2], stress[5]);
  free(gradient);
}

static int run_vacancy(void) {
  const double a = pow(2.0, 2.0 / 3.0);
  FixedSurface surface = {
      .cell = {2.0 * a, 0.0, 0.0, 0.0, 2.0 * a, 0.0, 0.0, 0.0,
               2.0 * a}};
  const double basis[4][3] = {{0.0, 0.0, 0.0},
                              {0.0, 0.5, 0.5},
                              {0.5, 0.0, 0.5},
                              {0.5, 0.5, 0.0}};
  const int64_t n = 31;
  const int64_t dof = 3 * n;
  double *reactant = calloc((size_t)dof, sizeof(*reactant));
  double *band = calloc((size_t)(NIMG * dof), sizeof(*band));
  if (reactant == NULL || band == NULL) {
    return 2;
  }
  int64_t atom = 0;
  int64_t moving = -1;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {
        for (int b = 0; b < 4; ++b) {
          if (i == 0 && j == 0 && k == 0 && b == 0) {
            continue;
          }
          for (int c = 0; c < 3; ++c) {
            const int ijk[3] = {i, j, k};
            reactant[3 * atom + c] = (ijk[c] + basis[b][c]) * a;
          }
          if (i == 0 && j == 0 && k == 0 && b == 1) {
            moving = atom;
          }
          ++atom;
        }
      }
    }
  }
  for (int image = 0; image < NIMG; ++image) {
    const double t = (double)image / (NIMG - 1);
    memcpy(band + image * dof, reactant, (size_t)dof * sizeof(*band));
    for (int c = 0; c < 3; ++c) {
      band[image * dof + 3 * moving + c] =
          (1.0 - t) * reactant[3 * moving + c];
    }
  }
  print_initial("vacancy", n, reactant, surface.cell);
  rgsaddle_band_config_t cfg = config(surface.cell);
  RgsaddleBand *session =
      rgsaddle_band_create(&cfg, NIMG, n, band);
  if (session == NULL) {
    return 3;
  }
  rgsaddle_report_t report;
  memset(&report, 0, sizeof(report));
  int iteration = 0;
  for (; iteration < 3000 && report.status != RGSADDLE_STATUS_CONVERGED;
       ++iteration) {
    const rgsaddle_status_t status =
        rgsaddle_band_step(session, fixed_surface, &surface, &report);
    if (status != RGSADDLE_OK) {
      fprintf(stderr, "vacancy step: %s\n", rgsaddle_status_name(status));
      return 4;
    }
  }
  double *out = calloc((size_t)(NIMG * dof), sizeof(*out));
  double *gradient = calloc((size_t)dof, sizeof(*gradient));
  double stress[9];
  double e0;
  double barrier = -INFINITY;
  rgsaddle_band_positions(session, out);
  lj(n, out, surface.cell, &e0, gradient, stress);
  for (int image = 0; image < NIMG; ++image) {
    double energy;
    lj(n, out + image * dof, surface.cell, &energy, gradient, stress);
    barrier = fmax(barrier, energy - e0);
  }
  printf("vacancy converged %d iterations %lld max_force %.15g "
         "barrier_eV %.15g\n",
         report.status == RGSADDLE_STATUS_CONVERGED,
         (long long)report.iteration, report.max_force, barrier);
  rgsaddle_band_free(session);
  free(out);
  free(gradient);
  free(reactant);
  free(band);
  return report.status == RGSADDLE_STATUS_CONVERGED ? 0 : 5;
}

static int run_fcc_variants(void) {
  const double short_axis = pow(2.0, 1.0 / 6.0);
  const double long_axis = sqrt(2.0) * short_axis;
  const int64_t n = 54;
  const int64_t dof = 3 * n;
  double *cells = calloc((size_t)(9 * NIMG), sizeof(*cells));
  double *band = calloc((size_t)(NIMG * dof), sizeof(*band));
  if (cells == NULL || band == NULL) {
    return 2;
  }
  for (int image = 0; image < NIMG; ++image) {
    const double t = (double)image / (NIMG - 1);
    double *cell = cells + 9 * image;
    cell[0] = 3.0 * ((1.0 - t) * short_axis + t * long_axis);
    cell[4] = 3.0 * short_axis;
    cell[8] = 3.0 * ((1.0 - t) * long_axis + t * short_axis);
    int64_t atom = 0;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          for (int b = 0; b < 2; ++b) {
            const double frac[3] = {(i + 0.5 * b) / 3.0,
                                    (j + 0.5 * b) / 3.0,
                                    (k + 0.5 * b) / 3.0};
            band[image * dof + 3 * atom] = frac[0] * cell[0];
            band[image * dof + 3 * atom + 1] = frac[1] * cell[4];
            band[image * dof + 3 * atom + 2] = frac[2] * cell[8];
            ++atom;
          }
        }
      }
    }
  }
  print_initial("fcc_variants", n, band, cells);
  rgsaddle_band_config_t cfg = config(NULL);
  RgsaddleBand *session =
      rgsaddle_band_create_solid(&cfg, NIMG, n, band, cells, 0.0, 1.0);
  if (session == NULL) {
    return 3;
  }
  rgsaddle_report_t report;
  memset(&report, 0, sizeof(report));
  int iteration = 0;
  for (; iteration < 5000 && report.status != RGSADDLE_STATUS_CONVERGED;
       ++iteration) {
    const rgsaddle_status_t status =
        rgsaddle_band_step_solid(session, solid_surface, NULL, &report);
    if (status != RGSADDLE_OK) {
      fprintf(stderr, "fcc_variants step: %s\n",
              rgsaddle_status_name(status));
      return 4;
    }
  }
  double *out = calloc((size_t)(NIMG * dof), sizeof(*out));
  double *out_cells = calloc((size_t)(9 * NIMG), sizeof(*out_cells));
  double *gradient = calloc((size_t)dof, sizeof(*gradient));
  double stress[9];
  double e0;
  double barrier = -INFINITY;
  rgsaddle_band_positions(session, out);
  rgsaddle_band_cells(session, out_cells);
  lj(n, out, out_cells, &e0, gradient, stress);
  for (int image = 0; image < NIMG; ++image) {
    double energy;
    lj(n, out + image * dof, out_cells + 9 * image, &energy, gradient,
       stress);
    barrier = fmax(barrier, energy - e0);
  }
  printf("fcc_variants converged %d iterations %lld max_force %.15g "
         "barrier_eV %.15g\n",
         report.status == RGSADDLE_STATUS_CONVERGED,
         (long long)report.iteration, report.max_force, barrier);
  rgsaddle_band_free(session);
  free(out);
  free(out_cells);
  free(gradient);
  free(cells);
  free(band);
  return report.status == RGSADDLE_STATUS_CONVERGED ? 0 : 5;
}

int main(void) {
  const int vacancy = run_vacancy();
  const int fcc_variants = run_fcc_variants();
  return vacancy != 0 ? vacancy : fcc_variants;
}
