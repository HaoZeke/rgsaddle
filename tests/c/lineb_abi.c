/* LI-NEB / instanton C ABI contract smoke (minor 46).
 * Built and run by tests/lineb_abi.rs against the cdylib. */
#include "rgsaddle.h"
#include "main_abi_contract.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define N_IMAGES 5
#define N_ATOMS 1
#define DOF (3 * N_ATOMS)

static rgsaddle_status_t surface(void *user, rgsaddle_surface_request_t *req) {
  (void)user;
  if (req->version.major != RGSADDLE_ABI_MAJOR || req->n_atoms != N_ATOMS) {
    return RGSADDLE_SHAPE;
  }
  const int64_t rows =
      (req->flags & RGSADDLE_REQ_ONE_IMAGE) ? 1 : req->n_images;
  for (int64_t r = 0; r < rows; ++r) {
    const double *p = req->positions + r * DOF;
    double *g = req->gradients + r * DOF;
    const double x = p[0], y = p[1], z = p[2];
    const double s = 1.0 / cosh(x);
    req->energies[r] = s * s + y * y + z * z;
    g[0] = -2.0 * s * s * tanh(x);
    g[1] = 2.0 * y;
    g[2] = 2.0 * z;
  }
  return RGSADDLE_OK;
}

static int fail(const char *msg, rgsaddle_status_t st) {
  fprintf(stderr, "%s (%s)\n", msg, rgsaddle_status_name(st));
  return 1;
}

static void seed_path(double *pos) {
  const double turning = acosh(sqrt(1.0 / 0.25));
  const double end = 0.5 * turning;
  memset(pos, 0, sizeof(double) * (size_t)N_IMAGES * DOF);
  for (int i = 0; i < N_IMAGES; ++i) {
    pos[i * DOF] = -end + 2.0 * end * (double)i / (double)(N_IMAGES - 1);
  }
}

int main(void) {
  if ((rgsaddle_abi_version() & 0xffff) != (int)RGSADDLE_ABI_MINOR) {
    return fail("abi minor", RGSADDLE_ABI_MISMATCH);
  }
  if (rgsaddle_lineb_config_default(1e-3, 1e-3, NULL) !=
      RGSADDLE_INVALID_PARAMETER) {
    return fail("null config default", RGSADDLE_INVALID_PARAMETER);
  }

  rgsaddle_lineb_config_t cfg;
  memset(&cfg, 0, sizeof cfg);
  if (rgsaddle_lineb_config_default(1e-3, 1e-3, &cfg) != RGSADDLE_OK) {
    return fail("config default", RGSADDLE_SOLVER);
  }
  if (cfg.opt_tol != 1e-3 || cfg.perp_tol != 1e-3 || cfg.spring != 0.1 ||
      cfg.endpoint_scale != 0.2 || cfg.time_step != 0.1 || cfg.max_move != 0.2 ||
      cfg.turning_width != 1e-3 || cfg.hbar != 1.0 || cfg.boltzmann != 1.0 ||
      cfg.quadrature_pairs != 4) {
    fprintf(stderr, "unexpected defaults\n");
    return 1;
  }

  double pos[N_IMAGES * DOF];
  double out_pos[N_IMAGES * DOF];
  double masses[N_ATOMS] = {1.0};
  rgsaddle_lineb_result_t result;
  memset(&result, 0, sizeof result);
  seed_path(pos);

  if (rgsaddle_lineb_solve(NULL, N_IMAGES, N_ATOMS, pos, masses, 0, 0.25, 1, 0,
                           surface, NULL, out_pos,
                           &result) != RGSADDLE_INVALID_PARAMETER) {
    return fail("null cfg", RGSADDLE_INVALID_PARAMETER);
  }
  if (rgsaddle_lineb_solve(&cfg, N_IMAGES, N_ATOMS, pos, masses, 0, 0.25, 1, 0,
                           NULL, NULL, out_pos, &result) != RGSADDLE_NULL_SURFACE) {
    return fail("null surface", RGSADDLE_NULL_SURFACE);
  }
  if (rgsaddle_lineb_solve(&cfg, N_IMAGES, N_ATOMS, pos, masses, 0, 0.25, 1, 0,
                           surface, NULL, out_pos, NULL) != RGSADDLE_NULL_REPORT) {
    return fail("null result", RGSADDLE_NULL_REPORT);
  }
  if (rgsaddle_lineb_solve(&cfg, 2, N_ATOMS, pos, masses, 0, 0.25, 1, 0, surface,
                           NULL, out_pos, &result) != RGSADDLE_SHAPE) {
    return fail("too few images", RGSADDLE_SHAPE);
  }
  if (rgsaddle_lineb_solve(&cfg, N_IMAGES, N_ATOMS, pos, masses, 2, 0.25, 1, 0,
                           surface, NULL, out_pos,
                           &result) != RGSADDLE_INVALID_PARAMETER) {
    return fail("bad molecular", RGSADDLE_INVALID_PARAMETER);
  }
  if (rgsaddle_lineb_solve(&cfg, N_IMAGES, N_ATOMS, pos, masses, 0, 0.25, 1, 1,
                           surface, NULL, out_pos,
                           &result) != RGSADDLE_INVALID_PARAMETER) {
    return fail("beads=1", RGSADDLE_INVALID_PARAMETER);
  }

  /* One step cannot meet the defaults; the ABI returns STALLED. */
  const rgsaddle_status_t stalled =
      rgsaddle_lineb_solve(&cfg, N_IMAGES, N_ATOMS, pos, masses, 0, 0.25, 1, 0,
                           surface, NULL, out_pos, &result);
  if (stalled != RGSADDLE_STALLED) {
    return fail("unconverged must be STALLED", stalled);
  }
  if (result.iterations < 1 || !isnan(result.log_rate_over_reactant)) {
    fprintf(stderr, "stalled result iterations=%lld log_rate=%g\n",
            (long long)result.iterations, result.log_rate_over_reactant);
    return 1;
  }

  /* Loose tolerances, path only: beads 0 leaves log_rate NaN. */
  if (rgsaddle_lineb_config_default(0.5, 0.5, &cfg) != RGSADDLE_OK) {
    return fail("loose config", RGSADDLE_SOLVER);
  }
  seed_path(pos);
  const rgsaddle_status_t ok =
      rgsaddle_lineb_solve(&cfg, N_IMAGES, N_ATOMS, pos, masses, 0, 0.25, 200, 0,
                           surface, NULL, out_pos, &result);
  if (ok != RGSADDLE_OK) {
    return fail("path-only solve", ok);
  }
  if (!isfinite(result.action) || !isfinite(result.half_period) ||
      !isfinite(result.half_period_dynamics) || !isfinite(result.temperature) ||
      !isfinite(result.temperature_dynamics) ||
      !isnan(result.log_rate_over_reactant) || result.rate_calls != 0 ||
      result.iterations < 1) {
    fprintf(stderr,
            "path result action=%g half=%g half_dyn=%g T=%g Td=%g log=%g "
            "rate_calls=%lld iter=%lld\n",
            result.action, result.half_period, result.half_period_dynamics,
            result.temperature, result.temperature_dynamics,
            result.log_rate_over_reactant, (long long)result.rate_calls,
            (long long)result.iterations);
    return 1;
  }

  printf("RGSADDLE_LINEB_ABI_OK action=%.6g iter=%lld\n", result.action,
         (long long)result.iterations);
  return 0;
}
