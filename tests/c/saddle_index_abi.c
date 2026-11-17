/* Saddle-index certification and escape through the C ABI (minor 40)
 * on V = (x^4 - 2 x^2) / 4 + (y^4 - y^2) / 4 + z^2 / 2, whose origin is
 * an index-2 saddle with curvatures -1 and -1/2 and whose points
 * (0, +-sqrt(1/2), 0) are index-1 saddles. Built and run by
 * tests/saddle_index_abi.rs against the cdylib. */
#include "rgsaddle.h"
#include "main_abi_contract.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define N_IMAGES 7
#define DOF 3

static void energy(const double *p, double *e, double *g) {
  const double a = p[0], b = p[1], c = p[2];
  *e = (a * a * a * a - 2.0 * a * a) / 4.0 + (b * b * b * b - b * b) / 4.0 + c * c / 2.0;
  g[0] = a * a * a - a;
  g[1] = b * b * b - 0.5 * b;
  g[2] = c;
}

static rgsaddle_status_t surface(void *user, rgsaddle_surface_request_t *req) {
  (void)user;
  const int64_t rows = (req->flags & RGSADDLE_REQ_ONE_IMAGE) ? 1 : req->n_images;
  for (int64_t r = 0; r < rows; ++r) {
    energy(req->positions + r * DOF, req->energies + r, req->gradients + r * DOF);
  }
  return RGSADDLE_OK;
}

static int fail(const char *msg, rgsaddle_status_t st) {
  const char *detail = rgsaddle_last_error();
  fprintf(stderr, "%s (%s): %s\n", msg, rgsaddle_status_name(st), detail ? detail : "");
  return 1;
}

static rgsaddle_minmode_config_t minmode_config(uint32_t minor) {
  rgsaddle_minmode_config_t cfg;
  memset(&cfg, 0, sizeof cfg);
  cfg.version.major = RGSADDLE_ABI_MAJOR;
  cfg.version.minor = minor;
  cfg.kind = RGSADDLE_MINMODE_DIMER;
  cfg.method = RGSADDLE_METHOD_FIRE;
  cfg.dr = 1e-3;
  cfg.rotation_tol = 1e-6;
  cfg.max_rotations = 20;
  cfg.krylov_dim = 3;
  cfg.force_tol = 1e-6;
  cfg.max_move = 0.1;
  cfg.difference = RGSADDLE_DIFFERENCE_FORWARD;
  return cfg;
}

/* Climb from (0.2, 0, 0) along x; the search never leaves the x axis. */
static int run_minmode(RgsaddleMinMode *session, int *pushes, double *x) {
  rgsaddle_report_t rep;
  memset(&rep, 0, sizeof rep);
  rgsaddle_saddle_index_t index;
  if (rgsaddle_minmode_saddle_index(session, &index, NULL, 0) != RGSADDLE_OK) {
    return 1;
  }
  int64_t escapes = index.escapes;
  *pushes = 0;
  int steps = 0;
  do {
    const rgsaddle_status_t st = rgsaddle_minmode_step(session, surface, NULL, &rep);
    if (st != RGSADDLE_OK) {
      return fail("minmode step", st);
    }
    if (rgsaddle_minmode_saddle_index(session, &index, NULL, 0) != RGSADDLE_OK) {
      return 1;
    }
    if (index.escapes != escapes) {
      if (rep.status != RGSADDLE_STATUS_RUNNING || index.certified) {
        fprintf(stderr, "a push must report running and leave the certified point\n");
        return 1;
      }
      escapes = index.escapes;
      ++*pushes;
    }
  } while (rep.status == RGSADDLE_STATUS_RUNNING && ++steps < 5000);
  if (rep.status != RGSADDLE_STATUS_CONVERGED) {
    fprintf(stderr, "minmode did not converge: %g\n", rep.max_force);
    return 1;
  }
  return rgsaddle_minmode_position(session, x) == RGSADDLE_OK ? 0 : 1;
}

static int check_minmode(void) {
  const double start[DOF] = {0.2, 0.0, 0.0};
  const double mode[DOF] = {1.0, 0.0, 0.0};
  double x[DOF];
  int pushes = 0;

  /* A config of minor 40 certifies by default: the top is index 2. */
  rgsaddle_minmode_config_t cfg = minmode_config(RGSADDLE_ABI_MINOR);
  RgsaddleMinMode *session = rgsaddle_minmode_create(&cfg, 1, start, mode);
  if (!session || run_minmode(session, &pushes, x) != 0) {
    return fail("minmode at minor 40", RGSADDLE_SOLVER);
  }
  double curvatures[8];
  rgsaddle_saddle_index_t index;
  rgsaddle_status_t st = rgsaddle_minmode_saddle_index(session, &index, curvatures, 8);
  if (st != RGSADDLE_OK) {
    return fail("minmode_saddle_index", st);
  }
  if (index.version.minor != RGSADDLE_ABI_MINOR || !index.certified || index.index != 2 ||
      !index.converged || index.n_curvatures < 3 || index.evaluations <= 0 ||
      index.total_evaluations != index.evaluations || index.escapes != 0 || pushes != 0 ||
      fabs(curvatures[0] + 1.0) > 1e-2 || fabs(curvatures[1] + 0.5) > 1e-2 ||
      fabs(x[0]) > 1e-5 || x[1] != 0.0) {
    fprintf(stderr, "top: certified %d index %d curvatures %g %g at %g %g\n", index.certified,
            index.index, curvatures[0], curvatures[1], x[0], x[1]);
    return 1;
  }

  /* The escape pushes once along y and climbs to the index-1 saddle. */
  rgsaddle_saddle_check_t check;
  if (rgsaddle_saddle_check_default(&check) != RGSADDLE_OK) {
    return 1;
  }
  check.escape_attempts = 2;
  if (rgsaddle_minmode_set_position(session, start, NULL) != RGSADDLE_OK ||
      rgsaddle_minmode_set_mode(session, mode) != RGSADDLE_OK) {
    return 1;
  }
  st = rgsaddle_minmode_set_saddle_check(session, &check);
  if (st != RGSADDLE_OK) {
    return fail("minmode_set_saddle_check", st);
  }
  if (run_minmode(session, &pushes, x) != 0) {
    return 1;
  }
  st = rgsaddle_minmode_saddle_index(session, &index, curvatures, 1);
  if (st != RGSADDLE_OK || !index.certified || index.index != 1 || index.escapes != 1 ||
      pushes != 1 || index.total_evaluations <= index.evaluations ||
      fabs(fabs(x[1]) - sqrt(0.5)) > 1e-4 || fabs(x[0]) > 1e-4 || curvatures[0] >= 0.0) {
    fprintf(stderr, "escape: index %d escapes %lld pushes %d at %g %g\n", index.index,
            (long long)index.escapes, pushes, x[0], x[1]);
    return 1;
  }
  st = rgsaddle_minmode_set_saddle_check(session, NULL);
  if (st != RGSADDLE_OK || rgsaddle_minmode_saddle_index(session, &index, NULL, 0) != RGSADDLE_OK ||
      index.certified || index.escapes != 1) {
    return fail("a cleared check keeps the counts and drops the certificate", st);
  }
  rgsaddle_minmode_free(session);

  /* A config of minor 39 keeps the old behaviour. */
  cfg = minmode_config(39);
  session = rgsaddle_minmode_create(&cfg, 1, start, mode);
  if (!session || run_minmode(session, &pushes, x) != 0) {
    return fail("minmode at minor 39", RGSADDLE_SOLVER);
  }
  if (rgsaddle_minmode_saddle_index(session, &index, NULL, 0) != RGSADDLE_OK ||
      index.certified || index.total_evaluations != 0) {
    fprintf(stderr, "a minor-39 config certified\n");
    return 1;
  }
  rgsaddle_minmode_free(session);
  return 0;
}

/* Images 4 to 6 are the inversion of images 2 to 0, so the middle one
 * stays on the top. */
static void symmetric_band(double *pos) {
  const double side = sqrt(0.5), pi = 3.14159265358979323846;
  for (int i = 0; i < 3; ++i) {
    const double t = (double)i / 6.0;
    pos[i * DOF] = -1.0 + 2.0 * t;
    pos[i * DOF + 1] = -side + 2.0 * side * t + 0.3 * sin(2.0 * pi * t);
    pos[i * DOF + 2] = 0.0;
  }
  memset(pos + 3 * DOF, 0, DOF * sizeof(double));
  for (int i = 4; i < N_IMAGES; ++i) {
    for (int k = 0; k < DOF; ++k) {
      pos[i * DOF + k] = -pos[(6 - i) * DOF + k];
    }
  }
}

static rgsaddle_band_config_t band_config(void) {
  rgsaddle_band_config_t cfg;
  memset(&cfg, 0, sizeof cfg);
  cfg.version.major = RGSADDLE_ABI_MAJOR;
  cfg.version.minor = RGSADDLE_ABI_MINOR;
  cfg.tangent = RGSADDLE_TANGENT_IMPROVED;
  cfg.spring = RGSADDLE_SPRING_UNIFORM;
  cfg.projection = RGSADDLE_PROJECTION_NEB;
  cfg.method = RGSADDLE_METHOD_FIRE;
  cfg.spring_k = 5.0;
  cfg.ci_trigger_factor = 0.5;
  cfg.force_tol = 1e-5;
  cfg.max_move = 0.1;
  return cfg;
}

static int check_band(void) {
  double pos[N_IMAGES * DOF];
  symmetric_band(pos);
  const rgsaddle_band_config_t cfg = band_config();
  RgsaddleBand *band = rgsaddle_band_create(&cfg, N_IMAGES, 1, pos);
  if (!band) {
    return fail("band_create", RGSADDLE_SHAPE);
  }
  rgsaddle_report_t rep;
  memset(&rep, 0, sizeof rep);
  int steps = 0;
  do {
    const rgsaddle_status_t st = rgsaddle_band_step(band, surface, NULL, &rep);
    if (st != RGSADDLE_OK) {
      return fail("band step", st);
    }
  } while (rep.status == RGSADDLE_STATUS_RUNNING && ++steps < 5000);
  double curvatures[2];
  rgsaddle_saddle_index_t index;
  if (rep.status != RGSADDLE_STATUS_CONVERGED || rep.ci_index != 3 ||
      rgsaddle_band_saddle_index(band, &index, curvatures, 2) != RGSADDLE_OK ||
      !index.certified || index.index != 2 || fabs(curvatures[0] + 1.0) > 1e-2 ||
      fabs(curvatures[1] + 0.5) > 1e-2) {
    fprintf(stderr, "band: ci %lld certified %d index %d\n", (long long)rep.ci_index,
            index.certified, index.index);
    return 1;
  }
  if (rgsaddle_band_saddle_index(band, &index, NULL, 1) != RGSADDLE_SHAPE ||
      rgsaddle_band_saddle_index(band, &index, curvatures, -1) != RGSADDLE_SHAPE ||
      rgsaddle_band_saddle_index(band, NULL, NULL, 0) != RGSADDLE_NULL_REPORT ||
      rgsaddle_band_saddle_index(NULL, &index, NULL, 0) != RGSADDLE_NULL_BAND) {
    return fail("band_saddle_index argument checks", RGSADDLE_SHAPE);
  }
  rgsaddle_saddle_check_t check;
  rgsaddle_saddle_check_default(&check);
  check.block = 0;
  if (rgsaddle_band_set_saddle_check(band, &check) != RGSADDLE_INVALID_PARAMETER ||
      rgsaddle_band_saddle_index(band, &index, NULL, 0) != RGSADDLE_OK || !index.certified) {
    return fail("a refused check must leave the band", RGSADDLE_INVALID_PARAMETER);
  }
  rgsaddle_saddle_check_default(&check);
  check.version.major += 1;
  if (rgsaddle_band_set_saddle_check(band, &check) != RGSADDLE_ABI_MISMATCH ||
      rgsaddle_band_set_saddle_check(NULL, NULL) != RGSADDLE_NULL_BAND) {
    return fail("band_set_saddle_check argument checks", RGSADDLE_ABI_MISMATCH);
  }
  rgsaddle_band_free(band);

  rgsaddle_ocineb_config_t oc;
  memset(&oc, 0, sizeof oc);
  oc.version.major = RGSADDLE_ABI_MAJOR;
  oc.version.minor = RGSADDLE_ABI_MINOR;
  oc.trigger_factor = 0.31;
  oc.angle_tol = 0.85;
  oc.stability_count = 5;
  oc.max_mmf_steps = 1000;
  const rgsaddle_minmode_config_t mm = minmode_config(RGSADDLE_ABI_MINOR);
  RgsaddleOcineb *ocineb = rgsaddle_ocineb_create(&cfg, &mm, &oc, N_IMAGES, 1, pos);
  if (!ocineb) {
    return fail("ocineb_create", RGSADDLE_SHAPE);
  }
  rgsaddle_ocineb_report_t orep;
  memset(&orep, 0, sizeof orep);
  steps = 0;
  do {
    const rgsaddle_status_t st = rgsaddle_ocineb_step(ocineb, surface, NULL, &orep);
    if (st != RGSADDLE_OK) {
      return fail("ocineb step", st);
    }
  } while (orep.status == RGSADDLE_STATUS_RUNNING && ++steps < 5000);
  if (orep.status != RGSADDLE_STATUS_CONVERGED ||
      rgsaddle_ocineb_saddle_index(ocineb, &index, NULL, 0) != RGSADDLE_OK ||
      !index.certified || index.index != 2) {
    fprintf(stderr, "ocineb: certified %d index %d\n", index.certified, index.index);
    return 1;
  }
  if (rgsaddle_ocineb_set_saddle_check(ocineb, NULL) != RGSADDLE_OK ||
      rgsaddle_ocineb_saddle_index(ocineb, &index, NULL, 0) != RGSADDLE_OK || index.certified ||
      rgsaddle_ocineb_saddle_index(NULL, &index, NULL, 0) != RGSADDLE_NULL_SESSION ||
      rgsaddle_ocineb_set_saddle_check(NULL, NULL) != RGSADDLE_NULL_SESSION) {
    return fail("ocineb saddle check", RGSADDLE_NULL_SESSION);
  }
  rgsaddle_ocineb_free(ocineb);
  return 0;
}

static int check_point(void) {
  rgsaddle_saddle_check_t check;
  if (rgsaddle_saddle_check_default(NULL) != RGSADDLE_NULL_REPORT ||
      rgsaddle_saddle_check_default(&check) != RGSADDLE_OK) {
    return 1;
  }
  if (check.version.major != RGSADDLE_ABI_MAJOR || check.version.minor != RGSADDLE_ABI_MINOR ||
      check.flags != 0 || check.block != 2 || check.lowest != 4 || check.max_actions != 100 ||
      check.difference != RGSADDLE_DIFFERENCE_FORWARD || check.dr != 1e-3 ||
      check.residual_tol != 1e-2 || check.negative_tol != 1e-3 || check.escape_attempts != 0 ||
      check.escape_push != 0.1) {
    fprintf(stderr, "unexpected defaults\n");
    return 1;
  }
  const double top[DOF] = {0.0, 0.0, 0.0};
  const double side[DOF] = {0.0, sqrt(0.5), 0.0};
  double curvatures[4];
  rgsaddle_saddle_index_t index;
  rgsaddle_status_t st =
      rgsaddle_saddle_index(&check, 1, top, NULL, NULL, NULL, surface, NULL, &index, curvatures, 4);
  if (st != RGSADDLE_OK || !index.certified || index.index != 2 || !index.converged ||
      index.n_curvatures != 3 || index.evaluations <= 0 ||
      index.total_evaluations != index.evaluations || fabs(curvatures[0] + 1.0) > 1e-2 ||
      fabs(curvatures[1] + 0.5) > 1e-2 || fabs(curvatures[2] - 1.0) > 1e-2) {
    fprintf(stderr, "top: index %d curvatures %g %g %g\n", index.index, curvatures[0],
            curvatures[1], curvatures[2]);
    return fail("saddle_index at the top", st);
  }
  st = rgsaddle_saddle_index(NULL, 1, side, NULL, NULL, NULL, surface, NULL, &index, NULL, 0);
  if (st != RGSADDLE_OK || index.index != 1) {
    return fail("saddle_index beside the top", st);
  }
  const uint8_t fixed[1] = {1};
  st = rgsaddle_saddle_index(&check, 1, top, NULL, NULL, fixed, surface, NULL, &index, NULL, 0);
  if (st != RGSADDLE_OK || index.index != 0 || index.n_curvatures != 0) {
    return fail("a fixed atom leaves nothing to certify", st);
  }
  const uint8_t two[1] = {2};
  if (rgsaddle_saddle_index(&check, 1, top, NULL, NULL, two, surface, NULL, &index, NULL, 0) !=
          RGSADDLE_INVALID_PARAMETER ||
      rgsaddle_saddle_index(&check, 1, top, NULL, NULL, NULL, NULL, NULL, &index, NULL, 0) !=
          RGSADDLE_NULL_SURFACE ||
      rgsaddle_saddle_index(&check, 1, top, NULL, NULL, NULL, surface, NULL, NULL, NULL, 0) !=
          RGSADDLE_NULL_REPORT ||
      rgsaddle_saddle_index(&check, 1, NULL, NULL, NULL, NULL, surface, NULL, &index, NULL, 0) !=
          RGSADDLE_SHAPE ||
      rgsaddle_saddle_index(&check, 0, top, NULL, NULL, NULL, surface, NULL, &index, NULL, 0) !=
          RGSADDLE_SHAPE ||
      rgsaddle_saddle_index(&check, 1, top, NULL, NULL, NULL, surface, NULL, &index, NULL, -1) !=
          RGSADDLE_SHAPE) {
    return fail("saddle_index argument checks", RGSADDLE_INVALID_PARAMETER);
  }
  check.max_actions = 1;
  if (rgsaddle_saddle_index(&check, 1, top, NULL, NULL, NULL, surface, NULL, &index, NULL, 0) !=
      RGSADDLE_INVALID_PARAMETER) {
    return fail("max_actions under block must be refused", RGSADDLE_INVALID_PARAMETER);
  }
  rgsaddle_saddle_check_default(&check);
  check.escape_attempts = 1;
  check.escape_push = 0.0;
  if (rgsaddle_saddle_index(&check, 1, top, NULL, NULL, NULL, surface, NULL, &index, NULL, 0) !=
      RGSADDLE_INVALID_PARAMETER) {
    return fail("an escape without a push must be refused", RGSADDLE_INVALID_PARAMETER);
  }
  return 0;
}

int main(void) {
  if (check_point() != 0 || check_minmode() != 0 || check_band() != 0) {
    return 1;
  }
  printf("RGSADDLE_SADDLE_INDEX_ABI_OK\n");
  return 0;
}
