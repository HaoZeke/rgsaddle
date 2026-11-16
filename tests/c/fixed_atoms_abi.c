/* rgsaddle_band_set_fixed_atoms and rgsaddle_band_project_fixed (ABI
 * minor 39) over three atoms: atoms 0 and 1 are tethered to points they
 * do not sit on, so they carry a force, and atom 2 crosses a double
 * well. Built and run by tests/fixed_atoms_abi.rs against the cdylib. */
#include "rgsaddle.h"
#include "main_abi_contract.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define N_IMAGES 7
#define N_ATOMS 3
#define DOF (3 * N_ATOMS)

static const double tether[2][3] = {{0.3, 0.2, 0.1}, {2.6, -0.2, 0.0}};

static void energy(const double *p, double *e, double *g) {
  *e = 0.0;
  for (int a = 0; a < 2; ++a) {
    for (int k = 0; k < 3; ++k) {
      const double d = p[3 * a + k] - tether[a][k];
      *e += 0.5 * d * d;
      g[3 * a + k] = d;
    }
  }
  const double x = p[6] - 1.0, y = p[7] - 1.5, z = p[8];
  *e += (x * x - 1.0) * (x * x - 1.0) + 2.0 * y * y + 2.0 * z * z;
  g[6] = 4.0 * x * (x * x - 1.0);
  g[7] = 4.0 * y;
  g[8] = 4.0 * z;
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

static void initial(double *pos) {
  for (int i = 0; i < N_IMAGES; ++i) {
    const double t = (double)i / (double)(N_IMAGES - 1);
    double *p = pos + i * DOF;
    p[0] = 0.0;
    p[1] = 0.0;
    p[2] = 0.0;
    p[3] = 2.5;
    p[4] = 0.0;
    p[5] = 0.0;
    p[6] = 2.0 * t;
    p[7] = 1.5 + 0.3 * sin(3.14159265358979 * t);
    p[8] = 0.0;
  }
}

static rgsaddle_band_config_t config(void) {
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
  cfg.force_tol = 1e-4;
  cfg.max_move = 0.1;
  return cfg;
}

/* Run to convergence and check that atoms 0 and 1 kept every bit. */
static int converge(RgsaddleBand *band, const double *start, const char *label) {
  rgsaddle_report_t rep;
  memset(&rep, 0, sizeof rep);
  int steps = 0;
  do {
    rgsaddle_status_t st = rgsaddle_band_step(band, surface, NULL, &rep);
    if (st != RGSADDLE_OK) {
      fprintf(stderr, "step %d: ", steps);
      return fail(label, st);
    }
    ++steps;
  } while (rep.status == RGSADDLE_STATUS_RUNNING && steps < 5000);
  if (rep.status != RGSADDLE_STATUS_CONVERGED || rep.ci_index < 1) {
    fprintf(stderr, "%s: max force %g, ci %lld\n", label, rep.max_force,
            (long long)rep.ci_index);
    return 1;
  }
  double out[N_IMAGES * DOF];
  if (rgsaddle_band_positions(band, out) != RGSADDLE_OK) {
    return 1;
  }
  if (memcmp(out, start, sizeof out) == 0) {
    fprintf(stderr, "%s: the band did not move\n", label);
    return 1;
  }
  for (int i = 0; i < N_IMAGES; ++i) {
    if (memcmp(out + i * DOF, start + i * DOF, 6 * sizeof(double)) != 0) {
      fprintf(stderr, "%s: a fixed atom moved in image %d\n", label, i);
      return 1;
    }
  }
  if (fabs(out[rep.ci_index * DOF + 6] - 1.0) > 1e-3) {
    fprintf(stderr, "%s: climbing image off the saddle: %g\n", label,
            out[rep.ci_index * DOF + 6]);
    return 1;
  }
  return 0;
}

int main(void) {
  double pos[N_IMAGES * DOF];
  initial(pos);
  const rgsaddle_band_config_t cfg = config();
  const uint8_t mask[N_ATOMS] = {1, 1, 0};

  if (rgsaddle_band_set_fixed_atoms(NULL, mask, N_ATOMS) != RGSADDLE_NULL_BAND) {
    return fail("a null band must be refused", RGSADDLE_NULL_BAND);
  }
  RgsaddleBand *band = rgsaddle_band_create(&cfg, N_IMAGES, N_ATOMS, pos);
  if (!band) {
    return fail("create", RGSADDLE_SHAPE);
  }
  if (rgsaddle_band_set_fixed_atoms(band, NULL, N_ATOMS) != RGSADDLE_SHAPE) {
    return fail("a null mask must be RGSADDLE_SHAPE", RGSADDLE_SHAPE);
  }
  if (rgsaddle_band_set_fixed_atoms(band, mask, N_ATOMS - 1) != RGSADDLE_SHAPE ||
      rgsaddle_band_set_fixed_atoms(band, mask, -1) != RGSADDLE_SHAPE) {
    return fail("a wrong n_atoms must be RGSADDLE_SHAPE", RGSADDLE_SHAPE);
  }
  const uint8_t two[N_ATOMS] = {1, 2, 0};
  if (rgsaddle_band_set_fixed_atoms(band, two, N_ATOMS) != RGSADDLE_INVALID_PARAMETER) {
    return fail("a byte other than 0 or 1 must be refused", RGSADDLE_INVALID_PARAMETER);
  }
  const uint8_t all[N_ATOMS] = {1, 1, 1};
  if (rgsaddle_band_set_fixed_atoms(band, all, N_ATOMS) != RGSADDLE_INVALID_PARAMETER) {
    return fail("every atom fixed must be refused", RGSADDLE_INVALID_PARAMETER);
  }
  rgsaddle_status_t st = rgsaddle_band_set_fixed_atoms(band, mask, N_ATOMS);
  if (st != RGSADDLE_OK) {
    return fail("set_fixed_atoms", st);
  }
  if (converge(band, pos, "FIRE") != 0) {
    return 1;
  }

  /* An endpoint that moves a fixed atom is refused and the band stays. */
  double moved[N_IMAGES * DOF];
  if (rgsaddle_band_positions(band, moved) != RGSADDLE_OK) {
    return 1;
  }
  double kept[N_IMAGES * DOF];
  memcpy(kept, moved, sizeof kept);
  moved[(N_IMAGES - 1) * DOF + 3] += 1e-6;
  if (rgsaddle_band_set_positions(band, moved) != RGSADDLE_INVALID_PARAMETER) {
    return fail("a moved fixed endpoint atom must be refused", RGSADDLE_INVALID_PARAMETER);
  }
  if (rgsaddle_band_positions(band, moved) != RGSADDLE_OK ||
      memcmp(moved, kept, sizeof kept) != 0) {
    return fail("a refused set_positions changed the band", RGSADDLE_INVALID_PARAMETER);
  }
  rgsaddle_band_free(band);

  /* Mismatched endpoints are refused by set_fixed_atoms as well. */
  double skew[N_IMAGES * DOF];
  memcpy(skew, pos, sizeof skew);
  skew[(N_IMAGES - 1) * DOF] = 0.5;
  band = rgsaddle_band_create(&cfg, N_IMAGES, N_ATOMS, skew);
  if (!band ||
      rgsaddle_band_set_fixed_atoms(band, mask, N_ATOMS) != RGSADDLE_INVALID_PARAMETER) {
    return fail("endpoints that differ on a fixed atom must be refused",
                RGSADDLE_INVALID_PARAMETER);
  }
  rgsaddle_band_free(band);

  /* The exponential metric with the mask, set in either order. */
  for (int order = 0; order < 2; ++order) {
    band = rgsaddle_band_create(&cfg, N_IMAGES, N_ATOMS, pos);
    if (!band) {
      return 1;
    }
    if (order == 0) {
      st = rgsaddle_band_set_fixed_atoms(band, mask, N_ATOMS);
    }
    if (st == RGSADDLE_OK) {
      st = rgsaddle_band_set_exp_precon(band, 3.0, 0.1, 0.0, 2.0);
    }
    if (st == RGSADDLE_OK && order == 1) {
      st = rgsaddle_band_set_fixed_atoms(band, mask, N_ATOMS);
    }
    if (st != RGSADDLE_OK) {
      return fail("exp precon with fixed atoms", st);
    }
    if (converge(band, pos, order == 0 ? "FIRE-Exp, mask first" : "FIRE-Exp, metric first") != 0) {
      return 1;
    }
    rgsaddle_band_free(band);
  }

  /* One-shot projection: a null mask is rgsaddle_band_project. */
  double energies[N_IMAGES];
  double gradients[N_IMAGES * DOF];
  for (int i = 0; i < N_IMAGES; ++i) {
    energy(pos + i * DOF, energies + i, gradients + i * DOF);
  }
  double plain[N_IMAGES * DOF], none[N_IMAGES * DOF], fixed[N_IMAGES * DOF];
  int64_t ci_plain = 0, ci_none = 0, ci_fixed = 0;
  if (rgsaddle_band_project(&cfg, N_IMAGES, N_ATOMS, pos, energies, gradients, plain,
                            &ci_plain) != RGSADDLE_OK ||
      rgsaddle_band_project_fixed(&cfg, N_IMAGES, N_ATOMS, pos, energies, gradients,
                                  NULL, none, &ci_none) != RGSADDLE_OK ||
      memcmp(plain, none, sizeof plain) != 0 || ci_plain != ci_none) {
    return fail("a null mask must match rgsaddle_band_project", RGSADDLE_OK);
  }
  st = rgsaddle_band_project_fixed(&cfg, N_IMAGES, N_ATOMS, pos, energies, gradients,
                                   mask, fixed, &ci_fixed);
  if (st != RGSADDLE_OK) {
    return fail("project_fixed", st);
  }
  for (int i = 0; i < N_IMAGES; ++i) {
    for (int k = 0; k < 6; ++k) {
      if (fixed[i * DOF + k] != 0.0) {
        return fail("a fixed atom carries a projected force", RGSADDLE_OK);
      }
    }
  }
  if (plain[3 * DOF] == 0.0) {
    return fail("the unmasked projection should load atom 0", RGSADDLE_OK);
  }
  if (rgsaddle_band_project_fixed(&cfg, N_IMAGES, N_ATOMS, pos, energies, gradients,
                                  two, fixed, &ci_fixed) != RGSADDLE_INVALID_PARAMETER ||
      rgsaddle_band_project_fixed(&cfg, N_IMAGES, N_ATOMS, pos, energies, gradients,
                                  all, fixed, &ci_fixed) != RGSADDLE_INVALID_PARAMETER) {
    return fail("project_fixed must refuse a bad mask", RGSADDLE_INVALID_PARAMETER);
  }

  printf("RGSADDLE_FIXED_ATOMS_ABI_OK\n");
  return 0;
}
