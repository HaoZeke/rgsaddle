/* The refine stepper is the config's refine_method and refine_threshold,
 * which mirror eOn's [Refine] section. A zeroed config and
 * RGSADDLE_METHOD_FIRE step with FIRE alone whatever the minor, a config
 * older than minor 45 never reads the refine members, a zero threshold is
 * eOn's 0.5, each step picks its stepper by the convergence force, the
 * OCI-NEB band takes the same stepper as a band, and a string band
 * ignores it. */
#include "rgsaddle.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define N_IMAGES 9
#define N_DOF (N_IMAGES * 3)
#define OLD_MINOR 44u
#define MAX_STEPS 4000

static int fail(const char *msg) {
  fprintf(stderr, "%s\n", msg);
  return 1;
}

static rgsaddle_status_t surface(void *user, rgsaddle_surface_request_t *req) {
  (void)user;
  if (req->version.major != RGSADDLE_ABI_MAJOR || req->n_atoms != 1) {
    return RGSADDLE_SHAPE;
  }
  for (long i = 0; i < req->n_images; ++i) {
    const double *p = req->positions + i * 3;
    double *g = req->gradients + i * 3;
    const double x = p[0], y = p[1], z = p[2];
    req->energies[i] = (x * x - 1.0) * (x * x - 1.0) + 2.0 * y * y + 2.0 * z * z;
    g[0] = 4.0 * x * (x * x - 1.0);
    g[1] = 4.0 * y;
    g[2] = 4.0 * z;
  }
  return RGSADDLE_OK;
}

static void bowed_path(double *pos, double bow) {
  memset(pos, 0, N_DOF * sizeof *pos);
  for (long i = 0; i < N_IMAGES; ++i) {
    const double t = (double)i / (double)(N_IMAGES - 1);
    pos[i * 3 + 0] = -1.0 + 2.0 * t;
    pos[i * 3 + 1] = bow * sin(3.141592653589793 * t);
  }
}

static void initial_path(double *pos) { bowed_path(pos, 0.3); }

static rgsaddle_band_config_t band_config(uint32_t minor, int32_t refine_method,
                                          double refine_threshold) {
  rgsaddle_band_config_t cfg;
  memset(&cfg, 0, sizeof cfg);
  cfg.version.major = RGSADDLE_ABI_MAJOR;
  cfg.version.minor = minor;
  cfg.tangent = RGSADDLE_TANGENT_IMPROVED;
  cfg.spring = RGSADDLE_SPRING_UNIFORM;
  cfg.projection = RGSADDLE_PROJECTION_NEB;
  cfg.method = RGSADDLE_METHOD_FIRE;
  cfg.refine_method = refine_method;
  cfg.refine_threshold = refine_threshold;
  cfg.spring_k = 5.0;
  cfg.ci_trigger_factor = 0.5;
  cfg.force_tol = 1e-3;
  cfg.max_move = 0.5;
  return cfg;
}

/* No climbing image and no min-mode refinement, so a band moved onto a
 * new path carries nothing but its steppers' state. */
static void plain_band(rgsaddle_band_config_t *cfg) {
  cfg->ci_trigger_factor = 0.0;
  cfg->flags |= RGSADDLE_BAND_NO_MIN_MODE;
  cfg->max_move = 5.0;
}

typedef struct {
  int steps;
  int converged;
  double positions[N_DOF];
} trajectory_t;

/* Steps a plain band to convergence. Returns 1 when create refuses. */
static int run_band(const rgsaddle_band_config_t *cfg, trajectory_t *out) {
  double pos[N_DOF];
  initial_path(pos);
  RgsaddleBand *band = rgsaddle_band_create(cfg, N_IMAGES, 1, pos);
  if (!band) {
    return 1;
  }
  rgsaddle_report_t rep;
  memset(&rep, 0, sizeof rep);
  memset(out, 0, sizeof *out);
  do {
    if (rgsaddle_band_step(band, surface, NULL, &rep) != RGSADDLE_OK) {
      rgsaddle_band_free(band);
      return 2;
    }
    ++out->steps;
  } while (rep.status == RGSADDLE_STATUS_RUNNING && out->steps < MAX_STEPS);
  out->converged = rep.status == RGSADDLE_STATUS_CONVERGED;
  rgsaddle_band_positions(band, out->positions);
  rgsaddle_band_free(band);
  return 0;
}

static double largest_difference(const double *a, const double *b) {
  double worst = 0.0;
  for (int k = 0; k < N_DOF; ++k) {
    worst = fmax(worst, fabs(a[k] - b[k]));
  }
  return worst;
}

static int same_path(const trajectory_t *a, const trajectory_t *b) {
  return a->steps == b->steps &&
         memcmp(a->positions, b->positions, sizeof a->positions) == 0;
}

static void band_after_one_step(const rgsaddle_band_config_t *cfg,
                                const double *from, double *out) {
  double pos[N_DOF];
  memcpy(pos, from, sizeof pos);
  RgsaddleBand *band = rgsaddle_band_create(cfg, N_IMAGES, 1, pos);
  rgsaddle_report_t rep;
  memset(&rep, 0, sizeof rep);
  if (band) {
    rgsaddle_band_step(band, surface, NULL, &rep);
    rgsaddle_band_positions(band, out);
    rgsaddle_band_free(band);
  } else {
    memset(out, 0, N_DOF * sizeof *out);
  }
}

/* Steps a band from the given path to convergence, moves it to the
 * second path and takes one step from there. */
static int step_after_move(const rgsaddle_band_config_t *cfg, const double *from,
                           const double *to, double *out) {
  double pos[N_DOF];
  memcpy(pos, from, sizeof pos);
  RgsaddleBand *band = rgsaddle_band_create(cfg, N_IMAGES, 1, pos);
  if (!band) {
    return 1;
  }
  rgsaddle_report_t rep;
  memset(&rep, 0, sizeof rep);
  int steps = 0;
  do {
    if (rgsaddle_band_step(band, surface, NULL, &rep) != RGSADDLE_OK) {
      rgsaddle_band_free(band);
      return 2;
    }
  } while (rep.status == RGSADDLE_STATUS_RUNNING && ++steps < MAX_STEPS);
  rgsaddle_band_set_positions(band, to);
  const rgsaddle_status_t status = rgsaddle_band_step(band, surface, NULL, &rep);
  rgsaddle_band_positions(band, out);
  rgsaddle_band_free(band);
  return status == RGSADDLE_OK ? 0 : 3;
}

static void ocineb_after_one_step(const rgsaddle_band_config_t *band_cfg,
                                  double *out) {
  rgsaddle_minmode_config_t mm;
  memset(&mm, 0, sizeof mm);
  mm.version.major = RGSADDLE_ABI_MAJOR;
  mm.version.minor = RGSADDLE_ABI_MINOR;
  mm.kind = RGSADDLE_MINMODE_DIMER;
  mm.method = RGSADDLE_METHOD_FIRE;
  mm.dr = 1e-3;
  mm.rotation_tol = 1e-4;
  mm.max_rotations = 20;
  mm.krylov_dim = 8;
  mm.force_tol = 1e-3;
  mm.max_move = 0.1;
  mm.difference = RGSADDLE_DIFFERENCE_FORWARD;

  rgsaddle_ocineb_config_t cfg;
  memset(&cfg, 0, sizeof cfg);
  cfg.version.major = RGSADDLE_ABI_MAJOR;
  cfg.version.minor = RGSADDLE_ABI_MINOR;
  cfg.trigger_factor = 0.31;
  cfg.angle_tol = 0.85;
  cfg.stability_count = 5;
  cfg.max_mmf_steps = 1000;

  double pos[N_DOF];
  initial_path(pos);
  memset(out, 0, N_DOF * sizeof *out);
  RgsaddleOcineb *session =
      rgsaddle_ocineb_create(band_cfg, &mm, &cfg, N_IMAGES, 1, pos);
  if (!session) {
    return;
  }
  rgsaddle_ocineb_report_t report;
  memset(&report, 0, sizeof report);
  if (rgsaddle_ocineb_step(session, surface, NULL, &report) == RGSADDLE_OK) {
    rgsaddle_ocineb_positions(session, out);
  }
  rgsaddle_ocineb_free(session);
}

_Static_assert(offsetof(rgsaddle_band_config_t, refine_threshold) ==
                   sizeof(rgsaddle_band_config_v44_t),
               "the first appended member starts where the old layout ends");
_Static_assert(offsetof(rgsaddle_band_config_t, refine_threshold) %
                       _Alignof(rgsaddle_band_config_t) ==
                   0,
               "an old layout placed at the end of a page stays aligned");

/* Puts a minor 44 config in the last bytes of a writable page. The
 * refine members would sit in the following guard page. */
static int old_minor_guard(void) {
  const long page = sysconf(_SC_PAGESIZE);
  const size_t old = offsetof(rgsaddle_band_config_t, refine_threshold);
  if (page < (long)old) {
    return 1;
  }
  void *mem = mmap(NULL, (size_t)page * 2, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (mem == MAP_FAILED) {
    return 1;
  }
  if (mprotect((char *)mem + page, (size_t)page, PROT_NONE) != 0) {
    munmap(mem, (size_t)page * 2);
    return 1;
  }
  char *base = (char *)mem + page - (long)old;
  memset(base, 0, old);
  rgsaddle_band_config_v44_t *prefix = (rgsaddle_band_config_v44_t *)base;
  prefix->version.major = RGSADDLE_ABI_MAJOR;
  prefix->version.minor = OLD_MINOR;
  prefix->tangent = RGSADDLE_TANGENT_IMPROVED;
  prefix->spring = RGSADDLE_SPRING_UNIFORM;
  prefix->projection = RGSADDLE_PROJECTION_NEB;
  prefix->method = RGSADDLE_METHOD_FIRE;
  prefix->spring_k = 5.0;
  prefix->ci_trigger_factor = 0.5;
  prefix->force_tol = 1e-3;
  prefix->max_move = 0.5;
  const rgsaddle_band_config_t *cfg = (const rgsaddle_band_config_t *)prefix;

  double pos[N_DOF];
  initial_path(pos);
  RgsaddleBand *band = rgsaddle_band_create(cfg, N_IMAGES, 1, pos);
  if (!band) {
    munmap(mem, (size_t)page * 2);
    return 1;
  }
  rgsaddle_band_free(band);

  double cells[N_IMAGES * 9];
  memset(cells, 0, sizeof cells);
  for (int i = 0; i < N_IMAGES; ++i) {
    cells[i * 9 + 0] = 4.0;
    cells[i * 9 + 4] = 4.0;
    cells[i * 9 + 8] = 4.0;
  }
  RgsaddleBand *solid =
      rgsaddle_band_create_solid(cfg, N_IMAGES, 1, pos, cells, 0.0, 1.0);
  if (!solid) {
    munmap(mem, (size_t)page * 2);
    return 1;
  }
  rgsaddle_band_free(solid);

  double energies[N_IMAGES];
  double gradients[N_DOF];
  double forces[N_DOF];
  int64_t ci = -2;
  memset(energies, 0, sizeof energies);
  memset(gradients, 0, sizeof gradients);
  if (rgsaddle_band_project(cfg, N_IMAGES, 1, pos, energies, gradients, forces,
                            &ci) != RGSADDLE_OK) {
    munmap(mem, (size_t)page * 2);
    return 1;
  }
  const uint8_t fixed = 0;
  if (rgsaddle_band_project_fixed(cfg, N_IMAGES, 1, pos, energies, gradients,
                                  &fixed, forces, &ci) != RGSADDLE_OK) {
    munmap(mem, (size_t)page * 2);
    return 1;
  }

  rgsaddle_minmode_config_t mm;
  memset(&mm, 0, sizeof mm);
  mm.version.major = RGSADDLE_ABI_MAJOR;
  mm.version.minor = OLD_MINOR;
  mm.kind = RGSADDLE_MINMODE_DIMER;
  mm.method = RGSADDLE_METHOD_FIRE;
  mm.dr = 1e-3;
  mm.rotation_tol = 1e-4;
  mm.max_rotations = 10;
  mm.krylov_dim = 4;
  mm.force_tol = 1e-3;
  mm.max_move = 0.2;

  rgsaddle_ocineb_config_t oc;
  memset(&oc, 0, sizeof oc);
  oc.version.major = RGSADDLE_ABI_MAJOR;
  oc.version.minor = OLD_MINOR;
  oc.trigger_factor = 0.5;
  oc.angle_tol = 0.85;
  oc.stability_count = 1;
  oc.max_mmf_steps = 10;
  RgsaddleOcineb *session =
      rgsaddle_ocineb_create(cfg, &mm, &oc, N_IMAGES, 1, pos);
  if (!session) {
    munmap(mem, (size_t)page * 2);
    return 1;
  }
  rgsaddle_ocineb_free(session);
  munmap(mem, (size_t)page * 2);
  return 0;
}

int main(void) {
  trajectory_t old_fire, new_fire, switched, scratch;
  const double over_everything = 1e9;

  /* A caller stamped minor 44 that fills the refine members, as a host
   * built on a newer header might, still gets FIRE to the end. */
  rgsaddle_band_config_t cfg =
      band_config(OLD_MINOR, RGSADDLE_REFINE_LBFGS, over_everything);
  if (run_band(&cfg, &old_fire) != 0 || !old_fire.converged) {
    return fail("a minor 44 FIRE band did not converge");
  }
  cfg = band_config(RGSADDLE_ABI_MINOR, RGSADDLE_REFINE_NONE, 0.0);
  if (run_band(&cfg, &new_fire) != 0 || !new_fire.converged) {
    return fail("a current minor FIRE band did not converge");
  }
  if (!same_path(&old_fire, &new_fire)) {
    return fail("RGSADDLE_METHOD_FIRE must step the same at every minor");
  }

  cfg = band_config(RGSADDLE_ABI_MINOR, RGSADDLE_REFINE_LBFGS, over_everything);
  if (run_band(&cfg, &switched) != 0 || !switched.converged) {
    return fail("a FIRE then L-BFGS band did not converge");
  }
  if (same_path(&switched, &new_fire)) {
    return fail("the refine switch must leave FIRE");
  }
  if (switched.steps >= new_fire.steps) {
    fprintf(stderr, "switch %d steps, FIRE %d steps\n", switched.steps,
            new_fire.steps);
    return fail("the switch must need fewer steps than FIRE alone");
  }

  /* The threshold decides when: one the band never reaches is FIRE to
   * the end. Zero is eOn's 0.5, and a negative or NaN threshold, or a
   * refine method the library does not know, is refused. */
  trajectory_t never, half;
  cfg = band_config(RGSADDLE_ABI_MINOR, RGSADDLE_REFINE_LBFGS, 1e-3 * 1e-3);
  if (run_band(&cfg, &never) != 0 || !never.converged) {
    return fail("a band with a threshold it never reaches did not converge");
  }
  if (!same_path(&never, &new_fire)) {
    return fail("a threshold below the tolerance must leave FIRE to the end");
  }
  cfg = band_config(RGSADDLE_ABI_MINOR, RGSADDLE_REFINE_LBFGS, 0.0);
  if (run_band(&cfg, &scratch) != 0 || !scratch.converged) {
    return fail("a band with the default threshold did not converge");
  }
  cfg = band_config(RGSADDLE_ABI_MINOR, RGSADDLE_REFINE_LBFGS, 0.5);
  if (run_band(&cfg, &half) != 0 || !same_path(&scratch, &half)) {
    return fail("a zero threshold must be eOn's 0.5");
  }
  if (same_path(&half, &new_fire)) {
    return fail("the default threshold must refine the band");
  }
  const double bad_thresholds[] = {-1.0, NAN};
  for (size_t i = 0; i < sizeof bad_thresholds / sizeof bad_thresholds[0]; ++i) {
    cfg = band_config(RGSADDLE_ABI_MINOR, RGSADDLE_REFINE_LBFGS, bad_thresholds[i]);
    if (run_band(&cfg, &scratch) != 1) {
      return fail("a negative or NaN threshold must be refused");
    }
  }
  cfg = band_config(RGSADDLE_ABI_MINOR, 99, over_everything);
  if (run_band(&cfg, &scratch) != 1) {
    return fail("an unknown refine method must be refused");
  }
  /* With no refine method the threshold is not read. */
  cfg = band_config(RGSADDLE_ABI_MINOR, RGSADDLE_REFINE_NONE, -1.0);
  if (run_band(&cfg, &scratch) != 0 || !same_path(&scratch, &new_fire)) {
    return fail("RGSADDLE_REFINE_NONE must ignore the threshold");
  }

  /* The choice is made on every step. A band that starts under the
   * threshold only ever steps with the refine stepper. Moved onto a
   * path whose force is over the threshold it steps with FIRE, so its
   * move from there is the first FIRE step of a fresh FIRE band, and
   * differs from the move of a band that keeps the refine stepper on. */
  double near[N_DOF], far[N_DOF], fire_far[N_DOF], back[N_DOF], kept[N_DOF];
  bowed_path(near, 0.05);
  bowed_path(far, 3.0);
  cfg = band_config(RGSADDLE_ABI_MINOR, RGSADDLE_REFINE_NONE, 0.0);
  plain_band(&cfg);
  band_after_one_step(&cfg, far, fire_far);
  cfg = band_config(RGSADDLE_ABI_MINOR, RGSADDLE_REFINE_LBFGS, 0.0);
  plain_band(&cfg);
  if (step_after_move(&cfg, near, far, back) != 0) {
    return fail("the refine band did not step");
  }
  cfg = band_config(RGSADDLE_ABI_MINOR, RGSADDLE_REFINE_LBFGS, over_everything);
  plain_band(&cfg);
  if (step_after_move(&cfg, near, far, kept) != 0) {
    return fail("the always refined band did not step");
  }
  if (largest_difference(fire_far, back) > 1e-12) {
    return fail("a force back over the threshold must step with FIRE");
  }
  if (largest_difference(fire_far, kept) < 1e-6) {
    return fail("the refine stepper must move that step differently");
  }

  /* A string band drives its own stepper, so the refine members change
   * nothing for it. */
  trajectory_t string_alone, string_refine;
  cfg = band_config(RGSADDLE_ABI_MINOR, RGSADDLE_REFINE_NONE, 0.0);
  cfg.flags |= RGSADDLE_BAND_STRING | RGSADDLE_BAND_NO_MIN_MODE;
  cfg.ci_trigger_factor = 0.0;
  if (run_band(&cfg, &string_alone) != 0) {
    return fail("the string band was refused");
  }
  cfg = band_config(RGSADDLE_ABI_MINOR, RGSADDLE_REFINE_LBFGS, over_everything);
  cfg.flags |= RGSADDLE_BAND_STRING | RGSADDLE_BAND_NO_MIN_MODE;
  cfg.ci_trigger_factor = 0.0;
  if (run_band(&cfg, &string_refine) != 0 || !same_path(&string_alone, &string_refine)) {
    return fail("a string band must ignore the refine stepper");
  }

  /* The OCI-NEB band is the same stepper: a zeroed refine method is the
   * first move of a FIRE band, at either minor, and a refine switch is
   * the move of the switched band. */
  double start[N_DOF], fire_one[N_DOF], switch_one[N_DOF], oci_one[N_DOF], oci_old_one[N_DOF],
      oci_switch_one[N_DOF];
  initial_path(start);
  cfg = band_config(RGSADDLE_ABI_MINOR, RGSADDLE_REFINE_NONE, 0.0);
  band_after_one_step(&cfg, start, fire_one);
  ocineb_after_one_step(&cfg, oci_one);
  cfg = band_config(OLD_MINOR, RGSADDLE_REFINE_LBFGS, over_everything);
  ocineb_after_one_step(&cfg, oci_old_one);
  cfg = band_config(RGSADDLE_ABI_MINOR, RGSADDLE_REFINE_LBFGS, over_everything);
  band_after_one_step(&cfg, start, switch_one);
  ocineb_after_one_step(&cfg, oci_switch_one);
  if (memcmp(fire_one, switch_one, sizeof fire_one) == 0) {
    return fail("one step must tell FIRE from the switch");
  }
  if (memcmp(fire_one, oci_one, sizeof fire_one) != 0) {
    return fail("an OCI-NEB band without a refine method must step as FIRE");
  }
  if (memcmp(fire_one, oci_old_one, sizeof fire_one) != 0) {
    return fail("a minor 44 OCI-NEB band must step as FIRE");
  }
  if (memcmp(switch_one, oci_switch_one, sizeof switch_one) != 0) {
    return fail("an OCI-NEB band with a refine method must switch as a band");
  }

  /* A minor 44 caller owns only the layout through memory, sixteen
   * bytes short of the enlarged struct. The allocation ends on a guard
   * page, and every entry point that takes the band config must accept
   * it without reading the refine members. */
  if (old_minor_guard() != 0) {
    return fail("a minor 44 allocation the size of the old layout was refused");
  }

  printf("RGSADDLE_BAND_REFINE_ABI_OK\n");
  return 0;
}
