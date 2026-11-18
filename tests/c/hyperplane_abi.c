/* C ABI contract and TI accuracy test for rgsaddle_hyperplane_*. */
#include "rgsaddle.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(int code, const char *msg) {
  fprintf(stderr, "%s (status=%d %s)\n", msg, code,
          rgsaddle_status_name((rgsaddle_status_t)code));
  return 1;
}

typedef struct {
  double c4;
  int saw_positions_flag_zero;
  int saw_positions_flag_one;
  int saw_finite_lambda;
  int saw_nan_derivative;
  int emit_projected_force;
  int fail_after;
  int calls;
} ti_ctx_t;

static double splitmix64(uint64_t *state) {
  uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  z = z ^ (z >> 31);
  return (double)(z >> 11) * (1.0 / 9007199254740992.0);
}

static double normal_box_muller(uint64_t *state) {
  double u1 = splitmix64(state);
  double u2 = splitmix64(state);
  if (u1 < 1e-15) {
    u1 = 1e-15;
  }
  return sqrt(-2.0 * log(u1)) * cos(6.28318530717958647692 * u2);
}

static int32_t sample_quartic_plane(
    void *user, const double *point, const double *normal, size_t dimension,
    const rgsaddle_hyperplane_budget_t *budget, rgsaddle_hyperplane_emit_fn emit,
    void *emit_context) {
  ti_ctx_t *ctx = (ti_ctx_t *)user;
  (void)normal;
  ctx->calls += 1;
  if (ctx->fail_after > 0 && ctx->calls >= ctx->fail_after) {
    return RGSADDLE_SURFACE_FAILED;
  }
  if (budget->version.major != RGSADDLE_ABI_MAJOR || dimension != 2) {
    return RGSADDLE_SHAPE;
  }
  if (budget->positions == 0) {
    ctx->saw_positions_flag_zero = 1;
  } else {
    ctx->saw_positions_flag_one = 1;
  }
  if (!isnan(budget->lambda)) {
    ctx->saw_finite_lambda = 1;
  }

  const double lambda = isnan(budget->lambda) ? 1.0 : budget->lambda;
  const double kt = budget->thermal;
  const double step_sigma = 0.6 * sqrt(kt);
  uint64_t rng = budget->seed ^ 0x123456789abcdefULL;
  double y = point[1];
  double v = 0.5 * y * y + lambda * ctx->c4 * y * y * y * y;

  const int64_t total =
      budget->thermalize + budget->stride * budget->samples;
  for (int64_t step = 0; step < total; ++step) {
    double y_prop = y + step_sigma * normal_box_muller(&rng);
    double v_prop =
        0.5 * y_prop * y_prop + lambda * ctx->c4 * y_prop * y_prop * y_prop * y_prop;
    double dv = v_prop - v;
    if (dv <= 0.0 || splitmix64(&rng) < exp(-dv / kt)) {
      y = y_prop;
      v = v_prop;
    }
    if (step >= budget->thermalize &&
        (step - budget->thermalize) % budget->stride == 0) {
      double pos[2] = {point[0], y};
      double pforce[2] = {0.0, -(y + 4.0 * lambda * ctx->c4 * y * y * y)};
      double deriv = ctx->c4 * y * y * y * y;
      rgsaddle_hyperplane_observation_t obs;
      obs.constraint_force = point[0];
      obs.position = budget->positions ? pos : NULL;
      obs.projected_force = ctx->emit_projected_force ? pforce : NULL;
      obs.thermodynamic_derivative = ctx->saw_nan_derivative ? NAN : deriv;
      int32_t rc = emit(emit_context, &obs);
      if (rc != RGSADDLE_OK) {
        return rc;
      }
    }
  }
  return RGSADDLE_OK;
}

static double exact_quartic_ti_free_energy(double c4, double kt) {
  const int n = 4000;
  const double limit = 8.0;
  const double h = (2.0 * limit) / (double)n;
  double z_harm = 0.0;
  double z_full = 0.0;
  for (int i = 0; i <= n; ++i) {
    double y = -limit + h * (double)i;
    double w = (i == 0 || i == n) ? 0.5 : 1.0;
    double vh = 0.5 * y * y;
    double vf = vh + c4 * y * y * y * y;
    z_harm += w * exp(-vh / kt);
    z_full += w * exp(-vf / kt);
  }
  return -kt * log(z_full / z_harm);
}

int main(void) {
  const char *obs_name = rgsaddle_thermodynamic_derivative_observable();
  if (!obs_name || strcmp(obs_name, "potential.full_minus_harmonic") != 0) {
    fprintf(stderr, "unexpected observable name: %s\n", obs_name ? obs_name : "NULL");
    return 1;
  }

  if (rgsaddle_hyperplane_sampler_create(NULL, NULL) != NULL) {
    return fail(RGSADDLE_INVALID_PARAMETER, "NULL callback must return NULL");
  }
  rgsaddle_hyperplane_sampler_free(NULL);

  ti_ctx_t ctx = {0.25, 0, 0, 0, 0, 1, 0, 0};
  RgsaddleHyperplaneSampler *sampler =
      rgsaddle_hyperplane_sampler_create(&ctx, sample_quartic_plane);
  if (!sampler) {
    return fail(RGSADDLE_ALLOC, "create sampler");
  }

  const double point[2] = {0.0, 0.0};
  const double normal[2] = {1.0, 0.0};
  const double zero_normal[2] = {0.0, 0.0};
  rgsaddle_hyperplane_ti_config_t cfg;
  memset(&cfg, 0, sizeof cfg);
  cfg.version.major = RGSADDLE_ABI_MAJOR;
  cfg.version.minor = RGSADDLE_ABI_MINOR;
  cfg.thermal = 0.5;
  cfg.nodes = 6;
  cfg.thermalize = 200;
  cfg.stride = 2;
  cfg.samples = 800;
  cfg.blocks = 8;
  cfg.seed = 17;

  rgsaddle_hyperplane_ti_result_t res;
  memset(&res, 0, sizeof res);

  if (rgsaddle_hyperplane_thermodynamic_integration(
          NULL, point, normal, 2, &cfg, &res) != RGSADDLE_NULL_SESSION) {
    return fail(RGSADDLE_NULL_SESSION, "null sampler");
  }
  if (rgsaddle_hyperplane_thermodynamic_integration(
          sampler, NULL, normal, 2, &cfg, &res) != RGSADDLE_NULL_SURFACE) {
    return fail(RGSADDLE_NULL_SURFACE, "null point");
  }
  if (rgsaddle_hyperplane_thermodynamic_integration(
          sampler, point, NULL, 2, &cfg, &res) != RGSADDLE_NULL_SURFACE) {
    return fail(RGSADDLE_NULL_SURFACE, "null normal");
  }
  if (rgsaddle_hyperplane_thermodynamic_integration(
          sampler, point, normal, 2, NULL, &res) != RGSADDLE_NULL_REPORT) {
    return fail(RGSADDLE_NULL_REPORT, "null config");
  }
  if (rgsaddle_hyperplane_thermodynamic_integration(
          sampler, point, normal, 2, &cfg, NULL) != RGSADDLE_NULL_REPORT) {
    return fail(RGSADDLE_NULL_REPORT, "null result");
  }
  if (rgsaddle_hyperplane_thermodynamic_integration(
          sampler, point, normal, 0, &cfg, &res) != RGSADDLE_NULL_REPORT) {
    return fail(RGSADDLE_NULL_REPORT, "zero dimension");
  }
  if (rgsaddle_hyperplane_thermodynamic_integration(
          sampler, point, zero_normal, 2, &cfg, &res) != RGSADDLE_INVALID_PARAMETER) {
    return fail(RGSADDLE_INVALID_PARAMETER, "zero normal");
  }

  rgsaddle_hyperplane_ti_config_t bad = cfg;
  bad.version.major = 99;
  if (rgsaddle_hyperplane_thermodynamic_integration(
          sampler, point, normal, 2, &bad, &res) != RGSADDLE_INVALID_PARAMETER) {
    return fail(RGSADDLE_INVALID_PARAMETER, "bad major");
  }
  bad = cfg;
  bad.nodes = 0;
  if (rgsaddle_hyperplane_thermodynamic_integration(
          sampler, point, normal, 2, &bad, &res) != RGSADDLE_INVALID_PARAMETER) {
    return fail(RGSADDLE_INVALID_PARAMETER, "non-positive nodes");
  }
  bad = cfg;
  bad.thermalize = -1;
  if (rgsaddle_hyperplane_thermodynamic_integration(
          sampler, point, normal, 2, &bad, &res) != RGSADDLE_INVALID_PARAMETER) {
    return fail(RGSADDLE_INVALID_PARAMETER, "negative thermalize");
  }
  bad = cfg;
  bad.stride = 0;
  if (rgsaddle_hyperplane_thermodynamic_integration(
          sampler, point, normal, 2, &bad, &res) != RGSADDLE_INVALID_PARAMETER) {
    return fail(RGSADDLE_INVALID_PARAMETER, "non-positive stride");
  }
  bad = cfg;
  bad.samples = 0;
  if (rgsaddle_hyperplane_thermodynamic_integration(
          sampler, point, normal, 2, &bad, &res) != RGSADDLE_INVALID_PARAMETER) {
    return fail(RGSADDLE_INVALID_PARAMETER, "non-positive samples");
  }
  bad = cfg;
  bad.blocks = 0;
  if (rgsaddle_hyperplane_thermodynamic_integration(
          sampler, point, normal, 2, &bad, &res) != RGSADDLE_INVALID_PARAMETER) {
    return fail(RGSADDLE_INVALID_PARAMETER, "non-positive blocks");
  }

  int32_t st = rgsaddle_hyperplane_thermodynamic_integration(
      sampler, point, normal, 2, &cfg, &res);
  if (st != RGSADDLE_OK) {
    return fail(st, "quartic TI failed");
  }
  const double expected = exact_quartic_ti_free_energy(ctx.c4, cfg.thermal);
  if (fabs(res.free_energy - expected) > 3.5 * res.error + 0.01 ||
      !(res.free_energy > 0.0) || !(res.error > 0.0) ||
      res.evaluations != cfg.nodes * cfg.samples) {
    fprintf(stderr, "TI mismatch: got %.6f +- %.6f, exact %.6f, evals=%lld\n",
            res.free_energy, res.error, expected, (long long)res.evaluations);
    rgsaddle_hyperplane_sampler_free(sampler);
    return 1;
  }
  if (!ctx.saw_positions_flag_zero || !ctx.saw_finite_lambda) {
    fprintf(stderr, "expected positions==0 and finite lambda\n");
    rgsaddle_hyperplane_sampler_free(sampler);
    return 1;
  }

  /* Callback omitting thermodynamic derivative must fail the TI node. */
  ctx.saw_nan_derivative = 1;
  st = rgsaddle_hyperplane_thermodynamic_integration(
      sampler, point, normal, 2, &cfg, &res);
  if (st == RGSADDLE_OK) {
    fprintf(stderr, "missing derivative should fail TI\n");
    rgsaddle_hyperplane_sampler_free(sampler);
    return 1;
  }

  /* Callback failure propagates as RGSADDLE_SURFACE_FAILED. */
  ctx.saw_nan_derivative = 0;
  ctx.calls = 0;
  ctx.fail_after = 2;
  st = rgsaddle_hyperplane_thermodynamic_integration(
      sampler, point, normal, 2, &cfg, &res);
  if (st != RGSADDLE_SURFACE_FAILED) {
    return fail(st, "expected callback failure propagation");
  }

  rgsaddle_hyperplane_sampler_free(sampler);
  printf("RGSADDLE_HYPERPLANE_ABI_OK dF=%.6f exact=%.6f\n", res.free_energy, expected);
  return 0;
}
