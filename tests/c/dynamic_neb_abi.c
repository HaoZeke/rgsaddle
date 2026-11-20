/* Dynamic NEB setters and the DyNEB mask, ABI minor 46. */
#include "rgsaddle.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int fail(const char *msg) {
  fprintf(stderr, "dynamic neb abi: %s\n", msg);
  return 1;
}

int main(void) {
  if (rgsaddle_abi_version() != (int)((RGSADDLE_ABI_MAJOR << 16) | RGSADDLE_ABI_MINOR)) {
    return fail("abi minor");
  }
  rgsaddle_version_t version = {0, 0};
  if (rgsaddle_abi_stamp(&version) != RGSADDLE_OK || version.major != 1 || version.minor != 46) {
    return fail("abi stamp");
  }
  if (rgsaddle_band_set_dynamic(NULL, 1, 0.05, 0.0) != RGSADDLE_NULL_BAND) {
    return fail("null band");
  }
  if (rgsaddle_dyneb_mask(2, 1, NULL, NULL, NULL, 0.05, 0.0, NULL) != RGSADDLE_SHAPE) {
    return fail("short band");
  }

  /* Three images, one atom. Interior force 0.01 is under fmax 0.05. */
  double positions[9] = {0, 0, 0, 1, 0, 0, 2, 0, 0};
  double energies[3] = {0.0, 1.0, 0.0};
  double forces[9] = {0, 0, 0, 0.01, 0, 0, 0, 0, 0};
  uint8_t frozen[3] = {0, 0, 0};
  if (rgsaddle_dyneb_mask(3, 1, positions, energies, forces, 0.05, 0.0, frozen) != RGSADDLE_OK) {
    return fail("mask");
  }
  /* The only interior image is the highest, so its force stays and it is frozen. */
  if (forces[3] != 0.01 || frozen[0] != 1 || frozen[1] != 1 || frozen[2] != 1) {
    return fail("highest image");
  }

  double pos[9];
  for (int i = 0; i < 3; ++i) {
    pos[3 * i] = -1.0 + (double)i;
    pos[3 * i + 1] = 0.0;
    pos[3 * i + 2] = 0.0;
  }
  rgsaddle_band_config_t cfg;
  memset(&cfg, 0, sizeof cfg);
  cfg.version.major = RGSADDLE_ABI_MAJOR;
  cfg.version.minor = RGSADDLE_ABI_MINOR;
  cfg.tangent = RGSADDLE_TANGENT_IMPROVED;
  cfg.spring = RGSADDLE_SPRING_UNIFORM;
  cfg.projection = RGSADDLE_PROJECTION_NEB;
  cfg.method = RGSADDLE_METHOD_FIRE;
  cfg.spring_k = 5.0;
  cfg.force_tol = 1e-3;
  cfg.max_move = 0.2;
  cfg.flags = RGSADDLE_BAND_NO_MIN_MODE;
  RgsaddleBand *band = rgsaddle_band_create(&cfg, 3, 1, pos);
  if (band == NULL) {
    return fail("create");
  }
  if (rgsaddle_band_set_dynamic(band, 2, 0.05, 0.0) != RGSADDLE_INVALID_PARAMETER) {
    return fail("enabled");
  }
  if (rgsaddle_band_set_dynamic(band, 1, NAN, 0.0) != RGSADDLE_NON_FINITE) {
    return fail("nan fmax");
  }
  if (rgsaddle_band_set_dynamic(band, 1, 0.05, 0.0) != RGSADDLE_OK) {
    return fail("set");
  }
  if (rgsaddle_band_set_dynamic(band, 0, -1.0, -1.0) != RGSADDLE_OK) {
    return fail("clear ignores the numbers");
  }
  rgsaddle_band_free(band);
  printf("RGSADDLE_DYNAMIC_NEB_ABI_OK\n");
  return 0;
}
