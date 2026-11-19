// Fixed-cell endpoint relaxation with the Sella minimum session.
#include "ss_geom.hpp"
#include "rgsaddle.h"
#include "rgpot/ForceStructs.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/fortran/FortranPots.hpp"

#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Surf {
  rgpot::LJPot *lj = nullptr;
  rgpot::fortranpots::TersoffPot *tersoff = nullptr;
  std::vector<int> z;
  long calls = 0;
};

struct FixedBox {
  Surf *surf;
  const double *cell;
};

bool eval_one(Surf &s, int n, const double *x, const double *h, double &energy, double *grad) {
  std::vector<double> force(static_cast<size_t>(3 * n));
  rgpot::ForceOut out{force.data(), 0.0, 0.0, {}, 0};
  rgpot::ForceInput in{static_cast<size_t>(n), x, s.z.data(), h};
  try {
    if (s.lj != nullptr) {
      s.lj->forceImpl(in, &out);
    } else {
      s.tersoff->forceImpl(in, &out);
    }
  } catch (const std::exception &) {
    return false;
  }
  energy = out.energy;
  for (int i = 0; i < 3 * n; ++i) {
    grad[i] = -force[static_cast<size_t>(i)];
  }
  s.calls += 1;
  return true;
}

rgsaddle_status_t fixed_box_cb(void *user, rgsaddle_surface_request_t *req) {
  auto *box = static_cast<FixedBox *>(user);
  const int64_t rows = (req->flags & RGSADDLE_REQ_ONE_IMAGE) ? 1 : req->n_images;
  const int64_t dof = 3 * req->n_atoms;
  for (int64_t row = 0; row < rows; ++row) {
    if (!eval_one(*box->surf, static_cast<int>(req->n_atoms), req->positions + row * dof, box->cell,
                  req->energies[row], req->gradients + row * dof)) {
      return RGSADDLE_SURFACE_FAILED;
    }
  }
  return RGSADDLE_OK;
}

void write_geom(const std::string &path, const Geom &g, const std::vector<double> &xyz) {
  std::ofstream geom(path);
  geom.setf(std::ios::scientific);
  geom.precision(16);
  geom << "n_images 1\nn_atoms " << g.n_atoms << "\npressure " << g.pressure << "\nepsilon "
       << g.epsilon << "\nsigma " << g.sigma << "\ncutoff " << g.cutoff << "\nsolid 0\npotential "
       << g.potential << "\nguard " << g.guard << "\nz " << g.z << "\nimage\n";
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      geom << g.cells[static_cast<size_t>(3 * r + c)] << (c == 2 ? "\n" : " ");
    }
  }
  for (int a = 0; a < g.n_atoms; ++a) {
    geom << xyz[static_cast<size_t>(3 * a)] << " " << xyz[static_cast<size_t>(3 * a + 1)] << " "
         << xyz[static_cast<size_t>(3 * a + 2)] << "\n";
  }
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "usage: ss_relax geom outdir tol max_iter\n";
    return 2;
  }
  const std::string outdir = argv[2];
  Geom g;
  if (!read_geom(argv[1], g)) {
    std::cerr << "could not read geometry\n";
    return 2;
  }
  if (g.n_images != 1) {
    std::cerr << "ss_relax expects one image\n";
    return 2;
  }
  Surf surf;
  surf.z.assign(static_cast<size_t>(g.n_atoms), g.z);
  rgpot::LJConfig lj_cfg;
  lj_cfg.u0 = g.epsilon;
  lj_cfg.psi = g.sigma;
  lj_cfg.cutoff = g.cutoff;
  lj_cfg.switch_width = 0.0;
  rgpot::LJPot lj(lj_cfg);
  rgpot::fortranpots::TersoffPot tersoff;
  if (g.potential == "lj") {
    surf.lj = &lj;
  } else if (g.potential == "tersoff") {
    surf.tersoff = &tersoff;
  } else {
    std::cerr << "unknown potential\n";
    return 2;
  }
  std::vector<double> masses(static_cast<size_t>(g.n_atoms), 1.0);
  rgsaddle_sella_min_config_t cfg{};
  rgsaddle_abi_stamp(&cfg.version);
  cfg.delta = 0.2;
  cfg.force_tol = std::atof(argv[3]);
  cfg.force_gate = RGSADDLE_FORCE_LINF;
  RgsaddleSellaMin *session =
      rgsaddle_sella_min_create(&cfg, g.n_atoms, g.xyz.data(), masses.data());
  if (session == nullptr) {
    std::cerr << "sella session refused\n";
    return 2;
  }
  FixedBox box{&surf, g.cells.data()};
  const long max_iter = std::atol(argv[4]);
  rgsaddle_report_t report{};
  int converged = 0;
  long iteration = 0;
  std::string status = "not_converged";
  for (; iteration < max_iter; ++iteration) {
    const rgsaddle_status_t step = rgsaddle_sella_min_step(session, fixed_box_cb, &box, &report);
    if (step != RGSADDLE_OK) {
      status = "surface_failed";
      break;
    }
    if (report.status == RGSADDLE_STATUS_CONVERGED) {
      converged = 1;
      status = "converged";
      iteration += 1;
      break;
    }
  }
  std::vector<double> xyz = g.xyz;
  if (rgsaddle_sella_min_position(session, xyz.data()) != RGSADDLE_OK) {
    status = "position_failed";
    converged = 0;
  }
  rgsaddle_sella_min_free(session);
  write_geom(outdir + "/relaxed.geom", g, xyz);
  std::ofstream out(outdir + "/result.txt");
  out.setf(std::ios::scientific);
  out.precision(16);
  out << "converged " << converged << "\n"
      << "iterations " << iteration << "\n"
      << "force_calls " << surf.calls << "\n"
      << "max_force " << report.max_force << "\n"
      << "status " << status << "\n"
      << "force_tol " << cfg.force_tol << "\n"
      << "force_gate linf\n"
      << "delta 0.2\n";
  return 0;
}
