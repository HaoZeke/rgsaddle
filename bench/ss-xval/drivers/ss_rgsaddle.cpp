// Solid-state and fixed-cell band runner. The surface is rgpot shifted LJ.
#include "rgsaddle.h"
#include "rgpot/ForceStructs.hpp"
#include "rgpot/LennardJones/LJPot.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Geom {
  int n_images = 0;
  int n_atoms = 0;
  double pressure = 0;
  double epsilon = 1;
  double sigma = 1;
  double cutoff = 1;
  int solid = 0;
  std::vector<double> cells;
  std::vector<double> xyz;
};

struct Surf {
  rgpot::LJPot *pot = nullptr;
  std::vector<int> z;
  long calls = 0;
};

bool read_geom(const std::string &path, Geom &g) {
  std::ifstream in(path);
  if (!in) {
    return false;
  }
  std::string key;
  while (in >> key) {
    if (key == "image") {
      break;
    }
    if (key == "n_images") {
      in >> g.n_images;
    } else if (key == "n_atoms") {
      in >> g.n_atoms;
    } else if (key == "pressure") {
      in >> g.pressure;
    } else if (key == "epsilon") {
      in >> g.epsilon;
    } else if (key == "sigma") {
      in >> g.sigma;
    } else if (key == "cutoff") {
      in >> g.cutoff;
    } else if (key == "solid") {
      in >> g.solid;
    } else {
      return false;
    }
  }
  if (g.n_images < 3 || g.n_atoms < 1) {
    return false;
  }
  g.cells.assign(static_cast<size_t>(g.n_images * 9), 0.0);
  g.xyz.assign(static_cast<size_t>(g.n_images * g.n_atoms * 3), 0.0);
  for (int image = 0; image < g.n_images; ++image) {
    if (image > 0) {
      in >> key;
      if (key != "image") {
        return false;
      }
    }
    for (int k = 0; k < 9; ++k) {
      in >> g.cells[static_cast<size_t>(image * 9 + k)];
    }
    for (int a = 0; a < g.n_atoms * 3; ++a) {
      in >> g.xyz[static_cast<size_t>(image * g.n_atoms * 3 + a)];
    }
  }
  return static_cast<bool>(in);
}

bool eval_one(Surf &s, int n, const double *x, const double *h, double &energy,
              double *grad, double *stress) {
  std::vector<double> force(static_cast<size_t>(3 * n));
  rgpot::ForceOut out{force.data(), 0.0, 0.0, {}, 0};
  rgpot::ForceInput in{static_cast<size_t>(n), x, s.z.data(), h};
  try {
    s.pot->forceImpl(in, &out);
  } catch (const std::exception &) {
    return false;
  }
  if (out.has_stress == 0) {
    return false;
  }
  energy = out.energy;
  for (int i = 0; i < 3 * n; ++i) {
    grad[i] = -force[static_cast<size_t>(i)];
  }
  for (int i = 0; i < 9; ++i) {
    stress[i] = out.stress[i];
  }
  s.calls += 1;
  return true;
}

rgsaddle_status_t solid_cb(void *user, rgsaddle_solid_surface_request_t *req) {
  auto *s = static_cast<Surf *>(user);
  const int64_t rows = (req->flags & RGSADDLE_REQ_ONE_IMAGE) ? 1 : req->n_images;
  const int64_t dof = 3 * req->n_atoms;
  for (int64_t row = 0; row < rows; ++row) {
    if (!eval_one(*s, static_cast<int>(req->n_atoms), req->positions + row * dof,
                  req->cells + 9 * row, req->energies[row],
                  req->gradients + row * dof, req->stresses + 9 * row)) {
      return RGSADDLE_SURFACE_FAILED;
    }
  }
  return RGSADDLE_OK;
}

struct FixedBox {
  Surf *surf;
  const double *cell;
};

rgsaddle_status_t fixed_box_cb(void *user, rgsaddle_surface_request_t *req) {
  auto *box = static_cast<FixedBox *>(user);
  const int64_t rows = (req->flags & RGSADDLE_REQ_ONE_IMAGE) ? 1 : req->n_images;
  const int64_t dof = 3 * req->n_atoms;
  double stress[9];
  for (int64_t row = 0; row < rows; ++row) {
    if (!eval_one(*box->surf, static_cast<int>(req->n_atoms),
                  req->positions + row * dof, box->cell, req->energies[row],
                  req->gradients + row * dof, stress)) {
      return RGSADDLE_SURFACE_FAILED;
    }
  }
  return RGSADDLE_OK;
}

double det3(const double *h) {
  return h[0] * (h[4] * h[8] - h[5] * h[7]) - h[1] * (h[3] * h[8] - h[5] * h[6]) +
         h[2] * (h[3] * h[7] - h[4] * h[6]);
}

bool inv3(const double *h, double *o) {
  const double d = det3(h);
  if (!(std::abs(d) > 1e-18)) {
    return false;
  }
  o[0] = (h[4] * h[8] - h[5] * h[7]) / d;
  o[1] = (h[2] * h[7] - h[1] * h[8]) / d;
  o[2] = (h[1] * h[5] - h[2] * h[4]) / d;
  o[3] = (h[5] * h[6] - h[3] * h[8]) / d;
  o[4] = (h[0] * h[8] - h[2] * h[6]) / d;
  o[5] = (h[2] * h[3] - h[0] * h[5]) / d;
  o[6] = (h[3] * h[7] - h[4] * h[6]) / d;
  o[7] = (h[1] * h[6] - h[0] * h[7]) / d;
  o[8] = (h[0] * h[4] - h[1] * h[3]) / d;
  return true;
}

double linear_h(double energy, const double *h, const double *h0, double pressure) {
  if (pressure == 0.0) {
    return energy;
  }
  double inv[9];
  if (!inv3(h0, inv)) {
    return energy;
  }
  double dh[9];
  for (int i = 0; i < 9; ++i) {
    dh[i] = h[i] - h0[i];
  }
  double trace = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int k = 0; k < 3; ++k) {
      trace += inv[3 * i + k] * dh[3 * k + i];
    }
  }
  return energy + pressure * trace * std::abs(det3(h0));
}

void write_geom(const std::string &path, const Geom &g, const std::vector<double> &xyz,
                const std::vector<double> &cells) {
  std::ofstream out(path);
  out.setf(std::ios::scientific);
  out.precision(16);
  out << "n_images " << g.n_images << "\n"
      << "n_atoms " << g.n_atoms << "\n"
      << "pressure " << g.pressure << "\n"
      << "epsilon " << g.epsilon << "\n"
      << "sigma " << g.sigma << "\n"
      << "cutoff " << g.cutoff << "\n"
      << "solid " << g.solid << "\n";
  for (int image = 0; image < g.n_images; ++image) {
    out << "image\n";
    for (int k = 0; k < 9; ++k) {
      out << cells[static_cast<size_t>(image * 9 + k)] << (k == 8 ? "\n" : " ");
    }
    for (int a = 0; a < g.n_atoms; ++a) {
      const size_t b = static_cast<size_t>(image * g.n_atoms * 3 + 3 * a);
      out << xyz[b] << " " << xyz[b + 1] << " " << xyz[b + 2] << "\n";
    }
  }
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 7) {
    std::cerr << "usage: ss_rgsaddle geom outdir spring ci tol max_iter\n";
    return 2;
  }
  Geom g;
  if (!read_geom(argv[1], g)) {
    std::cerr << "could not read geometry\n";
    return 2;
  }
  const std::string outdir = argv[2];
  const double spring = std::atof(argv[3]);
  const double ci = std::atof(argv[4]);
  const double tol = std::atof(argv[5]);
  const int max_iter = std::atoi(argv[6]);

  rgpot::LJConfig cfg;
  cfg.u0 = g.epsilon;
  cfg.psi = g.sigma;
  cfg.cutoff = g.cutoff;
  cfg.switch_width = 0.0;
  rgpot::LJPot pot(cfg);
  Surf surf;
  surf.pot = &pot;
  surf.z.assign(static_cast<size_t>(g.n_atoms), 18);

  rgsaddle_band_config_t band{};
  rgsaddle_abi_stamp(&band.version);
  band.flags = RGSADDLE_BAND_CI_AFTER | RGSADDLE_BAND_NO_MIN_MODE;
  band.tangent = RGSADDLE_TANGENT_IMPROVED;
  band.spring = RGSADDLE_SPRING_UNIFORM;
  band.projection = RGSADDLE_PROJECTION_NEB;
  band.method = RGSADDLE_METHOD_FIRE;
  band.spring_k = spring;
  band.ci_trigger_factor = 0.0;
  band.ci_trigger_force = ci;
  band.force_tol = tol;
  band.max_move = 0.2;
  if (!g.solid) {
    band.cell = g.cells.data();
  }

  RgsaddleBand *session = nullptr;
  if (g.solid) {
    session = rgsaddle_band_create_solid(&band, g.n_images, g.n_atoms, g.xyz.data(),
                                         g.cells.data(), g.pressure, 1.0);
  } else {
    session = rgsaddle_band_create(&band, g.n_images, g.n_atoms, g.xyz.data());
  }
  if (session == nullptr) {
    std::cerr << "band creation failed\n";
    return 3;
  }

  FixedBox box{&surf, g.cells.data()};
  rgsaddle_report_t report{};
  int converged = 0;
  std::string status = "not_converged";
  int steps = 0;
  for (; steps < max_iter; ++steps) {
    const rgsaddle_status_t code =
        g.solid ? rgsaddle_band_step_solid(session, solid_cb, &surf, &report)
                : rgsaddle_band_step(session, fixed_box_cb, &box, &report);
    if (code != RGSADDLE_OK) {
      status = rgsaddle_status_name(code);
      break;
    }
    if (report.status == RGSADDLE_STATUS_CONVERGED) {
      converged = 1;
      status = "converged";
      steps += 1;
      break;
    }
  }
  if (converged == 0 && status == "not_converged") {
    steps = max_iter;
  }

  std::vector<double> xyz(g.xyz.size());
  std::vector<double> cells(g.cells.size());
  rgsaddle_band_positions(session, xyz.data());
  if (g.solid) {
    rgsaddle_band_cells(session, cells.data());
  } else {
    for (int image = 0; image < g.n_images; ++image) {
      for (int k = 0; k < 9; ++k) {
        cells[static_cast<size_t>(image * 9 + k)] = g.cells[static_cast<size_t>(k)];
      }
    }
  }
  const long calls = surf.calls;
  double native_barrier = 0.0;
  {
    double highest = -1e300;
    double first = 0.0;
    std::vector<double> grad(static_cast<size_t>(3 * g.n_atoms));
    double stress[9];
    double energy = 0.0;
    for (int image = 0; image < g.n_images; ++image) {
      if (!eval_one(surf, g.n_atoms, xyz.data() + static_cast<size_t>(image * g.n_atoms * 3),
                    cells.data() + static_cast<size_t>(image * 9), energy, grad.data(),
                    stress)) {
        status = "final_eval_failed";
        break;
      }
      const double value =
          linear_h(energy, cells.data() + static_cast<size_t>(image * 9), cells.data(),
                   g.pressure);
      if (image == 0) {
        first = value;
      }
      highest = std::max(highest, value);
    }
    native_barrier = highest - first;
  }
  // The barrier evaluation is not part of the optimization.
  const long opt_calls = calls;

  write_geom(outdir + "/final.geom", g, xyz, cells);
  std::ofstream out(outdir + "/result.txt");
  out.setf(std::ios::scientific);
  out.precision(16);
  out << "converged " << converged << "\n"
      << "iterations " << (converged ? report.iteration : steps) << "\n"
      << "force_calls " << opt_calls << "\n"
      << "max_force " << report.max_force << "\n"
      << "status " << status << "\n"
      << "native_barrier " << native_barrier << "\n";
  rgsaddle_band_free(session);
  return 0;
}
