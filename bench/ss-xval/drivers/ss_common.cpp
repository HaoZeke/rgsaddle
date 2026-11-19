// Common-frame re-evaluation: one rgpot shifted-LJ build, exact H = E + pV.
#include "rgpot/ForceStructs.hpp"
#include "rgpot/LennardJones/LJPot.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
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
  if (g.n_images < 1 || g.n_atoms < 1) {
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

struct Eval {
  double energy = 0;
  double volume = 0;
  double enthalpy = 0;
  double max_force = 0;
  double max_stress = 0;
  double max_enthalpy_stress = 0;
  double max_cell_force = 0;
  bool ok = false;
};

Eval evaluate(rgpot::LJPot &pot, const std::vector<int> &z, int n, const double *x,
              const double *h, double pressure) {
  Eval out;
  std::vector<double> force(static_cast<size_t>(3 * n));
  rgpot::ForceOut raw{force.data(), 0.0, 0.0, {}, 0};
  rgpot::ForceInput in{static_cast<size_t>(n), x, z.data(), h};
  pot.forceImpl(in, &raw);
  if (raw.has_stress == 0) {
    return out;
  }
  out.energy = raw.energy;
  out.volume = std::abs(det3(h));
  out.enthalpy = raw.energy + pressure * out.volume;
  for (double f : force) {
    out.max_force = std::max(out.max_force, std::abs(f));
  }
  double inv[9];
  if (!inv3(h, inv)) {
    return out;
  }
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      const double sp = raw.stress[3 * i + j] + (i == j ? pressure : 0.0);
      out.max_enthalpy_stress = std::max(out.max_enthalpy_stress, std::abs(sp));
      out.max_stress = std::max(out.max_stress, std::abs(raw.stress[3 * i + j]));
    }
  }
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      double acc = 0.0;
      for (int k = 0; k < 3; ++k) {
        const double sk = raw.stress[3 * k + j] + (k == j ? pressure : 0.0);
        acc += inv[3 * k + i] * sk;
      }
      out.max_cell_force = std::max(out.max_cell_force, std::abs(out.volume * acc));
    }
  }
  out.ok = true;
  return out;
}

double finite_difference(rgpot::LJPot &pot, const std::vector<int> &z, int n, const double *x,
                         const double *h) {
  double h0[9];
  for (int i = 0; i < 9; ++i) {
    h0[i] = h[i];
  }
  double inv[9];
  if (!inv3(h0, inv)) {
    return -1.0;
  }
  std::vector<double> frac(static_cast<size_t>(3 * n));
  for (int a = 0; a < n; ++a) {
    for (int c = 0; c < 3; ++c) {
      frac[static_cast<size_t>(3 * a + c)] =
          x[3 * a] * inv[c] + x[3 * a + 1] * inv[3 + c] + x[3 * a + 2] * inv[6 + c];
    }
  }
  std::vector<double> force(static_cast<size_t>(3 * n));
  rgpot::ForceOut raw{force.data(), 0.0, 0.0, {}, 0};
  rgpot::ForceInput in{static_cast<size_t>(n), x, z.data(), h};
  pot.forceImpl(in, &raw);
  const double volume = std::abs(det3(h0));
  double worst = 0.0;
  const double eps = 1e-6;
  for (int ij = 0; ij < 9; ++ij) {
    auto energy_at = [&](double delta) {
      double hh[9];
      for (int k = 0; k < 9; ++k) {
        hh[k] = h0[k];
      }
      hh[ij] += delta;
      std::vector<double> xx(static_cast<size_t>(3 * n));
      for (int a = 0; a < n; ++a) {
        for (int c = 0; c < 3; ++c) {
          xx[static_cast<size_t>(3 * a + c)] = frac[static_cast<size_t>(3 * a)] * hh[c] +
                                              frac[static_cast<size_t>(3 * a + 1)] * hh[3 + c] +
                                              frac[static_cast<size_t>(3 * a + 2)] * hh[6 + c];
        }
      }
      return evaluate(pot, z, n, xx.data(), hh, 0.0).energy;
    };
    const double numeric = (energy_at(eps) - energy_at(-eps)) / (2.0 * eps);
    const int row = ij / 3;
    const int col = ij % 3;
    double predicted = 0.0;
    for (int k = 0; k < 3; ++k) {
      predicted += volume * inv[3 * k + row] * raw.stress[3 * k + col];
    }
    worst = std::max(worst, std::abs(predicted - numeric));
  }
  return worst;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: ss_common geom outdir\n";
    return 2;
  }
  Geom g;
  if (!read_geom(argv[1], g)) {
    std::cerr << "could not read geometry\n";
    return 2;
  }
  rgpot::LJConfig cfg;
  cfg.u0 = g.epsilon;
  cfg.psi = g.sigma;
  cfg.cutoff = g.cutoff;
  cfg.switch_width = 0.0;
  rgpot::LJPot pot(cfg);
  std::vector<int> z(static_cast<size_t>(g.n_atoms), 18);
  const double fd = finite_difference(pot, z, g.n_atoms, g.xyz.data(), g.cells.data());

  std::ofstream out(std::string(argv[2]) + "/common.txt");
  out.setf(std::ios::scientific);
  out.precision(16);
  out << "fd_cell_derivative_error " << fd << "\n";
  int saddle = 0;
  double h0 = 0.0;
  double hs = -1e300;
  double interior_max_atomic = 0.0;
  for (int image = 0; image < g.n_images; ++image) {
    const Eval ev = evaluate(pot, z, g.n_atoms,
                             g.xyz.data() + static_cast<size_t>(image * g.n_atoms * 3),
                             g.cells.data() + static_cast<size_t>(image * 9), g.pressure);
    if (!ev.ok) {
      std::cerr << "evaluation failed\n";
      return 3;
    }
    if (image == 0) {
      h0 = ev.enthalpy;
    }
    if (image > 0 && image + 1 < g.n_images) {
      interior_max_atomic = std::max(interior_max_atomic, ev.max_force);
      if (ev.enthalpy > hs) {
        hs = ev.enthalpy;
        saddle = image;
      }
    }
    const double limit = 1e-3 * std::cbrt(ev.volume);
    out << "image " << image << " energy " << ev.energy << " volume " << ev.volume << " enthalpy "
        << ev.enthalpy << " max_atomic " << ev.max_force << " max_stress " << ev.max_stress
        << " max_enthalpy_stress " << ev.max_enthalpy_stress << " stress_limit " << limit
        << " max_cell_force " << ev.max_cell_force << "\n";
  }
  if (g.n_images < 3) {
    hs = h0;
    saddle = 0;
  }
  const Eval saddle_ev =
      evaluate(pot, z, g.n_atoms, g.xyz.data() + static_cast<size_t>(saddle * g.n_atoms * 3),
               g.cells.data() + static_cast<size_t>(saddle * 9), g.pressure);
  const double limit = 1e-3 * std::cbrt(saddle_ev.volume);
  const int pass_force = saddle_ev.max_force <= 1e-3;
  const int pass_stress = saddle_ev.max_enthalpy_stress <= limit;
  out << "saddle_index " << saddle << "\n"
      << "barrier " << (hs - h0) << "\n"
      << "saddle_volume " << saddle_ev.volume << "\n"
      << "saddle_max_atomic " << saddle_ev.max_force << "\n"
      << "saddle_max_enthalpy_stress " << saddle_ev.max_enthalpy_stress << "\n"
      << "saddle_stress_limit " << limit << "\n"
      << "saddle_max_cell_force " << saddle_ev.max_cell_force << "\n"
      << "pass_force " << pass_force << "\n"
      << "pass_stress " << pass_stress << "\n"
      << "interior_max_atomic " << interior_max_atomic << "\n";
  return 0;
}
