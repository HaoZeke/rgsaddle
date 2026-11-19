#pragma once

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

struct Geom {
  int n_images = 0;
  int n_atoms = 0;
  double pressure = 0;
  double epsilon = 1;
  double sigma = 1;
  double cutoff = 1;
  int solid = 0;
  int z = 18;
  double guard = 0;
  std::string potential = "lj";
  std::vector<double> cells;
  std::vector<double> xyz;
};

inline double det3(const double *h) {
  return h[0] * (h[4] * h[8] - h[5] * h[7]) - h[1] * (h[3] * h[8] - h[5] * h[6]) +
         h[2] * (h[3] * h[7] - h[4] * h[6]);
}

inline bool inv3(const double *h, double *o) {
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

inline double min_perp_width(const double *h) {
  const double *a = h;
  const double *b = h + 3;
  const double *c = h + 6;
  const double bc[3] = {b[1] * c[2] - b[2] * c[1], b[2] * c[0] - b[0] * c[2],
                        b[0] * c[1] - b[1] * c[0]};
  const double ca[3] = {c[1] * a[2] - c[2] * a[1], c[2] * a[0] - c[0] * a[2],
                        c[0] * a[1] - c[1] * a[0]};
  const double ab[3] = {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                        a[0] * b[1] - a[1] * b[0]};
  auto norm = [](const double *v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); };
  const double area_a = norm(bc);
  const double area_b = norm(ca);
  const double area_c = norm(ab);
  if (!(area_a > 0.0 && area_b > 0.0 && area_c > 0.0)) {
    return 0.0;
  }
  const double volume = std::abs(det3(h));
  return std::min(volume / area_a, std::min(volume / area_b, volume / area_c));
}

inline bool read_geom(const std::string &path, Geom &g) {
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
    } else if (key == "potential") {
      in >> g.potential;
    } else if (key == "guard") {
      in >> g.guard;
    } else if (key == "z") {
      in >> g.z;
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

inline void write_geom(const std::string &path, const Geom &g, const std::vector<double> &xyz,
                       const std::vector<double> &cells) {
  std::ofstream out(path);
  out.setf(std::ios::scientific);
  out.precision(16);
  out << "n_images " << g.n_images << "\nn_atoms " << g.n_atoms << "\npressure " << g.pressure
      << "\nepsilon " << g.epsilon << "\nsigma " << g.sigma << "\ncutoff " << g.cutoff
      << "\nsolid " << g.solid << "\npotential " << g.potential << "\nguard " << g.guard << "\nz "
      << g.z << "\n";
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

struct WidthLog {
  double min_width = 1e300;
  std::vector<double> per_image;
  int ok = 1;
};

inline bool note_widths(const Geom &g, const double *cells, WidthLog &log) {
  if (log.per_image.empty()) {
    log.per_image.assign(static_cast<size_t>(g.n_images), 1e300);
  }
  bool pass = true;
  for (int image = 0; image < g.n_images; ++image) {
    const double width = min_perp_width(cells + static_cast<size_t>(image * 9));
    log.min_width = std::min(log.min_width, width);
    log.per_image[static_cast<size_t>(image)] =
        std::min(log.per_image[static_cast<size_t>(image)], width);
    if (g.guard > 0.0 && width < g.guard) {
      pass = false;
    }
  }
  if (!pass) {
    log.ok = 0;
  }
  return pass;
}
