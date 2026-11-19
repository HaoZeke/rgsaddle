// eOn band runner on rgpot shifted LJ or Tersoff.
#include "ss_geom.hpp"
#include "eon/Matter.h"
#include "eon/NudgedElasticBand.h"
#include "eon/Parameters.h"
#include "eon/potentials/RgpotAdapter/RgpotAdapter.h"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/fortran/FortranPots.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

double linear_h(double energy, const Matrix3d &h, const Matrix3d &h0, double pressure) {
  if (pressure == 0.0) {
    return energy;
  }
  const Matrix3d strain = h0.inverse() * (h - h0);
  return energy + pressure * strain.trace() * std::abs(h0.determinant());
}

// Interior images that need their own neighbor table hold a cloned potential.
// The parent's counter then sees only the two endpoints. The band total is the
// sum of forceCallCounter on the distinct potentials the images hold, with the
// parent's calls from before the band subtracted once.
long image_force_calls(eonc::NudgedElasticBand &band,
                       const std::shared_ptr<eonc::Potential> &root, size_t before) {
  std::unordered_set<const eonc::Potential *> seen;
  long total = 0;
  for (const auto &image : band.path) {
    if (!image) {
      continue;
    }
    const std::shared_ptr<eonc::Potential> pot = image->getPotential();
    if (!pot || !seen.insert(pot.get()).second) {
      continue;
    }
    long value = static_cast<long>(pot->forceCallCounter.load());
    if (pot.get() == root.get()) {
      value -= static_cast<long>(before);
    }
    if (value > 0) {
      total += value;
    }
  }
  return total;
}

void write_result(const std::string &outdir, int converged, long iterations,
                  long force_calls, double max_force, const std::string &status,
                  double native_barrier, const Geom &g, const std::vector<eonc::Matter> &images,
                  const WidthLog &widths) {
  if (!images.empty()) {
    std::ofstream geom(outdir + "/final.geom");
    geom.setf(std::ios::scientific);
    geom.precision(16);
    geom << "n_images " << g.n_images << "\nn_atoms " << g.n_atoms << "\npressure " << g.pressure
         << "\nepsilon " << g.epsilon << "\nsigma " << g.sigma << "\ncutoff " << g.cutoff
         << "\nsolid " << g.solid << "\npotential " << g.potential << "\nguard " << g.guard
         << "\nz " << g.z << "\n";
    for (const auto &image : images) {
      const Matrix3d cell = image.getCell();
      const AtomMatrix xyz = image.getPositions();
      geom << "image\n";
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
          geom << cell(r, c) << (c == 2 ? "\n" : " ");
        }
      }
      for (int a = 0; a < g.n_atoms; ++a) {
        geom << xyz(a, 0) << " " << xyz(a, 1) << " " << xyz(a, 2) << "\n";
      }
    }
  }
  std::ofstream out(outdir + "/result.txt");
  out.setf(std::ios::scientific);
  out.precision(16);
  out << "converged " << converged << "\n"
      << "iterations " << iterations << "\n"
      << "force_calls " << force_calls << "\n"
      << "force_count summed_distinct_image_potentials\n"
      << "max_force " << max_force << "\n"
      << "status " << status << "\n"
      << "native_barrier " << native_barrier << "\n"
      << "min_perp_width " << widths.min_width << "\n"
      << "guard " << g.guard << "\n"
      << "width_ok " << widths.ok << "\n";
  for (int image = 0; image < g.n_images && image < static_cast<int>(widths.per_image.size());
       ++image) {
    out << "image_min_width " << image << " " << widths.per_image[static_cast<size_t>(image)]
        << "\n";
  }
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 8) {
    std::cerr << "usage: ss_eon geom outdir optimizer spring ci tol max_iter\n";
    return 2;
  }
  Geom g;
  if (!read_geom(argv[1], g)) {
    std::cerr << "could not read geometry\n";
    return 2;
  }
  const std::string outdir = argv[2];
  const std::string optimizer = argv[3];
  const double spring = std::atof(argv[4]);
  const double ci = std::atof(argv[5]);
  const double tol = std::atof(argv[6]);
  const int max_iter = std::atoi(argv[7]);

  eonc::Parameters params;
  eonc::ParametersLoadAccess::main_options(params).parallel = false;
  auto &neb = eonc::ParametersLoadAccess::neb_options(params);
  neb.image_count = g.n_images - 2;
  neb.max_iterations = max_iter;
  neb.force_tolerance = tol;
  neb.spring.constant = spring;
  neb.climbing_image.enabled = true;
  neb.climbing_image.trigger_force = ci;
  neb.climbing_image.trigger_factor = 0.0;
  neb.solid_state.enabled = g.solid != 0;
  neb.solid_state.pressure = g.pressure;
  neb.solid_state.weight = 1.0;
  auto &opt = eonc::ParametersLoadAccess::optimizer_options(params);
  opt.max_move = 0.2;
  opt.max_iterations = static_cast<size_t>(max_iter);
  opt.converged_force = tol;
  if (optimizer == "fire") {
    neb.opt_method = eonc::OptType::FIRE;
    opt.method = eonc::OptType::FIRE;
    // Parameters() resolves time_step_input. Do not replace that value.
  } else if (optimizer == "lbfgs") {
    neb.opt_method = eonc::OptType::LBFGS;
    opt.method = eonc::OptType::LBFGS;
  } else {
    std::cerr << "optimizer must be fire or lbfgs\n";
    return 2;
  }

  if (g.potential == "tersoff" && g.solid) {
    std::cerr << "tersoff publishes no stress; the cell is not a degree of freedom\n";
    return 2;
  }
  WidthLog widths;
  if (!note_widths(g, g.cells.data(), widths)) {
    write_geom(outdir + "/final.geom", g, g.xyz, g.cells);
    write_result(outdir, 0, 0, 0, 0.0, "fail_cell_width", 0.0, g, {}, widths);
    return 0;
  }

  std::shared_ptr<eonc::Potential> potential;
  if (g.potential == "lj") {
    rgpot::LJConfig lj;
    lj.u0 = g.epsilon;
    lj.psi = g.sigma;
    lj.cutoff = g.cutoff;
    lj.switch_width = 0.0;
    potential = std::make_shared<RgpotAdapter<rgpot::LJPot>>(eonc::PotType::LJ, params, lj);
  } else if (g.potential == "tersoff") {
    potential =
        std::make_shared<RgpotAdapter<rgpot::fortranpots::TersoffPot>>(eonc::PotType::TERSOFF_SI,
                                                                        params);
  } else {
    std::cerr << "unknown potential\n";
    return 2;
  }
  const size_t calls_before = potential->forceCallCounter.load();

  std::vector<eonc::Matter> images;
  images.reserve(static_cast<size_t>(g.n_images));
  for (int image = 0; image < g.n_images; ++image) {
    eonc::Matter matter(potential, params);
    matter.resize(g.n_atoms);
    Matrix3d cell;
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        cell(r, c) = g.cells[static_cast<size_t>(image * 9 + 3 * r + c)];
      }
    }
    AtomMatrix xyz(g.n_atoms, 3);
    for (int a = 0; a < g.n_atoms; ++a) {
      matter.setAtomicNr(a, g.z);
      matter.setMass(a, 1.0);
      matter.setFixed(a, false);
      for (int c = 0; c < 3; ++c) {
        xyz(a, c) = g.xyz[static_cast<size_t>(image * g.n_atoms * 3 + 3 * a + c)];
      }
    }
    matter.setCell(cell);
    matter.setPeriodic(true);
    matter.setPositions(xyz);
    images.push_back(std::move(matter));
  }

  std::optional<eonc::NudgedElasticBand> band;
  try {
    band.emplace(std::move(images), params, potential);
    const auto state = band->compute();
    const long calls = image_force_calls(*band, potential, calls_before);
    int converged = state == eonc::NudgedElasticBand::NEBStatus::GOOD ? 1 : 0;
    std::string status = converged ? "converged" : "not_converged";
    double highest = -std::numeric_limits<double>::infinity();
    double first = 0.0;
    int interior = 0;
    std::vector<eonc::Matter> kept;
    std::vector<double> final_cells(static_cast<size_t>(g.n_images * 9), 0.0);
    kept.reserve(band->path.size());
    for (size_t i = 0; i < band->path.size(); ++i) {
      const Matrix3d cell = band->path[i]->getCell();
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
          final_cells[i * 9 + static_cast<size_t>(3 * r + c)] = cell(r, c);
        }
      }
      const double value = linear_h(band->path[i]->getPotentialEnergy(), cell,
                                    band->path.front()->getCell(), g.pressure);
      if (i == 0) {
        first = value;
      }
      if (i > 0 && i + 1 < band->path.size()) {
        highest = std::max(highest, value);
        interior += 1;
      }
      kept.push_back(*band->path[i]);
    }
    if (!note_widths(g, final_cells.data(), widths)) {
      converged = 0;
      status = "fail_cell_width";
    }
    const double barrier = interior > 0 ? highest - first : 0.0;
    write_result(outdir, converged, band->lastIteration(), calls, band->convergenceForce(), status,
                 barrier, g, kept, widths);
  } catch (const std::exception &err) {
    std::string message = err.what();
    for (char &ch : message) {
      if (ch == '\n' || ch == '\r') {
        ch = ' ';
      }
    }
    const long calls = band ? image_force_calls(*band, potential, calls_before)
                            : static_cast<long>(potential->forceCallCounter.load() - calls_before);
    write_result(outdir, 0, 0, calls,
                 std::numeric_limits<double>::quiet_NaN(), "abort " + message,
                 std::numeric_limits<double>::quiet_NaN(), g, {}, widths);
  }
  return 0;
}
