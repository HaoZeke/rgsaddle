// Shifted Lennard-Jones fixed-cell vacancy-hop input driver.
#include "eon/Matter.h"
#include "eon/NudgedElasticBand.h"
#include "eon/potentials/RgpotAdapter/RgpotAdapter.h"
#include "rgpot/LennardJones/LJPot.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>

int main() {
  Parameters params;
  params.main_options.parallel = false;
  params.main_options.removeNetForce = true;
  params.potential_options.potential = PotType::LJ;
  params.neb_options.image_count = 5;
  params.neb_options.max_iterations = 500;
  params.neb_options.opt_method = OptType::LBFGS;
  params.neb_options.force_tolerance = 1.0e-3;
  params.neb_options.spring.constant = 5.0;
  params.neb_options.climbing_image.enabled = true;
  params.neb_options.climbing_image.converged_only = false;
  params.neb_options.climbing_image.trigger_force =
      std::numeric_limits<double>::infinity();
  params.neb_options.climbing_image.trigger_factor = 0.0;
  params.optimizer_options.method = OptType::LBFGS;
  params.optimizer_options.max_iterations = 500;
  params.optimizer_options.max_move = 0.05;
  params.optimizer_options.converged_force = 1.0e-3;

  rgpot::LJConfig lj;
  lj.u0 = 1.0;
  lj.psi = 1.0;
  lj.cutoff = 1.5;
  auto potential =
      std::make_shared<RgpotAdapter<rgpot::LJPot>>(PotType::LJ, params, lj);

  const double a = std::pow(2.0, 2.0 / 3.0);
  const double basis[4][3] = {{0.0, 0.0, 0.0},
                              {0.0, 0.5, 0.5},
                              {0.5, 0.0, 0.5},
                              {0.5, 0.5, 0.0}};
  AtomMatrix positions(31, 3);
  long atom = 0;
  long moving = -1;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {
        for (int b = 0; b < 4; ++b) {
          if (i == 0 && j == 0 && k == 0 && b == 0) {
            continue;
          }
          positions(atom, 0) = (i + basis[b][0]) * a;
          positions(atom, 1) = (j + basis[b][1]) * a;
          positions(atom, 2) = (k + basis[b][2]) * a;
          if (i == 0 && j == 0 && k == 0 && b == 1) {
            moving = atom;
          }
          ++atom;
        }
      }
    }
  }
  Matrix3d cell = Matrix3d::Identity() * (2.0 * a);
  auto make_matter = [&](const AtomMatrix &xyz) {
    auto matter = std::make_shared<Matter>(potential, params);
    matter->resize(31);
    matter->setCell(cell);
    matter->setPeriodic(true);
    matter->setPositions(xyz);
    matter->setAtomicNrs(VectorXi::Ones(31));
    matter->setMasses(VectorXd::Ones(31));
    for (long i = 0; i < 31; ++i) {
      matter->setFixed(i, 0);
    }
    return matter;
  };

  auto initial = make_matter(positions);
  AtomMatrix product_positions = positions;
  product_positions.row(moving).setZero();
  auto product = make_matter(product_positions);

  const double initial_energy = initial->getPotentialEnergy();
  const double initial_force = initial->getForces().cwiseAbs().maxCoeff();
  std::cout.precision(15);
  std::cout << "vacancy initial_energy_eV " << initial_energy
            << " max_force_eV_per_A " << initial_force
            << " stress unavailable\n";

  NudgedElasticBand band(initial, product, params, potential);
  const auto status = band.compute();
  double maximum = -std::numeric_limits<double>::infinity();
  for (const auto &image : band.path) {
    maximum = std::max(maximum, image->getPotentialEnergy());
  }
  std::cout << "vacancy converged "
            << (status == NudgedElasticBand::NEBStatus::GOOD)
            << " status " << static_cast<int>(status)
            << " max_force " << band.convergenceForce()
            << " barrier_eV " << maximum - initial_energy << '\n';
  std::cout << "fcc_variants variable_cell_supported 0 stress_supported 0\n";
  return status == NudgedElasticBand::NEBStatus::GOOD ? 0 : 1;
}
