// CP2K solid-state Ar regtest expressed through eOn and rgpot.
#include "eon/Matter.h"
#include "eon/NudgedElasticBand.h"
#include "eon/Parameters.h"
#include "eon/potentials/RgpotAdapter/RgpotAdapter.h"
#include "rgpot/LennardJones/LJPot.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

namespace {

constexpr double kPressure = 50.0 / 1.602176634e6;

double enthalpy(const eonc::Matter &image, const eonc::Matter &reference) {
  const Matrix3d h0 = reference.getCell();
  const Matrix3d strain = h0.inverse() * (image.getCell() - h0);
  return image.getPotentialEnergy() +
         kPressure * std::abs(h0.determinant()) * strain.trace();
}

eonc::Matter make_image(const std::shared_ptr<eonc::Potential> &potential,
                        const eonc::Parameters &params, const Matrix3d &cell,
                        const AtomMatrix &positions) {
  eonc::Matter image(potential, params);
  image.resize(2);
  image.setAtomicNr(0, 18);
  image.setAtomicNr(1, 18);
  image.setMass(0, 39.948);
  image.setMass(1, 39.948);
  image.setCell(cell);
  image.setPeriodic(true);
  image.setPositions(positions);
  return image;
}

} // namespace

int main() {
  eonc::Parameters params;
  eonc::ParametersLoadAccess::main_options(params).parallel = false;
  auto &neb = eonc::ParametersLoadAccess::neb_options(params);
  neb.image_count = 1;
  neb.max_iterations = 10000;
  neb.opt_method = eonc::OptType::FIRE;
  neb.force_tolerance = 1.0e-6;
  neb.spring.constant =
      0.02 * 27.211386245988 / (0.529177210903 * 0.529177210903);
  neb.climbing_image.enabled = true;
  neb.climbing_image.trigger_force = INFINITY;
  neb.climbing_image.trigger_factor = 0.0;
  neb.climbing_image.band_slack = 1.0;
  neb.solid_state.enabled = true;
  neb.solid_state.pressure = kPressure;
  neb.solid_state.weight = 1.0;
  auto &opt = eonc::ParametersLoadAccess::optimizer_options(params);
  opt.method = eonc::OptType::FIRE;
  opt.max_move = 0.05;
  opt.max_iterations = 10000;
  opt.converged_force = 1.0e-6;
  opt.convergence_metric = "max_component";

  rgpot::LJConfig lj;
  lj.u0 = 119.8 * 8.617333262145e-5;
  lj.psi = 3.405;
  lj.cutoff = 4.8;
  auto potential = std::make_shared<RgpotAdapter<rgpot::LJPot>>(
      eonc::PotType::LJ, params, lj);

  Matrix3d h0;
  h0 << 10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0;
  Matrix3d hm;
  hm << 10.246950765959598, 0.0, 0.0, 0.4, 9.848857801796104, 0.0,
      -0.15, 0.2, 10.099504938362077;
  Matrix3d h1;
  h1 << 10.5, 0.0, 0.0, 0.8, 9.7, 0.0, -0.3, 0.4, 10.2;
  AtomMatrix x0(2, 3);
  x0 << 2.0, 2.0, 2.0, 5.3, 3.0, 2.0;
  AtomMatrix xm(2, 3);
  xm << 2.050676919316477, 2.000022291778578, 2.000098036812882,
      5.506269079841432, 2.797069031408468, 2.000098036812882;
  AtomMatrix x1(2, 3);
  x1 << 2.1, 2.0, 2.0, 5.7, 2.6, 2.0;

  std::vector<eonc::Matter> images;
  images.push_back(make_image(potential, params, h0, x0));
  images.push_back(make_image(potential, params, hm, xm));
  images.push_back(make_image(potential, params, h1, x1));
  const double initial_energy = images.front().getPotentialEnergy();
  const double max_force = images.front().getForces().cwiseAbs().maxCoeff();
  const Matrix3d initial_stress = images.front().cauchyStress();
  std::cout.precision(15);
  std::cout << "initial energy_eV " << initial_energy
            << " max_force_eV_per_A " << max_force
            << " max_stress_eV_per_A3 "
            << initial_stress.cwiseAbs().maxCoeff() << '\n';
  std::cout << "initial stress_eV_per_A3 " << initial_stress(0, 0) << ' '
            << initial_stress(1, 1) << ' ' << initial_stress(2, 2) << ' '
            << initial_stress(0, 1) << ' ' << initial_stress(0, 2) << ' '
            << initial_stress(1, 2) << '\n';

  eonc::NudgedElasticBand band(std::move(images), params, potential);
  const auto status = band.compute();
  double maximum = -INFINITY;
  for (const auto &image : band.path) {
    maximum = std::max(maximum, enthalpy(*image, *band.path.front()));
  }
  const auto &middle = *band.path[1];
  std::cout << "band converged "
            << (status == eonc::NudgedElasticBand::NEBStatus::GOOD)
            << " iterations " << band.lastIteration() << " max_force "
            << band.convergenceForce() << " middle_volume_A3 "
            << std::abs(middle.getCell().determinant())
            << " middle_enthalpy_eV "
            << enthalpy(middle, *band.path.front())
            << " barrier_from_first_eV "
            << maximum - enthalpy(*band.path.front(), *band.path.front())
            << '\n';
  return status == eonc::NudgedElasticBand::NEBStatus::GOOD ? 0 : 1;
}
