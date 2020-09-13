/*
 * Copyright 2019 Nils Funk, Imperial College London
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "se/voxel_implementations/MultiresOFusion/MultiresOFusion.hpp"
#include "se/str_utils.hpp"



bool MultiresOFusion::VoxelType::VoxelData::operator==(const MultiresOFusion::VoxelType::VoxelData& other) const {
  return (x == other.x) && (x_last == other.x_last) && (x_max == other.x_max)
      && (y == other.y) && (y_last == other.y_last)
      && (frame == other.frame) && (observed == other.observed);
}

bool MultiresOFusion::VoxelType::VoxelData::operator!=(const MultiresOFusion::VoxelType::VoxelData& other) const {
  return !(*this == other);
}

// Initialize static data members.
constexpr bool   MultiresOFusion::invert_normals;
float            MultiresOFusion::surface_boundary;
float            MultiresOFusion::min_occupancy;
float            MultiresOFusion::max_occupancy;
float            MultiresOFusion::max_weight;
int              MultiresOFusion::fs_integr_scale;
float            MultiresOFusion::factor;
float            MultiresOFusion::log_odd_min;
float            MultiresOFusion::log_odd_max;
bool             MultiresOFusion::const_surface_thickness;
float            MultiresOFusion::tau_min;
float            MultiresOFusion::tau_max;
float            MultiresOFusion::k_tau;
UncertaintyModel MultiresOFusion::uncertainty_model;
float            MultiresOFusion::sigma_min;
float            MultiresOFusion::sigma_max;
float            MultiresOFusion::k_sigma;

void MultiresOFusion::configure(const float voxel_dim) {
  surface_boundary  = 0;
  min_occupancy     = -50;
  max_occupancy     = 50;
  max_weight        = 100;
  fs_integr_scale   = 0; // Minimum integration scale for free-space
  log_odd_min       = -5.015;
  log_odd_max       = 5.015;
  const_surface_thickness = false;
  tau_min           = 0.06f;
  tau_max           = 0.16f;
  k_tau             = 0.052;
  uncertainty_model = UncertaintyModel::linear;
  sigma_min         = 0.02;
  sigma_max         = 0.045;
  k_sigma           = 0.0016;
  factor = (max_weight - 1) / max_weight;
}

void MultiresOFusion::configure(YAML::Node yaml_config, const float voxel_dim) {
  configure(voxel_dim);
  if (yaml_config.IsNull()) {
    return;
  }

  if (yaml_config["surface_boundary"]) {
    surface_boundary  = yaml_config["surface_boundary"].as<float>();
  }
  if (yaml_config["occupancy_min_max"]) {
    std::vector<float> occupancy_min_max = yaml_config["occupancy_min_max"].as<std::vector<float>>();
    min_occupancy = occupancy_min_max[0];
    max_occupancy = occupancy_min_max[1];
  }
  if (yaml_config["max_weight"]) {
    max_weight = yaml_config["max_weight"].as<float>();
  }
  if ((yaml_config["free_space_integr_scale"])) {
    fs_integr_scale = yaml_config["free_space_integr_scale"].as<int>(); // Minimum integration scale for free-space
  }
  if (yaml_config["log_odd_min_max"]) {
    std::vector<float> log_odd_min_max = yaml_config["log_odd_min_max"].as<std::vector<float>>();
    log_odd_min = log_odd_min_max[0];
    log_odd_max = log_odd_min_max[1];
  }
  if (yaml_config["const_surface_thickness"]) {
    const_surface_thickness = yaml_config["const_surface_thickness"].as<bool>();
  }
  if (yaml_config["tau_min_max"]) {
    std::vector<float> tau_min_max = yaml_config["tau_min_max"].as<std::vector<float>>();
    tau_min = tau_min_max[0];
    tau_max = tau_min_max[1];
  }
  if (yaml_config["uncertainty_model"]) {
    stringToModel.find(yaml_config["uncertainty_model"].as<std::string>())->second;
  }
  if (yaml_config["k_tau"]) {
    k_tau = yaml_config["k_tau"].as<float>();
  }
  if (yaml_config["sigma_min_max"]) {
    std::vector<float> sigma_min_max = yaml_config["sigma_min_max"].as<std::vector<float>>();
    sigma_min = sigma_min_max[0];
    sigma_max = sigma_min_max[1];
  }
  if (yaml_config["k_sigma"]) {
    k_sigma = yaml_config["k_sigma"].as<float>();
  }
  factor = (max_weight - 1) / max_weight;
}

std::string MultiresOFusion::printConfig() {

  std::stringstream out;
  out << str_utils::header_to_pretty_str("VOXEL IMPL") << "\n";
  out << str_utils::bool_to_pretty_str(MultiresOFusion::invert_normals,          "Invert normals") << "\n";
  out << str_utils::value_to_pretty_str(MultiresOFusion::surface_boundary,       "Surface boundary") << "\n";
  out << str_utils::value_to_pretty_str(MultiresOFusion::min_occupancy,          "Min occupancy") << "\n";
  out << str_utils::value_to_pretty_str(MultiresOFusion::max_occupancy,          "Max occupancy") << "\n";
  out << str_utils::value_to_pretty_str(MultiresOFusion::max_weight,             "Max weight") << "\n";
  out << str_utils::value_to_pretty_str(MultiresOFusion::fs_integr_scale,        "Free-space integration scale") << "\n";
  out << str_utils::value_to_pretty_str(MultiresOFusion::log_odd_min,            "Log-odd min per integration") << "\n";
  out << str_utils::value_to_pretty_str(MultiresOFusion::log_odd_max,            "Log-odd max per integration") << "\n";
  out << str_utils::bool_to_pretty_str(MultiresOFusion::const_surface_thickness, "Const surface thickness") << "\n";
  if (MultiresOFusion::const_surface_thickness) {
    out << str_utils::value_to_pretty_str(MultiresOFusion::tau_max,              "tau") << "\n";
  } else {
    out << str_utils::value_to_pretty_str(MultiresOFusion::tau_min,              "tau min") << "\n";
    out << str_utils::value_to_pretty_str(MultiresOFusion::tau_max,              "tau max") << "\n";
    out << str_utils::value_to_pretty_str(MultiresOFusion::k_tau,                "k tau") << "\n";
  }

  out << str_utils::str_to_pretty_str(modelToString.find(MultiresOFusion::uncertainty_model)->second,
                                                                         "Uncertainty model") << "\n";
  out << str_utils::value_to_pretty_str(MultiresOFusion::sigma_min,              "sigma min") << "\n";
  out << str_utils::value_to_pretty_str(MultiresOFusion::sigma_max,              "sigma max") << "\n";
  out << str_utils::value_to_pretty_str(MultiresOFusion::k_sigma,                "k sigma") << "\n";
  out << "\n";
  return out.str();
}

// Implement static member functions.
size_t MultiresOFusion::buildAllocationList(OctreeType&,
                                            const se::Image<float>&,
                                            const Eigen::Matrix4f&,
                                            const SensorImpl&,
                                            se::key_t*,
                                            size_t) {
  return 0;
}

void MultiresOFusion::dumpMesh(OctreeType&                map,
                               std::vector<se::Triangle>& mesh) {
  auto inside = [](const VoxelData& data) {
    return data.x > MultiresOFusion::surface_boundary;
  };

  auto select_value = [](const VoxelData& data) {
    return data.x;
  };

  se::algorithms::dual_marching_cube(map, select_value, inside, mesh);
}
