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

#ifndef __MultiresOFusion_HPP
#define __MultiresOFusion_HPP

#include <se/octree.hpp>
#include <se/image/image.hpp>
#include <se/voxel_implementations/MultiresOFusion/kernel_image.hpp>
#include "se/sensor_implementation.hpp"

#include <yaml-cpp/yaml.h>
#include <chrono>
#include <ctime>

/**
 * Minimal example of the structure of a potential voxel implementation. All
 * functions and data members are required. The signature of the functions
 * should not be changed. Additional static functions or data members may be
 * added freely.
 */

enum class UncertaintyModel {linear, quadratic};

static std::map<std::string, UncertaintyModel> stringToModel {
    { "linear",     UncertaintyModel::linear},
    { "quadratic",  UncertaintyModel::quadratic}
};

static std::map<UncertaintyModel, std::string> modelToString {
    { UncertaintyModel::linear, "linear"},
    { UncertaintyModel::quadratic, "quadratic"}
};

struct MultiresOFusion {

  /**
   * The voxel type used as the template parameter for se::Octree.
   *
   * \warning The struct name must always be `VoxelType`.
   */
  struct VoxelType{
    struct VoxelData {
      VoxelData() {};
      VoxelData(float x, float x_last, float x_max, float y, float y_last, int frame, bool observed) :
            x(x), x_last(x_last), x_max(x_max), y(y), y_last(y_last), frame(frame), observed(observed) {};

      float  x;             // Latest mean occupancy
      float  x_last;        // Child mean at time of up-propagation
      float  x_max;         // Max occupancy of children
      float  y;             // Mean number of integrations
      float  y_last;        // Child mean number of integrations at time of up-propagation
      int    frame;         // Latest integration frame
      bool   observed;      // All children have been observed at least once

      // Any other data stored in each voxel go here. Make sure to also update
      // invalid() and initData() to initialize all data members.
    };

    static inline VoxelData invalid()  { return {0.f, 0.f, 0.f, 0.f, 0.f, 0, false}; }
    static inline VoxelData initData() { return {0.f, 0.f, 0.f, 0.f, 0.f, 0, false}; }

    template <typename T>
    using MemoryPoolType = se::MemoryPool<T>;
    template <typename ElemT>
    using MemoryBufferType = std::vector<ElemT>;
  };

  /**
   * Set to true for TSDF maps, false for occupancy maps.
   *
   * \warning The name of this variable must always be `invert_normals`.
   */
  static constexpr bool invert_normals = false;

  // Any other constant parameters required for the implementation go here.
  static float surface_boundary;

  /**
   * Stored occupancy probabilities in log-odds are clamped to never be lower
   * than this value.
   */
  static float max_occupancy;

  /**
   * Stored occupancy probabilities in log-odds are clamped to never be lower
   * than this value.
   */
  static float min_occupancy;

  static float max_weight;

  static int   fs_integr_scale; // Minimum integration scale for free-space

  static float factor;

  static float log_odd_max;

  static float log_odd_min;

  static bool const_surface_thickness;

  static float tau_min;

  static float tau_max;

  static float k_tau;

  static UncertaintyModel uncertainty_model;

  static float sigma_min;

  static float sigma_max;

  static float k_sigma;

  static std::string type() { return "multiresofusion"; }

  /**
   * Configure the MultiresOFusion parameters
   */
  static void configure(YAML::Node yaml_config) {
    configure();
    if (yaml_config.IsNull()) return;

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
  };

  static void configure() {
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

  static std::string print_config() {
    std::stringstream ss;
    ss << "========== VOXEL IMPL ========== " << "\n";
    ss << "Invert normals:                  " << (MultiresOFusion::invert_normals
                                                   ? "true" : "false") << "\n";
    ss << "Surface boundary:                " << MultiresOFusion::surface_boundary << "\n";
    ss << "Min occupancy:                   " << MultiresOFusion::min_occupancy << "\n";
    ss << "Max occupancy:                   " << MultiresOFusion::max_occupancy << "\n";
    ss << "Max weight:                      " << MultiresOFusion::max_weight << "\n";
    ss << "Free-space integration scale:    " << MultiresOFusion::fs_integr_scale << "\n";
    ss << "Log-odd min per integration:     " << MultiresOFusion::log_odd_min << "\n";
    ss << "Log-odd max per integration:     " << MultiresOFusion::log_odd_max << "\n";
    ss << "Const surface thickness:         " << (MultiresOFusion::const_surface_thickness
                                                   ? "true" : "false") << "\n";
    if (MultiresOFusion::const_surface_thickness) {
    ss << "tau:                             " << MultiresOFusion::tau_max << "\n";
    } else {
    ss << "tau min:                         " << MultiresOFusion::tau_min << "\n";
    ss << "tau max:                         " << MultiresOFusion::tau_max << "\n";
    ss << "k tau:                           " << MultiresOFusion::k_tau << "\n";
    }
    ss << "Uncertainty model:               " << modelToString.find(MultiresOFusion::uncertainty_model)->second << "\n";
    ss << "sigma min:                       " << MultiresOFusion::sigma_min << "\n";
    ss << "sigma max:                       " << MultiresOFusion::sigma_max << "\n";
    ss << "k sigma:                         " << MultiresOFusion::k_sigma << "\n";
    ss << "\n";
    return ss.str();
  }

  /**
   * Compute the VoxelBlocks and Nodes that need to be allocated given the
   * camera pose.
   */
  static size_t buildAllocationList(
      se::Octree<MultiresOFusion::VoxelType>& map,
      const se::Image<float>&                 depth_image,
      const Eigen::Matrix4f&                  T_MC,
      const SensorImpl&                       sensor,
      se::key_t*                              allocation_list,
      size_t                                  reserved);

  /**
   * Integrate a depth image into the map.
   *
   * \warning The function signature must not be changed.
   */
  static void integrate(se::Octree<MultiresOFusion::VoxelType>& map,
                        const se::Image<float>&                 depth_image,
                        const Eigen::Matrix4f&                  T_CM,
                        const SensorImpl&                       sensor,
                        const unsigned                          frame);

  static Eigen::Vector4f raycast(
      const se::Octree<MultiresOFusion::VoxelType>& map,
      const Eigen::Vector3f&                        ray_origin_M,
      const Eigen::Vector3f&                        ray_dir_M,
      float                                         t_near,
      float                                         t_far,
      float,
      float);
};

#endif // MultiresOFusion_HPP

