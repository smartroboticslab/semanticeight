/*
 Copyright 2016 Emanuele Vespa, Imperial College London

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.

 3. Neither the name of the copyright holder nor the names of its contributors
 may be used to endorse or promote products derived from this software without
 specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef __OFUSION_HPP
#define __OFUSION_HPP

#include "se/octree.hpp"
#include "se/image/image.hpp"
#include "se/sensor_implementation.hpp"

#include <yaml-cpp/yaml.h>

/**
 * Occupancy mapping voxel implementation.
 */
struct OFusion {

  /**
   * The voxel type used as the template parameter for se::Octree.
   */
  struct VoxelType {
    /**
     * The struct stored in each se::Octree voxel.
     */
    struct VoxelData {
      float  x; /**< The occupancy value in log-odds. */
      double y; /**< The timestamp of the last update. */
    };

    static inline VoxelData invalid()     { return {0.f, 0.f}; }
    static inline VoxelData initData() { return {0.f, 0.f}; }

    template <typename T>
    using MemoryPoolType = se::PagedMemoryPool<T>;
    template <typename ElemT>
    using MemoryBufferType = se::PagedMemoryBuffer<ElemT>;
  };



  /**
   * No need to invert the normals when rendering an occupancy map.
   */
  static constexpr bool invert_normals = false;

  /**
   * The value of the time constant tau in equation (10) from \cite
   * VespaRAL18.
   */
  static float tau;
  static constexpr float default_tau = 4;

  /**
   * Stored occupancy probabilities in log-odds are clamped to never be greater
   * than this value.
   */
  static float max_occupancy;
  static constexpr float default_max_occupancy = 1000;

  /**
   * Stored occupancy probabilities in log-odds are clamped to never be lower
   * than this value.
   */
  static float min_occupancy;
  static constexpr float default_min_occupancy = -1000;

  /**
   * The surface is considered to be where the log-odds occupancy probability
   * crosses this value.
   */
  static float surface_boundary;
  static constexpr float default_surface_boundary = 0.f;

  /**
   * Configure the OFusion parameters
   */
  static void configure(YAML::Node yaml_config) {
    surface_boundary  = (yaml_config["surface_boundary"])
        ? yaml_config["surface_boundary"].as<float>() : default_surface_boundary;
    tau               = (yaml_config["tau"])
        ? yaml_config["tau"].as<float>() : default_tau;
    if (yaml_config["occupancy_min_max"]) {
      std::vector<float> occupancy_min_max = yaml_config["occupancy_min_max"].as<std::vector<float>>();
      min_occupancy = occupancy_min_max[0];
      max_occupancy = occupancy_min_max[1];
    } else {
      min_occupancy = default_min_occupancy;
      max_occupancy = default_max_occupancy;
    }
  };

  /**
   * Configure the OFusion parameters
   */
  static void configure() {
    surface_boundary  = default_surface_boundary;
    tau               = default_tau;
    max_occupancy     = default_max_occupancy;
    min_occupancy     = default_min_occupancy;
  };

  static std::ostream& print_config(std::ostream& out) {
    out << "Invert normals:                  " << (OFusion::invert_normals
                                                   ? "true" : "false") << "\n";
    out << "Surface boundary:                " << OFusion::surface_boundary << "\n";
    out << "Tau:                             " << OFusion::tau << "\n";
    out << "Max occupancy:                   " << OFusion::max_occupancy << "\n";
    out << "Min occupancy:                   " << OFusion::min_occupancy << "\n";
    out << "\n";
    return out;
  }
  /**
   * Compute the VoxelBlocks and Nodes that need to be allocated given the
   * camera pose.
   */
  static size_t buildAllocationList(
      se::Octree<OFusion::VoxelType>& map,
      const se::Image<float>&         depth_image,
      const Eigen::Matrix4f&          T_MC,
      const SensorImpl&               sensor,
      se::key_t*                      allocation_list,
      size_t                          reserved);



  /**
   * Integrate a depth image into the map.
   */
  static void integrate(
      se::Octree<OFusion::VoxelType>& map,
      const se::Image<float>&         depth_image,
      const Eigen::Matrix4f&          T_CM,
      const SensorImpl&               sensor,
      const unsigned                  frame);



  /**
   * Cast a ray and return the point where the surface was hit.
   */
  static Eigen::Vector4f raycast(
      const se::Octree<OFusion::VoxelType>& map,
      const Eigen::Vector3f&                ray_origin_M,
      const Eigen::Vector3f&                ray_dir_M,
      const float                           t_near,
      const float                           t_far,
      const float                           mu,
      const float                           step,
      const float                           large_step);
};

#endif

