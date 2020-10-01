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

#include "se/octree.hpp"
#include "se/image/image.hpp"
#include "se/algorithms/meshing.hpp"
#include "se/voxel_implementations/MultiresOFusion/DensePoolingImage.hpp"
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
  struct VoxelType {
    struct VoxelData {
      VoxelData() {};
      VoxelData(float x, short y, bool observed) : x(x), y(y), observed(observed) {};

      float  x;             // Latest mean occupancy
      short  y;             // Mean number of integrations
      bool   observed;      // All children have been observed at least once

      bool operator==(const VoxelData& other) const;
      bool operator!=(const VoxelData& other) const;
    };

    static inline VoxelData invalid()  { return {0.f, 0, false}; }
    static inline VoxelData initData() { return {0.f, 0, false}; }

    static float selectNodeValue(const VoxelData& data) {
      return data.x;
    };

    static float selectVoxelValue(const VoxelData& data) {
      return data.x;
    };

    static float selectNodeWeight(const VoxelData& data) {
      return data.y;
    };

    static float selectVoxelWeight(const VoxelData& data) {
      return data.y;
    };

    static float selectSliceNodeValue(const VoxelData& data) {
      if (data.observed) {
        return data.x * data.y;
      } else {
        return 0.f;
      }
    };

    static float selectSliceVoxelValue(const VoxelData& data) {
      if (data.observed) {
        return data.x * data.y;
      } else {
        return 0.f;
      }
    };

    static bool isInside(const VoxelData& data) {
      return data.x > surface_boundary;
    };

    static bool isValid(const VoxelData& data) {
      return data.observed;
    };

    static float threshold(const VoxelData& data) {
      return data.x * data.y;
    }

    using VoxelBlockType = se::VoxelBlockSingleMax<MultiresOFusion::VoxelType>;

    using MemoryPoolType = se::MemoryPool<MultiresOFusion::VoxelType>;
    template <typename ElemT>
    using MemoryBufferType = std::vector<ElemT>;
  };

  using VoxelData      = MultiresOFusion::VoxelType::VoxelData;
  using OctreeType     = se::Octree<MultiresOFusion::VoxelType>;
  using VoxelBlockType = typename MultiresOFusion::VoxelType::VoxelBlockType;

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

  static short max_weight;

  static int   fs_integr_scale; // Minimum integration scale for free-space

  static float factor;

  static float log_odd_max;

  static float log_odd_min;

  static bool const_surface_thickness;

  static float tau_min_factor;

  static float tau_max_factor;

  static float tau_min;

  static float tau_max;

  static float k_tau;

  static UncertaintyModel uncertainty_model;

  static float sigma_min_factor;

  static float sigma_max_factor;

  static float sigma_min;

  static float sigma_max;

  static float k_sigma;

  static std::string type() { return "multiresofusion"; }

  /**
   * Configure the MultiresOFusion parameters
   */
  static void configure(const float voxel_dim);
  static void configure(YAML::Node yaml_config, const float voxel_dim);

  static std::string printConfig();

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
  static void integrate(OctreeType&             map,
                        const se::Image<float>& depth_image,
                        const Eigen::Matrix4f&  T_CM,
                        const SensorImpl&       sensor,
                        const unsigned          frame);

  static Eigen::Vector4f raycast(const OctreeType&      map,
                                 const Eigen::Vector3f& ray_origin_M,
                                 const Eigen::Vector3f& ray_dir_M,
                                 float,
                                 float                  t_far);

  static void dumpMesh(OctreeType&                map,
                       std::vector<se::Triangle>& mesh);

};

#endif // MultiresOFusion_HPP

