/*
 * Copyright 2019 Sotiris Papatheodorou, Imperial College London
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
#include <se/continuous/volume_template.hpp>
#include <se/image/image.hpp>
#include <se/voxel_implementations/MultiresOFusion/kernel_image.hpp>

#include <sophus/se3.hpp>
#include <chrono>
#include <ctime>

/**
 * Minimal example of the structure of a potential voxel implementation. All
 * functions and data members are required. The signature of the functions
 * should not be changed. Additional static functions or data members may be
 * added freely.
 */
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
      // empty() and initValue() to initialize all data members.
    };

//    static inline VoxelData empty()     { return {min_occupancy, min_occupancy, min_occupancy, 1.f, true}; }
    static inline VoxelData empty()     { return {0.f, 0.f, 0.f, 0.f, 0.f, 0, false}; }
    static inline VoxelData initValue() { return {0.f, 0.f, 0.f, 0.f, 0.f, 0, false}; }

    template <typename ValueT>
    using MemoryPoolType = se::DummyMemoryPool<ValueT>;
    template <typename BufferT>
    using MemoryBufferType = std::vector<BufferT>;

//    template <typename ValueT>
//    using MemoryPoolType = se::MemoryPool<ValueT>;
//    template <typename BufferT>
//    using MemoryBufferType = se::MemoryBuffer<BufferT>;
  };

  /**
   * Set to true for TSDF maps, false for occupancy maps.
   *
   * \warning The name of this variable must always be `invert_normals`.
   */

  static constexpr bool invert_normals = false;

  // Any other constant parameters required for the implementation go here.
  static constexpr float surface_boundary = 0.f;

  /**
   * Stored occupancy probabilities in log-odds are clamped to never be lower
   * than this value.
   */
  static constexpr float max_occupancy = 50.f;

  /**
   * Stored occupancy probabilities in log-odds are clamped to never be lower
   * than this value.
   */
  static constexpr float min_occupancy = -50.f;

  static constexpr float max_weight = 10.f;

  static constexpr int   fs_integr_scale = 0; // Minimum integration scale for free-space

  /**
   * Compute the VoxelBlocks and Nodes that need to be allocated given the
   * camera pose.
   */
  static size_t buildAllocationList(
      se::key_t*                            allocation_list,
      size_t                                reserved,
      se::Octree<MultiresOFusion::VoxelType>& map,
      const Eigen::Matrix4f&                T_wc,
      const Eigen::Matrix4f&                K,
      const float*                          depth_image,
      const Eigen::Vector2i&                image_size,
      const float                           mu);

  /**
   * Integrate a depth image into the map.
   *
   * \warning The function signature must not be changed.
   */
  static void integrate(se::Octree<MultiresOFusion::VoxelType>& map,
                               const Sophus::SE3f&            T_cw,
                               const Eigen::Matrix4f&         K,
                               const se::Image<float>&        depth,
                               const float                    mu,
                               const unsigned                 frame);



//  /**
//   * Cast a ray and return the point where the surface was hit.
//   *
//   * \warning The function signature must not be changed.
//   */
//  static Eigen::Vector4f raycast(
//      const VolumeTemplate<MultiresOFusion, se::Octree>& volume,
//      const Eigen::Vector3f&                           origin,
//      const Eigen::Vector3f&                           direction,
//      const float                                      tnear,
//      const float                                      tfar,
//      const float                                      mu,
//      const float                                      step,
//      const float                                      large_step);


  static Eigen::Vector4f raycast(
      const VolumeTemplate<MultiresOFusion, se::Octree>& volume,
      const Eigen::Vector3f&                           origin,
      const Eigen::Vector3f&                           direction,
      float                                            tnear,
      float                                            tfar,
      float,
      float,
      float);
};

#endif // MultiresOFusion_HPP
