/*
 *
 * Copyright 2016 Emanuele Vespa, Imperial College London
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * */

#include "se/voxel_implementations/OFusion/OFusion.hpp"

#include "se/common.hpp"
#include "se/utils/math_utils.h"
#include "se/voxel_block_ray_iterator.hpp"
#include <type_traits>



Eigen::Vector4f OFusion::raycast(const OctreeType&      map,
                                 const Eigen::Vector3f& ray_origin_M,
                                 const Eigen::Vector3f& ray_dir_M,
                                 const float            t_near,
                                 const float            t_far) {
  se::VoxelBlockRayIterator<OFusion::VoxelType> ray(map, ray_origin_M, ray_dir_M,
                                                    t_near, t_far);
  ray.next();
  const float t_min = ray.tmin(); /* Get distance to the first intersected block */
  if (t_min <= 0.f) {
    return Eigen::Vector4f::Zero();
  }
  const float t_max = ray.tmax();
  float t = t_min;

  auto select_node_occupancy = [](const auto&){ return VoxelType::initData().x; };
  auto select_voxel_occupancy = [](const auto& data){ return data.x; };
  const float step_size = map.voxelDim() / 2;

  Eigen::Vector3f ray_pos_M = Eigen::Vector3f::Zero();

  float value_t  = 0;
  float value_tt = 0;
  Eigen::Vector3f point_M_t = Eigen::Vector3f::Zero();
  Eigen::Vector3f point_M_tt = Eigen::Vector3f::Zero();

  if (!find_valid_point(map, select_node_occupancy, select_voxel_occupancy,
                        ray_origin_M, ray_dir_M, step_size, t_far, t, value_t, point_M_t)) {
    return Eigen::Vector4f::Zero();
  }
  t += step_size;

  // if we are not already in it
  if (value_t <= OFusion::surface_boundary) {
    for (; t < t_max; t += step_size) {
      ray_pos_M = ray_origin_M + ray_dir_M * t;
      VoxelData data;
      map.getAtPoint(ray_pos_M, data);
      if (data.y == 0) {
        t += step_size;
        if (!find_valid_point(map, select_node_occupancy, select_voxel_occupancy,
                              ray_origin_M, ray_dir_M, step_size, t_far, t, value_t, point_M_t)) {
          return Eigen::Vector4f::Zero();
        }
        if (value_t > OFusion::surface_boundary) {
          break;
        }
        continue;
      }
      value_tt = data.x;
      point_M_tt = ray_pos_M;
      if (value_tt > -100.f) {
        bool is_valid = false;
        auto interp_res = map.interpAtPoint(ray_pos_M, select_node_occupancy, select_voxel_occupancy, 0, is_valid);
        value_tt = interp_res.first;
        if (!is_valid) {
          t += step_size;
          if (!find_valid_point(map, select_node_occupancy, select_voxel_occupancy,
                                ray_origin_M, ray_dir_M, step_size, t_far, t, value_t, point_M_t)) {
            return Eigen::Vector4f::Zero();
          }
          if (value_t > OFusion::surface_boundary) {
            break;
          }
          continue;
        }
      }
      if (value_tt > OFusion::surface_boundary) {
        break;
      }
      value_t = value_tt;
      point_M_t = point_M_tt;
    }
    if (value_tt > OFusion::surface_boundary && value_t < OFusion::surface_boundary) {
      // We overshot. Need to move backwards for zero crossing.
      t = t - (point_M_tt - point_M_t).norm() * (value_tt - OFusion::surface_boundary) / (value_tt - value_t);
      Eigen::Vector4f surface_point_M = (ray_origin_M + ray_dir_M * t).homogeneous();
      surface_point_M.w() = 0;
      return surface_point_M;
    }
  }
  return Eigen::Vector4f::Zero();
}