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

#include "se/voxel_implementations/MultiresTSDF/MultiresTSDF.hpp"

#include "se/utils/math_utils.h"
#include "se/voxel_block_ray_iterator.hpp"
#include <type_traits>



Eigen::Vector4f MultiresTSDF::raycast(
    const se::Octree<MultiresTSDF::VoxelType>& map,
    const Eigen::Vector3f&                     ray_origin_M,
    const Eigen::Vector3f&                     ray_dir_M,
    const float                                t_near,
    const float                                t_far,
    const float                                step,
    const float                                large_step) {

  se::VoxelBlockRayIterator<MultiresTSDF::VoxelType> ray(map, ray_origin_M, ray_dir_M,
      t_near, t_far);
  ray.next();
  const float t_min = ray.tmin(); /* Get distance to the first intersected block */
  if (t_min <= 0.f) {
    return Eigen::Vector4f::Zero();
  }
  const float t_max = ray.tmax();

  auto select_node_tsdf = [](const auto&){ return MultiresTSDF::VoxelType::initData().x; };
  auto select_voxel_tsdf = [](const auto& data){ return data.x; };
  // first walk with largesteps until we found a hit
  float t = t_min;
  float step_size = large_step;
  Eigen::Vector3f ray_pos_M = ray_origin_M + ray_dir_M * t;
  const int scale = 0;
  auto interp_res = map.interpAtPoint(ray_pos_M, select_node_tsdf, select_voxel_tsdf, scale);
  float f_t = interp_res.first;
  float f_tt = 0;
  if (f_t > 0) { // ups, if we were already in it, then don't render anything here
    for (; t < t_max; t += step_size) {
      auto data = map.getFineAtPoint(ray_pos_M, scale);
      if (data.y == 0) {
        step_size = large_step;
        ray_pos_M += step_size * ray_dir_M;
        continue;
      }
      f_tt = data.x;
      if (f_tt <= 0.1 && f_tt >= -0.5f) {
        interp_res = map.interpAtPoint(ray_pos_M, select_node_tsdf, select_voxel_tsdf, scale);
        f_tt = interp_res.first;
      }
      if (f_tt < 0.f)                  // got it, jump out of inner loop
        break;
      step_size = fmaxf(f_tt * MultiresTSDF::mu, step);
      ray_pos_M += step_size * ray_dir_M;
      f_t = f_tt;
    }
    if (f_tt < 0.f) {
      // got it, calculate accurate intersection
      t = t + step_size * f_tt / (f_t - f_tt);
      Eigen::Vector4f res = (ray_origin_M + ray_dir_M * t).homogeneous();
      res.w() = interp_res.second;
      return res;
    }
  }
  return Eigen::Vector4f::Constant(-1.f);
}

