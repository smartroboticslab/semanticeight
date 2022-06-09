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

#include <type_traits>

#include "se/common.hpp"
#include "se/utils/math_utils.h"
#include "se/voxel_block_ray_iterator.hpp"
#include "se/voxel_implementations/MultiresTSDF/MultiresTSDF.hpp"



Eigen::Vector4f MultiresTSDF::raycast(const OctreeType& map,
                                      const Eigen::Vector3f& ray_origin_M,
                                      const Eigen::Vector3f& ray_dir_M,
                                      const float t_near,
                                      const float t_far)
{
    se::VoxelBlockRayIterator<VoxelType> ray(map, ray_origin_M, ray_dir_M, t_near, t_far);
    ray.next();
    const float t_min = ray.tmin(); /* Get distance to the first intersected block */
    if (t_min <= 0.f) {
        return Eigen::Vector4f::Zero();
    }
    const float t_max = ray.tmax();

    // first walk with largesteps until we found a hit
    float t = t_min;
    float step_size = MultiresTSDF::mu / 2;
    Eigen::Vector3f ray_pos_M = Eigen::Vector3f::Zero();

    float value_t = 0;
    float value_tt = 0;
    Eigen::Vector3f point_M_t = Eigen::Vector3f::Zero();
    Eigen::Vector3f point_M_tt = Eigen::Vector3f::Zero();
    int scale_tt = 0;

    if (!find_valid_point(map,
                          VoxelType::selectNodeValue,
                          VoxelType::selectVoxelValue,
                          ray_origin_M,
                          ray_dir_M,
                          step_size,
                          t_max,
                          t,
                          value_t,
                          point_M_t)) {
        return Eigen::Vector4f::Zero();
    }
    step_size =
        se::math::clamp(value_t * MultiresTSDF::mu, MultiresTSDF::mu / 10, MultiresTSDF::mu / 2);
    t += step_size;

    if (value_t > 0) { // ups, if we were already in it, then don't render anything here
        for (; t < t_max; t += step_size) {
            ray_pos_M = ray_origin_M + ray_dir_M * t;
            VoxelData data;
            map.getAtPoint(ray_pos_M, data);
            if (data.y == 0) {
                t += step_size;
                if (!find_valid_point(map,
                                      VoxelType::selectNodeValue,
                                      VoxelType::selectVoxelValue,
                                      ray_origin_M,
                                      ray_dir_M,
                                      step_size,
                                      t_max,
                                      t,
                                      value_t,
                                      point_M_t)) {
                    return Eigen::Vector4f::Zero();
                }
                if (value_t < 0) {
                    break;
                }
                continue;
            }
            value_tt = data.getTsdf();
            point_M_tt = ray_pos_M;
            if (value_tt <= 0.1) {
                bool is_valid = false;
                auto interp_res = map.interpAtPoint(ray_pos_M,
                                                    VoxelType::selectNodeValue,
                                                    VoxelType::selectVoxelValue,
                                                    0,
                                                    is_valid);
                value_tt = interp_res.first;
                scale_tt = interp_res.second;
                if (!is_valid) {
                    t += step_size;
                    if (!find_valid_point(map,
                                          VoxelType::selectNodeValue,
                                          VoxelType::selectVoxelValue,
                                          ray_origin_M,
                                          ray_dir_M,
                                          step_size,
                                          t_max,
                                          t,
                                          value_t,
                                          point_M_t)) {
                        return Eigen::Vector4f::Zero();
                    }
                    if (value_t < 0) {
                        break;
                    }
                    continue;
                }
            }
            if (value_tt < 0) {
                break; // got it, jump out of inner loop
            }
            step_size = se::math::clamp(
                value_tt * MultiresTSDF::mu, MultiresTSDF::mu / 10, MultiresTSDF::mu / 2);
            value_t = value_tt;
            point_M_t = point_M_tt;
        }
        if (value_tt < 0 && value_t > 0) {
            // We overshot. Need to move backwards for zero crossing.
            t = t
                - (point_M_tt - point_M_t).norm() / (value_tt - value_t)
                    * value_tt; // (value_tt - 0)
            Eigen::Vector4f surface_point_M = (ray_origin_M + ray_dir_M * t).homogeneous();
            surface_point_M.w() = scale_tt;
            return surface_point_M;
        }
    }
    return Eigen::Vector4f::Constant(-1.f);
}



enum class VoxelState {
    Inside,
    Outside,
    Unknown,
};

Eigen::Vector4f MultiresTSDF::raycastBackFace(const OctreeType& map,
                                              const Eigen::Vector3f& ray_origin_M,
                                              const Eigen::Vector3f& ray_dir_M,
                                              const float t_near,
                                              const float t_far)
{
    se::VoxelBlockRayIterator<VoxelType> ray(map, ray_origin_M, ray_dir_M, t_near, t_far);
    ray.next();
    const float t_min = ray.tmin(); /* Get distance to the first intersected block */
    if (t_min <= 0.f) {
        return Eigen::Vector4f::Zero();
    }
    const float t_max = ray.tmax();

    // first walk with largesteps until we found a hit
    float t = t_min;
    float step_size = MultiresTSDF::mu / 2;
    Eigen::Vector3f ray_pos_M = Eigen::Vector3f::Zero();

    float value_t = 0;
    Eigen::Vector3f point_M_t = Eigen::Vector3f::Zero();
    VoxelState state_t = VoxelState::Unknown;
    VoxelState state_tt = VoxelState::Unknown;


    if (!find_valid_point(map,
                          VoxelType::selectNodeValue,
                          VoxelType::selectVoxelValue,
                          ray_origin_M,
                          ray_dir_M,
                          step_size,
                          t_max,
                          t,
                          value_t,
                          point_M_t)) {
        return Eigen::Vector4f::Zero();
    }
    state_t = (value_t <= 0.0f) ? VoxelState::Inside : VoxelState::Outside;
    step_size =
        se::math::clamp(value_t * MultiresTSDF::mu, MultiresTSDF::mu / 10, MultiresTSDF::mu / 2);
    t += step_size;

    for (; t < t_max; t += step_size) {
        ray_pos_M = ray_origin_M + ray_dir_M * t;
        VoxelData data;
        map.getAtPoint(ray_pos_M, data);
        if (!MultiresTSDF::VoxelType::isValid(data)) {
            state_tt = VoxelState::Unknown;
        }
        else if (MultiresTSDF::VoxelType::isInside(data)) {
            state_tt = VoxelState::Inside;
        }
        else {
            state_tt = VoxelState::Outside;
        }
        // All the possible state combinations between the two hit voxels:
        // inside  -> outside    back face hit
        //
        // outside -> inside     skip ray
        //
        // inside  -> inside     advance ray
        // outside -> outside    advance ray
        // unknown -> unknown    advance ray
        // unknown -> outside    advance ray
        // outside -> unknown    advance ray
        // unknown -> inside     advance ray
        // inside  -> unknown    advance ray
        if (state_t == VoxelState::Inside && state_tt == VoxelState::Outside) {
            // Hit a back-face.
            return Eigen::Vector4f(point_M_t.homogeneous());
        }
        else if (state_t == VoxelState::Outside && state_tt == VoxelState::Inside) {
            // Hit something other than a back-face.
            return Eigen::Vector4f::Zero();
        }
        state_tt = state_t;
    }
    // No back-face found.
    return Eigen::Vector4f::Zero();
}
