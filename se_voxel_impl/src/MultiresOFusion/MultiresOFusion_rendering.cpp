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

#include <se/utils/math_utils.h>
#include <type_traits>



/*!
 * \brief Compute the distance t in [m] travelled along the ray from the origin until the ray intersecs with the map.
 * Can be summariesed in 3 cases:
 * 1. Origin is inside the map                                      -> Return 0/valid
 * 2. Origin is outside the map and ray intersects the map          -> Return t/valid
 * 3. Origin is outside the map and ray will not intersect the map  -> Return 0/invalid
 *
 * \param curr_pos  current camera position in [m]
 * \param direction direction of the ray
 * \param map_dim   map dimension in [m]
 * \param t_far     maximum travel distance along the ray in [m]
 * \param is_valid  flag if the map intersection is valid
 * \return see above
 */
float computeMapIntersection( const Eigen::Vector3f& curr_pos, const Eigen::Vector3f& direction,
                              const int map_dim, float& t_far, bool& is_valid) {
  /*
  Fast Ray-Box Intersection
  by Andrew Woo
  from "Graphics Gems", Academic Press, 1990
  */
  const int map_min = 0;
  const int map_max = map_dim;
  int num_dim = 3;
  Eigen::Vector3f hit_point = -1*Eigen::Vector3f::Ones();				/* hit point */
  bool inside = true;
  Eigen::Vector3i quadrant;
  int which_plane;
  Eigen::Vector3f max_T;
  Eigen::Vector3f candidate_plane;

  /* Find candidate planes; this loop can be avoided if
     rays cast all from the eye(assume perpsective view) */
  for (int i = 0; i < num_dim; i++)
    if(curr_pos[i] < map_min) {
      quadrant[i] = 1; // LEFT := 1
      candidate_plane[i] = map_min;
      inside = false;
    }else if (curr_pos[i] > map_max) {
      quadrant[i] = 0; // RIGHT := 0
      candidate_plane[i] = map_max;
      inside = false;
    }else	{
      quadrant[i] = 2; // MIDDLE := 2
    }

  /* Ray origin inside bounding box */
  if(inside)	{
    return 0;
  }

  /* Calculate T distances to candidate planes */
  for (int i = 0; i < num_dim; i++) {
    if (quadrant[i] != 2 && direction[i] !=0.) // MIDDLE := 2
    {
      max_T[i] = (candidate_plane[i] - curr_pos[i]) / direction[i];
    }
    else {
      max_T[i] = -1.;
    }
  }

  /* Get largest of the max_T's for final choice of intersection */
  which_plane = 0;
  for (int i = 1; i < num_dim; i++)
    if (max_T[which_plane] < max_T[i])
      which_plane = i;

  /* Check final candidate actually inside box */
  if (max_T[which_plane] < 0.f) {
    is_valid = false;
    return 0;
  }
  for (int i = 0; i < num_dim; i++) {
    if (which_plane != i) {
      hit_point[i] = curr_pos[i] + max_T[which_plane] * direction[i];
      if (hit_point[i] < map_min || hit_point[i] > map_max) {
        is_valid = false;
        return 0;
      }
    } else {
      hit_point[i] = candidate_plane[i];
    }
  }

  float t = (hit_point - curr_pos).norm();
  if (t_far < t) {
    is_valid = false;
    return 0;
  }
  return t;
}

void advanceRay(const se::Octree<MultiresOFusion::VoxelType>* const map, const Eigen::Vector3f& origin_pos,
                            const Eigen::Vector3f& direction, float& t, const float& t_near, float& t_far, const float& map_res,
                            int max_scale, bool& is_valid) {
//  Eigen::IOFormat CleanFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "");
  int scale = max_scale;  // Initialize scale
  // Additional distance travelled in [voxel]
  float v_add  = 0;                     // TODO: I'll have to re-evaluate this value.
  float v      = 1 / map_res * t;       // t in voxel coordinates
  float v_near = 1 / map_res * t_near;  // t_near in voxel coordinates
  float v_far  = 1 / map_res * t_far;   // t_far in voxel coordinates
  Eigen::Vector3f origin_vox = 1 / map_res * origin_pos; // Origin in voxel coordinates

  // Current state of V in [voxel]
  Eigen::Vector3f V_max = Eigen::Vector3f::Ones();


  Eigen::Vector3f delta_V_map = map->size() / direction.array().abs(); // [voxel]/[-], potentionally dividing by 0

  Eigen::Vector3f map_frac  = origin_vox / map->size();
  // V at which the map boundary gets crossed (separate V for each dimension x-y-z)
  Eigen::Vector3f v_map;
  if(direction.x() < 0) {
    v_map.x() = map_frac.x() * delta_V_map.x();
  } else {
    v_map.x() = (1 - map_frac.x()) * delta_V_map.x();
  } if(direction.y() < 0) {
    v_map.y() = map_frac.y() * delta_V_map.y();
  } else {
    v_map.y() = (1 - map_frac.y()) * delta_V_map.y();
  } if(direction.z() < 0) {
    v_map.z() = map_frac.z() * delta_V_map.z();
  } else {
    v_map.z() = (1 - map_frac.z()) * delta_V_map.z();
  }

  // Maximum valid travelled distance in voxel is the minimum out of the far plane,
  // and the smallest distance that will make the ray cross the map boundary in either x, y or z direction.
  v_far = std::min(std::min(std::min(v_map.x(), v_map.y()), v_map.z()) + v, v_far); // [voxel]
  t_far = map_res * v_far;                                                          // [m]

  auto value = map->get_fine(origin_vox.x(), origin_vox.y(), origin_vox.z(), max_scale);
  while (value.x_max > -0.2f && scale > 2) {
    scale -= 1;
    value = map->get_fine(origin_vox.x(), origin_vox.y(), origin_vox.z(), scale);
  }

  Eigen::Vector3f curr_vox = origin_vox;

  auto start = std::chrono::system_clock::now();
  int iter_1 = 0;
  while ((v + v_add) < v_far) {
    if (scale <= 2) {
      t = map_res * (v + v_add - 4);
      return;
    }

    const int edge = 1 << scale;
    Eigen::Vector3i curr_node = edge*(((curr_vox).array().floor())/edge).cast<int>();


    // Fraction of the current position in [voxel] in the current node along the x-, y- and z-axis
    Eigen::Vector3f node_frac = (curr_vox - curr_node.cast<float>()) / edge;

    // Travelled distance needed in [voxel] to the whole edge in x, y and z direction
    Eigen::Vector3f delta_V = edge / direction.array().abs(); // [voxel]/[-]

    // Initalize V
    if(direction.x() < 0) {
      V_max.x() = node_frac.x() * delta_V.x();
    } else {
      V_max.x() = (1 - node_frac.x()) * delta_V.x();
    } if(direction.y() < 0) {
      V_max.y() = node_frac.y() * delta_V.y();
    } else {
      V_max.y() = (1 - node_frac.y()) * delta_V.y();
    } if(direction.z() < 0) {
      V_max.z() = node_frac.z() * delta_V.z();
    } else {
      V_max.z() = (1 - node_frac.z()) * delta_V.z();
    }

    const float zero_depth_band = 1.0e-6f;
    for (int i = 0; i < 3; i++) {
      if (std::fabs(direction[i]) < zero_depth_band)
        V_max[i] = std::numeric_limits<float>::infinity();
    }

    float V_min = std::min(std::min(V_max.x(), V_max.y()), V_max.z());

    v_add += V_min + 0.01;
    curr_vox = (v + v_add) * direction + origin_vox;

    value = map->get_fine(curr_vox.x(), curr_vox.y(), curr_vox.z(), scale);

    if (value.x_max > -0.2f) {
      while (value.x_max > -0.2f && scale > 2) {
        scale -= 1;
        value = map->get_fine(curr_vox.x(), curr_vox.y(), curr_vox.z(), scale);
      }
    } else {
      for (int s = scale + 1; s <= max_scale; s++) {
        value = map->get_fine(curr_vox.x(), curr_vox.y(), curr_vox.z(), s);

        if (value.x_max > -0.2f)
          break;
        scale += 1;
      }
    }
  }

  is_valid = false;
  return;
}

/*!
 * \brief Compute the intersection point and scale for a given ray
 * \param volume        Continuous map wrapper
 * \param origin_pos    Camera position in [m]
 * \param direction     Direction of the ray
 * \param p_near        Near plane distance in [m]
 * \param p_far         Far plane distance in [m]
 * \return              Surface intersection point in [m] and scale
 */
Eigen::Vector4f MultiresOFusion::raycast(const VolumeTemplate<MultiresOFusion, se::Octree>& volume,
                                         const Eigen::Vector3f& origin_pos,
                                         const Eigen::Vector3f& direction,
                                         float p_near,
                                         float p_far,
                                         float,
                                         float,
                                         float) {
  const int map_size = volume.size();             // map_size    := [voxel]
  const float map_res = volume.dim() / map_size;  // map_res     := [m / voxel];
  const float inv_map_res = 1.f / map_res;        // inv_map_res := [voxel / m];
  // inv_map_res := [m] to [voxel]; map_res := [voxel] to [m]
  float t_near = p_near;                          // max travel distance in [m]
  float t_far  = p_far;                           // min travel distance in [m]

  // Check if the ray origin is outside the map.
  // If so, compute the first point of contact with the map.
  // Stop if no intersection will occur (i.e. is_valid = false).
  bool is_valid = true;
  float t = computeMapIntersection(origin_pos, direction, volume.dim(), t_far, is_valid);

  if (!is_valid) {
    // Ray won't intersect with the map
    return Eigen::Vector4f::Zero();
  }

  int max_scale = 7;  // Max possible free space skipped per iteration (node size = 2^max_scale)
  max_scale = std::min(max_scale, volume.octree_->maxLevel() - 1);

  advanceRay(volume.octree_, origin_pos, direction, t, t_near, t_far, map_res, max_scale, is_valid);

  auto start = std::chrono::system_clock::now();
  if (!is_valid) {
    // Ray passes only through free space or intersects with the map before t_near or after t_far.
    return Eigen::Vector4f::Zero();
  }
  auto select_occupancy = [](const auto& val){ return val.x; };
  // first walk with largesteps until we found a hit
  float step_size = map_res / 2;
  Eigen::Vector3f curr_pos = origin_pos + direction * t;

  const int scale = 0;
  auto interp_res = volume.interp(curr_pos, scale, select_occupancy);
  float f_t = interp_res.first;
  float f_tt = 0;

  if (f_t <= MultiresOFusion::surface_boundary) { // ups, if we were already in it, then don't render anything here
    for (; t < t_far; t += step_size) {
      curr_pos =  origin_pos + direction * t;

      auto data = volume.get(curr_pos, scale);

      if (data.x > -0.2f && data.frame > 0.f) {
        interp_res = volume.interp(curr_pos, scale, select_occupancy);
        f_tt = interp_res.first;
      }
      if (f_tt > MultiresOFusion::surface_boundary)                  // got it, jump out of inner loop
      {
        break;
      }
      f_t = f_tt;
    }
    if (f_tt > MultiresOFusion::surface_boundary) {
      // got it, calculate accurate intersection
      t = t - step_size * (f_tt - MultiresOFusion::surface_boundary) / (f_tt - f_t);
      Eigen::Vector4f res = (origin_pos + direction * t).homogeneous();
      res.w() = interp_res.second;
      return res;
    }
  }

  return Eigen::Vector4f::Zero();
}

