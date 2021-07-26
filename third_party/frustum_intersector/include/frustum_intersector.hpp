// SPDX-FileCopyrightText: 2021 Smart Robotics Lab
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#ifndef __FRUSTUM_INTERSECTOR_HPP
#define __FRUSTUM_INTERSECTOR_HPP

#include <Eigen/Dense>

namespace fi {
  constexpr int num_frustum_vertices = 8;
  constexpr int num_frustum_normals = 6;

  float frustum_volume(const Eigen::Matrix<float, 4, num_frustum_vertices>& frustum_vertices_C);

  float frustum_intersection_pc(
      const Eigen::Matrix<float, 4, num_frustum_vertices>& frustum_vertices_C,
      const Eigen::Matrix4f& T_C0C1);

} // namespace fi

#endif // __FRUSTUM_INTERSECTOR_HPP
