// SPDX-FileCopyrightText: 2019-2020 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2019 Anna Dai
// SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#ifndef __INFORMATION_GAIN_HPP
#define __INFORMATION_GAIN_HPP

#include <se/octree.hpp>
#include <se/sensor_implementation.hpp>
#include <se/voxel_implementations.hpp>

namespace se {
  /** \brief Return the maximum possible entropy value for a single ray.
   * According to Nils, the maximum number of voxels a ray may cross is
   * 8 * sqrt(3) * (far_plane - near_plane) / voxel_dim
   * and since the maximum entropy is 1, this is also the maximum entropy for a ray.
   */
  float max_ray_entropy(const float voxel_dim,
                        const float near_plane,
                        const float far_plane);

  /** \brief Perform a 360 degree raycasting using a spherical camera model at t_MB computing the
   * map entropy. The size of entropy_image determines the number of rays cast. Frame B is
   * x-forward, z-up.
   */
  void raycast_entropy(Image<float>&                       entropy_image,
                       const Octree<VoxelImpl::VoxelType>& map,
                       const SensorImpl&                   sensor,
                       const Eigen::Vector3f&              t_MB);

  /** \brief Perform a 360 degree raycasting using a spherical camera model at t_MB computing the
   * depth along rays from t_MB. The size of depth_image determines the number of rays cast. Frame B
   * is x-forward, z-up.
   * \note Typically used only for visualization and debugging.
   */
  void raycast_depth(Image<float>&                       depth_image,
                     const Octree<VoxelImpl::VoxelType>& map,
                     const SensorImpl&                   sensor,
                     const Eigen::Vector3f&              t_MB);

  /** \brief Compute the yaw angle in the map frame M that maximizes the entropy.
   * \return The yaw angle (first) and the respective entropy (second).
   */
  std::pair<float, float> optimal_yaw(const Image<float>& entropy_image,
                                      const SensorImpl&   sensor);

  /** \brief Overlay the sensor FOV at yaw_M on the image.
   * \note Typically used only for visualization and debugging.
   */
  void overlay_yaw(Image<uint32_t>&  image,
                   const float       yaw_M,
                   const SensorImpl& sensor);



  /** Write the entropy values in a txt file.
   * \note DEBUG
   */
  int write_entropy(const std::string&  filename,
                    const Image<float>& entropy_image,
                    const SensorImpl&   sensor);
} // namespace se

#endif // __INFORMATION_GAIN_HPP
