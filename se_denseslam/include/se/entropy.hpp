// SPDX-FileCopyrightText: 2019-2020 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2019 Anna Dai
// SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#ifndef __INFORMATION_GAIN_HPP
#define __INFORMATION_GAIN_HPP

#include <se/octree.hpp>
#include <se/pose_grid_history.hpp>
#include <se/pose_vector_history.hpp>
#include <se/sensor_implementation.hpp>
#include <se/voxel_implementations.hpp>

namespace se {
/** \brief Return the maximum possible entropy value for a single ray.
 * According to Nils, the maximum number of voxels a ray may cross is
 * 8 * sqrt(3) * (far_plane - near_plane) / voxel_dim
 * and since the maximum entropy is 1, this is also the maximum entropy for a ray.
 */
float max_ray_entropy(const float voxel_dim, const float near_plane, const float far_plane);

/** \brief Perform a 360 degree raycasting using a spherical camera model at t_MC computing the
 * map entropy. The size of entropy_image determines the number of rays cast. Frame B is
 * x-forward, z-up. The raycasting is performed from the origin of the body frame instead of the
 * origin of the camera frame. This approximation works well enough if they are close.
 */
void raycast_entropy(Image<float>& entropy_image,
                     Image<Eigen::Vector3f>& entropy_hits_M,
                     const Octree<VoxelImpl::VoxelType>& map,
                     const SensorImpl& sensor,
                     const Eigen::Matrix4f& T_MB,
                     const Eigen::Matrix4f& T_BC);

void frustum_overlap(Image<float>& frustum_overlap_image,
                     const SensorImpl& sensor,
                     const Eigen::Matrix4f& T_MC,
                     const Eigen::Matrix4f& T_BC,
                     const PoseHistory* T_MB_history);

/** \brief Compute the yaw angle in the map frame M that maximizes the entropy.
 * \return The yaw angle (first) and the respective entropy (second).
 */
std::pair<float, float> optimal_yaw(const Image<float>& entropy_image,
                                    const Image<Eigen::Vector3f>& entropy_hits_M,
                                    const Image<float>& frustum_overlap_image,
                                    const SensorImpl& sensor,
                                    const Eigen::Matrix4f& T_MB,
                                    const Eigen::Matrix4f& T_BC);

Image<uint32_t> visualize_entropy(const Image<float>& entropy,
                                  const SensorImpl& sensor,
                                  const float yaw_M,
                                  const bool visualize_yaw = true);

Image<uint32_t> visualize_depth(const Image<Eigen::Vector3f>& entropy_hits_M,
                                const SensorImpl& sensor,
                                const Eigen::Matrix4f& T_MB,
                                const float yaw_M,
                                const bool visualize_yaw = true);

/** \brief Overlay the sensor FOV at yaw_M on the image.
 * \note Typically used only for visualization and debugging.
 */
void overlay_yaw(Image<uint32_t>& image, const float yaw_M, const SensorImpl& sensor);

void render_pose_entropy_depth(Image<uint32_t>& entropy,
                               Image<uint32_t>& depth,
                               const Octree<VoxelImpl::VoxelType>& map,
                               const SensorImpl& sensor,
                               const Eigen::Matrix4f& T_MB,
                               const Eigen::Matrix4f& T_BC,
                               const bool visualize_yaw = true);

} // namespace se

#endif // __INFORMATION_GAIN_HPP
