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

/** \brief Compute the azimuth angle given a column index, the image width and the horizontal
 * sensor FOV. It is assumed that the image spans an azimuth angle range of hfov and that azimuth
 * angle 0 corresponds to the middle column of the image.
 * \return The azimuth angle in the interval [-pi,pi).
 */
float index_to_azimuth(const int x_idx, const int width, const float hfov);

/** \brief Compute the polar angle given a row index, the image height and the vertical sensor
 * FOV. It is assumed that the image spans a polar angle range of vfov and that polar angle pi/2
 * corresponds to the middle row of the image.
 * \return The polar angle in the interval [0,pi].
 */
float index_to_polar(int y_idx, int height, float vfov, float pitch_offset);

/** \brief Compute the column index given an azimuth angle, the image width and the horizontal
 * sensor FOV. It is assumed that the image spans an azimuth angle range of hfov and that azimuth
 * angle 0 corresponds to the middle column of the image.
 * \return The column index in the interval [0,width-1].
 */
int azimuth_to_index(const float theta, const int width, const float hfov);

/** \brief Return the maximum possible entropy value for a single ray.
 * According to Nils, the maximum number of voxels a ray may cross is
 * 8 * sqrt(3) * (far_plane - near_plane) / voxel_dim
 * and since the maximum entropy is 1, this is also the maximum entropy for a ray.
 */
float max_ray_entropy(const float voxel_dim, const float near_plane, const float far_plane);

/** \brief Compute the ray directions in the map frame M (z-up) for all pixels of an image captured
 * by sensor.
 */
Image<Eigen::Vector3f> ray_M_image(const SensorImpl& sensor, const Eigen::Matrix4f& T_MC);

/** \brief Compute the ray directions in the map frame M (z-up) for all pixels of a width x height
 * 360 degree image.
 */
Image<Eigen::Vector3f> ray_M_360_image(const int width,
                                       const int height,
                                       const SensorImpl& sensor,
                                       const Eigen::Matrix4f& T_BC,
                                       const float roll_pitch_threshold = 0.0f);

/** \brief TODO
 */
void raycast_entropy(Image<float>& entropy_image,
                     Image<Eigen::Vector3f>& entropy_hits_M,
                     const Octree<VoxelImpl::VoxelType>& map,
                     const SensorImpl& sensor,
                     const Eigen::Matrix4f& T_MB,
                     const Eigen::Matrix4f& T_BC);

/** \brief Perform a 360 degree raycasting using a spherical camera model at t_MC computing the
 * map entropy. The size of entropy_image determines the number of rays cast. Frame B is
 * x-forward, z-up. The raycasting is performed from the origin of the body frame instead of the
 * origin of the camera frame. This approximation works well enough if they are close.
 */
void raycast_entropy_360(Image<float>& entropy_image,
                         Image<Eigen::Vector3f>& entropy_hits_M,
                         const Octree<VoxelImpl::VoxelType>& map,
                         const SensorImpl& sensor,
                         const Eigen::Matrix4f& T_MB,
                         const Eigen::Matrix4f& T_BC,
                         const float roll_pitch_threshold);

Image<float> mask_entropy_image(const Image<float>& entropy_image,
                                const Image<uint8_t>& frustum_overlap_mask);

/** \brief Compute the yaw angle in the map frame M that maximizes the entropy.
 * \return The yaw angle, the respective entropy, the index of the left edge of the window and the
 * window width.
 */
std::tuple<float, float, int, int> optimal_yaw(const Image<float>& entropy_image,
                                               const Image<Eigen::Vector3f>& entropy_hits_M,
                                               const SensorImpl& sensor,
                                               const Eigen::Matrix4f& T_MB,
                                               const Eigen::Matrix4f& T_BC);

Image<uint32_t> visualize_entropy(const Image<float>& entropy,
                                  const int window_idx,
                                  const int window_width,
                                  const bool visualize_yaw = true);

Image<uint32_t> visualize_depth(const Image<Eigen::Vector3f>& entropy_hits_M,
                                const SensorImpl& sensor,
                                const Eigen::Matrix4f& T_MB,
                                const int window_idx,
                                const int window_width,
                                const bool visualize_yaw = true);

/** \brief Overlay the sensor FOV at yaw_M on the image.
 * \note Typically used only for visualization and debugging.
 */
void overlay_yaw(Image<uint32_t>& image, const int window_idx, const int window_width);

void render_pose_entropy_depth(Image<uint32_t>& entropy,
                               Image<uint32_t>& depth,
                               const Octree<VoxelImpl::VoxelType>& map,
                               const SensorImpl& sensor,
                               const Eigen::Matrix4f& T_MB,
                               const Eigen::Matrix4f& T_BC,
                               const bool visualize_yaw = true,
                               const float roll_pitch_threshold = 0.0f);

} // namespace se

#endif // __INFORMATION_GAIN_HPP
