// SPDX-FileCopyrightText: 2022 Smart Robotics Lab
// SPDX-FileCopyrightText: 2022 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LOD_HPP
#define LOD_HPP

#include "se/image/image.hpp"
#include "se/object.hpp"
#include "se/octree.hpp"
#include "se/sensor_implementation.hpp"
#include "se/voxel_implementations.hpp"

namespace se {

/** \brief Compute the scale gain by observing a VoxelBlock with minimum observed scale min_scale at
 * scale expected_scale.
 */
int8_t scale_gain(const int8_t block_min_scale,
                  const int8_t block_expected_scale,
                  const int8_t desired_scale = 0,
                  const int8_t max_scale = VoxelImpl::VoxelBlockType::max_scale);

/** \brief Create a 360 degree scale gain image by getting the scale at the provided hits.
 * It is meant to be used with the hits returned by se::raycast_entropy_360().
 */
Image<float> bg_scale_gain(const Image<Eigen::Vector3f>& bg_hits_M,
                           const Octree<VoxelImpl::VoxelType>& map,
                           const SensorImpl& sensor,
                           const Eigen::Matrix4f& T_MB,
                           const Eigen::Matrix4f& T_BC,
                           const int8_t desired_scale = 0);

/** \brief Create a 360 degree scale gain image by getting the scale of objects along the provided
 * rays. It is meant to be used with the hits returned by se::raycast_entropy_360() and
 * se::raycast_entropy().
 */
Image<float> object_scale_gain(const Image<Eigen::Vector3f>& bg_hits_M,
                               const Objects& objects,
                               const SensorImpl& sensor,
                               const Eigen::Matrix4f& T_MB,
                               const Eigen::Matrix4f& T_BC,
                               const int8_t desired_scale = 0);

} // namespace se

#endif // LOD_HPP
