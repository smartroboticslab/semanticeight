// SPDX-FileCopyrightText: 2022 Smart Robotics Lab
// SPDX-FileCopyrightText: 2022 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DIST_HPP
#define DIST_HPP

#include "se/image/image.hpp"
#include "se/object.hpp"
#include "se/octree.hpp"
#include "se/sensor_implementation.hpp"
#include "se/voxel_implementations.hpp"

namespace se {

static constexpr float default_desired_distance_bg = 3.0f;
static constexpr float default_desired_distance_object = 1.0f;

/** \brief Compute the distance gain attained by observing a VoxelBlock whose minimum observation
 * distance is block_min_dist at block_expected_dist given that the desired observation distance is
 * desired_dist.
 */
float dist_gain(const float block_min_dist,
                const float block_expected_dist,
                const float desired_dist);

/** \brief Create a distance gain image by getting the minimum observation distance of the
 * background along the provided hit rays. It is meant to be used with the hits returned by
 * se::raycast_entropy_360() or se::raycast_entropy().
 */
Image<float> bg_dist_gain(const Image<Eigen::Vector3f>& bg_hits_M,
                          const Octree<VoxelImpl::VoxelType>& map,
                          const SensorImpl& sensor,
                          const Eigen::Matrix4f& T_MB,
                          const Eigen::Matrix4f& T_BC,
                          const float desired_dist = default_desired_distance_bg);

/** \brief Create a distance gain image by getting the minimum observation distance of objects along
 * the provided hit rays. It is meant to be used with the hits returned by se::raycast_entropy_360()
 * or se::raycast_entropy().
 */
Image<float> object_dist_gain(const Image<Eigen::Vector3f>& bg_hits_M,
                              const Objects& objects,
                              const SensorImpl& sensor,
                              const Eigen::Matrix4f& T_MB,
                              const Eigen::Matrix4f& T_BC,
                              const float desired_dist = default_desired_distance_object);

} // namespace se

#endif // DIST_HPP
