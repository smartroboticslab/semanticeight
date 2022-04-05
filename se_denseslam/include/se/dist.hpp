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

static constexpr float default_desired_distance = 1.0f;

/** \brief Compute the distance gain by observing a VoxelBlock with minimum observation distance
 * block_min_dist at block_expected_dist expected_scale.
 */
float dist_gain(const float block_min_dist,
                const float block_expected_dist,
                const float desired_dist = default_desired_distance);

/** \brief Create a 360 degree distance gain image by getting the minimum observation distance of
 * objects along the provided rays. It is meant to be used with the hits returned by
 * se::raycast_entropy().
 */
Image<float> object_dist_gain(const Image<Eigen::Vector3f>& bg_hits_M,
                              const Objects& objects,
                              const SensorImpl& sensor,
                              const Eigen::Matrix4f& T_MB,
                              const Eigen::Matrix4f& T_BC,
                              const float desired_dist = default_desired_distance);

} // namespace se

#endif // DIST_HPP
