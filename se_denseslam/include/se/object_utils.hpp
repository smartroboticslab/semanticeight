// SPDX-FileCopyrightText: 2021 Smart Robotics Lab
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#ifndef OBJECT_UTILS_HPP
#define OBJECT_UTILS_HPP

#include <set>

#include "se/object.hpp"

namespace se {

/** Return the set of object instance IDs that are visible by a sensor located at T_MC.
 */
std::set<int> get_visible_object_ids(const Objects& objects,
                                     const SensorImpl& sensor,
                                     const Eigen::Matrix4f& T_MC);

/** Return a subset of objects containing only the objects with instance IDs in
 * visible_object_ids.
 */
Objects filter_visible_objects(const Objects& objects, const std::set<int>& visible_object_ids);

/** Return the per-object Level-of-Detail gain by counting VoxelBlocks with minimum scales other
 * than 0.
 *
 * \note UNUSED
 */
std::vector<float> object_lod_gain_blocks(const Objects& objects,
                                          const SensorImpl& sensor,
                                          const Eigen::Matrix4f& T_MC);

/** Return the per-object Level-of-Detail gain by raycasting objects and counting hits with
 * minimum scales other than 0 .
 *
 * \note UNUSED
 */
std::vector<float> object_lod_gain_raycasting(const Objects& objects,
                                              const SensorImpl& sensor,
                                              const Eigen::Matrix4f& T_MC);

/** Return the per-image Level-of-Detail gain by raycasting objects. The current minimum scale is
 * compared with the expected integration scale. The gain is in the interval [0, 1].
 */
float lod_gain_raycasting(const Objects& objects,
                          const SensorImpl& sensor,
                          const SensorImpl& raycasting_sensor,
                          const Eigen::Matrix4f& T_MC,
                          Image<int8_t>& min_scale_image);

/** Return an array with the percentage of data at each scale aggregated over all objects.
 */
Object::ScaleArray<float> combinedPercentageAtScale(const Objects& objects);

} // namespace se

#endif // OBJECT_UTILS_HPP
