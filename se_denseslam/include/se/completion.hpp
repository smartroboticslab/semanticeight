// SPDX-FileCopyrightText: 2022 Smart Robotics Lab
// SPDX-FileCopyrightText: 2022 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#ifndef COMPLETION_HPP
#define COMPLETION_HPP

#include "se/image/image.hpp"
#include "se/object.hpp"
#include "se/sensor_implementation.hpp"

namespace se {

/** \brief Create a completion gain image by performing back-face raycasting on the objects. The
 * raycasting is performed along the rays in rays_M but not for any rays that correspond to hits in
 * hits_M. The returned image contains values in the set {0, 1}.
 */
Image<float> object_completion_gain(const Image<Eigen::Vector3f>& rays_M,
                                    const Image<Eigen::Vector3f>& hits_M,
                                    const Objects& objects,
                                    const SensorImpl& sensor,
                                    const Eigen::Matrix4f& T_MB,
                                    const Eigen::Matrix4f& T_BC);

} // namespace se

#endif // COMPLETION_HPP
