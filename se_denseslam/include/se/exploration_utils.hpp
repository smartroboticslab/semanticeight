// SPDX-FileCopyrightText: 2021 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#ifndef __EXPLORATION_UTILS_HPP
#define __EXPLORATION_UTILS_HPP

#include <se/octree.hpp>
#include <se/voxel_implementations.hpp>

namespace se {
/** Compute the explored volume of an se::Octree. All values are in m^3.
   */
struct ExploredVolume {
    ExploredVolume() = default;
    ExploredVolume(se::Octree<VoxelImpl::VoxelType>& map,
                   const Eigen::Vector3f& aabb_min_M = Eigen::Vector3f::Zero(),
                   const Eigen::Vector3f& aabb_max_M = Eigen::Vector3f::Zero());

    float free_volume = 0.0f;
    float occupied_volume = 0.0f;
    float explored_volume = 0.0f;
};

void freeSphere(se::Octree<VoxelImpl::VoxelType>& map, const Eigen::Vector3f& centre, float radius);
} // namespace se

#endif // __EXPLORATION_UTILS_HPP
