// SPDX-FileCopyrightText: 2021 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#include "se/exploration_utils.hpp"
#include "se/utils/math_utils.h"

namespace se {
  ExploredVolume::ExploredVolume(se::Octree<VoxelImpl::VoxelType>& map) {
    for (const auto& volume : map) {
      // The iterator will not return invalid (uninitialized) data so just focus on free and
      // occupied.
      if (se::math::cu(volume.dim) == 0.0f) {
        std::cout << "WAT?\n";
      }
      if (VoxelImpl::VoxelType::isFree(volume.data)) {
        free_volume += se::math::cu(volume.dim);
      } else {
        occupied_volume += se::math::cu(volume.dim);
      }
    }
    explored_volume = free_volume + occupied_volume;
  }



  void freeSphere(se::Octree<VoxelImpl::VoxelType>& map,
                  const Eigen::Vector3f&            centre_M,
                  float                             radius) {
    if (!std::is_same<VoxelImpl, MultiresOFusion>::value) {
      std::cerr << "Error: Only MultiresOFusion is supported\n";
      std::abort();
    }
    // Compute the sphere's AABB corners in metres and voxels
    const Eigen::Vector3f aabb_min_M = centre_M - Eigen::Vector3f::Constant(radius);
    const Eigen::Vector3f aabb_max_M = centre_M + Eigen::Vector3f::Constant(radius);
    // Compute the coordinates of all the points corresponding to voxels in the AABB
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> aabb_points_M;
    for (float z = aabb_min_M.z(); z <= aabb_max_M.z(); z += map.voxelDim()) {
      for (float y = aabb_min_M.y(); y <= aabb_max_M.y(); y += map.voxelDim()) {
        for (float x = aabb_min_M.x(); x <= aabb_max_M.x(); x += map.voxelDim()) {
          aabb_points_M.push_back(Eigen::Vector3f(x, y, z));
        }
      }
    }
    // Allocate the required VoxelBlocks
    std::set<se::key_t> code_set;
    for (const auto& point_M : aabb_points_M) {
      const Eigen::Vector3i voxel = map.pointToVoxel(point_M);
      if (map.contains(voxel)) {
        code_set.insert(map.hash(voxel.x(), voxel.y(), voxel.z(), map.blockDepth()));
      }
    }
    std::vector<se::key_t> codes (code_set.begin(), code_set.end());
    map.allocate(codes.data(), codes.size());
    // The data to store in the free voxels
    auto data = VoxelImpl::VoxelType::initData();
    data.x = -21; // The path planning threshold is -20.
    data.y = 1;
    data.observed = true;
    // Allocate the VoxelBlocks up to some scale
    constexpr int scale = 3;
    std::vector<VoxelImpl::VoxelBlockType*> blocks;
    map.getBlockList(blocks, false);
    for (auto& block : blocks) {
      block->active(true);
      block->allocateDownTo(scale);
    }
    // Set the sphere voxels to free
    for (const auto& point_M : aabb_points_M) {
      const Eigen::Vector3f voxel_dist_M = (centre_M - point_M).array().abs().matrix();
      if (voxel_dist_M.norm() <= radius) {
        map.setAtPoint(point_M, data, scale);
      }
    }
  }
} // namespace se

