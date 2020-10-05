// SPDX-FileCopyrightText: 2020 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#ifndef __FRONTIERS_HPP
#define __FRONTIERS_HPP

namespace se {
  // Must check that the supplied volume is free before calling
  // Not for Nodes
  template<typename T>
  bool isFrontier(const Eigen::Vector3i&            coord,
                  const int                         size_at_scale_li,
                  const int                         scale,
                  const typename T::VoxelBlockType& block,
                  const se::Octree<T>&              octree) {
    static const Eigen::Matrix<int, 3, 6> offsets = (Eigen::Matrix<int, 3, 6>()
        <<  0,  0, -1, 1, 0, 0,
            0, -1,  0, 0, 1, 0,
           -1,  0,  0, 0, 0, 1).finished();
    // Iterate over the 6 face neighbors
    for (int i = 0; i < 6; ++i) {
      const Eigen::Vector3i neighbour_coord = coord + size_at_scale_li * offsets.col(i);
      if (coord / T::VoxelBlockType::size_li == neighbour_coord / T::VoxelBlockType::size_li) {
        // The face neighbour is in the same VoxelBlock
        const auto& neighbour_data = block.data(neighbour_coord, scale);
        if (!T::isValid(neighbour_data)) {
          // The voxel has never been integrated into, found a frontier
          return true;
        }
      } else {
        // The face neighbour is in another VoxelBlock/Node
        typename T::VoxelData neighbour_data;
        (void) octree.get(neighbour_coord, neighbour_data, scale);
        if (!MultiresOFusion::VoxelType::isValid(neighbour_data)) {
          // The voxel/node has never been integrated into, found a frontier
          return true;
        }
      }
    }
    return false;
  }
} // namespace se

#endif // __FRONTIERS_HPP

