// SPDX-FileCopyrightText: 2020-2021 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2020-2021 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#ifndef __FRONTIERS_HPP
#define __FRONTIERS_HPP

namespace se {
  /** Test if the volume at the supplied coordinates and scale is a frontier.
   * Neighbors inside the same VoxelBlock are checked at the same scale. Neighbors at other
   * VoxelBlocks/Nodes are checked at the same or coarser scale.
   *
   * \warning Must check that the supplied volume is free before calling.
   */
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
        if (!T::isValid(neighbour_data)) {
          // The voxel/node has never been integrated into, found a frontier
          return true;
        }
      }
    }
    return false;
  }



  /** Test if the child at child_idx of node is a frontier.
   *
   * \warning Must check that the child is free before calling.
   */
  template<typename T>
  bool isFrontier(const int            child_idx,
                  const se::Node<T>&   node,
                  const se::Octree<T>& octree) {
    static const Eigen::Matrix<int, 3, 6> offsets = (Eigen::Matrix<int, 3, 6>()
        <<  0,  0, -1, 1, 0, 0,
            0, -1,  0, 0, 1, 0,
           -1,  0,  0, 0, 0, 1).finished();
    const int node_depth = octree.sizeToDepth(node.size());
    const int child_size = node.size() / 2;
    const Eigen::Vector3i child_coord = node.childCoord(child_idx);
    for (int i = 0; i < 6; ++i) {
      const Eigen::Vector3i neighbour_coord = child_coord + child_size * offsets.col(i);
      if (child_coord / child_size == neighbour_coord / child_size) {
        // The face neighbour is in the same Node
        const Eigen::Vector3i rel_coord = neighbour_coord - node.coordinates();
        const int sibling_idx = rel_coord.x() + rel_coord.y() * 2 + rel_coord.z() * 4;
        const auto& neighbour_data = node.childData(sibling_idx);
        if (!T::isValid(neighbour_data)) {
          // The Node has never been integrated into, found a frontier
          return true;
        }
      } else {
        // The face neighbour is in another Node
        const se::Node<T>& neighbour_parent = *(octree.fetchLowestNode(neighbour_coord, node_depth));
        const Eigen::Vector3i rel_coord = neighbour_coord - neighbour_parent.coordinates();
        const int neighbor_idx = rel_coord.x() + rel_coord.y() * 2 + rel_coord.z() * 4;
        const auto& neighbour_data = neighbour_parent.childData(neighbor_idx);
        if (!T::isValid(neighbour_data)) {
          // The Node has never been integrated into, found a frontier
          return true;
        }
      }
    }
    return true;
  }



  /** Update the frontier status of all voxels at the last updated scale of node and return their
   * total volume in primitive voxels.
  */
  template<typename T>
  int updateFrontierData(se::Node<T>*         node,
                         const se::Octree<T>& octree) {
    int frontier_volume = 0;
    if (node == nullptr) {
      return frontier_volume;
    }
    if (node->isBlock()) {
      // VoxelBlock
      typename T::VoxelBlockType* block = reinterpret_cast<typename T::VoxelBlockType*>(node);
      const int scale = block->current_scale();
      // The number of voxels per edge at this scale
      const int size_at_scale_li = block->scaleSize(scale);
      // The volume in primitive voxels of a voxel at scale
      const int voxel_volume_at_scale = se::math::cu(block->scaleVoxelSize(scale));
      // Iterate over each voxel at scale
      for (int i = 0; i < block->scaleNumVoxels(scale); i++) {
        auto data = block->data(i, scale);
        // Ensure it is free before checking whether it's a frontier
        if (T::isFree(data)) {
          const Eigen::Vector3i coords = block->voxelCoordinates(i, scale);
          // Update the frontier state and frontier volume
          if (se::isFrontier(coords, size_at_scale_li, scale, *block, octree)) {
            data.frontier = true;
            frontier_volume += voxel_volume_at_scale;
          } else {
            data.frontier = false;
          }
          block->setData(i, scale, data);
        }
      }
    } else {
      // Node
      // The volume in primitive voxels of a child Node
      const int child_node_volume = se::math::cu(node->size() / 2);
      // Iterate over all leaf children
      for (int i = 0; i < 8; i++) {
        if (node->child(i) == nullptr) {
          auto& data = node->childData(i);
          // Ensure it is free before checking whether it's a frontier
          if (T::isFree(data)) {
            // Update the frontier state and frontier volume
            if (se::isFrontier(i, *node, octree)) {
              data.frontier = true;
              frontier_volume += child_node_volume;
            } else {
              data.frontier = false;
            }
            node->childData(i, data);
          }
        }
      }
    }
    return frontier_volume;
  }
} // namespace se

#endif // __FRONTIERS_HPP

