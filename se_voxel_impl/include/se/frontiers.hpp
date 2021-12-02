// SPDX-FileCopyrightText: 2020-2021 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2020-2021 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#ifndef __FRONTIERS_HPP
#define __FRONTIERS_HPP

#include <se/octree.hpp>
#include <se/set_operations.hpp>

namespace se {
/** Test if the volume at the supplied coordinates and scale is a frontier.
   * Neighbors inside the same VoxelBlock are checked at the same scale. Neighbors at other
   * VoxelBlocks/Nodes are checked at the same or coarser scale.
   *
   * \warning Must check that the supplied volume is free before calling.
   */
template<typename T>
bool is_frontier(const Eigen::Vector3i& coord,
                 const int scale,
                 const typename T::VoxelBlockType& block,
                 const se::Octree<T>& octree)
{
    assert(octree.contains(coord));
    assert(block.contains(coord));
    assert(0 <= scale && scale <= T::VoxelBlockType::max_scale);

    static const Eigen::Matrix<int, 3, 6> offsets =
        (Eigen::Matrix<int, 3, 6>() << 0, 0, -1, 1, 0, 0, 0, -1, 0, 0, 1, 0, -1, 0, 0, 0, 0, 1)
            .finished();

    // The edge length of a volume at scale in primitive voxels
    const int scale_volume_size = VoxelBlock<T>::scaleVoxelSize(scale);
    // Iterate over the 6 face neighbors
    for (int i = 0; i < 6; ++i) {
        const Eigen::Vector3i neighbour_coord = coord + scale_volume_size * offsets.col(i);
        if (octree.contains(neighbour_coord)) {
            if (block.contains(neighbour_coord)) {
                // The face neighbour is in the same VoxelBlock
                const auto& neighbour_data = block.data(neighbour_coord, scale);
                if (!T::isValid(neighbour_data)) {
                    // The voxel has never been integrated into, found a frontier
                    return true;
                }
            }
            else {
                // The face neighbour is in another VoxelBlock/Node
                typename T::VoxelData neighbour_data;
                // get() will return the data at the lowest allocated scale up to `scale`. Thus no
                // erroneous frontiers will be detected if the neighbour is allocated at a higher scale
                // than the query volume.
                (void) octree.get(neighbour_coord, neighbour_data, scale);
                if (!T::isValid(neighbour_data)) {
                    // The voxel/node has never been integrated into, found a frontier
                    return true;
                }
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
bool is_frontier(const int child_idx, const se::Node<T>& node, const se::Octree<T>& octree)
{
    assert(0 <= child_idx && child_idx <= 8);

    static const Eigen::Matrix<int, 3, 6> offsets =
        (Eigen::Matrix<int, 3, 6>() << 0, 0, -1, 1, 0, 0, 0, -1, 0, 0, 1, 0, -1, 0, 0, 0, 0, 1)
            .finished();
    const int node_depth = octree.sizeToDepth(node.size());
    const int child_size = node.size() / 2;
    const Eigen::Vector3i child_coord = node.childCoord(child_idx);
    for (int i = 0; i < 6; ++i) {
        const Eigen::Vector3i neighbour_coord = child_coord + child_size * offsets.col(i);
        if (octree.contains(neighbour_coord)) {
            if (child_coord / child_size == neighbour_coord / child_size) {
                // The face neighbour is in the same Node
                const Eigen::Vector3i rel_coord = neighbour_coord - node.coordinates();
                const int sibling_idx = rel_coord.x() + rel_coord.y() * 2 + rel_coord.z() * 4;
                const auto& neighbour_data = node.childData(sibling_idx);
                if (!T::isValid(neighbour_data)) {
                    // The Node has never been integrated into, found a frontier
                    return true;
                }
            }
            else {
                // The face neighbour is in another Node
                const se::Node<T>& neighbour_parent =
                    *(octree.fetchLowestNode(neighbour_coord, node_depth));
                const Eigen::Vector3i rel_coord = neighbour_coord - neighbour_parent.coordinates();
                const int neighbor_idx = rel_coord.x() + rel_coord.y() * 2 + rel_coord.z() * 4;
                const auto& neighbour_data = neighbour_parent.childData(neighbor_idx);
                if (!T::isValid(neighbour_data)) {
                    // The Node has never been integrated into, found a frontier
                    return true;
                }
            }
        }
    }
    return true;
}



/** Update the frontier status of all voxels at the last updated scale of node and return their
   * total volume in primitive voxels.
  */
template<typename T>
int update_frontier_data(se::Node<T>& node, const se::Octree<T>& octree)
{
    int frontier_volume = 0;
    if (node.isBlock()) {
        // VoxelBlock
        typename T::VoxelBlockType& block = reinterpret_cast<typename T::VoxelBlockType&>(node);
        const int scale = block.current_scale();
        // The volume in primitive voxels of a voxel at scale
        const int voxel_volume_at_scale = se::math::cu(block.scaleVoxelSize(scale));
        // Iterate over each voxel at scale
        for (int i = 0; i < block.scaleNumVoxels(scale); i++) {
            auto data = block.data(i, scale);
            // Ensure it is free before checking whether it's a frontier
            if (T::isFree(data)) {
                const Eigen::Vector3i coords = block.voxelCoordinates(i, scale);
                // Update the frontier state and frontier volume
                if (se::is_frontier(coords, scale, block, octree)) {
                    data.frontier = true;
                    frontier_volume += voxel_volume_at_scale;
                }
                else {
                    data.frontier = false;
                }
                block.setData(i, scale, data);
            }
        }
    }
    else {
        // Node
        // The volume in primitive voxels of a child Node
        const int child_node_volume = se::math::cu(node.size() / 2);
        // Iterate over all leaf children
        for (int i = 0; i < 8; i++) {
            if (node.child(i) == nullptr) {
                auto& data = node.childData(i);
                // Ensure it is free before checking whether it's a frontier
                if (T::isFree(data)) {
                    // Update the frontier state and frontier volume
                    if (se::is_frontier(i, node, octree)) {
                        data.frontier = true;
                        frontier_volume += child_node_volume;
                    }
                    else {
                        data.frontier = false;
                    }
                    node.childData(i, data);
                }
            }
        }
    }
    return frontier_volume;
}



/** Update the frontier status of all voxels of VoxelBlocks in frontiers. VoxelBlocks with frontier
 * volume over total volume less than min_frontier_ratio will be removed from frontiers. If
 * min_frontier_ratio is set to 0, all frontiers will be kept in frontiers.
 */
template<typename T>
void update_frontiers(se::Octree<T>& octree,
                      std::set<se::key_t>& frontiers,
                      const float min_frontier_ratio)
{
    std::set<se::key_t> not_frontiers;
    // Remove VoxelBlocks that no longer correspond to frontiers
    for (auto code : frontiers) {
        const Eigen::Vector3i node_coord = se::keyops::decode(code);
        const int node_depth = se::keyops::depth(code);
        se::Node<T>* node = octree.fetchNode(node_coord, node_depth);
        // Remove unallocated nodes from the frontiers
        if (node == nullptr) {
            not_frontiers.insert(code);
            continue;
        }
        // Update the frontier status of the Node's voxel
        const int frontier_volume = update_frontier_data(*node, octree);
        const float frontier_ratio =
            static_cast<float>(frontier_volume) / se::math::cu(node->size());
        // Remove the Node if its frontiers are too small
        if (frontier_volume == 0 || frontier_ratio < min_frontier_ratio) {
            not_frontiers.insert(code);
        }
    }
    se::setminus(frontiers, not_frontiers);
}



/** Return all the frontier volumes in the octree. Search only the VoxelBlocks in frontiers
   */
template<typename T>
std::vector<se::Volume<T>> frontier_volumes(const se::Octree<T>& octree,
                                            const std::set<se::key_t>& frontiers)
{
    std::vector<se::Volume<T>> volumes;
    const float voxel_dim = octree.voxelDim();
    for (const auto& code : frontiers) {
        const int depth = se::keyops::depth(code);
        const Eigen::Vector3i coord = se::keyops::decode(code);
        if (depth == octree.blockDepth()) {
            // Frontier VoxelBlock, find the individual frontier voxels
            const typename T::VoxelBlockType* const block = octree.fetch(coord);
            if (block) {
                const int scale = block->current_scale();
                const int size = T::VoxelBlockType::scaleVoxelSize(scale);
                const float dim = voxel_dim * size;
                const Eigen::Vector3f voxel_centre_offset_M = Eigen::Vector3f::Constant(dim / 2.0f);
                for (int i = 0; i < T::VoxelBlockType::scaleNumVoxels(scale); ++i) {
                    const auto& data = block->data(i, scale);
                    if (data.frontier) {
                        const Eigen::Vector3f voxel_coord_M =
                            voxel_dim * block->voxelCoordinates(i, scale).template cast<float>();
                        const Eigen::Vector3f centre_M = voxel_coord_M + voxel_centre_offset_M;
                        volumes.emplace_back(centre_M, dim, size, data);
                    }
                }
            }
            else {
                assert(false && "This branch shouldn't have been reached since we store "
              "the Morton of the Node's parent. The data of the VoxelBlock will be stored "
              "in its parent Node whose Morton should be in frontiers.");
            }
        }
        else {
            // Node with frontier children, fetch it
            const se::Node<T>* const node = octree.fetchNode(coord, depth);
            if (node) {
                const int child_size = node->size() / 2;
                const float child_dim = voxel_dim * child_size;
                const Eigen::Vector3f child_centre_offset_M =
                    Eigen::Vector3f::Constant(child_dim / 2.0f);
                // Iterate over the unallocated children
                for (int i = 0; i < 8; i++) {
                    if (!node->child(i)) {
                        const auto& child_data = node->childData(i);
                        if (child_data.frontier) {
                            const Eigen::Vector3f child_coord_M =
                                voxel_dim * node->childCoord(i).template cast<float>();
                            const Eigen::Vector3f child_centre_M =
                                child_coord_M + child_centre_offset_M;
                            volumes.emplace_back(child_centre_M, child_dim, child_size, child_data);
                        }
                    }
                }
            }
            else {
                assert(false && "This branch shouldn't have been reached.");
            }
        }
    }
    return volumes;
}
} // namespace se

#endif // __FRONTIERS_HPP
