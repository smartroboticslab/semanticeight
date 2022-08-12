/**
 * Probabilistic Trajectory Planning, Map interface for supereight library of Emanuele Vespa.
 *
 * Copyright (C) 2018 Imperial College London.
 * Copyright (C) 2018 ETH Zürich.
 *
 * @file OccupancyWorld.cpp
 *
 *
 * @author Nils Funk
 * @date July 5, 2018
 */

#include "ptp/OccupancyWorld.hpp"

namespace ptp {
OccupancyWorld::OccupancyWorld(const se::Octree<MultiresOFusion::VoxelType>& map) : map_(map)
{
    updateMapBounds();
}

OccupancyWorld::OccupancyWorld(const se::Octree<VoxelImpl::VoxelType>& map,
                               const Eigen::Vector3f& planning_bounds_min_m,
                               const Eigen::Vector3f& planning_bounds_max_m) :
        OccupancyWorld::OccupancyWorld(map)
{
    const Eigen::Vector3f planning_bounds_min_v = map_.pointToVoxelF(planning_bounds_min_m);
    const Eigen::Vector3f planning_bounds_max_v = map_.pointToVoxelF(planning_bounds_max_m);
    map_bounds_min_ = map_bounds_min_.cwiseMax(planning_bounds_min_v);
    map_bounds_max_ = map_bounds_max_.cwiseMin(planning_bounds_max_v);
}

const se::Octree<VoxelImpl::VoxelType>& OccupancyWorld::getMap() const
{
    return map_;
}

void OccupancyWorld::getMapBounds(Eigen::Vector3f& map_bounds_min_v,
                                  Eigen::Vector3f& map_bounds_max_v) const
{
    map_bounds_min_v = map_bounds_min_;
    map_bounds_max_v = map_bounds_max_;
}

void OccupancyWorld::getMapBoundsMeter(Eigen::Vector3f& map_bounds_min_m,
                                       Eigen::Vector3f& map_bounds_max_m) const
{
    map_bounds_min_m = map_.voxelFToPoint(map_bounds_min_);
    map_bounds_max_m = map_.voxelFToPoint(map_bounds_max_);
}

bool OccupancyWorld::inMapBounds(const Eigen::Vector3f& voxel_coord) const
{
    return (map_bounds_min_.x() <= voxel_coord.x() && voxel_coord.x() <= map_bounds_max_.x()
            && map_bounds_min_.y() <= voxel_coord.y() && voxel_coord.y() <= map_bounds_max_.y()
            && map_bounds_min_.z() <= voxel_coord.z() && voxel_coord.z() <= map_bounds_max_.z());
}

bool OccupancyWorld::inMapBoundsMeter(const Eigen::Vector3f& point_M) const
{
    return inMapBounds(map_.pointToVoxelF(point_M));
}

bool OccupancyWorld::isFree(const Eigen::Vector3i& voxel_coord, const float threshold) const
{
    MultiresOFusion::VoxelType::VoxelData value;
    map_.getMax(voxel_coord, value);
    return (value.observed == true && MultiresOFusion::VoxelType::threshold(value) < threshold);
}

bool OccupancyWorld::isFreeAtPoint(const Eigen::Vector3f& point_M, const float threshold) const
{
    return isFree(map_.pointToVoxel(point_M), threshold);
}

Eigen::Vector3f OccupancyWorld::closestPointMeter(const Eigen::Vector3f& point_M) const
{
    const Eigen::Vector3f map_bounds_min_M = map_.voxelFToPoint(map_bounds_min_);
    const Eigen::Vector3f map_bounds_max_M = map_.voxelFToPoint(map_bounds_max_);
    Eigen::Vector3f closest_M = point_M;
    se::math::clamp(closest_M, map_bounds_min_M, map_bounds_max_M);
    return closest_M;
}

se::Node<MultiresOFusion::VoxelType>* OccupancyWorld::getNode(const Eigen::Vector3i& node_coord,
                                                              const int node_size) const
{
    return map_.fetchNode(node_coord, map_.sizeToDepth(node_size));
}

void OccupancyWorld::updateMapBounds()
{
    se::Node<MultiresOFusion::VoxelType>*
        nodeStack[se::Octree<MultiresOFusion::VoxelType>::max_voxel_depth * 8 + 1];
    size_t stack_idx = 0;

    se::Node<MultiresOFusion::VoxelType>* node = map_.root();

    map_bounds_min_ = Eigen::Vector3f(map_.size(), map_.size(), map_.size());
    map_bounds_max_ = Eigen::Vector3f(0, 0, 0);

    if (node) {
        se::Node<MultiresOFusion::VoxelType>* current;
        current = node;
        nodeStack[stack_idx++] = current;
        const int block_size = MultiresOFusion::VoxelBlockType::size_li;

        while (stack_idx != 0) {
            node = current;

            if (node->isBlock()) {
                MultiresOFusion::VoxelBlockType* block =
                    static_cast<MultiresOFusion::VoxelBlockType*>(node);
                const Eigen::Vector3i block_coord = block->coordinates();

                for (int i = 0; i < 3; i++) {
                    if (block_coord(i) < map_bounds_min_(i)) {
                        MultiresOFusion::VoxelBlockType* block =
                            map_.fetch(block_coord(0), block_coord(1), block_coord(2));
                        const Eigen::Vector3i blockCoord = block->coordinates();
                        int x, y, z;
                        int xlast = blockCoord(0) + block_size;
                        int ylast = blockCoord(1) + block_size;
                        int zlast = blockCoord(2) + block_size;
                        for (z = blockCoord(2); z < zlast; ++z) {
                            for (y = blockCoord(1); y < ylast; ++y) {
                                for (x = blockCoord(0); x < xlast; ++x) {
                                    MultiresOFusion::VoxelType::VoxelData value;
                                    const Eigen::Vector3i vox{x, y, z};
                                    value = block->data(Eigen::Vector3i(x, y, z),
                                                        block->current_scale());
                                    if (value.x != 0) {
                                        if (vox(i) < map_bounds_min_(i))
                                            map_bounds_min_(i) = vox(i);
                                    }
                                }
                            }
                        }
                    }
                }

                for (int i = 0; i < 3; i++) {
                    if ((block_coord(i) + block_size - 1) > map_bounds_min_(i)) {
                        MultiresOFusion::VoxelBlockType* block =
                            map_.fetch(block_coord(0), block_coord(1), block_coord(2));
                        const Eigen::Vector3i blockCoord = block->coordinates();
                        int x, y, z, blockSide;
                        blockSide = (int) MultiresOFusion::VoxelBlockType::size_li;
                        int xlast = blockCoord(0) + blockSide;
                        int ylast = blockCoord(1) + blockSide;
                        int zlast = blockCoord(2) + blockSide;
                        for (z = blockCoord(2); z < zlast; ++z) {
                            for (y = blockCoord(1); y < ylast; ++y) {
                                for (x = blockCoord(0); x < xlast; ++x) {
                                    MultiresOFusion::VoxelType::VoxelData value;
                                    const Eigen::Vector3i vox{x, y, z};
                                    value = block->data(Eigen::Vector3i(x, y, z),
                                                        block->current_scale());
                                    if (value.x != 0) {
                                        if (vox(i) > map_bounds_max_(i))
                                            map_bounds_max_(i) = vox(i);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            else {
                if (node->children_mask() == 0) {
                    const auto& data = node->data();
                    if (data.x != 0) {
                        const Eigen::Vector3f node_coord_min = node->coordinates().cast<float>();
                        const Eigen::Vector3f node_coord_max =
                            node_coord_min + Eigen::Vector3f::Constant(node->size() - 1);
                        map_bounds_min_ = map_bounds_min_.cwiseMin(node_coord_min);
                        map_bounds_max_ = map_bounds_max_.cwiseMax(node_coord_max);
                    }
                }
            }

            if (node->children_mask() == 0) {
                current = nodeStack[--stack_idx];
                continue;
            }

            for (int i = 0; i < 8; ++i) {
                se::Node<MultiresOFusion::VoxelType>* child = node->child(i);
                if (child != NULL) {
                    nodeStack[stack_idx++] = child;
                }
            }
            current = nodeStack[--stack_idx];
        }
    }
}
} // namespace ptp
