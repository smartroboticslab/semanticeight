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
OccupancyWorld::OccupancyWorld(std::shared_ptr<se::Octree<MultiresOFusion::VoxelType>> octree) :
        octree_(octree), res_(octree_->voxelDim())
{
    updateMapBounds();
}

std::shared_ptr<se::Octree<VoxelImpl::VoxelType>> OccupancyWorld::getMap() const
{
    return octree_;
}

float OccupancyWorld::getMapResolution() const
{
    return res_;
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
    map_bounds_min_m = map_bounds_min_ * res_;
    map_bounds_max_m = map_bounds_max_ * res_;
}

bool OccupancyWorld::inMapBounds(const Eigen::Vector3f& voxel_coord) const
{
    return (map_bounds_min_.x() <= voxel_coord.x() && map_bounds_max_.x() >= voxel_coord.x()
            && map_bounds_min_.y() <= voxel_coord.y() && map_bounds_max_.y() >= voxel_coord.y()
            && map_bounds_min_.z() <= voxel_coord.z() && map_bounds_max_.z() >= voxel_coord.z());
}

bool OccupancyWorld::inMapBoundsMeter(const Eigen::Vector3f& point_M) const
{
    return (
        map_bounds_min_.x() * res_ <= point_M.x() && map_bounds_max_.x() * res_ >= point_M.x()
        && map_bounds_min_.y() * res_ <= point_M.y() && map_bounds_max_.y() * res_ >= point_M.y()
        && map_bounds_min_.z() * res_ <= point_M.z() && map_bounds_max_.z() * res_ >= point_M.z());
}

bool OccupancyWorld::isFree(const Eigen::Vector3i& voxel_coord, const float threshold) const
{
    MultiresOFusion::VoxelType::VoxelData value;
    // TODO: change to getMax()
    octree_->get(voxel_coord.x(), voxel_coord.y(), voxel_coord.z(), value);
    if (value.x * value.y < threshold && value.observed == true) {
        return true;
    }
    return false;
}

bool OccupancyWorld::isFreeAtPoint(const Eigen::Vector3f& point_M, const float threshold) const
{
    const Eigen::Vector3i voxel_coord = (point_M / res_).cast<int>();
    MultiresOFusion::VoxelType::VoxelData value;
    // TODO: change to getMax()
    octree_->get(voxel_coord.x(), voxel_coord.y(), voxel_coord.z(), value);
    if (value.x * value.y < threshold && value.observed == true) {
        return true;
    }
    return false;
}

se::Node<MultiresOFusion::VoxelType>* OccupancyWorld::getNode(const Eigen::Vector3i& node_coord,
                                                              const int node_size) const
{
    const int node_depth = octree_->sizeToDepth(node_size);
    return octree_->fetchNode(node_coord.x(), node_coord.y(), node_coord.z(), node_depth);
}

void OccupancyWorld::updateMapBounds()
{
    se::Node<MultiresOFusion::VoxelType>*
        nodeStack[se::Octree<MultiresOFusion::VoxelType>::max_voxel_depth * 8 + 1];
    size_t stack_idx = 0;

    se::Node<MultiresOFusion::VoxelType>* node = octree_->root();

    Eigen::Vector3f map_bounds_min(octree_->size(), octree_->size(), octree_->size());
    Eigen::Vector3f map_bounds_max(0, 0, 0);

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
                    if (block_coord(i) < map_bounds_min(i)) {
                        MultiresOFusion::VoxelBlockType* block =
                            octree_->fetch(block_coord(0), block_coord(1), block_coord(2));
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
                                        if (vox(i) < map_bounds_min(i))
                                            map_bounds_min(i) = vox(i);
                                    }
                                }
                            }
                        }
                    }
                }

                for (int i = 0; i < 3; i++) {
                    if ((block_coord(i) + block_size - 1) > map_bounds_min(i)) {
                        MultiresOFusion::VoxelBlockType* block =
                            octree_->fetch(block_coord(0), block_coord(1), block_coord(2));
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
                                        if (vox(i) > map_bounds_max(i))
                                            map_bounds_max(i) = vox(i);
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
                        map_bounds_min = map_bounds_min.cwiseMin(node_coord_min);
                        map_bounds_max = map_bounds_max.cwiseMax(node_coord_max);
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

    map_bounds_min_ = map_bounds_min;
    map_bounds_max_ = map_bounds_max;
}
} // namespace ptp
