/**
 * Probabilistic Trajectory Planning, Map interface for supereight library of Emanuele Vespa.
 *
 * Copyright (C) 2018 Imperial College London.
 * Copyright (C) 2018 ETH Zürich.
 *
 * @file OccupancyWorld.hpp
 *
 *
 * @author Nils Funk
 * @date July 5, 2018
 */

#ifndef OCCUPANCYWORLD_HPP
#define OCCUPANCYWORLD_HPP

#include <Eigen/Dense>
#include <memory>
#include <ptp/Header.hpp>
#include <ptp/Path.hpp>
#include <ptp/common.hpp>
#include <se/node.hpp>
#include <se/node_iterator.hpp>
#include <se/octree.hpp>

#include "se/voxel_implementations.hpp"

namespace ptp {
class OccupancyWorld {
    public:
    OccupancyWorld(const se::Octree<VoxelImpl::VoxelType>& map);

    OccupancyWorld(const se::Octree<VoxelImpl::VoxelType>& map,
                   const Eigen::Vector3f& planning_bounds_min_m,
                   const Eigen::Vector3f& planning_bounds_max_m);

    const se::Octree<VoxelImpl::VoxelType>& getMap() const;

    void getMapBounds(Eigen::Vector3f& map_bounds_min_v, Eigen::Vector3f& map_bounds_max_v) const;
    void getMapBoundsMeter(Eigen::Vector3f& map_bounds_min_m,
                           Eigen::Vector3f& map_bounds_max_m) const;

    bool inMapBounds(const Eigen::Vector3f& voxel_coord) const;
    bool inMapBoundsMeter(const Eigen::Vector3f& point_M) const;

    bool isFree(const Eigen::Vector3i& voxel_coord, const float threshold = 0) const;
    bool isFreeAtPoint(const Eigen::Vector3f& point_M, const float threshold = 0) const;

    Eigen::Vector3f closestPointMeter(const Eigen::Vector3f& point_M) const;

    se::Node<MultiresOFusion::VoxelType>* getNode(const Eigen::Vector3i& node_coord,
                                                  const int node_size) const;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    private:
    const se::Octree<MultiresOFusion::VoxelType>& map_;
    Eigen::Vector3f map_bounds_min_;
    Eigen::Vector3f map_bounds_max_;

    void updateMapBounds();
};
} // namespace ptp

#endif //OCCUPANCYWORLD_HPP
