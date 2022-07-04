/**
 * Probabilistic Trajectory Planning, Probabilistic Collision Checker.
 *
 * Copyright (C) 2018 Imperial College London.
 * Copyright (C) 2018 ETH Zürich.
 *
 * @file OccupancyWorld.hpp
 *
 * @author Nils Funk
 * @date July 11, 2018
 */


#ifndef PTP_PROBCOLLISIONCHECKER_HPP
#define PTP_PROBCOLLISIONCHECKER_HPP

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <math.h>
#include <ptp/Header.hpp>
#include <ptp/OccupancyWorld.hpp>
#include <ptp/Path.hpp>
#include <ptp/PlanningParameter.hpp>
#include <ptp/common.hpp>

#include "se/voxel_implementations.hpp"

namespace ptp {
class ProbCollisionChecker {
    public:

    ProbCollisionChecker(const ptp::OccupancyWorld& ow, const PlanningParameter& pp);

    bool checkData(const MultiresOFusion::VoxelType::VoxelData& data,
                   const bool finest = false) const;

    bool checkPoint(const Eigen::Vector3f& point_M) const;

    bool checkLine(const Eigen::Vector3f& start_point_M,
                   const Eigen::Vector3f& connection_M,
                   const int num_subpos) const;

    bool checkSphereSkeleton(const Eigen::Vector3f& point_M, const float radius_m) const;

    bool checkCylinderSkeleton(const Eigen::Vector3f& start_point_M,
                               const Eigen::Vector3f& end_point_M,
                               const float radius_m) const;

    bool checkCorridorSkeleton(const Eigen::Vector3f& start_point_M,
                               const Eigen::Vector3f& end_point_M,
                               const float radius_m) const;

    bool checkNode(const Eigen::Vector3i& node_corner, const int node_size, bool& abort) const;

    bool checkNode(const Eigen::Vector3i& node_corner, const int node_size) const;

    bool checkInSphereFree(const Eigen::Vector3i& node_corner_min,
                           const int node_size,
                           const Eigen::Vector3f& position_m,
                           const float radius_m) const;

    bool checkBlockInSphereFree(const se::VoxelBlockSingleMax<MultiresOFusion::VoxelType>* block,
                                const Eigen::Vector3i& parent_coord,
                                const int node_size,
                                const Eigen::Vector3f& pointM,
                                const float radius_m) const;

    bool checkNodeInSphereFree(const se::Node<MultiresOFusion::VoxelType>* node,
                               const Eigen::Vector3i& parent_coord,
                               const int node_size,
                               const Eigen::Vector3f& point_M,
                               const float radius_m) const;

    bool checkSegmentFlightCorridor(const Eigen::Vector3f& start_point_M,
                                    const Eigen::Vector3f& end_point_M,
                                    const float radius_m) const;

    bool checkInCylinderFree(const Eigen::Vector3i& node_corner_min,
                             const int node_size,
                             const Eigen::Vector3f& start_point_M,
                             const Eigen::Vector3f& end_point_M,
                             const Eigen::Vector3f& axis,
                             const float radius_m) const;

    bool checkNodeInCylinderFree(const se::Node<MultiresOFusion::VoxelType>* parent_node,
                                 const Eigen::Vector3i& parent_coord,
                                 const int node_size,
                                 const Eigen::Vector3f& start_point_M,
                                 const Eigen::Vector3f& end_point_M,
                                 const Eigen::Vector3f& axis,
                                 const float radius_m) const;


    bool checkBlockInCylinderFree(const se::VoxelBlockSingleMax<MultiresOFusion::VoxelType>* block,
                                  const Eigen::Vector3i& parent_coord,
                                  const int node_size,
                                  const Eigen::Vector3f& start_point_M,
                                  const Eigen::Vector3f& end_point_M,
                                  const Eigen::Vector3f& axis,
                                  const float radius_m) const;

    bool checkCornerInCylinder(const Eigen::Vector3i& centre_v,
                               const Eigen::Vector3f& start_point_M,
                               const Eigen::Vector3f& end_point_M,
                               const Eigen::Vector3f& axis,
                               const float radius_m) const;

    bool checkCenterInCylinder(const Eigen::Vector3f& centre_v,
                               const int node_size,
                               const Eigen::Vector3f& start_point_M,
                               const Eigen::Vector3f& end_point_M,
                               const Eigen::Vector3f& axis,
                               const float radius_m) const;

    bool checkSphere(const Eigen::Vector3f& position_m, const float radius_m) const;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    private:
    const ptp::OccupancyWorld& ow_;
    const PlanningParameter& pp_;
    const float res_;
    static constexpr float free_threshold_ = -6.0f;
};

} // namespace ptp

#endif //PTP_PROBCOLLISIONCHECKER_HPP
