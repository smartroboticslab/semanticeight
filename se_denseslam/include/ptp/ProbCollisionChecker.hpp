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
#include "se/voxel_implementations.hpp"
#include <ptp/OccupancyWorld.hpp>
#include <ptp/PlanningParameter.hpp>
#include <math.h>
#include <ptp/common.hpp>
#include <ptp/Header.hpp>
#include <ptp/Path.hpp>

namespace ptp {
  class ProbCollisionChecker {
  public:
    typedef std::shared_ptr<ProbCollisionChecker> Ptr;

    ProbCollisionChecker(ptp::OccupancyWorld& ow, PlanningParameter pp);

    bool checkData(const MultiresOFusion::VoxelType::VoxelData& data, bool finest = false);

    bool checkPoint(const Eigen::Vector3f& point_M);

    bool checkLine(const Eigen::Vector3f& start_point_M,
                   const Eigen::Vector3f& connection_M,
                   const int              num_subpos);

    bool checkSphereSkeleton(const Eigen::Vector3f& point_M,
                             const float            radius_m);

    bool checkCylinderSkeleton(const Eigen::Vector3f& start_point_M,
                               const Eigen::Vector3f& end_point_M,
                               const float            radius_m);

    bool checkCorridorSkeleton(const Eigen::Vector3f& start_point_M,
                               const Eigen::Vector3f& end_point_M,
                               const float            radius_m);

    bool checkNode(const Eigen::Vector3i& node_corner,
                   const int              node_size,
                   bool&                  abort);

    bool checkNode(const Eigen::Vector3i& node_corner,
                   const int              node_size);

    bool checkInSphereFree(const Eigen::Vector3i& node_corner_min,
                           const int              node_size,
                           const Eigen::Vector3f& position_m,
                           const float            radius_m);

    bool checkBlockInSphereFree(const se::VoxelBlockSingleMax<MultiresOFusion::VoxelType>*
                                                       block,
                                const Eigen::Vector3i& parent_coord,
                                const int              node_size,
                                const Eigen::Vector3f& pointM,
                                const float            radius_m);

    bool checkNodeInSphereFree(const se::Node<MultiresOFusion::VoxelType>*
                                                      node,
                               const Eigen::Vector3i& parent_coord,
                               const int              node_size,
                               const Eigen::Vector3f& point_M,
                               const float            radius_m);

    bool checkSegmentFlightCorridor(const Eigen::Vector3f& start_point_M,
                                    const Eigen::Vector3f& end_point_M,
                                    const float            radius_m);

    bool checkInCylinderFree(const Eigen::Vector3i& node_corner_min,
                             const int              node_size,
                             const Eigen::Vector3f& start_point_M,
                             const Eigen::Vector3f& end_point_M,
                             const Eigen::Vector3f& axis,
                             const float            radius_m);

    bool checkNodeInCylinderFree(const se::Node<MultiresOFusion::VoxelType>*
                                                        parent_node,
                                 const Eigen::Vector3i& parent_coord,
                                 const int              node_size,
                                 const Eigen::Vector3f& start_point_M,
                                 const Eigen::Vector3f& end_point_M,
                                 const Eigen::Vector3f& axis,
                                 const float            radius_m);


    bool checkBlockInCylinderFree(const se::VoxelBlockSingleMax<MultiresOFusion::VoxelType>*
                                                                   block,
                                  const Eigen::Vector3i&           parent_coord,
                                  const int                        node_size,
                                  const Eigen::Vector3f&           start_point_M,
                                  const Eigen::Vector3f&           end_point_M,
                                  const Eigen::Vector3f&           axis,
                                  const float                      radius_m);

    bool checkCornerInCylinder(const Eigen::Vector3i& centre_v,
                               const Eigen::Vector3f& start_point_M,
                               const Eigen::Vector3f& end_point_M,
                               const Eigen::Vector3f& axis,
                               const float            radius_m);

    bool checkCenterInCylinder(const Eigen::Vector3f& centre_v,
                               const int              node_size,
                               const Eigen::Vector3f& start_point_M,
                               const Eigen::Vector3f& end_point_M,
                               const Eigen::Vector3f& axis,
                               const float            radius_m);

    bool checkSphere(const Eigen::Vector3f& position_m,
                     const float            radius_m);

  private:
    float threshold_;
    ptp::OccupancyWorld ow_;
    std::shared_ptr<se::Octree<MultiresOFusion::VoxelType>> octree_ = nullptr;
    PlanningParameter pp_;
    float res_;
  };

} // namespace ptp

#endif //PTP_PROBCOLLISIONCHECKER_HPP
