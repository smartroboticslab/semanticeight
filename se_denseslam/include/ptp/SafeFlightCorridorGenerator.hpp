/**
 * Probabilistic Trajectory Safe Flight Corridor Generator.
 *
 * Copyright (C) 2018 Imperial College London.
 * Copyright (C) 2018 ETH Zürich.
 *
 * @todo LICENSE
 *
 *
 * @file file PlanningParameter.hpp
 * @author Nils Funk
 * @date Juli, 2018
 */

#ifndef PTP_SAFEFLIGHTCORRIDORGENERATOR_H
#define PTP_SAFEFLIGHTCORRIDORGENERATOR_H

#include <ompl/base/Planner.h>
#include <ompl/base/ScopedState.h>
#include <ompl/base/StateSpace.h>
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/control/PathControl.h>
#include <ompl/geometric/PathGeometric.h>
#include <ompl/geometric/PathSimplifier.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/bitstar/BITstar.h>
#include <ompl/geometric/planners/rrt/InformedRRTstar.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <ompl/util/Time.h>
#include <ptp/MotionValidatorOccupancyDense.hpp>
#include <ptp/OccupancyWorld.hpp>
#include <ptp/OmplToEigen.hpp>
#include <ptp/Path.hpp>
#include <ptp/PlanningParameter.hpp>
#include <ptp/ProbCollisionChecker.hpp>
#include <ptp/common.hpp>

namespace ob = ompl::base;
namespace og = ompl::geometric;
namespace oc = ompl::control;

namespace ptp {
enum class PlanningResult {
    Success,
    Partial,
    Failed,
    StartOccupied,
    GoalOccupied,
};

std::string to_string(PlanningResult result);

class SafeFlightCorridorGenerator {
    public:
    SafeFlightCorridorGenerator(const std::shared_ptr<const se::Octree<VoxelImpl::VoxelType>> map,
                                const PlanningParameter& pp);
    /**
     * Plan the global path.
     * @param [in] start Start position for path planning. [m]
     * @param [in] goal Goal position for path planning. [m]
     * @return True if straight line planning was successful.
     */
    PlanningResult planPath(const Eigen::Vector3f& start_m, const Eigen::Vector3f& goal_m);

    /** Return the planned path. Its first vertex is always the start position.
     */
    Path<kDim>::Ptr getPath() const;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    private:
    void updateSpaceBoundaries();
    void setupPlanner(const Eigen::Vector3f& start_m, const Eigen::Vector3f& goal_m);
    void prunePath(og::PathGeometric& path);
    void simplifyPath(ompl::geometric::PathGeometric& path);

    const std::shared_ptr<const se::Octree<VoxelImpl::VoxelType>> map_;
    const OccupancyWorld ow_;
    const ProbCollisionChecker pcc_;

    const float min_flight_corridor_radius_;
    const float robot_radius_;
    const float solving_time_;

    ompl::base::StateSpacePtr space_;
    ompl::base::SpaceInformationPtr si_;
    ompl::base::OptimizationObjectivePtr objective_;
    ompl::base::ProblemDefinitionPtr pdef_;
    ompl::base::PlannerPtr optimizingPlanner_;

    Path<kDim>::Ptr path_;
};

} // namespace ptp



#endif //PTP_SAFEFLIGHTCORRIDORGENERATOR_H
