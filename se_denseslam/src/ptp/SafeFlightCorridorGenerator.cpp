/**
 * Probabilistic Trajectory Safe Flight Corridor Generator.
 *
 * Copyright (C) 2018 Imperial College London.
 * Copyright (C) 2018 ETH Zürich.
 *
 * @todo LICENSE
 *
 * @file file SafeFlightCorridorGenerator.cpp
 * @author Nils Funk
 * @date June, 2018
 */

#include "ptp/SafeFlightCorridorGenerator.hpp"

namespace ptp {

SafeFlightCorridorGenerator::SafeFlightCorridorGenerator(
    std::shared_ptr<se::Octree<VoxelImpl::VoxelType>> map,
    const PlanningParameter pp) :
        ow_(map), pcc_(ow_, pp), pp_(pp), solving_time_(pp.solving_time_)
{
    min_flight_corridor_radius_ =
        pp.robot_radius_ + pp.safety_radius_ + pp.min_control_point_radius_;
    robot_radius_ = pp.robot_radius_;
    flight_corridor_radius_reduction_ = pp.robot_radius_ + pp.safety_radius_;
    path_ = std::shared_ptr<Path<kDim>>(new Path<kDim>());
    path_not_simplified_ = std::shared_ptr<Path<kDim>>(new Path<kDim>());
}

bool SafeFlightCorridorGenerator::setupPlanner(const Eigen::Vector3f& start_point_M,
                                               const Eigen::Vector3f& goal_point_M)
{
    if (!pcc_.checkSphere(start_point_M, min_flight_corridor_radius_)) { ///< Verify start is free.
        start_end_occupied_ = true;
        return false;
    }

    // Don't test the goal point occupancy to allow planning partial paths.
    //if(!pcc_.checkSphere(goal_point_M, min_flight_corridor_radius_)) { ///< Verify goal is free.
    //  start_end_occupied_ = true;
    //  return false;
    //}

    Eigen::Vector3f min_boundary, max_boundary;
    ow_.getMapBoundsMeter(
        min_boundary,
        max_boundary); ///< Update min/max map boundary based on observed supereight octree.
    space_ = std::shared_ptr<ob::RealVectorStateSpace>(new ob::RealVectorStateSpace(kDim));
    setSpaceBoundaries(min_boundary, max_boundary, space_);

    si_ = std::shared_ptr<ob::SpaceInformation>(new ob::SpaceInformation(space_));
    // Set motion validity checking for this space (collision checking)
    auto motion_validator(std::shared_ptr<MotionValidatorOccupancyDense>(
        new MotionValidatorOccupancyDense(si_, &pcc_, min_flight_corridor_radius_)));
    si_->setMotionValidator(motion_validator);
    si_->setup();

    /// Set the start and goal states
    ob::ScopedState<ob::RealVectorStateSpace> ompl_start(space_), ompl_goal(space_);

    OmplToEigen::convertState(start_point_M, &ompl_start);
    OmplToEigen::convertState(goal_point_M, &ompl_goal);

    pdef_ = std::shared_ptr<ob::ProblemDefinition>(new ob::ProblemDefinition(si_));
    pdef_->setStartAndGoalStates(ompl_start, ompl_goal);

    /// Optimise for path length.
    pdef_->setOptimizationObjective(
        ob::OptimizationObjectivePtr(new ob::PathLengthOptimizationObjective(si_)));

    optimizingPlanner_ = std::shared_ptr<og::InformedRRTstar>(new og::InformedRRTstar(si_));
    optimizingPlanner_->setProblemDefinition(pdef_);
    optimizingPlanner_->setup();

    return true;
}

PlanningResult SafeFlightCorridorGenerator::planPath(const Eigen::Vector3f& start_point_M,
                                                     const Eigen::Vector3f& goal_point_M)
{
    // Setup the ompl planner
    if (!setupPlanner(start_point_M, goal_point_M)) {
        return PlanningResult::Failed;
    }

    path_->states.clear();
    path_not_simplified_->states.clear();
    // Attempt to solve the problem within x seconds of planning time
    ob::PlannerStatus solved = optimizingPlanner_->solve(
        solving_time_); ///< Limit time to solve by solving_time_ in the config file.

    if (solved) {
        // Get non-simplified path and convert to Eigen
        ompl::geometric::SimpleSetup ss(si_);
        ss.getProblemDefinition()->addSolutionPath(pdef_->getSolutionPath());
        og::PathGeometric path = ss.getSolutionPath();
        // Simplify path
        prunePath(path);
        // Convert final path to Eigen
        OmplToEigen::convertPath(path, path_, min_flight_corridor_radius_);
        reduceToControlPointCorridorRadius(path_);

        if (!pdef_->hasApproximateSolution()) {
            return PlanningResult::OK;
        }
        else {
            // TODO SEM check that the approximate solution is within some threshold of the goal,
            // otherwise return Failed
            return PlanningResult::Partial;
        }
    }
    else {
        ompl_failed_ = true;
        return PlanningResult::Failed;
    }
}

void SafeFlightCorridorGenerator::reduceToControlPointCorridorRadius(Path<kDim>::Ptr path_m)
{
    for (std::vector<State<kDim>>::iterator it_i = path_m->states.begin();
         it_i != path_m->states.end();
         ++it_i) {
        (*it_i).segment_radius = (*it_i).segment_radius - flight_corridor_radius_reduction_;
    }
}

void SafeFlightCorridorGenerator::prunePath(og::PathGeometric& path)
{
    if (pdef_) {
        const ob::PathPtr& p = pdef_->getSolutionPath();
        if (p) {
            ompl::time::point start = ompl::time::now();
            std::size_t numStates = path.getStateCount();

            // Simplify
            simplifyPath(path);

            float simplifyTime = ompl::time::seconds(ompl::time::now() - start);
            OMPL_INFORM(
                "SimpleSetup: Path simplification took %f seconds and "
                "changed from %d to %d states",
                simplifyTime,
                numStates,
                path.getStateCount());
            return;
        }
    }
    OMPL_WARN("No solution to simplify");
}

void SafeFlightCorridorGenerator::simplifyPath(ompl::geometric::PathGeometric& path)
{
    og::PathSimplifier simplifier(si_);

    // Termination condition
    const float max_time = 0.5; // TODO: parameterize
    ob::PlannerTerminationCondition ptc = ob::timedPlannerTerminationCondition(max_time);

    if (path.getStateCount() < 3) {
        return;
    }

    // Try a randomized step of connecting vertices
    bool tryMore = false;
    if (ptc == false) {
        tryMore = simplifier.reduceVertices(path);
    }

    // Try to collapse close-by vertices
    if (ptc == false) {
        simplifier.collapseCloseVertices(path);
    }

    // Try to reduce vertices some more, if there is any point in doing so
    int times = 0;
    while (tryMore && ptc == false && ++times <= 5) {
        tryMore = simplifier.reduceVertices(path);
    }
}

void SafeFlightCorridorGenerator::setSpaceBoundaries(const Eigen::Vector3f& min_boundary,
                                                     const Eigen::Vector3f& max_boundary,
                                                     ob::StateSpacePtr space)
{
    ob::RealVectorBounds bounds(kDim);

    bounds.setLow(0, min_boundary.x());
    bounds.setLow(1, min_boundary.y());
    bounds.setLow(2, min_boundary.z());
    bounds.setHigh(0, max_boundary.x());
    bounds.setHigh(1, max_boundary.y());
    bounds.setHigh(2, max_boundary.z());
    space->as<ob::RealVectorStateSpace>()->setBounds(bounds);
}

} // namespace ptp
