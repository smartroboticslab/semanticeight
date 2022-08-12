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

std::string to_string(PlanningResult result)
{
    switch (result) {
    case PlanningResult::Success:
        return "Success";
    case PlanningResult::Partial:
        return "Partial path planned";
    case PlanningResult::Failed:
        return "Planning failed";
    case PlanningResult::StartOccupied:
        return "Start position occupied";
    case PlanningResult::GoalOccupied:
        return "Goal position occupied";
    default:
        return "Unknown";
    }
}

SafeFlightCorridorGenerator::SafeFlightCorridorGenerator(
    const std::shared_ptr<const se::Octree<VoxelImpl::VoxelType>> map,
    const PlanningParameter& pp) :
        map_(map),
        ow_(*map, pp.sampling_min_M_, pp.sampling_max_M_),
        pcc_(ow_, pp),
        min_flight_corridor_radius_(pp.robot_radius_),
        robot_radius_(pp.robot_radius_),
        solving_time_(pp.solving_time_),
        space_(new ob::RealVectorStateSpace(kDim)),
        path_(new Path<kDim>())
{
    updateSpaceBoundaries();
    si_ = std::shared_ptr<ob::SpaceInformation>(new ob::SpaceInformation(space_));
    // Set motion validity checking for this space (collision checking)
    auto motion_validator(std::shared_ptr<MotionValidatorOccupancyDense>(
        new MotionValidatorOccupancyDense(si_, pcc_, min_flight_corridor_radius_)));
    si_->setMotionValidator(motion_validator);
    si_->setup();
}



PlanningResult SafeFlightCorridorGenerator::planPath(const Eigen::Vector3f& start_point_M,
                                                     const Eigen::Vector3f& goal_point_M)
{
    const bool start_inside = ow_.inMapBoundsMeter(start_point_M);
    const Eigen::Vector3f start_M =
        start_inside ? start_point_M : ow_.closestPointMeter(start_point_M);

    // Don't test the goal point occupancy to allow planning partial paths.
    //if(!pcc_.checkSphere(goal_point_M, min_flight_corridor_radius_)) {
    //  return PlanningResult::GoalOccupied;
    //}

    setupPlanner(start_M, goal_point_M);
    path_->states.clear();
    ob::PlannerStatus solved = optimizingPlanner_->solve(solving_time_);

    if (solved) {
        // Get non-simplified path and convert to Eigen
        ompl::geometric::SimpleSetup ss(si_);
        ss.getProblemDefinition()->addSolutionPath(pdef_->getSolutionPath());
        og::PathGeometric path = ss.getSolutionPath();
        // Simplify path
        prunePath(path);
        // Convert final path to Eigen
        OmplToEigen::convertPath(path, path_, min_flight_corridor_radius_);
        if (!start_inside) {
            path_->states.emplace(path_->states.begin(),
                                  State<kDim>{start_point_M, min_flight_corridor_radius_});
        }

        if (pdef_->hasApproximateSolution()) {
            // TODO SEM Consider checking that the approximate solution is within some threshold of
            // the goal, otherwise return PlanningResult::Failed.
            return PlanningResult::Partial;
        }
        else {
            return PlanningResult::Success;
        }
    }
    else {
        return PlanningResult::Failed;
    }
}



Path<kDim>::Ptr SafeFlightCorridorGenerator::getPath() const
{
    return path_;
}



void SafeFlightCorridorGenerator::updateSpaceBoundaries()
{
    Eigen::Vector3f min_boundary, max_boundary;
    ow_.getMapBoundsMeter(min_boundary, max_boundary);
    ob::RealVectorBounds bounds(kDim);
    bounds.setLow(0, min_boundary.x());
    bounds.setLow(1, min_boundary.y());
    bounds.setLow(2, min_boundary.z());
    bounds.setHigh(0, max_boundary.x());
    bounds.setHigh(1, max_boundary.y());
    bounds.setHigh(2, max_boundary.z());
    space_->as<ob::RealVectorStateSpace>()->setBounds(bounds);
}



void SafeFlightCorridorGenerator::setupPlanner(const Eigen::Vector3f& start_point_M,
                                               const Eigen::Vector3f& goal_point_M)
{
    /// Set the start and goal states
    ob::ScopedState<ob::RealVectorStateSpace> ompl_start(space_), ompl_goal(space_);
    OmplToEigen::convertState(start_point_M, &ompl_start);
    OmplToEigen::convertState(goal_point_M, &ompl_goal);

    pdef_ = std::shared_ptr<ob::ProblemDefinition>(new ob::ProblemDefinition(si_));
    pdef_->setStartAndGoalStates(ompl_start, ompl_goal, 1.5 * robot_radius_);

    // Optimise for path length.
    objective_ = ob::OptimizationObjectivePtr(new ob::PathLengthOptimizationObjective(si_));
    // Set a high cost (path length) threshold so that planning essentially stops on the first path
    // found.
    objective_->setCostThreshold(ob::Cost(1000.0));
    pdef_->setOptimizationObjective(objective_);

    optimizingPlanner_ = std::shared_ptr<og::InformedRRTstar>(new og::InformedRRTstar(si_));
    optimizingPlanner_->setProblemDefinition(pdef_);
    optimizingPlanner_->setup();
}



void SafeFlightCorridorGenerator::prunePath(og::PathGeometric& path)
{
    if (pdef_) {
        const ob::PathPtr& p = pdef_->getSolutionPath();
        if (p) {
            simplifyPath(path);
            return;
        }
    }
}



void SafeFlightCorridorGenerator::simplifyPath(ompl::geometric::PathGeometric& path)
{
    if (path.getStateCount() < 3) {
        return;
    }

    og::PathSimplifier simplifier(si_);

    // Termination condition
    const float max_time = 0.5; // TODO: parameterize
    ob::PlannerTerminationCondition ptc = ob::timedPlannerTerminationCondition(max_time);

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
} // namespace ptp
