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

#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/base/ScopedState.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/PathSimplifier.h>
#include <ompl/geometric/planners/bitstar/BITstar.h>
#include <ompl/geometric/planners/rrt/InformedRRTstar.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <ompl/base/Planner.h>
#include <ompl/base/StateSpace.h>
#include <ompl/geometric/PathGeometric.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/control/PathControl.h>
#include <ompl/util/Time.h>
#include <ptp/OmplToEigen.hpp>
#include <ptp/Path.hpp>
#include <ptp/OccupancyWorld.hpp>
#include <ptp/common.hpp>
#include <ptp/MotionValidatorOccupancyDense.hpp>
#include <ptp/ProbCollisionChecker.hpp>
#include <ptp/PlanningParameter.hpp>

namespace ob = ompl::base;
namespace og = ompl::geometric;
namespace oc = ompl::control;

namespace ptp {
  class SafeFlightCorridorGenerator {
  public:
    typedef std::shared_ptr<SafeFlightCorridorGenerator> Ptr;

    /**
     * @param [in] ow Map
     * @param ss_ SimplePlanner: Create the set of classes typically needed to solve a geometric problem
     *              StateSpacePtr:  Representation of a space in which planning can be performed. Topology specific sampling, interpolation and distance are defined.
     *              RealVectorStateSpace: A state space representing R^n. The distance function is the L2 norm.
     */
    SafeFlightCorridorGenerator(const OccupancyWorld&   ow,
                                ProbCollisionChecker&   pcc,
                                const PlanningParameter pp); // QUESTION: Why use const OccupancyWorld::Ptr&
    SafeFlightCorridorGenerator(); // QUESTION: Why use const OccupancyWorld::Ptr&

    /**
     * Set up the planner.
     * @param [in] start Start position for path planning. [m]
     * @param [in] goal Goal position for path planning. [m]
     */
    bool setupPlanner(const Eigen::Vector3f& start_m,
                      const Eigen::Vector3f& goal_m);

    Path<kDim>::Ptr getPathNotSimplified() {return path_not_simplified_;}

      /**
     * Plan the global path.
     * @param [in] start Start position for path planning. [m]
     * @param [in] goal Goal position for path planning. [m]
     * @return True if straight line planning was successful.
     *TODO: Instead of Eigen::Vector3f use a trajectory point type/message
     */
    bool planPath(const Eigen::Vector3f& start_m,
                  const Eigen::Vector3f& goal_m);

    bool start_end_occupied() {return start_end_occupied_;};
    bool ompl_failed() {return ompl_failed_;};

    void prunePath(og::PathGeometric& path);
    void simplifyPath(ompl::geometric::PathGeometric& path);

    Path<kDim>::Ptr getPath() { return path_; };

  private:
    /**
     * Set the space boundaries of the ompl planner from the map boundaries.
     * @param [in] min_boundary Lower boundaries in x, y and z of the map.
     * @param [in] max_boundary Upper boundaries in x, y and z of the map.
     * @param [out] space The ompl space to set the boundaries.
     */

    void setSpaceBoundaries(const Eigen::Vector3f& min_boundary,
                            const Eigen::Vector3f& max_boundary,
                            ompl::base::StateSpacePtr space);

    void reduceToControlPointCorridorRadius(Path<kDim>::Ptr path_m);

    ompl::base::StateSpacePtr space_ = nullptr;
    ompl::base::SpaceInformationPtr si_ = nullptr;
    ompl::base::ProblemDefinitionPtr pdef_ = nullptr;
    ompl::base::PlannerPtr optimizingPlanner_ = nullptr;

    Path<kDim>::Ptr path_ = nullptr;
    Path<kDim>::Ptr path_not_simplified_ = nullptr;

    OccupancyWorld ow_;
    ProbCollisionChecker pcc_;
    PlanningParameter pp_;
    float min_flight_corridor_radius_;
    float robot_radius_;
    float flight_corridor_radius_reduction_;
    float solving_time_;
    bool start_end_occupied_ = false;
    bool ompl_failed_ = false;
  };

}



#endif //PTP_SAFEFLIGHTCORRIDORGENERATOR_H

