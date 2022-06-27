/**
 * Probabilistic Trajectory Planning Parameter.
 *
 * Copyright (C) 2018 Imperial College London.
 * Copyright (C) 2018 ETH Zürich.
 *
 * @todo LICENSE
 *
 * @original file GlobalPlanningParameter.hpp
 * @original autor Marius Grimm
 * @original data May 26, 2017
 *
 * @file file PlanningParameter.hpp
 * @author Nils Funk
 * @date August, 2018
 */

#ifndef PTP_PLANNINGPARAMETER_HPP
#define PTP_PLANNINGPARAMETER_HPP

#include <eigen3/Eigen/Core>

namespace ptp {

/** Struct which hold the parameters for global path planning. */
struct PlanningParameter {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /** Default Constructor. */
    PlanningParameter() :
            start_t_MB_(Eigen::Vector3f::Zero()),
            goal_t_MB_(Eigen::Vector3f::Zero()),
            robot_radius_(0.15),
            safety_radius_(0),
            min_control_point_radius_(0.1),
            skeleton_sample_precision_(0.05),
            solving_time_(0.0)
    {
    }

    /**
   * Constructor.
   * @param [in] start Start position for path planning. [m]
   * @param [in] goal Goal position for path planning. [m]
   * @param [in] global_method Global planning method used.
   * @param [in] robot_radius Cubic bounding box size of UAV + safety margin. [m]
   * @param [in] solving_time Solving time to find a path. [s]
   */
    PlanningParameter(const Eigen::Vector3f& start_t_MB,
                      const Eigen::Vector3f& goal_t_MB,
                      const float& robot_radius,
                      const float& safety_radius,
                      const float& min_control_point_radius,
                      const float& skeleton_sample_precision,
                      const float& solving_time) :
            start_t_MB_(start_t_MB),
            goal_t_MB_(goal_t_MB),
            robot_radius_(robot_radius),
            safety_radius_(safety_radius),
            min_control_point_radius_(min_control_point_radius),
            skeleton_sample_precision_(skeleton_sample_precision),
            solving_time_(solving_time)
    {
    }

    Eigen::Vector3f start_t_MB_; ///> Start position for path planning
    Eigen::Vector3f goal_t_MB_;  ///> Goal position for path planning
    float robot_radius_;         ///> Robot bounding sphere radius
    float safety_radius_;        ///> Added to robot_radius_ for safety
    float min_control_point_radius_;
    float skeleton_sample_precision_;
    float solving_time_; ///> Solving time to find a path
};

} // namespace ptp

#endif //PTP_PLANNINGPARAMETER_HPP
