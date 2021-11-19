/**
 * Motion Planning, Convert Ompl path to Eigen path.
 *
 * Copyright (C) 2017 Imperial College London.
 * Copyright (C) 2017 ETH Zürich.
 *
 * @file OmplToEigen.hpp
 *
 * @ingroup common
 *
 * @author Marius Grimm (marius.grimm93@web.de)
 * @date May 22, 2017
 */

#ifndef PTP_CONVERTOMPLTOEIGEN_HPP
#define PTP_CONVERTOMPLTOEIGEN_HPP

#include <ompl/base/State.h>
#include <ompl/base/StateSpace.h>
#include <ompl/geometric/PathGeometric.h>
#include <ptp/Path.hpp>
#include <ptp/common.hpp>

namespace ptp {

/**
 * Class for Ompl to Eigen conversions.
 */
class OmplToEigen {
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /** Constructor. */
    OmplToEigen() = default;

    /** Delete copy constructor. */
    OmplToEigen(const OmplToEigen&) = delete;

    /** Use default virtual destructor. */
    virtual ~OmplToEigen() = default;

    /**
   * Convert Ompl path to Eigen path.
   * @param [in] ompl_path Path of type ompl::geometric::PathGeometric.
   * @param [out] eigen_path Path in Eigen format.
   *
   * @note Header of Eigen path message not set in this method.
   */
    static void
    convertPath(ompl::geometric::PathGeometric& ompl_path, Path<3>::Ptr eigen_path, float radius_m)
    {
        std::vector<ompl::base::State*>& states = ompl_path.getStates();

        for (ompl::base::State* state : states) {
            State<kDim> state_M;
            Eigen::Vector3f point_M(
                state->as<ompl::base::RealVectorStateSpace::StateType>()->values[0],
                state->as<ompl::base::RealVectorStateSpace::StateType>()->values[1],
                state->as<ompl::base::RealVectorStateSpace::StateType>()->values[2]);

            state_M.segment_end = point_M;
            state_M.segment_radius = radius_m;

            // Save the 3D path in output Eigen format
            eigen_path->states.push_back(state_M);
        }
    };

    /**
   * Convert Ompl state to Eigen Vector.
   * @param [in] state Position (state) in ompl::base::State type.
   * @return The converted state as Eigen::Vector3d type.
   *
   * @note Header of Eigen path message not set in this method.
   */
    static Eigen::Vector3f convertState(const ompl::base::State& state)
    {
        const Eigen::Vector3f eigen_point(
            state.as<ompl::base::RealVectorStateSpace::StateType>()->values[0],
            state.as<ompl::base::RealVectorStateSpace::StateType>()->values[1],
            state.as<ompl::base::RealVectorStateSpace::StateType>()->values[2]);

        return eigen_point;
    };

    /**
   * Convert Eigen Vector to ompl scoped state.
   * @param [in] state Position (state) as Eigen::Vector type.
   * @param [out] scoped_state Converted state as ompl::base::ScopedState type.
   */
    static void
    convertState(const Eigen::Vector3f& state,
                 ompl::base::ScopedState<ompl::base::RealVectorStateSpace>* scoped_state)
    {
        (*scoped_state)->values[0] = state.x();
        (*scoped_state)->values[1] = state.y();
        (*scoped_state)->values[2] = state.z();
    };
};

} // namespace ptp

#endif //PTP_CONVERTOMPLTOEIGEN_HPP
