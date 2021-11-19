/**
 * Motion Planning, Common important parameter declarations.
 *
 * Copyright (C) 2017 Imperial College London.
 * Copyright (C) 2017 ETH Zürich.
 *
 * @file common.hpp
 *
 * @ingroup common
 *
 * @author Marius Grimm (marius.grimm93@web.de)
 * @date August 28, 2017
 */

#ifndef PTP_COMMON_HPP
#define PTP_COMMON_HPP

//#include "mav_tube_trajectory_generation/motion_defines.h"

namespace ptp {

// Default parameter
static constexpr size_t kDim = 3;     /// Dimension for straight line planning
static constexpr size_t kDimTraj = 3; /// Dimension for trajectory optimization

} // namespace ptp

#endif //PTP_COMMON_HPP
