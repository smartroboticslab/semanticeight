/**
 * Motion Planning, Header.
 *
 * Copyright (C) 2017 Imperial College London.
 * Copyright (C) 2017 ETH Zürich.

 * @file Header.hpp
 *
 * @ingroup common
 *
 * @author Marius Grimm (marius.grimm93@web.de)
 * @date May 19, 2017
 */

#ifndef PTP_HEADER_HPP
#define PTP_HEADER_HPP

#include <chrono>
#include <iostream>

namespace ptp {

template<int TSize>
struct State {
    Eigen::Matrix<float, TSize, 1> segment_end;
    float segment_radius = -1;
};

/** Struct defining a Header with a timestamp and frame id. */
struct Header {
    std::chrono::nanoseconds time_nsec; ///> Timestamp in nanoseconds
    std::string frame_id;               ///> Frame id
};

} // namespace ptp

#endif //PTP_HEADER_HPP
