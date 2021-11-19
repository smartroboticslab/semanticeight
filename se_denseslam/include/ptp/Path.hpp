/**
 * Motion Planning, Path.
 *
 * Copyright (C) 2017 Imperial College London.
 * Copyright (C) 2017 ETH Zürich.
 *
 * @file Path.hpp
 *
 * @ingroup common
 *
 * @author Marius Grimm (marius.grimm93@web.de)
 * @date May 22, 2017
 */

#ifndef PTP_PATH_HPP
#define PTP_PATH_HPP

#include <eigen3/Eigen/Dense>
#include <memory>
#include <ptp/Header.hpp>
#include <vector>

namespace ptp {

/** Struct defining a Path with a Header and a vector of states. */
template<int TSize>
struct Path {
    /** Type definition for smart pointer. */
    typedef std::shared_ptr<Path<TSize>> Ptr;

    Header header;                    ///> Header holding timestamp and frame_id
    std::vector<State<TSize>> states; ///> States of the path
};

} // namespace ptp

#endif //PTP_PATH_HPP
