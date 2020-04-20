// SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#ifndef __SENSOR_HPP
#define __SENSOR_HPP

#include <cmath>

#include <Eigen/Dense>
#include <srl/projection/NoDistortion.hpp>
#include <srl/projection/OusterLidar.hpp>
#include <srl/projection/PinholeCamera.hpp>



namespace se {

  struct SensorConfig {
    // General
    int width = 0;
    int height = 0;
    float near_plane = 0.f;
    float far_plane = INFINITY;
    float mu = 0.1f;
    // Pinhole camera
    double fx = nan("");
    double fy = nan("");
    double cx = nan("");
    double cy = nan("");
    // LIDAR
    Eigen::VectorXd beam_azimuth_angles = Eigen::VectorXd(0);
    Eigen::VectorXd beam_elevation_angles = Eigen::VectorXd(0);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };



  struct PinholeCamera {
    PinholeCamera(const SensorConfig& c);

    srl::projection::PinholeCamera<srl::projection::NoDistortion> sensor;
    float near_plane;
    float far_plane;
    float mu;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };



  struct OusterLidar {
    OusterLidar(const SensorConfig& c);

    srl::projection::OusterLidar sensor;
    float near_plane;
    float far_plane;
    float mu;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

} // namespace se

#endif

