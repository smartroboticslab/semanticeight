// SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#ifndef __POINT_CLOUD_UTILS_HPP
#define __POINT_CLOUD_UTILS_HPP

#include <string>

#include <Eigen/Dense>

#include "se/image/image.hpp"



namespace se {
  /**
   * Save a point cloud as a PCD file.
   * Documentation for the PCD file format available here
   * https://pcl-tutorials.readthedocs.io/en/latest/pcd_file_format.html.
   *
   * \param[in] point_cloud The pointcloud to save.
   * \param[in] filename    The name of the PCD file to create.
   * \param[in] T_WC        The pose from which the point cloud was observed.
   * \return 0 on success, nonzero on error.
   */
  int save_point_cloud_pcd(se::Image<Eigen::Vector3f>& point_cloud,
                           const std::string&          filename,
                           const Eigen::Matrix4f&      T_WC);
} // namespace se

#endif

