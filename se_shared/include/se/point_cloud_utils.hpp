// SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#ifndef __POINT_CLOUD_UTILS_HPP
#define __POINT_CLOUD_UTILS_HPP

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <string>

#include "se/image/image.hpp"



namespace se {

/**
   * \brief Save a point cloud as a PCD file.
   *
   * Documentation for the PCD file format available here
   * https://pcl-tutorials.readthedocs.io/en/latest/pcd_file_format.html.
   *
   * \param[in] point_cloud The pointcloud to save.
   * \param[in] filename    The name of the PCD file to create.
   * \param[in] T_WC        The pose from which the point cloud was observed.
   * \return 0 on success, nonzero on error.
   */
int save_point_cloud_pcd(const se::Image<Eigen::Vector3f>& point_cloud,
                         const std::string& filename,
                         const Eigen::Matrix4f& T_WC);

/**
   * \brief Save a point cloud as a PLY file.
   *
   * Documentation for the PLY polygon file format available here
   * https://web.archive.org/web/20161204152348/http://www.dcs.ed.ac.uk/teaching/cs4/www/graphics/Web/ply.html.
   *
   * \param[in] point_cloud The pointcloud to save.
   * \param[in] filename    The name of the PCD file to create.
   * \param[in] T_WC        The pose from which the point cloud was observed.
   * \return 0 on success, nonzero on error.
   */
int save_point_cloud_ply(const se::Image<Eigen::Vector3f>& point_cloud,
                         const std::string& filename,
                         const Eigen::Matrix4f& T_WC);

/**
   * \brief Save a point cloud as a VTK file.
   *
   * Documentation for the VTK file format available here
   * https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf.
   *
   * \param[in] point_cloud The pointcloud to save.
   * \param[in] filename    The name of the PCD file to create.
   * \param[in] T_WC        The pose from which the point cloud was observed.
   * \return 0 on success, nonzero on error.
   */
int save_point_cloud_vtk(const se::Image<Eigen::Vector3f>& point_cloud,
                         const std::string& filename,
                         const Eigen::Matrix4f& T_WC);

/** Save rays in frame F as line segments in a PLY file.
 */
int save_rays_ply(
    const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& rays_F,
    const std::string& filename,
    const Eigen::Matrix4f& T_WF = Eigen::Matrix4f::Identity());

/** Save rays in frame F as line segments in a PLY file.
 */
int save_rays_ply(const se::Image<Eigen::Vector3f>& rays_F,
                  const std::string& filename,
                  const Eigen::Matrix4f& T_WF = Eigen::Matrix4f::Identity());

} // namespace se

#endif
