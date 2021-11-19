// SPDX-FileCopyrightText: 2018-2020 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2016 Emanuele Vespa, ETH Zürich
// SPDX-FileCopyrightText: 2019-2020 Nils Funk, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#ifndef __OCTREE_IO_HPP
#define __OCTREE_IO_HPP

#include <Eigen/Dense>
#include <string>

#include "se/octree.hpp"



namespace se {

/**
   * \brief Save a 3D slice of the octree values as a VTK file.
   *
   * \param[in] octree        The octree to be sliced.
   * \param[in] filename      The output filename.
   * \param[in] lower_coord   The lower, left, front coordinates of the 3D slice.
   * \param[in] upper_coord   The upper, right, back coordinates of the 3D slice.
   * \param[in] select_value  lambda function selecting the node and voxel value from the octant data to be saved.
   * \param[in] min_scale     The minimum scale to select the data from.
   * \return 0 on success, nonzero on error.
   */
template<typename VoxelT, typename ValueSelector>
int save_3d_value_slice_vtk(const se::Octree<VoxelT>& octree,
                            const std::string filename,
                            const Eigen::Vector3i& lower_coord,
                            const Eigen::Vector3i& upper_coord,
                            ValueSelector select_value,
                            const int min_scale = 0);

/**
   * \brief Save a 3D slice of the octree values as a VTK file.
   *
   * \param[in] octree             The octree to be sliced.
   * \param[in] filename           The output filename.
   * \param[in] lower_coord        The lower, left, front coordinates of the 3D slice.
   * \param[in] upper_coord        The upper, right, back coordinates of the 3D slice.
   * \param[in] select_node_value  Lambda function to select the node value to store in the slice.
   * \param[in] select_voxel_value Lambda function to select the voxel value to store in the slice.
   * \param[in] min_scale          The minimum scale to select the data from.
   * \return 0 on success, nonzero on error.
   */
template<typename VoxelT, typename NodeValueSelector, typename VoxelValueSelector>
int save_3d_value_slice_vtk(const se::Octree<VoxelT>& octree,
                            const std::string filename,
                            const Eigen::Vector3i& lower_coord,
                            const Eigen::Vector3i& upper_coord,
                            NodeValueSelector select_node_value,
                            VoxelValueSelector select_voxel_value,
                            const int min_scale = 0);

template<typename VoxelT>
int save_3d_scale_slice_vtk(const se::Octree<VoxelT>& octree,
                            const std::string filename,
                            const Eigen::Vector3i& lower_coord,
                            const Eigen::Vector3i& upper_coord,
                            const int min_scale = 0);

template<typename ValueGetter>
int save_3d_slice_vtk(const std::string filename,
                      const Eigen::Vector3i& lower_coord,
                      const Eigen::Vector3i& upper_coord,
                      ValueGetter get_value);

/**
   * \brief Save the octree structure as a PLY file.
   *
   * Documentation for the PLY polygon file format available here
   * https://web.archive.org/web/20161204152348/http://www.dcs.ed.ac.uk/teaching/cs4/www/graphics/Web/ply.html.
   *
   * \note Aggregated voxel blocks are written as a single octant of size block_size^3.
   *
   * \param[in] octree   The octree providing the structure to be saved.
   * \param[in] filename The output filename.
   * \return 0 on success, nonzero on error.
   */
template<typename VoxelT>
int save_octree_structure_ply(const se::Octree<VoxelT>& octree, const std::string filename);

} // namespace se

#include "octree_io_impl.hpp"

#endif // __OCTREE_IO_HPP
