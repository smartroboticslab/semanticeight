/*
 Copyright 2016 Emanuele Vespa, Imperial College London

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.

 3. Neither the name of the copyright holder nor the names of its contributors
 may be used to endorse or promote products derived from this software without
 specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef VTK_IO_H
#define VTK_IO_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

#include <Eigen/Dense>

#include "se/octree.hpp"


/**
 * \brief Save a 3D slice of the octree values as a VTK file.
 *
 * \param[in] octree       The octree to be sliced.
 * \param[in] filename     The output filename.
 * \param[in] lower_coord  The lower, left, front coordinates of the 3D slice.
 * \param[in] upper_coord  The upper, right, back coordinates of the 3D slice.
 * \param[in] select_value lambda function selecting the value from the voxel data to be saved.
 * \param[in] scale        The minimum scale to select the data from.
 * \return 0 on success, nonzero on error.
 */
template <typename VoxelT, typename ValueSelector>
int save_3d_slice_vtk(const se::Octree<VoxelT>& octree,
                      const std::string         filename,
                      const Eigen::Vector3i&    lower_coord,
                      const Eigen::Vector3i&    upper_coord,
                      ValueSelector             select_value,
                      const int                 scale) {

  // Open the file for writing.
  std::ofstream file (filename.c_str());
  if (!file.is_open()) {
    std::cerr << "Unable to write file " << filename << "\n";
    return 1;
  }

  std::stringstream ss_x_coord, ss_y_coord, ss_z_coord, ss_scalars;

  const int stride = 1 << scale;
  const int dimX = std::max(1, (upper_coord.x() - lower_coord.x()) / stride);
  const int dimY = std::max(1, (upper_coord.y() - lower_coord.y()) / stride);
  const int dimZ = std::max(1, (upper_coord.z() - lower_coord.z()) / stride);

  file << "# vtk DataFile Version 1.0" << std::endl;
  file << "vtk mesh generated from KFusion" << std::endl;
  file << "ASCII" << std::endl;
  file << "DATASET RECTILINEAR_GRID" << std::endl;
  file << "DIMENSIONS " << dimX << " " << dimY << " " << dimZ << std::endl;

  for (int x = lower_coord.x(); x < upper_coord.x(); x += stride) {
    ss_x_coord << x << " ";
  }
  for (int y = lower_coord.y(); y < upper_coord.y(); y += stride) {
    ss_y_coord << y << " ";
  }
  for (int z = lower_coord.z(); z < upper_coord.z(); z += stride) {
    ss_z_coord << z << " ";
  }

  for (int z = lower_coord.z(); z < upper_coord.z(); z += stride) {
    for (int y = lower_coord.y(); y < upper_coord.y(); y += stride) {
      for (int x = lower_coord.x(); x < upper_coord.x(); x += stride) {
        const auto value = select_value(octree.getFine(x, y, z, scale));
        ss_scalars << value << std::endl;
      }
    }
  }

  file << "X_COORDINATES " << dimX << " int " << std::endl;
  file << ss_x_coord.str() << std::endl;

  file << "Y_COORDINATES " << dimY << " int " << std::endl;
  file << ss_y_coord.str() << std::endl;

  file << "Z_COORDINATES " << dimZ << " int " << std::endl;
  file << ss_z_coord.str() << std::endl;

  file << "POINT_DATA " << dimX * dimY * dimZ << std::endl;
  file << "SCALARS scalars float 1" << std::endl;
  file << "LOOKUP_TABLE default" << std::endl;
  file << ss_scalars.str() << std::endl;

  file.close();
  return 0;
}

#endif // VTK_IO_H
