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

#ifndef MESH_IO_H
#define MESH_IO_H
#include <fstream>
#include <sstream>
#include <iostream>
#include "se/utils/math_utils.h"
#include "se/io/meshing_io.hpp"
#include <algorithm>

namespace se {
inline void writeVtkMesh(const char*                  vtk_filename,
                         const std::vector<Triangle>& mesh,
                         const Eigen::Matrix4f&       T_WM,
                         const float*                 point_data = nullptr,
                         const float*                 cell_data = nullptr){
  std::stringstream ss_points_W;
  std::stringstream ss_polygons;
  std::stringstream ss_point_data;
  std::stringstream ss_cell_data;
  int point_count = 0;
  int triangle_count = 0;
  bool has_point_data = point_data != nullptr;
  bool has_cell_data = cell_data != nullptr;

  for(unsigned int i = 0; i < mesh.size(); ++i ){
    const Triangle& triangle_M = mesh[i];

    Eigen::Vector3f vertex_0_W = (T_WM * triangle_M.vertexes[0].homogeneous()).head(3);
    Eigen::Vector3f vertex_1_W = (T_WM * triangle_M.vertexes[1].homogeneous()).head(3);
    Eigen::Vector3f vertex_2_W = (T_WM * triangle_M.vertexes[2].homogeneous()).head(3);

    ss_points_W << vertex_0_W.x() << " "
                << vertex_0_W.y() << " "
                << vertex_0_W.z() << std::endl;

    ss_points_W << vertex_1_W.x() << " "
                << vertex_1_W.y() << " "
                << vertex_1_W.z() << std::endl;

    ss_points_W << vertex_2_W.x() << " "
                << vertex_2_W.y() << " "
                << vertex_2_W.z() << std::endl;

    ss_polygons << "3 " << point_count << " " << point_count+1 <<
                " " << point_count+2 << std::endl;

    if(has_point_data){
      ss_point_data << point_data[i*3] << std::endl;
      ss_point_data << point_data[i*3 + 1] << std::endl;
      ss_point_data << point_data[i*3 + 2] << std::endl;
    }

    if(has_cell_data){
      ss_cell_data << cell_data[i] << std::endl;
    }

    point_count +=3;
    triangle_count++;
  }

  std::ofstream f;
  f.open(vtk_filename);
  f << "# vtk DataFile Version 1.0" << std::endl;
  f << "vtk mesh generated from KFusion" << std::endl;
  f << "ASCII" << std::endl;
  f << "DATASET POLYDATA" << std::endl;

  f << "POINTS " << point_count << " FLOAT" << std::endl;
  f << ss_points_W.str();

  f << "POLYGONS " << triangle_count << " " << triangle_count * 4 << std::endl;
  f << ss_polygons.str() << std::endl;
  if(has_point_data){
    f << "POINT_DATA " << point_count << std::endl;
    f << "SCALARS vertex_scalars float 1" << std::endl;
    f << "LOOKUP_TABLE default" << std::endl;
    f << ss_point_data.str();
  }

  if(has_cell_data){
    f << "CELL_DATA " << triangle_count << std::endl;
    f << "SCALARS cell_scalars float 1" << std::endl;
    f << "LOOKUP_TABLE default" << std::endl;
    f << ss_cell_data.str();
  }
  f.close();
}

inline void writeObjMesh(const char*                  obj_filename,
                         const std::vector<Triangle>& mesh){
  std::stringstream points_M;
  std::stringstream faces;
  int point_count = 0;
  int face_count = 0;

  for(unsigned int i = 0; i < mesh.size(); i++){
    const Triangle& triangle_M = mesh[i];
    points_M << "v " << triangle_M.vertexes[0].x() << " "
             << triangle_M.vertexes[0].y() << " "
             << triangle_M.vertexes[0].z() << std::endl;
    points_M << "v " << triangle_M.vertexes[1].x() << " "
             << triangle_M.vertexes[1].y() << " "
             << triangle_M.vertexes[1].z() << std::endl;
    points_M << "v " << triangle_M.vertexes[2].x() << " "
             << triangle_M.vertexes[2].y() << " "
             << triangle_M.vertexes[2].z() << std::endl;

    faces  << "f " << (face_count*3)+1 << " " << (face_count*3)+2
           << " " << (face_count*3)+3 << std::endl;

    point_count +=3;
    face_count += 1;
  }

  std::ofstream f(obj_filename);
  f << "# OBJ file format with ext .obj" << std::endl;
  f << "# vertex count = " << point_count << std::endl;
  f << "# face count = " << face_count << std::endl;
  f << points_M.str();
  f << faces.str();
  f.close();
  std::cout << "Written " << face_count << " faces and " << point_count
            << " points" << std::endl;
}
} // namespace se
#endif
