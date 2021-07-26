// SPDX-FileCopyrightText: 2021 Smart Robotics Lab
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#include <frustum_intersector.hpp>

#include <iostream>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/corefinement.h>
#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/Surface_mesh.h>

namespace fi {
  typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
  typedef CGAL::Surface_mesh<K::Point_3> Mesh;
  typedef CGAL::Surface_mesh<K::Point_3>::Vertex_index Vertex_index;



  Mesh frustum_to_mesh(const Eigen::Matrix<float, 4, num_frustum_vertices>& frustum_vertices_C) {
    Mesh m;
    // Add the frustum vertices to the mesh
    for (int i = 0; i < num_frustum_vertices; i++) {
      const Eigen::Vector3f& v = frustum_vertices_C.col(i).head<3>();
      m.add_vertex(K::Point_3(v.x(), v.y(), v.z()));
    }
    // Set the triangle mesh faces
    // Near plane
    m.add_face(Vertex_index(2), Vertex_index(1), Vertex_index(0));
    m.add_face(Vertex_index(3), Vertex_index(2), Vertex_index(0));
    // Far plane
    m.add_face(Vertex_index(4), Vertex_index(5), Vertex_index(7));
    m.add_face(Vertex_index(5), Vertex_index(6), Vertex_index(7));
    // Top plane
    m.add_face(Vertex_index(0), Vertex_index(1), Vertex_index(5));
    m.add_face(Vertex_index(5), Vertex_index(4), Vertex_index(0));
    // Bottom plane
    m.add_face(Vertex_index(7), Vertex_index(6), Vertex_index(2));
    m.add_face(Vertex_index(2), Vertex_index(3), Vertex_index(7));
    // Right plane
    m.add_face(Vertex_index(1), Vertex_index(2), Vertex_index(5));
    m.add_face(Vertex_index(2), Vertex_index(6), Vertex_index(5));
    // Left plane
    m.add_face(Vertex_index(0), Vertex_index(4), Vertex_index(7));
    m.add_face(Vertex_index(7), Vertex_index(3), Vertex_index(0));
    return m;
  }



  float frustum_volume(const Eigen::Matrix<float, 4, num_frustum_vertices>& frustum_vertices_C) {
    Mesh m = frustum_to_mesh(frustum_vertices_C);
    return CGAL::Polygon_mesh_processing::volume(m);
  }



  float frustum_intersection_pc(
      const Eigen::Matrix<float, 4, num_frustum_vertices>& frustum_vertices_C,
      const Eigen::Matrix4f& T_C0C1) {
    // We are doing the computation in the camera frame of the previous camera pose (C0) so there is
    // no need to transform the frutum vertices of the previous camera pose.
    const Eigen::Matrix<float, 4, num_frustum_vertices> frustum_vertices_0_C0 = frustum_vertices_C;
    // Transform the frustum vertices of the current camera pose to the frame of the previous camera
    // pose (C0).
    Eigen::Matrix<float, 4, num_frustum_vertices> frustum_vertices_1_C0;
    for (int i = 0; i < num_frustum_vertices; i++) {
      frustum_vertices_1_C0.col(i) = T_C0C1 * frustum_vertices_C.col(i);
    }
    // Compute the frustum intersection.
    Mesh frustum_0 = frustum_to_mesh(frustum_vertices_0_C0);
    Mesh frustum_1 = frustum_to_mesh(frustum_vertices_1_C0);
    Mesh intersection;
    const bool valid_intersection = CGAL::Polygon_mesh_processing::corefine_and_compute_intersection(
        frustum_0, frustum_1, intersection);
    // Return the percentage of overlap of the previous and currect frustum.
    if (valid_intersection) {
      return CGAL::Polygon_mesh_processing::volume(intersection)
        / CGAL::Polygon_mesh_processing::volume(frustum_0);
    } else {
      return 0.0f;
    }
  }
} // namespace fi

