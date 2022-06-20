/*
 * SPDX-FileCopyrightText: 2016-2019 Emanuele Vespa
 * SPDX-FileCopyrightText: 2019-2021 Smart Robotics Lab, Imperial College London, Technical University of Munich
 * SPDX-FileCopyrightText: 2021 Nils Funk
 * SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef SE_MESH_FACE_HPP
#define SE_MESH_FACE_HPP

#include <Eigen/Dense>
#include <array>
#include <cstdint>
#include <vector>

namespace se {

template<size_t NumVertexes>
struct MeshFace {
    std::array<Eigen::Vector3f, NumVertexes> vertexes;
    std::array<uint8_t, NumVertexes> r;
    std::array<uint8_t, NumVertexes> g;
    std::array<uint8_t, NumVertexes> b;
    int8_t max_vertex_scale;
    float min_dist_updated;
    float fg;
    static constexpr size_t num_vertexes = NumVertexes;

    MeshFace() : max_vertex_scale(0), min_dist_updated(INFINITY), fg(0.0f)
    {
        vertexes.fill(Eigen::Vector3f::Zero());
        r.fill(0u);
        g.fill(0u);
        b.fill(0u);
    }
};

template<typename FaceT>
using Mesh = std::vector<FaceT>;

typedef MeshFace<3> Triangle;
typedef Mesh<Triangle> TriangleMesh;

typedef MeshFace<4> Quad;
typedef Mesh<Quad> QuadMesh;

} // namespace se

#endif // SE_MESH_FACE_HPP
