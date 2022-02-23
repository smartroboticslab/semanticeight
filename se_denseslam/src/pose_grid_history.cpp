// SPDX-FileCopyrightText: 2021 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#include "se/pose_grid_history.hpp"

#include <algorithm>
#include <fstream>
#include <limits>

#include "se/entropy.hpp"

namespace se {

PoseGridHistory::PoseGridHistory(const Eigen::Vector3f& dimensions,
                                 const Eigen::Vector4f& resolution) :
        dim_(dimensions.x(), dimensions.y(), dimensions.z(), M_TAU_F),
        res_(resolution),
        inv_res_((1.0f / res_.array()).matrix()),
        size_((dim_.array() / res_.array()).ceil().matrix().cast<int>()),
        num_cells_(size_.prod()),
        grid_(num_cells_, 0)
{
}



void PoseGridHistory::record(const Eigen::Vector4f& pose)
{
    const size_t linear_idx = poseToIndex(pose);
    // Avoid overflow.
    if (grid_[linear_idx] < std::numeric_limits<DataType>::max()) {
        grid_[linear_idx]++;
    }
}



void PoseGridHistory::record(const Eigen::Matrix4f& pose)
{
    const Eigen::Vector3f t = pose.topRightCorner<3, 1>();
    record(Eigen::Vector4f(t.x(), t.y(), t.z(), getYaw(pose)));
}



PoseGridHistory::DataType PoseGridHistory::get(const Eigen::Vector4f& pose) const
{
    return grid_[poseToIndex(pose)];
}



PoseGridHistory::DataType PoseGridHistory::get(const Eigen::Matrix4f& pose) const
{
    const Eigen::Vector3f t = pose.topRightCorner<3, 1>();
    return get(Eigen::Vector4f(t.x(), t.y(), t.z(), getYaw(pose)));
}



float PoseGridHistory::rejectionProbability(const Eigen::Vector3f& position,
                                            const SensorImpl& /* sensor */) const
{
    // Count the visited yaw angles for this position.
    float visited_yaw = 0;
    for (float yaw = 0.0f; yaw < M_TAU_F; yaw += res_[3]) {
        const Eigen::Vector4f pose(position.x(), position.y(), position.z(), yaw);
        visited_yaw += (get(pose) > 0);
    }
    return visited_yaw / size_[3];
}



Eigen::Vector3f PoseGridHistory::dimensions() const
{
    return dim_.head<3>();
}



Eigen::Vector4f PoseGridHistory::resolution() const
{
    return res_;
}



Eigen::Vector4i PoseGridHistory::dimensionsCells() const
{
    return size_;
}



size_t PoseGridHistory::size() const
{
    return num_cells_;
}



PoseVector PoseGridHistory::neighbourPoses(const Eigen::Matrix4f& pose,
                                           const SensorImpl& /* sensor */) const
{
    const Eigen::Vector3f t = pose.topRightCorner<3, 1>();
    PoseVector neighbours;
    for (int yaw_idx = 0; yaw_idx < size_.w(); yaw_idx++) {
        Eigen::Vector4i indices = poseToIndices(Eigen::Vector4f(t.x(), t.y(), t.z(), 0.0f));
        indices.w() = yaw_idx;
        if (grid_[indicesToLinearIndex(indices)]) {
            const Eigen::Vector4f p = indicesToPose(indices);
            Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
            T.topLeftCorner<3, 3>() = se::math::yaw_to_rotm(p.w());
            T.topRightCorner<3, 1>() = p.head<3>();
            neighbours.push_back(T);
        }
    }
    return neighbours;
}



void PoseGridHistory::frustumOverlap(Image<uint8_t>& frustum_overlap_mask,
                                     const SensorImpl& /* sensor */,
                                     const Eigen::Matrix4f& T_MB,
                                     const Eigen::Matrix4f& /* T_BC */) const
{
#pragma omp parallel for
    for (int x = 0; x < frustum_overlap_mask.width(); x++) {
        const float yaw_M = se::index_to_azimuth(x, frustum_overlap_mask.width(), M_TAU_F);
        Eigen::Vector4f test_T_MB = T_MB.topRightCorner<3, 1>().homogeneous();
        test_T_MB.w() = yaw_M;
        const uint8_t value = get(test_T_MB) > 0 ? UINT8_MAX : 0u;
        for (int y = 0; y < frustum_overlap_mask.height(); y++) {
            frustum_overlap_mask(x, y) = value;
        }
    }
}



PoseGridHistory::XYZYawVector PoseGridHistory::visitedPoses() const
{
    PoseGridHistory::XYZYawVector visited_poses;
    for (size_t i = 0; i < size(); i++) {
        if (grid_[i]) {
            visited_poses.push_back(indexToPose(i));
        }
    }
    return visited_poses;
}



QuadMesh PoseGridHistory::gridMesh() const
{
    QuadMesh mesh;
    // Iterate over all grid cells
    for (int x = 0; x < size_.x(); x++) {
        for (int y = 0; y < size_.y(); y++) {
            for (int z = 0; z < size_.z(); z++) {
                // Aggregate the yaw
                size_t data_sum = 0;
                for (int yaw = 0; yaw < size_.w(); yaw++) {
                    data_sum += grid_[indicesToLinearIndex(Eigen::Vector4i(x, y, z, yaw))];
                }
                // Compute the vertices of the current xyz cell
                static constexpr int cell_num_vertexes = 8;
                Eigen::Array3f cell_vertexes[cell_num_vertexes];
                for (int v = 0; v < cell_num_vertexes; v++) {
                    static const Eigen::Array3f vertex_offsets[cell_num_vertexes] = {{0, 0, 0},
                                                                                     {1, 0, 0},
                                                                                     {0, 1, 0},
                                                                                     {1, 1, 0},
                                                                                     {0, 0, 1},
                                                                                     {1, 0, 1},
                                                                                     {0, 1, 1},
                                                                                     {1, 1, 1}};
                    const Eigen::Array3f point = res_.head<3>().array() * Eigen::Array3f(x, y, z);
                    const Eigen::Array3f offset = res_.head<3>().array() * vertex_offsets[v];
                    cell_vertexes[v] = point + offset;
                }
                // Add the faces to the mesh
                static constexpr int cell_num_faces = 6;
                for (int f = 0; f < cell_num_faces; f++) {
                    mesh.emplace_back();
                    for (size_t v = 0; v < Quad::num_vertexes; v++) {
                        static constexpr size_t
                            face_vertex_idx[cell_num_faces][Quad::num_vertexes] = {{0, 1, 3, 2},
                                                                                   {1, 5, 7, 3},
                                                                                   {5, 7, 6, 4},
                                                                                   {0, 2, 6, 4},
                                                                                   {0, 1, 5, 4},
                                                                                   {2, 3, 7, 6}};
                        mesh.back().vertexes[v] = cell_vertexes[face_vertex_idx[f][v]];
                        mesh.back().r[v] = std::min(data_sum, static_cast<size_t>(UINT8_MAX));
                        mesh.back().g[v] = mesh.back().r[v];
                        mesh.back().b[v] = mesh.back().r[v];
                    }
                }
            }
        }
    }
    return mesh;
}



TriangleMesh PoseGridHistory::wedgeMesh() const
{
    TriangleMesh mesh;
    // Iterate over all visited poses
    for (const auto& t : visitedPoses()) {
        // Compute the wedge vertices
        static constexpr int wedge_num_vertexes = 6;
        Eigen::Array3f wedge_vertexes[wedge_num_vertexes];
        const Eigen::Array3f half_xy_res(res_.x() / 2, res_.y() / 2, 0);
        // Bottom centre vertex
        wedge_vertexes[0] = t.head<3>().array() + half_xy_res;
        // Bottom right vertex
        const float right_yaw = t.w();
        const Eigen::Array3f right_normal(cos(right_yaw), sin(right_yaw), 0);
        wedge_vertexes[1] = wedge_vertexes[0] + half_xy_res * right_normal;
        // Bottom left vertex
        const float left_yaw = t.w() + res_.w();
        const Eigen::Array3f left_normal(cos(left_yaw), sin(left_yaw), 0);
        wedge_vertexes[2] = wedge_vertexes[0] + half_xy_res * left_normal;
        // The top vertices are just translated along the z axis
        for (int i = 0; i < 3; i++) {
            wedge_vertexes[i + 3] = wedge_vertexes[i] + Eigen::Array3f(0, 0, res_.z());
        }
        // Add the faces to the mesh
        static constexpr int wedge_num_faces = 8;
        for (size_t f = 0; f < wedge_num_faces; f++) {
            mesh.emplace_back();
            for (size_t v = 0; v < Triangle::num_vertexes; v++) {
                static constexpr size_t face_vertex_idx[wedge_num_faces][Triangle::num_vertexes] = {
                    {0, 1, 2},
                    {3, 4, 5},
                    {0, 1, 4},
                    {0, 4, 3},
                    {1, 2, 4},
                    {2, 5, 4},
                    {0, 5, 2},
                    {0, 3, 5}};
                const size_t idx = face_vertex_idx[f][v];
                mesh.back().vertexes[v] = wedge_vertexes[idx];
                if (idx != 0 && idx != 3) {
                    mesh.back().r[v] = grid_[poseToIndex(t)];
                    mesh.back().g[v] = mesh.back().r[v];
                    mesh.back().b[v] = mesh.back().r[v];
                }
            }
        }
    }
    return mesh;
}



size_t PoseGridHistory::indicesToLinearIndex(const Eigen::Vector4i& indices) const
{
    assert((0 <= indices.x() && "The x index is non-negative"));
    assert((0 <= indices.y() && "The y index is non-negative"));
    assert((0 <= indices.z() && "The z index is non-negative"));
    assert((0 <= indices.w() && "The yaw index is non-negative"));
    assert((indices.x() < size_.x() && "The x index isn't greater than the size"));
    assert((indices.y() < size_.y() && "The y index isn't greater than the size"));
    assert((indices.z() < size_.z() && "The z index isn't greater than the size"));
    assert((indices.w() < size_.w() && "The yaw index isn't greater than the size"));
    // https://en.wikipedia.org/wiki/Row-major_order#Address_calculation_in_general
    // Row-major order allows for faster iteration over the yaw angles for a single position.
    return indices[3] + size_[3] * (indices[2] + size_[2] * (indices[1] + size_[1] * indices[0]));
}



Eigen::Vector4i PoseGridHistory::linearIndexToIndices(const size_t linear_idx) const
{
    assert((linear_idx < num_cells_ && "The linear index isn't greater than the size"));
    Eigen::Vector4i indices;
    size_t tmp = linear_idx;
    for (int i = 3; i > 0; i--) {
        indices[i] = tmp % size_[i];
        tmp = tmp / size_[i];
    }
    indices[0] = tmp;
    return indices;
}



Eigen::Vector4i PoseGridHistory::poseToIndices(const Eigen::Vector4f& pose) const
{
    assert((0.0f <= pose.x() && "The x coordinate is non-negative"));
    assert((0.0f <= pose.y() && "The y coordinate is non-negative"));
    assert((0.0f <= pose.z() && "The z coordinate is non-negative"));
    assert((pose.x() < dim_.x() && "The x coordinate is smaller than the upper bound"));
    assert((pose.y() < dim_.y() && "The y coordinate is smaller than the upper bound"));
    assert((pose.z() < dim_.z() && "The z coordinate is smaller than the upper bound"));
    // Discretize the pose into cell indices.
    return (inv_res_.array() * wrapYaw(pose).array()).matrix().cast<int>();
}



Eigen::Vector4f PoseGridHistory::indicesToPose(const Eigen::Vector4i& indices) const
{
    assert((0 <= indices.x() && "The x index is non-negative"));
    assert((0 <= indices.y() && "The y index is non-negative"));
    assert((0 <= indices.z() && "The z index is non-negative"));
    assert((0 <= indices.w() && "The yaw index is non-negative"));
    assert((indices.x() < size_.x() && "The x index isn't greater than the size"));
    assert((indices.y() < size_.y() && "The y index isn't greater than the size"));
    assert((indices.z() < size_.z() && "The z index isn't greater than the size"));
    assert((indices.w() < size_.w() && "The yaw index isn't greater than the size"));
    return (res_.array() * indices.array().cast<float>()).matrix();
}



size_t PoseGridHistory::poseToIndex(const Eigen::Vector4f& pose) const
{
    return indicesToLinearIndex(poseToIndices(pose));
}



Eigen::Vector4f PoseGridHistory::indexToPose(const size_t linear_idx) const
{
    return indicesToPose(linearIndexToIndices(linear_idx));
}



Eigen::Vector4f PoseGridHistory::wrapYaw(const Eigen::Vector4f& pose)
{
    return Eigen::Vector4f(pose.x(), pose.y(), pose.z(), se::math::wrap_angle_2pi(pose.w()));
}



float PoseGridHistory::getYaw(const Eigen::Matrix4f& pose)
{
    // Keep only the rotation around the z axis from the quaternion.
    // https://stackoverflow.com/questions/5782658/extracting-yaw-from-a-quaternion#5783030
    Eigen::Quaternionf q(pose.topLeftCorner<3, 3>());
    q.x() = 0.0f;
    q.y() = 0.0f;
    q.normalize();
    return se::math::wrap_angle_2pi(2 * atan2(q.z(), q.w()));
}

} // namespace se
