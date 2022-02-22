// SPDX-FileCopyrightText: 2021 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#ifndef POSE_GRID_HISTORY_HPP
#define POSE_GRID_HISTORY_HPP

#include <cstdint>
#include <string>

#include "se/algorithms/mesh_face.hpp"
#include "se/pose_history.hpp"
#include "se/utils/math_utils.h"

namespace se {

/** A dense grid storing the number of poses (x,y,z,yaw) encountered in each cell.
 */
class PoseGridHistory : public PoseHistory {
    public:
    typedef uint8_t DataType;
    typedef std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f>> XYZYawVector;

    /** Initialize a PoseGridHistory with the given x,y,z dimensions and x,y,z,yaw resolution. The
     * grid cells are all initialized to 0.
     */
    PoseGridHistory(const Eigen::Vector3f& dimensions,
                    const Eigen::Vector4f& resolution =
                        Eigen::Vector4f(0.5f, 0.5f, 0.5f, se::math::deg_to_rad(22.5f)));

    void record(const Eigen::Vector4f& pose);

    void record(const Eigen::Matrix4f& pose);

    DataType get(const Eigen::Vector4f& pose) const;

    DataType get(const Eigen::Matrix4f& pose) const;

    Eigen::Vector3f dimensions() const;

    Eigen::Vector4f resolution() const;

    Eigen::Vector4i dimensionsCells() const;

    size_t size() const;

    PoseVector neighbourPoses(const Eigen::Matrix4f& pose, const SensorImpl& sensor) const;

    void frustumOverlap(Image<uint8_t>& frustum_overlap_mask,
                        const SensorImpl& sensor,
                        const Eigen::Matrix4f& T_MB,
                        const Eigen::Matrix4f& T_BC) const;

    XYZYawVector visitedPoses() const;

    QuadMesh gridMesh() const;

    TriangleMesh wedgeMesh() const;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW



    private:
    const Eigen::Vector4f dim_;
    const Eigen::Vector4f res_;
    const Eigen::Vector4f inv_res_;
    const Eigen::Vector4i size_;
    const size_t num_cells_;
    std::vector<DataType> grid_;

    size_t indicesToLinearIndex(const Eigen::Vector4i& indices) const;

    Eigen::Vector4i linearIndexToIndices(const size_t linear_idx) const;

    Eigen::Vector4i poseToIndices(const Eigen::Vector4f& pose) const;

    Eigen::Vector4f indicesToPose(const Eigen::Vector4i& pose) const;

    size_t poseToIndex(const Eigen::Vector4f& pose) const;

    Eigen::Vector4f indexToPose(const size_t idx) const;

    float rejectionProbability(const Eigen::Vector3f& position, const SensorImpl& sensor) const;

    static Eigen::Vector4f wrapYaw(const Eigen::Vector4f& pose);

    static float getYaw(const Eigen::Matrix4f& pose);
};

} // namespace se

#endif // POSE_GRID_HISTORY_HPP
