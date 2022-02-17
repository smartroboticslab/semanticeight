// SPDX-FileCopyrightText: 2021 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#ifndef POSE_VECTOR_HISTORY_HPP
#define POSE_VECTOR_HISTORY_HPP

#include "se/pose_history.hpp"

namespace se {

class PoseVectorHistory : public PoseHistory {
    public:
    PoseVector poses;

    void record(const Eigen::Vector4f& pose);

    void record(const Eigen::Matrix4f& pose);

    float rejectionProbability(const Eigen::Vector3f& pos, const SensorImpl& sensor) const;

    size_t size() const;

    PoseVector neighbourPoses(const Eigen::Matrix4f& pose, const SensorImpl& sensor) const;

    void frustumOverlap(Image<float>& frustum_overlap_image,
                        const SensorImpl& sensor,
                        const Eigen::Matrix4f& T_MB,
                        const Eigen::Matrix4f& T_BC) const;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

} // namespace se

#endif // POSE_VECTOR_HISTORY_HPP
