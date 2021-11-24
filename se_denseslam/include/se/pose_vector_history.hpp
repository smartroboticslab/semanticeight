// SPDX-FileCopyrightText: 2021 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#ifndef POSE_VECTOR_HISTORY_HPP
#define POSE_VECTOR_HISTORY_HPP

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <random>
#include <se/sensor_implementation.hpp>
#include <vector>

namespace se {
typedef std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> PoseVector;

class PoseVectorHistory {
    public:
    PoseVector poses;

    PoseVectorHistory();

    bool rejectSampledPos(const Eigen::Vector3f& pos, const SensorImpl& sensor) const;

    bool rejectSampledPose(const Eigen::Matrix4f& pose, const SensorImpl& sensor) const;

    float rejectionProbabilityPos(const Eigen::Vector3f& pos, const SensorImpl& sensor) const;

    float rejectionProbabilityPose(const Eigen::Matrix4f& pose, const SensorImpl& sensor) const;

    PoseVector neighbourPoses(const Eigen::Matrix4f& pose, const SensorImpl& sensor) const;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    private:
    mutable std::mt19937 gen_;
    mutable std::uniform_real_distribution<float> uniform_;
};
} // namespace se

#endif // POSE_VECTOR_HISTORY_HPP
