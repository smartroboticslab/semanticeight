// SPDX-FileCopyrightText: 2021 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#ifndef POSE_HISTORY_HPP
#define POSE_HISTORY_HPP

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <random>
#include <vector>

#include "se/sensor_implementation.hpp"

namespace se {
typedef std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> PoseVector;

class PoseHistory {
    public:
    PoseHistory();

    virtual size_t size() const = 0;

    bool rejectPosition(const Eigen::Vector3f& pos, const SensorImpl& sensor) const;

    virtual void frustumOverlap(Image<uint8_t>& frustum_overlap_mask,
                                const SensorImpl& sensor,
                                const Eigen::Matrix4f& T_MC,
                                const Eigen::Matrix4f& T_BC) const = 0;

    private:
    mutable std::mt19937 gen_;
    mutable std::uniform_real_distribution<float> uniform_;

    virtual float rejectionProbability(const Eigen::Vector3f& position,
                                       const SensorImpl& sensor) const = 0;
};

} // namespace se

#endif // POSE_HISTORY_HPP
