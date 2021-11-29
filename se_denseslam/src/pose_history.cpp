// SPDX-FileCopyrightText: 2021 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#include "se/pose_history.hpp"

namespace se {

PoseHistory::PoseHistory() : uniform_(0.0f, 1.0f)
{
}



bool PoseHistory::rejectPosition(const Eigen::Vector3f& pos, const SensorImpl& sensor) const
{
    const float rejection_probability = rejectionProbability(pos, sensor);
    // A random value in the interval [0,1].
    const float sample = uniform_(gen_);
    return (sample < rejection_probability);
}

} // namespace se
