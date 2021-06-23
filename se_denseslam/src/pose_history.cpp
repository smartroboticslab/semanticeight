// SPDX-FileCopyrightText: 2021 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <random>

#include "se/pose_history.hpp"
#include "se/utils/math_utils.h"

float normal_pdf(const Eigen::Vector3f& x,
                 const Eigen::Vector3f& mu,
                 const Eigen::Matrix3f& sigma) {
  const Eigen::Vector3f n = x - mu;
  const Eigen::Matrix3f sigma_inv = (Eigen::Matrix3f() <<
        1.0f / sigma(0,0), 0.0f, 0.0f,
        0.0f, 1.0f / sigma(1,1), 0.0f,
        0.0f, 0.0f, 1.0f / sigma(2,2)).finished();
  const float sigma_det = sigma.determinant();
  return expf(-0.5f * n.transpose() * sigma_inv * n) / sqrtf( se::math::cu(2.0f * M_PI_F) * sigma_det);
}



namespace se {
  PoseHistory::PoseHistory()
    : uniform_(0.0f, 1.0f) {
  }



  bool PoseHistory::rejectSampledPos(const Eigen::Vector3f& pos, const SensorImpl& sensor) const {
    const float prob = rejectionProbabilityPos(pos, sensor);
    // A random value in the interval [0,1].
    const float sample = uniform_(gen_);
    return (sample < prob);
  }


  bool PoseHistory::rejectSampledPose(const Eigen::Matrix4f& pose, const SensorImpl& sensor) const {
    return PoseHistory::rejectSampledPos(pose.topRightCorner<3,1>(), sensor);
  }



  float PoseHistory::rejectionProbabilityPos(const Eigen::Vector3f& pos, const SensorImpl& sensor) const {
    // Compute the normal distribution parameters.
    const float stddev_xy = sensor.far_plane / 3.0f;
    // TODO SEM Should depend on vfov and far plane
    const float stddev_z = stddev_xy / 3.0f;
    const Eigen::Matrix3f sigma = (Eigen::Matrix3f() <<
        stddev_xy, 0.0f, 0.0f,
        0.0f, stddev_xy, 0.0f,
        0.0f, 0.0f, stddev_z).finished();
    // Compute the rejection probability of the supplied position with all previous positions.
    std::vector<float> probs (poses.size());
#pragma omp parallel for
    for (size_t i = 0; i < probs.size(); i++) {
      probs[i] = normal_pdf(pos, poses[i].topRightCorner<3,1>(), sigma);
    }
    // Return the maximum rejection probability.
    return *std::max_element(probs.begin(), probs.end());
  }



  float PoseHistory::rejectionProbabilityPose(const Eigen::Matrix4f& pose, const SensorImpl& sensor) const {
    return PoseHistory::rejectionProbabilityPos(pose.topRightCorner<3,1>(), sensor);
  }



  PoseVector PoseHistory::neighbourPoses(const Eigen::Matrix4f& pose, const SensorImpl& sensor) const {
    PoseVector neighbours;
    for (const auto& p : poses) {
      if ((pose.topRightCorner<3,1>() - p.topRightCorner<3,1>()).norm() <= 2 * sensor.radius) {
        neighbours.push_back(p);
      }
    }
    return neighbours;
  }
} // namespace se

