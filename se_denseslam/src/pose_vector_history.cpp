// SPDX-FileCopyrightText: 2021 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#include "se/pose_vector_history.hpp"

#include <algorithm>
#include <frustum_intersector.hpp>
#include <random>

#include "se/utils/math_utils.h"

float normal_pdf(const Eigen::Vector3f& x, const Eigen::Vector3f& mu, const Eigen::Matrix3f& sigma)
{
    const Eigen::Vector3f n = x - mu;
    const Eigen::Matrix3f sigma_inv = (Eigen::Matrix3f() << 1.0f / sigma(0, 0),
                                       0.0f,
                                       0.0f,
                                       0.0f,
                                       1.0f / sigma(1, 1),
                                       0.0f,
                                       0.0f,
                                       0.0f,
                                       1.0f / sigma(2, 2))
                                          .finished();
    const float sigma_det = sigma.determinant();
    return expf(-0.5f * n.transpose() * sigma_inv * n)
        / sqrtf(se::math::cu(2.0f * M_PI_F) * sigma_det);
}



namespace se {

void PoseVectorHistory::record(const Eigen::Vector4f& pose)
{
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T.topRightCorner<3, 1>() = pose.head<3>();
    T(0, 0) = cos(pose.w());
    T(0, 1) = -sin(pose.w());
    T(1, 0) = sin(pose.w());
    T(1, 1) = cos(pose.w());
    poses.push_back(T);
}



void PoseVectorHistory::record(const Eigen::Matrix4f& pose)
{
    poses.push_back(pose);
}



float PoseVectorHistory::rejectionProbability(const Eigen::Vector3f& pos,
                                              const SensorImpl& sensor) const
{
    // Compute the normal distribution parameters.
    const float stddev_xy = sensor.far_plane / 3.0f;
    // TODO SEM Should depend on vfov and far plane
    const float stddev_z = stddev_xy / 3.0f;
    const Eigen::Matrix3f sigma =
        (Eigen::Matrix3f() << stddev_xy, 0.0f, 0.0f, 0.0f, stddev_xy, 0.0f, 0.0f, 0.0f, stddev_z)
            .finished();
    // Compute the rejection probability of the supplied position with all previous positions.
    std::vector<float> probs(poses.size());
#pragma omp parallel for
    for (size_t i = 0; i < probs.size(); i++) {
        probs[i] = normal_pdf(pos, poses[i].topRightCorner<3, 1>(), sigma);
    }
    // Return the maximum rejection probability.
    return *std::max_element(probs.begin(), probs.end());
}



size_t PoseVectorHistory::size() const
{
    return poses.size();
}



PoseVector PoseVectorHistory::neighbourPoses(const Eigen::Matrix4f& pose,
                                             const SensorImpl& sensor) const
{
    PoseVector neighbours;
    for (const auto& p : poses) {
        if ((pose.topRightCorner<3, 1>() - p.topRightCorner<3, 1>()).norm() <= 2 * sensor.radius) {
            neighbours.push_back(p);
        }
    }
    return neighbours;
}



void PoseVectorHistory::frustumOverlap(Image<uint8_t>& frustum_overlap_mask,
                                       const SensorImpl& sensor,
                                       const Eigen::Matrix4f& T_MB,
                                       const Eigen::Matrix4f& T_BC) const
{
    const Eigen::Matrix4f T_CM = se::math::to_inverse_transformation(T_MB * T_BC);
    const se::PoseVector neighbors = neighbourPoses(T_MB, sensor);
#pragma omp parallel for
    for (int x = 0; x < frustum_overlap_mask.width(); x++) {
        std::vector<float> overlap;
        overlap.reserve(neighbors.size());
        for (const auto& n_T_MB : neighbors) {
            // Convert the neighbor pose to the candidate frame.
            const Eigen::Matrix4f T_CCn = T_CM * n_T_MB * T_BC;
            overlap.push_back(fi::frustum_intersection_pc(sensor.frustum_vertices_, T_CCn));
        }
        // FIXME SEM something should depend on x
        const uint8_t value = UINT8_MAX
            * (overlap.empty() ? 0.0f : *std::max_element(overlap.begin(), overlap.end()));
        for (int y = 0; y < frustum_overlap_mask.height(); y++) {
            frustum_overlap_mask(x, y) = value;
        }
    }
}

} // namespace se
