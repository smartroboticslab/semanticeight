// SPDX-FileCopyrightText: 2022 Smart Robotics Lab
// SPDX-FileCopyrightText: 2022 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#include "se/completion.hpp"

#include "se/object_rendering.hpp"

namespace se {

Image<float> object_completion_gain(const Image<Eigen::Vector3f>& rays_M,
                                    const Image<Eigen::Vector3f>& hits_M,
                                    const Objects& objects,
                                    const SensorImpl& sensor,
                                    const Eigen::Matrix4f& T_MB,
                                    const Eigen::Matrix4f& T_BC)
{
    Image<float> gain_image(rays_M.width(), rays_M.height(), 0.0f);
    const Eigen::Vector3f t_MC = se::math::to_translation(T_MB * T_BC);
    // The map (M) and body (B) frames have the same orientation so C_CB is the same as C_CM.
    const Eigen::Matrix3f C_CM = se::math::to_inverse_rotation(T_BC);
#pragma omp parallel for
    for (int y = 0; y < gain_image.height(); ++y) {
#pragma omp simd
        for (int x = 0; x < gain_image.width(); ++x) {
            // Only raycast the back-face on pixels where there was no foreground hit.
            if (std::isnan(hits_M(x, y).x())) {
                const Eigen::Vector3f ray_dir_C = C_CM * rays_M(x, y);
                const ObjectHit hit = raycast_objects(objects,
                                                      std::map<int, cv::Mat>(),
                                                      Eigen::Vector2f(x, y),
                                                      t_MC,
                                                      rays_M(x, y),
                                                      sensor.nearDist(ray_dir_C),
                                                      sensor.farDist(ray_dir_C),
                                                      ObjVoxelImpl::raycastBackFace);
                if (hit.valid()) {
                    gain_image(x, y) = 1.0f;
                }
            }
        }
    }
    return gain_image;
}

} // namespace se
