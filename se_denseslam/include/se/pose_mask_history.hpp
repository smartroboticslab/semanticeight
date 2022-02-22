// SPDX-FileCopyrightText: 2022 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2022 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#ifndef POSE_MASK_HISTORY_HPP
#define POSE_MASK_HISTORY_HPP

#include <cstdint>
#include <string>

#include "se/algorithms/mesh_face.hpp"
#include "se/pose_history.hpp"
#include "se/utils/math_utils.h"

namespace se {

/** A dense (x,y,z) grid storing a 360-degree raycasting mask of invalid hits.
 */
class PoseMaskHistory : public PoseHistory {
    public:
    typedef uint8_t MaskType;

    PoseMaskHistory(const Eigen::Vector2i& raycast_res,
                    const SensorImpl& sensor,
                    const Eigen::Matrix4f& T_BC,
                    const Eigen::Vector3f& dimensions,
                    const Eigen::Vector3f& resolution = Eigen::Vector3f::Constant(0.5f));

    size_t size() const;

    void frustumOverlap(Image<uint8_t>& frustum_overlap_mask,
                        const SensorImpl& sensor,
                        const Eigen::Matrix4f& T_MC,
                        const Eigen::Matrix4f& T_BC) const;

    void record(const Eigen::Matrix4f& T_MB, const se::Image<float>& depth);

    const se::Image<MaskType>& getMask(const Eigen::Vector3f& t_MB) const;
    const se::Image<MaskType>& getMask(const Eigen::Matrix4f& T_MB) const;

    int writeMasks(const std::string& directory, const bool modified_only = true) const;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW



    private:
    const Eigen::Vector2i raycast_res_;
    const Eigen::Vector3f dim_;
    const Eigen::Vector3f res_;
    const Eigen::Vector3f inv_res_;
    const Eigen::Vector3i size_;
    const SensorImpl sensor_;
    const Eigen::Matrix4f T_BC_;
    const Eigen::Matrix4f T_CB_;
    const Image<Eigen::Vector3f> rays_M_;
    std::vector<se::Image<MaskType>> grid_;

    /** \brief Return the ratio of invalid pixels in the corresponding mask.
     */
    float rejectionProbability(const Eigen::Vector3f& t_MB, const SensorImpl& sensor) const;

    size_t numInvalidPixels(const size_t idx) const;

    se::Image<MaskType>& get(const Eigen::Vector3f& t_MB);
    se::Image<MaskType>& get(const Eigen::Matrix4f& T_MB);

    size_t positionToIndex(const Eigen::Vector3f& pos) const;

    Eigen::Vector3f indexToPosition(const size_t idx) const;
};

} // namespace se

#endif // POSE_MASK_HISTORY_HPP
