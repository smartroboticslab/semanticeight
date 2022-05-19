// SPDX-FileCopyrightText: 2022 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2022 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#include "se/pose_mask_history.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <lodepng.h>
#include <sstream>

#include "se/entropy.hpp"
#include "se/filesystem.hpp"
#include "se/preprocessing.hpp"

namespace se {

PoseMaskHistory::PoseMaskHistory(const Eigen::Vector2i& raycast_res,
                                 const SensorImpl& sensor,
                                 const Eigen::Matrix4f& T_BC,
                                 const Eigen::Vector3f& dimensions,
                                 const Eigen::Vector3f& resolution) :
        raycast_res_(raycast_res),
        dim_(dimensions),
        res_(resolution),
        inv_res_((1.0f / res_.array()).matrix()),
        size_((dim_.array() / res_.array()).ceil().matrix().cast<int>()),
        sensor_(sensor),
        T_BC_(T_BC),
        T_CB_(se::math::to_inverse_transformation(T_BC)),
        rays_M_(se::ray_M_360_image(raycast_res.x(), raycast_res.y(), sensor, T_BC)),
        grid_(size_.prod(), Image<PoseMaskHistory::MaskType>(raycast_res.x(), raycast_res.y(), 0))
{
}



size_t PoseMaskHistory::size() const
{
    return grid_.size();
}



void PoseMaskHistory::frustumOverlap(Image<uint8_t>& frustum_overlap_mask,
                                     const SensorImpl& /* sensor */,
                                     const Eigen::Matrix4f& T_MC,
                                     const Eigen::Matrix4f& T_BC) const
{
    const Eigen::Matrix4f T_MB = T_MC * se::math::to_inverse_transformation(T_BC);
    frustum_overlap_mask = getMask(T_MB);
}



void PoseMaskHistory::record(const Eigen::Matrix4f& T_MB, const se::Image<float>& depth)
{
    const Eigen::Matrix4f T_CM = T_CB_ * se::math::to_inverse_transformation(T_MB);
    Image<PoseMaskHistory::MaskType>& invalid_mask = get(T_MB);
#pragma omp parallel for
    for (int y = 0; y < rays_M_.height(); ++y) {
        for (int x = 0; x < rays_M_.width(); ++x) {
            const Eigen::Vector3f ray_C = T_CM.topLeftCorner<3, 3>() * rays_M_(x, y);
            Eigen::Vector2f image_point;
            if (sensor_.model.project(ray_C, &image_point)
                == srl::projection::ProjectionStatus::Successful) {
                const Eigen::Vector2f pt = image_point + Eigen::Vector2f::Constant(0.5f);
                const float d = depth(pt.x(), pt.y());
                if ((d < 1e-5) || std::isnan(d)) {
                    invalid_mask(x, y) = UINT8_MAX;
                }
            }
            // Do nothing on invalid projections as these would happen on regions of the 360 degree
            // image that are invisible from this pose.
        }
    }
}



const Image<PoseMaskHistory::MaskType>& PoseMaskHistory::getMask(const Eigen::Vector3f& t_MB) const
{
    return grid_[positionToIndex(t_MB)];
}



const Image<PoseMaskHistory::MaskType>& PoseMaskHistory::getMask(const Eigen::Matrix4f& T_MB) const
{
    const Eigen::Vector3f& t_MB = T_MB.topRightCorner<3, 1>();
    return getMask(t_MB);
}



int PoseMaskHistory::writeMasks(const std::string& directory, const bool modified_only) const
{
    stdfs::create_directories(directory);
    int status = 0;
    for (size_t i = 0; i < grid_.size(); ++i) {
        const auto& mask = grid_[i];
        if (modified_only && numInvalidPixels(i) == 0) {
            continue;
        }
        const Eigen::Vector3f t_MB = indexToPosition(i);
        const std::string filename(directory + "/mask_" + std::to_string(t_MB.x()) + "_"
                                   + std::to_string(t_MB.y()) + "_" + std::to_string(t_MB.z())
                                   + ".png");
        status += lodepng_encode_file(filename.c_str(),
                                      reinterpret_cast<const unsigned char*>(mask.data()),
                                      mask.width(),
                                      mask.height(),
                                      LCT_GREY,
                                      8 * sizeof(PoseMaskHistory::MaskType));
    }
    return status;
}



float PoseMaskHistory::rejectionProbability(const Eigen::Vector3f& t_MB,
                                            const SensorImpl& /* sensor */) const
{
    return static_cast<float>(numInvalidPixels(positionToIndex(t_MB))) / raycast_res_.prod();
}



size_t PoseMaskHistory::numInvalidPixels(const size_t idx) const
{
    const Image<PoseMaskHistory::MaskType>& mask = grid_[idx];
    return std::count_if(mask.begin(), mask.end(), [](auto x) { return x; });
}



Image<PoseMaskHistory::MaskType>& PoseMaskHistory::get(const Eigen::Vector3f& t_MB)
{
    return grid_[positionToIndex(t_MB)];
}



Image<PoseMaskHistory::MaskType>& PoseMaskHistory::get(const Eigen::Matrix4f& T_MB)
{
    const Eigen::Vector3f& t_MB = T_MB.topRightCorner<3, 1>();
    return get(t_MB);
}



size_t PoseMaskHistory::positionToIndex(const Eigen::Vector3f& pos) const
{
    assert((0.0f <= pos.x() && "The x coordinate is non-negative"));
    assert((0.0f <= pos.y() && "The y coordinate is non-negative"));
    assert((0.0f <= pos.z() && "The z coordinate is non-negative"));
    assert((pos.x() < dim_.x() && "The x coordinate is smaller than the upper bound"));
    assert((pos.y() < dim_.y() && "The y coordinate is smaller than the upper bound"));
    assert((pos.z() < dim_.z() && "The z coordinate is smaller than the upper bound"));
    // Discretize the position into (x,y,z) indices.
    const Eigen::Vector3i indices = (inv_res_.array() * pos.array()).matrix().cast<int>();
    assert((0 <= indices.x() && "The x index is non-negative"));
    assert((0 <= indices.y() && "The y index is non-negative"));
    assert((0 <= indices.z() && "The z index is non-negative"));
    assert((indices.x() < size_.x() && "The x index isn't greater than the size"));
    assert((indices.y() < size_.y() && "The y index isn't greater than the size"));
    assert((indices.z() < size_.z() && "The z index isn't greater than the size"));
    // Convert the (x,y,z) indices into a row-major linear index.
    // https://en.wikipedia.org/wiki/Row-major_order#Address_calculation_in_general
    const size_t idx = indices[2] + size_[2] * (indices[1] + size_[1] * indices[0]);
    assert((idx < grid_.size() && "The linear index isn't greater than the size"));
    return idx;
}



Eigen::Vector3f PoseMaskHistory::indexToPosition(const size_t idx) const
{
    assert((idx < grid_.size() && "The linear index isn't greater than the size"));
    // Convert the row-major linear index into (x,y,z) indices.
    Eigen::Vector3i indices;
    size_t tmp = idx;
    for (int i = indices.size() - 1; i > 0; i--) {
        indices[i] = tmp % size_[i];
        tmp = tmp / size_[i];
    }
    indices[0] = tmp;
    assert((0 <= indices.x() && "The x index is non-negative"));
    assert((0 <= indices.y() && "The y index is non-negative"));
    assert((0 <= indices.z() && "The z index is non-negative"));
    assert((indices.x() < size_.x() && "The x index isn't greater than the size"));
    assert((indices.y() < size_.y() && "The y index isn't greater than the size"));
    assert((indices.z() < size_.z() && "The z index isn't greater than the size"));
    // Convert the (x,y,z) indices into a position.
    const Eigen::Vector3f pos = (res_.array() * indices.array().cast<float>()).matrix();
    assert((0.0f <= pos.x() && "The x coordinate is non-negative"));
    assert((0.0f <= pos.y() && "The y coordinate is non-negative"));
    assert((0.0f <= pos.z() && "The z coordinate is non-negative"));
    assert((pos.x() < dim_.x() && "The x coordinate is smaller than the upper bound"));
    assert((pos.y() < dim_.y() && "The y coordinate is smaller than the upper bound"));
    assert((pos.z() < dim_.z() && "The z coordinate is smaller than the upper bound"));
    return pos;
}

} // namespace se
