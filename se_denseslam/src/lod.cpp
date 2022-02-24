// SPDX-FileCopyrightText: 2022 Smart Robotics Lab
// SPDX-FileCopyrightText: 2022 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#include "se/lod.hpp"

namespace se {

template<typename VoxelT>
int8_t block_scale_gain(const typename VoxelT::VoxelBlockType* block,
                        const Octree<VoxelT>& map,
                        const SensorImpl& sensor,
                        const Eigen::Matrix4f& T_CM,
                        const int8_t desired_scale)
{
    if (!block) {
        return 0;
    }
    const int8_t block_min_scale = block->minScaleReached();
    // Get the coordinates of the VoxelBlock's centre in the sensor frame C.
    const Eigen::Vector3f block_centre_coord_f = se::get_sample_coord(
        block->coordinates(), VoxelT::VoxelBlockType::size_li, Eigen::Vector3f::Constant(0.5f));
    const Eigen::Vector3f block_centre_M = map.voxelFToPoint(block_centre_coord_f);
    const Eigen::Vector3f block_centre_C = (T_CM * block_centre_M.homogeneous()).head(3);
    const int8_t block_expected_scale = sensor.targetIntegrationScale(
        block_centre_C, map.voxelDim(), VoxelT::VoxelBlockType::max_scale);
    return scale_gain(
        block_min_scale, block_expected_scale, desired_scale, VoxelT::VoxelBlockType::max_scale);
}



int8_t scale_gain(const int8_t block_min_scale,
                  const int8_t block_expected_scale,
                  const int8_t desired_scale,
                  const int8_t max_scale)
{
    // The fact that the VoxelBlock is allocated means that its scale is at least max_scale.
    if (desired_scale >= max_scale) {
        return 0.0f;
    }
    // Test if the desired scale has already been achieved in this VoxelBlock.
    if (block_min_scale <= desired_scale) {
        return 0.0f;
    }
    // Test if there's possibility of improving the scale from this view.
    if (block_expected_scale >= block_min_scale) {
        return 0.0f;
    }
    // The following should be true since we reached this place:
    // desired_scale < max_scale
    // desired_scale < block_min_scale
    // block_expected_scale < block_min_scale
    // Even if the expected scale is smaller than the desired scale, only count gain up to the
    // desired scale.
    return block_min_scale - std::max(block_expected_scale, desired_scale);
}



Image<float> bg_scale_gain(const Image<Eigen::Vector3f>& bg_hits_M,
                           const Octree<VoxelImpl::VoxelType>& map,
                           const SensorImpl& sensor,
                           const Eigen::Matrix4f& T_MB,
                           const Eigen::Matrix4f& T_BC,
                           const int8_t desired_scale)
{
    Image<float> gain_image(bg_hits_M.width(), bg_hits_M.height(), 0.0f);
    const float max_scale_gain = VoxelImpl::VoxelBlockType::max_scale - desired_scale;
    const Eigen::Matrix4f T_CM = se::math::to_inverse_transformation(T_MB * T_BC);
#pragma omp parallel for
    for (int y = 0; y < gain_image.height(); ++y) {
#pragma omp simd
        for (int x = 0; x < gain_image.width(); ++x) {
            const Eigen::Vector3f& hit_M = bg_hits_M(x, y);
            if (!isnan(hit_M.x())) {
                const auto* block = map.fetch(map.pointToVoxel(hit_M));
                const float gain = block_scale_gain(block, map, sensor, T_CM, desired_scale);
                gain_image(x, y) = gain / max_scale_gain;
            }
        }
    }
    return gain_image;
}

} // namespace se
