// SPDX-FileCopyrightText: 2022 Smart Robotics Lab
// SPDX-FileCopyrightText: 2022 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#include "se/dist.hpp"

#include "se/object_rendering.hpp"

namespace se {

template<typename VoxelT>
float block_dist_gain(const typename VoxelT::VoxelBlockType* block,
                      const Octree<VoxelT>& map,
                      const SensorImpl& sensor,
                      const Eigen::Matrix4f& T_CM,
                      const float desired_dist)
{
    if (!block) {
        return 0.0f;
    }
    const float block_min_dist = block->minDistUpdated();
    // Get the coordinates of the VoxelBlock's centre in the sensor frame C.
    const Eigen::Vector3f block_centre_coord_f = se::get_sample_coord(
        block->coordinates(), VoxelT::VoxelBlockType::size_li, Eigen::Vector3f::Constant(0.5f));
    const Eigen::Vector3f block_centre_M = map.voxelFToPoint(block_centre_coord_f);
    const Eigen::Vector3f block_centre_C = (T_CM * block_centre_M.homogeneous()).head(3);
    const float block_expected_dist = sensor.measurementFromPoint(block_centre_C);
    const float block_half_diag = block->size() * map.voxelDim() * sqrt(3.0f) / 2.0f;
    if (sensor.near_plane <= block_expected_dist - block_half_diag
        && block_expected_dist + block_half_diag <= sensor.far_plane) {
        return dist_gain(block_min_dist, block_expected_dist, desired_dist);
    }
    else {
        return 0.0f;
    }
}



float dist_gain(const float block_min_dist,
                const float block_expected_dist,
                const float desired_dist)
{
    // Test if the desired distance has already been achieved in this VoxelBlock.
    if (block_min_dist <= block_expected_dist) {
        return 0.0f;
    }
    // Test if there's possibility of improving the distance from this view.
    if (block_expected_dist >= block_min_dist) {
        return 0.0f;
    }
    // The following should be true since we reached this place:
    // desired_dist < block_min_dist
    // block_expected_dist < block_min_dist
    // Even if the expected distance is smaller than the desired distance, only count gain up to the
    // desired distance.
    return block_min_dist - std::max(block_expected_dist, desired_dist);
}



Image<float> object_dist_gain(const Image<Eigen::Vector3f>& bg_hits_M,
                              const Objects& objects,
                              const SensorImpl& sensor,
                              const Eigen::Matrix4f& T_MB,
                              const Eigen::Matrix4f& T_BC,
                              const float desired_dist)
{
    Image<float> gain_image(bg_hits_M.width(), bg_hits_M.height(), 0.0f);
    const float max_dist_gain = sensor.far_plane;
    const Eigen::Matrix4f T_MC = T_MB * T_BC;
    const Eigen::Vector3f t_MC = se::math::to_translation(T_MC);
    const Eigen::Matrix4f T_CM = se::math::to_inverse_transformation(T_MC);
#pragma omp parallel for
    for (int y = 0; y < gain_image.height(); ++y) {
#pragma omp simd
        for (int x = 0; x < gain_image.width(); ++x) {
            // The background contains the objects so we can skip raycasting the objects if there is
            // no background hit.
            if (isnan(bg_hits_M(x, y).x())) {
                continue;
            }
            const Eigen::Vector3f ray_dir_M = (bg_hits_M(x, y) - t_MC).normalized();
            const ObjectHit hit = raycast_objects(objects,
                                                  std::map<int, cv::Mat>(),
                                                  Eigen::Vector2f(x, y),
                                                  t_MC,
                                                  ray_dir_M,
                                                  sensor.near_plane,
                                                  sensor.far_plane);
            if (hit.instance_id != ObjectHit().instance_id) {
                const Object& object = *(objects[hit.instance_id]);
                const auto& map = *(object.map_);
                const Eigen::Vector3f hit_O = (object.T_OM_ * hit.hit_M.homogeneous()).head<3>();
                const auto* block = map.fetch(map.pointToVoxel(hit_O));
                const float gain = block_dist_gain(block, map, sensor, T_CM, desired_dist);
                gain_image(x, y) = gain / max_dist_gain;
            }
        }
    }
    return gain_image;
}

} // namespace se
