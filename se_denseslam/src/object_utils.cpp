// SPDX-FileCopyrightText: 2021 Smart Robotics Lab
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#include "se/object_utils.hpp"

#include <algorithm>
#include <numeric>
#include <opencv2/opencv.hpp>

#include "se/object_rendering.hpp"

namespace se {

std::set<int> get_visible_object_ids(const Objects& objects,
                                     const SensorImpl& sensor,
                                     const Eigen::Matrix4f& T_MC)
{
    std::set<int> v;
    // Iterate over all current objects.
    for (size_t i = 0; i < objects.size(); ++i) {
        const Object& object = *(objects[i]);
#if SE_BOUNDING_VOLUME != SE_BV_NONE
        // Add the instance ID to the visible_objects if the test was successful.
        if (object.bounding_volume_M_.isVisible(T_MC, sensor)) {
            v.insert(object.instance_id);
        }
#else
        // Consider all objects visible if there's no bounding volume.
        v.insert(object.instance_id);
#endif
    }
    return v;
}



Objects filter_visible_objects(const Objects& objects, const std::set<int>& visible_object_ids)
{
    Objects visible_objects;
    const auto push_back_visible = [&](const auto& o) {
        if (visible_object_ids.count(o->instance_id) == 1) {
            visible_objects.push_back(o);
        }
    };
    std::for_each(objects.begin(), objects.end(), push_back_visible);
    return visible_objects;
}



std::vector<float> object_lod_gain_blocks(const Objects& objects,
                                          const SensorImpl& sensor,
                                          const Eigen::Matrix4f& T_MC)
{
    std::vector<float> gains(objects.size(), 0.0f);
    // Update the gains of all visible objects.
    const std::set<int> visible_objects = get_visible_object_ids(objects, sensor, T_MC);
    for (size_t i = 0; i < objects.size(); i++) {
        const Object& o = *objects[i];
        if (visible_objects.count(o.instance_id) == 1) {
            // The gain is the percentage of blocks whose min scale is not 0.
            const int num_blocks = std::accumulate(
                o.num_blocks_per_min_scale.begin(), o.num_blocks_per_min_scale.end(), 0);
            gains[i] =
                (num_blocks - o.num_blocks_per_min_scale[0]) / static_cast<float>(num_blocks);
        }
    }
    return gains;
}



std::vector<float> object_lod_gain_raycasting(const Objects& objects,
                                              const SensorImpl& sensor,
                                              const Eigen::Matrix4f& T_MC)
{
    std::vector<float> gains(objects.size(), 0.0f);
    std::vector<float> num_pixels(objects.size(), 0.0f);
    const std::set<int> vo = get_visible_object_ids(objects, sensor, T_MC);
    const std::vector<int> visible_objects(vo.begin(), vo.end());
    const Eigen::Vector2i image_res(sensor.model.imageWidth(), sensor.model.imageHeight());
#if SE_BOUNDING_VOLUME > 0
    // Compute the visible object raycasting masks based on their bounding volumes.
    std::vector<cv::Mat> raycasting_masks;
    for (const auto object_id : visible_objects) {
        raycasting_masks.push_back(
            objects[object_id]->bounding_volume_M_.raycastingMask(image_res, T_MC, sensor));
    }
#endif
#pragma omp parallel for
    for (int y = 0; y < image_res.y(); ++y) {
#pragma omp simd
        for (int x = 0; x < image_res.x(); ++x) {
            // Compute the ray for this pixel.
            const Eigen::Vector2f pixel_f(x, y);
            Eigen::Vector3f ray_dir_C;
            sensor.model.backProject(pixel_f, &ray_dir_C);
            const Eigen::Vector3f ray_dir_M = se::math::to_rotation(T_MC) * ray_dir_C.normalized();
            const Eigen::Vector4f t_MC = se::math::to_translation(T_MC).homogeneous();

            // Bookkeeping for finding the scale and instance ID of the nearest object hit.
            float nearest_hit_dist = INFINITY;
            float nearest_hit_prob = 0.0f;
            int nearest_hit_min_scale = -1;
            int nearest_hit_instance_id = se::instance_invalid;

            // Iterate over all visible objects.
            for (size_t i = 0; i < visible_objects.size(); i++) {
                const Object& object = *objects[visible_objects[i]];

#if SE_BOUNDING_VOLUME > 0
                // Skip pixels outside the bounding volume mask.
                if (!raycasting_masks[i].at<se::mask_elem_t>(y, x)) {
                    continue;
                }
#endif

                // Change from the map to the object frame.
                const Eigen::Matrix4f& T_OM = object.T_OM_;
                const Eigen::Vector3f ray_dir_O = se::math::to_rotation(T_OM) * ray_dir_M;
                const Eigen::Vector3f t_OC = (T_OM * t_MC).head(3);
                // Raycast the object.
                const Eigen::Vector4f hit_O = ObjVoxelImpl::raycast(*object.map_,
                                                                    t_OC,
                                                                    ray_dir_O,
                                                                    sensor.nearDist(ray_dir_C),
                                                                    sensor.farDist(ray_dir_C));
                if (hit_O.w() >= 0.f) {
                    float hit_distance = (t_OC - hit_O.head<3>()).norm();
                    // Slightly increase the hit distance for the background and "stuff"
                    // so that the foreground has priority
                    if (object.classId() == se::semantic_classes.backgroundId()) {
                        hit_distance += object.voxelDim();
                    }
                    else if (!se::semantic_classes.enabled(object.classId())) {
                        hit_distance += 0.5f * object.voxelDim();
                    }
                    // Skip hits further than the closest hit
                    if (hit_distance > nearest_hit_dist) {
                        continue;
                    }
                    // Compute the foreground probability
                    float fg_prob =
                        object.map_
                            ->interpAtPoint(hit_O.head<3>(),
                                            [](const auto& data) { return data.getFg(); })
                            .first;
                    // Compute the complement of the probability if this is the
                    // background. This allows comparing it with the foreground
                    // probabilities of objects as it becomes a "valid hit" probability.
                    if (object.classId() == se::semantic_classes.backgroundId()) {
                        fg_prob = 1.0f - fg_prob;
                    }
                    // Skip hits with low foreground probability, i.e. belonging to the
                    // background
                    if (fg_prob <= 0.5f) {
                        continue;
                    }
                    // Skip hits with the same distance and lower foreground probability
                    if ((hit_distance == nearest_hit_dist) && (fg_prob <= nearest_hit_prob)) {
                        continue;
                    }
                    // Good hit found.
                    nearest_hit_prob = fg_prob;
                    nearest_hit_dist = hit_distance;
                    nearest_hit_instance_id = object.instance_id;
                    // Fetch the VoxelBlock containing the hit and get its minimum updated scale.
                    // TODO SEM Not sure why block would ever be nullptr if we got a valid hit but it
                    // happens.
                    const auto* block =
                        object.map_->fetch(object.map_->pointToVoxel(hit_O.head<3>()));
                    if (block) {
                        nearest_hit_min_scale = block->min_scale();
                    }
                    else {
                        nearest_hit_min_scale = -1;
                    }
                }
            }
            // Update the gain of the nearest object hit.
            if (nearest_hit_instance_id != se::instance_invalid) {
                if (nearest_hit_min_scale > 0) {
                    gains[nearest_hit_instance_id]++;
                }
                num_pixels[nearest_hit_instance_id]++;
            }
        }
    }
    // Normalize the object gains in the interval [0-1] by dividing the number of pixels with scale
    // greater than 0 with the total number of pixels.
    for (size_t i = 0; i < gains.size(); i++) {
        if (num_pixels[i] > 0.0f) {
            gains[i] /= num_pixels[i];
        }
    }
    return gains;
}



float lod_gain_raycasting(const Objects& objects,
                          const SensorImpl& sensor,
                          const SensorImpl& raycasting_sensor,
                          const Eigen::Matrix4f& T_MC,
                          Image<int8_t>& min_scale_image)
{
    size_t num_nonzero_scale = 0;
    const Eigen::Matrix4f T_CM = se::math::to_inverse_transformation(T_MC);
    const std::set<int> vo = get_visible_object_ids(objects, raycasting_sensor, T_MC);
    const std::vector<int> visible_objects(vo.begin(), vo.end());
    const Eigen::Vector2i image_res(raycasting_sensor.model.imageWidth(),
                                    raycasting_sensor.model.imageHeight());
    if (min_scale_image.width() != image_res.x() || min_scale_image.height() != image_res.y()) {
        min_scale_image = Image<int8_t>(image_res.x(), image_res.y());
    }
#if SE_BOUNDING_VOLUME > 0
    // Compute the visible object raycasting masks based on their bounding volumes.
    std::vector<cv::Mat> raycasting_masks;
    for (const auto object_id : visible_objects) {
        raycasting_masks.push_back(objects[object_id]->bounding_volume_M_.raycastingMask(
            image_res, T_MC, raycasting_sensor));
    }
#endif
#pragma omp parallel for
    for (int y = 0; y < image_res.y(); ++y) {
#pragma omp simd
        for (int x = 0; x < image_res.x(); ++x) {
            // Initialize the image to an invalid scale.
            min_scale_image(x, y) = -1;
            // Compute the ray for this pixel.
            const Eigen::Vector2f pixel_f(x, y);
            Eigen::Vector3f ray_dir_C;
            raycasting_sensor.model.backProject(pixel_f, &ray_dir_C);
            const Eigen::Vector3f ray_dir_M = se::math::to_rotation(T_MC) * ray_dir_C.normalized();
            const Eigen::Vector4f t_MC = se::math::to_translation(T_MC).homogeneous();

            // Bookkeeping for finding the scale and instance ID of the nearest object hit.
            float nearest_hit_dist = INFINITY;
            float nearest_hit_prob = 0.0f;
            int nearest_hit_min_scale = -1;
            int nearest_hit_expected_scale = -1;
            int nearest_hit_instance_id = se::instance_invalid;

            // Iterate over all visible objects.
            for (size_t i = 0; i < visible_objects.size(); i++) {
                const Object& object = *objects[visible_objects[i]];

#if SE_BOUNDING_VOLUME > 0
                // Skip pixels outside the bounding volume mask.
                if (!raycasting_masks[i].at<se::mask_elem_t>(y, x)) {
                    continue;
                }
#endif

                // Change from the map to the object frame.
                const Eigen::Matrix4f& T_OM = object.T_OM_;
                const Eigen::Vector3f ray_dir_O = se::math::to_rotation(T_OM) * ray_dir_M;
                const Eigen::Vector3f t_OC = (T_OM * t_MC).head(3);
                // Raycast the object.
                const Eigen::Vector4f hit_O =
                    ObjVoxelImpl::raycast(*object.map_,
                                          t_OC,
                                          ray_dir_O,
                                          raycasting_sensor.nearDist(ray_dir_C),
                                          raycasting_sensor.farDist(ray_dir_C));
                if (hit_O.w() >= 0.f) {
                    float hit_distance = (t_OC - hit_O.head<3>()).norm();
                    // Slightly increase the hit distance for the background and "stuff"
                    // so that the foreground has priority
                    if (object.classId() == se::semantic_classes.backgroundId()) {
                        hit_distance += object.voxelDim();
                    }
                    else if (!se::semantic_classes.enabled(object.classId())) {
                        hit_distance += 0.5f * object.voxelDim();
                    }
                    // Skip hits further than the closest hit
                    if (hit_distance > nearest_hit_dist) {
                        continue;
                    }
                    // Compute the foreground probability
                    float fg_prob =
                        object.map_
                            ->interpAtPoint(hit_O.head<3>(),
                                            [](const auto& data) { return data.getFg(); })
                            .first;
                    // Compute the complement of the probability if this is the
                    // background. This allows comparing it with the foreground
                    // probabilities of objects as it becomes a "valid hit" probability.
                    if (object.classId() == se::semantic_classes.backgroundId()) {
                        fg_prob = 1.0f - fg_prob;
                    }
                    // Skip hits with low foreground probability, i.e. belonging to the
                    // background
                    if (fg_prob <= 0.5f) {
                        continue;
                    }
                    // Skip hits with the same distance and lower foreground probability
                    if ((hit_distance == nearest_hit_dist) && (fg_prob <= nearest_hit_prob)) {
                        continue;
                    }
                    // Good hit found.
                    nearest_hit_prob = fg_prob;
                    nearest_hit_dist = hit_distance;
                    nearest_hit_instance_id = object.instance_id;
                    // Fetch the VoxelBlock containing the hit and get its minimum updated scale.
                    // TODO SEM Not sure why block would ever be nullptr if we got a valid hit but it
                    // happens.
                    const auto* block =
                        object.map_->fetch(object.map_->pointToVoxel(hit_O.head<3>()));
                    if (block) {
                        // Get the coordinates of the VoxelBlock's centre in the sensor frame C.
                        const Eigen::Vector3f block_centre_coord_f =
                            se::get_sample_coord(block->coordinates(),
                                                 ObjVoxelImpl::VoxelBlockType::size_li,
                                                 Eigen::Vector3f::Constant(0.5f));
                        const Eigen::Vector3f block_centre_O =
                            object.map_->voxelFToPoint(block_centre_coord_f);
                        const Eigen::Vector3f block_centre_C =
                            (T_CM * object.T_MO_ * block_centre_O.homogeneous()).head(3);
                        // Keep track of the expected scale when this VoxelBlock is observed and its current
                        // minimum scale.
                        nearest_hit_expected_scale =
                            sensor.computeIntegrationScale(block_centre_C,
                                                           object.map_->voxelDim(),
                                                           block->current_scale(),
                                                           block->min_scale(),
                                                           ObjVoxelImpl::VoxelBlockType::max_scale);
                        nearest_hit_min_scale = block->min_scale();
                    }
                    else {
                        nearest_hit_expected_scale = -1;
                        nearest_hit_min_scale = -1;
                    }
                }
            }
            // Update the gain of the nearest object hit.
            if (nearest_hit_instance_id != se::instance_invalid) {
                if (nearest_hit_expected_scale < nearest_hit_min_scale) {
                    min_scale_image(x, y) = nearest_hit_min_scale;
                    num_nonzero_scale++;
                }
            }
        }
    }
    return num_nonzero_scale / static_cast<float>(image_res.prod());
}



Object::ScaleArray<float> combinedPercentageAtScale(const Objects& objects)
{
    // Accumulate the per-scale blocks of all objects.
    Object::ScaleArray<size_t> num_blocks_per_min_scale{};
    for (const auto& o : objects) {
        std::transform(num_blocks_per_min_scale.cbegin(),
                       num_blocks_per_min_scale.cend(),
                       o->num_blocks_per_min_scale.cbegin(),
                       num_blocks_per_min_scale.begin(),
                       std::plus<>{});
    }
    // Compute the percentages.
    const size_t num_blocks = std::accumulate(
        num_blocks_per_min_scale.cbegin(), num_blocks_per_min_scale.cend(), static_cast<size_t>(0));
    Object::ScaleArray<float> pc{};
    if (num_blocks != 0) {
        std::transform(num_blocks_per_min_scale.cbegin(),
                       num_blocks_per_min_scale.cend(),
                       pc.begin(),
                       [num_blocks](auto b) { return 100.0f * b / num_blocks; });
    }
    return pc;
}

} // namespace se
