/*
 * Created by binbin on 25/04/18.
 * Modified by Sotiris Papatheodorou
 */

#include "se/object_rendering.hpp"

#include "se/semanticeight_definitions.hpp"
//#define SE_DEBUG_IMAGES
#include "se/debug_images.hpp"



void raycastObjectListKernel(const Objects& objects,
                             const std::set<int>& visible_objects,
                             se::Image<Eigen::Vector3f>& surface_point_cloud_M,
                             se::Image<Eigen::Vector3f>& surface_normals_M,
                             cv::Mat& instance_id_image,
                             se::Image<int8_t>& scale_image,
                             se::Image<int8_t>& min_scale_image,
                             const Eigen::Matrix4f& raycast_T_MC,
                             const SensorImpl& sensor,
                             const int frame)
{
    TICKD("raycastObjectListKernel");
    const std::vector<int> visible_objects_vec(visible_objects.begin(), visible_objects.end());
#if SE_BOUNDING_VOLUME > 0
    // Compute the visible object raycasting masks based on their bounding volumes.
    std::vector<cv::Mat> raycasting_masks;
    for (const auto& object_id : visible_objects_vec) {
        Object& object = *(objects[object_id]);
        raycasting_masks.push_back(object.bounding_volume_M_.raycastingMask(
            Eigen::Vector2i(surface_point_cloud_M.width(), surface_point_cloud_M.height()),
            raycast_T_MC,
            sensor));
    }
#else
#endif

    dbg::images.init(objects.size(), surface_point_cloud_M.width(), surface_point_cloud_M.height());
#pragma omp parallel for
    for (int y = 0; y < surface_point_cloud_M.height(); ++y) {
#pragma omp simd
        for (int x = 0; x < surface_point_cloud_M.width(); ++x) {
            // Count how many object volumes this ray has missed
            size_t miss_count = 0;

            const size_t pixel_idx = x + surface_point_cloud_M.width() * y;
            const Eigen::Vector2f pixel_f(x, y);
            Eigen::Vector3f ray_dir_C;
            sensor.model.backProject(pixel_f, &ray_dir_C);
            const Eigen::Vector3f ray_dir_M =
                se::math::to_rotation(raycast_T_MC) * ray_dir_C.normalized();
            const Eigen::Vector4f t_MC = se::math::to_translation(raycast_T_MC).homogeneous();

            // Keep the distance to the nearest hit and the respective foreground
            // probability
            float nearest_hit_dist = INFINITY;
            float nearest_hit_prob = 0.0f;

            // Iterate over all visible objects.
            for (size_t i = 0; i < visible_objects_vec.size(); i++) {
                const Object& object = *(objects[visible_objects_vec[i]]);
                const int instance_id = object.instance_id;
                const int class_id = object.classId();

#if SE_BOUNDING_VOLUME > 0
                // Skip pixels outside the bounding volume mask
                if ((instance_id != se::instance_bg)
                    && !raycasting_masks[i].at<se::mask_elem_t>(y, x)) {
                    miss_count++;
                    //std::cout << "Skipping masked" << "\n";
                    dbg::images.set(instance_id, x, y, dbg::raycast_outside_bounding);
                    continue;
                }
#endif

                // Change from map to object frame
                const Eigen::Matrix4f& T_OM = object.T_OM_;
                const Eigen::Vector3f ray_dir_O = se::math::to_rotation(T_OM) * ray_dir_M;
                const Eigen::Vector3f t_OC = (T_OM * t_MC).head(3);
                // Raycast the object
                const Eigen::Vector4f surface_intersection_O =
                    ObjVoxelImpl::raycast(*object.map_,
                                          t_OC,
                                          ray_dir_O,
                                          sensor.nearDist(ray_dir_C),
                                          sensor.farDist(ray_dir_C));

                if (surface_intersection_O.w() >= 0.f) {
                    float hit_distance = (t_OC - surface_intersection_O.head<3>()).norm();
                    // Slightly increase the hit distance for the background and "stuff"
                    // so that the foreground has priority
                    if (class_id == se::semantic_classes.backgroundId()) {
                        hit_distance += object.voxelDim();
                    }
                    else if (!se::semantic_classes.enabled(class_id)) {
                        hit_distance += 0.5f * object.voxelDim();
                    }
                    // Skip hits further than the closest hit
                    if (hit_distance > nearest_hit_dist) {
                        miss_count++;
                        //std::cout << "Skipping far hit" << "\n";
                        dbg::images.set(instance_id, x, y, dbg::raycast_far);
                        continue;
                    }
                    // Compute the foreground probability
                    float fg_prob =
                        object.map_
                            ->interpAtPoint(surface_intersection_O.head<3>(),
                                            [](const auto& data) { return data.getFg(); })
                            .first;
                    // Compute the complement of the probability if this is the
                    // background. This allows comparing it with the foreground
                    // probabilities of objects as it becomes a "valid hit" probability.
                    if (class_id == se::semantic_classes.backgroundId()) {
                        fg_prob = 1.0f - fg_prob;
                    }
                    // Skip hits with low foreground probability, i.e. belonging to the
                    // background
                    if (fg_prob <= 0.5f) {
                        miss_count++;
                        //std::cout << "Skipping FG prob " << fg_prob << " <= 0.5 hit" << "\n";
                        dbg::images.set(instance_id, x, y, dbg::raycast_low_fg);
                        continue;
                    }
                    // Skip hits with the same distance and lower foreground probability
                    if ((hit_distance == nearest_hit_dist) && (fg_prob <= nearest_hit_prob)) {
                        miss_count++;
                        //std::cout << "Skipping lower prob hit" << "\n";
                        dbg::images.set(instance_id, x, y, dbg::raycast_same_dist_lower_fg);
                        continue;
                    }
                    // Good hit found.
                    nearest_hit_prob = fg_prob;
                    nearest_hit_dist = hit_distance;
                    const Eigen::Matrix4f T_MO = se::math::to_inverse_transformation(T_OM);
                    // Update the point cloud
                    surface_point_cloud_M[pixel_idx] =
                        (T_MO * surface_intersection_O.head(3).homogeneous()).head(3);
                    // Update the normals
                    Eigen::Vector3f surface_normal_O = object.map_->gradAtPoint(
                        surface_intersection_O.head<3>(),
                        ObjVoxelImpl::VoxelType::selectNodeValue,
                        ObjVoxelImpl::VoxelType::selectVoxelValue,
                        static_cast<int>(surface_intersection_O.w() + 0.5f));
                    if (surface_normal_O.norm() == 0.f) {
                        surface_normals_M[pixel_idx] = Eigen::Vector3f(INVALID, 0.f, 0.f);
                    }
                    else {
                        // Invert normals for TSDF representations.
                        surface_normals_M[pixel_idx] = se::math::to_rotation(T_MO)
                            * (ObjVoxelImpl::invert_normals ? (-1.f * surface_normal_O).normalized()
                                                            : surface_normal_O.normalized())
                                  .head(3);
                    }
                    // Update the instance mask
                    instance_id_image.at<se::instance_mask_elem_t>(y, x) = instance_id;
                    scale_image[pixel_idx] = static_cast<int8_t>(surface_intersection_O.w());
                    // Fetch the VoxelBlock containing the hit and get its minimum updated scale.
                    // TODO SEM not sure why block would ever be nullptr if we got a valid hit but it happens.
                    const auto* block = object.map_->fetch(
                        object.map_->pointToVoxel(surface_intersection_O.head<3>()));
                    if (block) {
                        min_scale_image[pixel_idx] = block->min_scale();
                    }
                    else {
                        min_scale_image[pixel_idx] = -1;
                    }
                    //std::cout << "Hit!" << "\n";
                    dbg::images.set(instance_id, x, y, dbg::raycast_ok);
                }
                else {
                    // No hit was made
                    miss_count++;
                    continue;
                }
            }

            // All objects were missed.
            if (miss_count == visible_objects_vec.size()) {
                surface_point_cloud_M[pixel_idx] = Eigen::Vector3f::Zero();
                surface_normals_M[pixel_idx] = Eigen::Vector3f(INVALID, 0.f, 0.f);
                instance_id_image.at<se::instance_mask_elem_t>(y, x) = se::instance_bg;
                scale_image[pixel_idx] = -1;
                min_scale_image[pixel_idx] = -1;
            }
        }
    }
    std::stringstream p;
    p << "object_raycast_" << std::setw(5) << std::setfill('0') << frame;
    dbg::images.save("/home/srl/renders", p.str());
    TOCK("raycastObjectListKernel");
}



void renderObjectListKernel(uint32_t* output_image_data,
                            const Eigen::Vector2i& output_image_res,
                            const Eigen::Vector3f& light_M,
                            const Eigen::Vector3f& ambient_M,
                            const Objects& objects,
                            const se::Image<Eigen::Vector3f>& object_point_cloud_M,
                            const se::Image<Eigen::Vector3f>& object_normals_M,
                            const cv::Mat& instance_id_image,
                            const se::Image<int8_t>& scale_image,
                            const se::Image<int8_t>& min_scale_image,
                            const RenderMode render_mode)
{
    TICKD("raycastObjectListKernel");
#pragma omp parallel for
    for (int y = 0; y < output_image_res.y(); ++y) {
#pragma omp simd
        for (int x = 0; x < output_image_res.x(); ++x) {
            const size_t pixel_idx = x + output_image_res.x() * y;

            const Eigen::Vector3f& object_point_M = object_point_cloud_M[pixel_idx];
            const Eigen::Vector3f& object_normal_M = object_normals_M[pixel_idx];
            const int instance_id = instance_id_image.at<se::instance_mask_elem_t>(y, x);

            // Iterate over all objects to find the one with the instance ID returned
            // by raycasting.
            for (const auto& object_ptr : objects) {
                Object& object = *object_ptr;
                if (instance_id == object.instance_id) {
                    if (object_normal_M.x() != INVALID && object_normal_M.norm() > 0.f) {
                        const Eigen::Vector3f diff = (object_point_M - light_M).normalized();
                        const Eigen::Vector3f dir = Eigen::Vector3f::Constant(
                            fmaxf(object_normal_M.normalized().dot(diff), 0.f));

                        Eigen::Vector3f col = Eigen::Vector3f::Constant(0.0f);

                        switch (render_mode) {
                        case RenderMode::Color: {
                            // Convert the hit to object coordinates to use for real color
                            // inerpolation.
                            const Eigen::Matrix4f& T_OM = object.T_OM_;
                            const Eigen::Vector3f point_O =
                                (T_OM * object_point_M.homogeneous()).head<3>();
                            const float interp_r =
                                object.map_
                                    ->interpAtPoint(point_O,
                                                    [](const auto& data) { return data.r; })
                                    .first;
                            const float interp_g =
                                object.map_
                                    ->interpAtPoint(point_O,
                                                    [](const auto& data) { return data.g; })
                                    .first;
                            const float interp_b =
                                object.map_
                                    ->interpAtPoint(point_O,
                                                    [](const auto& data) { return data.b; })
                                    .first;
                            const Eigen::Vector3f rgb = Eigen::Vector3f(
                                interp_r / 255.0f, interp_g / 255.0f, interp_b / 255.0f);
                            col = dir.cwiseProduct(rgb) + ambient_M;
                        } break;
                        case RenderMode::InstanceID:
                            if (instance_id == se::instance_bg) {
                                // No color for background.
                                col = dir + ambient_M;
                            }
                            else {
                                // Colored objects.
                                const Eigen::Vector3f rgb =
                                    se::internal::color_map[instance_id
                                                            % se::internal::color_map.size()]
                                    / 255.0f;
                                col = dir.cwiseProduct(rgb) + ambient_M;
                            }
                            break;
                        case RenderMode::ClassID: {
                            const int class_id = object.classId();
                            if (class_id == se::semantic_classes.backgroundId()) {
                                // No color for background.
                                col = dir + ambient_M;
                            }
                            else {
                                // Colored objects.
                                const Eigen::Vector3f rgb =
                                    se::internal::color_map[class_id
                                                            % se::internal::color_map.size()]
                                    / 255.0f;
                                col = dir.cwiseProduct(rgb) + ambient_M;
                            }
                        } break;
                        case RenderMode::Scale: {
                            const int8_t scale = scale_image[pixel_idx];
                            if (scale >= 0) {
                                col = (dir + ambient_M)
                                          .cwiseProduct(se::internal::color_map[scale] / 255.0f);
                            }
                            else {
                                col = Eigen::Vector3f::Zero();
                            }
                        } break;
                        case RenderMode::MinScale: {
                            const int8_t scale = min_scale_image[pixel_idx];
                            if (scale >= 0) {
                                col = (dir + ambient_M)
                                          .cwiseProduct(se::internal::color_map[scale] / 255.0f);
                            }
                            else {
                                col = Eigen::Vector3f::Zero();
                            }
                        } break;
                        case RenderMode::ForegroundProb:
                            const Eigen::Matrix4f& T_OM = object.T_OM_;
                            const Eigen::Vector3f point_O =
                                (T_OM * object_point_M.homogeneous()).head<3>();
                            const float interp_fg =
                                object.map_
                                    ->interpAtPoint(point_O,
                                                    [](const auto& data) { return data.getFg(); })
                                    .first;
                            if (0.0f <= interp_fg && interp_fg <= 1.0f) {
                                col = (dir + ambient_M)
                                          .cwiseProduct(
                                              Eigen::Vector3f(interp_fg, 0.0f, 1.0f - interp_fg));
                            }
                            else {
                                col = Eigen::Vector3f::Zero();
                            }
                            break;
                        }
                        se::math::clamp(col, Eigen::Vector3f::Zero(), Eigen::Vector3f::Ones());
                        col *= 255.0f;
                        output_image_data[pixel_idx] =
                            se::pack_rgba(col.x(), col.y(), col.z(), 0xFF);
                    }
                    // Found the object
                    break;
                }
            }
        }
    }
    TOCK("renderObjectListKernel");
}



void overlayBoundingVolumeKernel(uint32_t* output_image_data,
                                 const Eigen::Vector2i& output_image_res,
                                 const Objects& objects,
                                 const Eigen::Matrix4f& T_MC,
                                 const SensorImpl& sensor,
                                 const float opacity)
{
    for (size_t i = 0; i < objects.size(); ++i) {
#if SE_BOUNDING_VOLUME > 0
        objects[i]->bounding_volume_M_.overlay(
            output_image_data, output_image_res, T_MC, sensor, opacity);
#else
#endif
    }
}
