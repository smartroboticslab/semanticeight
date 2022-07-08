/*
 * Created by binbin on 25/04/18.
 * Modified by Sotiris Papatheodorou
 */

#include "se/object_rendering.hpp"

#include "se/semanticeight_definitions.hpp"
//#define SE_DEBUG_IMAGES
#include "se/debug_images.hpp"



void raycastObjectListKernel(const Objects& objects,
                             const std::set<int>& visible_object_ids,
                             se::Image<Eigen::Vector3f>& surface_point_cloud_M,
                             se::Image<Eigen::Vector3f>& surface_normals_M,
                             cv::Mat& instance_id_image,
                             se::Image<int8_t>& scale_image,
                             se::Image<int8_t>& min_scale_image,
                             const Eigen::Matrix4f& raycast_T_MC,
                             const SensorImpl& sensor,
                             const int /* frame */)
{
    TICKD("raycastObjectListKernel");
    Objects visible_objects;
    visible_objects.reserve(visible_object_ids.size());
    std::transform(visible_object_ids.begin(),
                   visible_object_ids.end(),
                   std::back_inserter(visible_objects),
                   [&](int id) { return objects[id]; });

    std::map<int, cv::Mat> raycasting_masks;
#if SE_BOUNDING_VOLUME > 0
    // Compute the visible object raycasting masks based on their bounding volumes.
    for (const auto& o : visible_objects) {
        raycasting_masks[o->instance_id] = o->bounding_volume_M_.raycastingMask(
            Eigen::Vector2i(surface_point_cloud_M.width(), surface_point_cloud_M.height()),
            raycast_T_MC,
            sensor);
    }
#endif

    const Eigen::Vector3f t_MC = se::math::to_translation(raycast_T_MC);
    const Eigen::Matrix3f C_MC = se::math::to_rotation(raycast_T_MC);
#pragma omp parallel for
    for (int y = 0; y < surface_point_cloud_M.height(); ++y) {
#pragma omp simd
        for (int x = 0; x < surface_point_cloud_M.width(); ++x) {
            const size_t pixel_idx = x + surface_point_cloud_M.width() * y;
            const Eigen::Vector2f pixel(x, y);
            Eigen::Vector3f ray_dir_C;
            sensor.model.backProject(pixel, &ray_dir_C);
            const Eigen::Vector3f ray_dir_M = C_MC * ray_dir_C.normalized();
            const ObjectHit hit = raycast_objects(visible_objects,
                                                  raycasting_masks,
                                                  pixel,
                                                  t_MC,
                                                  ray_dir_M,
                                                  sensor.nearDist(ray_dir_C),
                                                  sensor.farDist(ray_dir_C));
            instance_id_image.at<se::instance_mask_elem_t>(y, x) = hit.instance_id;
            scale_image[pixel_idx] = hit.scale;
            min_scale_image[pixel_idx] = hit.min_scale;
            surface_point_cloud_M[pixel_idx] = hit.hit_M;
            surface_normals_M[pixel_idx] = hit.normal_M;
        }
    }
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
        const Eigen::Vector3f rgb =
            se::internal::color_map[objects[i]->instance_id % se::internal::color_map.size()];
        objects[i]->bounding_volume_M_.overlay(output_image_data,
                                               output_image_res,
                                               T_MC,
                                               sensor,
                                               cv::Scalar(rgb.x(), rgb.y(), rgb.z(), 255),
                                               opacity);
#else
#endif
    }
}
