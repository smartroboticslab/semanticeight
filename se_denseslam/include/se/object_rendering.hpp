/*
 * Created by binbin on 25/04/18.
 * Modified by Sotiris Papatheodorou
 */

#ifndef __OBJECT_RAYCASTING_HPP
#define __OBJECT_RAYCASTING_HPP

#include <Eigen/Dense>
#include <cstdint>
#include <map>
#include <opencv2/opencv.hpp>
#include <set>
#include <vector>

#include "se/image/image.hpp"
#include "se/object.hpp"
#include "se/sensor_implementation.hpp"



enum RenderMode : uint8_t { Color, InstanceID, ClassID, Scale, MinScale, ForegroundProb };



struct ObjectHit {
    int instance_id = se::instance_bg;
    int8_t scale = -1;
    int8_t min_scale = -1;
    Eigen::Vector3f hit_M = Eigen::Vector3f::Zero();
    Eigen::Vector3f normal_M = Eigen::Vector3f(INVALID, 0.f, 0.f);

    bool valid() const
    {
        return instance_id != se::instance_bg;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/** Raycast each object along the ray passing through pixel and return an ObjectHit for the nearest
 * object. If no object is hit, the return struct will contain its default values.
 */
template<typename RaycastF = decltype(ObjVoxelImpl::raycast)>
ObjectHit raycast_objects(const Objects& objects,
                          const std::map<int, cv::Mat>& raycasting_masks,
                          const Eigen::Vector2f pixel,
                          const Eigen::Vector3f& ray_origin_MC,
                          const Eigen::Vector3f& ray_dir_M,
                          const float near_dist,
                          const float far_dist,
                          RaycastF raycast = ObjVoxelImpl::raycast);

/**
 * Raycast each object and create unified vertex and normal maps.
 *
 * \todo There are some missing regions in the render, both with do_raycast and
 * without.
 */
void raycastObjectListKernel(const Objects& objects,
                             const std::set<int>& visible_object_ids,
                             se::Image<Eigen::Vector3f>& surface_point_cloud_M,
                             se::Image<Eigen::Vector3f>& surface_normals_M,
                             cv::Mat& instance_id_image,
                             se::Image<int8_t>& scale_image,
                             se::Image<int8_t>& min_scale_image,
                             const Eigen::Matrix4f& raycast_T_MC,
                             const SensorImpl& sensor,
                             const int frame);

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
                            const RenderMode render_mode);

void overlayBoundingVolumeKernel(uint32_t* output_image_data,
                                 const Eigen::Vector2i& output_image_res,
                                 const Objects& objects,
                                 const Eigen::Matrix4f& T_MC,
                                 const SensorImpl& sensor,
                                 const float opacity);

#include "se/object_rendering_impl.hpp"

#endif
