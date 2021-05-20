/*
 * Created by binbin on 25/04/18.
 * Modified by Sotiris Papatheodorou
 */

#ifndef __OBJECT_RAYCASTING_HPP
#define __OBJECT_RAYCASTING_HPP

#include <cstdint>
#include <set>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "se/image/image.hpp"
#include "se/sensor_implementation.hpp"
#include "se/object.hpp"



enum RenderMode : uint8_t {
  Color,
  InstanceID,
  ClassID,
  Scale,
  MinScale,
  ForegroundProb
};

/**
 * Raycast each object and create unified vertex and normal maps.
 *
 * \todo There are some missing regions in the render, both with do_raycast and
 * without.
 */
void raycastObjectListKernel(const Objects&              objects,
                             const std::set<int>&        visible_objects,
                             se::Image<Eigen::Vector3f>& surface_point_cloud_M,
                             se::Image<Eigen::Vector3f>& surface_normals_M,
                             cv::Mat&                    instance_id_image,
                             se::Image<int8_t>&          scale_image,
                             se::Image<int8_t>&          min_scale_image,
                             const Eigen::Matrix4f&      raycast_T_MC,
                             const SensorImpl&           sensor);

void renderObjectListKernel(uint32_t*                         output_image_data,
                            const Eigen::Vector2i&            output_image_res,
                            const Eigen::Vector3f&            light_M,
                            const Eigen::Vector3f&            ambient_M,
                            const Objects&                    objects,
                            const se::Image<Eigen::Vector3f>& object_point_cloud_M,
                            const se::Image<Eigen::Vector3f>& object_normals_M,
                            const cv::Mat&                    instance_id_image,
                            const se::Image<int8_t>&          scale_image,
                            const se::Image<int8_t>&          min_scale_image,
                            const RenderMode                  render_mode);

void overlayBoundingVolumeKernel(uint32_t*              output_image_data,
                                 const Eigen::Vector2i& output_image_res,
                                 const Objects&         objects,
                                 const Eigen::Matrix4f& T_MC,
                                 const SensorImpl&      sensor,
                                 const float            opacity);

#endif

