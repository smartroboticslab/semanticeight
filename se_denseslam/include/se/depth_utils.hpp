/*
 * Copyright (C) 2020 Sotiris Papatheodorou
 */

#ifndef __DEPTH_UTILS_HPP
#define __DEPTH_UTILS_HPP

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

#include "se/image/image.hpp"
#include "se/segmentation_result.hpp"
#include "se/sensor_implementation.hpp"



namespace se {
/**
   * Compute the error between two depth images.
   * The error is computed as the sum of the squared difference in depth
   * values for each pixel. The squared error between depth values is
   * computed only at pixels for which the mask is nonzero. Invalid depth
   * measurements (with a value of 0) are not considered. If the error
   * returned is -1 then no pixels were compared.
   */
inline float depth_image_error(const cv::Mat& depth_1, const cv::Mat& depth_2, const cv::Mat& mask)
{
    assert((depth_1.type() == CV_32FC1)
           && "Error: depth_1 must be a float image with the depth in meters");
    assert((depth_2.type() == CV_32FC1)
           && "Error: depth_2 must be a float image with the depth in meters");
    assert((depth_1.cols == depth_2.cols) && "Error: depth_1 and depth_2 must have the same width");
    assert((depth_1.rows == depth_2.rows)
           && "Error: depth_1 and depth_2 must have the same height");
    assert((depth_1.cols == mask.cols) && "Error: depth_1 and mask must have the same width");
    assert((depth_1.rows == mask.rows) && "Error: depth_1 and mask must have the same height");

    float error = 0.f;
    bool error_updated = false;

    for (size_t p = 0; p < depth_1.total(); p++) {
        // Only compare depth values inside the mask.
        if (mask.at<uint8_t>(p)) {
            // Only compare valid depth measurements.
            if ((depth_1.at<float>(p) != 0.f) && (depth_2.at<float>(p) != 0.f)) {
                // Compute the squared depth error.
                const float diff = depth_1.at<float>(p) - depth_2.at<float>(p);
                error += diff * diff;
                error_updated = true;
            }
        }
    }
    if (!error_updated) {
        error = -1.f;
    }
    return error;
}



inline float depth_image_error(const se::Image<float>& depth_1,
                               const se::Image<float>& depth_2,
                               const cv::Mat& mask)
{
    // Need to cast away the const because OpenCV does not have a constructor
    // accepting a pointer to const data. It should be safe since the created
    // Mats are const.
    const cv::Mat depth_1_cv(
        depth_1.height(), depth_1.width(), CV_32FC1, const_cast<float*>(depth_1.data()));
    const cv::Mat depth_2_cv(
        depth_2.height(), depth_2.width(), CV_32FC1, const_cast<float*>(depth_2.data()));
    return depth_image_error(depth_1_cv, depth_2_cv, mask);
}



/**
   * Perform segmentation based on geometric edges. This results in masks for
   * contiguous convex regions with no depth discontinuities. This tends to
   * oversegment objects so the resulting masks should be merged with
   * semantic masks.
   */
se::SegmentationResult geometric_segmentation(const se::Image<float>& depth,
                                              const SensorImpl& sensor);

/** Return a mask for the parts of the image where the scene background is occluding all
   * objects, i.e. it's in the foreground.
   */
cv::Mat occlusion_mask(const se::Image<float>& object_depth,
                       const se::Image<float>& background_depth,
                       const float background_voxel_dim);

cv::Mat occlusion_mask(const se::Image<Eigen::Vector3f>& object_point_cloud_M,
                       const se::Image<Eigen::Vector3f>& background_point_cloud_M,
                       const float background_voxel_dim,
                       const Eigen::Matrix4f& T_CM);

void add_depth_measurement_noise(se::Image<float>& depth,
                                 const float k_sigma,
                                 const float min_sigma,
                                 const float max_sigma);

} // namespace se

#endif // __DEPTH_UTILS_HPP
