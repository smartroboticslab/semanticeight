/*
 * Copyright (C) 2020 Sotiris Papatheodorou
 */

#include "se/depth_utils.hpp"

#include <algorithm>

#include "se/commons.h"
#include "se/preprocessing.hpp"



namespace se {
  static constexpr float geometric_threshold = 0.02f; // edgeness
  static constexpr float geometric_lambda = 0.1f;
  static constexpr int geometric_component_size = 0;
  static constexpr size_t morph_diam = 3;



  float calc_distance(const se::Image<Eigen::Vector3f>& point_cloud,
                      const se::Image<Eigen::Vector3f>& normals,
                      const size_t                      idx1,
                      const size_t                      idx2) {

    if (   normals[idx1] == Eigen::Vector3f::Constant(INVALID)
        || normals[idx2] == Eigen::Vector3f::Constant(INVALID)) {
      return -1.f;
    } else {
      return normals[idx1].transpose() * (point_cloud[idx2] - point_cloud[idx1]);
    }
  }





  float calc_concavity(const se::Image<Eigen::Vector3f>& normals,
                       const size_t                      idx1,
                       const size_t                      idx2) {

    if (   normals[idx1] == Eigen::Vector3f::Constant(INVALID)
        || normals[idx2] == Eigen::Vector3f::Constant(INVALID)) {
      return -1.f;
    } else {
      return 1.f - normals[idx1].transpose() * normals[idx2];
    }
  }


  /**
   * Compute the geometric edges from the depth image. Adds edges to regions
   * with depth discontinuities and concavities.
   *
   * \note See Martin Runz, Lourdes Agapito, 'MaskFusion: Real-Time
   * Recognition, Tracking and Reconstruction of Multiple Moving Objects'
   * Section 5.2 for more details. The way the lambda parameter was slightly
   * changed.
   *
   * \todo Generate the mask directly, see compute_geom_edges().
   */
  void geometric_edge_kernel(se::Image<uint8_t>&               is_edge,
                             const se::Image<Eigen::Vector3f>& point_cloud,
                             const se::Image<Eigen::Vector3f>& normals,
                             const float                       lambda,
                             const float                       threshold) {

    const Eigen::Vector2i input_size (point_cloud.width(), point_cloud.height());
#pragma omp parallel for
    for (int y = 0; y < input_size.y(); y++) {
      for (int x = 0; x < input_size.x(); x++) {
        const size_t pixel_idx = x + y * input_size.x();

        // Skip boundary
        if (   (x == 0) || (x == input_size.x() - 1)
            || (y == 0) || (y == input_size.y() - 1)) {
          is_edge[pixel_idx] = 0;
          continue;
        }

        // Compute distance and concavity with the 8 surrounding vertices.
        std::vector<float> distance;
        distance.reserve(8);
        std::vector<float> concavity;
        concavity.reserve(8);
        for (int i = x-1; i <= x+1; ++i) {
          for (int j = y-1; j <= y+1; ++j) {
            if (   (i != x) && (j != y)
                && (i >= 0) && (i < input_size.x())
                && (j >= 0) && (j < input_size.y())) {

              const int neighbor_idx = i + j * input_size.x();

              const float neighbor_distance
                  = calc_distance(point_cloud, normals, pixel_idx, neighbor_idx);
              distance.push_back(std::abs(neighbor_distance));

              const float neighbor_concavity = neighbor_distance >= 0
                  ? calc_concavity(normals, pixel_idx, neighbor_idx)
                  : 0;
              concavity.push_back(neighbor_concavity);
            }
          }
        }

        // Compute edgeness using the max distance and concavity of the
        // surrounding vertices.
        float edgeness = threshold;
        if (distance.size() > 0) {
          const float max_concavity = *std::max_element(concavity.begin(), concavity.end());
          const float max_distance = *std::max_element(distance.begin(), distance.end());
          edgeness = max_distance + lambda * max_concavity;
        }

        // Edges are located in regions with large concavity or depth
        // discontinuities.
        if (edgeness > threshold) {
          is_edge[pixel_idx] = UINT8_MAX;
        } else {
          is_edge[pixel_idx] = 0;
        }
      }
    }
  }





  se::SegmentationResult geometric_segmentation(const se::Image<float>& depth,
                                                const SensorImpl&       sensor) {

    const Eigen::Vector2i depth_res (depth.width(), depth.height());

    // Create point cloud and normals from depth image
    se::Image<Eigen::Vector3f> point_cloud (depth_res.x(), depth_res.y());
    se::Image<Eigen::Vector3f> normals (depth_res.x(), depth_res.y());
    depthToPointCloudKernel(point_cloud, depth, sensor);
    if (sensor.left_hand_frame) {
      pointCloudToNormalKernel<true>(normals, point_cloud);
    } else {
      pointCloudToNormalKernel<false>(normals, point_cloud);
    }

    // Compute the geometric edges.
    se::Image<uint8_t> is_edge (depth_res.x(), depth_res.y());
    geometric_edge_kernel(is_edge, point_cloud, normals, geometric_lambda, geometric_threshold);

    // Invert the edge mask to get a convex region mask.
    const cv::Mat edge_mask (depth_res.y(), depth_res.x(), CV_8UC1, is_edge.data());
    cv::Mat convex_mask;
    cv::bitwise_not(edge_mask, convex_mask);

    //cv::imwrite("/home/srl/edge_mask_og.png", convex_mask);

    // Perform morphological transformations to connect small edges and remove
    // small holes.
    const cv::Mat element = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, cv::Size(morph_diam, morph_diam));
    cv::morphologyEx(convex_mask, convex_mask, cv::MORPH_OPEN, element);
    //cv::morphologyEx(convex_mask, convex_mask, cv::MORPH_CLOSE, element);

    //cv::imwrite("/home/srl/edge_mask.png", convex_mask);

    // Find connected patches in the convex mask.
    cv::Mat geometric_label;
    cv::Mat label_stats;
    cv::Mat label_centroids; // Unused
    const int num_labels = cv::connectedComponentsWithStats(
        convex_mask, geometric_label, label_stats, label_centroids, 4, CV_16U);
    SegmentationResult geometric_segmentation(depth_res);
    //geometric_segmentation.combined_mask_ = geometric_label.clone(); // TODO not needed?

    for (int label_id = 0; label_id < num_labels; ++label_id) {
      if (label_stats.at<int32_t>(label_id, cv::CC_STAT_AREA) > geometric_component_size) {
        cv::Mat mask = (geometric_label == label_id);
        const cv::Mat element = cv::getStructuringElement(
            cv::MORPH_ELLIPSE, cv::Size(morph_diam, morph_diam));
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, element);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, element);
        const InstanceSegmentation geo_seg(label_id, se::class_invalid, mask);
        geometric_segmentation.object_instances.push_back(geo_seg);
      }
    }

    return geometric_segmentation;
  }



  cv::Mat occlusion_mask(const se::Image<float>& object_depth,
                         const se::Image<float>& background_depth,
                         const float             background_voxel_dim) {
    assert(background_depth.width() == object_depth.width());
    assert(background_depth.height() == object_depth.height());

    // Interpret the depth images as Mats.
    cv::Size s (object_depth.width(), object_depth.height());
    cv::Mat object_depth_mat (s, CV_32FC1, (void*) object_depth.data());
    cv::Mat background_depth_mat (s, CV_32FC1, (void*) background_depth.data());

    // Get a mask for the regions where the background depth is closer than the object depth. Add a
    // threshold to take the coarser background voxels into account.
    cv::Mat threshold (s, CV_32FC1, cv::Scalar(2.0f * background_voxel_dim));
    cv::Mat mask = abs(object_depth_mat - background_depth_mat) > threshold;
    return mask;
  }



  cv::Mat occlusion_mask(const se::Image<Eigen::Vector3f>& object_point_cloud_M,
                         const se::Image<Eigen::Vector3f>& background_point_cloud_M,
                         const float                       background_voxel_dim,
                         const Eigen::Matrix4f&            T_CM) {
    // Convert the point clouds to depth images.
    se::Image<float> object_depth (object_point_cloud_M.width(), object_point_cloud_M.height());
    pointCloudToDepthKernel(object_depth, object_point_cloud_M, T_CM);
    se::Image<float> background_depth (background_point_cloud_M.width(), background_point_cloud_M.height());
    pointCloudToDepthKernel(background_depth, background_point_cloud_M, T_CM);
    // Compute the occlusion.
    return occlusion_mask(object_depth, background_depth, background_voxel_dim);
  }

} // namespace se

