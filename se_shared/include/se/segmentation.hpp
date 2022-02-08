/*
 * Created by binbin on 25/04/18.
 * Modified by Sotiris Papatheodorou
 */

#ifndef SEGMENTATION_HPP
#define SEGMENTATION_HPP

#include <cstdint>
#include <opencv2/opencv.hpp>
#include <set>
#include <string>
#include <vector>

#include "se/semantic_classes.hpp"

namespace se {
extern se::SemanticClasses semantic_classes;

// If you change any of the mask types below, ensure you also change the
// respective element type.
// cv::Mat and element types for binary masks.
constexpr int mask_t = CV_8UC1;
typedef uint8_t mask_elem_t;
// cv::Mat and element types for class ID masks.
constexpr int class_mask_t = CV_8SC1;
typedef int8_t class_mask_elem_t;
// cv::Mat and element types for instance ID masks.
constexpr int instance_mask_t = CV_32SC1;
typedef int32_t instance_mask_elem_t;
// cv::Mat and element types for integration (foreground probability) masks.
constexpr int integration_mask_t = CV_32FC1;
typedef float integration_mask_elem_t;

// Some constants used throughout the code.
constexpr instance_mask_elem_t instance_bg = -1;
constexpr instance_mask_elem_t instance_new = -2;
constexpr instance_mask_elem_t instance_invalid = -3;

/**
   * The directory that contains the ground truth segmentation. It must be a
   * subdirectory of the input directory.
   */
const std::string segmentation_base_dir = "segmentation";
const std::string segmentation_boxes_dir = "boxes";
const std::string segmentation_class_ids_dir = "class_ids";
const std::string segmentation_confidence_dir = "confidence";
const std::string segmentation_confidence_all_dir = "confidence_all";
const std::string segmentation_masks_dir = "masks";



/** Compute the Intersection over Union (IoU) of two masks.
   */
float IoU(const cv::Mat& mask1, const cv::Mat& mask2);

/** Something that looks kinda like IoU. Threshold for this was 0.5.
   */
float notIoU(const cv::Mat& mask1, const cv::Mat& mask2);

/**
   * Given a mask of type se::instance_mask_t containing several object instances,
   * extract a mask of a specific instance. Returns a mask of type se::mask_t.
   */
cv::Mat extract_instance(const cv::Mat& instance_mask, const se::instance_mask_elem_t instance);

} // namespace se

#endif
