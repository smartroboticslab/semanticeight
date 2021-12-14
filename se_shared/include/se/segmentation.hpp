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

namespace se {
/** An std::vector used to map semantic class IDs to semantic class names. */
extern std::vector<std::string> class_names;
/** An std::set of the semantic class names that are considered stuff. */
extern std::set<std::string> stuff_class_names;
/** An std::vector used to map semantic class IDs to voxel resolutions in metres. */
extern std::vector<float> class_res;
/** The map resolution used for most objects. */
constexpr float default_res = 0.05f;

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
constexpr class_mask_elem_t class_bg = 0;
constexpr class_mask_elem_t class_invalid = -2;
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



void use_coco_classes();

void use_matterport3d_classes();

void set_thing(const std::string& class_name);

void set_stuff(const std::string& class_name);

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

/**
   * Convert a class ID to a string containing the class name. For example
   * class_id_to_str(CLASS_BACKGROUND) returns "BACKGROUND".
   */
std::string class_id_to_str(const int class_id);

/** Test whether the supplied class ID is a "stuff" object.
   */
bool is_class_stuff(const int class_id);

/** Print all recognized class IDs and names.
   */
void print_classes();

} // namespace se

#endif
