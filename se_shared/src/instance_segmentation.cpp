/*
 * Created by binbin on 25/04/18.
 * Modified by Sotiris Papatheodorou
 */

#include "se/instance_segmentation.hpp"

namespace se {
  constexpr size_t InstanceSegmentation::morph_diam_;
  constexpr float InstanceSegmentation::skip_integration;
  constexpr float InstanceSegmentation::skip_fg_update;
  constexpr uint8_t InstanceSegmentation::instance_mask_threshold;

  InstanceSegmentation::InstanceSegmentation()
    : instance_id(instance_invalid),
      instance_mask(cv::Mat::zeros(0, 0, se::mask_t))
  {
  }



  InstanceSegmentation::InstanceSegmentation(const int      instance_id,
                                             const int      class_id,
                                             const cv::Mat& instance_mask)
    : instance_id(instance_id),
      instance_mask(instance_mask),
      conf(class_id)
  {
    cv::threshold(instance_mask, instance_mask, instance_mask_threshold, UINT8_MAX, cv::THRESH_BINARY);
  }



  InstanceSegmentation::InstanceSegmentation(const int                  instance_id,
                                             const cv::Mat&             instance_mask,
                                             const DetectionConfidence& confidence)
    : instance_id(instance_id),
      instance_mask(instance_mask),
      conf(confidence) {
    cv::threshold(instance_mask, instance_mask, instance_mask_threshold, UINT8_MAX, cv::THRESH_BINARY);
  }



  InstanceSegmentation::InstanceSegmentation(const int                  instance_id,
                                             const cv::Mat&             instance_mask)
    : instance_id(instance_id),
      instance_mask(instance_mask) {
    cv::threshold(instance_mask, instance_mask, instance_mask_threshold, UINT8_MAX, cv::THRESH_BINARY);
  }



  int InstanceSegmentation::classId() const {
    return conf.classId();
  }



  float InstanceSegmentation::confidence() const {
    return conf.confidence();
  }



  cv::Mat InstanceSegmentation::generateIntegrationMask() const {
    cv::Size mask_size = instance_mask.size();

    // Perform morphological dilation on the instance mask
    cv::Mat dilated_mask;
    const cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
        cv::Size(morph_diam_, morph_diam_));
    cv::dilate(instance_mask, dilated_mask, element);

    // Get the background from the dilated mask (inverse mask)
    cv::Mat bg_from_dilated;
    cv::bitwise_not(dilated_mask, bg_from_dilated);

    // Difference between dilated and original mask
    cv::Mat dilation_diff = dilated_mask - instance_mask;

    // Initialize the generalized instance mask with the foreground probability
    // value that characterizes the object. 1 for normal objects and 0 for
    // background objects.
    const se::integration_mask_elem_t base_value = (classId() == se::class_bg ? 0.0f : 1.0f);
    cv::Mat integration_mask = cv::Mat(mask_size, se::integration_mask_t, cv::Scalar(base_value));

    // Set regions that should not be fused to skip_integration. These are the regions not contained
    // in the inverse of the dilated mask.
    cv::Mat dont_fuse = cv::Mat(mask_size, se::integration_mask_t, skip_integration);
    dont_fuse.copyTo(integration_mask, bg_from_dilated);

    // Set regions that should be not be considered as part of the object.
    // These are the regions in the difference between the original and dilated
    // masks.
    const se::integration_mask_elem_t complementaty_value = 1.0f - base_value;
    cv::Mat bg = cv::Mat(mask_size, se::integration_mask_t, cv::Scalar(complementaty_value));
    bg.copyTo(integration_mask, dilation_diff);

    // Set all positive parts of the mask to skip_fg_update if this is an undetected instance.
    if (!detected()) {
      cv::Mat positive_mask = integration_mask > 0.0f;
      integration_mask.setTo(skip_fg_update, positive_mask);
    }

    return integration_mask;
  }



  void InstanceSegmentation::resize(const int width, const int height) {
    cv::Mat tmp_image;
    cv::resize(instance_mask, tmp_image, cv::Size(width, height));
    instance_mask = tmp_image.clone();
  }



  void InstanceSegmentation::morphologicalRefinement(
      const size_t element_diameter) {
    const cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
        cv::Size(element_diameter, element_diameter));
    cv::morphologyEx(instance_mask, instance_mask, cv::MORPH_OPEN, element);
    cv::morphologyEx(instance_mask, instance_mask, cv::MORPH_CLOSE, element);
  }



  int InstanceSegmentation::merge(const InstanceSegmentation& other,
                                  const float                 overlap_thres) {
    // Compute the amount of overlap.
    cv::Mat intersection;
    cv::bitwise_and(instance_mask, other.instance_mask, intersection);
    const float intersection_area = cv::countNonZero(intersection);
    const float other_area = cv::countNonZero(other.instance_mask);
    const float ratio = intersection_area / other_area;
    if (ratio > overlap_thres) {
      // Merge the masks.
      cv::Mat combined_mask;
      cv::bitwise_or(instance_mask, other.instance_mask, combined_mask);
      combined_mask.copyTo(instance_mask);
      return 1;
    } else {
      // Do not merge.
      return 0;
    }
  }



  bool InstanceSegmentation::detected() const {
    return conf.valid();
  }



  std::ostream& operator<<(std::ostream& os, const InstanceSegmentation& o) {
    const std::string object_type
        = is_class_stuff(o.classId()) ? "STUFF " : "THING ";
    std::streamsize p = os.precision();
    const std::ios_base::fmtflags f (os.flags());
    os << std::setw(3) << o.instance_id << "   "
        << o.instance_mask.cols << "x" << o.instance_mask.rows << " "
        << std::setw(5) << std::setprecision(3)
        << cv::countNonZero(o.instance_mask) / static_cast<float>(o.instance_mask.total()) << "   "
        << std::setprecision(p)
        << std::setw(3) << o.classId() << " "
        << std::setw(3) << std::fixed << std::setprecision(0) << 100.0f * o.confidence() << "% "
        << std::setprecision(p)
        << (o.detected() ? "  detected " : "undetected ")
        << object_type << class_id_to_str(o.classId());
    os.flags(f);
    return os;
  }

} // namespace se

