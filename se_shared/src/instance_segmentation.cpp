/*
 * Created by binbin on 25/04/18.
 * Modified by Sotiris Papatheodorou
 */

#include "se/instance_segmentation.hpp"

namespace se {
constexpr size_t InstanceSegmentation::morph_diam;
constexpr float InstanceSegmentation::skip_integration;
constexpr float InstanceSegmentation::skip_fg_update;
constexpr uint8_t InstanceSegmentation::instance_mask_threshold;



InstanceSegmentation::InstanceSegmentation() :
        instance_id(instance_invalid),
        instance_mask(cv::Mat::zeros(0, 0, se::mask_t)),
        detected(false)
{
}



InstanceSegmentation::InstanceSegmentation(const int instance_id,
                                           const int class_id,
                                           const cv::Mat& instance_mask,
                                           const bool detected) :
        InstanceSegmentation(instance_id, DetectionConfidence(class_id), instance_mask, detected)
{
}



InstanceSegmentation::InstanceSegmentation(const int instance_id,
                                           const DetectionConfidence& confidence,
                                           const cv::Mat& instance_mask,
                                           const bool detected) :
        instance_id(instance_id), instance_mask(instance_mask), conf(confidence), detected(detected)
{
    cv::threshold(
        instance_mask, instance_mask, instance_mask_threshold, UINT8_MAX, cv::THRESH_BINARY);
}



int InstanceSegmentation::classId() const
{
    return conf.classId();
}



float InstanceSegmentation::confidence() const
{
    return conf.confidence();
}



cv::Mat InstanceSegmentation::generateIntegrationMask() const
{
    // Initialize the integration mask to "don't integrate".
    cv::Mat integration_mask =
        cv::Mat(instance_mask.size(), se::integration_mask_t, cv::Scalar(skip_integration));
    const se::integration_mask_elem_t base_value =
        (classId() == se::semantic_classes.backgroundId() ? 0.0f : 1.0f);
    // Update parts of the integration mask covered by the instance mask.
    integration_mask.setTo(cv::Scalar(base_value), instance_mask);
    // Set all positive parts of the mask to skip_fg_update if this is an undetected instance.
    if (!detected) {
        cv::Mat positive_mask = integration_mask > 0.0f;
        integration_mask.setTo(skip_fg_update, positive_mask);
    }
    return integration_mask;
}



cv::Mat InstanceSegmentation::generateIntegrationMask(const cv::Mat& raycasted_object_mask) const
{
    // Get the normal integration mask.
    cv::Mat integration_mask = generateIntegrationMask();

    const se::integration_mask_elem_t base_value =
        (classId() == se::semantic_classes.backgroundId() ? 0.0f : 1.0f);
    const se::integration_mask_elem_t complementaty_value = 1.0f - base_value;

    // Decrease the probability in undetected parts of the object.
    cv::Mat inverse_instance_mask;
    cv::bitwise_not(instance_mask, inverse_instance_mask);
    cv::Mat undetected_mask;
    cv::bitwise_and(raycasted_object_mask, inverse_instance_mask, undetected_mask);
    integration_mask.setTo(cv::Scalar(complementaty_value), undetected_mask);

    return integration_mask;
}



void InstanceSegmentation::resize(const int width, const int height)
{
    cv::Mat tmp_image;
    cv::resize(instance_mask, tmp_image, cv::Size(width, height), cv::INTER_NEAREST);
    instance_mask = tmp_image.clone();
}



void InstanceSegmentation::morphologicalRefinement(const size_t element_diameter)
{
    const cv::Mat element =
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(element_diameter, element_diameter));
    cv::morphologyEx(instance_mask, instance_mask, cv::MORPH_OPEN, element);
    cv::morphologyEx(instance_mask, instance_mask, cv::MORPH_CLOSE, element);
}



void InstanceSegmentation::removeDepthOutliers(const cv::Mat& depth)
{
    cv::Scalar mu_cv, stddev_cv;
    cv::meanStdDev(depth, mu_cv, stddev_cv, instance_mask);
    const float stddev = stddev_cv[0];
    // Arbitrary distance threshold in meters.
    if (stddev > 0.5f) {
        cv::Mat inliers = (depth - mu_cv) <= 3 * stddev_cv;
        cv::bitwise_and(instance_mask, inliers, instance_mask);
    }
}



int InstanceSegmentation::merge(const InstanceSegmentation& other, const float overlap_thres)
{
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
    }
    else {
        // Do not merge.
        return 0;
    }
}



void InstanceSegmentation::print(FILE* f) const
{
    const std::string object_type = semantic_classes.enabled(classId()) ? "THING" : "STUFF";
    fprintf(f,
            "%3d   %dx%d %5.3f   %3d %3.0f%% %s %s %s",
            instance_id,
            instance_mask.cols,
            instance_mask.rows,
            cv::countNonZero(instance_mask) / static_cast<float>(instance_mask.total()),
            classId(),
            100.0f * confidence(),
            (detected ? "  detected" : "undetected"),
            object_type.c_str(),
            semantic_classes.name(classId()).c_str());
}



std::ostream& operator<<(std::ostream& os, const InstanceSegmentation& o)
{
    const std::string object_type = semantic_classes.enabled(o.classId()) ? "THING " : "STUFF ";
    std::streamsize p = os.precision();
    const std::ios_base::fmtflags f(os.flags());
    os << std::setw(3) << o.instance_id << "   " << o.instance_mask.cols << "x"
       << o.instance_mask.rows << " " << std::setw(5) << std::setprecision(3)
       << cv::countNonZero(o.instance_mask) / static_cast<float>(o.instance_mask.total()) << "   "
       << std::setprecision(p) << std::setw(3) << o.classId() << " " << std::setw(3) << std::fixed
       << std::setprecision(0) << 100.0f * o.confidence() << "% " << std::setprecision(p)
       << (o.detected ? "  detected " : "undetected ") << object_type
       << semantic_classes.name(o.classId());
    os.flags(f);
    return os;
}

} // namespace se
