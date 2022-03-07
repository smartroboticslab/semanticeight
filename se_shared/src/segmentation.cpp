/*
 * Created by binbin on 25/04/18.
 * Modified by Sotiris Papatheodorou
 */

#include "se/segmentation.hpp"

namespace se {
// Use COCO classes by default.
se::SemanticClasses semantic_classes = se::SemanticClasses::coco_classes();



float IoU(const cv::Mat& mask1, const cv::Mat& mask2)
{
    assert((mask1.rows == mask2.rows) && "Masks have different number of rows\n");
    assert((mask1.cols == mask2.cols) && "Masks have different number of columns\n");
    assert((mask1.type() == mask2.type()) && "Masks have different type\n");

    cv::Mat mask_intersection;
    cv::bitwise_and(mask1, mask2, mask_intersection);
    cv::Mat mask_union;
    cv::bitwise_or(mask1, mask2, mask_union);

    const float intersection_area = cv::countNonZero(mask_intersection);
    const float union_area = cv::countNonZero(mask_union);
    const float iou = intersection_area / union_area;
    return iou;
}



float notIoU(const cv::Mat& mask1, const cv::Mat& mask2)
{
    assert((mask1.rows == mask2.rows) && "Masks have different number of rows\n");
    assert((mask1.cols == mask2.cols) && "Masks have different number of columns\n");
    assert((mask1.type() == mask2.type()) && "Masks have different type\n");

    cv::Mat mask_intersection(mask1.size(), mask1.type());
    cv::bitwise_and(mask1, mask2, mask_intersection);

    const float intersection_area = cv::countNonZero(mask_intersection);
    const float mask1_area = cv::countNonZero(mask1);
    const float mask2_area = cv::countNonZero(mask2);

    const float mask1_ratio = intersection_area / mask1_area;
    const float mask2_ratio = intersection_area / mask2_area;

    const float ratio = fmaxf(mask1_ratio, mask2_ratio);

    return ratio;
}



cv::Mat extract_instance(const cv::Mat& instance_mask, const se::instance_mask_elem_t instance_id)
{
    cv::Mat individual_mask(instance_mask.rows, instance_mask.cols, se::mask_t);
    // Set all elements matching the instance ID to 255 and all others to 0.
    std::transform(instance_mask.begin<se::instance_mask_elem_t>(),
                   instance_mask.end<se::instance_mask_elem_t>(),
                   individual_mask.begin<se::mask_elem_t>(),
                   [instance_id](const auto& value) { return (value == instance_id) ? 255 : 0; });
    return individual_mask;
}

} // namespace se
