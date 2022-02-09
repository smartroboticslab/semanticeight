/*
 * Created by binbin on 25/04/18.
 * Modified by Sotiris Papatheodorou
 */

#ifndef INSTANCE_SEGMENTATION_HPP
#define INSTANCE_SEGMENTATION_HPP

#include <Eigen/StdVector>

#include "se/detection_confidence.hpp"

namespace se {
/** Segmentation data for a single object instance. Objects that were not detected by the
   * segmentation network will have an invalid conf.
   */
struct InstanceSegmentation {
    /** The ID of this particular object instance. Used to distinguish objects of the same class.
     */
    int instance_id;

    /** The value of the mask is 255 for pixels corresponding to the object and 0 elsewhere. Its
     * type is se::mask_t.
     *
     * \note 255 is used instead of 1 so that when saved, the masks are black and white. This helps
     * significantly in debugging.
     */
    cv::Mat instance_mask;

    /** An object containing the confidence of the object detector for each object class.
     */
    DetectionConfidence conf;

    /** True if this instance was detected by the segmentation network and false if it was generated
     * from an existing object through raycasting.
     */
    bool detected;



    /** Create an invalid InstanceSegmentation.
     */
    InstanceSegmentation();

    /** If class_id is a valid semantic class ID, the corresponding element of conf will be set to
     * 1 and all other elements to 0.
     */
    InstanceSegmentation(const int instance_id,
                         const int class_id,
                         const cv::Mat& instance_mask,
                         const bool detected = true);

    InstanceSegmentation(const int instance_id,
                         const DetectionConfidence& confidence,
                         const cv::Mat& instance_mask,
                         const bool detected = true);

    /** Retern the ID of the detected object class.
     */
    int classId() const;

    /** Return the confidence the object detector assigned to this detection.
     */
    float confidence() const;

    /** Create an integration mask that can be used for integrating the masked object to the volume.
     * Its type is se::integration_mask_t. The mask values have the following meaning:
     * - [0,1]                                      Foreground probability
     * - se::InstanceSegmentation::skip_integration Do not integrate a measurement
     * - se::InstanceSegmentation::skip_fg_update   Fuse but don't update the foreground probability
     *
     * \param[out] integration_mask The resulting mask.
     *
     * \note Currently the possible mask element values are {-1, 0, 1}.
     */
    cv::Mat generateIntegrationMask() const;

    cv::Mat generateIntegrationMask(const cv::Mat& raycasted_object_mask) const;

    /** Changes the resolution of the mask to the one supplied.
     */
    void resize(const int width, const int height);

    /** Use morphological opening and closing to remove small patches and small holes from the
     * InstanceSegmentation::instance_mask.
     */
    void morphologicalRefinement(const size_t element_diameter = morph_diam);

    void removeDepthOutliers(const cv::Mat& depth);

    int merge(const InstanceSegmentation& other, const float overlap_thres);

    void print(FILE* f = stdout) const;

    /** Diameter of the circle used for morphological transformations in
     * InstanceSegmentation::generateIntegrationMask() and
     * InstanceSegmentation::morphologicalRefinement().
     */
    static constexpr size_t morph_diam = 5;

    static constexpr float skip_integration = -1.0f;

    static constexpr float skip_fg_update = -2.0f;

    /** Threshold used to convert instance mask elements from the interval [0,UINT8_MAX] to the set
     * {0,UINT8_MAX}.
     */
    static constexpr uint8_t instance_mask_threshold = UINT8_MAX / 2;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

std::ostream& operator<<(std::ostream& os, const InstanceSegmentation& o);



/** A vector containing InstanceSegmentation structs.
   */
typedef std::vector<InstanceSegmentation, Eigen::aligned_allocator<InstanceSegmentation>>
    InstanceSegmentationVec;
typedef std::vector<InstanceSegmentation, Eigen::aligned_allocator<InstanceSegmentation>>::iterator
    InstanceSegmentationVecIt;
typedef std::vector<InstanceSegmentation,
                    Eigen::aligned_allocator<InstanceSegmentation>>::const_iterator
    InstanceSegmentationVecCIt;

} // namespace se

#endif // INSTANCE_SEGMENTATION_HPP
