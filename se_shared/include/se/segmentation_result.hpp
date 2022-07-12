/*
 * Created by binbin on 25/04/18.
 * Modified by Sotiris Papatheodorou
 */

#ifndef SEGMENTATION_RESULT_HPP
#define SEGMENTATION_RESULT_HPP

#include "se/instance_segmentation.hpp"

namespace se {
/**
   * Segmentation data for a single RGB frame. This struct contains multiple
   * InstanceSegmentation structs.
   */
struct SegmentationResult {
    int width;
    int height;

    /**
     * A vector containing all the InstanceSegmentation for a single frame.
     */
    InstanceSegmentationVec object_instances;



    SegmentationResult(const int width, const int height);

    SegmentationResult(const Eigen::Vector2i& img_size);

    /**
     * Read the segmentation for a single frame from base_dir.
     *
     * \param[in] base_dir The base directory of the segmentation results
     * containing the folders confidence_all, class_ids and masks.
     * \param[in] base_name The base name of the files to read. This should be
     * the current frame timestamp for TUM datasets.
     *
     * \note The width and height must have been set properly in the
     * SegmentationResult constructor as there is no way to get the frame
     * resolution if no objects have been detected.
     */
    bool read(const std::string& base_dir, const std::string& base_name);

    /**
     * Write the segmentation for a single frame in numpy format.
     *
     * \param[in] base_dir  The base directory of the segmentation results. The subdirectories
     *                      confidence_all and masks will be created if they don't exist.
     * \param[in] base_name The base name of the files to write. This should be the current frame
     *                      timestamp for TUM datasets.
     */
    bool write(const std::string& base_dir, const std::string& base_name) const;

    /**
     * Write the segmentation masks and the combined mask for a single frame as
     * PNG images inside base_dir.
     *
     * \param[in] base_dir The base directory where the images will be created.
     * \return True for success, false for error.
     */
    bool write(const std::string& base_dir) const;

    /**
     * Return object_instances_.begin().
     */
    InstanceSegmentationVecIt begin();
    InstanceSegmentationVecCIt begin() const;
    InstanceSegmentationVecCIt cbegin() const;

    /**
     * Return object_instances_.end().
     */
    InstanceSegmentationVecIt end();
    InstanceSegmentationVecCIt end() const;
    InstanceSegmentationVecCIt cend() const;

    /**
     * Search the object_instances_ for the given instance ID. Returns an
     * iterator to the first match or an iterator to object_instances_.end() if
     * no match was found.
     */
    InstanceSegmentationVecIt find(const int instance_id);
    InstanceSegmentationVecCIt find(const int instance_id) const;

    bool empty() const;

    /**
     * Clear all stored segmentation data.
     */
    void clear();

    /**
     * Add another object instance containing the background. Its mask is the
     * inverse of the union of all object masks.
     */
    void generateBackgroundInstance();

    /**
     * Create a mask containing the object instance ID for each pixel. Its type
     * is se::instance_mask_t. The values of the mask are as follows:
     * - se::instance_invalid invalid object detection
     * - se::instance_new     object without an assigned instance ID
     * - se::instance_bg      background
     * - 1+               object with the number corresponding to the object
     *                    instance ID
     */
    cv::Mat instanceMask() const;

    /**
     * Create a mask containing the object class for each pixel. Its type is
     * se::class_mask_t. The values of the mask are as follows:
     * - se::semantic_classes.invalidId()      invalid object detection
     * - se::semantic_classes.backgroundId()   background
     * - 1+                                    object with the number corresponding to the object
     *                                         class
     */
    cv::Mat classMask() const;

    /**
     * Change the resolution of all masks to the one supplied.
     */
    void resize(const int width, const int height);

    /**
     * Use morphological opening and closing to remove small patches and small
     * holes from each InstanceSegmentation::instance_mask.
     */
    void morphologicalRefinement(const size_t element_diameter);

    void removeDepthOutliers(const cv::Mat& depth);

    /**
     * Remove all object instances whose masks' percentage (in the interval [0,
     * 1]) of nonzero elements is lower than percent_nonzero_threshold.
     */
    void removeSmall(const float percent_nonzero_threshold);

    /**
     * Remove all object instances whose class is class_id.
     */
    void removeClass(const int class_id);

    /**
     * Remove all object instances whose class is in SEGMENTATION_CLASSES_STUFF.
     */
    void removeStuff();

    /**
     * Remove all object instances whose instance ID is se::instance_invalid.
     */
    void removeInvalid();

    /**
     * Filter the object masks to remove pixels where the depth measurements are invalid.
     */
    void filterInvalidDepth(const cv::Mat& valid_depth_mask);

    /**
     * Remove all object instances whose detected class confidence is below
     * conf_thres.
     */
    void removeLowConfidence(const float conf_thres);

    int merge(const InstanceSegmentation& other, const float overlap_thres);

    int merge(const SegmentationResult& other, const float overlap_thres);

    void print(FILE* f = stdout) const;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};



std::ostream& operator<<(std::ostream& os, const SegmentationResult& s);

} // namespace se

#endif // SEGMENTATION_RESULT_HPP
