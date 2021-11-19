/*
 * Created by binbin on 25/04/18.
 * Modified by Sotiris Papatheodorou
 */

#include "se/segmentation_result.hpp"

#include <cnpy.h>

#include "se/semanticeight_definitions.hpp"

// Types used for reading segmentation data from NumPy arrays.
typedef uint8_t numpy_mask_t;
typedef uint64_t numpy_class_id_t;
typedef float numpy_confidence_t;



bool read_masks(const std::string& filename, std::vector<cv::Mat>& masks)
{
#if SE_VERBOSE >= SE_VERBOSE_DETAILED
    printf("Reading masks from:               %s\n", filename.c_str());
#endif
    // Empty the vector
    masks.resize(0);
    // Load the numpy array
    cnpy::NpyArray masks_npy;
    try {
        masks_npy = cnpy::npy_load(filename);
    }
    catch (std::runtime_error&) {
        return false;
    }
    // Get the number of masks
    const size_t num_masks = masks_npy.shape[0];
    if (num_masks == 0) {
        // Return if no objects were detected
#if SE_VERBOSE >= SE_VERBOSE_DETAILED
        printf("Read %zu masks\n", masks.size());
#endif
        return true;
    }
    else {
        // Test for correct word size and order
        if (masks_npy.word_size != sizeof(numpy_mask_t)) {
            std::cerr << "Error reading masks: Numpy word size must be " << sizeof(numpy_mask_t)
                      << ", not " << masks_npy.word_size << std::endl;
            return false;
        }
        if (masks_npy.fortran_order == true) {
            std::cerr << "Error: Numpy fortran order must be false" << std::endl;
            return false;
        }

        // Get mask size
        const size_t width = masks_npy.shape[2];
        const size_t height = masks_npy.shape[1];
        // Allocate masks
        // NOTE: do not initialize masks to zero here as opencv will try to be
        // clever and will allocate a single matrix with zeroes for all masks.
        // When you later try to overwrite each mask using memcpy they will all
        // be overwritten. There might be a better way to do this though.
        masks = std::vector<cv::Mat>(num_masks);

        // Read masks
        const numpy_mask_t* current_mask_npy = masks_npy.data<numpy_mask_t>();
        for (size_t i = 0; i < num_masks; ++i) {
            // Initialize mask
            masks[i] = cv::Mat(cv::Size(width, height), se::mask_t);
            // Copy mask data
            std::memcpy(masks[i].data,
                        &(current_mask_npy[i * width * height]),
                        width * height * sizeof(numpy_mask_t));
            // Change elements from {0,1} to {0,255}.
            masks[i] *= 255;
        }
#if SE_VERBOSE >= SE_VERBOSE_DETAILED
        printf("Read %zu masks at %zux%zu\n", masks.size(), width, height);
#endif
        return true;
    }
}



bool read_class_ids(const std::string& filename, std::vector<uint8_t>& class_ids)
{
#if SE_VERBOSE >= SE_VERBOSE_DETAILED
    printf("Reading class IDs from:           %s\n", filename.c_str());
#endif
    // Empty the vector
    class_ids.resize(0);
    // Load the numpy array
    cnpy::NpyArray class_ids_npy;
    try {
        class_ids_npy = cnpy::npy_load(filename);
    }
    catch (std::runtime_error&) {
        return false;
    }
    // Get the number of detected objects
    const size_t num_objects = class_ids_npy.shape[0];
    if (num_objects == 0) {
        // Return if no objects were detected
#if SE_VERBOSE >= SE_VERBOSE_DETAILED
        printf("Read %zu class IDs\n", class_ids.size());
#endif
        return true;
    }
    else {
        // Test for correct word size and order
        if (class_ids_npy.word_size != sizeof(numpy_class_id_t)) {
            std::cerr << "Error reading class IDs: Numpy word size must be "
                      << sizeof(numpy_class_id_t) << ", not " << class_ids_npy.word_size
                      << std::endl;
            return false;
        }
        if (class_ids_npy.fortran_order == true) {
            std::cerr << "Error: Numpy fortran order must be false" << std::endl;
            return false;
        }

        // Allocate vector
        class_ids.resize(num_objects);
#if SE_VERBOSE >= SE_VERBOSE_DETAILED
        printf("Read %zu class IDs: ", class_ids.size());
#endif

        // Read class IDs
        for (size_t i = 0; i < num_objects; ++i) {
            // The NumPy arrays store the final class IDs, with the background as a class.
            class_ids[i] = static_cast<uint8_t>(class_ids_npy.data<numpy_class_id_t>()[i]);
#if SE_VERBOSE >= SE_VERBOSE_DETAILED
            printf("%u ", (unsigned int) class_ids[i]);
#endif
        }
#if SE_VERBOSE >= SE_VERBOSE_DETAILED
        printf("\n");
#endif
        return true;
    }
}



bool read_all_confs(const std::string& filename, se::VecDetectionConfidence& all_probs)
{
#if SE_VERBOSE >= SE_VERBOSE_DETAILED
    printf("Reading class probabilities from: %s\n", filename.c_str());
#endif
    // Empty the vector
    all_probs.resize(0);
    // Load the numpy array
    cnpy::NpyArray all_probs_npy;
    try {
        all_probs_npy = cnpy::npy_load(filename);
    }
    catch (std::runtime_error&) {
        return false;
    }
    // Get the number of detected objects
    const size_t num_objects = all_probs_npy.shape[0];
    if (num_objects == 0) {
        // Return if no objects were detected
#if SE_VERBOSE >= SE_VERBOSE_DETAILED
        printf("Read %zu class probabilities\n", all_probs.size());
#endif
        return true;
    }
    else {
        // Test for correct word size and order
        if (all_probs_npy.word_size != sizeof(numpy_confidence_t)) {
            std::cerr << "Error reading all class confidence: Numpy word size must be "
                      << sizeof(numpy_confidence_t) << ", not " << all_probs_npy.word_size
                      << std::endl;
            return false;
        }
        if (all_probs_npy.fortran_order == true) {
            std::cerr << "Error: Numpy fortran order must be false" << std::endl;
            return false;
        }
        // Test for correct number of classes
        if (all_probs_npy.shape[1] != se::class_names.size() - 1) {
            std::cerr << "Error: The number of classes must be " << se::class_names.size() - 1
                      << ", not " << all_probs_npy.shape[1] << std::endl;
            return false;
        }
        // Allocate vector
        all_probs = se::VecDetectionConfidence(num_objects);
#if SE_VERBOSE >= SE_VERBOSE_DETAILED
        printf("Read %zu class probabilities: ", all_probs.size());
#endif
        // Read class probabilities
        numpy_confidence_t* current_probs_npy = all_probs_npy.data<numpy_confidence_t>();
        for (size_t i = 0; i < num_objects; ++i) {
            all_probs[i] = se::DetectionConfidence(current_probs_npy);
            current_probs_npy += (se::class_names.size() - 1);
#if SE_VERBOSE >= SE_VERBOSE_DETAILED
            printf("%5.3f ", all_probs[i].confidence());
#endif
        }
#if SE_VERBOSE >= SE_VERBOSE_DETAILED
        printf("\n");
#endif
        return true;
    }
}



namespace se {
// SegmentationResult ///////////////////////////////////////////////////////
SegmentationResult::SegmentationResult(const int w, const int h) : width(w), height(h)
{
}



SegmentationResult::SegmentationResult(const Eigen::Vector2i& img_size) :
        SegmentationResult::SegmentationResult(img_size.x(), img_size.y())
{
}



bool SegmentationResult::read(const std::string& base_dir, const std::string& base_name)
{
    // TODO: accept width, height as parameters for when there are no detections
    // Reset data
    clear();

#if SE_VERBOSE >= SE_VERBOSE_DETAILED
    printf("Loading segmentation data from:   %s/*/%s.npy\n", base_dir.c_str(), base_name.c_str());
#endif

    bool success;

    // Read masks
    std::string filename_masks = base_dir + "/" + segmentation_masks_dir + "/" + base_name + ".npy";
    std::vector<cv::Mat> masks;
    success = read_masks(filename_masks, masks);
    if (not success) {
        std::cerr << "Could not read segmentation masks from " << filename_masks << std::endl;
        return false;
    }

    //// Read class IDs
    //std::string filename_class_ids = base_dir + "/"
    //    + segmentation_class_ids_dir + "/" + base_name + ".npy";
    //std::vector<uint8_t> class_ids;
    //success = read_class_ids(filename_class_ids, class_ids);
    //if (not success) {
    //  std::cerr << "Could not read class IDs from " << filename_class_ids
    //      << std::endl;
    //  return false;
    //}

    // Read class probabilities
    std::string filename_all_probs =
        base_dir + "/" + segmentation_confidence_all_dir + "/" + base_name + ".npy";
    se::VecDetectionConfidence all_probs;
    success = read_all_confs(filename_all_probs, all_probs);
    if (not success) {
        std::cerr << "Could not read class confidence from " << filename_all_probs << std::endl;
        return false;
    }

    // Add the detected objects to the SegmentationResult
    size_t num_objects = all_probs.size();
    for (size_t i = 0; i < num_objects; ++i) {
        object_instances.push_back(InstanceSegmentation(se::instance_new, masks[i], all_probs[i]));
    }

    return true;
}



bool SegmentationResult::write(const std::string& base_dir) const
{
    bool success;
    size_t i = 0;
    for (const auto& inst : object_instances) {
        success = cv::imwrite(base_dir + "/instance_mask" + std::to_string(i) + ".png",
                              inst.instance_mask);
        if (!success)
            return false;
        i++;
    }
    success = cv::imwrite(base_dir + "/instance_mask.png", instanceMask());
    if (!success)
        return false;
    success = cv::imwrite(base_dir + "/class_mask.png", classMask());
    return success;
}



InstanceSegmentationVecIt SegmentationResult::begin()
{
    return object_instances.begin();
}



InstanceSegmentationVecCIt SegmentationResult::begin() const
{
    return object_instances.cbegin();
}



InstanceSegmentationVecCIt SegmentationResult::cbegin() const
{
    return object_instances.cbegin();
}



InstanceSegmentationVecIt SegmentationResult::end()
{
    return object_instances.end();
}

InstanceSegmentationVecCIt SegmentationResult::end() const
{
    return object_instances.cend();
}

InstanceSegmentationVecCIt SegmentationResult::cend() const
{
    return object_instances.cend();
}



InstanceSegmentationVecIt SegmentationResult::find(const int instance_id)
{
    for (InstanceSegmentationVecIt it = object_instances.begin(); it != object_instances.end();
         ++it) {
        if (it->instance_id == instance_id)
            return it;
    }
    // No match found.
    return object_instances.end();
}



InstanceSegmentationVecCIt SegmentationResult::find(const int instance_id) const
{
    for (InstanceSegmentationVecCIt it = object_instances.cbegin(); it != object_instances.cend();
         ++it) {
        if (it->instance_id == instance_id)
            return it;
    }
    // No match found.
    return object_instances.cend();
}



bool SegmentationResult::empty() const
{
    return object_instances.empty();
}



void SegmentationResult::clear()
{
    object_instances.clear();
}



void SegmentationResult::generateBackgroundInstance()
{
    // Remove the background object if it already exists
    auto it = find(se::instance_bg);
    if (it != object_instances.end()) {
        object_instances.erase(it);
    }

    // Initialize the background mask
    cv::Mat bg_mask = cv::Mat::ones(cv::Size(width, height), se::mask_t) * 255;
    // Subtract the masks of all other objects
    for (auto& object : object_instances) {
        cv::Mat inverse_object_mask;
        cv::bitwise_not(object.instance_mask, inverse_object_mask);
        cv::bitwise_and(bg_mask, inverse_object_mask, bg_mask);
    }

    // Add the background object to the object map
    object_instances.push_back(InstanceSegmentation(se::instance_new, se::class_bg, bg_mask));
}



cv::Mat SegmentationResult::instanceMask() const
{
    cv::Mat mask(cv::Size(width, height), se::instance_mask_t, se::instance_bg);

    if (object_instances.size() > 0) {
        for (const auto& inst : object_instances) {
            // Create a matrix filled with the object instance ID.
            cv::Mat id_mat(cv::Size(width, height), se::instance_mask_t, inst.instance_id);
            // Copy it to the mask using the object instance mask.
            id_mat.copyTo(mask, inst.instance_mask);
        }
    }

    return mask;
}



cv::Mat SegmentationResult::classMask() const
{
    cv::Mat mask(cv::Size(width, height), se::class_mask_t, se::class_bg);

    if (object_instances.size() > 0) {
        for (const auto& inst : object_instances) {
            // Create a matrix filled with the object class ID.
            cv::Mat class_mat(cv::Size(width, height), se::class_mask_t, inst.classId());
            // Copy it to the mask using the object instance mask.
            class_mat.copyTo(mask, inst.instance_mask);
        }
    }

    return mask;
}



void SegmentationResult::resize(const int w, const int h)
{
    if (width != w || height != h) {
        // Resize all individual masks
        for (auto& instance : object_instances) {
            instance.resize(w, h);
        }
        width = w;
        height = h;
    }
}



void SegmentationResult::morphologicalRefinement(const size_t element_diameter)
{
    // Process all individual masks
    for (auto& instance : object_instances) {
        instance.morphologicalRefinement(element_diameter);
    }
}



void SegmentationResult::removeDepthOutliers(const cv::Mat& depth)
{
    for (auto& instance : object_instances) {
        instance.removeDepthOutliers(depth);
    }
}



void SegmentationResult::removeSmall(const float percent_nonzero_threshold)
{
    // Loop over all detected instances in reverse to make multiple removals
    // faster.
    for (int i = object_instances.size() - 1; i >= 0; --i) {
        // Count the nonzero elements.
        const size_t nonzero = cv::countNonZero(object_instances[i].instance_mask);
        // Total number of elements.
        const size_t total_elements =
            object_instances[i].instance_mask.rows * object_instances[i].instance_mask.cols;

        const float nonzero_percent = (float) nonzero / (float) total_elements;
        if (nonzero_percent < percent_nonzero_threshold) {
#if SE_VERBOSE >= SE_VERBOSE_NORMAL
            printf("Removed ");
            (object_instances.begin() + i)->print();
            printf(" with %6.2f%% nonzero mask elements\n", nonzero_percent);
#endif
            object_instances.erase(object_instances.begin() + i);
        }
    }
}



void SegmentationResult::removeClass(const int class_id)
{
    // Loop over all detected instances in reverse to make multiple removals
    // faster.
    for (int i = object_instances.size() - 1; i >= 0; --i) {
        if (object_instances[i].classId() == class_id) {
#if SE_VERBOSE >= SE_VERBOSE_NORMAL
            printf("Removed ");
            (object_instances.begin() + i)->print();
            printf(" with class %d\n", class_id);
#endif
            object_instances.erase(object_instances.begin() + i);
        }
    }
}



void SegmentationResult::removeStuff()
{
    // Loop over all detected instances in reverse to make multiple removals
    // faster.
    for (int i = object_instances.size() - 1; i >= 0; --i) {
        if (is_class_stuff(object_instances[i].classId())) {
#if SE_VERBOSE >= SE_VERBOSE_NORMAL
            printf("Removed ");
            (object_instances.begin() + i)->print();
            printf(" as STUFF\n");
#endif
            object_instances.erase(object_instances.begin() + i);
        }
    }
}



void SegmentationResult::removeInvalid()
{
    // Loop over all detected instances in reverse to make multiple removals
    // faster.
    for (int i = object_instances.size() - 1; i >= 0; --i) {
        if (object_instances[i].instance_id == se::instance_invalid) {
#if SE_VERBOSE >= SE_VERBOSE_NORMAL
            printf("Removed ");
            (object_instances.begin() + i)->print();
            printf(" as invalid instance\n");
#endif
            object_instances.erase(object_instances.begin() + i);
        }
    }
}



void SegmentationResult::removeInvalidDepth(const cv::Mat& valid_depth_mask)
{
    // Loop over all detected instances in reverse to make multiple removals
    // faster.
    for (int i = object_instances.size() - 1; i >= 0; --i) {
        cv::Mat mask_intersection;
        cv::bitwise_and(object_instances[i].instance_mask, valid_depth_mask, mask_intersection);
        if (cv::countNonZero(mask_intersection) == 0) {
#if SE_VERBOSE >= SE_VERBOSE_NORMAL
            printf("Removed ");
            (object_instances.begin() + i)->print();
            printf(" due to invalid depth\n");
#endif
            object_instances.erase(object_instances.begin() + i);
        }
    }
}



void SegmentationResult::removeLowConfidence(const float conf_thres)
{
    // Loop over all detected instances in reverse to make multiple removals
    // faster.
    for (int i = object_instances.size() - 1; i >= 0; --i) {
        if (object_instances[i].confidence() < conf_thres) {
#if SE_VERBOSE >= SE_VERBOSE_NORMAL
            printf("Removed ");
            (object_instances.begin() + i)->print();
            printf(" with confidence %3.0f%% < %3.0f%%\n",
                   100.0f * object_instances[i].confidence(),
                   100.0f * conf_thres);
#endif
            object_instances.erase(object_instances.begin() + i);
        }
    }
}



int SegmentationResult::merge(const InstanceSegmentation& other, const float overlap_thres)
{
    for (auto& instance : object_instances) {
        if (instance.merge(other, overlap_thres)) {
            // Only merge to one object.
            return 1;
        }
    }
    // Could not merge into any object.
    return 0;
}



int SegmentationResult::merge(const SegmentationResult& other, const float overlap_thres)
{
    int num_merged = 0;
    for (auto& other_instance : other) {
        if (merge(other_instance, overlap_thres)) {
            num_merged++;
        }
    }
    return num_merged;
}



void SegmentationResult::print(FILE* f) const
{
    for (const auto& object : object_instances) {
        object.print(f);
        fprintf(f, "\n");
    }
}



std::ostream& operator<<(std::ostream& os, const SegmentationResult& s)
{
    for (const auto& object : s.object_instances) {
        os << object << "\n";
    }
    return os;
}

} // namespace se
