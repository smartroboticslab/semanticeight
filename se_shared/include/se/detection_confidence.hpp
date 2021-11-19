// SPDX-FileCopyrightText: 2021 Smart Robotics Lab
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DETECTION_CONFIDENCE_HPP
#define DETECTION_CONFIDENCE_HPP

#include <Eigen/Dense>

#include "se/segmentation.hpp"

namespace se {
class DetectionConfidence {
    public:
    /** Initialize for an invalid detection.
       */
    DetectionConfidence();

    /** Initialize so that the confidence of class_id is 1 and that of all other classes is 0.
       */
    DetectionConfidence(int class_id, float confidence = 1.0f);

    /** Initialize the confidence of all clases from an array of length
       * se::class_names.size() - 1. The confidence of the background is set to 0. Used to
       * initialize from NumPy arrays.
       */
    DetectionConfidence(float* confidence_array);

    float operator[](std::size_t idx) const;

    /** Return the ID of the semantic class with the greatest confidence.
       */
    int classId() const;

    /** Return the greatest confidence.
       */
    float confidence() const;

    /** Combine the current confidence vector with that of other using a mean.
       */
    void merge(const DetectionConfidence& other);

    bool valid() const;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    private:
    Eigen::VectorXf conf_;

    int class_id_;

    float num_updates_;

    void updateClassId();
};



typedef std::vector<DetectionConfidence, Eigen::aligned_allocator<DetectionConfidence>>
    VecDetectionConfidence;



inline float DetectionConfidence::operator[](std::size_t idx) const
{
    return conf_(idx);
}

inline int DetectionConfidence::classId() const
{
    return class_id_;
}

inline float DetectionConfidence::confidence() const
{
    if (class_id_ >= 0) {
        return conf_(class_id_);
    }
    else {
        return 0.0f;
    }
}
} // namespace se

#endif // DETECTION_CONFIDENCE_HPP
