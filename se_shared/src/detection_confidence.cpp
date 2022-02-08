// SPDX-FileCopyrightText: 2021 Smart Robotics Lab
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#include "se/detection_confidence.hpp"

namespace se {
DetectionConfidence::DetectionConfidence() :
        conf_(se::semantic_classes.size()),
        class_id_(se::semantic_classes.invalidId()),
        num_updates_(0)
{
    conf_ = Eigen::VectorXf::Zero(se::semantic_classes.size());
}

DetectionConfidence::DetectionConfidence(int class_id, float confidence) :
        conf_(se::semantic_classes.size()), class_id_(class_id), num_updates_(1)
{
    conf_ = Eigen::VectorXf::Zero(se::semantic_classes.size());
    conf_(class_id_) = confidence;
}

DetectionConfidence::DetectionConfidence(float* confidence_array) :
        conf_(se::semantic_classes.size()),
        class_id_(se::semantic_classes.invalidId()),
        num_updates_(1)
{
    // TODO SEM the commented out code is the correct version but the dataset is wrong. The dataset
    // is storing the confidence values one element after the correct one, thus the last class won't
    // have a confidence value.
    //conf_(0) = 0.0f;
    //for (size_t i = 1; i < se::semantic_classes.size(); i++) {
    //  conf_(i) = confidence_array[i - 1];
    //}
    conf_(se::semantic_classes.size() - 1) = 0.0f;
    for (size_t i = 0; i < se::semantic_classes.size() - 1; i++) {
        conf_(i) = confidence_array[i];
    }
    updateClassId();
}

void DetectionConfidence::merge(const DetectionConfidence& other)
{
    if (num_updates_ > 0 || other.num_updates_ > 0) {
        // Average the confidence vectors.
        conf_ = (num_updates_ * conf_ + other.num_updates_ * other.conf_)
            / (num_updates_ + other.num_updates_);
        num_updates_ += other.num_updates_;
        updateClassId();
    }
}

bool DetectionConfidence::valid() const
{
    return (class_id_ != se::semantic_classes.invalidId());
}

void DetectionConfidence::updateClassId()
{
    if (conf_.isApproxToConstant(0.0f)) {
        class_id_ = se::semantic_classes.invalidId();
    }
    else {
        conf_.maxCoeff(&class_id_);
    }
}
} // namespace se
