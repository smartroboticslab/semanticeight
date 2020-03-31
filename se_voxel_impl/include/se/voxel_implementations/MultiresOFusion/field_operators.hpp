/**
 *
 * SLAMcore Confidential
 * ---------------------
 *
 * SLAMcore Limited
 * All Rights Reserved.
 * (C) Copyright 2019
 *
 * NOTICE:
 *
 * All information contained herein is, and remains the property of SLAMcore
 * Limited and its suppliers, if any. The intellectual and technical concepts
 * contained herein are proprietary to SLAMcore Limited and its suppliers and
 * may be covered by patents in process, and are protected by trade secret or
 * copyright law. Dissemination of this information or reproduction of this
 * material is strictly forbidden unless prior written permission is obtained
 * from SLAMcore Limited.
 *
 */

#ifndef __FIELD_OPERATORS_HPP
#define __FIELD_OPERATORS_HPP

/**
 * @file
 * @brief Define useful operators on field data types
 */

/// @todo This file will be set to standards once new traits are introduced in SE.

/// @todo this is not the place to declare this. Also, the name is not best, something like MeanWeightField(MWF) would
/// be more generic and independent of the particular sensor model we are using.
using MultiresOFusionData = MultiresOFusion::VoxelType::VoxelData;

namespace field_operations
{
  static constexpr float factor = (MultiresOFusion::max_weight - 1) / MultiresOFusion::max_weight;

  inline bool hasInformation(const MultiresOFusionData& field) { return field.observed; }

  inline float value(const MultiresOFusionData& field) { return field.x; }

  inline void addValue2Block(MultiresOFusionData& field, float value, const unsigned frame, const int scale)
  {
    field.x        = (field.x * field.y + value) / (field.y + 1);
    field.y        = std::min(field.y + 1, MultiresOFusion::max_weight);
//    field.x_max    = (scale == 0) ? field.x * field.y : field.x_max + value;
//    field.x_max    = std::max(std::min(field.x_max, MultiresOFusion::max_occupancy), MultiresOFusion::min_occupancy);
    field.x_max    = std::max(std::min(field.x * field.y, MultiresOFusion::max_occupancy), MultiresOFusion::min_occupancy);
    field.frame    = frame;
    field.observed = true;
  }

  inline void addValue2Node(MultiresOFusionData& field, float value, const unsigned frame) {
    if (field.y == MultiresOFusion::max_weight) {
      field.x_max    = std::max(std::min(factor * field.x_max + value, MultiresOFusion::max_occupancy), MultiresOFusion::min_occupancy);
    } else {
      field.x_max    = std::max(std::min(field.x_max + value, MultiresOFusion::max_occupancy), MultiresOFusion::min_occupancy);
    }
    field.y        = std::min(field.y + 1, MultiresOFusion::max_weight);
    field.observed = true;
  }
}

#endif // __FIELD_OPERATORS_HPP
