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

#ifndef __SENSOR_MODEL_HPP
#define __SENSOR_MODEL_HPP

/**
 * @file
 * @brief sensor_model for abstracting diferent sensor inverse models from the alocation and evaluation logic.
 */

#include <se/voxel_implementations/MultiresOFusion/field_operators.hpp>
#include "../../../../../se_denseslam/include/se/constant_parameters.h"

/**
 * A generic sensor model class implementing a generic interface with an inverse sensor model.
 * @tparam FieldType Data storage type.
 */
template <typename Derived>
class sensor_model
{
public:
  /**
   * @brief Return a conservative measure of the expected variance of a sensor model inside a voxel
   *  given its position and depth variance.
   * @param[in] depth_min Depth measurement max value inside voxel.
   * @param[in] depth_max Depth measurement min value inside voxel.
   * @param[in] voxel_min Voxel depth min value.
   * @param[in] voxel_max Voxel depth max value.
   * @param[in] sigma Model uncertainty value for sensor, equivalent to TSDF narrow band
   * @param[in] projScale Scaling due to ray position.
   * @param[in] hasUnknownData Is there unknown data on the voxel area.
   * @return Estimate of variance
   */
  inline static int lowVariance(float depth_min, float depth_max, float voxel_min_m, float voxel_max_m,
                                                         float mu)
  {
    return Derived::lowVariance(depth_min, depth_max, voxel_min_m, voxel_max_m, mu);
  }

  inline static float computeSigma()
  {
    return Derived::computeSigma();
  }

  /**
   * @brief Update a field with a new measurement, a weighting of 1 is considered for the new measurement.
   * @param diffMeasToVox Difference from the depth measurement to the evaluation position (depth_meas - voxel_pos).
   * @param sigma Uncertainty or truncation band assigned to the model.
   * @param field Field to update.
   */
  inline static void updateBlock(float pos_z, float depth_sample, float sigma, float voxel_size, MultiresOFusion::VoxelType::VoxelData& field, const unsigned frame, const int scale, const float proj_scale = 1)
  {
    Derived::updateBlock(pos_z, depth_sample, sigma, voxel_size, field, frame, scale, proj_scale);
  }

  inline static void freeBlock(MultiresOFusion::VoxelType::VoxelData& field, const unsigned frame, const int scale)
  {
    Derived::freeBlock(field, frame, scale);
  }

  inline static void freeNode(MultiresOFusion::VoxelType::VoxelData& field, const unsigned frame)
  {
    Derived::freeNode(field, frame);
  }



  /**
   * @brief Check if a field should be marked as occupied.
   * @param value Field to evaluate.
   * @param hysteresis An hysteresis band over the 0 crossing.
   * @return True if considered occupied.
   */
  inline static bool isOccupied(const MultiresOFusion::VoxelType::VoxelData& value, const float hysteresis)
  {
    return Derived::is_occupied(value, hysteresis);
  }

  /**
   * @brief Check if a field should be marked as free.
   * @param value Field to evaluate.
   * @param hysteresis An hysteresis band over the 0 crossing.
   * @return True if considered free.
   */
  inline static bool isFree(const MultiresOFusion::VoxelType::VoxelData& value, const float hysteresis)
  {
    return Derived::is_free(value, hysteresis);
  }
};

#include "se/voxel_implementations/MultiresOFusion/sensor_model_impl.hpp"

#endif // __SENSOR_MODEL_HPP