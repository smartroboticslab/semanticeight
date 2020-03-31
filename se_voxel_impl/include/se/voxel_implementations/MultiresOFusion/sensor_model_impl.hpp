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

/**
 * @file
 * @brief sensor_model for abstracting diferent sensor inverse models from the alocation and evaluation logic.
 */

/**
 * Class implementing a "Half Extended" SDF model:
 *                                          -1
 * |                   /|
 * |                  / |invalid____________ 0
 * |                 /|
 * |----------------  |                      1
 * |                 depth meassurement
 * |-----------------------------> distance
 * @tparam FieldType Data storage type.
 */

template <typename FieldType>
class OFusionModel : public sensor_model<OFusionModel<MultiresOFusion::VoxelType>>
{
public:
//#define tau 0.32
#define log_odd_min_ -5.015
#define log_odd_max_ 5.015

#define tau_max_ 0.16;
//#define tau_max_ 0.2 // Cases: 10cm res
//#define k_tau_ 0.052f // Cases: 1cm res + x0.25 image sampling, 2cm res + x0.25 image sampling
#define k_tau_ 0.026f // Cases: 1cm res + x0.5 image sampling, 2cm res + x0.5 image sampling

  /**
   * @brief Return a conservative meassure of the expected variance of a sensor model inside a voxel
   *  given its position and depth variance.
   * @param[in] depthMin Depth measurement max value inside voxel.
   * @param[in] depthMax Depth measurement min value inside voxel.
   * @param[in] voxelMin Voxel depth min value.
   * @param[in] voxelMax Voxel depth max value.
   * @param[in] sigma Model uncertainty value for sensor, equivalent to TSDF narrow band
   * @param[in] projScale Scaling due to ray position.
   * @param[in] hasUnknownData Is there unknown data on the voxel area.
   * @return Estimate of variance
   */
  static int lowVariance(float depth_min, float depth_max, float voxel_min_m, float voxel_max_m, float mu)
  {

    // Assume worst case scenario -> no multiplication with projScale
    float diff_max = (voxel_max_m - depth_min); // * projScale;
    float diff_min = (voxel_min_m - depth_max); // * projScale;


    // TODO: if 10cm
//    const float tau_max   =     tau_max_; // Case: 10cm res
    // TODO: if <10cm
    const float tau_max   =     se::math::clamp(k_tau_ * depth_max, 0.06f, 0.16f);

    // TODO: CHANGE FOR DATASET
//    const float sigma_min = se::math::clamp(k_tau_ * depth_min, 0.045f, 0.18f); // Livingroom dataset
    const float sigma_min = 3 * std::max(0.0016f * depth_max * depth_max, 0.02f); // Cow and lady

    if (diff_min > tau_max) { // behind of surface
      return  1;
    } else if (diff_max < -sigma_min) {
      return -1;
    } else {
      return  0;
    }
  }

  inline static float computeSigma() {
    // Return tau in this case
    return tau_max_;
  }

  inline static void updateBlock(float pos_z, float depth_sample, float mu, float voxel_size, MultiresOFusion::VoxelType::VoxelData& field, const unsigned frame, const int scale, const float proj_scale)
  {
    const float diff = (pos_z - depth_sample) * proj_scale;

    // TODO: if 10cm
//    const float tau   =     tau_max_; // Case: 10cm res
    // TODO: if <10cm
    const float tau   =     se::math::clamp(k_tau_ * depth_sample, 0.06f, 0.16f);

    // TODO: CHANGE FOR DATASET
//    const float sigma = se::math::clamp(k_tau_ * depth_sample, 0.045f, 0.18f); // Livingroom dataset
    const float sigma = 3 * std::max(0.0016f * depth_sample * depth_sample, 0.02f); // Cow and lady

    float sample;
    if (diff < -sigma) {
      sample = log_odd_min_;
    } else if (diff < tau / 2) {
      sample = std::min(log_odd_min_ - log_odd_min_ / sigma * (diff + sigma), log_odd_max_);
    } else if (diff < tau) {
      sample = std::min(-log_odd_min_ * tau / (2 * sigma), log_odd_max_);
    } else {
      return;
    }
    field_operations::addValue2Block(field, sample, frame, scale);
  }

  inline static void freeBlock(MultiresOFusion::VoxelType::VoxelData& field, const unsigned frame, const int scale)
  {
    field_operations::addValue2Block(field, log_odd_min_, frame, scale);
  }

  inline static void freeNode(MultiresOFusion::VoxelType::VoxelData& field, const unsigned frame)
  {
    field_operations::addValue2Node(field, log_odd_min_, frame);
  }

  inline static bool isOccupied(const MultiresOFusion::VoxelType::VoxelData& field, const float hysteresis)
  {
    return field_operations::value(field) > hysteresis;
  }

  inline static bool isFree(const MultiresOFusion::VoxelType::VoxelData& value, const float hysteresis)
  {
    return field_operations::value(value) < -hysteresis;
  }
};