/*
 * Copyright 2019 Nils Funk, Imperial College London
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

template <typename FieldType>
class OFusionModel : public sensor_model<OFusionModel<MultiresOFusion::VoxelType>>
{
public:
  inline static float computeSigma(float depth_sample) {
    if (MultiresOFusion::uncertainty_model == UncertaintyModel::linear) {

      return 3 * se::math::clamp(MultiresOFusion::k_sigma * depth_sample, MultiresOFusion::sigma_min, MultiresOFusion::sigma_max); // Livingroom dataset

    } else if (MultiresOFusion::uncertainty_model == UncertaintyModel::quadratic) {

      return 3 * se::math::clamp(MultiresOFusion::k_sigma * depth_sample * depth_sample, MultiresOFusion::sigma_min, MultiresOFusion::sigma_max); // Cow and lady

    }
  }

  inline static float computeTau(float depth_sample) {
    if (MultiresOFusion::const_surface_thickness == true) {

      return MultiresOFusion::tau_max; // Livingroom dataset

    } else {

      return se::math::clamp(MultiresOFusion::k_tau * depth_sample, MultiresOFusion::tau_min, MultiresOFusion::tau_max);

    }
  }

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
  static int lowVariance(float depth_min,
                         float depth_max,
                         float voxel_min_m,
                         float voxel_max_m,
                         float /* mu */) {

    // Assume worst case scenario -> no multiplication with projScale
    float diff_max = (voxel_max_m - depth_min); // * projScale;
    float diff_min = (voxel_min_m - depth_max); // * projScale;

    float tau_max   = computeTau(depth_max);
    float sigma_min = computeSigma(depth_max);


    if (diff_min > tau_max) { // behind of surface
      return  1;
    } else if (diff_max < -sigma_min) {
      return -1;
    } else {
      return  0;
    }
  }

  inline static void updateBlock(float          pos_z,
                                 float          depth_sample,
                                 float          /* mu */,
                                 float          /* voxel_dim */,
                                 MultiresOFusion::VoxelType::VoxelData& field,
                                 const unsigned frame,
                                 const int      scale,
                                 const float    proj_scale) {
    const float diff = (pos_z - depth_sample) * proj_scale;

    float tau   = computeTau(depth_sample);
    float sigma = computeSigma(depth_sample);

    float sample;
    if (diff < -sigma) {
      sample = MultiresOFusion::log_odd_min;
    } else if (diff < tau / 2) {
      sample = std::min(MultiresOFusion::log_odd_min - MultiresOFusion::log_odd_min / sigma * (diff + sigma), MultiresOFusion::log_odd_max);
    } else if (diff < tau) {
      sample = std::min(-MultiresOFusion::log_odd_min * tau / (2 * sigma), MultiresOFusion::log_odd_max);
    } else {
      return;
    }
    field_operations::addValueToBlock(field, sample, frame, scale);
  }

  inline static void freeBlock(MultiresOFusion::VoxelType::VoxelData& field,
                               const unsigned frame,
                               const int      scale) {
    field_operations::addValueToBlock(field, MultiresOFusion::log_odd_min, frame, scale);
  }

  inline static void freeNode(MultiresOFusion::VoxelType::VoxelData& field,
                              const unsigned frame) {
    field_operations::addValueToNode(field, MultiresOFusion::log_odd_min, frame);
  }

  inline static bool isOccupied(const MultiresOFusion::VoxelType::VoxelData& field,
                                const float hysteresis) {
    return field_operations::value(field) > hysteresis;
  }

  inline static bool isFree(const MultiresOFusion::VoxelType::VoxelData& value,
                            const float hysteresis) {
    return field_operations::value(value) < -hysteresis;
  }
};
