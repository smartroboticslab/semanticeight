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

#ifndef FIELD_OPERATORS_HPP
#define FIELD_OPERATORS_HPP

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

#endif // FIELD_OPERATORS_HPP
