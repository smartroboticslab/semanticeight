/*
 * Copyright 2016 Emanuele Vespa, Imperial College London
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * */

#ifndef ACTIVE_LIST_HPP
#define ACTIVE_LIST_HPP

#include "se/utils/math_utils.h"
#include "se/node.hpp"
#include "se/utils/memory_pool.hpp"
#include "se/utils/morton_utils.hpp"
#include "se/sensor_implementation.hpp"

namespace se {
namespace algorithms {

  template <typename VoxelBlockType>
    static inline bool in_frustum(const VoxelBlockType*  block,
                                  const float            voxel_size,
                                  const Eigen::Matrix4f& T_CW,
                                  const SensorImpl&      sensor,
                                  const Eigen::Vector2i& image_size) {

      const int side = VoxelBlockType::side;
      const static Eigen::Matrix<int, 4, 8> offsets =
        (Eigen::Matrix<int, 4, 8>() << 0, side, 0   , side, 0   , side, 0   , side,
                                       0, 0   , side, side, 0   , 0   , side, side,
                                       0, 0   , 0   , 0   , side, side, side, side,
                                       0, 0   , 0   , 0   , 0   , 0   , 0   , 0   ).finished();
      const Eigen::Matrix3Xf block_corners_C = (T_CW * Eigen::Vector4f(voxel_size, voxel_size, voxel_size, 1.f).asDiagonal() *
          (offsets.colwise() + block->coordinates().homogeneous()).template cast<float>()).topRows(3);
      Eigen::Matrix2Xf projected_corners(2, 8);
      std::vector<srl::projection::ProjectionStatus> projection_stati;
      sensor.model.projectBatch(block_corners_C, &projected_corners, &projection_stati);
      return std::all_of(projection_stati.begin(), projection_stati.end(),
          [](const auto it){ return it == srl::projection::ProjectionStatus::Successful; });
    }

  template <typename ValueType, typename P>
    bool satisfies(const ValueType& el, P predicate) {
      return predicate(el);
    }

  template <typename ValueType, typename P, typename... Ps>
    bool satisfies(const ValueType& el, P predicate, Ps... others) {
      return predicate(el) || satisfies(el, others...);
    }

  template <typename BufferType, typename... Predicates>
  void filter(std::vector<BufferType *>&               out,
              const se::PagedMemoryBuffer<BufferType>& buffer,
              Predicates...                            ps) {
#ifdef _OPENMP
#pragma omp declare reduction (merge : std::vector<BufferType *> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for reduction(merge: out)
    for (unsigned int i = 0; i < buffer.size(); ++i) {
      if (satisfies(buffer[i], ps...)) {
        out.push_back(buffer[i]);
      }
    }
#else
    for (unsigned int i = 0; i < buffer.size(); ++i) {
      if (satisfies(buffer[i], ps...)) {
        out.push_back(buffer[i]);
      }
    }
#endif
  }
}
}
#endif
