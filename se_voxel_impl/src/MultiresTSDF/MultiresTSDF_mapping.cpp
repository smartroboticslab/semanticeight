/*
 *
 * Copyright 2019 Emanuele Vespa, Imperial College London
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
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

#include "se/voxel_implementations/MultiresTSDF/MultiresTSDF.hpp"

#include "se/node.hpp"
#include "se/octree.hpp"
#include "se/image/image.hpp"
#include "se/filter.hpp"
#include "se/functors/for_each.hpp"



namespace se {
  namespace multires {

/**
 * Update the subgrids of a voxel block starting from a given scale up
 * to a maximum scale.
 *
 * \param[in] block VoxelBlock to be updated
 * \param[in] scale scale from which propagate up voxel values
*/
    void propagateUp(se::VoxelBlock<MultiresTSDF::VoxelType>* block, const int scale) {
      const Eigen::Vector3i block_coord = block->coordinates();
      const int block_size = se::VoxelBlock<MultiresTSDF::VoxelType>::size;
      for (int voxel_scale = scale; voxel_scale < se::math::log2_const(block_size); ++voxel_scale) {
        const int stride = 1 << (voxel_scale + 1);
        for (int z = 0; z < block_size; z += stride)
          for (int y = 0; y < block_size; y += stride)
            for (int x = 0; x < block_size; x += stride) {
              const Eigen::Vector3i voxel_coord = block_coord + Eigen::Vector3i(x, y, z);

              float mean = 0;
              int sample_count = 0;
              float weight = 0;
              for (int k = 0; k < stride; k += stride / 2)
                for (int j = 0; j < stride; j += stride / 2)
                  for (int i = 0; i < stride; i += stride / 2) {
                    auto child_data = block->data(voxel_coord + Eigen::Vector3i(i, j, k), voxel_scale);
                    if (child_data.y != 0) {
                      mean += child_data.x;
                      weight += child_data.y;
                      sample_count++;
                    }
                  }
              auto voxel_data = block->data(voxel_coord, voxel_scale + 1);

              if (sample_count != 0) {
                mean /= sample_count;
                weight /= sample_count;
                voxel_data.x = mean;
                voxel_data.x_last = mean;
                voxel_data.y = ceil(weight);
              } else {
                voxel_data = MultiresTSDF::VoxelType::initData();
              }
              voxel_data.delta_y = 0;
              block->data(voxel_coord, voxel_scale + 1, voxel_data);
            }
      }
    }

    void propagateUp(se::Node<MultiresTSDF::VoxelType>* node,
                     const int                          voxel_depth,
                     const unsigned                     timestamp) {

      if (!node->parent()) {
        node->timestamp(timestamp);
        return;
      }

      float mean = 0;
      int sample_count = 0;
      float weight = 0;
      for (int i = 0; i < 8; ++i) {
        const auto &child_data = node->data_[i];
        if (child_data.y != 0) {
          mean += child_data.x;
          weight += child_data.y;
          sample_count++;
        }
      }

      const unsigned int child_idx = se::child_idx(node->code_,
                                           se::keyops::code(node->code_), voxel_depth);
      if (sample_count > 0) {
        auto &node_data = node->parent()->data_[child_idx];
        mean /= sample_count;
        weight /= sample_count;
        node_data.x = mean;
        node_data.x_last = mean;
        node_data.y = ceil(weight);
        node_data.delta_y = 0;
      }
      node->timestamp(timestamp);
    }


    template<typename FieldSelector>
    float interp(const Eigen::Vector3i&                         voxel_coord,
                 const int                                      scale,
                 const se::Octree<MultiresTSDF::VoxelType>&     map,
                 const se::VoxelBlock<MultiresTSDF::VoxelType>* block,
                 FieldSelector                                  select_value,
                 bool&                                          is_valid) {

      auto select_weight = [](const auto &data) { return data.y; };

      // The return types of the select_value() and select_weight() functions. Since they
      // can be lambda functions, an argument needs to be passed to them before
      // deducing the return type.
      typedef decltype(select_value(MultiresTSDF::VoxelType::initData())) select_value_t;
      typedef decltype(select_weight(MultiresTSDF::VoxelType::initData())) select_weight_t;

      // Compute base point in parent block
      const int block_size = se::VoxelBlock<MultiresTSDF::VoxelType>::size >> (scale + 1);
      const int stride = 1 << (scale + 1);

      const Eigen::Vector3f& centre_offset = se::Octree<MultiresTSDF::VoxelType>::offset_;
      Eigen::Vector3i base =
          stride * (voxel_coord.cast<float>() / stride - centre_offset).cast<int>().cwiseMax(Eigen::Vector3i::Zero());
      base = (base.array() == block_size - 1).select(base - Eigen::Vector3i::Constant(1), base);

      select_value_t values[8];
      internal::gather_values(
          map, block->coordinates() + base, scale + 1, select_value, values);

      select_weight_t weights[8];
      internal::gather_values(
          map, block->coordinates() + base, scale + 1, select_weight, weights);
      for (int i = 0; i < 8; ++i) {
        if (weights[i] == 0) {
          is_valid = false;
          return select_value(MultiresTSDF::VoxelType::initData());
        }
      }
      is_valid = true;

      const Eigen::Vector3f voxel_coord_f = voxel_coord.cast<float>() + centre_offset * (stride / 2);
      const Eigen::Vector3f base_f = base.cast<float>() + centre_offset * (stride);
      const Eigen::Vector3f factor = (voxel_coord_f - base_f) / stride;

      const float v_000 = values[0] * (1 - factor.x()) + values[1] * factor.x();
      const float v_001 = values[2] * (1 - factor.x()) + values[3] * factor.x();
      const float v_010 = values[4] * (1 - factor.x()) + values[5] * factor.x();
      const float v_011 = values[6] * (1 - factor.x()) + values[7] * factor.x();

      const float v_0 = v_000 * (1 - factor.y()) + v_001 * (factor.y());
      const float v_1 = v_010 * (1 - factor.y()) + v_011 * (factor.y());

      const float val = v_0 * (1 - factor.z()) + v_1 * factor.z();
      return val;
    }


/**
 * Update the subgrids of a voxel block starting from a given scale
 * down to the finest grid.
 *
 * \param[in] block VoxelBlock to be updated
 * \param[in] scale scale from which propagate down voxel values
*/
    void propagateDown(const se::Octree<MultiresTSDF::VoxelType>& map,
                       se::VoxelBlock<MultiresTSDF::VoxelType>*  block,
                       const int                                 scale,
                       const int                                 min_scale,
                       const int                                 max_weight = INT_MAX) {
      const Eigen::Vector3i block_coord = block->coordinates();
      const int block_size = se::VoxelBlock<MultiresTSDF::VoxelType>::size;
      for (int voxel_scale = scale; voxel_scale > min_scale; --voxel_scale) {
        const int stride = 1 << voxel_scale;
        for (int z = 0; z < block_size; z += stride)
          for (int y = 0; y < block_size; y += stride)
            for (int x = 0; x < block_size; x += stride) {
              const Eigen::Vector3i parent_coord = block_coord + Eigen::Vector3i(x, y, z);
              auto parent_data = block->data(parent_coord, voxel_scale);
              float delta_x = parent_data.x - parent_data.x_last;
              const int half_stride = stride / 2;
              for (int k = 0; k < stride; k += half_stride) {
                for (int j = 0; j < stride; j += half_stride) {
                  for (int i = 0; i < stride; i += half_stride) {
                    const Eigen::Vector3i voxel_coord = parent_coord + Eigen::Vector3i(i, j, k);
                    auto voxel_data = block->data(voxel_coord, voxel_scale - 1);
                    if (voxel_data.y == 0) {
                      bool is_valid;
                      voxel_data.x = se::math::clamp(
                          interp(voxel_coord - block_coord, voxel_scale - 1, map, block,
                                 [](const auto &data) { return data.x; }, is_valid), -1.f, 1.f);
                      voxel_data.y = is_valid ? parent_data.y : 0;
                      voxel_data.x_last = voxel_data.x;
                      voxel_data.delta_y = 0;
                    } else {
                      voxel_data.x = std::max(voxel_data.x + delta_x, -1.f);
                      voxel_data.y = fminf(voxel_data.y + parent_data.delta_y, max_weight);
                      voxel_data.delta_y = parent_data.delta_y;
                    }
                    block->data(voxel_coord, voxel_scale - 1, voxel_data);
                  }
                }
              }
              parent_data.x_last = parent_data.x;
              parent_data.delta_y = 0;
              block->data(parent_coord, voxel_scale, parent_data);
            }
      }
    }

/**
 * Update a voxel block at a given scale by first propagating down the parent
 * values and then integrating the new measurement;
*/
    void propagateUpdate(se::VoxelBlock<MultiresTSDF::VoxelType>*   block,
                         const int                                  scale,
                         const se::Octree<MultiresTSDF::VoxelType>& map,
                         const se::Image<float>&                    depth_image,
                         const Eigen::Matrix4f&                     T_CM,
                         const SensorImpl&                          sensor,
                         const float                                map_res,
                         const int                                  max_weight,
                         const Eigen::Vector3f&                     centre_offset,
                         const float                                mu) {

      const int block_size = se::VoxelBlock<MultiresTSDF::VoxelType>::size;
      const int parent_scale = scale + 1;
      const int stride = 1 << parent_scale;
      const int half_stride = stride >> 1;
      bool is_visible = false;

      const Eigen::Vector3i block_coord = block->coordinates();

      for (unsigned int z = 0; z < block_size; z += stride) {
        for (unsigned int y = 0; y < block_size; y += stride) {
          for (unsigned int x = 0; x < block_size; x += stride) {
            const Eigen::Vector3i parent_coord = block_coord + Eigen::Vector3i(x, y, z);
            auto parent_data = block->data(parent_coord, parent_scale);
            float delta_x = parent_data.x - parent_data.x_last;
            for (int k = 0; k < stride; k += half_stride) {
              for (int j = 0; j < stride; j += half_stride) {
                for (int i = 0; i < stride; i += half_stride) {
                  const Eigen::Vector3i voxel_coord = parent_coord + Eigen::Vector3i(i, j, k);
                  auto voxel_data = block->data(voxel_coord, scale);
                  if (voxel_data.y == 0) {
                    bool is_valid;
                    voxel_data.x = se::math::clamp(interp(voxel_coord - block_coord, scale, map, block,
                        [](const auto &val) { return val.x; }, is_valid), -1.f, 1.f);
                    voxel_data.y = is_valid ? parent_data.y : 0;
                    voxel_data.x_last = voxel_data.x;
                    voxel_data.delta_y = 0;
                  } else {
                    voxel_data.x = se::math::clamp(voxel_data.x + delta_x, -1.f, 1.f);
                    voxel_data.y = fminf(voxel_data.y + parent_data.delta_y, max_weight);
                    voxel_data.delta_y = parent_data.delta_y;
                  }

                  const Eigen::Vector3f point_C = (T_CM * (map_res * (voxel_coord.cast<float>() +
                      centre_offset)).homogeneous()).head(3);
                  
                  Eigen::Vector2f pixel_f;
                  if (sensor.model.project(point_C, &pixel_f) != srl::projection::ProjectionStatus::Successful) {
                    block->data(voxel_coord, scale, voxel_data);
                    continue;
                  }
                  const Eigen::Vector2i pixel = (pixel_f + Eigen::Vector2f::Constant(0.5f)).cast<int>();
                
                  is_visible = true;

                  const float depth_value = depth_image[pixel.x() + depth_image.width() * pixel.y()];
                  // continue on invalid depth measurement
                  if (depth_value <= 0) {
                    block->data(voxel_coord, scale, voxel_data);
                    continue;
                  }

                  // Update the TSDF
                  const float tsdf_value = (depth_value - point_C.z())
                                     * std::sqrt(1 + se::math::sq(point_C.x() / point_C.z()) +
                                                 se::math::sq(point_C.y() / point_C.z()));
                  if (tsdf_value > -mu) {
                    const float tsdf_value = fminf(1.f, tsdf_value / mu);
                    voxel_data.x = se::math::clamp(
                        (static_cast<float>(voxel_data.y) * voxel_data.x + tsdf_value) /
                        (static_cast<float>(voxel_data.y) + 1.f),
                        -1.f, 1.f);
                    voxel_data.y = fminf(voxel_data.y + 1, max_weight);
                    voxel_data.delta_y++;
                  }
                  block->data(voxel_coord, scale, voxel_data);
                }
              }
            }
            parent_data.x_last = parent_data.x;
            parent_data.delta_y = 0;
            block->data(parent_coord, parent_scale, parent_data);
          }
        }
      }
      block->current_scale(scale);
      block->active(is_visible);
    }

    struct multires_block_update {
      multires_block_update(
          const se::Octree<MultiresTSDF::VoxelType>& map,
          const se::Image<float>&                    depth_image,
          const Eigen::Matrix4f&                     T_CM,
          const SensorImpl                           sensor,
          const float                                voxel_dim,
          const int                                  max_weight) :
          map(map),
          depth_image(depth_image),
          T_CM(T_CM),
          sensor(sensor),
          voxel_dim(voxel_dim),
          max_weight(max_weight),
          sample_offset_fraction(map.offset_),
          mu(sensor.mu) {}

      const se::Octree<MultiresTSDF::VoxelType>& map;
      const se::Image<float>& depth_image;
      const Eigen::Matrix4f& T_CM;
      const SensorImpl sensor;
      const float voxel_dim;
      const int max_weight;
      const Eigen::Vector3f& sample_offset_fraction;
      const float mu;

      void operator()(se::VoxelBlock<MultiresTSDF::VoxelType>* block) {

        constexpr int block_size = se::VoxelBlock<MultiresTSDF::VoxelType>::size;
        const Eigen::Vector3i block_coord = block->coordinates();
        const Eigen::Vector3f block_sample_offset = (sample_offset_fraction.array().colwise() *
            Eigen::Vector3f::Constant(block_size).array());
        const float block_diff = (T_CM * (voxel_dim * (block_coord.cast<float>() +
            block_sample_offset)).homogeneous()).head(3).z();
        const int last_scale = block->current_scale();
        const int scale = std::max(std::min(sensor.computeIntegrationScale(block_diff, voxel_dim),
        se::math::log2_const(block_size >> 1)), last_scale - 1);
        block->min_scale(block->min_scale() < 0 ? scale : std::min(block->min_scale(), scale));
        if (last_scale > scale) {
          propagateUpdate(block, scale, map, depth_image, T_CM, sensor,
              voxel_dim, max_weight, sample_offset_fraction, mu);
          return;
        }
        bool is_visible = false;
        block->current_scale(scale);
        const int stride = 1 << scale;

        const Eigen::Vector3f voxel_sample_offset = sample_offset_fraction;
        for (unsigned int z = 0; z < block_size; z += stride) {
          for (unsigned int y = 0; y < block_size; y += stride) {
#pragma omp simd
            for (unsigned int x = 0; x < block_size; x += stride) {
              const Eigen::Vector3i voxel_coord = block_coord + Eigen::Vector3i(x, y, z);
              const Eigen::Vector3f point_C = (T_CM * (voxel_dim *
                  (voxel_coord.cast<float>() + voxel_sample_offset)).homogeneous()).head(3);
              
              Eigen::Vector2f pixel_f;
              if (sensor.model.project(point_C, &pixel_f) != srl::projection::ProjectionStatus::Successful) {
                continue;
              }
              const Eigen::Vector2i pixel = (pixel_f + Eigen::Vector2f::Constant(0.5f)).cast<int>();

              is_visible = true;
              // const Eigen::Vector2i pixel = (pixel_f + Eigen::Vector2f::Constant(0.5f)).cast<int>();
              const float depth_value = depth_image[pixel.x() + depth_image.width() * pixel.y()];
              // continue on invalid depth measurement
              if (depth_value <= 0) continue;

              // Update the TSDF
              const float point_dist = (depth_value - point_C.z())
                                 * std::sqrt(1 + se::math::sq(point_C.x() / point_C.z()) +
                                             se::math::sq(point_C.y() / point_C.z()));
              if (point_dist > -mu) {
                const float tsdf_value = fminf(1.f, point_dist / mu);
                auto voxel_data = block->data(voxel_coord, scale);
                voxel_data.x = se::math::clamp(
                    (static_cast<float>(voxel_data.y) * voxel_data.x + tsdf_value) /
                    (static_cast<float>(voxel_data.y) + 1.f),
                    -1.f, 1.f);
                voxel_data.y = fminf(voxel_data.y + 1, max_weight);
                voxel_data.delta_y++;
                block->data(voxel_coord, scale, voxel_data);
              }
            }
          }
        }
        propagateUp(block, scale);
        block->active(is_visible);
      }
    };
} // namespace multires
} // namespace se

void MultiresTSDF::integrate(se::Octree<MultiresTSDF::VoxelType>& map,
                             const se::Image<float>&              depth_image,
                             const Eigen::Matrix4f&               T_CM,
                             const SensorImpl&                    sensor,
                             const unsigned                       frame) {

  using namespace std::placeholders;

  /* Retrieve the active list */
  std::vector<se::VoxelBlock<MultiresTSDF::VoxelType> *> active_list;
  auto& block_buffer = map.pool().blockBuffer();

  /* Predicates definition */
  const float voxel_dim = map.dim() / map.size();
  auto in_frustum_predicate =
  std::bind(se::algorithms::in_frustum<se::VoxelBlock<MultiresTSDF::VoxelType>>,
      std::placeholders::_1, voxel_dim, T_CM, sensor);
  auto is_active_predicate = [](const se::VoxelBlock<MultiresTSDF::VoxelType> *block) {
    return block->active();
  };
  se::algorithms::filter(active_list, block_buffer, is_active_predicate,
                         in_frustum_predicate);

  std::deque<se::Node<MultiresTSDF::VoxelType> *> node_queue;
  std::mutex deque_mutex;
  struct se::multires::multires_block_update block_update_funct(
      map, depth_image, T_CM, sensor, voxel_dim, MultiresTSDF::max_weight);
  se::functor::internal::parallel_for_each(active_list, block_update_funct);

  for (const auto &block : active_list) {
    if (block->parent()) node_queue.push_back(block->parent());
  }

  while (!node_queue.empty()) {
    se::Node<MultiresTSDF::VoxelType>* node = node_queue.front();
    node_queue.pop_front();
    if (node->timestamp() == frame) continue;
    se::multires::propagateUp(node, map.voxelDepth(), frame);
    if (node->parent()) node_queue.push_back(node->parent());
  }
}
