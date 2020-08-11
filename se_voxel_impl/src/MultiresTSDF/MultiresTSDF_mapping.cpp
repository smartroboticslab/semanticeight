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
#include "se/image_utils.hpp"
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
    void propagate_up(MultiresTSDF::VoxelBlockType* block,
                      const int                     scale) {
      const Eigen::Vector3i block_coord = block->coordinates();
      const int block_size = MultiresTSDF::VoxelBlockType::size_li;
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
              block->setData(voxel_coord, voxel_scale + 1, voxel_data);
            }
      }
    }

    void propagate_up(se::Node<MultiresTSDF::VoxelType>* node,
                      const int                          voxel_depth,
                      const unsigned                     timestamp) {

      if (!node->parent()) {
        node->timestamp(timestamp);
        return;
      }

      float mean = 0;
      int sample_count = 0;
      float weight = 0;
      for (int child_idx = 0; child_idx < 8; ++child_idx) {
        const auto &child_data = node->childData(child_idx);
        if (child_data.y != 0) {
          mean += child_data.x;
          weight += child_data.y;
          sample_count++;
        }
      }

      const unsigned int child_idx = se::child_idx(node->code(),
          se::keyops::code(node->code()), voxel_depth);
      if (sample_count > 0) {
        auto &node_data = node->parent()->childData(child_idx);
        mean /= sample_count;
        weight /= sample_count;
        node_data.x = mean;
        node_data.x_last = mean;
        node_data.y = ceil(weight);
        node_data.delta_y = 0;
      }
      node->timestamp(timestamp);
    }

    /**
     * Update the subgrids of a voxel block starting from a given scale
     * down to the finest grid.
     *
     * \param[in] block VoxelBlock to be updated
     * \param[in] scale scale from which propagate down voxel values
     */
    void propagate_down(const se::Octree<MultiresTSDF::VoxelType>& map,
                        MultiresTSDF::VoxelBlockType*              block,
                        const int                                  scale,
                        const int                                  min_scale) {
      const Eigen::Vector3i block_coord = block->coordinates();
      const int block_size = MultiresTSDF::VoxelBlockType::size_li;
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
                      const Eigen::Vector3f voxel_sample_coord_f =
                          get_sample_coord(voxel_coord, stride, map.sample_offset_frac_);
                      voxel_data.x = se::math::clamp(map.interp(voxel_sample_coord_f,
                          [](const auto &data) { return data.x; }, voxel_scale - 1, is_valid).first, -1.f, 1.f);
                      voxel_data.y = is_valid ? parent_data.y : 0;
                      voxel_data.x_last = voxel_data.x;
                      voxel_data.delta_y = 0;
                    } else {
                      voxel_data.x = std::max(voxel_data.x + delta_x, -1.f);
                      voxel_data.y = fminf(voxel_data.y + parent_data.delta_y, MultiresTSDF::max_weight);
                      voxel_data.delta_y = parent_data.delta_y;
                    }
                    block->setData(voxel_coord, voxel_scale - 1, voxel_data);
                  }
                }
              }
              parent_data.x_last = parent_data.x;
              parent_data.delta_y = 0;
              block->setData(parent_coord, voxel_scale, parent_data);
            }
      }
    }

    /**
     * Update a voxel block at a given scale by first propagating down the parent
     * values and then integrating the new measurement;
     */
    void propagate_update(MultiresTSDF::VoxelBlockType*              block,
                          const int                                  voxel_scale,
                          const se::Octree<MultiresTSDF::VoxelType>& map,
                          const se::Image<float>&                    depth_image,
                          const Eigen::Matrix4f&                     T_CM,
                          const SensorImpl&                          sensor,
                          const float                                voxel_dim,
                          const Eigen::Vector3f&                     sample_offset_frac) {

      const int block_size = MultiresTSDF::VoxelBlockType::size_li;
      const int parent_scale = voxel_scale + 1;
      const int parent_stride = 1 << parent_scale;
      const int voxel_stride = parent_stride >> 1;
      bool is_visible = false;

      const Eigen::Vector3i block_coord = block->coordinates();

      for (unsigned int z = 0; z < block_size; z += parent_stride) {
        for (unsigned int y = 0; y < block_size; y += parent_stride) {
          for (unsigned int x = 0; x < block_size; x += parent_stride) {
            const Eigen::Vector3i parent_coord = block_coord + Eigen::Vector3i(x, y, z);
            auto parent_data = block->data(parent_coord, parent_scale);
            float delta_x = parent_data.x - parent_data.x_last;
            for (int k = 0; k < parent_stride; k += voxel_stride) {
              for (int j = 0; j < parent_stride; j += voxel_stride) {
                for (int i = 0; i < parent_stride; i += voxel_stride) {
                  const Eigen::Vector3i voxel_coord = parent_coord + Eigen::Vector3i(i, j, k);
                  auto voxel_data = block->data(voxel_coord, voxel_scale);
                  const Eigen::Vector3f voxel_sample_coord_f =
                      get_sample_coord(voxel_coord, voxel_stride, sample_offset_frac);
                  if (voxel_data.y == 0) {
                    bool is_valid;
                    voxel_data.x = se::math::clamp(map.interp(voxel_sample_coord_f,
                        [](const auto &data) { return data.x; }, voxel_scale + 1, is_valid).first, -1.f, 1.f);
                    voxel_data.y = is_valid ? parent_data.y : 0;
                    voxel_data.x_last = voxel_data.x;
                    voxel_data.delta_y = 0;
                  } else {
                    voxel_data.x = se::math::clamp(voxel_data.x + delta_x, -1.f, 1.f);
                    voxel_data.y = fminf(voxel_data.y + parent_data.delta_y, MultiresTSDF::max_weight);
                    voxel_data.delta_y = parent_data.delta_y;
                  }

                  const Eigen::Vector3f point_C = (T_CM * (voxel_dim * voxel_sample_coord_f).homogeneous()).head(3);

                  // Don't update the point if the sample point is behind the far plane
                  if (point_C.norm() > sensor.farDist(point_C)) {
                    continue;
                  }

                  float depth_value(0);
                  if (!sensor.projectToPixelValue(point_C, depth_image, depth_value,
                                                  [](float depth_value){ return depth_value > 0; })) {
                    continue;
                  }

                  is_visible = true;

                  // Update the TSDF
                  const float sdf_value = (depth_value - point_C.z())
                                     * std::sqrt(1 + se::math::sq(point_C.x() / point_C.z()) +
                                                 se::math::sq(point_C.y() / point_C.z()));
                  if (sdf_value > -MultiresTSDF::mu) {
                    const float tsdf_value = fminf(1.f, sdf_value / MultiresTSDF::mu);
                    voxel_data.x = se::math::clamp(
                        (static_cast<float>(voxel_data.y) * voxel_data.x + tsdf_value) /
                        (static_cast<float>(voxel_data.y) + 1.f),
                        -1.f, 1.f);
                    voxel_data.y = fminf(voxel_data.y + 1, MultiresTSDF::max_weight);
                    voxel_data.delta_y++;
                  }
                  block->setData(voxel_coord, voxel_scale, voxel_data);
                }
              }
            }
            parent_data.x_last = parent_data.x;
            parent_data.delta_y = 0;
            block->setData(parent_coord, parent_scale, parent_data);
          }
        }
      }
      block->current_scale(voxel_scale);
      block->active(is_visible);
    }

    struct MultiresTSDFUpdate {
      MultiresTSDFUpdate(
          const se::Octree<MultiresTSDF::VoxelType>& map,
          const se::Image<float>&                    depth_image,
          const Eigen::Matrix4f&                     T_CM,
          const SensorImpl                           sensor,
          const float                                voxel_dim) :
          map(map),
          depth_image(depth_image),
          T_CM(T_CM),
          sensor(sensor),
          voxel_dim(voxel_dim),
          sample_offset_frac(map.sample_offset_frac_) {}

      const se::Octree<MultiresTSDF::VoxelType>& map;
      const se::Image<float>& depth_image;
      const Eigen::Matrix4f& T_CM;
      const SensorImpl sensor;
      const float voxel_dim;
      const Eigen::Vector3f& sample_offset_frac;

      void operator()(MultiresTSDF::VoxelBlockType* block) {

        constexpr int block_size = MultiresTSDF::VoxelBlockType::size_li;
        const Eigen::Vector3i block_coord = block->coordinates();
        const Eigen::Vector3f block_centre_coord_f =
            se::get_sample_coord(block_coord, block_size, Eigen::Vector3f::Constant(0.5f));
        const float block_centre_point_C_z = (T_CM * (voxel_dim * block_centre_coord_f).homogeneous()).head(3).z();
        const int last_scale = block->current_scale();

        const int scale = std::max(sensor.computeIntegrationScale(
            block_centre_point_C_z, voxel_dim, last_scale, block->min_scale(), map.maxBlockScale()), last_scale - 1);
        block->min_scale(block->min_scale() < 0 ? scale : std::min(block->min_scale(), scale));
        if (last_scale > scale) {
          propagate_update(block, scale, map, depth_image, T_CM, sensor,
              voxel_dim, sample_offset_frac);
          return;
        }
        bool is_visible = false;
        block->current_scale(scale);
        const int stride = 1 << scale;
        for (unsigned int z = 0; z < block_size; z += stride) {
          for (unsigned int y = 0; y < block_size; y += stride) {
#pragma omp simd
            for (unsigned int x = 0; x < block_size; x += stride) {
              const Eigen::Vector3i voxel_coord = block_coord + Eigen::Vector3i(x, y, z);
              const Eigen::Vector3f voxel_sample_coord_f =
                  get_sample_coord(voxel_coord, stride, sample_offset_frac);
              const Eigen::Vector3f point_C = (T_CM * (voxel_dim * voxel_sample_coord_f).homogeneous()).head(3);

              // Don't update the point if the sample point is behind the far plane
              if (point_C.norm() > sensor.farDist(point_C)) {
                continue;
              }
              float depth_value(0);
              if (!sensor.projectToPixelValue(point_C, depth_image, depth_value,
                  [&](float depth_value){ return depth_value >= sensor.near_plane; })) {
                continue;
              }

              is_visible = true;

              // Update the TSDF
              const float point_dist = (depth_value - point_C.z())
                                 * std::sqrt(1 + se::math::sq(point_C.x() / point_C.z()) +
                                             se::math::sq(point_C.y() / point_C.z()));
              if (point_dist > -MultiresTSDF::mu) {
                const float tsdf_value = fminf(1.f, point_dist / MultiresTSDF::mu);
                auto voxel_data = block->data(voxel_coord, scale);
                voxel_data.x = se::math::clamp(
                    (static_cast<float>(voxel_data.y) * voxel_data.x + tsdf_value) /
                    (static_cast<float>(voxel_data.y) + 1.f),
                    -1.f, 1.f);
                voxel_data.y = fminf(voxel_data.y + 1, MultiresTSDF::max_weight);
                voxel_data.delta_y++;
                block->setData(voxel_coord, scale, voxel_data);
              }
            }
          }
        }
        propagate_up(block, scale);
        block->active(is_visible);
      }
    };
} // namespace multires
} // namespace se

void MultiresTSDF::integrate(OctreeType&             map,
                             const se::Image<float>& depth_image,
                             const Eigen::Matrix4f&  T_CM,
                             const SensorImpl&       sensor,
                             const unsigned          frame) {

  using namespace std::placeholders;

  /* Retrieve the active list */
  std::vector<VoxelBlockType *> active_list;
  auto& block_buffer = map.pool().blockBuffer();

  /* Predicates definition */
  const float voxel_dim = map.dim() / map.size();
  auto in_frustum_predicate =
  std::bind(se::algorithms::in_frustum<VoxelBlockType>,
      std::placeholders::_1, voxel_dim, T_CM, sensor);
  auto is_active_predicate = [](const VoxelBlockType* block) {
    return block->active();
  };
  se::algorithms::filter(active_list, block_buffer, is_active_predicate,
                         in_frustum_predicate);

  std::deque<se::Node<VoxelType> *> node_queue;
  std::mutex deque_mutex;
  struct se::multires::MultiresTSDFUpdate block_update_funct(
      map, depth_image, T_CM, sensor, voxel_dim);
  se::functor::internal::parallel_for_each(active_list, block_update_funct);

  for (const auto &block : active_list) {
    if (block->parent()) {
      node_queue.push_back(block->parent());
    }
  }

  while (!node_queue.empty()) {
    se::Node<VoxelType>* node = node_queue.front();
    node_queue.pop_front();
    if (node->timestamp() == frame) {
      continue;
    }
    se::multires::propagate_up(node, map.voxelDepth(), frame);
    if (node->parent()) {
      node_queue.push_back(node->parent());
    }
  }
}
