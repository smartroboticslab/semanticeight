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

#include <se/voxel_implementations/MultiresOFusion/MultiresOFusion.hpp>
#include <se/voxel_implementations/MultiresOFusion/sensor_model.hpp>
#include "../../../se_denseslam/include/se/constant_parameters.h"
#include <set>
#include <se/utils/math_utils.h>
#include <Eigen/Core>
#include "se/image_utils.hpp"

struct AllocateAndUpdateRecurse {

AllocateAndUpdateRecurse(se::Octree<MultiresOFusion::VoxelType>&                        map,
                         std::vector<se::VoxelBlock<MultiresOFusion::VoxelType>*>&      block_list,
                         std::vector<std::set<se::Node<MultiresOFusion::VoxelType>*>>&  node_list,
                         const se::Image<float>&                                        depth_image,
                         const se::KernelImage* const                                   kernel_depth_image,
                         SensorImpl                                                     sensor,
                         const Eigen::Matrix4f&                                         T_CM,
                         const float                                                    voxel_dim,
                         const Eigen::Vector3f&                                         offset,
                         const size_t                                                   voxel_depth,
                         const float                                                    max_depth_value,
                         const unsigned                                                 frame) :
                         map_(map),
                         pool_(map.pool()),
                         block_list_(block_list),
                         node_list_(node_list),
                         depth_image_(depth_image),
                         kernel_depth_image_(kernel_depth_image),
                         sensor_(sensor),
                         T_CM_(T_CM),
                         mu_(sensor.mu),
                         voxel_dim_(voxel_dim),
                         offset_(offset),
                         voxel_depth_(voxel_depth),
                         max_depth_value_(max_depth_value),
                         frame_(frame),
                         zero_depth_band_(1.0e-6f),
                         size_to_radius(std::sqrt(3.0f) / 2.0f) {
  point_offset_ << 0, 1, 0, 1, 0, 1, 0, 1,
                   0, 0, 1, 1, 0, 0, 1, 1,
                   0, 0, 0, 0, 1, 1, 1, 1;
  };

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  se::Octree<MultiresOFusion::VoxelType>& map_;
  se::MemoryPool<MultiresOFusion::VoxelType>& pool_;
  std::vector<se::VoxelBlock<MultiresOFusion::VoxelType>*>& block_list_;
  std::vector<std::set<se::Node<MultiresOFusion::VoxelType>*>>& node_list_;
  const se::Image<float>&      depth_image_;
  const se::KernelImage* const kernel_depth_image_;
  SensorImpl sensor_;
  const Eigen::Matrix4f& T_CM_;
  const float mu_;
  const float voxel_dim_;
  const Eigen::Vector3f& offset_;
  const size_t voxel_depth_;
  const float max_depth_value_;
  const int frame_;
  Eigen::Matrix<float, 3, 8> point_offset_;
  const float zero_depth_band_;
  const float size_to_radius;

  static inline constexpr int max_block_scale_ =  se::math::log2_const(se::VoxelBlock<MultiresOFusion::VoxelType>::size);;

  /**
   * \brief Propagate a summary of the eight nodes children to its parent
   * \param node        Node to be summariesed
   * \param voxel_depth Maximum depth of the octree
   * \param frame       Current frame
   * \return data       Summary of the node
   */
  static MultiresOFusionData propagateUp(se::Node<MultiresOFusion::VoxelType>* node,
                                         const int                             voxel_depth,
                                         const unsigned                        frame) {
    if(!node->parent()) {
      node->timestamp(frame);
      return MultiresOFusion::VoxelType::invalid();
    }

    int data_count = 0;
    int y_max    = 0;
    int observed_count = 0;
    unsigned last_frame = 0;
    float x_max = 2 * MultiresOFusion::min_occupancy;
    for(int child_idx = 0; child_idx < 8; ++child_idx) {
      const auto& child_data = node->data_[child_idx];
      if (child_data.y > 0) { // At least 1 integration
        data_count++;
        if (child_data.x_max > x_max) {
          x_max = child_data.x_max;
          y_max = child_data.y;
        }
        if (child_data.frame > last_frame)
          last_frame = child_data.frame;
      }
      if (child_data.observed == true)
      {
        observed_count++;
      }
    }

    const unsigned int child_idx = se::child_idx(node->code_,
                                         se::keyops::depth(node->code_), voxel_depth);
    auto& node_data = node->parent()->data_[child_idx];
    if(data_count != 0) {
      node_data.x_max  = x_max;
      node_data.y      = y_max;
      node_data.frame  = last_frame;
      if (observed_count == 8)
        node_data.observed = true;
    }
    
    node->timestamp(frame);
    return node_data;
  }

  /**
   * \brief Summariese the values from the current integration scale recursively
   * up to the block's max scale.
   * \param block Voxel block to be updated
   * \param scale Scale from which propagate up voxel values
  */
  static void propagateUp(se::VoxelBlock<MultiresOFusion::VoxelType>* block,
                          const int                                   scale) {
    const Eigen::Vector3i block_coord = block->coordinates();
    const int block_size = se::VoxelBlock<MultiresOFusion::VoxelType>::size;
    for(int voxel_scale = scale; voxel_scale < max_block_scale_; ++voxel_scale) {
      const int stride = 1 << (voxel_scale + 1);
      for(int z = 0; z < block_size; z += stride) {
        for (int y = 0; y < block_size; y += stride) {
          for (int x = 0; x < block_size; x += stride) {
            const Eigen::Vector3i voxel_coord = block_coord + Eigen::Vector3i(x, y, z);
            float x_mean = 0;
            float y_mean = 0;
            int data_count = 0;
            int observed_count = 0;
            unsigned last_frame = 0;
            float x_max = MultiresOFusion::min_occupancy;
            for (int k = 0; k < stride; k += stride / 2)
              for (int j = 0; j < stride; j += stride / 2)
                for (int i = 0; i < stride; i += stride / 2) {
                  auto child_data = block->data(voxel_coord + Eigen::Vector3i(i, j, k), voxel_scale);
                  if (child_data.y > 0) {
                    x_mean += child_data.x;
                    y_mean += child_data.y;
                    data_count++;
                    if (child_data.x_max > x_max)
                      x_max = child_data.x_max;
                    if (child_data.frame > last_frame)
                      last_frame = child_data.frame;
                  }
                  if (child_data.observed) {
                    observed_count++;
                  }
                }
            auto voxel_data = block->data(voxel_coord, voxel_scale + 1);

            if (data_count != 0) {
              x_mean /= data_count;
              y_mean /= data_count;
              voxel_data.x = x_mean;
              voxel_data.x_last = x_mean;
              voxel_data.y = y_mean;
              voxel_data.y_last = y_mean;
              voxel_data.x_max = x_max;
              voxel_data.frame = last_frame;
              if (observed_count == 8)
                voxel_data.observed = true;
              block->data(voxel_coord, voxel_scale + 1, voxel_data);
            }
          }
        }
      }
    }
  }

  /**
   * \brief Interpolate occupancy log-odd via tri-linear interpolation based on
   * the neighboring values at a finest common scale
   * \param map     Currently saved map
   * \param block   Block containing the voxel value to be interpolated
   * \param voxel_coord     Voxel coordinates of the occupancy log-odd to be interpolated
   * \param scale   Scale at which to interpolate the occupancy log-odd
   * \param is_valid   Flag indicating that the interpolation was successful
   * \return val    Interpolated occupancy log-odd
  */
  static float interp(const se::Octree<MultiresOFusion::VoxelType>&     map,
                      const se::VoxelBlock<MultiresOFusion::VoxelType>* block,
                      const Eigen::Vector3i&                            voxel_coord,
                      const int                                         scale,
                      bool&                                             is_valid) {
    // Compute base point in parent block
    const int voxel_size   = se::VoxelBlock<MultiresOFusion::VoxelType>::size >> (scale + 1);
    const int stride = 1 << (scale + 1);

    const Eigen::Vector3f& offset = se::Octree<MultiresOFusion::VoxelType>::offset_;
    Eigen::Vector3i base_coord = stride * (voxel_coord.cast<float>()/stride - offset).cast<int>().cwiseMax(Eigen::Vector3i::Constant(0));

    const Eigen::Vector3f voxel_coord_f  = voxel_coord.cast<float>() + offset * (stride/2);
    Eigen::Vector3f base_coord_f = base_coord.cast<float>() + offset*(stride);
    Eigen::Vector3f factor = (voxel_coord_f - base_coord_f) / stride;

    for (int i = 0; i < 3; i++) {
      if (factor[i] < 0) {
        factor[i]  = 1 + factor[i];
        base_coord[i]   -= stride;
        base_coord_f[i] -= stride;
      }
    }

    float occupancies[8];
    se::internal_multires::gather_values(map, block->coordinates() + base_coord, scale + 1,
                                         [](const auto& data) { return data.x; }, occupancies);

    bool observed[8];
    se::internal_multires::gather_values(map, block->coordinates() + base_coord, scale + 1,
                            [](const auto& data) { return data.observed; }, observed);
    for(int i = 0; i < 8; ++i) {
      if(observed[i] == 0) {
        is_valid = false;
        return MultiresOFusion::VoxelType::initData().x;
      }
    }
    is_valid = true;

    const float v_000 = occupancies[0] * (1 - factor.x()) + occupancies[1] * factor.x();
    const float v_001 = occupancies[2] * (1 - factor.x()) + occupancies[3] * factor.x();
    const float v_010 = occupancies[4] * (1 - factor.x()) + occupancies[5] * factor.x();
    const float v_011 = occupancies[6] * (1 - factor.x()) + occupancies[7] * factor.x();

    const float v_0 = v_000 * (1 - factor.y()) + v_001 * (factor.y());
    const float v_1 = v_010 * (1 - factor.y()) + v_011 * (factor.y());

    const float val = v_0 * (1 - factor.z()) + v_1 * factor.z();

    return val;
  }

  /**
   * \brief Update a voxel block at a given scale by first propagating down the parent
   * values and then integrating the new measurement;
  */
  void propagateDownAndUpdate(se::VoxelBlock<MultiresOFusion::VoxelType>* block,
                              const int                                   scale) {

    const int block_size = se::VoxelBlock<MultiresOFusion::VoxelType>::size;
    const int parent_scale = scale + 1;
    const int stride = 1 << parent_scale;
    const int half_stride = stride >> 1;

    const Eigen::Vector3i block_coord = block->coordinates();
    const Eigen::Vector3f voxel_sample_offset = offset_;

    for(int z = 0; z < block_size; z += stride) {
      for(int y = 0; y < block_size; y += stride) {
        for(int x = 0; x < block_size; x += stride) {
          const Eigen::Vector3i parent_coord = block_coord + Eigen::Vector3i(x, y, z);
          auto data = block->data(parent_coord, parent_scale);
          float delta_x = data.x - data.x_last;
          float delta_y = data.y - data.y_last;
          for(int k = 0; k < stride; k += half_stride) {
            for(int j = 0; j < stride; j += half_stride) {
              for(int i = 0; i < stride; i += half_stride) {
                const Eigen::Vector3i voxel_coord = parent_coord + Eigen::Vector3i(i, j , k);
                auto voxel_data = block->data(voxel_coord, scale);

                bool is_valid = true;
                auto occupancy = interp(map_, block, voxel_coord - block_coord, scale, is_valid);

                if (voxel_data.observed == false) {
                  if (is_valid) {
                    voxel_data.x        = occupancy;
                    voxel_data.x_max    = voxel_data.x * data.y;
                    voxel_data.y        = data.y;
                    voxel_data.y_last   = data.y;
                    voxel_data.frame    = data.frame;
                    voxel_data.observed = true;
                    block->data(voxel_coord, scale, voxel_data);
                  }
                } else {
                  float delta_y_curr = (delta_y + voxel_data.y > MultiresOFusion::max_weight) ? MultiresOFusion::max_weight - voxel_data.y : delta_y;
                  float delta_x_curr = (voxel_data.x + delta_x < -5.015) ? -5.015 - voxel_data.x : delta_x;
                  voxel_data.x = voxel_data.x + delta_x_curr;
                  voxel_data.x = se::math::clamp(voxel_data.x + delta_x, -5.015f, 5.015f);
                  voxel_data.frame = data.frame;
                  voxel_data.observed = true;
                  if (is_valid) {
                    voxel_data.x = (voxel_data.x * voxel_data.y + occupancy * delta_y) / (voxel_data.y + delta_y);
                  }
                  voxel_data.y = voxel_data.y + delta_y_curr;
                  voxel_data.x_max = se::math::clamp(voxel_data.x * voxel_data.y, MultiresOFusion::min_occupancy, MultiresOFusion::max_occupancy);
                  block->data(voxel_coord, scale, voxel_data);
                }

                const Eigen::Vector3f point_C = (T_CM_ * (voxel_dim_ *
                                                          (voxel_coord.cast<float>() + voxel_sample_offset)).homogeneous()).head(3);
                Eigen::Vector2f pixel_f;
                if (sensor_.model.project(point_C, &pixel_f) != srl::projection::ProjectionStatus::Successful) {
                  block->data(voxel_coord, scale, voxel_data);
                  continue;
                }
                const Eigen::Vector2i pixel = round_pixel(pixel_f);
                const float depth_value = depth_image_[pixel.x() + depth_image_.width() * pixel.y()];
                if (depth_value <=  0) continue;

                const float proj_scale =  std::sqrt(1 + se::math::sq(point_C.x() / point_C.z()) +
                                                    se::math::sq(point_C.y() / point_C.z()));

                // Update the LogOdd
                sensor_model<OFusionModel<MultiresOFusion::VoxelType::VoxelData>>::updateBlock(point_C.z(), depth_value, mu_,
                                                                                               voxel_dim_, voxel_data, frame_, scale, proj_scale);
              }
            }
          }
        }
      }
    }
    block->current_scale(scale);
  }

  /**
   * \brief Update a voxel block at a given scale by first updating the observed state of all voxels at the
   * scale to true if the have been partially observed (y > 0);
  */
  void propagateUpAndUpdate(se::VoxelBlock<MultiresOFusion::VoxelType>* block,
                       const int scale) {
    const Eigen::Vector3i block_coord = block->coordinates();
    constexpr int block_size = se::VoxelBlock<MultiresOFusion::VoxelType>::size;
    block->current_scale(scale);
    const int stride = 1 << scale;

    const Eigen::Vector3f voxel_sample_offset = offset_;
    for (unsigned int z = 0; z < block_size; z += stride) {
      for (unsigned int y = 0; y < block_size; y += stride) {
#pragma omp simd
        for (unsigned int x = 0; x < block_size; x += stride) {
          const Eigen::Vector3i voxel_coord = block_coord + Eigen::Vector3i(x, y, z);
          const Eigen::Vector3f point_C = (T_CM_ * (voxel_dim_ *
                                                    (voxel_coord.cast<float>() + voxel_sample_offset)).homogeneous()).head(3);
          auto voxel_data = block->data(voxel_coord, scale);
          Eigen::Vector2f pixel_f;
          if (sensor_.model.project(point_C, &pixel_f) != srl::projection::ProjectionStatus::Successful) {
            block->data(voxel_coord, scale, voxel_data);
            continue;
          }
          const Eigen::Vector2i pixel = round_pixel(pixel_f);
          const float depth_value = depth_image_[pixel.x() + depth_image_.width() * pixel.y()];
          if (depth_value <=  0) continue;

          const float proj_scale =  std::sqrt(1 + se::math::sq(point_C.x() / point_C.z()) +
                                              se::math::sq(point_C.y() / point_C.z()));

          // Update the LogOdd
          sensor_model<OFusionModel<MultiresOFusion::VoxelType::VoxelData>>::updateBlock(point_C.z(), depth_value, mu_,
                                                                                         voxel_dim_, voxel_data, frame_, scale, proj_scale);
          block->data(voxel_coord, scale, voxel_data);
        }
      }
    }
  }

  /**
   * \brief Compute integration scale for a given voxel block and update all voxels that project into the image plane.
  */
  void updateBlock(se::VoxelBlock<MultiresOFusion::VoxelType>* block, int min_scale = 0) {
    constexpr int block_size = se::VoxelBlock<MultiresOFusion::VoxelType>::size;
    const Eigen::Vector3i block_coord = block->coordinates();
    const Eigen::Vector3f block_sample_offset = (offset_.array().colwise() *
                                                 Eigen::Vector3f::Constant(block_size).array());
    const float block_diff = (T_CM_ * (voxel_dim_ * (block_coord.cast<float>() +
                                                   block_sample_offset)).homogeneous()).head(3).z();
    const int last_scale = block->current_scale();
    const int scale = std::max(sensor_.computeIntegrationScale(
        block_diff, voxel_dim_, last_scale, block->min_scale(), map_.maxBlockScale()), last_scale - 1);
    block->min_scale(block->min_scale() < 0 ? scale : std::min(block->min_scale(), scale));

    if(last_scale > scale) {
      // Down propagate values first
      propagateDownAndUpdate(block, scale);
      return;
    } else if (last_scale < scale) {
      // Update observed state at new scale
      propagateUpAndUpdate(block, scale);
      return;
    }

    block->current_scale(scale);
    const int stride = 1 << scale;

    const Eigen::Vector3f voxel_sample_offset = offset_;
    for (unsigned int z = 0; z < block_size; z += stride) {
      for (unsigned int y = 0; y < block_size; y += stride) {
#pragma omp simd
        for (unsigned int x = 0; x < block_size; x += stride) {
          const Eigen::Vector3i voxel_coord = block_coord + Eigen::Vector3i(x, y, z);
          const Eigen::Vector3f point_C = (T_CM_ * (voxel_dim_ *
                                                   (voxel_coord.cast<float>() + voxel_sample_offset)).homogeneous()).head(3);
          Eigen::Vector2f pixel_f;
          if (sensor_.model.project(point_C, &pixel_f) != srl::projection::ProjectionStatus::Successful) {
            continue;
          }
          const Eigen::Vector2i pixel = round_pixel(pixel_f);
          const float depth_value = depth_image_[pixel.x() + depth_image_.width() * pixel.y()];
          if (depth_value <=  0) continue;
          auto voxel_data = block->data(voxel_coord, scale);

          const float proj_scale =  std::sqrt(1 + se::math::sq(point_C.x() / point_C.z()) +
                                              se::math::sq(point_C.y() / point_C.z()));

          // Update the LogOdd
          sensor_model<OFusionModel<MultiresOFusion::VoxelType::VoxelData>>::updateBlock(point_C.z(), depth_value, mu_,
              voxel_dim_, voxel_data, frame_, scale, proj_scale);
          block->data(voxel_coord, scale, voxel_data);
        }
      }
    }
  }

  /**
   * \brief Recursively reduce all children by the minimum occupancy log-odd for a single integration.
  */
  void freeNodeRecurse(se::Node<MultiresOFusion::VoxelType>* node, int depth) {
    for (int child_idx = 0; child_idx < 8; child_idx++) {
      auto child = node->child(child_idx);
      if (child) {
        if (child->isBlock()) {
          se::VoxelBlock<MultiresOFusion::VoxelType>* block = dynamic_cast<se::VoxelBlock<MultiresOFusion::VoxelType>*>(child);
          // Voxel block has a low variance (unknown data and frustum crossing allowed). Update data at a minimum
          // free space integration scale or coarser (depending on later scale selection).
          updateBlock(block, MultiresOFusion::fs_integr_scale);
#pragma omp critical (voxel_lock)
          { // Add voxel block to voxel block list for later up propagation
            block_list_.push_back(dynamic_cast<se::VoxelBlock<MultiresOFusion::VoxelType>*>(block));
          }
        } else {
          freeNodeRecurse(child, depth + 1);
        }
      } else {
        sensor_model<OFusionModel<MultiresOFusion::VoxelType::VoxelData>>::freeNode(node->data_[child_idx], frame_);
#pragma omp critical (node_lock)
        { // Add node to node list for later up propagation (finest node for this branch)
          node_list_[depth].insert(node);
        }
      }
    }
  }

  /**
   * \brief Reduce the nodes occupancy log-odd by the minimum occupancy log-odd for a single integration.
  */
  void freeNode(MultiresOFusion::VoxelType::VoxelData& data) {
    sensor_model<OFusionModel<MultiresOFusion::VoxelType::VoxelData>>::freeNode(data, frame_);
  }

  /**
   * \brief Get reference to a child value for a given parent node.
   * \param parent Pointer to the parent node
   * \param rel_step    Child relative position with respect to the parent
   * \return Reference to child's value
   */
  MultiresOFusion::VoxelType::VoxelData& getChildValue(se::Node<MultiresOFusion::VoxelType>* parent,
                                                       const Eigen::Vector3i&                rel_step)
  {
    assert(!parent->isBlock() && "Father node must not be a leaf");

    return parent->data_[rel_step.x() + rel_step.y() * 2 + rel_step.z() * 4];
  }

  /**
   * \brief Allocate a node for a given parent node
   * \param parent      Pointer to the parent node
   * \param node_coord  Coordinates of the node in voxel coordinates
   * \param rel_step    Node relative direction with respect to the parent
   * \param node_size   Size of the node in voxel units
   * \param depth       Depth of the node in the octree
   * \return node       Pointer to the allocated node
   */
  auto allocateNode(se::Node<MultiresOFusion::VoxelType>* parent,
                    const Eigen::Vector3i&                node_coord,
                    const Eigen::Vector3i&                rel_step,
                    const int                             node_size,
                    const unsigned                        depth) -> se::Node<MultiresOFusion::VoxelType>* {
    int node_idx = rel_step.x() + rel_step.y() * 2 + rel_step.z() * 4;
    se::Node<MultiresOFusion::VoxelType>*& node = parent->child(node_idx);
    if(node) {
      // Return node if already allocated
      return node;
    }

    /*
     * Otherwise allocate and initalise child
     */

    // Update child mask
    parent->children_mask_ = parent->children_mask_ | (1 << node_idx);

    if(node_size <= static_cast<int>(se::VoxelBlock<MultiresOFusion::VoxelType>::size)) {
      // Allocate voxel block
#pragma omp critical (voxel_alloc_lock)
      {
        auto init_data = parent->data_[node_idx]; // Initalise child with parent value
        // Initialise remaining values that are not kept track of at node depth
        init_data.x      = (init_data.y > 0) ? init_data.x_max / init_data.y : 0;
        init_data.x_last = init_data.x;
        init_data.y_last = init_data.y;
        node = map_.pool().acquireBlock(init_data);
        se::VoxelBlock<MultiresOFusion::VoxelType>* block = dynamic_cast<se::VoxelBlock<MultiresOFusion::VoxelType>*>(node);
        block->coordinates(node_coord);
        block->size_ = node_size;
        block->code_ = se::keyops::encode(node_coord.x(), node_coord.y(), node_coord.z(),
                                          map_.blockDepth(), voxel_depth_);
        block->parent() = parent;
      }
    } else {
      // Allocate node
#pragma omp critical (node_alloc_lock)
      {
        node = map_.pool().acquireNode(parent->data_[node_idx]);
        node->size_ = node_size;
        node->code_ = se::keyops::encode(node_coord.x(), node_coord.y(), node_coord.z(), depth, voxel_depth_);
        node->parent() = parent;
        // Initalise all children with parent value if it differs from the default value
        if (node->parent()->data_[node_idx].x_max != MultiresOFusion::VoxelType::initData().x_max) {
          for (int child_idx = 0; child_idx < 8; child_idx++) {
            node->data_[child_idx] = node->parent()->data_[node_idx];
          }
        }
      }
    }

    assert(node && "Voxel alloc failure");
    return node;
  }

  /**
   * \brief Verify if the camera is inside a given node
   * \param node_coord  Coordinates of the node in voxel coordinates
   * \param node_size   Size of the child node in voxel units
   * \param t_wc        Translational component of the camera position
   * \return True/false Statement if the camera is inside the node.
   */
  bool cameraInNode(const Eigen::Vector3i& node_coord,
                    const int              node_size,
                    const Eigen::Matrix4f& T_MC) {
    const Eigen::Vector3f voxel_coord = se::math::toTranslation(T_MC) / voxel_dim_;
    if (   voxel_coord.x() >= node_coord.x() && voxel_coord.x() <= node_coord.x() + node_size
        && voxel_coord.y() >= node_coord.y() && voxel_coord.y() <= node_coord.y() + node_size
        && voxel_coord.z() >= node_coord.z() && voxel_coord.z() <= node_coord.z() + node_size) {
      return true;
    }
    return false;
  }

  /**
   * \brief Verify if the node crosses the camera frustum excluding the case of the camera in the octant
   * \param proj_corner_stati Stati of the projection of the eight octant corners into the image plane
   * \return True/false statement node crosses the camera frustum.
   */
  bool crossesFrustum(std::vector<srl::projection::ProjectionStatus>&  proj_corner_stati) {
    for (int corner_idx = 0; corner_idx < 8; corner_idx++) {
      if (proj_corner_stati[corner_idx] == srl::projection::ProjectionStatus::Successful) {
        return true;
      }
    }
    return false;
  }

  /**
   * \brief Update and allocate a given node and all its children recursively
   * \param node_coord  Corner of the node to be processed (bottom, left, front coordinates)
   * \param node_size   Size in [voxel] of the node to be processed
   * \param depth       Depth of the node to be processed (root = 0)
   * \param rel_step    Relative direction of node within parent node (e.g. [1, 0, 1], [0, 1, 1])
   * \param parent      Pointer to the nodes parent
   */
  void operator()(const Eigen::Vector3i&                 node_coord,
                  const int                              node_size,
                  const int                              depth,
                  const Eigen::Vector3i&                 rel_step,
                  se::Node <MultiresOFusion::VoxelType>* parent) {
    /*
     * Approximate max and min depth to quickly check if the node is behind the camera or maximum depth
     */
    // Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", " << ", ";");
    // std::cout << "=====" << std::endl;
    // std::cout << "node_coord " << node_coord.format(CommaInitFmt) << std::endl;
    // std::cout << "node_size  << " << node_size << std::endl;

    // Compute the node centre's depth in the camera frame
    const Eigen::Vector3f node_centre_point_M = voxel_dim_ * (node_coord.cast<float>() + Eigen::Vector3f(node_size, node_size, node_size) / 2.f);
    const Eigen::Vector3f node_centre_point_C = (T_CM_ * node_centre_point_M.homogeneous()).head(3);

    // Extend and reduce the depth by the sphere radius covering the entire cube
    const float approx_depth_value_max = node_centre_point_C.z() + node_size * size_to_radius * voxel_dim_;
    const float approx_depth_value_min = node_centre_point_C.z() - node_size * size_to_radius * voxel_dim_;

    // CASE 0.1 (OUT OF BOUNDS): Block is behind the camera or behind the maximum depth value
    if (approx_depth_value_min > max_depth_value_ ||
        // std::cout << "CASE 0.1" << std::endl;
        approx_depth_value_max <= 0) {
      return;
    }

    /*
     * Approximate a 2D bounding box covering the projected node in the image plane
     */
    bool should_split = false;

    // Compute the 8 corners of the node to be evaluated
    Eigen::Matrix<float, 3, 8> node_corner_coords_f = (node_size * point_offset_).colwise() + node_coord.cast<float>();
    Eigen::Matrix<float, 3, 8> node_corner_points_C =
        (T_CM_  * Eigen::Vector4f(voxel_dim_, voxel_dim_, voxel_dim_, 1.f).asDiagonal() * node_corner_coords_f.colwise().homogeneous()).topRows(3);

    // std::cout << "node_corner_points_C " << node_corner_points_C.row(2).format(CommaInitFmt) << std::endl;

    Eigen::VectorXi node_corners_infront(8);
    node_corners_infront << 1, 1, 1, 1, 1, 1, 1, 1;
    for (int corner_idx = 0; corner_idx < 8; corner_idx++) {
      if (node_corner_points_C(2, corner_idx) < zero_depth_band_) {
        node_corners_infront(corner_idx) = 0;
      }
    }

    // std::cout << "node_corners_infront << " << node_corners_infront(0) << ", "
    //                                         << node_corners_infront[1] << ", "
    //                                         << node_corners_infront[2] << ", "
    //                                         << node_corners_infront[3] << ", "
    //                                         << node_corners_infront[4] << ", "
    //                                         << node_corners_infront[5] << ", "
    //                                         << node_corners_infront[6] << ", "
    //                                         << node_corners_infront[7] << std::endl;

    int num_node_corners_infront = node_corners_infront.sum();

    // CASE 0.2 (OUT OF BOUNDS): Node is behind the camera
    if (num_node_corners_infront == 0) {
      // std::cout << "CASE 0.2" << std::endl;
      return;
    }

    // Project the 8 corners into the image plane
    Eigen::Matrix2Xf proj_node_corner_pixels_f(2, 8);
    Eigen::VectorXf node_corners_diff = node_corner_points_C.row(2);
    std::vector<srl::projection::ProjectionStatus> proj_node_corner_stati;
    sensor_.model.projectBatch(node_corner_points_C, &proj_node_corner_pixels_f, &proj_node_corner_stati);

    int low_variance; // -1 := low variance infront of the surface, 0 := high variance, 1 = low_variance behind the surface
    se::KernelImage::Pixel kernel_pixel; // min, max pixel batch depth + crossing frustum state + contains unknown values state

    if(depth < map_.blockDepth() + 1) {
      if (num_node_corners_infront < 8) {
        // CASE 1 (CAMERA IN NODE):
        if (cameraInNode(node_coord, node_size, T_CM_)) {
          should_split = true;
        } else if (crossesFrustum(proj_node_corner_stati)) {
//          // EXCEPTION: The node is crossing the frustum boundary, but already reached the min allocation scale for crossing nodes
//          if (node_size == se::VoxelBlock<MultiresOFusion::VoxelType>::size) {
//            return;
//          }
          should_split = true;
        } else {
          return;
        }
        // CASE 2 (FRUSTUM BOUNDARY): Node partly behind the camera and crosses the the frustum boundary
      } else {
        // Compute the minimum and maximum pixel values to generate the bounding box
        Eigen::Vector2i image_bb_min = proj_node_corner_pixels_f.rowwise().minCoeff().cast<int>();
        Eigen::Vector2i image_bb_max = proj_node_corner_pixels_f.rowwise().maxCoeff().cast<int>();
        float node_dist_min = node_corners_diff.minCoeff();
        float node_dist_max = node_corners_diff.maxCoeff();

        kernel_pixel = kernel_depth_image_->conservativeQuery(image_bb_min, image_bb_max);

        // CASE 0.3 (OUT OF BOUNDS): The node is behind surface
        if (approx_depth_value_min > kernel_pixel.max + sensor_model<OFusionModel<MultiresOFusion::VoxelType::VoxelData>>::computeSigma()) {
          return;
        }

        // CASE 0.4 (OUT OF BOUNDS): The node is outside frustum (i.e left, right, below, above) or all pixel values are unknown -> return intermediately
        if(kernel_pixel.status_known == se::KernelImage::Pixel::statusKnown::unknown) {
          return;
        }

        low_variance = sensor_model<OFusionModel<MultiresOFusion::VoxelType>>::lowVariance(
            kernel_pixel.min, kernel_pixel.max, node_dist_min, node_dist_max, mu_);

        const se::key_t node_key = se::keyops::encode(node_coord.x(), node_coord.y(), node_coord.z(), depth, voxel_depth_);
        const unsigned int child_idx = se::child_idx(node_key, depth, map_.voxelDepth());

        // CASE 1 (REDUNDANT DATA): Depth values in the bounding box are far away from the node or unknown (1).
        //                          The node to be evaluated is free (2) and fully observed (3),
        if(low_variance != 0 && parent->data_[child_idx].x_max <= MultiresOFusion::min_occupancy && parent->data_[child_idx].observed == true) {
          return;
        }

        // CASE 2 (FRUSTUM BOUNDARY): The node is crossing the frustum boundary
        if(kernel_pixel.status_crossing == se::KernelImage::Pixel::statusCrossing::crossing) {
          // EXCEPTION: The node is crossing the frustum boundary, but already reached the min allocation scale for crossing nodes
//          if (node_size == se::VoxelBlock<MultiresOFusion::VoxelType>::size) {
//            return;
//          }
          should_split = true;
        }

        // CASE 3: The node is inside the frustum, but projects into partially known pixel
        else if(kernel_pixel.status_known == se::KernelImage::Pixel::statusKnown::part_known) {
          should_split = true;
        }

        // CASE 4: The node is inside the frustum with only known data + node has a potential high variance
        else if (low_variance == 0){
          should_split = true;
        }
      }
    }

    if(should_split) {
      // Returns a pointer to the according node if it has previously been allocated.
      se::Node<MultiresOFusion::VoxelType>* node = allocateNode(parent, node_coord, rel_step, node_size, depth);
      if(node->isBlock()) { // Evaluate the node directly if it is a voxel block
        node->active(true);
        // Cast from node to voxel block
        se::VoxelBlock<MultiresOFusion::VoxelType>* block = dynamic_cast<se::VoxelBlock<MultiresOFusion::VoxelType>*>(node);
        if (low_variance != 0) {
          // Voxel block has a low variance (unknown data and frustum crossing allowed). Update data at a minimum
          // free space integration scale or coarser (depending on later scale selection).
          updateBlock(block, MultiresOFusion::fs_integr_scale);
        } else {
          // Otherwise update values at the finest integration scale or coarser (depending on later scale selection).
          updateBlock(block, 0);
        }
#pragma omp critical (voxel_lock)
        { // Add voxel block to voxel block list for later up propagation
          block_list_.push_back(dynamic_cast<se::VoxelBlock<MultiresOFusion::VoxelType>*>(node));
        }
      }
      else {
        // Split! Start recursive process
#pragma omp parallel for
        for(int child_idx = 0; child_idx < 8; ++child_idx) {
          int child_size = node_size / 2;
          Eigen::Vector3i child_rel_step = Eigen::Vector3i((child_idx & 1) > 0, (child_idx & 2) > 0, (child_idx & 4) > 0);
          Eigen::Vector3i child_coord = node_coord + child_rel_step * child_size;
          (*this)(child_coord, child_size, depth + 1, child_rel_step, node);
        }
      }
    } else {
      assert(depth);
      int node_idx = rel_step.x() + rel_step.y() * 2 + rel_step.z() * 4;
      se::Node<MultiresOFusion::VoxelType>* node = parent->child(node_idx);
      if (!node) {
        // Node does not exist -> Does NOT have children that need to be updated
        if (low_variance == -1) {
          // Free node
          MultiresOFusion::VoxelType::VoxelData& node_value = getChildValue(parent, rel_step);
          freeNode(node_value);
#pragma omp critical (node_lock)
          { // Add node to node list for later up propagation (finest node for this branch)
            node_list_[depth - 1].insert(parent);
          }
        } // else node has low variance behind surface (ignore)
      } else {
        // Node does exist -> Does POTENTIALLY have children that that need to be updated
          if (node->isBlock()) {
            // Node is a voxel block -> Does NOT have children that need to be updated
            se::VoxelBlock<MultiresOFusion::VoxelType>* block = dynamic_cast<se::VoxelBlock<MultiresOFusion::VoxelType>*>(node);
            // Node has a low variance (unknown data and frustum crossing allowed). Update data at a minimum
            // free space integration scale or coarser (depending on later scale selection).
            updateBlock(block, MultiresOFusion::fs_integr_scale);
#pragma omp critical (voxel_lock)
            { // Add voxel block to voxel block list for later up propagation
              block_list_.push_back(dynamic_cast<se::VoxelBlock<MultiresOFusion::VoxelType>*>(node));
            }
          } else {
            // Node has children
            if (low_variance == -1) {
              //Free node recursively
              freeNodeRecurse(node, depth);
            } // else node has low variance behind surface (ignore)
          }
      }
    }
  };

  /**
   * \brief Propage all newly integrated values from the voxel block depth up to the root of the octree
   */
  void propagateNodes() {
    for(const auto& block : block_list_) {
      if(block->parent()) {
        propagateUp(block, block->current_scale());
        node_list_[map_.blockDepth() - 1].insert(block->parent());
        const unsigned int child_idx = se::child_idx(block->code_,
                                             se::keyops::depth(block->code_), map_.voxelDepth());
        auto data = block->data(block->coordinates(), se::math::log2_const(se::VoxelBlock<MultiresOFusion::VoxelType>::size));
        auto& parent_data = block->parent()->data_[child_idx];
        parent_data = data;

        auto test = block->parent()->data_[child_idx];
        if (data.observed && data.x_max <= MultiresOFusion::min_occupancy + 0.01) { //&& data.x * data.y <= MultiresOFusion::min_occupancy
          pool_.deleteBlock(block, voxel_depth_);
        }
      }
    }

    for (int d = map_.blockDepth() - 1; d > 0; d--) {
      std::set<se::Node<MultiresOFusion::VoxelType>*>::iterator it;
      for (it = node_list_[d].begin(); it != node_list_[d].end(); ++it) {
        se::Node<MultiresOFusion::VoxelType>* node = *it;
        if(node->timestamp() == frame_) continue;
        if(node->parent()) {
          auto node_data = AllocateAndUpdateRecurse::propagateUp(node, voxel_depth_, frame_);
          node_list_[d-1].insert(node->parent());
          if (node_data.observed && node_data.x_max <= MultiresOFusion::min_occupancy + 0.01) {
            pool_.deleteNode(node, voxel_depth_);
          }
        }
      }
    }
  }
};

/**
 * \brief Update and allocate all nodes and voxel blocks in the camera frustum using a map-to-camera integration scheme.
 * Starting from the eight roots children each node is projected into the image plane and an appropriate allocation and
 * updating scale is choosen depending on the variation of occupancy log-odds within each node/voxel block
 */
void MultiresOFusion::integrate(se::Octree<MultiresOFusion::VoxelType>& map,
                                const se::Image<float>&                 depth_image,
                                const Eigen::Matrix4f&                  T_CM,
                                const SensorImpl&                       sensor,
                                const unsigned                          frame) {
  // Create min/map depth pooling image for different bounding box sizes
  const std::unique_ptr<se::KernelImage> kernel_depth_image(new se::KernelImage(depth_image));

  const float max_depth_value = std::min(sensor.far_plane, kernel_depth_image->maxValue() + sensor_model<OFusionModel<MultiresOFusion::VoxelType::VoxelData>>::computeSigma());
  const float voxel_dim = map.dim() / map.size();
  const Eigen::Vector3f offset = se::Octree<MultiresOFusion::VoxelType>::offset_;

  std::vector<se::VoxelBlock<MultiresOFusion::VoxelType>*> block_list;
  std::vector<std::set<se::Node<MultiresOFusion::VoxelType>*>> node_list(map.blockDepth());
  AllocateAndUpdateRecurse funct(map, block_list, node_list, depth_image, kernel_depth_image.get(), sensor, T_CM,
      voxel_dim, offset, map.voxelDepth(), max_depth_value, frame);

  // Launch on the 8 voxels of the first depth
#pragma omp parallel for
  for(int child_idx = 0; child_idx < 8; ++child_idx)
  {
    int child_size = map.size() / 2;
    Eigen::Vector3i child_rel_step = Eigen::Vector3i((child_idx & 1) > 0, (child_idx & 2) > 0, (child_idx & 4) > 0);
    Eigen::Vector3i child_coord = child_rel_step * child_size; // Because, + corner is (0, 0, 0) at root depth
    funct(child_coord, child_size, 1, child_rel_step, map.root());
  }

  funct.propagateNodes();
}