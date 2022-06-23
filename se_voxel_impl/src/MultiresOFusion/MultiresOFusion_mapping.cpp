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

#include <Eigen/Core>
#include <set>

#include "../../../se_denseslam/include/se/constant_parameters.h"
#include "se/frontiers.hpp"
#include "se/image_utils.hpp"
#include "se/str_utils.hpp"
#include "se/utils/math_utils.h"
#include "se/voxel_implementations/MultiresOFusion/MultiresOFusion.hpp"
#include "se/voxel_implementations/MultiresOFusion/updating_model.hpp"

using OctreeType = se::Octree<MultiresOFusion::VoxelType>;
using VoxelData = MultiresOFusion::VoxelType::VoxelData;
using NodeType = se::Node<MultiresOFusion::VoxelType>;
using VoxelBlockType = MultiresOFusion::VoxelType::VoxelBlockType;

template<typename SensorImplType>
struct MultiresOFusionUpdate {
    /**
   * \param[in]  map                  The reference to the map to be updated.
   * \param[out] block_list           The list of blocks to be updated (used for up-propagation in later stage).
   * \param[out] node_list            The list of nodes that have been updated (used for up-propagation in later stage.
   * \param[out] free_list            The list verifying if the updated block should be freed. <bool>
   * \param[out] low_variance_list    The list verifying if the updated block has low variance. <bool>
   * \param[out] projects_inside_list The list verifying if the updated block projects completely into the image. <bool>
   * \param[in]  depth_image          The depth image to be integrated.
   * \param[in]  pooling_depth_image  The pointer to the pooling image created from the depth image.
   * \param[in]  sensor               The sensor model.
   * \param[in]  T_CM                 The transformation from map to camera frame.
   * \param[in]  voxel_dim            The dimension in meters of the finest voxel / map resolution.
   * \param[in]  voxel_depth          The tree depth of the finest voxel.
   * \param[in]  max_depth_value      The maximum depth value in the image.
   * \param[in]  frame                The frame number to be integrated.
   */
    MultiresOFusionUpdate(OctreeType& map,
                          std::vector<VoxelBlockType*>& block_list,
                          std::vector<std::set<NodeType*>>& node_list,
                          std::vector<bool>& free_list,
                          std::vector<bool>& low_variance_list,
                          std::vector<bool>& projects_inside_list,
                          const se::Image<float>& depth_image,
                          const se::Image<uint32_t>& rgba_image,
                          const cv::Mat& fg_image,
                          const se::DensePoolingImage<SensorImpl>* const pooling_depth_image,
                          const SensorImplType sensor,
                          const Eigen::Matrix4f& T_CM,
                          const float voxel_dim,
                          const size_t voxel_depth,
                          const float max_depth_value,
                          const unsigned frame) :
            map_(map),
            pool_(map.pool()),
            block_list_(block_list),
            node_list_(node_list),
            free_list_(free_list),
            low_variance_list_(low_variance_list),
            projects_inside_list_(projects_inside_list),
            depth_image_(depth_image),
            rgba_image_(rgba_image),
            fg_image_(fg_image),
            pooling_depth_image_(pooling_depth_image),
            sensor_(sensor),
            T_CM_(T_CM),
            voxel_dim_(voxel_dim),
            sample_offset_frac_(map.sample_offset_frac_),
            voxel_depth_(voxel_depth),
            max_depth_value_(max_depth_value),
            frame_(frame),
            zero_depth_band_(1.0e-6f),
            size_to_radius_(std::sqrt(3.0f) / 2.0f)
    {
        corner_rel_steps_ << 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1;
    };

    OctreeType& map_;
    MultiresOFusion::VoxelType::MemoryPoolType& pool_;
    std::vector<VoxelBlockType*>& block_list_;
    std::vector<std::set<NodeType*>>& node_list_;
    std::vector<bool>& free_list_;
    std::vector<bool>& low_variance_list_;
    std::vector<bool>& projects_inside_list_;
    const se::Image<float>& depth_image_;
    const se::Image<uint32_t>& rgba_image_;
    const cv::Mat& fg_image_;
    const se::DensePoolingImage<SensorImpl>* const pooling_depth_image_;
    const SensorImplType sensor_;
    const Eigen::Matrix4f& T_CM_;
    const float voxel_dim_;
    const Eigen::Vector3f& sample_offset_frac_;
    const size_t voxel_depth_;
    const float max_depth_value_;
    const unsigned frame_;
    Eigen::Matrix<float, 3, 8> corner_rel_steps_;
    const float zero_depth_band_;
    const float size_to_radius_;

    static constexpr int max_block_scale_ =
        se::math::log2_const(MultiresOFusion::VoxelBlockType::size_li);

    void freeBlock(VoxelBlockType* block)
    {
        // Compute the point of the block centre in the sensor frame
        const unsigned int block_size = VoxelBlockType::size_li;
        const Eigen::Vector3i block_coord = block->coordinates();
        const Eigen::Vector3f block_centre_coord_f =
            se::get_sample_coord(block_coord, block_size, Eigen::Vector3f::Constant(0.5f));
        const Eigen::Vector3f block_centre_point_C =
            (T_CM_ * (voxel_dim_ * block_centre_coord_f).homogeneous()).head(3);

        /// Compute the integration scale
        // The last integration scale
        const int last_scale = block->current_scale();

        // The recommended integration scale
        const int computed_integration_scale = sensor_.computeIntegrationScale(
            block_centre_point_C, voxel_dim_, last_scale, block->min_scale(), map_.maxBlockScale());

        // The minimum integration scale (change to last if data has already been integrated)
        const int min_integration_scale =
            ((block->min_scale() == -1 || block->maxValue() < 0.95 * MultiresOFusion::log_odd_min))
            ? MultiresOFusion::fs_integr_scale
            : std::max(0, last_scale - 1);
        const int max_integration_scale = (block->min_scale() == -1)
            ? max_block_scale_
            : std::min(max_block_scale_, last_scale + 1);

        // The final integration scale
        const int recommended_scale = std::min(
            std::max(min_integration_scale, computed_integration_scale), max_integration_scale);

        int integration_scale = last_scale;

        /// If no data has been integrated in the block before (block->min_scale() == -1), use the computed integration scale.
        if (block->min_scale() == -1) {
            // Make sure the block is allocated up to the integration scale
            integration_scale = recommended_scale;
            block->allocateDownTo(integration_scale);
            block->current_scale(integration_scale);
            block->initCurrCout();
            block->setInitData(MultiresOFusion::VoxelType::initData());
        }
        else if (recommended_scale != last_scale) { ///<< Potential double integration

            if (recommended_scale
                != block->buffer_scale()) { ///<< Start from scratch and initialise buffer

                block->initBuffer(recommended_scale);

                if (recommended_scale < last_scale) {
                    const int parent_scale = last_scale;
                    const unsigned int size_at_parent_scale_li = block->size_li >> parent_scale;
                    const unsigned int size_at_parent_scale_sq =
                        se::math::sq(size_at_parent_scale_li);

                    const unsigned int size_at_buffer_scale_li = size_at_parent_scale_li << 1;
                    const unsigned int size_at_buffer_scale_sq =
                        se::math::sq(size_at_buffer_scale_li);

                    for (unsigned int z = 0; z < size_at_parent_scale_li; z++) {
                        for (unsigned int y = 0; y < size_at_parent_scale_li; y++) {
#pragma omp simd
                            for (unsigned int x = 0; x < size_at_parent_scale_li; x++) {
                                const int parent_idx =
                                    x + y * size_at_parent_scale_li + z * size_at_parent_scale_sq;
                                const auto& parent_data =
                                    block->currData(parent_idx); // TODO: CAN BE MADE FASTER

                                for (unsigned int k = 0; k < 2; k++) {
                                    for (unsigned int j = 0; j < 2; j++) {
                                        for (unsigned int i = 0; i < 2; i++) {
                                            const int buffer_idx = (2 * x + i)
                                                + (2 * y + j) * size_at_buffer_scale_li
                                                + (2 * z + k) * size_at_buffer_scale_sq;
                                            auto& buffer_data = block->bufferData(
                                                buffer_idx); ///<< Fetch value from buffer.

                                            buffer_data.x = parent_data.x;
                                            buffer_data.fg = parent_data.fg;
                                            buffer_data.fg_count = parent_data.fg_count;
                                            buffer_data.y =
                                                parent_data.y; // (parent_data.y > 0) ? 1 : 0;
                                            buffer_data.r = parent_data.r;
                                            buffer_data.g = parent_data.g;
                                            buffer_data.b = parent_data.b;
                                            buffer_data.observed =
                                                false; ///<< Set falls such that the observe count can work properly
                                            buffer_data.frontier =
                                                parent_data
                                                    .frontier; // TODO SEM does this make sense?

                                        } // i
                                    }     // j
                                }         // k

                            } // x
                        }     // y
                    }         // z
                }
            }

            /// Integrate data into buffer.
            const unsigned int size_at_recommended_scale_li = block->size_li >> recommended_scale;
            const unsigned int size_at_recommended_scale_sq =
                se::math::sq(size_at_recommended_scale_li);

            for (unsigned int z = 0; z < size_at_recommended_scale_li; z++) {
                for (unsigned int y = 0; y < size_at_recommended_scale_li; y++) {
#pragma omp simd
                    for (unsigned int x = 0; x < size_at_recommended_scale_li; x++) {
                        const int buffer_idx =
                            x + y * size_at_recommended_scale_li + z * size_at_recommended_scale_sq;
                        auto& buffer_data =
                            block->bufferData(buffer_idx); /// \note pass by reference now.
                        block->incrBufferObservedCount(updating_model::freeVoxel(buffer_data));

                    } // x
                }     // y
            }         // z

            block->incrBufferIntegrCount();

            if (block->switchData()) {
                return;
            }
        }
        else {
            block->resetBuffer();
        }

        const unsigned int size_at_integration_scale_li = block->size_li >> integration_scale;
        const unsigned int size_at_integration_scale_sq =
            se::math::sq(size_at_integration_scale_li);

        for (unsigned int z = 0; z < size_at_integration_scale_li; z++) {
            for (unsigned int y = 0; y < size_at_integration_scale_li; y++) {
                //#pragma omp simd // Can't have omp critical in omp simd
                for (unsigned int x = 0; x < size_at_integration_scale_li; x++) {
                    // Update the voxel data based using the depth measurement
                    const int voxel_idx =
                        x + y * size_at_integration_scale_li + z * size_at_integration_scale_sq;
                    auto& voxel_data = block->currData(voxel_idx); /// \note pass by reference now.
                    block->incrCurrObservedCount(updating_model::freeVoxel(voxel_data));
                } // x
            }     // y
        }         // z

        block->incrCurrIntegrCount();
    }

    /**
   * \brief Compute integration scale for a given voxel block and update all voxels that project into the image plane.
   *
   * \note The minimum integration scale has only an effect if no data has been integrated into the block yet, i.e.
   *       the integration scale of the block has not been initialised yet.
   *
   * \param[out] block                 The block to be updated.
   * \param[out] min_integration_scale The minimum integration scale.
   */
    void updateBlock(VoxelBlockType* block, bool low_variance, bool project_inside)
    {
        // Compute the point of the block centre in the sensor frame
        const unsigned int block_size = VoxelBlockType::size_li;
        const Eigen::Vector3i block_coord = block->coordinates();

        const Eigen::Vector3f block_centre_coord_f =
            se::get_sample_coord(block_coord, block_size, Eigen::Vector3f::Constant(0.5f));
        const Eigen::Vector3f block_centre_point_C =
            (T_CM_ * (voxel_dim_ * block_centre_coord_f).homogeneous()).head(3);

        // Convert block centre to measurement >> PinholeCamera -> .z() | OusterLidar -> .norm()
        const float block_point_C_m = sensor_.measurementFromPoint(block_centre_point_C);

        // Compute one tau and 3x sigma value for the block
        float tau = updating_model::computeTau(block_point_C_m);
        float three_sigma = updating_model::computeThreeSigma(block_point_C_m);

        /// Compute the integration scale
        // The last integration scale
        const int last_scale = block->current_scale();

        // The recommended integration scale
        const int computed_integration_scale = sensor_.computeIntegrationScale(
            block_centre_point_C, voxel_dim_, last_scale, block->min_scale(), map_.maxBlockScale());

        // The minimum integration scale (change to last if data has already been integrated)
        const int min_integration_scale =
            (low_variance
             && (block->min_scale() == -1
                 || block->maxValue() < 0.95 * MultiresOFusion::log_odd_min))
            ? MultiresOFusion::fs_integr_scale
            : std::max(0, last_scale - 1);
        const int max_integration_scale = (block->min_scale() == -1)
            ? max_block_scale_
            : std::min(max_block_scale_, last_scale + 1);

        // The final integration scale
        const int recommended_scale = std::min(
            std::max(min_integration_scale, computed_integration_scale), max_integration_scale);

        int integration_scale = last_scale;

        /// If no data has been integrated in the block before (block->min_scale() == -1), use the computed integration scale.
        if (block->min_scale() == -1) {
            // Make sure the block is allocated up to the integration scale
            integration_scale = recommended_scale;
            block->allocateDownTo(integration_scale);
            block->current_scale(integration_scale);
            block->initCurrCout();
            block->setInitData(MultiresOFusion::VoxelType::initData());
        }
        else if (recommended_scale != last_scale) { ///<< Potential double integration

            if (recommended_scale
                != block->buffer_scale()) { ///<< Start from scratch and initialise buffer

                block->initBuffer(recommended_scale);

                if (recommended_scale < last_scale) {
                    const int parent_scale = last_scale;
                    const unsigned int size_at_parent_scale_li = block->size_li >> parent_scale;
                    const unsigned int size_at_parent_scale_sq =
                        se::math::sq(size_at_parent_scale_li);

                    const unsigned int size_at_buffer_scale_li = size_at_parent_scale_li << 1;
                    const unsigned int size_at_buffer_scale_sq =
                        se::math::sq(size_at_buffer_scale_li);

                    for (unsigned int z = 0; z < size_at_parent_scale_li; z++) {
                        for (unsigned int y = 0; y < size_at_parent_scale_li; y++) {
#pragma omp simd
                            for (unsigned int x = 0; x < size_at_parent_scale_li; x++) {
                                const int parent_idx =
                                    x + y * size_at_parent_scale_li + z * size_at_parent_scale_sq;
                                const auto& parent_data =
                                    block->currData(parent_idx); // TODO: CAN BE MADE FASTER

                                for (unsigned int k = 0; k < 2; k++) {
                                    for (unsigned int j = 0; j < 2; j++) {
                                        for (unsigned int i = 0; i < 2; i++) {
                                            const int buffer_idx = (2 * x + i)
                                                + (2 * y + j) * size_at_buffer_scale_li
                                                + (2 * z + k) * size_at_buffer_scale_sq;
                                            auto& buffer_data = block->bufferData(
                                                buffer_idx); ///<< Fetch value from buffer.

                                            buffer_data.x = parent_data.x;
                                            buffer_data.fg = parent_data.fg;
                                            buffer_data.fg_count = parent_data.fg_count;
                                            buffer_data.y =
                                                parent_data.y; // (parent_data.y > 0) ? 1 : 0;
                                            buffer_data.r = parent_data.r;
                                            buffer_data.g = parent_data.g;
                                            buffer_data.b = parent_data.b;
                                            buffer_data.observed =
                                                false; ///<< Set falls such that the observe count can work properly
                                            buffer_data.frontier =
                                                parent_data
                                                    .frontier; // TODO SEM does this make sense?

                                        } // i
                                    }     // j
                                }         // k

                            } // x
                        }     // y
                    }         // z
                }
            }

            /// Integrate data into buffer.
            const unsigned int recommended_stride = 1 << recommended_scale;
            const unsigned int size_at_recommended_scale_li = block->size_li >> recommended_scale;
            const unsigned int size_at_recommended_scale_sq =
                se::math::sq(size_at_recommended_scale_li);

            const Eigen::Vector3i voxel_coord_base = block->coordinates();
            const Eigen::Vector3f voxel_sample_coord_base_f =
                se::get_sample_coord(voxel_coord_base, recommended_stride, sample_offset_frac_);
            const Eigen::Vector3f sample_point_base_C =
                (T_CM_ * (voxel_dim_ * voxel_sample_coord_base_f).homogeneous()).head(3);

            const Eigen::Matrix3f sample_point_delta_matrix_C =
                (se::math::to_rotation(T_CM_)
                 * (voxel_dim_
                    * (Eigen::Matrix3f() << recommended_stride,
                       0,
                       0,
                       0,
                       recommended_stride,
                       0,
                       0,
                       0,
                       recommended_stride)
                          .finished()));

            auto valid_predicate = [&](float depth_value,
                                       uint32_t /* rgba_value */,
                                       se::integration_mask_elem_t fg_value) {
                return depth_value >= sensor_.near_plane
                    && fg_value != se::InstanceSegmentation::skip_integration;
            };

            // TODO SEM maybe omp parallel for will speed this up
            for (unsigned int z = 0; z < size_at_recommended_scale_li; z++) {
                for (unsigned int y = 0; y < size_at_recommended_scale_li; y++) {
#pragma omp simd
                    for (unsigned int x = 0; x < size_at_recommended_scale_li; x++) {
                        const Eigen::Vector3f sample_point_C = sample_point_base_C
                            + sample_point_delta_matrix_C * Eigen::Vector3f(x, y, z);

                        // Fetch image value
                        float depth_value(0);
                        uint32_t rgba_value(0);
                        se::integration_mask_elem_t fg_value(0);
                        if (!sensor_.projectToPixelValue(sample_point_C,
                                                         depth_image_,
                                                         depth_value,
                                                         rgba_image_,
                                                         rgba_value,
                                                         fg_image_,
                                                         fg_value,
                                                         valid_predicate)) {
                            continue;
                        }

                        const int buffer_idx =
                            x + y * size_at_recommended_scale_li + z * size_at_recommended_scale_sq;
                        auto& buffer_data =
                            block->bufferData(buffer_idx); /// \note pass by reference now.

                        if (low_variance) {
                            block->incrBufferObservedCount(updating_model::freeVoxel(buffer_data));
                        }
                        else {
                            const float sample_point_C_m =
                                sensor_.measurementFromPoint(sample_point_C);
                            const float range = sample_point_C.norm();
                            const float range_diff =
                                (sample_point_C_m - depth_value) * (range / sample_point_C_m);
                            block->incrBufferObservedCount(updating_model::updateVoxel(
                                range_diff, tau, three_sigma, rgba_value, fg_value, buffer_data));
                        }
                    } // x
                }     // y
            }         // z

            block->incrBufferIntegrCount(project_inside);

            if (block->switchData()) {
                return;
            }
        }
        else {
            block->resetBuffer();
        }

        const unsigned int integration_stride = 1 << integration_scale;
        const unsigned int size_at_integration_scale_li = block->size_li >> integration_scale;
        const unsigned int size_at_integration_scale_sq =
            se::math::sq(size_at_integration_scale_li);

        const Eigen::Vector3i voxel_coord_base = block->coordinates();
        const Eigen::Vector3f voxel_sample_coord_base_f =
            se::get_sample_coord(voxel_coord_base, integration_stride, sample_offset_frac_);
        const Eigen::Vector3f sample_point_base_C =
            (T_CM_ * (voxel_dim_ * voxel_sample_coord_base_f).homogeneous()).head(3);
        block->minDistUpdated(sensor_.measurementFromPoint(sample_point_base_C));

        const Eigen::Matrix3f sample_point_delta_matrix_C =
            (se::math::to_rotation(T_CM_)
             * (voxel_dim_
                * (Eigen::Matrix3f() << integration_stride,
                   0,
                   0,
                   0,
                   integration_stride,
                   0,
                   0,
                   0,
                   integration_stride)
                      .finished()));

        auto valid_predicate = [&](float depth_value,
                                   uint32_t /* rgba_value */,
                                   se::integration_mask_elem_t fg_value) {
            return depth_value >= sensor_.near_plane && 0.0f <= fg_value && fg_value <= 1.0f;
        };

        // TODO SEM maybe omp parallel for will speed this up
        for (unsigned int z = 0; z < size_at_integration_scale_li; z++) {
            for (unsigned int y = 0; y < size_at_integration_scale_li; y++) {
                //#pragma omp simd // Can't have omp critical in omp simd
                for (unsigned int x = 0; x < size_at_integration_scale_li; x++) {
                    const Eigen::Vector3f sample_point_C = sample_point_base_C
                        + sample_point_delta_matrix_C * Eigen::Vector3f(x, y, z);

                    // Fetch image value
                    float depth_value(0);
                    uint32_t rgba_value(0);
                    se::integration_mask_elem_t fg_value(0);
                    if (!sensor_.projectToPixelValue(sample_point_C,
                                                     depth_image_,
                                                     depth_value,
                                                     rgba_image_,
                                                     rgba_value,
                                                     fg_image_,
                                                     fg_value,
                                                     valid_predicate)) {
                        continue;
                    }

                    // Update the voxel data based using the depth measurement
                    const int voxel_idx =
                        x + y * size_at_integration_scale_li + z * size_at_integration_scale_sq;
                    auto& voxel_data = block->currData(voxel_idx); /// \note pass by reference now.
                    if (low_variance) {
                        block->incrCurrObservedCount(updating_model::freeVoxel(voxel_data));
                    }
                    else {
                        const float sample_point_C_m = sensor_.measurementFromPoint(sample_point_C);
                        const float range = sample_point_C.norm();
                        const float range_diff =
                            (sample_point_C_m - depth_value) * (range / sample_point_C_m);
                        block->incrCurrObservedCount(updating_model::updateVoxel(
                            range_diff, tau, three_sigma, rgba_value, fg_value, voxel_data));
                    }
                } // x
            }     // y
        }         // z

        block->incrCurrIntegrCount();
    }



    /**
   * \brief Recursively reduce all children by the minimum occupancy log-odd for a single integration.
   */
    void freeNodeRecurse(NodeType* node, int depth)
    {
        for (int child_idx = 0; child_idx < 8; child_idx++) {
            auto child = node->child(child_idx);
            if (child) {
                if (child->isBlock()) {
                    VoxelBlockType* block = dynamic_cast<VoxelBlockType*>(child);
                    // Voxel block has a low variance. Update data at a minimum
                    // free space integration scale or finer/coarser (depending on later scale selection).
#pragma omp critical(voxel_lock)
                    { // Add voxel block to voxel block list for later update and up-propagation
                        block_list_.push_back(dynamic_cast<VoxelBlockType*>(block));
                        free_list_.push_back(true);
                        low_variance_list_.push_back(true);
                        projects_inside_list_.push_back(true);
                    }
                }
                else {
                    freeNodeRecurse(child, depth + 1);
                }
            }
            else {
                updating_model::freeNode(node->childData(child_idx));
#pragma omp critical(node_lock)
                { // Add node to node list for later up-propagation (finest node for this tree-branch)
                    node_list_[depth].insert(node);
                }
            }
        }
    }



    /**
   * \brief Get reference to a child value for a given parent node.
   *
   * \param[in] parent   The pointer to the parent node.
   * \param[in] rel_step The child's relative position with respect to the parent.
   *
   * \return Reference to child's value.
   */
    VoxelData& getChildValue(NodeType* parent, const Eigen::Vector3i& rel_step)
    {
        assert(!parent->isBlock() && "Father node must not be a leaf");

        return parent->childData(rel_step.x() + rel_step.y() * 2 + rel_step.z() * 4);
    }



    /**
   * \brief Allocate a node for a given parent node.
   *
   * \param[in] parent      The pointer to the parent node.
   * \param[in] node_coord  The coordinates of the node in voxel coordinates.
   * \param[in] rel_step    The node relative direction with respect to the parent.
   * \param[in] node_size   The size of the node in voxel units.
   * \param[in] depth       The depth of the node in the octree.
   *
   * \return The pointer to the allocated node.
   */
    auto allocateNode(NodeType* parent,
                      const Eigen::Vector3i& node_coord,
                      const Eigen::Vector3i& rel_step,
                      const unsigned int node_size,
                      const unsigned int depth) -> NodeType*
    {
        const unsigned int node_idx = rel_step.x() + rel_step.y() * 2 + rel_step.z() * 4;
        NodeType*& node = parent->child(node_idx);
        if (node) {
            /// Return node if already allocated.
            return node;
        }

        /// Otherwise allocate and initalise child.

        // Update child mask
        parent->children_mask(parent->children_mask() | (1 << node_idx));

        if (node_size <= VoxelBlockType::size_li) {
            // Allocate voxel block
#pragma omp critical(voxel_alloc_lock)
            {
                auto init_data = parent->childData(node_idx); // Initalise child with parent value
                // Initialise remaining values that are not kept track of at node depth
                node = map_.pool().acquireBlock(init_data);
                VoxelBlockType* block = dynamic_cast<VoxelBlockType*>(node);
                block->coordinates(node_coord);
                block->code(se::keyops::encode(node_coord.x(),
                                               node_coord.y(),
                                               node_coord.z(),
                                               map_.blockDepth(),
                                               voxel_depth_));
                block->parent() = parent;
            }
        }
        else {
            // Allocate node
#pragma omp critical(node_alloc_lock)
            {
                node = map_.pool().acquireNode(parent->childData(node_idx));
                node->size(node_size);
                node->code(se::keyops::encode(
                    node_coord.x(), node_coord.y(), node_coord.z(), depth, voxel_depth_));
                node->parent() = parent;
                // Initalise all children with parent value if it differs from the default value
                if (node->parent()->childData(node_idx).x
                    != MultiresOFusion::VoxelType::initData().x) { //TODO VERIFY
                    for (int child_idx = 0; child_idx < 8; child_idx++) {
                        node->childData(child_idx, node->parent()->childData(node_idx));
                    }
                }
            }
        }

        assert(node && "Voxel alloc failure");
        return node;
    }



    /**
   * \brief Verify if the camera is inside a given node.
   *
   * \param[in] node_coord  The coordinates of the node in voxel coordinates.
   * \param[in] node_size   The size of the node in voxel units.
   * \param[in] T_MC        The transformation from camera to map frame.
   *
   * \return True/false statement if the camera is inside the node.
   */
    bool cameraInNode(const Eigen::Vector3i& node_coord,
                      const int node_size,
                      const Eigen::Matrix4f& T_MC)
    {
        const Eigen::Vector3f voxel_coord = se::math::to_translation(T_MC) / voxel_dim_;
        if (voxel_coord.x() >= node_coord.x() && voxel_coord.x() <= node_coord.x() + node_size
            && voxel_coord.y() >= node_coord.y() && voxel_coord.y() <= node_coord.y() + node_size
            && voxel_coord.z() >= node_coord.z() && voxel_coord.z() <= node_coord.z() + node_size) {
            return true;
        }
        return false;
    }



    /**
   * \brief Verify if the node crosses the camera frustum excluding the case of the camera in the node.
   *
   * \param[in] proj_corner_stati The stati of the projection of the eight octant corners into the image plane
   *
   * \return True/false statement if node crosses the camera frustum.
   */
    bool crossesFrustum(std::vector<srl::projection::ProjectionStatus>& proj_corner_stati)
    {
        for (int corner_idx = 0; corner_idx < 8; corner_idx++) {
            if (proj_corner_stati[corner_idx] == srl::projection::ProjectionStatus::Successful) {
                return true;
            }
        }
        return false;
    }



    /**
   * \brief Update and allocate a given node and all its children recursively.
   *
   * \param[in] node_coord  The corner of the node to be processed (bottom, left, front coordinates).
   * \param[in] node_size   The size in [voxel] of the node to be processed.
   * \param[in] depth       The depth of the node to be processed (root = 0).
   * \param[in] rel_step    The relative direction of node within parent node (e.g. [1, 0, 1], [0, 1, 1]).
   * \param[in] parent      The pointer to the nodes parent.
   */
    void operator()(const Eigen::Vector3i& /* node_coord */,
                    const int /* node_size */,
                    const int /* depth */,
                    const Eigen::Vector3i& /* rel_step */,
                    NodeType* /* parent */)
    {
    }

    /**
   * \brief Propage all newly integrated values from the voxel block depth up to the root of the octree.
   */
    void propagateToRoot()
    {
        for (const auto& block : block_list_) {
            if (block->parent()) {
                node_list_[map_.blockDepth() - 1].insert(block->parent());
                const unsigned int child_idx = se::child_idx(
                    block->code(), se::keyops::depth(block->code()), map_.voxelDepth());
                auto max_data = block->maxData();
                auto& parent_data = block->parent()->childData(child_idx);
                parent_data = max_data;

                if (max_data.observed && !max_data.frontier
                    && max_data.x * max_data.y <= 0.95 * MultiresOFusion::min_occupancy) {
                    pool_.deleteBlock(block, voxel_depth_);
                }

                // TODO: ^SWITCH 1 - Alternative approach (conservative)
                // Delete block if it's in free space and it's max value already surpassed a lower threshold.
                // Approach only saves time.
                //        if (   data.observed
                //            && data.x <= 0.95 * MultiresOFusion::log_odd_min
                //            && data.y > MultiresOFusion::max_weight / 2) {
                //          pool_.deleteBlock(block, voxel_depth_);
                //        }
            }
        }

        for (int d = map_.blockDepth() - 1; d > 0; d--) {
            std::set<NodeType*>::iterator it;
            for (it = node_list_[d].begin(); it != node_list_[d].end(); ++it) {
                NodeType* node = *it;
                if (node->timestamp() == frame_) {
                    continue;
                }
                if (node->parent()) {
                    auto node_data =
                        updating_model::propagateToNoteAtCoarserScale(node, voxel_depth_, frame_);
                    node_list_[d - 1].insert(node->parent());

                    if (node_data.observed && !node_data.frontier
                        && node_data.x * node_data.y <= 0.95 * MultiresOFusion::min_occupancy) {
                        pool_.deleteNode(node, voxel_depth_);
                    }

                    // TODO: ^SWITCH 1 - Alternative approach (conservative)
                    // Delete node if it's in free space and it's max value already surpassed a lower threshold.
                    // Approach only saves time.
                    //          if (   node_data.observed
                    //              && node_data.x <= 0.95 * MultiresOFusion::log_odd_min
                    //              && node_data.y > MultiresOFusion::max_weight / 2) {
                    //            pool_.deleteNode(node, voxel_depth_);
                    //          }

                } // if parent
            }     // nodes at depth d
        }         // depth d
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};



template<typename SensorImplType>
constexpr int MultiresOFusionUpdate<SensorImplType>::max_block_scale_;



template<>
void MultiresOFusionUpdate<se::PinholeCamera>::operator()(const Eigen::Vector3i& node_coord,
                                                          const int node_size,
                                                          const int depth,
                                                          const Eigen::Vector3i& rel_step,
                                                          NodeType* parent)
{
    /// Approximate max and min depth to quickly check if the node is behind the camera or maximum depth.
    // Compute the node centre's depth in the camera frame
    const Eigen::Vector3f node_centre_point_M = voxel_dim_
        * (node_coord.cast<float>() + Eigen::Vector3f(node_size, node_size, node_size) / 2.f);
    const Eigen::Vector3f node_centre_point_C = (T_CM_ * node_centre_point_M.homogeneous()).head(3);

    // Extend and reduce the depth by the sphere radius covering the entire cube
    const float approx_depth_value_max =
        node_centre_point_C.z() + node_size * size_to_radius_ * voxel_dim_;
    const float approx_depth_value_min =
        node_centre_point_C.z() - node_size * size_to_radius_ * voxel_dim_;

    /// CASE 0.1 (OUT OF BOUNDS): Block is behind the camera or behind the maximum depth value
    if (approx_depth_value_min > max_depth_value_
        || approx_depth_value_max < zero_depth_band_) { // TODO: Alternative sensor_.near_plane.

        return;
    }

    // Compute the 8 corners of the node to be evaluated
    Eigen::Matrix<float, 3, 8> node_corner_coords_f =
        (node_size * corner_rel_steps_).colwise() + node_coord.cast<float>();
    Eigen::Matrix<float, 3, 8> node_corner_points_C =
        (T_CM_ * Eigen::Vector4f(voxel_dim_, voxel_dim_, voxel_dim_, 1.f).asDiagonal()
         * node_corner_coords_f.colwise().homogeneous())
            .topRows(3);

    Eigen::VectorXi node_corners_infront(8);
    node_corners_infront << 1, 1, 1, 1, 1, 1, 1, 1;
    for (int corner_idx = 0; corner_idx < 8; corner_idx++) {
        if (node_corner_points_C(2, corner_idx) < zero_depth_band_) {
            node_corners_infront(corner_idx) = 0;
        }
    }

    int num_node_corners_infront = node_corners_infront.sum();

    /// CASE 0.2 (OUT OF BOUNDS): Node is behind the camera.
    if (num_node_corners_infront == 0) {
        return;
    }

    // Project the 8 corners into the image plane
    Eigen::Matrix2Xf proj_node_corner_pixels_f(2, 8);
    const Eigen::VectorXf node_corners_diff = node_corner_points_C.row(2);
    std::vector<srl::projection::ProjectionStatus> proj_node_corner_stati;
    sensor_.model.projectBatch(
        node_corner_points_C, &proj_node_corner_pixels_f, &proj_node_corner_stati);

    /// Approximate a 2D bounding box covering the projected node in the image plane.
    bool should_split = false;
    bool projects_inside = false;
    int low_variance =
        0; ///<< -1 := low variance infront of the surface, 0 := high variance, 1 = low_variance behind the surface.
    se::Pixel pooling_pixel = se::Pixel::
        crossingUnknownPixel(); ///<< min, max pixel batch depth + crossing frustum state + contains unknown values state.

    if (depth < map_.blockDepth() + 1) {
        if (num_node_corners_infront < 8) {
            /// CASE 1 (CAMERA IN NODE):
            if (cameraInNode(node_coord, node_size, se::math::to_inverse_transformation(T_CM_))) {
                should_split = true;

                /// CASE 2 (FRUSTUM BOUNDARY): Node partly behind the camera and crosses the the frustum boundary
            }
            else if (crossesFrustum(proj_node_corner_stati)) {
                should_split = true;

                /// CASE 2 (FRUSTUM BOUNDARY): Node partly behind the camera and crosses the the frustum boundary without a corner reprojecting
            }
            else if (sensor_.sphereInFrustumInf(node_centre_point_C,
                                                node_size * size_to_radius_ * voxel_dim_)) {
                should_split = true;
            }
            else {
                return;
            }
        }
        else {
            // Compute the minimum and maximum pixel values to generate the bounding box
            const Eigen::Vector2i image_bb_min =
                proj_node_corner_pixels_f.rowwise().minCoeff().cast<int>();
            const Eigen::Vector2i image_bb_max =
                proj_node_corner_pixels_f.rowwise().maxCoeff().cast<int>();
            const float node_dist_min_m = node_corners_diff.minCoeff();
            const float node_dist_max_m = node_corners_diff.maxCoeff();

            pooling_pixel = pooling_depth_image_->conservativeQuery(image_bb_min, image_bb_max);

            /// CASE 0.3 (OUT OF BOUNDS): The node is outside frustum (i.e left, right, below, above) or
            ///                           all pixel values are unknown -> return intermediately
            if (pooling_pixel.status_known == se::Pixel::statusKnown::unknown) {
                return;
            }

            /// CASE 0.4 (OUT OF BOUNDS): The node is behind surface
            if (node_dist_min_m > pooling_pixel.max
                    + MultiresOFusion::tau_max) { // TODO: Can be changed to node_dist_max_m?

                return;
            }

            low_variance = updating_model::lowVariance(
                pooling_pixel.min, pooling_pixel.max, node_dist_min_m, node_dist_max_m);

            const se::key_t node_key = se::keyops::encode(
                node_coord.x(), node_coord.y(), node_coord.z(), depth, voxel_depth_);
            const unsigned int child_idx = se::child_idx(node_key, depth, map_.voxelDepth());

            /// CASE 1 (REDUNDANT DATA): Depth values in the bounding box are far away from the node or unknown (1).
            ///                          The node to be evaluated is free (2) and fully observed (3),
            if (low_variance != 0 && parent->childData(child_idx).observed
                && parent->childData(child_idx).x * parent->childData(child_idx).y
                    <= 0.95 * MultiresOFusion::min_occupancy) {
                return;
            }

            // TODO: ^SWITCH 1 - Alternative approach (conservative)
            // Don't free node even more under given conditions.
            //        if (   low_variance != 0
            //            && parent->childData(child_idx).observed
            //            && parent->childData(child_idx).x <= 0.95 * MultiresOFusion::log_odd_min
            //            && parent->childData(child_idx).y > MultiresOFusion::max_weight / 2) {
            //          return;
            //        }

            /// CASE 2 (FRUSTUM BOUNDARY): The node is crossing the frustum boundary
            if (pooling_pixel.status_crossing == se::Pixel::statusCrossing::crossing) {
                should_split = true;
            }

            /// CASE 3: The node is inside the frustum, but projects into partially known pixel
            else if (pooling_pixel.status_known == se::Pixel::statusKnown::part_known) {
                // TODO: SWITCH 1 - Alternative approach (conservative)
                // If the entire node is already free don't bother splitting it to free only parts of it even more because
                // of missing data and just move on.
                // Approach only saves time.
                //          if (   low_variance != 0
                //              && parent->childData(child_idx).observed
                //              && parent->childData(child_idx).x <= 0.95 * MultiresOFusion::log_odd_min
                //              && parent->childData(child_idx).y > MultiresOFusion::max_weight / 2) {
                //            return;
                //          }

                should_split = true;
            }

            /// CASE 4: The node is inside the frustum with only known data + node has a potential high variance
            else if (low_variance == 0) {
                should_split = true;
            }

            projects_inside = pooling_pixel.status_known == se::Pixel::known;
        }
    }

    if (should_split) {
        // Returns a pointer to the according node if it has previously been allocated.
        NodeType* node = allocateNode(parent, node_coord, rel_step, node_size, depth);
        if (node->isBlock()) { // Evaluate the node directly if it is a voxel block
            node->active(true);
            // Cast from node to voxel block
#pragma omp critical(voxel_lock)
            { // Add voxel block to voxel block list for later update and up-propagation
                block_list_.push_back(dynamic_cast<VoxelBlockType*>(node));
                free_list_.push_back(false);
                low_variance_list_.push_back((low_variance == -1));
                projects_inside_list_.push_back(projects_inside);
            }
        }
        else {
            // Split! Start recursive process
#pragma omp parallel for
            for (int child_idx = 0; child_idx < 8; ++child_idx) {
                int child_size = node_size / 2;
                const Eigen::Vector3i child_rel_step =
                    Eigen::Vector3i((child_idx & 1) > 0, (child_idx & 2) > 0, (child_idx & 4) > 0);
                const Eigen::Vector3i child_coord = node_coord + child_rel_step * child_size;
                (*this)(child_coord, child_size, depth + 1, child_rel_step, node);
            }
        }
    }
    else {
        assert(depth);
        int node_idx = rel_step.x() + rel_step.y() * 2 + rel_step.z() * 4;
        NodeType* node = parent->child(node_idx);
        if (!node) {
            // Node does not exist -> Does NOT have children that need to be updated
            if (low_variance == -1) {
                // Free node
                VoxelData& node_value = getChildValue(parent, rel_step);
                updating_model::freeNode(node_value);
#pragma omp critical(node_lock)
                { // Add node to node list for later up propagation (finest node for this branch)
                    node_list_[depth - 1].insert(parent);
                }
            } // else node has low variance behind surface (ignore)
        }
        else {
            // Node does exist -> Does POTENTIALLY have children that that need to be updated
            if (node->isBlock()) {
                // Node is a voxel block -> Does NOT have children that need to be updated
#pragma omp critical(voxel_lock)
                { // Add voxel block to voxel block list for later update and up-propagation
                    block_list_.push_back(dynamic_cast<VoxelBlockType*>(node));
                    free_list_.push_back(false);
                    low_variance_list_.push_back((low_variance == -1));
                    projects_inside_list_.push_back(projects_inside);
                }
            }
            else {
                // Node has children
                if (low_variance == -1) {
                    //Free node recursively
                    freeNodeRecurse(node, depth);
                } // else node has low variance behind surface (ignore)
            }
        }
    }
}

template<>
void MultiresOFusionUpdate<se::OusterLidar>::operator()(const Eigen::Vector3i& node_coord,
                                                        const int node_size,
                                                        const int depth,
                                                        const Eigen::Vector3i& rel_step,
                                                        NodeType* parent)
{
    /// Approximate max and min depth to quickly check if the node is behind the camera or maximum depth.
    // Compute the node centre's depth in the camera frame
    const Eigen::Vector3f node_centre_point_M = voxel_dim_
        * (node_coord.cast<float>() + Eigen::Vector3f(node_size, node_size, node_size) / 2.f);
    const Eigen::Vector3f node_centre_point_C = (T_CM_ * node_centre_point_M.homogeneous()).head(3);

    // Extend and reduce the depth by the sphere radius covering the entire cube
    const float approx_depth_value_min =
        node_centre_point_C.norm() - node_size * size_to_radius_ * voxel_dim_;
    const float approx_depth_value_max =
        node_centre_point_C.norm() + node_size * size_to_radius_ * voxel_dim_;

    /// CASE 0.1 (OUT OF BOUNDS): Block is behind the camera or behind the maximum depth value
    if (approx_depth_value_min > max_depth_value_) {
        return;
    }

    /// Approximate a 2D bounding box covering the projected node in the image plane.
    bool should_split = false;
    bool projects_inside = false;
    int low_variance =
        0; ///<< -1 := low variance infront of the surface, 0 := high variance, 1 = low_variance behind the surface.
    se::Pixel pooling_pixel = se::Pixel::
        crossingUnknownPixel(); ///<< min, max pixel batch depth + crossing frustum state + contains unknown values state.

    if (depth < map_.blockDepth() + 1) {
        if (cameraInNode(node_coord, node_size, se::math::to_inverse_transformation(T_CM_))) {
            should_split = true;
        }
        else {
            // Compute the 8 corners of the node to be evaluated
            Eigen::Matrix<float, 3, 8> node_corner_coords_f =
                (node_size * corner_rel_steps_).colwise() + node_coord.cast<float>();
            Eigen::Matrix<float, 3, 8> node_corner_points_C =
                (T_CM_ * Eigen::Vector4f(voxel_dim_, voxel_dim_, voxel_dim_, 1.f).asDiagonal()
                 * node_corner_coords_f.colwise().homogeneous())
                    .topRows(3);

            // Project the 8 corners into the image plane
            Eigen::Matrix2Xf proj_node_corner_pixels_f(2, 8);
            std::vector<srl::projection::ProjectionStatus> proj_node_corner_stati;
            sensor_.model.projectBatch(
                node_corner_points_C, &proj_node_corner_pixels_f, &proj_node_corner_stati);

            Eigen::Vector2f proj_node_centre_pixel_f;
            sensor_.model.project(node_centre_point_C, &proj_node_centre_pixel_f);

            float proj_node_centre_pixel_u_f = proj_node_centre_pixel_f.x();
            float proj_node_centre_pixel_u_comp_f =
                proj_node_centre_pixel_u_f + depth_image_.width() / 2;

            if (proj_node_centre_pixel_u_comp_f > depth_image_.width() - 0.5) {
                proj_node_centre_pixel_u_comp_f =
                    proj_node_centre_pixel_u_comp_f - depth_image_.width();
            }

            Eigen::VectorXf proj_node_corner_pixels_u_f(8);
            for (int i = 0; i < 8; i++) {
                if (proj_node_centre_pixel_u_f < proj_node_centre_pixel_u_comp_f
                    && proj_node_corner_pixels_f(0, i) > proj_node_centre_pixel_u_f
                    && proj_node_corner_pixels_f(0, i) > proj_node_centre_pixel_u_comp_f) {
                    proj_node_corner_pixels_u_f(i) =
                        proj_node_corner_pixels_f(0, i) - depth_image_.width();
                }
                else if (proj_node_centre_pixel_u_f > proj_node_centre_pixel_u_comp_f
                         && proj_node_corner_pixels_f(0, i) < proj_node_centre_pixel_u_f
                         && proj_node_corner_pixels_f(0, i) < proj_node_centre_pixel_u_comp_f) {
                    proj_node_corner_pixels_u_f(i) =
                        proj_node_corner_pixels_f(0, i) + depth_image_.width();
                }
                else {
                    proj_node_corner_pixels_u_f(i) = proj_node_corner_pixels_f(0, i);
                }
            }
            Eigen::VectorXf proj_node_corner_pixels_v_f = proj_node_corner_pixels_f.row(1);

            int u_min = proj_node_corner_pixels_u_f.minCoeff();
            int u_max = proj_node_corner_pixels_u_f.maxCoeff();
            int v_min = proj_node_corner_pixels_v_f.minCoeff();
            int v_max = proj_node_corner_pixels_v_f.maxCoeff();

            // Compute the minimum and maximum pixel values to generate the bounding box
            const Eigen::Vector2i image_bb_min(u_min, v_min);
            const Eigen::Vector2i image_bb_max(u_max, v_max);

            pooling_pixel = pooling_depth_image_->conservativeQuery(image_bb_min, image_bb_max);

            /// CASE 0.3 (OUT OF BOUNDS): The node is outside frustum (i.e left, right, below, above) or
            ///                           all pixel values are unknown -> return intermediately
            if (pooling_pixel.status_known == se::Pixel::statusKnown::unknown) {
                return;
            }

            /// CASE 0.4 (OUT OF BOUNDS): The node is behind surface
            if (approx_depth_value_min > pooling_pixel.max
                    + MultiresOFusion::tau_max) { // TODO: Can be changed to node_dist_max_m?

                return;
            }

            low_variance = updating_model::lowVariance(pooling_pixel.min,
                                                       pooling_pixel.max,
                                                       approx_depth_value_min,
                                                       approx_depth_value_max);

            const se::key_t node_key = se::keyops::encode(
                node_coord.x(), node_coord.y(), node_coord.z(), depth, voxel_depth_);
            const unsigned int child_idx = se::child_idx(node_key, depth, map_.voxelDepth());

            /// CASE 1 (REDUNDANT DATA): Depth values in the bounding box are far away from the node or unknown (1).
            ///                          The node to be evaluated is free (2) and fully observed (3),
            if (low_variance != 0 && parent->childData(child_idx).observed
                && parent->childData(child_idx).x * parent->childData(child_idx).y
                    <= 0.95 * MultiresOFusion::min_occupancy) {
                return;
            }

            // TODO: ^SWITCH 1 - Alternative approach (conservative)
            // Don't free node even more under given conditions.
            //        if (   low_variance != 0
            //            && parent->childData(child_idx).observed
            //            && parent->childData(child_idx).x <= 0.95 * MultiresOFusion::log_odd_min
            //            && parent->childData(child_idx).y > MultiresOFusion::max_weight / 2) {
            //          return;
            //        }

            /// CASE 2 (FRUSTUM BOUNDARY): The node is crossing the frustum boundary
            if (pooling_pixel.status_crossing == se::Pixel::statusCrossing::crossing) {
                should_split = true;
            }

            /// CASE 3: The node is inside the frustum, but projects into partially known pixel
            else if (pooling_pixel.status_known == se::Pixel::statusKnown::part_known) {
                // TODO: SWITCH 1 - Alternative approach (conservative)
                // If the entire node is already free don't bother splitting it to free only parts of it even more because
                // of missing data and just move on.
                // Approach only saves time.
                //          if (   low_variance != 0
                //              && parent->childData(child_idx).observed
                //              && parent->childData(child_idx).x <= 0.95 * MultiresOFusion::log_odd_min
                //              && parent->childData(child_idx).y > MultiresOFusion::max_weight / 2) {
                //            return;
                //          }

                should_split = true;
            }

            /// CASE 4: The node is inside the frustum with only known data + node has a potential high variance
            else if (low_variance == 0) {
                should_split = true;
            }

            projects_inside = pooling_pixel.status_known == se::Pixel::known;
        }
    }

    if (should_split) {
        // Returns a pointer to the according node if it has previously been allocated.
        NodeType* node = allocateNode(parent, node_coord, rel_step, node_size, depth);
        if (node->isBlock()) { // Evaluate the node directly if it is a voxel block
            node->active(true);
            // Cast from node to voxel block
#pragma omp critical(voxel_lock)
            { // Add voxel block to voxel block list for later update and up-propagation
                block_list_.push_back(dynamic_cast<VoxelBlockType*>(node));
                free_list_.push_back(false);
                low_variance_list_.push_back((low_variance == -1));
                projects_inside_list_.push_back(projects_inside);
            }
        }
        else {
            // Split! Start recursive process
#pragma omp parallel for
            for (int child_idx = 0; child_idx < 8; ++child_idx) {
                int child_size = node_size / 2;
                const Eigen::Vector3i child_rel_step =
                    Eigen::Vector3i((child_idx & 1) > 0, (child_idx & 2) > 0, (child_idx & 4) > 0);
                const Eigen::Vector3i child_coord = node_coord + child_rel_step * child_size;
                (*this)(child_coord, child_size, depth + 1, child_rel_step, node);
            }
        }
    }
    else {
        assert(depth);
        int node_idx = rel_step.x() + rel_step.y() * 2 + rel_step.z() * 4;
        NodeType* node = parent->child(node_idx);
        if (!node) {
            // Node does not exist -> Does NOT have children that need to be updated
            if (low_variance == -1) {
                // Free node
                VoxelData& node_value = getChildValue(parent, rel_step);
                updating_model::freeNode(node_value);
#pragma omp critical(node_lock)
                { // Add node to node list for later up propagation (finest node for this branch)
                    node_list_[depth - 1].insert(parent);
                }
            } // else node has low variance behind surface (ignore)
        }
        else {
            // Node does exist -> Does POTENTIALLY have children that that need to be updated
            if (node->isBlock()) {
                // Node is a voxel block -> Does NOT have children that need to be updated
#pragma omp critical(voxel_lock)
                { // Add voxel block to voxel block list for later update and up-propagation
                    block_list_.push_back(dynamic_cast<VoxelBlockType*>(node));
                    free_list_.push_back(false);
                    low_variance_list_.push_back((low_variance == -1));
                    projects_inside_list_.push_back(projects_inside);
                }
            }
            else {
                // Node has children
                if (low_variance == -1) {
                    //Free node recursively
                    freeNodeRecurse(node, depth);
                } // else node has low variance behind surface (ignore)
            }
        }
    }
}

/**
 * \brief Update and allocate all nodes and voxel blocks in the camera frustum using a map-to-camera integration scheme.
 * Starting from the eight roots children each node is projected into the image plane and an appropriate allocation and
 * updating scale is choosen depending on the variation of occupancy log-odds within each node/voxel block
 */
void MultiresOFusion::integrate(OctreeType& map,
                                const se::Image<float>& depth_image,
                                const se::Image<uint32_t>& rgba_image,
                                const cv::Mat& fg_image,
                                const Eigen::Matrix4f& T_CM,
                                const SensorImpl& sensor,
                                const unsigned frame,
                                std::set<se::key_t>* updated_nodes)
{
    TICKD("updateMap")
    // Create min/map depth pooling image for different bounding box sizes
    const std::unique_ptr<se::DensePoolingImage<SensorImpl>> pooling_depth_image(
        new se::DensePoolingImage<SensorImpl>(depth_image));

    TICKD("mapToCameraAllocation")
    const float max_depth_value =
        std::min(sensor.far_plane, pooling_depth_image->maxValue() + MultiresOFusion::tau_max);
    const float voxel_dim = map.dim() / map.size();

    std::vector<VoxelBlockType*> block_list; ///< List of blocks to be updated.
    std::vector<bool> free_list;             ///< Should updated block be freed? <bool>
    std::vector<bool> low_variance_list;     ///< Has updated block low variance? <bool>
    std::vector<bool>
        projects_inside_list; ///< Does the updated block reproject completely into the image? <bool>
    std::vector<std::set<NodeType*>> node_list(map.blockDepth());
    MultiresOFusionUpdate<SensorImpl> funct(map,
                                            block_list,
                                            node_list,
                                            free_list,
                                            low_variance_list,
                                            projects_inside_list,
                                            depth_image,
                                            rgba_image,
                                            fg_image,
                                            pooling_depth_image.get(),
                                            sensor,
                                            T_CM,
                                            voxel_dim,
                                            map.voxelDepth(),
                                            max_depth_value,
                                            frame);

    // Launch on the 8 voxels of the first depth
#pragma omp parallel for
    for (int child_idx = 0; child_idx < 8; ++child_idx) {
        int child_size = map.size() / 2;
        Eigen::Vector3i child_rel_step =
            Eigen::Vector3i((child_idx & 1) > 0, (child_idx & 2) > 0, (child_idx & 4) > 0);
        Eigen::Vector3i child_coord =
            child_rel_step * child_size; // Because, + corner is (0, 0, 0) at root depth
        funct(child_coord, child_size, 1, child_rel_step, map.root());
    }

    TOCK("mapToCameraAllocation")

    // Everything in the block_list is a candidate frontier.
    if (updated_nodes) {
        for (const auto block : block_list) {
            updated_nodes->insert(block->code());
        }
    }

    TICKD("updateBlock")
#pragma omp parallel for
    for (unsigned int i = 0; i < block_list.size(); ++i) {
        if (free_list[i]) {
            funct.freeBlock(block_list[i]);
        }
        else {
            funct.updateBlock(block_list[i], low_variance_list[i], projects_inside_list[i]);
        }
    }
    TOCK("updateBlock")
    TOCK("updateMap")

    /// Propagation
    TICKD("propagation")

    TICKD("propagationInBlock")
#pragma omp parallel for
    for (unsigned int i = 0; i < block_list.size(); ++i) {
        updating_model::propagateBlockToCoarsestScale(block_list[i]);
    }
    TOCK("propagationInBlock")

    TICKD("propagationInNode")
    funct.propagateToRoot();
    TOCK("propagationInNode")

    TOCK("propagation")
}



void MultiresOFusion::propagateToRoot(OctreeType& map)
{
    const int voxel_depth = map.voxelDepth();
    const int block_depth = map.blockDepth();
    std::vector<VoxelBlockType*> block_list;
    map.getBlockList(block_list, false);
    // Propagate the VoxelBlock data to their coarsest scale
#pragma omp parallel for
    for (size_t i = 0; i < block_list.size(); ++i) {
        updating_model::propagateBlockToCoarsestScale(block_list[i]);
    }

    // Propagate the VoxelBlock data up to their parent nodes and keep track of the parent nodes
    std::vector<std::set<NodeType*>> node_list(block_depth);
    for (const auto block : block_list) {
        if (block->parent()) {
            // Keep track of the VoxelBlock's parent Node for later use
            node_list[block_depth - 1].insert(block->parent());
            // Get the VoxelBlock's max-aggregated data
            const auto& block_max_data = block->maxData();
            // Propagate the VoxelBlock's data to its parent Node
            const unsigned child_idx = se::child_idx(block->code(), block_depth, voxel_depth);
            block->parent()->childData(child_idx) = block_max_data;
            // Prune the VoxelBlock if it's all free
            if (block_max_data.observed
                && block_max_data.x * block_max_data.y <= 0.95 * MultiresOFusion::min_occupancy) {
                map.pool().deleteBlock(block, voxel_depth);
            }
        }
        else {
            // Delete orphan VoxelBlocks
            map.pool().deleteBlock(block, voxel_depth);
        }
    }

    // Propagate the data to all the nodes up to the root
    for (int d = block_depth - 1; d > 0; d--) {
        for (NodeType* node : node_list[d]) {
            if (node->timestamp() == 1) {
                continue;
            }
            if (node->parent()) {
                // Insert the Node's parent to the list of Nodes for processing in the next iteration
                node_list[d - 1].insert(node->parent());
                // Propagate a summary of the Node's eight children to its parent
                const auto& node_data =
                    updating_model::propagateToNoteAtCoarserScale(node, voxel_depth, 1);
                // Prune the Node if needed
                if (node_data.observed
                    && node_data.x * node_data.y <= 0.95 * MultiresOFusion::min_occupancy) {
                    map.pool().deleteNode(node, voxel_depth);
                }
            }
        }
    }
}
