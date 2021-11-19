// SPDX-FileCopyrightText: 2019-2020 Nils Funk, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#ifndef UPDATING_MODEL_HPP
#define UPDATING_MODEL_HPP

#include <limits>

#include "../../../../../se_denseslam/include/se/constant_parameters.h"
#include "se/instance_segmentation.hpp"
#include "se/str_utils.hpp"


class updating_model {
    public:
    using OctreeType = se::Octree<MultiresOFusion::VoxelType>;
    using VoxelData = MultiresOFusion::VoxelType::VoxelData;
    using NodeType = se::Node<MultiresOFusion::VoxelType>;
    using VoxelBlockType = MultiresOFusion::VoxelType::VoxelBlockType;



    /**
   * \brief Compute the estimated uncertainty boundary for a given depth measurement.
   *
   * \param[in] depth_value The measured depth of the depth image.
   *
   * \return Three sigma uncertainty.
   */
    static inline float computeThreeSigma(const float depth_value)
    {
        if (MultiresOFusion::uncertainty_model == UncertaintyModel::linear) {
            return 3
                * se::math::clamp(MultiresOFusion::k_sigma * depth_value,
                                  MultiresOFusion::sigma_min,
                                  MultiresOFusion::sigma_max); // Livingroom dataset
        }
        else {
            return 3
                * se::math::clamp(MultiresOFusion::k_sigma * se::math::sq(depth_value),
                                  MultiresOFusion::sigma_min,
                                  MultiresOFusion::sigma_max); // Cow and lady
        }
    }



    /**
   * \brief Compute the estimated wall thickness tau for a given depth measurement.
   *
   * \param[in] depth_value The measured depth of the depth image.
   *
   * \return The estimated wall thickness.
   */
    static inline float computeTau(const float depth_value)
    {
        if (MultiresOFusion::const_surface_thickness == true) {
            return MultiresOFusion::tau_max; ///<< e.g. used in ICL-NUIM livingroom dataset.
        }
        else {
            return se::math::clamp(MultiresOFusion::k_tau * depth_value,
                                   MultiresOFusion::tau_min,
                                   MultiresOFusion::tau_max);
        }
    }



    /**
   * \brief Return a conservative meassure of the expected variance of a sensor model inside a voxel
   *        given its position and depth variance.
   *
   * \param[in] depth_value_min Depth measurement max value inside voxel.
   * \param[in] depth_value_max Depth measurement min value inside voxel.
   * \param[in] node_dist_min_m Minimum node distance along z-axis in meter.
   * \param[in] node_dist_max_m Maximum node distance along z-axis in meter.
   *
   * \return Estimate of the variance
   */
    static int lowVariance(const float depth_value_min,
                           const float depth_value_max,
                           const float node_dist_min_m,
                           const float node_dist_max_m)
    {
        // Assume worst case scenario -> no multiplication with proj_scale
        float z_diff_max = (node_dist_max_m - depth_value_min); // * proj_scale;
        float z_diff_min = (node_dist_min_m - depth_value_max); // * proj_scale;

        float tau_max = computeTau(depth_value_max);
        float three_sigma_min = computeThreeSigma(depth_value_max);

        if (z_diff_min > 1.25 * tau_max) { // behind of surface
            return 1;
        }
        else if (z_diff_max < -1.25 * three_sigma_min) { // guranteed free space
            return -1;
        }
        else {
            return 0;
        }
    }



    /**
   * \brief Update the weighted mean log-odd octant occupancy and set the octant to observed.
   *
   * \param[in,out] data         The data in the octant.
   * \param[in] sample_value The sample occupancy to be integrated.
   *
   * \return True/false if the voxel has been observed the first time
   */
    static inline bool weightedMeanUpdate(VoxelData& data, const float sample_value)
    {
        data.x = (data.x * data.y + sample_value) / (data.y + 1);
        data.y = std::min((short) (data.y + 1), MultiresOFusion::max_weight);
        if (data.observed) {
            return false;
        }
        else {
            data.observed = true;
            return true;
        }
    }



    /**
   * \brief Update a field with a new measurement, a weighting of 1 is considered for the new measurement.
   *
   * \param[in]     range_diff  The range difference between the voxel sample point and the depth value of the reprojection.
   * \param[in]     tau         The estimated wall thickness.
   * \param[in]     three_sigma The 3x sigma uncertainty.
   * \param[in,out] voxel_data  The reference to the voxel data of the voxel to be updated.
   *
   * \return True/false if the node has been observed the first time
   */
    static inline bool updateVoxel(const float range_diff,
                                   const float tau,
                                   const float three_sigma,
                                   const uint32_t rgba_value,
                                   const se::integration_mask_elem_t fg_value,
                                   VoxelData& voxel_data)
    {
        float sample_value;

        if (range_diff < -three_sigma) {
            sample_value = MultiresOFusion::log_odd_min;
        }
        else if (range_diff < tau / 2) {
            sample_value = std::min(MultiresOFusion::log_odd_min
                                        - MultiresOFusion::log_odd_min / three_sigma
                                            * (range_diff + three_sigma),
                                    MultiresOFusion::log_odd_max);
        }
        else if (range_diff < tau) {
            sample_value = std::min(-MultiresOFusion::log_odd_min * tau / (2 * three_sigma),
                                    MultiresOFusion::log_odd_max);
        }
        else {
            return false;
        }

        // Update the foreground probability.
        if (fg_value != se::InstanceSegmentation::skip_fg_update) {
            voxel_data.fg =
                (fg_value + voxel_data.fg * voxel_data.fg_count) / (voxel_data.fg_count + 1);
            voxel_data.fg_count++;
        }
        // Update the color.
        voxel_data.r =
            (se::r_from_rgba(rgba_value) + voxel_data.r * voxel_data.y) / (voxel_data.y + 1);
        voxel_data.g =
            (se::g_from_rgba(rgba_value) + voxel_data.g * voxel_data.y) / (voxel_data.y + 1);
        voxel_data.b =
            (se::b_from_rgba(rgba_value) + voxel_data.b * voxel_data.y) / (voxel_data.y + 1);

        return weightedMeanUpdate(voxel_data, sample_value);
    }



    /**
   * \brief Reduce the node data by the minimum log-odd occupancy update per iteration.
   *        This function can be used to faster update a octant if it's know that it is in free space.
   *        The aim is to increase computation time by avoiding to compute the sample value from scratch.
   *
   * \param[in,out] node_data The reference to the node data.
   */
    static inline void freeNode(MultiresOFusion::VoxelType::VoxelData& node_data)
    {
        weightedMeanUpdate(node_data, MultiresOFusion::log_odd_min);
    }


    /**
   * \brief Reduce the node data by the minimum log-odd occupancy update per iteration.
   *        This function can be used to faster update a octant if it's know that it is in free space.
   *        The aim is to increase computation time by avoiding to compute the sample value from scratch.
   *
   * \param[in,out] node_data The reference to the node data.
   */
    static inline bool freeVoxel(MultiresOFusion::VoxelType::VoxelData& voxel_data)
    {
        return weightedMeanUpdate(voxel_data, MultiresOFusion::log_odd_min);
    }


    /**
   * \brief Propagate a summary of the eight nodes children to its parent
   *
   * \param[in] node        Node to be summariesed
   * \param[in] voxel_depth Maximum depth of the octree
   * \param[in] frame       Current frame
   *
   * \return data Summary of the node
   */
    static inline VoxelData propagateToNoteAtCoarserScale(NodeType* node,
                                                          const unsigned int voxel_depth,
                                                          const unsigned int frame)
    {
        if (!node->parent()) {
            node->timestamp(frame);
            return MultiresOFusion::VoxelType::invalid();
        }

        float x_max = 0;
        short y_max = 0;
        float fg_max = 0.0f;
        uint16_t fg_count_max = 0u;
        uint8_t r_max = 0u;
        uint8_t g_max = 0u;
        uint8_t b_max = 0u;
        float o_max = -std::numeric_limits<float>::max();
        unsigned int observed_count = 0;
        unsigned int data_count = 0;
        bool frontier = false;
        for (unsigned int child_idx = 0; child_idx < 8; ++child_idx) {
            const auto& child_data = node->childData(child_idx);
            if (child_data.y > 0 && child_data.x * child_data.y > o_max) { // At least 1 integration
                data_count++;
                x_max = child_data.x;
                y_max = child_data.y;
                fg_max = child_data.fg;
                fg_count_max = child_data.fg_count;
                r_max = child_data.r;
                g_max = child_data.g;
                b_max = child_data.b;
                o_max = x_max * y_max;
                // Set the frontier flag of the node if at least 1 of its children is a frontier
                frontier = (frontier || child_data.frontier);
            }
            if (child_data.observed == true) {
                observed_count++;
            }
        }

        const unsigned int child_idx =
            se::child_idx(node->code(), se::keyops::depth(node->code()), voxel_depth);
        auto& node_data = node->parent()->childData(child_idx);

        if (data_count > 0) {
            node_data.x = x_max; // TODO: Need to check update?
            node_data.y = y_max;
            node_data.fg = fg_max;
            node_data.fg_count = fg_count_max;
            node_data.r = r_max;
            node_data.g = g_max;
            node_data.b = b_max;
            if (observed_count == 8) {
                node_data.observed = true;
            }
            node_data.frontier = frontier;
        }
        return node_data;
    }



    /**
   * \brief Summariese the values from the current integration scale recursively
   *        up to the block's max scale.
   *
   * \param[in] block         The block to be updated.
   * \param[in] initial_scale Scale from which propagate up voxel values
  */
    static inline void propagateBlockToCoarsestScale(VoxelBlockType* block)
    {
        if (block->current_scale() < VoxelBlockType::max_scale) {
            int target_scale = block->current_scale() + 1;
            unsigned int size_at_target_scale_li = block->size_li >> target_scale;
            unsigned int size_at_target_scale_sq = se::math::sq(size_at_target_scale_li);

            int child_scale = target_scale - 1;
            unsigned int size_at_child_scale_li = block->size_li >> child_scale;
            unsigned int size_at_child_scale_sq = se::math::sq(size_at_child_scale_li);

            if (block->buffer_scale() > block->current_scale()) {
                VoxelData* max_data_at_target_scale = block->blockMaxDataAtScale(target_scale);
                VoxelData* max_data_at_child_scale = block->blockDataAtScale(child_scale);

                for (unsigned int z = 0; z < size_at_target_scale_li; z++) {
                    for (unsigned int y = 0; y < size_at_target_scale_li; y++) {
                        for (unsigned int x = 0; x < size_at_target_scale_li; x++) {
                            const int target_max_data_idx =
                                x + y * size_at_target_scale_li + z * size_at_target_scale_sq;
                            auto& target_max_data = max_data_at_target_scale[target_max_data_idx];

                            float x_max = 0;
                            short y_max = 0;
                            float fg_max = 0.0f;
                            uint16_t fg_count_max = 0u;
                            uint8_t r_max = 0u;
                            uint8_t g_max = 0u;
                            uint8_t b_max = 0u;
                            float o_max = -std::numeric_limits<float>::max();

                            int observed_count = 0;
                            int data_count = 0;
                            bool frontier = false;

                            for (unsigned int k = 0; k < 2; k++) {
                                for (unsigned int j = 0; j < 2; j++) {
                                    for (unsigned int i = 0; i < 2; i++) {
                                        const int child_max_data_idx = (2 * x + i)
                                            + (2 * y + j) * size_at_child_scale_li
                                            + (2 * z + k) * size_at_child_scale_sq;
                                        const auto child_data =
                                            max_data_at_child_scale[child_max_data_idx];

                                        if (child_data.y > 0
                                            && (child_data.x * child_data.y) > o_max) {
                                            data_count++;
                                            // Update max
                                            x_max = child_data.x;
                                            y_max = child_data.y;
                                            fg_max = child_data.fg;
                                            fg_count_max = child_data.fg_count;
                                            r_max = child_data.r;
                                            g_max = child_data.g;
                                            b_max = child_data.b;
                                            o_max = x_max * y_max;
                                            // Set the frontier flag of the scale if at least 1 of its children is a frontier
                                            frontier = (frontier || child_data.frontier);
                                        }

                                        if (child_data.observed) {
                                            observed_count++;
                                        }

                                    } // i
                                }     // j
                            }         // k

                            if (data_count > 0) {
                                target_max_data.x = x_max;
                                target_max_data.y = y_max;
                                target_max_data.fg = fg_max;
                                target_max_data.fg_count = fg_count_max;
                                target_max_data.r = r_max;
                                target_max_data.g = g_max;
                                target_max_data.b = b_max;
                                if (observed_count == 8) {
                                    target_max_data.observed =
                                        true; // TODO: We don't set the observed count to true for mean values
                                }
                                target_max_data.frontier = frontier;
                            }

                        } // x
                    }     // y
                }         // z
            }
            else {
                VoxelData* max_data_at_target_scale = block->blockMaxDataAtScale(target_scale);
                VoxelData* data_at_target_scale = block->blockDataAtScale(target_scale);
                VoxelData* data_at_child_scale = block->blockDataAtScale(child_scale);

                for (unsigned int z = 0; z < size_at_target_scale_li; z++) {
                    for (unsigned int y = 0; y < size_at_target_scale_li; y++) {
                        for (unsigned int x = 0; x < size_at_target_scale_li; x++) {
                            const int target_data_idx =
                                x + y * size_at_target_scale_li + z * size_at_target_scale_sq;
                            auto& target_data = data_at_target_scale[target_data_idx];
                            auto& target_max_data = max_data_at_target_scale[target_data_idx];


                            float x_mean = 0;
                            short y_mean = 0;
                            float fg_mean = 0.0f;
                            uint32_t fg_count_mean = 0u;
                            uint16_t r_mean = 0u;
                            uint16_t g_mean = 0u;
                            uint16_t b_mean = 0u;

                            float x_max = 0;
                            short y_max = 0;
                            float fg_max = 0.0f;
                            uint16_t fg_count_max = 0u;
                            uint8_t r_max = 0u;
                            uint8_t g_max = 0u;
                            uint8_t b_max = 0u;
                            float o_max = -std::numeric_limits<float>::max();

                            int observed_count = 0;
                            int data_count = 0;
                            bool frontier = false;

                            for (unsigned int k = 0; k < 2; k++) {
                                for (unsigned int j = 0; j < 2; j++) {
                                    for (unsigned int i = 0; i < 2; i++) {
                                        const int child_data_idx = (2 * x + i)
                                            + (2 * y + j) * size_at_child_scale_li
                                            + (2 * z + k) * size_at_child_scale_sq;
                                        const auto child_data = data_at_child_scale[child_data_idx];

                                        if (child_data.y > 0) {
                                            // Update mean
                                            data_count++;
                                            x_mean += child_data.x;
                                            y_mean += child_data.y;
                                            fg_mean += child_data.fg;
                                            fg_count_mean += child_data.fg_count;
                                            r_mean += child_data.r;
                                            g_mean += child_data.g;
                                            b_mean += child_data.b;
                                            // Set the frontier flag of the scale if at least 1 of its children is a frontier
                                            frontier = (frontier || child_data.frontier);

                                            if ((child_data.x * child_data.y) > o_max) {
                                                // Update max
                                                x_max = child_data.x;
                                                y_max = child_data.y;
                                                fg_max = child_data.fg;
                                                fg_count_max = child_data.fg_count;
                                                r_max = child_data.r;
                                                g_max = child_data.g;
                                                b_max = child_data.b;
                                                o_max = x_max * y_max;
                                            }
                                        }

                                        if (child_data.observed) {
                                            observed_count++;
                                        }

                                    } // i
                                }     // j
                            }         // k

                            if (data_count > 0) {
                                target_data.x = x_mean / data_count;
                                target_data.y = ceil((float) y_mean) / data_count;
                                target_data.fg = fg_mean / data_count;
                                target_data.fg_count = fg_count_mean / data_count;
                                target_data.r = r_mean / data_count;
                                target_data.g = g_mean / data_count;
                                target_data.b = b_mean / data_count;
                                target_data.observed = false;
                                target_data.frontier = frontier;

                                //              target_data.x = x_mean / data_count;
                                //              target_data.y = y_max;
                                //              if (observed_count == 8) {
                                //                target_data.observed = true; // TODO: We don't set the observed count to true for mean values
                                //              }

                                //              target_data.x = x_max;
                                //              target_data.y = y_max;
                                //              if (observed_count == 8) {
                                //                target_data.observed = true; // TODO: We don't set the observed count to true for mean values
                                //              }

                                target_max_data.x = x_max;
                                target_max_data.y = y_max;
                                target_max_data.fg = fg_max;
                                target_max_data.fg_count = fg_count_max;
                                target_max_data.r = r_max;
                                target_max_data.g = g_max;
                                target_max_data.b = b_max;
                                if (observed_count == 8) {
                                    target_max_data.observed =
                                        true; // TODO: We don't set the observed count to true for mean values
                                }
                                target_max_data.frontier = frontier;

                                //              if (abs(target_data.x - target_max_data.x) > 1) {
                                //                std::cout << "-----" << std::endl;
                                //                std::cout << target_data.x << "/" << target_max_data.x << "/" << data_count << std::endl;
                                //                for (int k = 0; k < 2; k++) {
                                //                  for (int j = 0; j < 2; j++) {
                                //                    for (int i = 0; i < 2; i++) {
                                //
                                //                      const int child_data_idx = (2 * x + i) + (2 * y + j) * size_at_child_scale_li + (2 * z + k) * size_at_child_scale_sq;
                                //                      const auto child_data = data_at_child_scale[child_data_idx];
                                //
                                //                      std::cout << str_utils::value_to_pretty_str(child_data.x, "child.x") << std::endl;
                                //                      std::cout << str_utils::value_to_pretty_str(child_data.y, "child.y") << std::endl;
                                //
                                //                    } // i
                                //                  } // j
                                //                } // k
                                //              }
                            }

                        } // x
                    }     // y
                }         // z
            }



            for (target_scale += 1; target_scale <= VoxelBlockType::max_scale; ++target_scale) {
                unsigned int size_at_target_scale_li = block->size_li >> target_scale;
                unsigned int size_at_target_scale_sq = se::math::sq(size_at_target_scale_li);

                int child_scale = target_scale - 1;
                unsigned int size_at_child_scale_li = block->size_li >> child_scale;
                unsigned int size_at_child_scale_sq = se::math::sq(size_at_child_scale_li);

                VoxelData* max_data_at_target_scale = block->blockMaxDataAtScale(target_scale);
                VoxelData* data_at_target_scale = block->blockDataAtScale(target_scale);
                VoxelData* max_data_at_child_scale = block->blockMaxDataAtScale(child_scale);
                VoxelData* data_at_child_scale = block->blockDataAtScale(child_scale);

                for (unsigned int z = 0; z < size_at_target_scale_li; z++) {
                    for (unsigned int y = 0; y < size_at_target_scale_li; y++) {
                        for (unsigned int x = 0; x < size_at_target_scale_li; x++) {
                            const int target_data_idx =
                                x + y * size_at_target_scale_li + z * size_at_target_scale_sq;
                            auto& target_data = data_at_target_scale[target_data_idx];
                            auto& target_max_data = max_data_at_target_scale[target_data_idx];

                            float x_mean = 0;
                            short y_mean = 0;
                            float fg_mean = 0.0f;
                            uint32_t fg_count_mean = 0u;
                            uint16_t r_mean = 0u;
                            uint16_t g_mean = 0u;
                            uint16_t b_mean = 0u;

                            float x_max = 0;
                            short y_max = 0;
                            float fg_max = 0.0f;
                            uint16_t fg_count_max = 0u;
                            uint8_t r_max = 0u;
                            uint8_t g_max = 0u;
                            uint8_t b_max = 0u;
                            float o_max = -std::numeric_limits<float>::max();

                            int observed_count = 0;
                            int data_count = 0;
                            bool frontier = false;

                            for (unsigned int k = 0; k < 2; k++) {
                                for (unsigned int j = 0; j < 2; j++) {
                                    for (unsigned int i = 0; i < 2; i++) {
                                        const int child_data_idx = (2 * x + i)
                                            + (2 * y + j) * size_at_child_scale_li
                                            + (2 * z + k) * size_at_child_scale_sq;
                                        const auto child_data = data_at_child_scale[child_data_idx];
                                        const auto child_max_data =
                                            max_data_at_child_scale[child_data_idx];

                                        if (child_max_data.y > 0) {
                                            // Update mean
                                            data_count++;
                                            x_mean += child_data.x;
                                            y_mean += child_data.y;
                                            fg_mean += child_data.fg;
                                            fg_count_mean += child_data.fg_count;
                                            r_mean += child_data.r;
                                            g_mean += child_data.g;
                                            b_mean += child_data.b;
                                            // Set the frontier flag of the scale if at least 1 of its children is a frontier
                                            frontier = (frontier || child_data.frontier);

                                            if ((child_max_data.x * child_max_data.y) > o_max) {
                                                // Update max
                                                x_max = child_max_data.x;
                                                y_max = child_max_data.y;
                                                fg_max = child_max_data.fg;
                                                fg_count_max = child_max_data.fg_count;
                                                r_max = child_max_data.r;
                                                g_max = child_max_data.g;
                                                b_max = child_max_data.b;
                                                o_max = x_max * y_max;
                                            }
                                        }

                                        if (child_max_data.observed) {
                                            observed_count++;
                                        }

                                    } // i
                                }     // j
                            }         // k

                            if (data_count > 0) {
                                target_data.x = x_mean / data_count;
                                target_data.y = ceil((float) y_mean) / data_count;
                                target_data.fg = fg_mean / data_count;
                                target_data.fg_count = fg_count_mean / data_count;
                                target_data.r = r_mean / data_count;
                                target_data.g = g_mean / data_count;
                                target_data.b = b_mean / data_count;
                                target_data.observed = false;
                                target_data.frontier = frontier;

                                //              target_data.x = x_mean / data_count;
                                //              target_data.y = ceil((float) y_mean) / data_count;
                                //              if (observed_count == 8) {
                                //                target_data.observed = true; // TODO: We don't set the observed count to true for mean values
                                //              }

                                //              target_data.x = x_max;
                                //              target_data.y = y_max;
                                //              if (observed_count == 8) {
                                //                target_data.observed = true; // TODO: We don't set the observed count to true for mean values
                                //              }

                                target_max_data.x = x_max;
                                target_max_data.y = y_max;
                                target_max_data.fg = fg_max;
                                target_max_data.fg_count = fg_count_max;
                                target_max_data.r = r_max;
                                target_max_data.g = g_max;
                                target_max_data.b = b_max;
                                if (observed_count == 8) {
                                    target_max_data.observed =
                                        true; // TODO: We don't set the observed count to true for mean values
                                }
                                target_max_data.frontier = frontier;
                            }

                        } // x
                    }     // y
                }         // z
            }
        }
    }



    /**
   * \brief Propagate the maximum log-odd occupancy of the eight children up to the next scale.
   *
   * \note UNUSED FUNCTION
   *
   * \note The maximum log-odd occupancy is the maximum partly observed log-odd occupancy.
   *       To check if the node is safe for planning, verify if the log-odd occupancy is below a chosen threshold
   *       and that the node is observed (data.observed == true -> node or all children have been seen).
   *
   * \note The maximum log-odd occupancy is the maximum product of data.x * data.y and
   *       not the maximum mean log-odd Occupancy data.x.
   *
   * \param[in,out] block           The block in which the target voxel is included.
   * \param[in]     voxel_coord     The coordinates of the target voxel (corner).
   * \param[in]     target_scale    The scale to which to propage the data to.
   * \param[in]     target_stride   The stride in voxel units of the target scale.
   * \param[in,out] target_data     The data reference to the target voxel.
   */
    static inline void maxCoarsePropagation(const VoxelBlockType* block,
                                            const Eigen::Vector3i target_coord,
                                            const int target_scale,
                                            const unsigned int target_stride,
                                            VoxelData& target_data)
    {
        float x_max = -std::numeric_limits<float>::max();
        short y_max = 0;
        float fg_max = 0.0f;
        uint16_t fg_count_max = 0u;
        uint8_t r_max = 0u;
        uint8_t g_max = 0u;
        uint8_t b_max = 0u;
        float o_max = -std::numeric_limits<float>::max();
        unsigned int observed_count = 0;
        unsigned int data_count = 0;

        int child_scale = target_scale - 1;
        unsigned int child_stride = target_stride >> 1; ///<< Halfen target stride

        for (unsigned int k = 0; k < target_stride; k += child_stride) {
            for (unsigned int j = 0; j < target_stride; j += child_stride) {
                for (unsigned int i = 0; i < target_stride; i += child_stride) {
                    const auto child_data =
                        block->data(target_coord + Eigen::Vector3i(i, j, k), child_scale);
                    /// Only compare partly observed children (child_data.y > 0)
                    /// x_max is the product of data.x * data.y (i.e. not the mean log-odd occupancy)
                    if (child_data.y > 0 && ((child_data.x * child_data.y) > o_max)) {
                        data_count++;
                        x_max = child_data.x;
                        y_max = child_data.y;
                        fg_max = child_data.fg;
                        fg_count_max = child_data.fg_count;
                        r_max = child_data.r;
                        g_max = child_data.g;
                        b_max = child_data.b;
                        o_max = x_max * y_max;
                    }

                    if (child_data.observed) {
                        observed_count++;
                    }

                } // i
            }     // j
        }         // k

        if (data_count > 0) {
            target_data.x = x_max;
            target_data.y = y_max;
            target_data.fg = fg_max;
            target_data.fg_count = fg_count_max;
            target_data.r = r_max;
            target_data.g = g_max;
            target_data.b = b_max;
            if (observed_count
                == 8) { ///<< If all children have been observed, set parent/target to observed.
                target_data.observed = true;
            }
        }
    }



    static inline void meanCoarsePropagation(const VoxelBlockType* block,
                                             const Eigen::Vector3i target_coord,
                                             const int target_scale,
                                             const unsigned int target_stride,
                                             VoxelData& target_data)
    {
        float x_mean = 0;
        short y_mean = 0;
        float fg_mean = 0.0f;
        uint32_t fg_count_mean = 0u;
        uint16_t r_mean = 0u;
        uint16_t g_mean = 0u;
        uint16_t b_mean = 0u;
        unsigned int observed_count = 0;
        unsigned int data_count = 0;

        int child_scale = target_scale - 1;
        unsigned int child_stride = target_stride >> 1;

        for (unsigned int k = 0; k < target_stride; k += child_stride) {
            for (unsigned int j = 0; j < target_stride; j += child_stride) {
                for (unsigned int i = 0; i < target_stride; i += child_stride) {
                    auto child_data =
                        block->data(target_coord + Eigen::Vector3i(i, j, k), child_scale);
                    if (child_data.y > 0) {
                        data_count++;
                        x_mean += child_data.x;
                        y_mean += child_data.y;
                        fg_mean += child_data.fg;
                        fg_count_mean += child_data.fg_count;
                        r_mean += child_data.r;
                        g_mean += child_data.g;
                        b_mean += child_data.b;
                    }
                    if (child_data.observed) {
                        observed_count++;
                    }
                }
            }
        }

        if (data_count == 8) {
            target_data.x = x_mean / data_count;
            target_data.y = ((float) y_mean) / data_count;
            target_data.fg_count = fg_count_mean / data_count;
            target_data.r = r_mean / data_count;
            target_data.g = g_mean / data_count;
            target_data.b = b_mean / data_count;
            target_data.observed = true;

            //      // TODO: ^SWITCH 2 - Set observed if all children are known.
            //      if (observed_count == 8) {
            //        target_data.observed = true;
            //      }
        }
    }

    static inline void propagateToVoxelAtCoarserScale(const VoxelBlockType* block,
                                                      const Eigen::Vector3i voxel_coord,
                                                      const int target_scale,
                                                      const unsigned int target_stride,
                                                      VoxelData& voxel_data)
    {
        maxCoarsePropagation(block, voxel_coord, target_scale, target_stride, voxel_data);
    }
};


#endif //UPDATING_MODEL_HPP
