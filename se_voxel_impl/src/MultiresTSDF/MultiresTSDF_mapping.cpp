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

#include "se/filter.hpp"
#include "se/functors/for_each.hpp"
#include "se/image/image.hpp"
#include "se/image_utils.hpp"
#include "se/instance_segmentation.hpp"
#include "se/node.hpp"
#include "se/octree.hpp"
#include "se/voxel_implementations/MultiresTSDF/MultiresTSDF.hpp"



struct MultiresTSDFUpdate {
    using VoxelType = MultiresTSDF::VoxelType;
    using VoxelData = MultiresTSDF::VoxelType::VoxelData;
    using OctreeType = se::Octree<MultiresTSDF::VoxelType>;
    using NodeType = se::Node<MultiresTSDF::VoxelType>;
    using VoxelBlockType = typename MultiresTSDF::VoxelType::VoxelBlockType;

    MultiresTSDFUpdate(const OctreeType& map,
                       const se::Image<float>& depth_image,
                       const se::Image<uint32_t>& rgba_image,
                       const cv::Mat& fg_image,
                       const Eigen::Matrix4f& T_CM,
                       const SensorImpl sensor,
                       const float voxel_dim) :
            map_(map),
            depth_image_(depth_image),
            rgba_image_(rgba_image),
            fg_image_(fg_image),
            T_CM_(T_CM),
            sensor_(sensor),
            voxel_dim_(voxel_dim),
            sample_offset_frac_(map.sample_offset_frac_)
    {
    }

    const OctreeType& map_;
    const se::Image<float>& depth_image_;
    const se::Image<uint32_t>& rgba_image_;
    const cv::Mat& fg_image_;
    const Eigen::Matrix4f& T_CM_;
    const SensorImpl sensor_;
    const float voxel_dim_;
    const Eigen::Vector3f& sample_offset_frac_;

    /**
   * Update the subgrids of a voxel block starting from a given scale up
   * to a maximum scale.
   *
   * \param[in] block VoxelBlock to be updated
   * \param[in] scale scale from which propagate up voxel values
   */
    static void propagateUp(VoxelBlockType* block, const int scale)
    {
        const Eigen::Vector3i block_coord = block->coordinates();
        const int block_size = VoxelBlockType::size_li;
        for (int voxel_scale = scale; voxel_scale < se::math::log2_const(block_size);
             ++voxel_scale) {
            const int stride = 1 << (voxel_scale + 1);
            for (int z = 0; z < block_size; z += stride)
                for (int y = 0; y < block_size; y += stride)
                    for (int x = 0; x < block_size; x += stride) {
                        const Eigen::Vector3i voxel_coord = block_coord + Eigen::Vector3i(x, y, z);

                        float mean = 0;
                        int sample_count = 0;
                        float weight = 0;
                        float fg = 0.0f;
                        float fg_count = 0.0f;
                        float r = 0;
                        float g = 0;
                        float b = 0;
                        for (int k = 0; k < stride; k += stride / 2) {
                            for (int j = 0; j < stride; j += stride / 2) {
                                for (int i = 0; i < stride; i += stride / 2) {
                                    VoxelData child_data = block->data(
                                        voxel_coord + Eigen::Vector3i(i, j, k), voxel_scale);
                                    if (child_data.y != 0) {
                                        mean += child_data.getTsdf();
                                        weight += child_data.y;
                                        fg += child_data.getFg();
                                        fg_count += child_data.fg_count;
                                        r += child_data.r;
                                        g += child_data.g;
                                        b += child_data.b;
                                        sample_count++;
                                    }
                                }
                            }
                        }
                        VoxelData voxel_data = block->data(voxel_coord, voxel_scale + 1);

                        if (sample_count != 0) {
                            mean /= sample_count;
                            weight /= sample_count;
                            fg /= sample_count;
                            fg_count /= sample_count;
                            r /= sample_count;
                            g /= sample_count;
                            b /= sample_count;
                            voxel_data.setTsdf(mean);
                            voxel_data.setTsdfLast(mean);
                            voxel_data.y = weight + 0.5f;
                            voxel_data.setFg(fg);
                            voxel_data.fg_count = fg_count + 0.5f;
                            voxel_data.r = r + 0.5f;
                            voxel_data.g = g + 0.5f;
                            voxel_data.b = b + 0.5f;
                        }
                        else {
                            voxel_data = VoxelType::initData();
                        }
                        voxel_data.delta_y = 0;
                        block->setData(voxel_coord, voxel_scale + 1, voxel_data);
                    }
        }
    }



    static void propagateUp(NodeType* node, const int voxel_depth, const unsigned timestamp)
    {
        if (!node->parent()) {
            node->timestamp(timestamp);
            return;
        }

        float mean = 0;
        int sample_count = 0;
        float weight = 0;
        float fg = 0.0f;
        float fg_count = 0.0f;
        float r = 0;
        float g = 0;
        float b = 0;
        for (int child_idx = 0; child_idx < 8; ++child_idx) {
            const VoxelData& child_data = node->childData(child_idx);
            if (child_data.y != 0) {
                mean += child_data.getTsdf();
                weight += child_data.y;
                fg += child_data.getFg();
                fg_count += child_data.fg_count;
                r += child_data.r;
                g += child_data.g;
                b += child_data.b;
                sample_count++;
            }
        }

        const unsigned int child_idx =
            se::child_idx(node->code(), se::keyops::depth(node->code()), voxel_depth);
        if (sample_count > 0) {
            VoxelData& node_data = node->parent()->childData(child_idx);
            mean /= sample_count;
            weight /= sample_count;
            fg /= sample_count;
            fg_count /= sample_count;
            r /= sample_count;
            g /= sample_count;
            b /= sample_count;
            node_data.setTsdf(mean);
            node_data.setTsdfLast(mean);
            node_data.y = weight + 0.5f;
            node_data.setFg(fg);
            node_data.fg_count = fg_count + 0.5f;
            node_data.r = r + 0.5f;
            node_data.g = g + 0.5f;
            node_data.b = b + 0.5f;
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
    static void propagateDown(const OctreeType& map,
                              VoxelBlockType* block,
                              const int scale,
                              const int min_scale)
    {
        const Eigen::Vector3i block_coord = block->coordinates();
        const int block_size = VoxelBlockType::size_li;
        for (int voxel_scale = scale; voxel_scale > min_scale; --voxel_scale) {
            const int stride = 1 << voxel_scale;
            for (int z = 0; z < block_size; z += stride) {
                for (int y = 0; y < block_size; y += stride) {
                    for (int x = 0; x < block_size; x += stride) {
                        const Eigen::Vector3i parent_coord = block_coord + Eigen::Vector3i(x, y, z);
                        VoxelData parent_data = block->data(parent_coord, voxel_scale);
                        float delta_x = parent_data.getTsdf() - parent_data.getTsdfLast();
                        const int half_stride = stride / 2;
                        for (int k = 0; k < stride; k += half_stride) {
                            for (int j = 0; j < stride; j += half_stride) {
                                for (int i = 0; i < stride; i += half_stride) {
                                    const Eigen::Vector3i voxel_coord =
                                        parent_coord + Eigen::Vector3i(i, j, k);
                                    VoxelData voxel_data =
                                        block->data(voxel_coord, voxel_scale - 1);
                                    if (voxel_data.y == 0) {
                                        bool is_valid;
                                        const Eigen::Vector3f voxel_sample_coord_f =
                                            se::get_sample_coord(
                                                voxel_coord, stride, map.sample_offset_frac_);
                                        voxel_data.setTsdf(
                                            se::math::clamp(map.interp(voxel_sample_coord_f,
                                                                       VoxelType::selectNodeValue,
                                                                       VoxelType::selectVoxelValue,
                                                                       voxel_scale - 1,
                                                                       is_valid)
                                                                .first,
                                                            -1.f,
                                                            1.f));
                                        voxel_data.y = is_valid ? parent_data.y : 0;
                                        voxel_data.setTsdfLast(voxel_data.getTsdf());
                                        voxel_data.delta_y = 0;
                                        voxel_data.setFg(
                                            map.interp(
                                                   voxel_sample_coord_f,
                                                   [](const auto&) {
                                                       return MultiresTSDF::VoxelType::initData()
                                                           .getFg();
                                                   },
                                                   [](const auto& x) { return x.getFg(); },
                                                   voxel_scale - 1,
                                                   is_valid)
                                                .first);
                                        voxel_data.fg_count = is_valid ? parent_data.fg_count : 0;
                                        voxel_data.r = parent_data.r;
                                        voxel_data.g = parent_data.g;
                                        voxel_data.b = parent_data.b;
                                    }
                                    else {
                                        voxel_data.setTsdfLast(
                                            std::max(voxel_data.getTsdf() + delta_x, -1.f));
                                        se::math::increment_clamp(voxel_data.y,
                                                                  parent_data.delta_y,
                                                                  MultiresTSDF::max_weight);
                                        voxel_data.delta_y = parent_data.delta_y;
                                        voxel_data.setFg(parent_data.getFg());
                                        se::math::increment_clamp(voxel_data.fg_count,
                                                                  parent_data.fg_count,
                                                                  MultiresTSDF::max_weight);
                                        voxel_data.r = parent_data.r;
                                        voxel_data.g = parent_data.g;
                                        voxel_data.b = parent_data.b;
                                    }
                                    block->setData(voxel_coord, voxel_scale - 1, voxel_data);
                                }
                            }
                        }
                        parent_data.setTsdfLast(parent_data.getTsdf());
                        parent_data.delta_y = 0;
                        block->setData(parent_coord, voxel_scale, parent_data);
                    }
                }
            }
        }
    }



    /**
   * Update a voxel block at a given scale by first propagating down the parent
   * values and then integrating the new measurement;
   */
    void propagateUpdate(VoxelBlockType* block, const int voxel_scale)
    {
        const int block_size = VoxelBlockType::size_li;
        const int parent_scale = voxel_scale + 1;
        const int parent_stride = 1 << parent_scale;
        const int voxel_stride = parent_stride >> 1;
        bool is_visible = false;

        const Eigen::Vector3i block_coord = block->coordinates();

        for (unsigned int z = 0; z < block_size; z += parent_stride) {
            for (unsigned int y = 0; y < block_size; y += parent_stride) {
                for (unsigned int x = 0; x < block_size; x += parent_stride) {
                    const Eigen::Vector3i parent_coord = block_coord + Eigen::Vector3i(x, y, z);
                    VoxelData parent_data = block->data(parent_coord, parent_scale);
                    float delta_x = parent_data.getTsdf() - parent_data.getTsdfLast();
                    for (int k = 0; k < parent_stride; k += voxel_stride) {
                        for (int j = 0; j < parent_stride; j += voxel_stride) {
                            for (int i = 0; i < parent_stride; i += voxel_stride) {
                                const Eigen::Vector3i voxel_coord =
                                    parent_coord + Eigen::Vector3i(i, j, k);
                                VoxelData voxel_data = block->data(voxel_coord, voxel_scale);
                                const Eigen::Vector3f voxel_sample_coord_f = se::get_sample_coord(
                                    voxel_coord, voxel_stride, sample_offset_frac_);
                                if (voxel_data.y == 0) {
                                    bool is_valid;
                                    voxel_data.setTsdf(
                                        se::math::clamp(map_.interp(voxel_sample_coord_f,
                                                                    VoxelType::selectNodeValue,
                                                                    VoxelType::selectVoxelValue,
                                                                    voxel_scale + 1,
                                                                    is_valid)
                                                            .first,
                                                        -1.f,
                                                        1.f));
                                    voxel_data.y = is_valid ? parent_data.y : 0;
                                    voxel_data.setTsdfLast(voxel_data.getTsdf());
                                    voxel_data.delta_y = 0;
                                    voxel_data.setFg(
                                        map_.interp(
                                                voxel_sample_coord_f,
                                                [](const auto&) {
                                                    return MultiresTSDF::VoxelType::initData()
                                                        .getFg();
                                                },
                                                [](const auto& x) { return x.getFg(); },
                                                voxel_scale - 1,
                                                is_valid)
                                            .first);
                                    voxel_data.fg_count = is_valid ? parent_data.fg_count : 0;
                                    voxel_data.r = parent_data.r;
                                    voxel_data.g = parent_data.g;
                                    voxel_data.b = parent_data.b;
                                }
                                else {
                                    voxel_data.setTsdf(
                                        se::math::clamp(voxel_data.getTsdf() + delta_x, -1.f, 1.f));
                                    se::math::increment_clamp(voxel_data.y,
                                                              parent_data.delta_y,
                                                              MultiresTSDF::max_weight);
                                    voxel_data.delta_y = parent_data.delta_y;
                                    voxel_data.setFg(parent_data.getFg());
                                    se::math::increment_clamp(voxel_data.fg_count,
                                                              parent_data.fg_count,
                                                              MultiresTSDF::max_weight);
                                    voxel_data.r = parent_data.r;
                                    voxel_data.g = parent_data.g;
                                    voxel_data.b = parent_data.b;
                                }

                                const Eigen::Vector3f point_C =
                                    (T_CM_ * (voxel_dim_ * voxel_sample_coord_f).homogeneous())
                                        .head(3);

                                // Don't update the point if the sample point is behind the far plane
                                if (point_C.norm() > sensor_.farDist(point_C)) {
                                    continue;
                                }

                                float depth_value(0);
                                uint32_t rgba_value(0);
                                se::integration_mask_elem_t fg_value(0);
                                if (!sensor_.projectToPixelValue(
                                        point_C,
                                        depth_image_,
                                        depth_value,
                                        rgba_image_,
                                        rgba_value,
                                        fg_image_,
                                        fg_value,
                                        [&](float depth_value,
                                            uint32_t,
                                            se::integration_mask_elem_t fg_value) {
                                            return depth_value >= sensor_.near_plane
                                                && fg_value
                                                != se::InstanceSegmentation::skip_integration;
                                        })) {
                                    continue;
                                }

                                is_visible = true;

                                // Update the TSDF
                                const float m = sensor_.measurementFromPoint(point_C);
                                const float sdf_value = (depth_value - m) / m * point_C.norm();
                                if (sdf_value > -MultiresTSDF::mu * (1 << voxel_scale)) {
                                    const float tsdf_value =
                                        fminf(1.f, sdf_value / MultiresTSDF::mu);
                                    voxel_data.setTsdf(se::math::clamp(
                                        (static_cast<float>(voxel_data.y) * voxel_data.getTsdf()
                                         + tsdf_value)
                                            / (static_cast<float>(voxel_data.y) + 1.f),
                                        -1.f,
                                        1.f));
                                    // Update the foreground probability.
                                    if (fg_value != se::InstanceSegmentation::skip_fg_update) {
                                        voxel_data.setFg(
                                            (fg_value + voxel_data.getFg() * voxel_data.fg_count)
                                            / (voxel_data.fg_count + 1));
                                        se::math::increment_clamp(
                                            voxel_data.fg_count,
                                            static_cast<MultiresTSDF::weight_t>(1),
                                            MultiresTSDF::max_weight);
                                    }
                                    // Update the color.
                                    voxel_data.r =
                                        (se::r_from_rgba(rgba_value) + voxel_data.r * voxel_data.y)
                                        / (voxel_data.y + 1);
                                    voxel_data.g =
                                        (se::g_from_rgba(rgba_value) + voxel_data.g * voxel_data.y)
                                        / (voxel_data.y + 1);
                                    voxel_data.b =
                                        (se::b_from_rgba(rgba_value) + voxel_data.b * voxel_data.y)
                                        / (voxel_data.y + 1);
                                    se::math::increment_clamp(
                                        voxel_data.y,
                                        static_cast<MultiresTSDF::weight_t>(1),
                                        MultiresTSDF::max_weight);
                                    se::math::increment_clamp(
                                        voxel_data.delta_y,
                                        static_cast<MultiresTSDF::weight_t>(1),
                                        MultiresTSDF::max_weight);
                                }
                                block->setData(voxel_coord, voxel_scale, voxel_data);
                            }
                        }
                    }
                    parent_data.setTsdfLast(parent_data.getTsdf());
                    parent_data.delta_y = 0;
                    block->setData(parent_coord, parent_scale, parent_data);
                }
            }
        }
        block->current_scale(voxel_scale);
        block->active(is_visible);
    }



    void operator()(VoxelBlockType* block)
    {
        constexpr int block_size = VoxelBlockType::size_li;
        const Eigen::Vector3i block_coord = block->coordinates();
        const Eigen::Vector3f block_centre_coord_f =
            se::get_sample_coord(block_coord, block_size, Eigen::Vector3f::Constant(0.5f));
        const Eigen::Vector3f block_centre_point_C =
            (T_CM_ * (voxel_dim_ * block_centre_coord_f).homogeneous()).head(3);
        const int last_scale = block->current_scale();

        const int scale = std::max(sensor_.computeIntegrationScale(block_centre_point_C,
                                                                   voxel_dim_,
                                                                   last_scale,
                                                                   block->min_scale(),
                                                                   map_.maxBlockScale()),
                                   last_scale - 1);
        block->min_scale(block->min_scale() < 0 ? scale : std::min(block->min_scale(), scale));
        block->minDistUpdated(sensor_.measurementFromPoint(block_centre_point_C));
        if (last_scale > scale) {
            propagateUpdate(block, scale);
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
                        se::get_sample_coord(voxel_coord, stride, sample_offset_frac_);
                    const Eigen::Vector3f point_C =
                        (T_CM_ * (voxel_dim_ * voxel_sample_coord_f).homogeneous()).head(3);

                    // Don't update the point if the sample point is behind the far plane
                    if (point_C.norm() > sensor_.farDist(point_C)) {
                        continue;
                    }
                    float depth_value(0);
                    uint32_t rgba_value(0);
                    se::integration_mask_elem_t fg_value(0);
                    if (!sensor_.projectToPixelValue(
                            point_C,
                            depth_image_,
                            depth_value,
                            rgba_image_,
                            rgba_value,
                            fg_image_,
                            fg_value,
                            [&](float depth_value, uint32_t, se::integration_mask_elem_t fg_value) {
                                return depth_value >= sensor_.near_plane
                                    && fg_value != se::InstanceSegmentation::skip_integration;
                            })) {
                        continue;
                    }

                    is_visible = true;

                    // Update the TSDF
                    const float m = sensor_.measurementFromPoint(point_C);
                    const float sdf_value = (depth_value - m) / m * point_C.norm();
                    if (sdf_value > -MultiresTSDF::mu * (1 << scale)) {
                        const float tsdf_value = fminf(1.f, sdf_value / MultiresTSDF::mu);
                        VoxelData voxel_data = block->data(voxel_coord, scale);
                        voxel_data.setTsdf(se::math::clamp(
                            (static_cast<float>(voxel_data.y) * voxel_data.getTsdf() + tsdf_value)
                                / (static_cast<float>(voxel_data.y) + 1.f),
                            -1.f,
                            1.f));
                        // Update the foreground probability.
                        if (fg_value != se::InstanceSegmentation::skip_fg_update) {
                            voxel_data.setFg((fg_value + voxel_data.getFg() * voxel_data.fg_count)
                                             / (voxel_data.fg_count + 1));
                            se::math::increment_clamp(voxel_data.fg_count,
                                                      static_cast<MultiresTSDF::weight_t>(1),
                                                      MultiresTSDF::max_weight);
                        }
                        // Update the color.
                        voxel_data.r = (se::r_from_rgba(rgba_value) + voxel_data.r * voxel_data.y)
                            / (voxel_data.y + 1);
                        voxel_data.g = (se::g_from_rgba(rgba_value) + voxel_data.g * voxel_data.y)
                            / (voxel_data.y + 1);
                        voxel_data.b = (se::b_from_rgba(rgba_value) + voxel_data.b * voxel_data.y)
                            / (voxel_data.y + 1);
                        se::math::increment_clamp(voxel_data.y,
                                                  static_cast<MultiresTSDF::weight_t>(1),
                                                  MultiresTSDF::max_weight);
                        se::math::increment_clamp(
                            voxel_data.delta_y,
                            static_cast<MultiresTSDF::MultiresTSDF::weight_t>(1),
                            MultiresTSDF::max_weight);
                        block->setData(voxel_coord, scale, voxel_data);
                    }
                }
            }
        }
        propagateUp(block, scale);
        block->active(is_visible);
    }
};

void MultiresTSDF::integrate(OctreeType& map,
                             const se::Image<float>& depth_image,
                             const se::Image<uint32_t>& rgba_image,
                             const cv::Mat& fg_image,
                             const Eigen::Matrix4f& T_CM,
                             const SensorImpl& sensor,
                             const unsigned frame)
{
    using namespace std::placeholders;

    /* Retrieve the active list */
    std::vector<VoxelBlockType*> active_list;
    auto& block_buffer = map.pool().blockBuffer();

    /* Predicates definition */
    const float voxel_dim = map.dim() / map.size();
    auto in_frustum_predicate = std::bind(
        se::algorithms::in_frustum<VoxelBlockType>, std::placeholders::_1, voxel_dim, T_CM, sensor);
    auto is_active_predicate = [](const VoxelBlockType* block) { return block->active(); };
    se::algorithms::filter(active_list, block_buffer, is_active_predicate, in_frustum_predicate);

    std::deque<se::Node<VoxelType>*> node_queue;
    struct MultiresTSDFUpdate block_update_funct(
        map, depth_image, rgba_image, fg_image, T_CM, sensor, voxel_dim);
    se::functor::internal::parallel_for_each(active_list, block_update_funct);

    for (const auto& block : active_list) {
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
        MultiresTSDFUpdate::propagateUp(node, map.voxelDepth(), frame);
        if (node->parent()) {
            node_queue.push_back(node->parent());
        }
    }
}
