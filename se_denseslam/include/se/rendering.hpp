/*

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.


 Copyright 2016 Emanuele Vespa, Imperial College London

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.

 3. Neither the name of the copyright holder nor the names of its contributors
 may be used to endorse or promote products derived from this software without
 specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef __RENDERING_HPP
#define __RENDERING_HPP

#include <cstdint>

#include "lodepng.h"
#include "se/commons.h"
#include "se/image/image.hpp"
#include "se/octree.hpp"
#include "se/sensor_implementation.hpp"
#include "se/timings.h"
#include "se/tracking.hpp"
#include "se/utils/math_utils.h"
#include "se/voxel_implementations.hpp"



namespace se {
namespace internal {
static se::Image<int> scale_image(2048, 960); // TODO: Adjust scale image based on image dimensions.
static std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> color_map = {
    {102, 194, 165},
    {252, 141, 98},
    {141, 160, 203},
    {231, 138, 195},
    {166, 216, 84},
    {255, 217, 47},
    {229, 196, 148},
    {179, 179, 179},
};
} // namespace internal
} // namespace se



template<typename VoxelImplT>
void raycastKernel(const se::Octree<typename VoxelImplT::VoxelType>& map,
                   se::Image<Eigen::Vector3f>& surface_point_cloud_M,
                   se::Image<Eigen::Vector3f>& surface_normals_M,
                   se::Image<int8_t>& scale_image,
                   se::Image<int8_t>& min_scale_image,
                   const Eigen::Matrix4f& raycast_T_MC,
                   const SensorImpl& sensor)
{
    TICKD("raycastKernel");
#pragma omp parallel for
    for (int y = 0; y < surface_point_cloud_M.height(); y++) {
#pragma omp simd
        for (int x = 0; x < surface_point_cloud_M.width(); x++) {
            Eigen::Vector4f surface_intersection_M;

            const Eigen::Vector2i pixel(x, y);
            const Eigen::Vector2f pixel_f = pixel.cast<float>();
            Eigen::Vector3f ray_dir_C;
            sensor.model.backProject(pixel_f, &ray_dir_C);
            const Eigen::Vector3f ray_dir_M =
                (se::math::to_rotation(raycast_T_MC) * ray_dir_C.normalized()).head(3);
            const Eigen::Vector3f t_MC = se::math::to_translation(raycast_T_MC);

            surface_intersection_M = VoxelImplT::raycast(
                map, t_MC, ray_dir_M, sensor.nearDist(ray_dir_C), sensor.farDist(ray_dir_C));
            if (surface_intersection_M.w() >= 0.f) {
                surface_point_cloud_M[x + y * surface_point_cloud_M.width()] =
                    surface_intersection_M.head<3>();
                Eigen::Vector3f surface_normal =
                    map.gradAtPoint(surface_intersection_M.head<3>(),
                                    VoxelImplT::VoxelType::selectNodeValue,
                                    VoxelImplT::VoxelType::selectVoxelValue,
                                    static_cast<int>(surface_intersection_M.w() + 0.5f));
                se::internal::scale_image(x, y) = static_cast<int>(surface_intersection_M.w());
                if (surface_normal.norm() == 0.f) {
                    surface_normals_M[pixel.x() + pixel.y() * surface_normals_M.width()] =
                        Eigen::Vector3f(INVALID, 0.f, 0.f);
                }
                else {
                    // Invert surface normals for TSDF representations.
                    surface_normals_M[pixel.x() + pixel.y() * surface_normals_M.width()] =
                        VoxelImplT::invert_normals ? (-1.f * surface_normal).normalized()
                                                   : surface_normal.normalized();
                }
                // Fetch the VoxelBlock containing the hit and get its minimum updated scale.
                const auto* block = map.fetch(map.pointToVoxel(surface_intersection_M.head<3>()));
                if (block) {
                    scale_image(x, y) = surface_intersection_M.w();
                    min_scale_image(x, y) = block->min_scale();
                }
                else {
                    scale_image(x, y) = -1;
                    min_scale_image(x, y) = -1;
                }
            }
            else {
                surface_point_cloud_M[pixel.x() + pixel.y() * surface_point_cloud_M.width()] =
                    Eigen::Vector3f::Zero();
                surface_normals_M[pixel.x() + pixel.y() * surface_normals_M.width()] =
                    Eigen::Vector3f(INVALID, 0.f, 0.f);
            }
        }
    }
    TOCK("raycastKernel");
}



void renderRGBAKernel(uint32_t* output_RGBA_image_data,
                      const Eigen::Vector2i& output_RGBA_image_res,
                      const se::Image<uint32_t>& input_RGBA_image);



void renderDepthKernel(uint32_t* depth_RGBA_image_data,
                       float* depth_image_data,
                       const Eigen::Vector2i& depth_RGBA_image_res,
                       const float near_plane,
                       const float far_plane);



void renderTrackKernel(uint32_t* tracking_RGBA_image_data,
                       const TrackData* tracking_result_data,
                       const Eigen::Vector2i& tracking_RGBA_image_res);



template<typename VoxelImplT>
void renderVolumeKernel(uint32_t* volume_RGBA_image_data,
                        const Eigen::Vector2i& volume_RGBA_image_res,
                        const Eigen::Vector3f& light_M,
                        const Eigen::Vector3f& ambient_M,
                        const se::Image<Eigen::Vector3f>& surface_point_cloud_M,
                        const se::Image<Eigen::Vector3f>& surface_normals_M,
                        const se::Image<int8_t>& scale_image,
                        const bool render_scale = false)
{
    TICKD("renderVolumeKernel");
    const int h = volume_RGBA_image_res.y(); // clang complains if this is inside the for loop
    const int w = volume_RGBA_image_res.x(); // clang complains if this is inside the for loop
#pragma omp parallel for
    for (int y = 0; y < h; y++) {
#pragma omp simd
        for (int x = 0; x < w; x++) {
            const size_t pixel_idx = x + w * y;

            const Eigen::Vector3f surface_point_M = surface_point_cloud_M[pixel_idx];
            const Eigen::Vector3f surface_normal_M = surface_normals_M[pixel_idx];

            if (surface_normal_M.x() != INVALID && surface_normal_M.norm() > 0.f) {
                const Eigen::Vector3f diff = (surface_point_M - light_M).normalized();
                const Eigen::Vector3f dir =
                    Eigen::Vector3f::Constant(fmaxf(surface_normal_M.normalized().dot(diff), 0.f));
                Eigen::Vector3f col = dir + ambient_M;
                se::math::clamp(col, Eigen::Vector3f::Zero(), Eigen::Vector3f::Ones());
                if (render_scale) {
                    col = col.cwiseProduct(se::internal::color_map[scale_image(x, y)]);
                }
                else {
                    col *= 255.0f;
                }
                volume_RGBA_image_data[pixel_idx] = se::pack_rgba(col.x(), col.y(), col.z(), 0xFF);
            }
            else {
                volume_RGBA_image_data[pixel_idx] = 0xFF000000;
            }
        }
    }
    TOCK("renderVolumeKernel");
}



inline void printNormals(const se::Image<Eigen::Vector3f>& normals, const char* filename);



// Find ALL the intersection along a ray till the far_plane.
template<typename VoxelT>
void raycast_full(
    const se::Octree<VoxelT>& map,
    std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f>>& surface_points_M,
    const Eigen::Vector3f& ray_origin_M,
    const Eigen::Vector3f& ray_dir_M,
    const float far_plane,
    const float step,
    const float large_step)
{
    float t = 0;
    float step_size = large_step;
    float f_t =
        map.interpAtPoint(ray_origin_M + ray_dir_M * t, [](const auto& data) { return data.x; })
            .first;
    t += step;
    float f_tt = 1.f;

    for (; t < far_plane; t += step_size) {
        f_tt =
            map.interpAtPoint(ray_origin_M + ray_dir_M * t, [](const auto& data) { return data.x; })
                .first;
        if (f_tt < 0.f && f_t > 0.f
            && std::abs(f_tt - f_t) < 0.5f) { // got it, jump out of inner loop
            typename VoxelT::VoxelData data_t;
            map.getAtPoint(ray_origin_M + ray_dir_M * (t - step_size), data_t);
            typename VoxelT::VoxelData data_tt;
            map.getAtPoint(ray_origin_M + ray_dir_M * t, data_tt);

            if (f_t == 1.0 || f_tt == 1.0 || data_t.y == 0 || data_tt.y == 0) {
                f_t = f_tt;
                continue;
            }
            t = t + step_size * f_tt / (f_t - f_tt);
            surface_points_M.push_back((ray_origin_M + ray_dir_M * t).homogeneous());
        }
        if (f_tt < std::abs(0.8f)) {
            // coming closer, reduce step_size
            step_size = step;
        }
        f_t = f_tt;
    }
}



// Semanticeight-only /////////////////////////////////////////////////////////
template<typename T>
void renderMaskKernel(uint32_t* output_image_data,
                      const Eigen::Vector2i& output_image_res,
                      const se::Image<uint32_t>& rgba_image,
                      const cv::Mat& mask,
                      const float mask_opacity = 0.5f)
{
    TICKD("renderMaskKernel");
    const bool valid_mask = (mask.cols != 0 && mask.rows != 0);
    assert((output_image_res.x() == rgba_image.width())
           && "Input and output image dimension mismatch");
    assert((output_image_res.y() == rgba_image.height())
           && "Input and output image dimension mismatch");
    assert((output_image_res.x() == mask.cols) && valid_mask
           && "Mask and output image dimension mismatch");
    assert((output_image_res.y() == mask.rows) && valid_mask
           && "Mask and output image dimension mismatch");

    const int h = output_image_res.y(); // clang complains if this is inside the for loop
    const int w = output_image_res.x(); // clang complains if this is inside the for loop
#pragma omp parallel for
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            const size_t pixel_idx = x + w * y;

            // Initialize color to pink for invalid.
            uint32_t new_color = 0xFF8080FF;

            const uint64_t mask_value = valid_mask ? mask.at<T>(y, x) : static_cast<T>(-1);
            if (mask_value < 0) {
                // Show RGB image.
                new_color = rgba_image[pixel_idx];
            }
            else if (mask_value == 0) {
                // Blend RGB image with black.
                new_color = se::blend(rgba_image[pixel_idx], 0xFF000000, 1.f - mask_opacity);
            }
            else {
                // Blend RGB image with color.
                const Eigen::Vector3i obj_color =
                    se::internal::color_map[mask_value % se::internal::color_map.size() - 1]
                        .cast<int>();
                new_color =
                    se::blend(rgba_image[pixel_idx], se::pack_rgba(obj_color), 1.f - mask_opacity);
            }
            output_image_data[pixel_idx] = new_color;
        }
    }
    TOCK("renderMaskKernel");
}

#endif
