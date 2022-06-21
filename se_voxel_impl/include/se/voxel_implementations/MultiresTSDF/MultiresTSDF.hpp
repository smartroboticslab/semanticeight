/*
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

#ifndef __MULTIRESTSDF_HPP
#define __MULTIRESTSDF_HPP

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "se/algorithms/meshing.hpp"
#include "se/image/image.hpp"
#include "se/octree.hpp"
#include "se/sensor_implementation.hpp"

/** Kinect Fusion Truncated Signed Distance Function voxel implementation for
 * integration at multiple scales. */
struct MultiresTSDF {
    typedef int16_t tsdf_t;
    typedef uint8_t weight_t;
    typedef uint8_t fg_t;

    /**
     * The voxel type used as the template parameter for se::Octree.
     */
    struct VoxelType {
        /**
         * The struct stored in each se::Octree voxel.
         */
        struct VoxelData {
            tsdf_t x; /**< The value of the TSDF. */
            tsdf_t x_last;
            weight_t y;
            weight_t delta_y;
            fg_t fg;           // Foreground probability
            weight_t fg_count; // Foreground probability update count
            uint8_t r;         // Red channel
            uint8_t g;         // Green channel
            uint8_t b;         // Blue channel

            bool operator==(const VoxelData& other) const;
            bool operator!=(const VoxelData& other) const;

            inline float getTsdf() const
            {
                return x / static_cast<float>(std::numeric_limits<tsdf_t>::max());
            }

            inline void setTsdf(float tsdf)
            {
                x = tsdf * static_cast<float>(std::numeric_limits<tsdf_t>::max());
            }

            inline float getTsdfLast() const
            {
                return x / static_cast<float>(std::numeric_limits<tsdf_t>::max());
            }

            inline void setTsdfLast(float tsdf)
            {
                x = tsdf * static_cast<float>(std::numeric_limits<tsdf_t>::max());
            }

            inline float getFg() const
            {
                return fg / static_cast<float>(std::numeric_limits<fg_t>::max());
            }

            inline void setFg(float fg_prob)
            {
                fg = fg_prob * std::numeric_limits<fg_t>::max();
            }
        };

        static inline VoxelData invalid()
        {
            return {std::numeric_limits<tsdf_t>::max(),
                    std::numeric_limits<tsdf_t>::max(),
                    0u,
                    0u,
                    0u,
                    0u,
                    0u,
                    0u,
                    0u};
        }
        static inline VoxelData initData()
        {
            return {std::numeric_limits<tsdf_t>::max(),
                    std::numeric_limits<tsdf_t>::max(),
                    0u,
                    0u,
                    0u,
                    0u,
                    0u,
                    0u,
                    0u};
        }

        static float selectNodeValue(const VoxelData& /* data */)
        {
            return VoxelType::initData().getTsdf();
        };

        static float selectVoxelValue(const VoxelData& data)
        {
            return data.getTsdf();
        };

        static bool isInside(const VoxelData& data)
        {
            return data.x < static_cast<tsdf_t>(0);
        };

        static bool isValid(const VoxelData& data)
        {
            return data.y > 0;
        };

        using VoxelBlockType = se::VoxelBlockFull<MultiresTSDF::VoxelType>;

        using MemoryPoolType = se::PagedMemoryPool<MultiresTSDF::VoxelType>;
        template<typename ElemT>
        using MemoryBufferType = se::PagedMemoryBuffer<ElemT>;
    };

    using VoxelData = MultiresTSDF::VoxelType::VoxelData;
    using OctreeType = se::Octree<MultiresTSDF::VoxelType>;
    using VoxelBlockType = typename MultiresTSDF::VoxelType::VoxelBlockType;

    /**
   * The normals must be inverted when rendering a TSDF map.
   */
    static constexpr bool invert_normals = true;

    /**
   * The factor the voxel dim is multiplied with to compute mu
   *
   *  <br>\em Default: 8
   */
    static float mu_factor;

    /**
   * The MultiresTSDF truncation bound. Values of the MultiresTSDF are assumed to be in the
   * interval ±mu. See Section 3.3 of \cite NewcombeISMAR2011 for more
   * details.
   *  <br>\em Default: 0.1
   */
    static float mu;

    /**
   * The maximum value of the weight factor
   * MultiresTSDF::VoxelType::VoxelData::y.
   */
    static weight_t max_weight;

    static std::string type()
    {
        return "multirestsdf";
    }

    /**
   * Configure the MultiresTSDF parameters
   */
    static void configure(const float voxel_dim);
    static void configure(YAML::Node yaml_config, const float voxel_dim);

    static std::string printConfig();

    /**
   * Compute the VoxelBlocks and Nodes that need to be allocated given the
   * camera pose.
   */
    static size_t buildAllocationList(OctreeType& map,
                                      const se::Image<float>& depth_image,
                                      const Eigen::Matrix4f& T_MC,
                                      const SensorImpl& sensor,
                                      se::key_t* allocation_list,
                                      size_t reserved);



    /**
   * Integrate a depth image into the map.
   */
    static void integrate(OctreeType& map,
                          const se::Image<float>& depth_image,
                          const se::Image<uint32_t>& rgba_image,
                          const cv::Mat& fg_image,
                          const Eigen::Matrix4f& T_CM,
                          const SensorImpl& sensor,
                          const unsigned frame);



    /**
     * Cast a ray and return the point where the surface was hit.
     */
    static Eigen::Vector4f raycast(const OctreeType& map,
                                   const Eigen::Vector3f& ray_origin_M,
                                   const Eigen::Vector3f& ray_dir_M,
                                   const float t_near,
                                   const float t_far);

    /**
     * Cast a ray and return the point where the back of the surface was hit. Unline
     * MultiresTSDF::raycast(), no interpolation is performed since the only thing we care about is
     * that the back of the surface was hit by this ray.
     */
    static Eigen::Vector4f raycastBackFace(const OctreeType& map,
                                           const Eigen::Vector3f& ray_origin_M,
                                           const Eigen::Vector3f& ray_dir_M,
                                           const float t_near,
                                           const float t_far);

    static void dumpMesh(OctreeType& map,
                         std::vector<se::Triangle>& mesh,
                         se::meshing::ScaleMode scale_mode = se::meshing::ScaleMode::Current);
};

#endif
