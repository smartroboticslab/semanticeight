/*
 *
 * Copyright 2016 Emanuele Vespa, Imperial College London
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

#include "se/voxel_implementations/TSDF/TSDF.hpp"

#include "se/utils/math_utils.h"
#include "se/node.hpp"
#include "se/utils/morton_utils.hpp"



/*
 * \brief Given a depth map and camera matrix it computes the list of
 * voxels intersected but not allocated by the rays around the measurement m in
 * a region comprised between m +/- band.
 * \param map indexing structure used to index voxel blocks
 * \param T_wc camera to world frame transformation
 * \param sensor model
 * \param size discrete extent of the map, in number of voxels
 * \param allocation_list output list of keys corresponding to voxel blocks to
 * be allocated
 * \param reserved allocated size of allocation_list
 */
size_t TSDF::buildAllocationList(
    se::Octree<TSDF::VoxelType>& map,
    const se::Image<float>&      depth_image,
    const Eigen::Matrix4f&       T_WC,
    const SensorImpl&            sensor,
    se::key_t*                   allocation_list,
    size_t                       reserved) {

  const Eigen::Vector2i image_size(depth_image.width(), depth_image.height());
  const float voxel_size = map.dim() / map.size();
  const float inverse_voxel_size = 1.f / voxel_size;
  const int volume_size = map.size();
  const unsigned block_depth = map.blockDepth();
  const float band = 2.f * sensor.mu;



#ifdef _OPENMP
  std::atomic<unsigned int> voxel_count (0);
#else
  unsigned int voxel_count = 0;
#endif

  const Eigen::Vector3f t_WC = T_WC.topRightCorner<3, 1>();
  const int num_steps = ceil(band * inverse_voxel_size);
#pragma omp parallel for
  for (int y = 0; y < image_size.y(); ++y) {
    for (int x = 0; x < image_size.x(); ++x) {
      if (depth_image[x + y*image_size.x()] == 0.f)
        continue;

      const float depth_value = depth_image[x + y * image_size.x()];

      Eigen::Vector3f ray_direction_C;
      const Eigen::Vector2f image_point(x + 0.5f,y + 0.5f);
      sensor.model.backProject(image_point, &ray_direction_C);
      const Eigen::Vector3f surface_vertex_W = (T_WC * (depth_value * ray_direction_C).homogeneous()).head<3>();

      const Eigen::Vector3f reverse_ray_direction_W = (t_WC - surface_vertex_W).normalized();

      const Eigen::Vector3f ray_origin_W = surface_vertex_W - (band * 0.5f) * reverse_ray_direction_W;
      const Eigen::Vector3f step = (reverse_ray_direction_W * band) / num_steps;

      Eigen::Vector3f ray_position_W = ray_origin_W;
      for (int i = 0; i < num_steps; i++) {

        const Eigen::Vector3i voxel_W = (ray_position_W * inverse_voxel_size).cast<int>();
        if (   (voxel_W.x() < volume_size)
            && (voxel_W.y() < volume_size)
            && (voxel_W.z() < volume_size)
            && (voxel_W.x() >= 0)
            && (voxel_W.y() >= 0)
            && (voxel_W.z() >= 0)) {
          se::VoxelBlock<TSDF::VoxelType> * node_ptr = map.fetch(
              voxel_W.x(), voxel_W.y(), voxel_W.z());
          if (node_ptr == nullptr) {
            const se::key_t voxel_key = map.hash(voxel_W.x(), voxel_W.y(), voxel_W.z(),
                block_depth);
            const unsigned int idx = voxel_count++;
            if (idx < reserved) {
              allocation_list[idx] = voxel_key;
            } else {
              break;
            }
          } else {
            node_ptr->active(true);
          }
        }
        ray_position_W += step;
      }
    }
  }
  const size_t written = voxel_count;
  return written >= reserved ? reserved : written;
}

