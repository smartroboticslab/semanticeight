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

#include "se/DenseSLAMSystem.h"

#include <cstring>

#include "se/voxel_block_ray_iterator.hpp"
#include "se/algorithms/meshing.hpp"
#include "se/io/meshing_io.hpp"
#include "se/io/octree_io.hpp"
#include "se/geometry/octree_collision.hpp"
#include "se/algorithms/balancing.hpp"
#include "se/functors/for_each.hpp"
#include "se/timings.h"
#include "se/perfstats.h"
#include "se/rendering.hpp"


extern PerfStats stats;
static bool print_kernel_timing = false;

DenseSLAMSystem::DenseSLAMSystem(const Eigen::Vector2i& image_res,
                                 const Eigen::Vector3i& map_size,
                                 const Eigen::Vector3f& map_dim,
                                 const Eigen::Vector3f& t_MW,
                                 std::vector<int> & pyramid,
                                 const Configuration& config):
      DenseSLAMSystem(image_res, map_size, map_dim,
          se::math::to_transformation(t_MW), pyramid, config) { }

DenseSLAMSystem::DenseSLAMSystem(const Eigen::Vector2i& image_res,
                                 const Eigen::Vector3i& map_size,
                                 const Eigen::Vector3f& map_dim,
                                 const Eigen::Matrix4f& T_MW,
                                 std::vector<int> & pyramid,
                                 const Configuration& config) :
  image_res_(image_res),
  config_(config),
  surface_point_cloud_M_(image_res_.x(), image_res_.y()),
  surface_normals_M_(image_res_.x(), image_res_.y()),
  depth_image_(image_res_.x(), image_res_.y()),
  rgba_image_(image_res_.x(), image_res_.y())
  {
    // Initalise poses
    T_MW_ = T_MW;
    T_MC_ = T_MW;
    init_T_MC_ = T_MC_;
    raycast_T_MC_ = T_MC_;
    render_T_MC_ =  &T_MC_;
    this->map_dim_ = map_dim;
    this->map_size_ = map_size;

    this->iterations_.clear();
    for(std::vector<int>::iterator it = pyramid.begin();
        it != pyramid.end(); it++) {
      this->iterations_.push_back(*it);
    }

    if (getenv("KERNEL_TIMINGS"))
      print_kernel_timing = true;

    // internal buffers to initialize
    reduction_output_.resize(8 * 32);
    tracking_result_.resize(image_res_.x() * image_res_.y());

    for (unsigned int i = 0; i < iterations_.size(); ++i) {
      int downsample = 1 << i;
      scaled_depth_image_.push_back(se::Image<float>(image_res_.x() / downsample,
            image_res_.y() / downsample));

      input_point_cloud_C_.push_back(se::Image<Eigen::Vector3f>(image_res_.x() / downsample,
            image_res_.y() / downsample));

      input_normals_C_.push_back(se::Image<Eigen::Vector3f>(image_res_.x() / downsample,
            image_res_.y() / downsample));
    }

    // ********* BEGIN : Generate the gaussian *************
    size_t gaussianS = radius * 2 + 1;
    gaussian_.reserve(gaussianS);
    int x;
    for (unsigned int i = 0; i < gaussianS; i++) {
      x = i - 2;
      gaussian_[i] = expf(-(x * x) / (2 * delta * delta));
    }

    // ********* END : Generate the gaussian *************

    map_ = std::make_shared<se::Octree<VoxelImpl::VoxelType> >();
    map_->init(map_size_.x(), map_dim_.x());
}





bool DenseSLAMSystem::preprocessDepth(const float*           input_depth_image_data,
                                      const Eigen::Vector2i& input_depth_image_res,
                                      const bool             filter_depth){

  mm2metersKernel(depth_image_, input_depth_image_data, input_depth_image_res);

  if (filter_depth) {
    bilateralFilterKernel(scaled_depth_image_[0], depth_image_, gaussian_,
        e_delta, radius);
  } else {
    std::memcpy(scaled_depth_image_[0].data(), depth_image_.data(),
        sizeof(float) * image_res_.x() * image_res_.y());
  }
  return true;
}



bool DenseSLAMSystem::preprocessColor(const uint32_t*        input_RGBA_image_data,
                                      const Eigen::Vector2i& input_RGBA_image_res) {

  downsampleImageKernel(input_RGBA_image_data, input_RGBA_image_res, rgba_image_);

  return true;
}



bool DenseSLAMSystem::track(const SensorImpl& sensor,
                            const float       icp_threshold) {

  // half sample the input depth maps into the pyramid levels
  for (unsigned int i = 1; i < iterations_.size(); ++i) {
    halfSampleRobustImageKernel(scaled_depth_image_[i], scaled_depth_image_[i - 1], e_delta * 3, 1);
  }

  // prepare the 3D information from the input depth maps
  for (unsigned int i = 0; i < iterations_.size(); ++i) {
    const float scaling_factor = 1.f / (1 << i);
    const SensorImpl scaled_sensor(sensor, scaling_factor);
    depthToPointCloudKernel(input_point_cloud_C_[i], scaled_depth_image_[i], scaled_sensor);
    if(sensor.left_hand_frame)
      pointCloudToNormalKernel<true>(input_normals_C_[i], input_point_cloud_C_[i]);
    else
      pointCloudToNormalKernel<false>(input_normals_C_[i], input_point_cloud_C_[i]);
  }

  previous_T_MC_ = T_MC_;

  for (int level = iterations_.size() - 1; level >= 0; --level) {
    Eigen::Vector2i reduction_output_res(
        image_res_.x() / (int) pow(2, level),
        image_res_.y() / (int) pow(2, level));
    for (int i = 0; i < iterations_[level]; ++i) {

      trackKernel(tracking_result_.data(), input_point_cloud_C_[level], input_normals_C_[level],
                  surface_point_cloud_M_, surface_normals_M_, T_MC_, sensor, dist_threshold, normal_threshold);

      reduceKernel(reduction_output_.data(), reduction_output_res, tracking_result_.data(), image_res_);

      if (updatePoseKernel(T_MC_, reduction_output_.data(), icp_threshold))
        break;

    }
  }
  return checkPoseKernel(T_MC_, previous_T_MC_, reduction_output_.data(),
      image_res_, track_threshold);
}



bool DenseSLAMSystem::integrate(const SensorImpl&  sensor,
                                const unsigned     frame) {

  const int num_blocks_per_pixel = map_->size()
    / ((VoxelBlockType::size_li));
  const size_t num_blocks_total = num_blocks_per_pixel
    * image_res_.x() * image_res_.y();
  allocation_list_.reserve(num_blocks_total);

  const Eigen::Matrix4f T_CM = se::math::to_inverse_transformation(T_MC_); // TODO:
  const size_t num_voxel = VoxelImpl::buildAllocationList(
      *map_,
      depth_image_,
      T_MC_,
      sensor,
      allocation_list_.data(),
      allocation_list_.capacity());

  map_->allocate(allocation_list_.data(), num_voxel);

  VoxelImpl::integrate(
      *map_,
      depth_image_,
      T_CM,
      sensor,
      frame);
  return true;
}



bool DenseSLAMSystem::raycast(const SensorImpl& sensor) {

  raycast_T_MC_ = T_MC_;
  raycastKernel<VoxelImpl>(*map_, surface_point_cloud_M_, surface_normals_M_,
      raycast_T_MC_, sensor);
  return true;
}



void DenseSLAMSystem::dump_volume(std::string ) {

}

void DenseSLAMSystem::renderVolume(unsigned char*         volume_RGBA_image_data,
                                   const Eigen::Vector2i& volume_RGBA_image_res,
                                   const SensorImpl&      sensor) {
  
  se::Image<Eigen::Vector3f> render_surface_point_cloud_M (image_res_.x(), image_res_.y());
  se::Image<Eigen::Vector3f> render_surface_normals_M (image_res_.x(), image_res_.y());
  if (render_T_MC_->isApprox(raycast_T_MC_)) {
    // Copy the raycast from the camera viewpoint. Can't safely use memcpy with
    // Eigen objects it seems.
    for (size_t i = 0; i < surface_point_cloud_M_.size(); ++i) {
      render_surface_point_cloud_M[i] = surface_point_cloud_M_[i];
      render_surface_normals_M[i] = surface_normals_M_[i];
    }
  } else {
    // Raycast the map from the render viewpoint.
    raycastKernel<VoxelImpl>(*map_, render_surface_point_cloud_M,
        render_surface_normals_M, *render_T_MC_, sensor);
  }
  renderVolumeKernel<VoxelImpl>(volume_RGBA_image_data, volume_RGBA_image_res,
      se::math::to_translation(*render_T_MC_), ambient,
      render_surface_point_cloud_M, render_surface_normals_M);
}

void DenseSLAMSystem::renderTrack(unsigned char*         tracking_RGBA_image_data,
                                  const Eigen::Vector2i& tracking_RGBA_image_res) {
  renderTrackKernel(tracking_RGBA_image_data, tracking_result_.data(), tracking_RGBA_image_res);
}

void DenseSLAMSystem::renderDepth(unsigned char*         depth_RGBA_image_data,
                                  const Eigen::Vector2i& depth_RGBA_image_res,
                                  const SensorImpl&      sensor) {
  renderDepthKernel(depth_RGBA_image_data, depth_image_.data(), depth_RGBA_image_res,
      sensor.near_plane, sensor.far_plane);
}



void DenseSLAMSystem::renderRGBA(uint8_t*               output_RGBA_image_data,
                                 const Eigen::Vector2i& output_RGBA_image_res) {

  renderRGBAKernel(output_RGBA_image_data, output_RGBA_image_res, rgba_image_);
}



void DenseSLAMSystem::dump_mesh(const std::string filename){

  se::functor::internal::parallel_for_each(map_->pool().blockBuffer(),
      [](auto block) {
        if(std::is_same<VoxelImpl, MultiresTSDF>::value) {
          block->current_scale(block->min_scale());
        } else {
          block->current_scale(0);
        }
      });

  auto interp_down = [this](auto block) {
    if(block->min_scale() == 0) return;
    const Eigen::Vector3f& sample_offset_frac = map_->sample_offset_frac_;
    const Eigen::Vector3i block_coord = block->coordinates();
    const int block_size = VoxelBlockType::size_li;
    bool is_valid;
    for(int z = 0; z < block_size; ++z)
      for(int y = 0; y < block_size; ++y)
        for(int x = 0; x < block_size; ++x) {
          const Eigen::Vector3i voxel_coord = block_coord + Eigen::Vector3i(x, y , z);
          auto voxel_data = block->data(voxel_coord, 0);
          auto voxel_value = (map_->interp(
              se::get_sample_coord(voxel_coord, 1, sample_offset_frac),
              [](const auto& data) { return data.x; }, 0, is_valid)).first;
          if(is_valid) {
            voxel_data.x = voxel_value;
            voxel_data.y = map_->interp(
                se::get_sample_coord(voxel_coord, 1, sample_offset_frac),
                [](const auto& data) { return data.y; }).first;
          } else {
            voxel_data.y = 0;
          }
          block->setData(voxel_coord, 0, voxel_data);
        }
  };

  se::functor::internal::parallel_for_each(map_->pool().blockBuffer(),
      interp_down);
  se::functor::internal::parallel_for_each(map_->pool().blockBuffer(),
      [](auto block) {
          block->current_scale(0);
      });

    std::cout << "Saving triangle mesh to file :" << filename  << std::endl;

    std::vector<se::Triangle> mesh;
    auto inside = [](const VoxelImpl::VoxelType::VoxelData& data) {
      return data.x < 0.f;
    };

    auto select_value = [](const VoxelImpl::VoxelType::VoxelData& data) {
      return data.x;
    };

    se::algorithms::marching_cube(*map_, select_value, inside, mesh);
    save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(this->T_MW_));
}

void DenseSLAMSystem::dump_dual_mesh(const std::string filename){

  std::cout << "Saving triangle mesh to file :" << filename  << std::endl;

  std::vector<se::Triangle> mesh;
  auto inside = [](const VoxelImpl::VoxelType::VoxelData& data) {
    return data.x < 0.f;
  };

  auto select_value = [](const VoxelImpl::VoxelType::VoxelData& data) {
    return data.x;
  };

  se::algorithms::dual_marching_cube(*map_, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(this->T_MW_));
}