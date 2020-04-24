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
#include "se/voxel_block_ray_iterator.hpp"
#include "se/algorithms/meshing.hpp"
#include "se/geometry/octree_collision.hpp"
#include "se/io/vtk-io.h"
#include "se/io/ply_io.hpp"
#include "se/algorithms/balancing.hpp"
#include "se/functors/for_each.hpp"
#include "se/timings.h"
#include "se/perfstats.h"
#include "se/rendering.hpp"


extern PerfStats Stats;
static bool print_kernel_timing = false;

DenseSLAMSystem::DenseSLAMSystem(const Eigen::Vector2i& input_size,
                                 const Eigen::Vector3i& volume_resolution,
                                 const Eigen::Vector3f& volume_dimensions,
                                 const Eigen::Vector3f& init_pose,
                                 std::vector<int> & pyramid,
                                 const Configuration& config):
      DenseSLAMSystem(input_size, volume_resolution, volume_dimensions,
          se::math::toMatrix4f(init_pose), pyramid, config) { }

DenseSLAMSystem::DenseSLAMSystem(const Eigen::Vector2i& input_size,
                                 const Eigen::Vector3i& volume_resolution,
                                 const Eigen::Vector3f& volume_dimensions,
                                 const Eigen::Matrix4f& init_T_WC,
                                 std::vector<int> & pyramid,
                                 const Configuration& config) :
  computation_size_(input_size),
  config_(config),
  sensor_({input_size.x(), input_size.y(), config.left_hand_frame,
        nearPlane, farPlane, config.mu,
        config.camera[0] / config.compute_size_ratio, config.camera[1] / config.compute_size_ratio,
        config.camera[2] / config.compute_size_ratio, config.camera[3] / config.compute_size_ratio,
        Eigen::VectorXf(0), Eigen::VectorXf(0)}),
  vertex_(computation_size_.x(), computation_size_.y()),
  normal_(computation_size_.x(), computation_size_.y()),
  float_depth_(computation_size_.x(), computation_size_.y()),
  rgba_(computation_size_.x(), computation_size_.y())
  {

    this->init_position_M_ = init_T_WC.block<3,1>(0,3);
    this->volume_dimension_ = volume_dimensions;
    this->volume_resolution_ = volume_resolution;
    this->mu_ = config.mu;
    T_WC_ = init_T_WC;
    raycast_T_WC_ = init_T_WC;

    this->iterations_.clear();
    for (std::vector<int>::iterator it = pyramid.begin();
        it != pyramid.end(); it++) {
      this->iterations_.push_back(*it);
    }

    render_T_WC_ = &T_WC_;

    if (getenv("KERNEL_TIMINGS"))
      print_kernel_timing = true;

    // internal buffers to initialize
    reduction_output_.resize(8 * 32);
    tracking_result_.resize(computation_size_.x() * computation_size_.y());

    for (unsigned int i = 0; i < iterations_.size(); ++i) {
      int downsample = 1 << i;
      scaled_depth_.push_back(se::Image<float>(computation_size_.x() / downsample,
            computation_size_.y() / downsample));

      input_vertex_.push_back(se::Image<Eigen::Vector3f>(computation_size_.x() / downsample,
            computation_size_.y() / downsample));

      input_normal_.push_back(se::Image<Eigen::Vector3f>(computation_size_.x() / downsample,
            computation_size_.y() / downsample));
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

    discrete_vol_ptr_ = std::make_shared<se::Octree<VoxelImpl::VoxelType> >();
    discrete_vol_ptr_->init(volume_resolution_.x(), volume_dimension_.x());
    volume_ = Volume<VoxelImpl>(volume_resolution_.x(), volume_dimension_.x(),
        discrete_vol_ptr_.get());
}



bool DenseSLAMSystem::preprocessDepth(const uint16_t*        input_depth,
                                      const Eigen::Vector2i& input_size,
                                      const bool             filter_depth){

  mm2metersKernel(float_depth_, input_depth, input_size);

  if (filter_depth) {
    bilateralFilterKernel(scaled_depth_[0], float_depth_, gaussian_,
        e_delta, radius);
  } else {
    std::memcpy(scaled_depth_[0].data(), float_depth_.data(),
        sizeof(float) * computation_size_.x() * computation_size_.y());
  }
  return true;
}



bool DenseSLAMSystem::preprocessColor(const uint8_t*         input_RGB,
                                      const Eigen::Vector2i& input_size) {

  downsampleImageKernel(input_RGB, input_size, rgba_);

  return true;
}



bool DenseSLAMSystem::track(float icp_threshold) {

  // half sample the input depth maps into the pyramid levels
  for (unsigned int i = 1; i < iterations_.size(); ++i) {
    halfSampleRobustImageKernel(scaled_depth_[i], scaled_depth_[i - 1], e_delta * 3, 1);
  }

  // prepare the 3D information from the input depth maps
  Eigen::Vector2i local_image_size = computation_size_;
  for (unsigned int i = 0; i < iterations_.size(); ++i) {
    float scaling_factor = 1.f / float(1 << i);
    SensorImpl scaled_sensor(sensor_, scaling_factor);
    depth2vertexKernel(input_vertex_[i], scaled_depth_[i], scaled_sensor);
    if(sensor_.left_hand_frame)
      vertex2normalKernel<true>(input_normal_[i], input_vertex_[i]);
    else
      vertex2normalKernel<false>(input_normal_[i], input_vertex_[i]);
    local_image_size /= 2;
  }

  previous_T_WC_ = T_WC_;

  for (int level = iterations_.size() - 1; level >= 0; --level) {
    Eigen::Vector2i local_image_size(
        computation_size_.x() / (int) pow(2, level),
        computation_size_.y() / (int) pow(2, level));
    for (int i = 0; i < iterations_[level]; ++i) {

      trackKernel(tracking_result_.data(), input_vertex_[level], input_normal_[level],
          vertex_, normal_, T_WC_, sensor_, dist_threshold, normal_threshold);

      reduceKernel(reduction_output_.data(), tracking_result_.data(), computation_size_,
          local_image_size);

      if (updatePoseKernel(T_WC_, reduction_output_.data(), icp_threshold))
        break;

    }
  }
  return checkPoseKernel(T_WC_, previous_T_WC_, reduction_output_.data(),
      computation_size_, track_threshold);
}



bool DenseSLAMSystem::integrate(unsigned int frame) {

  const float voxel_size = volume_.dim() / volume_.size();
  const int num_vox_per_pix = volume_.dim()
    / ((se::VoxelBlock<VoxelImpl::VoxelType>::side) * voxel_size);
  const size_t total = num_vox_per_pix
    * computation_size_.x() * computation_size_.y();
  allocation_list_.reserve(total);

  const Sophus::SE3f& T_CW = Sophus::SE3f(T_WC_).inverse();
  const size_t allocated = VoxelImpl::buildAllocationList(
      allocation_list_.data(),
      allocation_list_.capacity(),
      *volume_.octree_,
      T_WC_,
      sensor_,
      float_depth_);

  volume_.octree_->allocate(allocation_list_.data(), allocated);

  VoxelImpl::integrate(
      *volume_.octree_,
      T_CW,
      float_depth_,
      sensor_,
      frame);
  return true;
}



bool DenseSLAMSystem::raycast() {

  raycast_T_WC_ = T_WC_;
  float step = volume_dimension_.x() / volume_resolution_.x();
  raycastKernel(volume_, vertex_, normal_, raycast_T_WC_, sensor_, step, step*BLOCK_SIDE);

  return true;
}



void DenseSLAMSystem::dump_volume(std::string ) {

}

void DenseSLAMSystem::renderVolume(unsigned char*         output,
                                   const Eigen::Vector2i& output_size) {

  const float step = volume_dimension_.x() / volume_resolution_.x();
  const float large_step = 0.75 * sensor_.mu;
  renderVolumeKernel(volume_, output, output_size,
      *(this->render_T_WC_), sensor_, step, large_step,
      this->render_T_WC_->topRightCorner<3, 1>(), ambient,
      !(this->render_T_WC_->isApprox(raycast_T_WC_)), vertex_,
      normal_);
}

void DenseSLAMSystem::renderTrack(unsigned char* out,
    const Eigen::Vector2i& output_size) {
        renderTrackKernel(out, tracking_result_.data(), output_size);
}

void DenseSLAMSystem::renderDepth(unsigned char* out,
    const Eigen::Vector2i& output_size) {
        renderDepthKernel(out, float_depth_.data(), output_size, sensor_.near_plane, sensor_.far_plane);
}



void DenseSLAMSystem::renderRGBA(uint8_t*               output_RGBA,
                                 const Eigen::Vector2i& output_size) {

  renderRGBAKernel(output_RGBA, output_size, rgba_);
}



void DenseSLAMSystem::dump_mesh(const std::string filename){

//  se::functor::internal::parallel_for_each(volume_.octree_->pool().blockBuffer(),
//      [](auto block) {
//        if(std::is_same<VoxelImpl, MultiresTSDF>::value) {
//          block->current_scale(block->min_scale());
//        } else {
//          block->current_scale(0);
//        }
//      });

  auto interp_down = [this](auto block) {
    if(block->min_scale() == 0) return;
    const Eigen::Vector3f& offset = this->volume_.octree_->_offset;
    const Eigen::Vector3i base = block->coordinates();
    const int side = block->side;
    for(int z = 0; z < side; ++z)
      for(int y = 0; y < side; ++y)
        for(int x = 0; x < side; ++x) {
          const Eigen::Vector3i vox = base + Eigen::Vector3i(x, y , z);
          auto curr = block->data(vox, 0);
          auto res = this->volume_.octree_->interp_checked(
              vox.cast<float>() + offset, 0, [](const auto& val) { return val.x; });
          if(res.second >= 0) {
            curr.x = res.first;
            curr.y = this->volume_.octree_->interp(
                vox.cast<float>() + offset, [](const auto& val) { return val.y; }).first;
          } else {
            curr.y = 0;
          }
          block->data(vox, 0, curr);
        }
  };

  se::functor::internal::parallel_for_each(volume_.octree_->pool().blockBuffer(),
      interp_down);
  se::functor::internal::parallel_for_each(volume_.octree_->pool().blockBuffer(),
      [](auto block) {
          block->current_scale(0);
      });

    std::cout << "saving triangle mesh to file :" << filename  << std::endl;

    std::vector<Triangle> mesh;
    auto inside = [](const VoxelImpl::VoxelType::VoxelData& val) {
      return val.x < 0.f;
    };

    auto select = [](const VoxelImpl::VoxelType::VoxelData& val) {
      return val.x;
    };

    se::algorithms::marching_cube(*volume_.octree_, select, inside, mesh);
    writeVtkMesh(filename.c_str(), mesh, this->init_position_M_);
}
