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

#include<Eigen/StdVector>

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
#include "se/set_operations.hpp"
#include "se/voxel_implementations/MultiresOFusion/updating_model.hpp"
#include "se/frontiers.hpp"
#include "se/semanticeight_definitions.hpp"
#include "se/depth_utils.hpp"
#include "se/object_utils.hpp"
#include "se/exploration_utils.hpp"

extern PerfStats stats;

DenseSLAMSystem::DenseSLAMSystem(const Eigen::Vector2i&   image_res,
                                 const Eigen::Vector3i&   map_size,
                                 const Eigen::Vector3f&   map_dim,
                                 const Eigen::Vector3f&   t_MW,
                                 std::vector<int>&        pyramid,
                                 const se::Configuration& config,
                                 const std::string        voxel_impl_yaml_path)
  : DenseSLAMSystem(image_res, map_size, map_dim,
      se::math::to_transformation(t_MW), pyramid, config, voxel_impl_yaml_path) {}

DenseSLAMSystem::DenseSLAMSystem(const Eigen::Vector2i&   image_res,
                                 const Eigen::Vector3i&   map_size,
                                 const Eigen::Vector3f&   map_dim,
                                 const Eigen::Matrix4f&   T_MW,
                                 std::vector<int>&        pyramid,
                                 const se::Configuration& config,
                                 const std::string        voxel_impl_yaml_path)
  : image_res_(image_res),
    depth_image_(image_res_.x(), image_res_.y()),
    rgba_image_(image_res_.x(), image_res_.y()),
    map_dim_(map_dim),
    map_size_(map_size),
    config_(config),
    init_T_MC_(T_MW),
    T_MC_(init_T_MC_),
    previous_T_MC_(T_MC_),
    iterations_(pyramid),
    reduction_output_(8 * 32, 0.0f),
    tracking_result_(image_res_.prod(), TrackData()),
    raycast_T_MC_(T_MC_),
    surface_point_cloud_M_(image_res_.x(), image_res_.y(), Eigen::Vector3f::Zero()),
    surface_normals_M_(image_res_.x(), image_res_.y(), Eigen::Vector3f::Zero()),
    render_T_MC_(&T_MC_),
    T_MW_(T_MW),
    T_WM_(se::math::to_inverse_transformation(T_MW_)),
    input_segmentation_(image_res_.x(), image_res_.y()),
    processed_segmentation_(image_res_.x(), image_res_.y()),
    object_surface_point_cloud_M_(image_res_.x(), image_res_.y()),
    object_surface_normals_M_(image_res_.x(), image_res_.y()),
    object_scale_image_(image_res_.x(), image_res_.y(), -1),
    object_min_scale_image_(image_res_.x(), image_res_.y(), -1)
  {

    bool has_yaml_voxel_impl_config = false;
    YAML::Node yaml_voxel_impl_config = YAML::Load("");

    if (voxel_impl_yaml_path != "") {
      if (YAML::LoadFile(voxel_impl_yaml_path)["voxel_impl"]) {
        yaml_voxel_impl_config = YAML::LoadFile(voxel_impl_yaml_path)["voxel_impl"];
        has_yaml_voxel_impl_config = true;
      }
    }

    const float voxel_dim = map_dim_.x() / map_size_.x();
    if (has_yaml_voxel_impl_config) {
      VoxelImpl::configure(yaml_voxel_impl_config, voxel_dim);
      ObjVoxelImpl::configure(yaml_voxel_impl_config, voxel_dim);
    } else {
      VoxelImpl::configure(voxel_dim);
      ObjVoxelImpl::configure(voxel_dim);
    }

    // Initialize the Gaussian for the bilateral filter
    constexpr int gaussian_size = gaussian_radius * 2 + 1;
    gaussian_.reserve(gaussian_size);
    for (int i = 0; i < gaussian_size; i++) {
      const int x = i - 2;
      gaussian_[i] = expf(-(x * x) / (2 * delta * delta));
    }

    // Initialize the scaled images
    for (unsigned int i = 0; i < iterations_.size(); ++i) {
      const int downsample = 1 << i;
      const Eigen::Vector2i res = image_res_ / downsample;
      scaled_depth_image_.emplace_back(res.x(), res.y(), 0.0f);
      input_point_cloud_C_.emplace_back(res.x(), res.y(), Eigen::Vector3f::Zero());
      input_normals_C_.emplace_back(res.x(), res.y(), Eigen::Vector3f::Zero());
    }

    // Initialize the map
    map_ = std::shared_ptr<se::Octree<VoxelImpl::VoxelType> >(new se::Octree<VoxelImpl::VoxelType>());
    map_->init(map_size_.x(), map_dim_.x());

    // Semanticeight-only /////////////////////////////////////////////////////
    valid_depth_mask_ = cv::Mat(cv::Size(image_res_.x(), image_res_.y()), se::mask_t, cv::Scalar(255));
    raycasted_instance_mask_ = cv::Mat(cv::Size(image_res_.x(), image_res_.y()),
        se::instance_mask_t, cv::Scalar(se::instance_bg));
}



bool DenseSLAMSystem::preprocessDepth(const float*           input_depth_image_data,
                                      const Eigen::Vector2i& input_depth_image_res,
                                      const bool             filter_depth){
  TICKD("preprocessDepth")
  downsampleDepthKernel(input_depth_image_data, input_depth_image_res, depth_image_);

  if (filter_depth) {
    bilateralFilterKernel(scaled_depth_image_[0], depth_image_, gaussian_,
        e_delta, gaussian_radius);
  } else {
    std::memcpy(scaled_depth_image_[0].data(), depth_image_.data(),
        sizeof(float) * image_res_.x() * image_res_.y());
  }
  TOCK("preprocessDepth")
  return true;
}



bool DenseSLAMSystem::preprocessColor(const uint32_t*        input_RGBA_image_data,
                                      const Eigen::Vector2i& input_RGBA_image_res) {

  TICKD("preprocessColor")
  downsampleImageKernel(input_RGBA_image_data, input_RGBA_image_res, rgba_image_);
  TOCK("preprocessColor")
  return true;
}



bool DenseSLAMSystem::track(const SensorImpl& sensor,
                            const float       icp_threshold) {

  TICK("TRACKING")
  // half sample the input depth maps into the pyramid levels
  for (unsigned int i = 1; i < iterations_.size(); ++i) {
    halfSampleRobustImageKernel(scaled_depth_image_[i], scaled_depth_image_[i - 1], e_delta * 3, 1);
  }

  // prepare the 3D information from the input depth maps
  for (unsigned int i = 0; i < iterations_.size(); ++i) {
    const float scaling_factor = 1.f / (1 << i);
    const SensorImpl scaled_sensor(sensor, scaling_factor);
    depthToPointCloudKernel(input_point_cloud_C_[i], scaled_depth_image_[i], scaled_sensor);
    if(sensor.left_hand_frame) {
      pointCloudToNormalKernel<true>(input_normals_C_[i], input_point_cloud_C_[i]);
    }
    else {
      pointCloudToNormalKernel<false>(input_normals_C_[i], input_point_cloud_C_[i]);
    }
  }

  previous_T_MC_ = T_MC_;

  for (int level = iterations_.size() - 1; level >= 0; --level) {
    Eigen::Vector2i reduction_output_res(
        image_res_.x() / (int) pow(2, level),
        image_res_.y() / (int) pow(2, level));
    for (int i = 0; i < iterations_[level]; ++i) {

      trackKernel(tracking_result_.data(), input_point_cloud_C_[level], input_normals_C_[level],
          surface_point_cloud_M_, surface_normals_M_, T_MC_, raycast_T_MC_, sensor, dist_threshold, normal_threshold);

      reduceKernel(reduction_output_.data(), reduction_output_res, tracking_result_.data(), image_res_);

      if (updatePoseKernel(T_MC_, reduction_output_.data(), icp_threshold))
        break;

    }
  }
  TOCK("TRACKING")
  return checkPoseKernel(T_MC_, previous_T_MC_, reduction_output_.data(),
      image_res_, track_threshold);
}



bool DenseSLAMSystem::integrate(const SensorImpl&  sensor,
                                const unsigned     frame) {

  TICK("INTEGRATION")
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

  if (num_voxel > 0) {
    TICKD("allocate")
    map_->allocate(allocation_list_.data(), num_voxel);
    TOCK("allocate")
  }

  VoxelImpl::integrate(
      *map_,
      depth_image_,
      rgba_image_,
      cv::Mat(image_res_.x(), image_res_.y(), se::integration_mask_t, cv::Scalar(0.0f)),
      T_CM,
      sensor,
      frame,
      &updated_nodes_);
  TOCK("INTEGRATION")
  TICKD("FRONTIERS")
  se::setunion(frontiers_, updated_nodes_);
  update_frontiers(*map_, frontiers_, min_frontier_volume_);
  TOCK("FRONTIERS")
  // Update the free/occupied volume.
  se::ExploredVolume ev (*map_);
  free_volume = ev.free_volume;
  occupied_volume = ev.occupied_volume;
  explored_volume = ev.explored_volume;
  return true;
}



bool DenseSLAMSystem::raycast(const SensorImpl& sensor) {

  TICK("RAYCASTING")
  raycast_T_MC_ = T_MC_;
  raycastKernel<VoxelImpl>(*map_, surface_point_cloud_M_, surface_normals_M_,
      raycast_T_MC_, sensor);
  TOCK("RAYCASTING")
  return true;
}



void DenseSLAMSystem::renderVolume(uint32_t*              volume_RGBA_image_data,
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
    TICK("RAYCASTING")
    // Raycast the map from the render viewpoint.
    raycastKernel<VoxelImpl>(*map_, render_surface_point_cloud_M,
        render_surface_normals_M, *render_T_MC_, sensor);
    TOCK("RAYCASTING")
  }

  TICKD("renderVolume")
  renderVolumeKernel<VoxelImpl>(volume_RGBA_image_data, volume_RGBA_image_res,
      se::math::to_translation(*render_T_MC_), ambient,
      render_surface_point_cloud_M, render_surface_normals_M);
  TOCK("renderVolume")
}

void DenseSLAMSystem::renderTrack(uint32_t*              tracking_RGBA_image_data,
                                  const Eigen::Vector2i& tracking_RGBA_image_res) {

  TICKD("renderTrack")
  renderTrackKernel(tracking_RGBA_image_data, tracking_result_.data(), tracking_RGBA_image_res);
  TOCK("renderTrack")
}



void DenseSLAMSystem::renderDepth(uint32_t*              depth_RGBA_image_data,
                                  const Eigen::Vector2i& depth_RGBA_image_res,
                                  const SensorImpl&      sensor) {

  TICKD("renderDepth")
  renderDepthKernel(depth_RGBA_image_data, depth_image_.data(), depth_RGBA_image_res,
      sensor.near_plane, sensor.far_plane);
  TOCK("renderDepth")
}



void DenseSLAMSystem::renderRGBA(uint32_t*              output_RGBA_image_data,
                                 const Eigen::Vector2i& output_RGBA_image_res) {

  TICKD("renderRGBA")
  //renderRGBAKernel(output_RGBA_image_data, output_RGBA_image_res, rgba_image_);
  renderMaskKernel<se::class_mask_elem_t>(output_RGBA_image_data, output_RGBA_image_res,
      rgba_image_, input_segmentation_.classMask());
  TOCK("renderRGBA")
}



void DenseSLAMSystem::dumpMesh(const std::string filename_voxel,
                               const std::string filename_meter,
                               const bool print_path) {

  TICK("dumpMesh")

  std::vector<se::Triangle> mesh;
  VoxelImpl::dumpMesh(*map_, mesh);

  if (filename_voxel != "") {

    if (print_path) {
      std::cout << "Saving triangle mesh in map frame to file [voxel]:" << filename_voxel  << std::endl;
    }

    if (str_utils::ends_with(filename_voxel, ".ply")) {
      save_mesh_ply(mesh, filename_voxel.c_str());
    } else {
      save_mesh_vtk(mesh, filename_voxel.c_str());
    }


  }

  if (filename_meter != "") {

    if (print_path) {
      std::cout << "Saving triangle mesh in world frame to file [meter]:" << filename_meter  << std::endl;
    }

    if (str_utils::ends_with(filename_meter, ".ply")) {
      save_mesh_ply(mesh, filename_meter.c_str(), se::math::to_inverse_transformation(this->T_MW_), map_->voxelDim());
    } else {
      save_mesh_vtk(mesh, filename_meter.c_str(), se::math::to_inverse_transformation(this->T_MW_), map_->voxelDim());
    }

  }

  TOCK("dumpMesh")
}



void DenseSLAMSystem::saveStructure(const std::string base_filename) {

  TICK("saveStructure")
  std::stringstream f_s;
  f_s << base_filename << ".ply";
  se::save_octree_structure_ply(*map_, f_s.str().c_str());

  Eigen::Vector3i slice_coord = (map_->size() / map_->dim() * t_MC()).cast<int>();

  int scale = 0;
  // Save x plane
  std::stringstream fv_x;
  fv_x << base_filename << "_value_x.vtk";
  save_3d_value_slice_vtk(*map_, fv_x.str().c_str(),
                          Eigen::Vector3i(slice_coord.x(),     0,            0           ),
                          Eigen::Vector3i(slice_coord.x() + 1, map_->size(), map_->size()),
                          VoxelImpl::VoxelType::selectNodeValue, VoxelImpl::VoxelType::selectVoxelValue,
                          scale);

  // Save y plane
  std::stringstream fv_y;
  fv_y << base_filename << "_value_y.vtk";
  save_3d_value_slice_vtk(*map_, fv_y.str().c_str(),
                          Eigen::Vector3i(0,            slice_coord.y(),     0           ),
                          Eigen::Vector3i(map_->size(), slice_coord.y() + 1, map_->size()),
                          VoxelImpl::VoxelType::selectNodeValue, VoxelImpl::VoxelType::selectVoxelValue,
                          scale);

  // Save z plane
  std::stringstream fv_z;
  fv_z << base_filename << "_value_z.vtk";
  save_3d_value_slice_vtk(*map_, fv_z.str().c_str(),
                          Eigen::Vector3i(0,            0,            slice_coord.z()    ),
                          Eigen::Vector3i(map_->size(), map_->size(), slice_coord.z() + 1),
                          VoxelImpl::VoxelType::selectNodeValue, VoxelImpl::VoxelType::selectVoxelValue,
                          scale);

  // Save x plane
  std::stringstream fs_x;
  fs_x << base_filename << "_scale_x.vtk";
  save_3d_scale_slice_vtk(*map_, fs_x.str().c_str(),
                          Eigen::Vector3i(slice_coord.x(),     0,            0),
                          Eigen::Vector3i(slice_coord.x() + 1, map_->size(), map_->size()),
                          scale);

  // Save y plane
  std::stringstream fs_y;
  fs_y << base_filename << "_scale_y.vtk";
  save_3d_scale_slice_vtk(*map_, fs_y.str().c_str(),
                          Eigen::Vector3i(0, slice_coord.y(), 0),
                          Eigen::Vector3i(map_->size(), slice_coord.y() + 1, map_->size()),
                          scale);

  // Save z plane
  std::stringstream fs_z;
  fs_z << base_filename << "_scale_z.vtk";
  save_3d_scale_slice_vtk(*map_, fs_z.str().c_str(),
                          Eigen::Vector3i(0, 0, slice_coord.z()),
                          Eigen::Vector3i(map_->size(), map_->size(), slice_coord.z() + 1),
                          scale);
  TOCK("saveStructure")
}



void DenseSLAMSystem::structureStats(size_t& num_nodes,
                                     size_t& num_blocks,
                                     std::vector<size_t>& num_blocks_per_scale) {
  TICK("structureStats")
  num_nodes            = map_->pool().nodeBufferSize();
  num_blocks           = map_->pool().blockBufferSize();
  num_blocks_per_scale = map_->pool().blockBufferSizeDetailed();
  TOCK("structureStats")
}



// Semanticeight-only /////////////////////////////////////////////////////
bool DenseSLAMSystem::preprocessSegmentation(
    const se::SegmentationResult& segmentation) {

#if SE_VERBOSE >= SE_VERBOSE_NORMAL
  std::cout << "Preprocessing in: "
      << "   Masks " << segmentation.width << "x" << segmentation.height
      << "   Objects " << segmentation.object_instances.size()
      << "\n";
#endif

  // Copy the segmentation output and resize if needed
  input_segmentation_ = segmentation;
  input_segmentation_.resize(image_res_.x(), image_res_.y());
#if SE_VERBOSE >= SE_VERBOSE_NORMAL
  std::cout << input_segmentation_;
  std::cout << "Preprocessing out:"
      << "   RGB " << rgba_image_.width() << "x" << rgba_image_.height()
      << "   Depth " << depth_image_.width() << "x" << depth_image_.height()
      << "   Masks " << input_segmentation_.width << "x" << input_segmentation_.height
      << "   Objects " << input_segmentation_.object_instances.size()
      << "\n\n";
#endif
  return true;
}



bool DenseSLAMSystem::trackObjects(const SensorImpl& sensor) {
#if SE_VERBOSE >= SE_VERBOSE_NORMAL
  std::cout << "trackObjects in:   Number of current objects: "
      << objects_.size() << "\n";
#endif
  updateValidDepthMask(depth_image_);

  // Save the resulting masks in processed_segmentation_
  processed_segmentation_ = se::SegmentationResult(input_segmentation_);

  // Process masks to improve segmentation results.
  processed_segmentation_.removeLowConfidence(class_confidence_threshold_);
  processed_segmentation_.removeStuff();
  processed_segmentation_.removeSmall(small_mask_threshold_);
  processed_segmentation_.removeInvalidDepth(valid_depth_mask_);
  //processed_segmentation_.morphologicalRefinement(10);
  // TODO: more mask prerocessing

  // Compute the objects visible from the camera pose computed by tracking based on their bounding
  // volumes.
  visible_objects_ = se::get_visible_object_ids(objects_, sensor, T_MC_);

  // Raycast the background and objects from the current pose.
  raycastObjectsAndBg(sensor);

  // Match new objects to existing visible object instances.
  matchObjectInstances(processed_segmentation_, iou_threshold_);

  // Create object instances for all existing visible objects that were not
  // present in the segmentation.
  generateUndetectedInstances(processed_segmentation_);

  // Add the new detected objects to the object list and to the
  // visible_objects_.
  generateObjects(processed_segmentation_, sensor);

#if SE_VERBOSE >= SE_VERBOSE_NORMAL
  std::cout << processed_segmentation_;
  std::cout << "trackObjects out:  Number of current objects: "
      << objects_.size() << "\n\n";
#endif
  return true;
}



bool DenseSLAMSystem::integrateObjects(const SensorImpl& sensor,
                                       const size_t      frame) {
#if SE_VERBOSE >= SE_VERBOSE_NORMAL
  std::cout << "Integration in:    Number of objects to integrate: "
      << processed_segmentation_.object_instances.size() << "\n";
#endif
  // Integrate each object detection, including the background.
  for (auto& object_detection : processed_segmentation_.object_instances) {
    const int object_instance = object_detection.instance_id;
    Object& object = *(objects_[object_instance]);
    object.integrate(depth_image_, rgba_image_, object_detection, T_MC_, sensor, frame);

#if SE_BOUNDING_VOLUME != SE_BV_NONE
    // Update the bounding volume from the new measurement.
    if (object_instance != se::instance_bg) {
      object.bounding_volume_M_.merge(input_point_cloud_C_[0], T_MC_,
          object_detection.instance_mask);
    }
#else
#endif
#if SE_VERBOSE >= SE_VERBOSE_NORMAL
    std::cout << object_detection << "\n";
#endif
  }
#if SE_VERBOSE >= SE_VERBOSE_NORMAL
  std::cout << "Integration out:    " << "\n\n";
#endif
#if SE_VERBOSE == SE_VERBOSE_MINIMAL
  for (const auto& object : objects_) {
    std::cout << *object << "\n";
  }
#endif
  return true;
}



bool DenseSLAMSystem::raycastObjectsAndBg(const SensorImpl& sensor) {
#if SE_VERBOSE >= SE_VERBOSE_NORMAL
  std::cout << "Raycasting in:     " << "\n";
#endif

  // Raycast the background.
  raycast(sensor);
  // Raycast all objects.
  raycastObjectListKernel(objects_, visible_objects_, object_surface_point_cloud_M_,
      object_surface_normals_M_, raycasted_instance_mask_, object_scale_image_,
      object_min_scale_image_, raycast_T_MC_, sensor);
  // Compute regions where objects are occluded by the background.
  cv::Mat mask = se::occlusion_mask(object_surface_point_cloud_M_, surface_point_cloud_M_,
      map_->voxelDim(), raycast_T_MC_);
  // Occlude the object raycasts by the background.
#pragma omp parallel for
  for (int pixel_idx = 0; pixel_idx < image_res_.prod(); ++pixel_idx) {
    const bool object_occluded = mask.at<se::mask_elem_t>(pixel_idx);
    if (object_occluded) {
      object_surface_point_cloud_M_[pixel_idx] = surface_point_cloud_M_[pixel_idx];
      object_surface_normals_M_[pixel_idx] = surface_normals_M_[pixel_idx];
      raycasted_instance_mask_.at<se::instance_mask_elem_t>(pixel_idx) = se::instance_bg;
      object_scale_image_[pixel_idx] = -1;
    }
  }

#if SE_VERBOSE >= SE_VERBOSE_FLOOD
  for (const auto& instance_id : visible_objects_) {
    std::cout << instance_id << " ";
  }
  std::cout << "\n";
#endif
#if SE_VERBOSE >= SE_VERBOSE_NORMAL
  std::cout << "Raycasting out:    Visible objects: "
    << visible_objects_.size() << "/" << objects_.size()
    << "\n\n";
#endif
  return true;
}



void DenseSLAMSystem::renderObjects(uint32_t*              output_image_data,
                                    const Eigen::Vector2i& output_image_res,
                                    const SensorImpl&      sensor,
                                    const RenderMode       render_mode,
                                    const bool             render_bounding_volumes) {

  // Render the background normally.
  renderVolume(output_image_data, output_image_res, sensor);

  // Overlay the objects using the current raycast.
  renderObjectListKernel(output_image_data, output_image_res,
      se::math::to_translation(*render_T_MC_), ambient, objects_, object_surface_point_cloud_M_,
      object_surface_normals_M_, raycasted_instance_mask_, object_scale_image_,
      object_min_scale_image_, render_mode);

  if (render_bounding_volumes) {
    overlayBoundingVolumeKernel(output_image_data, output_image_res, objects_,
        *render_T_MC_, sensor, 1.0f);
  }
}



void DenseSLAMSystem::renderObjectClasses(uint32_t*              output_image_data,
                                          const Eigen::Vector2i& output_image_res) const {
  renderMaskKernel<se::class_mask_elem_t>(output_image_data, output_image_res,
      rgba_image_, processed_segmentation_.classMask());
}



void DenseSLAMSystem::renderObjectInstances(uint32_t*              output_image_data,
                                            const Eigen::Vector2i& output_image_res) const {
  renderMaskKernel<se::instance_mask_elem_t>(output_image_data, output_image_res,
      rgba_image_, processed_segmentation_.instanceMask());
}



void DenseSLAMSystem::renderRaycast(uint32_t*              output_image_data,
                                    const Eigen::Vector2i& output_image_res) {
  renderMaskKernel<se::instance_mask_elem_t>(output_image_data, output_image_res,
      rgba_image_, raycasted_instance_mask_, 0.8);
}



void DenseSLAMSystem::dumpObjectMeshes(const std::string filename, const bool print_path) {
  for (const auto& object : objects_) {
    std::vector<se::Triangle> mesh;
    ObjVoxelImpl::dumpMesh(*(object->map_), mesh);
    const std::string f = filename + "_" + std::to_string(object->instance_id) + ".vtk";
    const Eigen::Matrix4f T_WO = se::math::to_inverse_transformation(object->T_OM_ * T_MW_);
    if (print_path) {
      std::cout << "Saving triangle mesh to file :" << f  << "\n";
    }
    save_mesh_vtk(mesh, f.c_str(), T_WO, object->voxelDim());
  }
}





// Exploration only ///////////////////////////////////////////////////////
void DenseSLAMSystem::freeInitialPosition(const SensorImpl& sensor) {
  freeInitCylinder(sensor);
  // Update the frontier status
  update_frontiers(*map_, frontiers_, min_frontier_volume_);
  // Up-propagate free space to the root
  VoxelImpl::propagateToRoot(*map_);
}



std::vector<se::Volume<VoxelImpl::VoxelType>> DenseSLAMSystem::frontierVolumes() const {
  return se::frontier_volumes(*map_, frontiers_);
}



bool DenseSLAMSystem::goalReached() const {
  if (se::math::position_error(goal_T_MC_, T_MC_).norm() > goal_position_threshold_) {
    return false;
  }
  if (fabsf(se::math::yaw_error(goal_T_MC_, T_MC_)) > goal_yaw_threshold_) {
    return false;
  }
  if (fabsf(se::math::pitch_error(goal_T_MC_, T_MC_)) > goal_roll_pitch_threshold_) {
    return false;
  }
  if (fabsf(se::math::roll_error(goal_T_MC_, T_MC_)) > goal_roll_pitch_threshold_) {
    return false;
  }
  return true;
}



se::Path DenseSLAMSystem::computeNextPath_WC(const SensorImpl& sensor) {
  se::ExplorationConfig config {
      config_.num_candidates,
      {
        config_.raycast_width,
        config_.raycast_height,
        config_.linear_velocity,
        config_.angular_velocity,
        {
          "", Eigen::Vector3f::Zero(), Eigen::Vector3f::Zero(),
          config_.robot_radius,
          config_.safety_radius,
          config_.min_control_point_radius,
          config_.skeleton_sample_precision,
          config_.solving_time}}};
  const std::vector<se::key_t> frontier_vec(frontiers_.begin(), frontiers_.end());
  se::SinglePathExplorationPlanner planner (map_, frontier_vec, objects_, sensor, T_MC_, config);
  candidate_views_ = planner.views();
  rejected_candidate_views_ = planner.rejectedViews();
  goal_view_ = planner.bestView();
  path_M_ = planner.bestPath();
  if (path_M_.empty()) {
    goal_T_MC_ = T_MC_;
    path_W_.clear();
  } else {
    goal_T_MC_ = path_M_.back();
    // Convert the path to the world frame
    path_W_ = se::Path(path_M_.size());
    for (size_t i = 0; i < path_M_.size(); ++i) {
      path_W_[i] = T_WM_ * path_M_[i];
    }
  }
  return path_W_;
}



std::vector<se::CandidateView> DenseSLAMSystem::candidateViews() const {
  return candidate_views_;
}



std::vector<se::CandidateView> DenseSLAMSystem::rejectedCandidateViews() const {
  return rejected_candidate_views_;
}



se::CandidateView DenseSLAMSystem::goalView() const {
  return goal_view_;
}



se::Image<uint32_t> DenseSLAMSystem::renderEntropy(const SensorImpl& sensor,
                                                   const bool        visualize_yaw) {
  return goal_view_.renderEntropy(*map_, sensor, visualize_yaw);
}



se::Image<uint32_t> DenseSLAMSystem::renderEntropyDepth(const SensorImpl& sensor,
                                                        const bool        visualize_yaw) {
  return goal_view_.renderDepth(*map_, sensor, visualize_yaw);
}





// Private functions //////////////////////////////////////////////////////////
void DenseSLAMSystem::computeNewObjectParameters(
    Eigen::Matrix4f&       T_OM,
    int&                   map_size,
    float&                 map_dim,
    const cv::Mat&         mask,
    const int              class_id,
    const Eigen::Matrix4f& T_MC) {

  // Reference to the vertex map in camera coordinates.
  const se::Image<Eigen::Vector3f>& point_cloud_C = input_point_cloud_C_[0];

  // Compute the minimum, maximum and mean vertex values in world coordinates.
  Eigen::Vector3f vertex_min;
  Eigen::Vector3f vertex_max;
  Eigen::Vector3f vertex_mean;

#if SE_VERBOSE >= SE_VERBOSE_DETAILED
  const int count =
#endif
  vertexMapStats(point_cloud_C, mask, T_MC, vertex_min, vertex_max, vertex_mean);

  // Compute map dimensions
  const float max_dim = (vertex_max - vertex_min).maxCoeff();
  map_dim = fminf(2.5 * max_dim, 5.0);

#if SE_VERBOSE >= SE_VERBOSE_DETAILED
  std::cout<< __func__ << " max/min x/y/z: "
      << vertex_min.x() << " " << vertex_max.x() << " "
      << vertex_min.y() << " " << vertex_max.y() << " "
      << vertex_min.z() << " " << vertex_max.z()
      << std::endl;
  std::cout<< __func__ << " average of vertex out of " << count << ": "
      << vertex_mean.x() << " " << vertex_mean.y() << " " << vertex_mean.z()
      << ", with the max size " << max_dim
      << std::endl;
#endif

  // Select the map size depending on the object class.
  if (se::is_class_stuff(class_id)) {
    // "Stuff" should have the same resolution as the background.
    map_size = objects_[0]->mapSize();
  } else {
    // For "things" compute the volume size to achieve a voxel size of se::class_res[class_id].
    if (map_dim == 0.0f) {
      map_size = 0;
    } else {
      const float size_f = map_dim / se::class_res[class_id];
      // Round up to the nearest power of 2.
      map_size = 2 << static_cast<int>(ceil(log2(size_f)));
      // Saturate to 2048 voxels.
      map_size = std::min(map_size, 2048);
    }
  }

  // Put the origin of the Object frame at the corner of the object map. t_OM
  // is the origin of the Map frame expressed in the Object frame.
  T_OM = Eigen::Matrix4f::Identity();
  T_OM.topRightCorner<3,1>() = Eigen::Vector3f::Constant(map_dim / 2.0f) - vertex_mean;
}



void DenseSLAMSystem::matchObjectInstances(
    se::SegmentationResult& detections,
    const float             matching_threshold) {
  // Loop over all detected objects.
  for (auto& detection : detections) {
    // Loop over all visible objects to find the best match.
    float best_score = 0.f;
    int best_instance_id = se::instance_new;
    for (const auto& object_id : visible_objects_) {
      const Object& object = *(objects_[object_id]);
      // Compute the IoU of the masks.
      const cv::Mat object_mask = se::extract_instance(raycasted_instance_mask_, object.instance_id);
      const float iou = se::notIoU(detection.instance_mask, object_mask);
      const float score = iou;
      //const int same_class = (detection.classId() == object.classId());
      //const float score = iou * same_class;

      // Test if a better match was found.
      if ((score > matching_threshold) and (score > best_score)) {
        best_score = score;
        best_instance_id = object.instance_id;
      }
    }

    // Set the instance ID to that of the best match.
    detection.instance_id = best_instance_id;
#if SE_VERBOSE >= SE_VERBOSE_NORMAL
    std::cout << "Matched " << detection << " with IoU: " << best_score << std::endl;
#endif
  }
}



void DenseSLAMSystem::generateObjects(se::SegmentationResult& masks,
                                      const SensorImpl&       sensor) {
  // TODO: Only call the below function once when tracking is run.
  // Generate the vertex map from the current depth frame. When tracking is
  // performed this step is redundant.
  depthToPointCloudKernel(input_point_cloud_C_[0], scaled_depth_image_[0], sensor);

  // Iterate over all detected objects
  for (auto& object_detection : masks) {
    // If this is not a new object, skip generating it.
    const int& instance_id = object_detection.instance_id;
    if (instance_id != se::instance_new)
      continue;

    // Get references to object data.
    const cv::Mat& mask =  object_detection.instance_mask;
    const int& class_id = object_detection.classId();

    if ((class_id == se::class_bg) || (class_id == 255))
      continue;

    // Determine the new object volume size and pose.
    int map_size;
    float map_dim;
    Eigen::Matrix4f T_OM;
    computeNewObjectParameters(T_OM, map_size, map_dim,
        mask, class_id, T_MC_);

    // The mask may contain no valid depth measurements. In that case the
    // detected object instance must be removed from the SegmentationResult.
    // Set the instance ID to se::instance_invalid so that it is removed
    // afterwards.
    if (map_size == 0) {
      object_detection.instance_id = se::instance_invalid;
      continue;
    }

    // Add the object to the object list.
    const int new_object_instance_id = objects_.size();
    objects_.emplace_back(new Object(image_res_,
        Eigen::Vector3i::Constant(map_size),
        Eigen::Vector3f::Constant(map_dim),
        T_OM, T_MC_, new_object_instance_id));
    // Update the instance ID of the object detection as well.
    object_detection.instance_id = new_object_instance_id;
    // Add it to the visible_objects_.
    visible_objects_.insert(object_detection.instance_id);

#if SE_VERBOSE >= SE_VERBOSE_DETAILED
    std::cout <<  __func__ << ":"
        << "   volume extent: " << map_dim
        << "   volume size: " << map_size
        << "   volume step: " << map_dim / map_size
        << "   class id: "<< class_id
        << "   T_OM:\n" << T_OM << std::endl;
#endif
  }

  // Remove all masks whose corresponding depth measurements are all invalid.
  masks.removeInvalid();
}



void DenseSLAMSystem::updateValidDepthMask(const se::Image<float>& depth) {
#pragma omp parallel for
  for (int y = 0; y < depth.height(); y++) {
    for (int x = 0; x < depth.width(); x++) {
      if (depth[x + y * depth.width()] != 0.f) {
        valid_depth_mask_.at<se::mask_elem_t>(y, x) = 255;
      } else {
        valid_depth_mask_.at<se::mask_elem_t>(y, x) = 0;
      }
    }
  }
}



void DenseSLAMSystem::generateUndetectedInstances(se::SegmentationResult& detections) {
  // Find the set of visible but not detected/matched objects. No need to
  // consider the background.
  std::set<int> undetected_objects (visible_objects_);
  undetected_objects.erase(se::instance_bg);
  for (const auto& detection : detections) {
    if (detection.instance_id != se::instance_new) {
      undetected_objects.erase(detection.instance_id);
    }
  }
#if SE_VERBOSE >= SE_VERBOSE_NORMAL
  std::cout << "Unmatched objects " << undetected_objects.size() << "\n";
#endif

  // Create a detection for each undetected object.
  for (const auto& undetected_id : undetected_objects) {
    const cv::Mat undetected_mask = se::extract_instance(raycasted_instance_mask_, undetected_id);
    const se::InstanceSegmentation undetected_instance (undetected_id, undetected_mask);
    detections.object_instances.push_back(undetected_instance);
  }
}



// Exploration only ///////////////////////////////////////////////////////
void DenseSLAMSystem::freeInitCylinder(const SensorImpl& sensor) {
  if (!std::is_same<VoxelImpl, MultiresOFusion>::value) {
    std::cerr << "Error: Only MultiresOFusion is supported\n";
    std::abort();
  }
  // Compute the cylinder parameters and increase the height by some percentage
  const float height = 3.0f * 2.0f * (config_.robot_radius + config_.safety_radius);
  const float radius = std::max(height, (height / 2.0f) / tan(sensor.vertical_fov / 2.0f));
  const Eigen::Vector3f centre_M = T_MC_.topRightCorner<3,1>();
  // Compute the cylinder's AABB corners in metres and voxels
  const Eigen::Vector3f aabb_min_M = centre_M - Eigen::Vector3f(radius, radius, height);
  const Eigen::Vector3f aabb_max_M = centre_M + Eigen::Vector3f(radius, radius, height);
  // Compute the coordinates of all the points corresponding to voxels in the AABB
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> aabb_points_M;
  for (float z = aabb_min_M.z(); z <= aabb_max_M.z(); z += map_->voxelDim()) {
    for (float y = aabb_min_M.y(); y <= aabb_max_M.y(); y += map_->voxelDim()) {
      for (float x = aabb_min_M.x(); x <= aabb_max_M.x(); x += map_->voxelDim()) {
        aabb_points_M.push_back(Eigen::Vector3f(x, y, z));
      }
    }
  }
  // Allocate the required VoxelBlocks
  std::set<se::key_t> code_set;
  for (const auto& point_M : aabb_points_M) {
    const Eigen::Vector3i voxel = map_->pointToVoxel(point_M);
    if (map_->contains(voxel)) {
      code_set.insert(map_->hash(voxel.x(), voxel.y(), voxel.z(), map_->blockDepth()));
    }
  }
  std::vector<se::key_t> codes (code_set.begin(), code_set.end());
  map_->allocate(codes.data(), codes.size());
  // Add the allocated VoxelBlocks to the frontier set
  se::setunion(frontiers_, code_set);
  // The data to store in the free voxels
  auto data = VoxelImpl::VoxelType::initData();
  data.x = -21; // The path planning threshold is -20.
  data.y = 1;
  data.observed = true;
  // Allocate the VoxelBlocks up to some scale
  constexpr int scale = 0;
  std::vector<VoxelImpl::VoxelBlockType*> blocks;
  map_->getBlockList(blocks, false);
  for (auto& block : blocks) {
    block->active(true);
    block->allocateDownTo(scale);
  }
  // Set the cylinder voxels to free
  for (const auto& point_M : aabb_points_M) {
    const Eigen::Vector3f voxel_dist_M = (centre_M - point_M).array().abs().matrix();
    if (voxel_dist_M.head<2>().norm() <= radius && voxel_dist_M.z() <= height) {
      map_->setAtPoint(point_M, data, scale);
    }
  }
}

