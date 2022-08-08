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

#include <Eigen/StdVector>
#include <cstring>

#include "se/algorithms/balancing.hpp"
#include "se/algorithms/meshing.hpp"
#include "se/depth_utils.hpp"
#include "se/exploration_utils.hpp"
#include "se/frontiers.hpp"
#include "se/functors/for_each.hpp"
#include "se/geometry/octree_collision.hpp"
#include "se/io/meshing_io.hpp"
#include "se/io/octree_io.hpp"
#include "se/object_utils.hpp"
#include "se/perfstats.h"
#include "se/rendering.hpp"
#include "se/semanticeight_definitions.hpp"
#include "se/set_operations.hpp"
#include "se/timings.h"
#include "se/voxel_block_ray_iterator.hpp"
#include "se/voxel_implementations/MultiresOFusion/updating_model.hpp"

extern PerfStats stats;

DenseSLAMSystem::DenseSLAMSystem(const Eigen::Vector2i& image_res,
                                 const Eigen::Vector3i& map_size,
                                 const Eigen::Vector3f& map_dim,
                                 const Eigen::Vector3f& t_MW,
                                 std::vector<int>& pyramid,
                                 const se::Configuration& config,
                                 const std::string voxel_impl_yaml_path) :
        DenseSLAMSystem(image_res,
                        map_size,
                        map_dim,
                        se::math::to_transformation(t_MW),
                        pyramid,
                        config,
                        voxel_impl_yaml_path)
{
}

DenseSLAMSystem::DenseSLAMSystem(const Eigen::Vector2i& image_res,
                                 const Eigen::Vector3i& map_size,
                                 const Eigen::Vector3f& map_dim,
                                 const Eigen::Matrix4f& T_MW,
                                 std::vector<int>& pyramid,
                                 const se::Configuration& config,
                                 const std::string voxel_impl_yaml_path) :
        image_res_(image_res),
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
        scale_image_(image_res_.x(), image_res_.y(), -1),
        min_scale_image_(image_res_.x(), image_res_.y(), -1),
        input_segmentation_(image_res_.x(), image_res_.y()),
        processed_segmentation_(image_res_.x(), image_res_.y()),
        object_surface_point_cloud_M_(image_res_.x(), image_res_.y()),
        object_surface_normals_M_(image_res_.x(), image_res_.y()),
        object_scale_image_(image_res_.x(), image_res_.y(), -1),
        object_min_scale_image_(image_res_.x(), image_res_.y(), -1),
        aabb_min_M_((T_MW * config.aabb_min_W.homogeneous()).head<3>()),
        aabb_max_M_((T_MW * config.aabb_max_W.homogeneous()).head<3>()),
        aabb_edges_M_(se::AABB(aabb_min_M_, aabb_max_M_).edges())
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
        ObjVoxelImpl::configure(yaml_voxel_impl_config, se::SemanticClass::default_res);
    }
    else {
        VoxelImpl::configure(voxel_dim);
        ObjVoxelImpl::configure(se::SemanticClass::default_res);
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
    map_ =
        std::shared_ptr<se::Octree<VoxelImpl::VoxelType>>(new se::Octree<VoxelImpl::VoxelType>());
    map_->init(map_size_.x(), map_dim_.x());

    // Semanticeight-only /////////////////////////////////////////////////////
    valid_depth_mask_ =
        cv::Mat(cv::Size(image_res_.x(), image_res_.y()), se::mask_t, cv::Scalar(255));
    raycasted_instance_mask_ = cv::Mat(
        cv::Size(image_res_.x(), image_res_.y()), se::instance_mask_t, cv::Scalar(se::instance_bg));
    occlusion_mask_ = cv::Mat(cv::Size(image_res_.x(), image_res_.y()), se::mask_t, cv::Scalar(0));
}



bool DenseSLAMSystem::preprocessDepth(const float* input_depth_image_data,
                                      const Eigen::Vector2i& input_depth_image_res,
                                      const bool filter_depth)
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    TICKD("preprocessDepth")
    //downsampleDepthKernel(input_depth_image_data, input_depth_image_res, depth_image_);
    downsampleNearestNeighborKernel(input_depth_image_data, input_depth_image_res, depth_image_);

    if (filter_depth) {
        bilateralFilterKernel(
            scaled_depth_image_[0], depth_image_, gaussian_, e_delta, gaussian_radius);
    }
    else {
        std::memcpy(scaled_depth_image_[0].data(),
                    depth_image_.data(),
                    sizeof(float) * image_res_.x() * image_res_.y());
    }
    if (!config_.isExperiment()) {
        const float far_plane = 10.0f;
        const float min_sigma = 0.005f;
        const float max_sigma = 0.200f;
        // k_sigma * far_plane^2 == max_sigma
        const float k_sigma = max_sigma / (far_plane * far_plane);
        add_depth_measurement_noise(depth_image_, k_sigma, min_sigma, max_sigma);
    }
    TOCK("preprocessDepth")
    return true;
}



bool DenseSLAMSystem::preprocessColor(const uint32_t* input_RGBA_image_data,
                                      const Eigen::Vector2i& input_RGBA_image_res)
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    TICKD("preprocessColor")
    //downsampleImageKernel(input_RGBA_image_data, input_RGBA_image_res, rgba_image_);
    downsampleNearestNeighborKernel(input_RGBA_image_data, input_RGBA_image_res, rgba_image_);
    TOCK("preprocessColor")
    return true;
}



bool DenseSLAMSystem::track(const SensorImpl& sensor, const float icp_threshold)
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    TICK("TRACKING")
    // half sample the input depth maps into the pyramid levels
    for (unsigned int i = 1; i < iterations_.size(); ++i) {
        halfSampleRobustImageKernel(
            scaled_depth_image_[i], scaled_depth_image_[i - 1], e_delta * 3, 1);
    }

    // prepare the 3D information from the input depth maps
    for (unsigned int i = 0; i < iterations_.size(); ++i) {
        const float scaling_factor = 1.f / (1 << i);
        const SensorImpl scaled_sensor(sensor, scaling_factor);
        depthToPointCloudKernel(input_point_cloud_C_[i], scaled_depth_image_[i], scaled_sensor);
        if (sensor.left_hand_frame) {
            pointCloudToNormalKernel<true>(input_normals_C_[i], input_point_cloud_C_[i]);
        }
        else {
            pointCloudToNormalKernel<false>(input_normals_C_[i], input_point_cloud_C_[i]);
        }
    }

    previous_T_MC_ = T_MC_;

    for (int level = iterations_.size() - 1; level >= 0; --level) {
        Eigen::Vector2i reduction_output_res(image_res_.x() / (int) pow(2, level),
                                             image_res_.y() / (int) pow(2, level));
        for (int i = 0; i < iterations_[level]; ++i) {
            trackKernel(tracking_result_.data(),
                        input_point_cloud_C_[level],
                        input_normals_C_[level],
                        surface_point_cloud_M_,
                        surface_normals_M_,
                        T_MC_,
                        raycast_T_MC_,
                        sensor,
                        dist_threshold,
                        normal_threshold);

            reduceKernel(reduction_output_.data(),
                         reduction_output_res,
                         tracking_result_.data(),
                         image_res_);

            if (updatePoseKernel(T_MC_, reduction_output_.data(), icp_threshold))
                break;
        }
    }
    TOCK("TRACKING")
    return checkPoseKernel(
        T_MC_, previous_T_MC_, reduction_output_.data(), image_res_, track_threshold);
}



bool DenseSLAMSystem::integrate(const SensorImpl& sensor, const unsigned frame)
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    TICK("INTEGRATION")
    const int num_blocks_per_pixel = map_->size() / ((VoxelBlockType::size_li));
    const size_t num_blocks_total = num_blocks_per_pixel * image_res_.x() * image_res_.y();
    allocation_list_.reserve(num_blocks_total);

    const Eigen::Matrix4f T_CM = se::math::to_inverse_transformation(T_MC_); // TODO:
    const size_t num_voxel = VoxelImpl::buildAllocationList(
        *map_, depth_image_, T_MC_, sensor, allocation_list_.data(), allocation_list_.capacity());

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
    update_frontiers(*map_, frontiers_, config_.frontier_cluster_min_ratio);
    TOCK("FRONTIERS")
    // Update the free/occupied volume.
    se::ExploredVolume ev(*map_, aabb_min_M_, aabb_max_M_);
    free_volume = ev.free_volume;
    occupied_volume = ev.occupied_volume;
    explored_volume = ev.explored_volume;
    return true;
}



bool DenseSLAMSystem::raycast(const SensorImpl& sensor)
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    TICK("RAYCASTING")
    raycast_T_MC_ = T_MC_;
    raycastKernel<VoxelImpl>(*map_,
                             surface_point_cloud_M_,
                             surface_normals_M_,
                             scale_image_,
                             min_scale_image_,
                             raycast_T_MC_,
                             sensor);
    TOCK("RAYCASTING")
    return true;
}



void DenseSLAMSystem::renderVolume(uint32_t* volume_RGBA_image_data,
                                   const Eigen::Vector2i& volume_RGBA_image_res,
                                   const SensorImpl& sensor,
                                   const bool render_scale)
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    se::Image<Eigen::Vector3f> render_surface_point_cloud_M(image_res_.x(), image_res_.y());
    se::Image<Eigen::Vector3f> render_surface_normals_M(image_res_.x(), image_res_.y());
    se::Image<int8_t> scale_image(image_res_.x(), image_res_.y());
    se::Image<int8_t> min_scale_image(image_res_.x(), image_res_.y());
    if (render_T_MC_->isApprox(raycast_T_MC_)) {
        // Copy the raycast from the camera viewpoint. Can't safely use memcpy with
        // Eigen objects it seems.
        for (size_t i = 0; i < surface_point_cloud_M_.size(); ++i) {
            render_surface_point_cloud_M[i] = surface_point_cloud_M_[i];
            render_surface_normals_M[i] = surface_normals_M_[i];
            scale_image[i] = scale_image_[i];
            min_scale_image[i] = min_scale_image_[i];
        }
    }
    else {
        TICK("RAYCASTING")
        // Raycast the map from the render viewpoint.
        raycastKernel<VoxelImpl>(*map_,
                                 render_surface_point_cloud_M,
                                 render_surface_normals_M,
                                 scale_image,
                                 min_scale_image,
                                 *render_T_MC_,
                                 sensor);
        TOCK("RAYCASTING")
    }

    TICKD("renderVolume")
    renderVolumeKernel<VoxelImpl>(volume_RGBA_image_data,
                                  volume_RGBA_image_res,
                                  se::math::to_translation(*render_T_MC_),
                                  ambient,
                                  render_surface_point_cloud_M,
                                  render_surface_normals_M,
                                  scale_image_,
                                  render_scale);
    TOCK("renderVolume")
}

void DenseSLAMSystem::renderTrack(uint32_t* tracking_RGBA_image_data,
                                  const Eigen::Vector2i& tracking_RGBA_image_res)
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    TICKD("renderTrack")
    renderTrackKernel(tracking_RGBA_image_data, tracking_result_.data(), tracking_RGBA_image_res);
    TOCK("renderTrack")
}



void DenseSLAMSystem::renderDepth(uint32_t* depth_RGBA_image_data,
                                  const Eigen::Vector2i& depth_RGBA_image_res,
                                  const SensorImpl& sensor)
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    TICKD("renderDepth")
    renderDepthKernel(depth_RGBA_image_data,
                      depth_image_.data(),
                      depth_RGBA_image_res,
                      sensor.near_plane,
                      sensor.far_plane);
    TOCK("renderDepth")
}



void DenseSLAMSystem::renderRGBA(uint32_t* output_RGBA_image_data,
                                 const Eigen::Vector2i& output_RGBA_image_res)
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    TICKD("renderRGBA")
    renderRGBAKernel(output_RGBA_image_data, output_RGBA_image_res, rgba_image_);
    TOCK("renderRGBA")
}



int DenseSLAMSystem::saveMesh(const std::string& filename, const Eigen::Matrix4f& T_FW) const
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    TICK("saveMesh")
    se::TriangleMesh mesh;
    VoxelImpl::dumpMesh(*map_, mesh);
    // Scale voxels to meters.
    Eigen::Matrix4f T_FM = T_FW * T_WM_;
    T_FM.topLeftCorner<3, 3>() *= map_->voxelDim();
    const std::string metadata = "voxel resolution: " + std::to_string(map_->voxelDim()) + " m";
    if (str_utils::ends_with(filename, ".ply")) {
        return se::io::save_mesh_ply(mesh, filename, T_FM, metadata);
    }
    else if (str_utils::ends_with(filename, ".vtk")) {
        return se::io::save_mesh_vtk(mesh, filename, T_FM, metadata);
    }
    else if (str_utils::ends_with(filename, ".obj")) {
        return se::io::save_mesh_obj(mesh, filename, T_FM, metadata);
    }
    TOCK("saveMesh")
    std::cerr << "Error saving mesh: unknown file extension in " << filename << "\n";
    return 2;
}



std::vector<se::Triangle> DenseSLAMSystem::triangleMeshV(const se::meshing::ScaleMode scale_mode)
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    TICK("triangleMesh")
    std::vector<se::Triangle> mesh;
    VoxelImpl::dumpMesh(*map_, mesh, scale_mode);
    TOCK("triangleMesh")
    return mesh;
}



void DenseSLAMSystem::saveStructure(const std::string base_filename)
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    TICK("saveStructure")
    std::stringstream f_s;
    f_s << base_filename << ".ply";
    se::save_octree_structure_ply(*map_, f_s.str().c_str());

    Eigen::Vector3i slice_coord = (map_->size() / map_->dim() * t_MC()).cast<int>();

    int scale = 0;
    // Save x plane
    std::stringstream fv_x;
    fv_x << base_filename << "_value_x.vtk";
    save_3d_value_slice_vtk(*map_,
                            fv_x.str().c_str(),
                            Eigen::Vector3i(slice_coord.x(), 0, 0),
                            Eigen::Vector3i(slice_coord.x() + 1, map_->size(), map_->size()),
                            VoxelImpl::VoxelType::selectNodeValue,
                            VoxelImpl::VoxelType::selectVoxelValue,
                            scale);

    // Save y plane
    std::stringstream fv_y;
    fv_y << base_filename << "_value_y.vtk";
    save_3d_value_slice_vtk(*map_,
                            fv_y.str().c_str(),
                            Eigen::Vector3i(0, slice_coord.y(), 0),
                            Eigen::Vector3i(map_->size(), slice_coord.y() + 1, map_->size()),
                            VoxelImpl::VoxelType::selectNodeValue,
                            VoxelImpl::VoxelType::selectVoxelValue,
                            scale);

    // Save z plane
    std::stringstream fv_z;
    fv_z << base_filename << "_value_z.vtk";
    save_3d_value_slice_vtk(*map_,
                            fv_z.str().c_str(),
                            Eigen::Vector3i(0, 0, slice_coord.z()),
                            Eigen::Vector3i(map_->size(), map_->size(), slice_coord.z() + 1),
                            VoxelImpl::VoxelType::selectNodeValue,
                            VoxelImpl::VoxelType::selectVoxelValue,
                            scale);

    // Save x plane
    std::stringstream fs_x;
    fs_x << base_filename << "_scale_x.vtk";
    save_3d_scale_slice_vtk(*map_,
                            fs_x.str().c_str(),
                            Eigen::Vector3i(slice_coord.x(), 0, 0),
                            Eigen::Vector3i(slice_coord.x() + 1, map_->size(), map_->size()),
                            scale);

    // Save y plane
    std::stringstream fs_y;
    fs_y << base_filename << "_scale_y.vtk";
    save_3d_scale_slice_vtk(*map_,
                            fs_y.str().c_str(),
                            Eigen::Vector3i(0, slice_coord.y(), 0),
                            Eigen::Vector3i(map_->size(), slice_coord.y() + 1, map_->size()),
                            scale);

    // Save z plane
    std::stringstream fs_z;
    fs_z << base_filename << "_scale_z.vtk";
    save_3d_scale_slice_vtk(*map_,
                            fs_z.str().c_str(),
                            Eigen::Vector3i(0, 0, slice_coord.z()),
                            Eigen::Vector3i(map_->size(), map_->size(), slice_coord.z() + 1),
                            scale);
    TOCK("saveStructure")
}



bool DenseSLAMSystem::saveThresholdSliceZ(const std::string filename, const float z_M)
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    const int z_voxel = map_->pointToVoxel(Eigen::Vector3f(0, 0, z_M)).z();
    return save_3d_value_slice_vtk(*map_,
                                   filename,
                                   Eigen::Vector3i(0, 0, z_voxel),
                                   Eigen::Vector3i(map_->size(), map_->size(), z_voxel + 1),
                                   VoxelImpl::VoxelType::threshold);
}



void DenseSLAMSystem::structureStats(size_t& num_nodes,
                                     size_t& num_blocks,
                                     std::vector<size_t>& num_blocks_per_scale)
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    TICK("structureStats")
    num_nodes = map_->pool().nodeBufferSize();
    num_blocks = map_->pool().blockBufferSize();
    num_blocks_per_scale = map_->pool().blockBufferSizeDetailed();
    TOCK("structureStats")
}



const se::Image<float>& DenseSLAMSystem::getDepth() const
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    return depth_image_;
}



// Semanticeight-only /////////////////////////////////////////////////////
bool DenseSLAMSystem::preprocessSegmentation(const se::SegmentationResult& segmentation)
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
#if SE_VERBOSE >= SE_VERBOSE_NORMAL
    printf("Preprocessing in:    Masks %dx%d   Objects %zu\n",
           segmentation.width,
           segmentation.height,
           segmentation.object_instances.size());
#endif

    // Copy the segmentation output and resize if needed
    input_segmentation_ = segmentation;
    input_segmentation_.resize(image_res_.x(), image_res_.y());
#if SE_VERBOSE >= SE_VERBOSE_NORMAL
    input_segmentation_.print();
    printf("Preprocessing out:   RGB %dx%d   Depth %dx%d   Masks %dx%d   Objects %zu\n\n",
           rgba_image_.width(),
           rgba_image_.height(),
           depth_image_.width(),
           depth_image_.height(),
           input_segmentation_.width,
           input_segmentation_.height,
           input_segmentation_.object_instances.size());
#endif
    return true;
}



bool DenseSLAMSystem::trackObjects(const SensorImpl& sensor, const int frame)
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
#if SE_VERBOSE >= SE_VERBOSE_NORMAL
    printf("trackObjects in:   Number of current objects: %zu\n", objects_.size());
#endif
    updateValidDepthMask(depth_image_, sensor);

    // Save the resulting masks in processed_segmentation_
    processed_segmentation_ = se::SegmentationResult(input_segmentation_);

    // Process masks to improve segmentation results.
    processed_segmentation_.removeStuff();
    processed_segmentation_.removeLowConfidence(class_confidence_threshold_);
    processed_segmentation_.filterInvalidDepth(valid_depth_mask_);
    processed_segmentation_.removeSmall(small_mask_threshold_);
    //processed_segmentation_.morphologicalRefinement(10);
    //cv::Mat depth_cv (cv::Size(depth_image_.width(), depth_image_.height()), CV_32FC1, depth_image_.data());
    //processed_segmentation_.removeDepthOutliers(depth_cv);
    // TODO: more mask prerocessing

    // Compute the objects visible from the camera pose computed by tracking based on their bounding
    // volumes.
    visible_objects_ = se::get_visible_object_ids(objects_, sensor, T_MC_);
#if SE_VERBOSE >= SE_VERBOSE_NORMAL
    printf("                   Number of visible objects: %zu\n", visible_objects_.size());
#endif

    // Raycast the background and objects from the current pose.
    raycastObjectsAndBg(sensor, frame);

    // Match new objects to existing visible object instances.
    matchObjectInstances(processed_segmentation_, iou_threshold_);

    // Create object instances for all existing visible objects that were not
    // present in the segmentation.
    generateUndetectedInstances(processed_segmentation_);

    // Add the new detected objects to the object list and to the
    // visible_objects_.
    generateObjects(processed_segmentation_, sensor);

#if SE_VERBOSE >= SE_VERBOSE_NORMAL
    processed_segmentation_.print();
    printf("trackObjects out:  Number of current objects: %zu\n\n", objects_.size());
#endif
    return true;
}



bool DenseSLAMSystem::integrateObjects(const SensorImpl& sensor, const size_t frame)
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
#if SE_VERBOSE >= SE_VERBOSE_NORMAL
    printf("Integration in:    Number of objects to integrate: %zu\n",
           processed_segmentation_.object_instances.size());
#endif
    // Integrate each object detection, including the background.
    for (auto& object_detection : processed_segmentation_.object_instances) {
        const int object_instance = object_detection.instance_id;
        Object& object = *(objects_[object_instance]);
        const cv::Mat raycasted_object_mask =
            se::extract_instance(raycasted_instance_mask_, object.instance_id);
        object.integrate(depth_image_,
                         rgba_image_,
                         object_detection,
                         raycasted_object_mask,
                         T_MC_,
                         sensor,
                         frame);

#if SE_BOUNDING_VOLUME != SE_BV_NONE
        // Update the bounding volume from the new measurement.
        if (object_instance != se::instance_bg) {
            object.bounding_volume_M_.merge(
                input_point_cloud_C_[0], T_MC_, object_detection.instance_mask);
        }
#else
#endif
#if SE_VERBOSE >= SE_VERBOSE_NORMAL
        object_detection.print();
        printf("\n");
#endif
    }
#if SE_VERBOSE >= SE_VERBOSE_NORMAL
    printf("Integration out:    \n\n");
#endif
#if SE_VERBOSE == SE_VERBOSE_MINIMAL
    for (const auto& object : objects_) {
        object->print();
        printf("\n");
    }
#endif
    return true;
}



bool DenseSLAMSystem::raycastObjectsAndBg(const SensorImpl& sensor, const int frame)
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
#if SE_VERBOSE >= SE_VERBOSE_NORMAL
    printf("Raycasting in:     \n");
#endif

    // Raycast the background.
    raycast(sensor);
    // Raycast all objects.
    raycastObjectListKernel(objects_,
                            visible_objects_,
                            object_surface_point_cloud_M_,
                            object_surface_normals_M_,
                            raycasted_instance_mask_,
                            object_scale_image_,
                            object_min_scale_image_,
                            raycast_T_MC_,
                            sensor,
                            frame);
    // Compute regions where objects are occluded by the background.
    occlusion_mask_ = se::occlusion_mask(
        object_surface_point_cloud_M_, surface_point_cloud_M_, map_->voxelDim(), raycast_T_MC_);
    //std::stringstream p;
    //p << "/home/srl/raycast_" << std::setw(5) << std::setfill('0') << frame << ".png";
    //cv::imwrite(p.str(), raycasted_instance_mask_ + 1);
    //std::stringstream q;
    //q << "/home/srl/occlusion_" << std::setw(5) << std::setfill('0') << frame << ".png";
    //cv::imwrite(q.str(), occlusion_mask_);
    // Occlude the object raycasts by the background.
#pragma omp parallel for
    for (int pixel_idx = 0; pixel_idx < image_res_.prod(); ++pixel_idx) {
        const bool object_occluded = occlusion_mask_.at<se::mask_elem_t>(pixel_idx);
        if (object_occluded) {
            object_surface_point_cloud_M_[pixel_idx] = surface_point_cloud_M_[pixel_idx];
            object_surface_normals_M_[pixel_idx] = surface_normals_M_[pixel_idx];
            raycasted_instance_mask_.at<se::instance_mask_elem_t>(pixel_idx) = se::instance_bg;
            object_scale_image_[pixel_idx] = -1;
            object_min_scale_image_[pixel_idx] = -1;
        }
    }

#if SE_VERBOSE >= SE_VERBOSE_FLOOD
    for (const auto& instance_id : visible_objects_) {
        printf("%d ", instance_id);
    }
    printf("\n");
#endif
#if SE_VERBOSE >= SE_VERBOSE_NORMAL
    printf("Raycasting out:    Visible objects: %zu/%zu\n\n",
           visible_objects_.size(),
           objects_.size());
#endif
    return true;
}



void DenseSLAMSystem::renderObjects(uint32_t* output_image_data,
                                    const Eigen::Vector2i& output_image_res,
                                    const SensorImpl& sensor,
                                    const RenderMode render_mode,
                                    const bool render_bounding_volumes)
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    se::Image<Eigen::Vector3f> render_surface_point_cloud_M(image_res_.x(), image_res_.y());
    se::Image<Eigen::Vector3f> render_surface_normals_M(image_res_.x(), image_res_.y());
    se::Image<int8_t> scale_image(image_res_.x(), image_res_.y());
    se::Image<int8_t> min_scale_image(image_res_.x(), image_res_.y());
    se::Image<Eigen::Vector3f> object_surface_point_cloud_M(image_res_.x(), image_res_.y());
    se::Image<Eigen::Vector3f> object_surface_normals_M(image_res_.x(), image_res_.y());
    se::Image<int8_t> object_scale_image(image_res_.x(), image_res_.y());
    se::Image<int8_t> object_min_scale_image(image_res_.x(), image_res_.y());
    cv::Mat raycasted_instance_mask = cv::Mat(
        cv::Size(image_res_.x(), image_res_.y()), se::instance_mask_t, cv::Scalar(se::instance_bg));
    std::set<int> visible_objects;
    if (render_T_MC_->isApprox(raycast_T_MC_)) {
        // Copy the raycast from the camera viewpoint. Can't safely use memcpy with
        // Eigen objects it seems.
        for (size_t i = 0; i < surface_point_cloud_M_.size(); ++i) {
            render_surface_point_cloud_M[i] = surface_point_cloud_M_[i];
            render_surface_normals_M[i] = surface_normals_M_[i];
            scale_image[i] = scale_image_[i];
            min_scale_image[i] = min_scale_image_[i];
            object_surface_point_cloud_M[i] = object_surface_point_cloud_M_[i];
            object_surface_normals_M[i] = object_surface_normals_M_[i];
            object_scale_image[i] = object_scale_image_[i];
            object_min_scale_image[i] = object_min_scale_image_[i];
        }
        raycasted_instance_mask = raycasted_instance_mask_;
        visible_objects = visible_objects_;
    }
    else {
        visible_objects = se::get_visible_object_ids(objects_, sensor, *render_T_MC_);
        // Raycast the map from the render viewpoint.
        raycastKernel<VoxelImpl>(*map_,
                                 render_surface_point_cloud_M,
                                 render_surface_normals_M,
                                 scale_image,
                                 min_scale_image,
                                 *render_T_MC_,
                                 sensor);
        // Raycast the objects from the render viewpoint.
        raycastObjectListKernel(objects_,
                                visible_objects,
                                object_surface_point_cloud_M,
                                object_surface_normals_M,
                                raycasted_instance_mask,
                                object_scale_image,
                                object_min_scale_image,
                                *render_T_MC_,
                                sensor,
                                -1);
        // Compute regions where objects are occluded by the background.
        cv::Mat occlusion_mask = se::occlusion_mask(object_surface_point_cloud_M,
                                                    render_surface_point_cloud_M,
                                                    map_->voxelDim(),
                                                    *render_T_MC_);
        // Occlude the object raycasts by the background.
#pragma omp parallel for
        for (int pixel_idx = 0; pixel_idx < image_res_.prod(); ++pixel_idx) {
            const bool object_occluded = occlusion_mask.at<se::mask_elem_t>(pixel_idx);
            if (object_occluded) {
                object_surface_point_cloud_M[pixel_idx] = render_surface_point_cloud_M[pixel_idx];
                object_surface_normals_M[pixel_idx] = render_surface_normals_M[pixel_idx];
                raycasted_instance_mask.at<se::instance_mask_elem_t>(pixel_idx) = se::instance_bg;
                object_scale_image[pixel_idx] = -1;
                object_min_scale_image_[pixel_idx] = -1;
            }
        }
    }

    // Render the background normally.
    renderVolumeKernel<VoxelImpl>(output_image_data,
                                  output_image_res,
                                  se::math::to_translation(*render_T_MC_),
                                  ambient,
                                  render_surface_point_cloud_M,
                                  render_surface_normals_M,
                                  scale_image);

    // Overlay the objects using the raycast.
    renderObjectListKernel(output_image_data,
                           output_image_res,
                           se::math::to_translation(*render_T_MC_),
                           ambient,
                           objects_,
                           object_surface_point_cloud_M,
                           object_surface_normals_M,
                           raycasted_instance_mask,
                           object_scale_image,
                           object_min_scale_image,
                           render_mode);

    if (render_bounding_volumes) {
        overlayBoundingVolumeKernel(
            output_image_data, output_image_res, objects_, *render_T_MC_, sensor, 1.0f);
    }
}



void DenseSLAMSystem::renderObjectClasses(uint32_t* output_image_data,
                                          const Eigen::Vector2i& output_image_res) const
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    renderMaskKernel<se::class_mask_elem_t>(
        output_image_data, output_image_res, rgba_image_, processed_segmentation_.classMask());
}



void DenseSLAMSystem::renderObjectInstances(uint32_t* output_image_data,
                                            const Eigen::Vector2i& output_image_res) const
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    renderMaskKernel<se::instance_mask_elem_t>(
        output_image_data, output_image_res, rgba_image_, processed_segmentation_.instanceMask());
}



void DenseSLAMSystem::renderRaycast(uint32_t* output_image_data,
                                    const Eigen::Vector2i& output_image_res)
{
    renderMaskKernel<se::instance_mask_elem_t>(
        output_image_data, output_image_res, rgba_image_, raycasted_instance_mask_, 0.8);
}



void DenseSLAMSystem::saveObjectMeshes(const std::string& filename,
                                       const Eigen::Matrix4f& T_FW) const
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    for (const auto& object : objects_) {
        std::vector<se::Triangle> mesh;
        ObjVoxelImpl::dumpMesh(*(object->map_), mesh, se::meshing::ScaleMode::Min);
        // Scale voxels to meters.
        Eigen::Matrix4f T_FO = T_FW * se::math::to_inverse_transformation(object->T_OM_ * T_MW_);
        T_FO.topLeftCorner<3, 3>() *= object->voxelDim();
        std::stringstream f;
        f << filename << "_" << std::setw(3) << std::setfill('0') << object->instance_id << ".ply";
        std::stringstream metadata_ss;
        metadata_ss << "class: " << se::semantic_classes.name(object->conf.classId()) << "\n"
                    << "class ID: " << object->conf.classId() << "\n"
                    << "instance ID: " << object->instance_id << "\n"
                    << "voxel resolution: " << object->voxelDim() << " m";
        se::io::save_mesh_ply(mesh, f.str(), T_FO, metadata_ss.str());
    }
}



std::vector<std::vector<se::Triangle>>
DenseSLAMSystem::objectTriangleMeshesV(const se::meshing::ScaleMode scale_mode)
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    std::vector<std::vector<se::Triangle>> meshes(objects_.size());
    for (size_t i = 0; i < objects_.size(); i++) {
        const auto& object = *objects_[i];
        ObjVoxelImpl::dumpMesh(*object.map_, meshes[i], scale_mode);
    }
    return meshes;
}



// Exploration only ///////////////////////////////////////////////////////
void DenseSLAMSystem::freeInitialPosition(const SensorImpl& sensor, const std::string& type)
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    const Eigen::Vector3f centre_M = T_MC_.topRightCorner<3, 1>();
    if (type == "cylinder") {
        constexpr float min_height = 0.5f;
        const float robot_height = 2.0f * config_.robot_radius;
        const float height_M = std::max(3.0f * robot_height, min_height);
        const float radius_M =
            std::max(height_M, (height_M / 2.0f) / tan(sensor.vertical_fov / 2.0f));
        freeCylinder(centre_M, radius_M, height_M);
    }
    else {
        // 2.4 because the same ratio was used in the ICRA 2020 paper.
        constexpr float min_radius = 0.5f;
        const float radius_M = std::max(2.4f * config_.robot_radius, min_radius);
        freeSphere(centre_M, radius_M);
    }
}



const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>&
DenseSLAMSystem::environmentAABBEdgesM() const
{
    // Only uses const members, no need to lock.
    return aabb_edges_M_;
}



std::vector<se::Volume<VoxelImpl::VoxelType>> DenseSLAMSystem::frontierVolumes() const
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    return se::frontier_volumes(*map_, frontiers_);
}



// Private functions //////////////////////////////////////////////////////////
void DenseSLAMSystem::computeNewObjectParameters(Eigen::Matrix4f& T_OM,
                                                 int& map_size,
                                                 float& map_dim,
                                                 const cv::Mat& mask,
                                                 const int class_id,
                                                 const Eigen::Matrix4f& T_MC)
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
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
    printf("%s max/min x/y/z: %f %f %f %f %f %f\n",
           __func__,
           vertex_min.x(),
           vertex_max.x(),
           vertex_min.y(),
           vertex_max.y(),
           vertex_min.z(),
           vertex_max.z());
    printf("%s average of vertex out of %d: %f %f %f, with the max size %f\n",
           __func__,
           count,
           vertex_mean.x(),
           vertex_mean.y(),
           vertex_mean.z(),
           max_dim);
#endif

    // Select the map size depending on the object class.
    if (se::semantic_classes.enabled(class_id)) {
        // For "things" compute the volume size to achieve a voxel size of
        // se::semantic_classes.res(class_id).
        if (map_dim == 0.0f) {
            map_size = 0;
        }
        else {
            const float size_f = map_dim / se::semantic_classes.res(class_id);
            // Round up to the nearest power of 2.
            map_size = 2 << static_cast<int>(ceil(log2(size_f)));
            // Saturate to 2048 voxels.
            map_size = std::min(map_size, 2048);
        }
    }
    else {
        // "Stuff" should have the same resolution as the background.
        map_size = objects_[0]->mapSize();
    }

    // Put the origin of the Object frame at the corner of the object map. t_OM
    // is the origin of the Map frame expressed in the Object frame.
    T_OM = Eigen::Matrix4f::Identity();
    T_OM.topRightCorner<3, 1>() = Eigen::Vector3f::Constant(map_dim / 2.0f) - vertex_mean;
}



void DenseSLAMSystem::matchObjectInstances(se::SegmentationResult& detections,
                                           const float matching_threshold)
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    // Loop over all detected objects.
    for (auto& detection : detections) {
        // Loop over all visible objects to find the best match.
        float best_score = 0.f;
        int best_instance_id = se::instance_new;
        for (const auto& object_id : visible_objects_) {
            const Object& object = *(objects_[object_id]);
            // Compute the IoU of the masks.
            const cv::Mat object_mask =
                se::extract_instance(raycasted_instance_mask_, object.instance_id);
            const float iou = se::notIoU(detection.instance_mask, object_mask);
            const float score = iou;
            //const int same_class = (detection.classId() == object.classId());
            //const float score = iou * same_class;

            // Test if a better match was found.
            if (score > best_score) {
                best_score = score;
                if (score > matching_threshold) {
                    best_instance_id = object.instance_id;
                }
            }
        }

        // Set the instance ID to that of the best match.
        detection.instance_id = best_instance_id;
#if SE_VERBOSE >= SE_VERBOSE_NORMAL
        printf("Matched ");
        detection.print();
        printf(" with IoU: %f\n", best_score);
#endif
    }
}



void DenseSLAMSystem::generateObjects(se::SegmentationResult& masks, const SensorImpl& sensor)
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
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
        const cv::Mat& mask = object_detection.instance_mask;
        const int& class_id = object_detection.classId();

        if ((class_id == se::semantic_classes.backgroundId()) || (class_id == 255))
            continue;

        // Determine the new object volume size and pose.
        int map_size;
        float map_dim;
        Eigen::Matrix4f T_OM;
        computeNewObjectParameters(T_OM, map_size, map_dim, mask, class_id, T_MC_);

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
                                         T_OM,
                                         T_MC_,
                                         new_object_instance_id));
        // Update the instance ID of the object detection as well.
        object_detection.instance_id = new_object_instance_id;
        // Add it to the visible_objects_.
        visible_objects_.insert(object_detection.instance_id);

#if SE_VERBOSE >= SE_VERBOSE_DETAILED
        printf(
            "%s:   volume extent: %f   volume size: %d   volume step: %f   class id: %d   T_OM:\n",
            __func__,
            map_dim,
            map_size,
            map_dim / map_size,
            class_id);
        std::cout << T_OM << "\n";
#endif
    }

    // Remove all masks whose corresponding depth measurements are all invalid.
    masks.removeInvalid();
}



void DenseSLAMSystem::updateValidDepthMask(const se::Image<float>& depth, const SensorImpl& sensor)
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
#pragma omp parallel for
    for (int y = 0; y < depth.height(); y++) {
        for (int x = 0; x < depth.width(); x++) {
            const float d = depth[x + y * depth.width()];
            if (d != 0.0f && !isnan(d) && !isinf(d) && d >= sensor.near_plane
                && d <= sensor.far_plane) {
                valid_depth_mask_.at<se::mask_elem_t>(y, x) = 255;
            }
            else {
                valid_depth_mask_.at<se::mask_elem_t>(y, x) = 0;
            }
        }
    }
}



void DenseSLAMSystem::generateUndetectedInstances(se::SegmentationResult& detections)
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    // Find the set of visible but not detected/matched objects. No need to
    // consider the background.
    std::set<int> undetected_objects(visible_objects_);
    undetected_objects.erase(se::instance_bg);
    for (const auto& detection : detections) {
        if (detection.instance_id != se::instance_new) {
            undetected_objects.erase(detection.instance_id);
        }
    }
#if SE_VERBOSE >= SE_VERBOSE_NORMAL
    printf("Unmatched objects %zu\n", undetected_objects.size());
#endif

    // Create a detection for each undetected object.
    for (const auto& undetected_id : undetected_objects) {
        const cv::Mat undetected_mask =
            se::extract_instance(raycasted_instance_mask_, undetected_id);
        const se::InstanceSegmentation undetected_instance(
            undetected_id, objects_[undetected_id]->conf, undetected_mask, false);
        detections.object_instances.push_back(undetected_instance);
    }
}



// Exploration only ///////////////////////////////////////////////////////
void DenseSLAMSystem::freeCylinder(const Eigen::Vector3f& centre_M,
                                   const float radius_M,
                                   const float height_M)
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    static_assert(std::is_same<VoxelImpl, MultiresOFusion>::value,
                  "Only MultiresOFusion is supported");
    const Eigen::Vector3f aabb_min_M = centre_M - Eigen::Vector3f(radius_M, radius_M, height_M);
    const Eigen::Vector3f aabb_max_M = centre_M + Eigen::Vector3f(radius_M, radius_M, height_M);
    // Compute the coordinates of all the points corresponding to voxels in the AABB.
    const float voxel_dim = map_->voxelDim();
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> aabb_points_M;
    for (float z = aabb_min_M.z(); z <= aabb_max_M.z(); z += voxel_dim) {
        for (float y = aabb_min_M.y(); y <= aabb_max_M.y(); y += voxel_dim) {
            for (float x = aabb_min_M.x(); x <= aabb_max_M.x(); x += voxel_dim) {
                aabb_points_M.emplace_back(x, y, z);
            }
        }
    }
    // Allocate the required VoxelBlocks.
    std::set<se::key_t> code_set;
    for (const auto& point_M : aabb_points_M) {
        const Eigen::Vector3i voxel = map_->pointToVoxel(point_M);
        if (map_->contains(voxel)) {
            code_set.insert(map_->hash(voxel.x(), voxel.y(), voxel.z(), map_->blockDepth()));
        }
    }
    std::vector<se::key_t> codes(code_set.begin(), code_set.end());
    map_->allocate(codes.data(), codes.size());
    // Add the allocated VoxelBlocks to the frontier set
    se::setunion(frontiers_, code_set);
    // The data to store in the free voxels
    auto new_data = VoxelImpl::VoxelType::initData();
    new_data.x = -21; // The path planning threshold is -20.
    new_data.y = 1;
    new_data.observed = true;
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
        if (voxel_dist_M.head<2>().norm() <= radius_M && voxel_dist_M.z() <= height_M) {
            VoxelImpl::VoxelType::VoxelData current_data;
            map_->getAtPoint(point_M, current_data, scale);
            if (!VoxelImpl::VoxelType::isInside(current_data)) {
                map_->setAtPoint(point_M, new_data, scale);
            }
        }
    }
    // Update the frontier status
    update_frontiers(*map_, frontiers_, config_.frontier_cluster_min_ratio);
    // Up-propagate free space to the root
    VoxelImpl::propagateToRoot(*map_);
}



void DenseSLAMSystem::freeSphere(const Eigen::Vector3f& centre_M, const float radius_M)
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    static_assert(std::is_same<VoxelImpl, MultiresOFusion>::value,
                  "Only MultiresOFusion is supported");
    const Eigen::Vector3f aabb_min_M = centre_M - Eigen::Vector3f::Constant(radius_M);
    const Eigen::Vector3f aabb_max_M = centre_M + Eigen::Vector3f::Constant(radius_M);
    // Compute the coordinates of all the points corresponding to voxels in the AABB
    const float voxel_dim = map_->voxelDim();
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> aabb_points_M;
    for (float z = aabb_min_M.z(); z <= aabb_max_M.z(); z += voxel_dim) {
        for (float y = aabb_min_M.y(); y <= aabb_max_M.y(); y += voxel_dim) {
            for (float x = aabb_min_M.x(); x <= aabb_max_M.x(); x += voxel_dim) {
                aabb_points_M.emplace_back(x, y, z);
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
    std::vector<se::key_t> codes(code_set.begin(), code_set.end());
    map_->allocate(codes.data(), codes.size());
    // Add the allocated VoxelBlocks to the frontier set
    se::setunion(frontiers_, code_set);
    // The data to store in the free voxels
    auto new_data = VoxelImpl::VoxelType::initData();
    new_data.x = -21; // The path planning threshold is -20.
    new_data.y = 1;
    new_data.observed = true;
    // Allocate the VoxelBlocks up to some scale
    constexpr int scale = 0;
    std::vector<VoxelImpl::VoxelBlockType*> blocks;
    map_->getBlockList(blocks, false);
    for (auto& block : blocks) {
        block->active(true);
        block->allocateDownTo(scale);
    }
    // Set the sphere voxels to free
    for (const auto& point_M : aabb_points_M) {
        const Eigen::Vector3f voxel_dist_M = (centre_M - point_M).array().abs().matrix();
        if (voxel_dist_M.norm() <= radius_M) {
            VoxelImpl::VoxelType::VoxelData current_data;
            map_->getAtPoint(point_M, current_data, scale);
            if (!VoxelImpl::VoxelType::isInside(current_data)) {
                map_->setAtPoint(point_M, new_data, scale);
            }
        }
    }
    // Update the frontier status
    update_frontiers(*map_, frontiers_, config_.frontier_cluster_min_ratio);
    // Up-propagate free space to the root
    VoxelImpl::propagateToRoot(*map_);
}
