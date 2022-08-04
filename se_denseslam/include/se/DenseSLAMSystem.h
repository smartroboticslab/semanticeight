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

#ifndef _KERNELS_
#define _KERNELS_

#include <Eigen/Dense>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <octomap/octomap.h>
#include <set>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "object.hpp"
#include "object_rendering.hpp"
#include "preprocessing.hpp"
#include "se/commons.h"
#include "se/config.h"
#include "se/image/image.hpp"
#include "se/io/octomap_io.hpp"
#include "se/octree.hpp"
#include "se/perfstats.h"
#include "se/pose_vector_history.hpp"
#include "se/segmentation_result.hpp"
#include "se/sensor_implementation.hpp"
#include "se/single_path_exploration_planner.hpp"
#include "se/timings.h"
#include "se/voxel_implementations.hpp"
#include "tracking.hpp"



class DenseSLAMSystem {
    using VoxelBlockType = typename VoxelImpl::VoxelType::VoxelBlockType;

    private:
    mutable std::recursive_mutex mutex_;
    // Input images
    Eigen::Vector2i image_res_;
    se::Image<float> depth_image_;
    se::Image<uint32_t> rgba_image_;

    // Pipeline config
    Eigen::Vector3f map_dim_;
    Eigen::Vector3i map_size_;
    se::Configuration config_;
    std::vector<float> gaussian_;

    // Camera pose
    Eigen::Matrix4f init_T_MC_; // Initial camera pose in map frame
    Eigen::Matrix4f T_MC_;      // Camera pose in map frame

    // Tracking
    Eigen::Matrix4f previous_T_MC_; // Camera pose of the previous image in map frame
    std::vector<int> iterations_;
    std::vector<se::Image<float>> scaled_depth_image_;
    std::vector<se::Image<Eigen::Vector3f>> input_point_cloud_C_;
    std::vector<se::Image<Eigen::Vector3f>> input_normals_C_;
    std::vector<float> reduction_output_;
    std::vector<TrackData> tracking_result_;

    // Raycasting
    Eigen::Matrix4f raycast_T_MC_; // Raycasting camera pose in map frame
    se::Image<Eigen::Vector3f> surface_point_cloud_M_;
    se::Image<Eigen::Vector3f> surface_normals_M_;

    // Rendering
    Eigen::Matrix4f* render_T_MC_; // Rendering camera pose in map frame
    bool need_render_ = false;

    // Map
    Eigen::Matrix4f T_MW_; // Constant world to map frame transformation
    Eigen::Matrix4f T_WM_;
    std::vector<se::key_t> allocation_list_;
    std::shared_ptr<se::Octree<VoxelImpl::VoxelType>> map_;

    // Semanticeight-only /////////////////////////////////////////////////////
    /** Only used for rendering.
     */
    se::Image<int8_t> scale_image_;
    /** Contains the minimum scale at which the background has been integrated as of the last
     * raycast.
     */
    se::Image<int8_t> min_scale_image_;

    /**
     * The input segmentation for the current frame.
     */
    se::SegmentationResult input_segmentation_;

    /**
     * The segmentation for the current frame after it has been post processed
     * and the detections matched to any existing objects.
     */
    se::SegmentationResult processed_segmentation_;

    /**
     * The objects detected and tracked by the pipeline. The background is
     * always the first object.
     */
    Objects objects_;

    /**
     * The instance IDs of the objects that are currently visible by the
     * camera.
     */
    std::set<int> visible_objects_;

    /**
     * The value of the mask is 1 for pixels with a valid depth measurement
     * (not 0) and 0 elsewhere. Its type is se::mask_t.
     */
    cv::Mat valid_depth_mask_;

    /**
     * The value of the mask corresponds to the instance ID of the object that
     * was raycasted at the corresponding pixel. Its type is se::instance_mask_t.
     */
    cv::Mat raycasted_instance_mask_;
    cv::Mat occlusion_mask_;

    se::Image<Eigen::Vector3f> object_surface_point_cloud_M_;
    se::Image<Eigen::Vector3f> object_surface_normals_M_;

    /**
     * Contains the scale at which each pixel was hit in the last raycasting.
     */
    se::Image<int8_t> object_scale_image_;
    se::Image<int8_t> object_min_scale_image_;

    /**
     * IOU threshold in [0, 1] to consider two masks matched.
     */
    static constexpr float iou_threshold_ = 0.25;

    /**
     * Small mask removal threshold in [0, 1].
     */
    static constexpr float small_mask_threshold_ = 0.02;

    /**
     * Minimum acceptable confidence for class detections [0, 1].
     */
    static constexpr float class_confidence_threshold_ = 0.75;

    // Exploration only ///////////////////////////////////////////////////////
    std::set<se::key_t> updated_nodes_;
    std::set<se::key_t> frontiers_;
    const Eigen::Vector3f aabb_min_M_;
    const Eigen::Vector3f aabb_max_M_;
    const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> aabb_edges_M_;



    /**
     * Given a mask, compute the appropriate volume extent and size as well as
     * its pose. Return volume extent and volume size of 0 if the object is
     * invalid (e.g. if the masked region has no valid depth meausurements.
     * Also compute the object bounding sphere.
     *
     * \note depth2vertexKernel() must have been called so that
     * DenseSLAMSystem::input_point_cloud_C_[0] contains the vertex map
     * produced from the current depth frame before calling this function.
     *
     * \todo TODO Improve the heuristics used for the object size.
     */
    void computeNewObjectParameters(Eigen::Matrix4f& T_OM,
                                    int& volume_size,
                                    float& volume_extent,
                                    const cv::Mat& mask,
                                    const int class_id,
                                    const Eigen::Matrix4f& T_MC);

    /**
     * Compare the current detections with the raycasted_object_mask_ created
     * by DenseSLAMSystem::raycastObjectsAndBg() and match them using IoU
     * (Intersection over Union).  When one of the current detections is
     * matched to a mask of an existing object, set its instance ID to that of
     * the object.
     *
     * \note The raycasted_object_mask_ must have been created by raycasting
     * the current object list using DenseSLAMSystem::raycastObjectsAndBg().
     */
    void matchObjectInstances(se::SegmentationResult& detections, const float matching_threshold);

    /**
     * Generate new objects based on the segmentation masks. Only generate
     * objects for the masks whose respective instance ids are not already in
     * object_list_. The instance_id_s of masks are updated where needed to
     * reflect the instance IDs of the new objects.
     *
     * \param[in] masks The object masks resulting from the segmentation.
     * \param[in] sensor
     * ::Configuration.camera for details.
     */
    void generateObjects(se::SegmentationResult& masks, const SensorImpl& sensor);

    /**
     * Update DenseSLAMSystem::valid_depth_mask_ to contain 255 at pixels with valid depth
     * measurements within the near and far planes and 0 everywhere else.
     */
    void updateValidDepthMask(const se::Image<float>& depth, const SensorImpl& sensor);

    void generateUndetectedInstances(se::SegmentationResult& detections);



    public:
    /**
     * Constructor using the initial camera position.
     *
     * \param[in] input_res The size (width and height) of the input frames.
     * \param[in] map_size_ The x, y and z resolution of the
     * reconstructed volume in voxels.
     * \param[in] map_dim_ The x, y and z dimensions of the
     * reconstructed volume in meters.
     * \param[in] t_MW The x, y and z coordinates of the world to map frame translation.
     * The map frame rotation is assumed to be aligned with the world frame.
     * \param[in] pyramid See se::Configuration.pyramid for more details.
     * \param[in] config_ The pipeline options.
     */
    DenseSLAMSystem(const Eigen::Vector2i& image_res,
                    const Eigen::Vector3i& map_size,
                    const Eigen::Vector3f& map_dim,
                    const Eigen::Vector3f& t_MW,
                    std::vector<int>& pyramid,
                    const se::Configuration& config,
                    const std::string voxel_impl_yaml_path = "");
    /**
     * Constructor using the initial camera position.
     *
     * \param[in] input_res The size (width and height) of the input frames.
     * \param[in] map_size_ The x, y and z resolution of the
     * reconstructed volume in voxels.
     * \param[in] map_dim_ The x, y and z dimensions of the
     * reconstructed volume in meters.
     * \param[in] T_MW The world to map frame transformation encoded in a 4x4 matrix.
     * \param[in] pyramid See se::Configuration.pyramid for more details.
     * \param[in] config_ The pipeline options.
     */
    DenseSLAMSystem(const Eigen::Vector2i& image_res,
                    const Eigen::Vector3i& map_size,
                    const Eigen::Vector3f& map_dim,
                    const Eigen::Matrix4f& T_MW,
                    std::vector<int>& pyramid,
                    const se::Configuration& config,
                    const std::string voxel_impl_yaml_path = "");

    /**
     * Preprocess a single depth frame and add it to the pipeline.
     * This is the first stage of the pipeline.
     *
     * \param[in] input_depth Pointer to the depth frame data. Each pixel is
     * represented by a single float containing the depth value in meters.
     * \param[in] input_res Size of the depth frame in pixels (width and
     * height).
     * \param[in] filter_depth Whether to filter the depth frame using a
     * bilateral filter to reduce the measurement noise.
     * \return true (does not fail).
     */
    bool preprocessDepth(const float* input_depth_image_data,
                         const Eigen::Vector2i& input_depth_image_res,
                         const bool filter_depth_image);

    /**
     * Preprocess an RGBA frame and add it to the pipeline.
     * This is the first stage of the pipeline.
     *
     * \param[in] input_RGBA Pointer to the RGBA frame data, 4 channels, 8
     * bits per channel.
     * \param[in] input_res Size of the depth and RGBA frames in pixels
     * (width and height).
     * \param[in] filter_depth Whether to filter the depth frame using a
     * bilateral filter to reduce the measurement noise.
     * \return true (does not fail).
     */
    bool preprocessColor(const uint32_t* input_RGBA_image_data,
                         const Eigen::Vector2i& input_RGBA_image_res);

    /**
     * Update the camera pose. Create a 3D reconstruction from the current
     * depth frame and compute a transformation using ICP. The 3D
     * reconstruction of the current frame is matched to the 3D reconstruction
     * obtained from all of the previous frames. This is the second stage of
     * the pipeline.
     *
	 * \param[in] k The intrinsic camera parameters. See
	 * se::Configuration.camera for details.
     * \param[in] icp_threshold The ICP convergence threshold.
     * \return true if the camera pose was updated and false if it wasn't.
     */
    bool track(const SensorImpl& sensor, const float icp_threshold);

    /**
     * Integrate the 3D reconstruction resulting from the current frame to the
     * existing reconstruction. This is the third stage of the pipeline.
     *
     * \param[in] k The intrinsic camera parameters. See
     * se::Configuration.camera for details.
     * \param[in] mu TSDF truncation bound. See se::Configuration.mu for more
     * details.
     * \param[in] frame The index of the current frame (starts from 0).
     * \return true (does not fail).
     */
    bool integrate(const SensorImpl& sensor, const unsigned frame);

    /**
     * \brief Raycast the map from the current pose to create a point cloud (point cloud map)
     *        and respective normal vectors (normal map). The point cloud and normal
     *        maps are then used to track the next frame in DenseSLAMSystem::tracking.
     *        This is the fourth stage of the pipeline.
     *
     * \note Raycast is not performed on the first 3 frames (those with an index up to 2).
     *
     * \param[in] k  The intrinsic camera parameters. See ::Configuration.camera for details.
     * \param[in] mu TSDF truncation bound. See ::Configuration.mu for more details.
     *
     * \return true (does not fail).
     */
    bool raycast(const SensorImpl& sensor);

    /**
     * \brief Extract the mesh from the map using a marching cube algorithm and save it to a file.
     *
     * \param[in] filename The file where the mesh will be saved. The file format will be selected
     *                     based on the file extension. Allowed file extensions are `.ply`, `.vtk` and
     *                     `.obj`.
     * \param[in] T_WM     Transformation from the world frame where the mesh is generated to the world
     *                     frame. Defaults to identity.
     * \return Zero on success and non-zero on error.
     */
    int saveMesh(const std::string& filename,
                 const Eigen::Matrix4f& T_FW = Eigen::Matrix4f::Identity()) const;

    std::vector<se::Triangle>
    triangleMeshV(const se::meshing::ScaleMode scale_mode = se::meshing::ScaleMode::Current);

    /** \brief Export the octree structure and slices.
     *
     * \param[in] base_filename The base name of the file without suffix.
     */
    void saveStructure(const std::string base_filename);

    bool saveThresholdSliceZ(const std::string filename, const float z_M);

    void structureStats(size_t& num_nodes,
                        size_t& num_blocks,
                        std::vector<size_t>& num_blocks_per_scale);

    /** \brief Render the current 3D reconstruction.
     * This function performs raycasting if needed, otherwise it uses the point
     * cloud and normal maps created in DenseSLAMSystem::raycasting.
     *
     * \param[out] output_image_data A pointer to an array where the image will
     *                               be rendered. The array must be allocated
     *                               before calling this function, one uint32_t
     *                               per pixel.
     * \param[in] output_image_res   The dimensions of the output array (width
     *                               and height in pixels).
     */
    void renderVolume(uint32_t* output_image_data,
                      const Eigen::Vector2i& output_image_res,
                      const SensorImpl& sensor,
                      const bool render_scale = false);

    /**
     * Render the output of the tracking algorithm. The meaning of the colors
     * is as follows:
     *
     * | Color  | Meaning |
     * | ------ | ------- |
     * | grey   | Successful tracking. |
     * | black  | No input data. |
     * | red    | Not in image. |
     * | green  | No correspondence. |
     * | blue   | Too far away. |
     * | yellow | Wrong normal. |
     * | orange | Tracking not performed. |
     *
     * \param[out] output_image_data A pointer to an array where the image will
     *                               be rendered. The array must be allocated
     *                               before calling this function, one uint32_t
     *                               per pixel.
     * \param[in] output_image_res   The dimensions of the output array (width
     *                               and height in pixels).
     */
    void renderTrack(uint32_t* output_image_data, const Eigen::Vector2i& output_image_res);

    /**
	 * Render the current depth frame. The frame is rendered before
	 * preprocessing while taking into account the values of
	 * se::Configuration::near_plane and se::Configuration::far_plane. Regions
	 * closer to the camera than se::Configuration::near_plane appear white and
	 * regions further than se::Configuration::far_plane appear black.
     *
     * \param[out] output_image_data A pointer to an array where the image will
     *                               be rendered. The array must be allocated
     *                               before calling this function, one uint32_t
     *                               per pixel.
     * \param[in] output_image_res   The dimensions of the output array (width
     *                               and height in pixels).
     */
    void renderDepth(uint32_t* output_image_data,
                     const Eigen::Vector2i& output_image_res,
                     const SensorImpl& sensor);

    /**
     * Render the RGB frame currently in the pipeline.
     *
     * \param[out] output_RGBA_image_data A pointer to an array where the image
     *                                    will be rendered. The array must be
     *                                    allocated before calling this
     *                                    function, one uint32_t per pixel.
     * \param[in] output_RGBA_image_res   The dimensions of the output image
     *                                    (width and height in pixels).
     */
    void renderRGBA(uint32_t* output_RGBA_image_data, const Eigen::Vector2i& output_RGBA_image_res);

    //
    // Getters
    //

    /*
     * TODO Document this.
     */
    std::shared_ptr<se::Octree<VoxelImpl::VoxelType>> getMap()
    {
        const std::lock_guard<std::recursive_mutex> lock(mutex_);
        return map_;
    }

    std::shared_ptr<const se::Octree<VoxelImpl::VoxelType>> getMap() const
    {
        const std::lock_guard<std::recursive_mutex> lock(mutex_);
        return map_;
    }

    void saveMap(const std::string& map_filename)
    {
        const std::lock_guard<std::recursive_mutex> lock(mutex_);
        map_->save(map_filename);
    }

    void loadMap(const std::string& map_filename)
    {
        const std::lock_guard<std::recursive_mutex> lock(mutex_);
        map_->load(map_filename);
    }

    template<typename ValueSelector>
    void saveOctoMapBinary(const std::string& octomap_binary_filename,
                           const float threshold,
                           ValueSelector value_selector,
                           const int x_lb = 0,
                           const int y_lb = 0,
                           const int z_lb = 0,
                           int x_ub = 0,
                           int y_ub = 0,
                           int z_ub = 0)
    {
        const std::lock_guard<std::recursive_mutex> lock(mutex_);
        // Initialise default boundaries. Can't be set above as map_size_ is non-static.
        x_ub = (x_ub == 0) ? map_size_.x() : x_ub;
        y_ub = (y_ub == 0) ? map_size_.y() : y_ub;
        z_ub = (z_ub == 0) ? map_size_.z() : z_ub;

        // Check if boundaries are valid
        if (x_ub <= x_lb || y_ub <= y_lb || z_ub <= z_lb) {
            return;
        }

        // Convert boundaries to meter units
        const float x_lb_m = x_lb * map_->voxelDim();
        const float y_lb_m = y_lb * map_->voxelDim();
        const float z_lb_m = z_lb * map_->voxelDim();
        const float x_ub_m = x_ub * map_->voxelDim();
        const float y_ub_m = y_ub * map_->voxelDim();
        const float z_ub_m = z_ub * map_->voxelDim();

        // Create a lambda function to set the state of a single voxel.
        const auto set_node_value =
            [&](octomap::OcTree& octomap,
                const octomap::point3d& voxel_coord,
                const typename VoxelImpl::VoxelType::VoxelData& voxel_data) {
                // Check if the voxel is in the boundaries
                if (voxel_coord.x() >= x_lb_m && voxel_coord.y() >= y_lb_m
                    && voxel_coord.z() >= z_lb_m && voxel_coord.x() < x_ub_m
                    && voxel_coord.y() < y_ub_m && voxel_coord.z() < z_ub_m) {
                    // Do not update unknown voxels.
                    if (VoxelImpl::VoxelType::isValid(voxel_data)) {
                        if (value_selector(voxel_data) < threshold) {
                            // Free
                            octomap.updateNode(voxel_coord, false, false);
                        }
                        else {
                            octomap.updateNode(voxel_coord, true, false);
                        }
                    }
                }
            };

        octomap::OcTree* octomap = se::to_octomap(*map_, set_node_value);
        octomap->writeBinary(octomap_binary_filename);
        delete octomap;
    }

    /**
     * Get the translation of the world frame to the map frame.
     *
     * \return A vector containing the x, y and z coordinates of the translation.
     */
    Eigen::Vector3f t_MW() const
    {
        const std::lock_guard<std::recursive_mutex> lock(mutex_);
        return se::math::to_translation(T_MW_);
    }

    /**
     * Get the transformation of the world frame to map frame.
     *
     * \return The rotation (3x3 rotation matrix) and translation (3x1 vector) encoded in a 4x4 matrix.
     */
    Eigen::Matrix4f T_MW() const
    {
        const std::lock_guard<std::recursive_mutex> lock(mutex_);
        return T_MW_;
    }

    /**
     * Get the translation of the map frame to world frame.
     *
     * \return A vector containing the x, y and z coordinates of the translation.
     */
    Eigen::Vector3f t_WM() const
    {
        const std::lock_guard<std::recursive_mutex> lock(mutex_);
        Eigen::Vector3f t_WM = se::math::to_inverse_translation(T_MW_);
        return t_WM;
    }

    /**
     * Get the transformation of the map frame to world frame.
     *
     * \return The rotation (3x3 rotation matrix) and translation (3x1 vector) encoded in a 4x4 matrix.
     */
    Eigen::Matrix4f T_WM() const
    {
        Eigen::Matrix4f T_WM = se::math::to_inverse_transformation(T_MW_);
        return T_WM;
    }

    /**
     * Get the current camera position in map frame.
     *
     * \return A vector containing the x, y and z coordinates t_MC.
     */
    Eigen::Vector3f t_MC() const
    {
        const std::lock_guard<std::recursive_mutex> lock(mutex_);
        return se::math::to_translation(T_MC_);
    }

    /**
     * Get the current camera position in world frame.
     *
     * \return A vector containing the x, y and z coordinates of t_WC.
     */
    Eigen::Vector3f t_WC() const
    {
        const std::lock_guard<std::recursive_mutex> lock(mutex_);
        Eigen::Matrix4f T_WC = se::math::to_inverse_transformation(T_MW_) * T_MC_;
        Eigen::Vector3f t_WC = se::math::to_translation(T_WC);
        return t_WC;
    }

    /**
     * Get the initial camera position in map frame.
     *
     * \return A vector containing the x, y and z coordinates of initt_MC.
     */
    Eigen::Vector3f initt_MC() const
    {
        const std::lock_guard<std::recursive_mutex> lock(mutex_);
        return se::math::to_translation(T_MC_);
    }

    /**
     * Get the initial camera position in world frame.
     *
     * \return A vector containing the x, y and z coordinates of initt_WC.
     */
    Eigen::Vector3f initt_WC() const
    {
        const std::lock_guard<std::recursive_mutex> lock(mutex_);
        Eigen::Matrix4f init_T_WC = se::math::to_inverse_transformation(T_MW_) * init_T_MC_;
        Eigen::Vector3f initt_WC = se::math::to_translation(init_T_WC);
        return initt_WC;
    }

    /**
     * Get the current camera pose in map frame.
     *
     * \return The current camera pose T_MC encoded in a 4x4 matrix.
     */
    Eigen::Matrix4f T_MC() const
    {
        const std::lock_guard<std::recursive_mutex> lock(mutex_);
        return T_MC_;
    }

    /**
     * Get the current camera pose in world frame.
     *
     * \return The current camera pose T_MC encoded in a 4x4 matrix.
     */
    Eigen::Matrix4f T_WC() const
    {
        const std::lock_guard<std::recursive_mutex> lock(mutex_);
        Eigen::Matrix4f T_WC = se::math::to_inverse_transformation(T_MW_) * T_MC_;
        return T_WC;
    }

    /**
     * Get the initial camera pose in map frame.
     *
     * \return The initial camera pose init_T_MC_ encoded in a 4x4 matrix.
     */
    Eigen::Matrix4f initT_MC() const
    {
        const std::lock_guard<std::recursive_mutex> lock(mutex_);
        return init_T_MC_;
    }

    /**
     * Get the inital camera pose in world frame.
     *
     * \return The initial camera pose T_MC encoded in a 4x4 matrix.
     */
    Eigen::Matrix4f initT_WC() const
    {
        const std::lock_guard<std::recursive_mutex> lock(mutex_);
        Eigen::Matrix4f init_T_WC = se::math::to_inverse_transformation(T_MW_) * init_T_MC_;
        return init_T_WC;
    }

    /**
     * Set the current camera pose provided in map frame.
     *
     * \param[in] T_MC The desired camera pose encoded in a 4x4 matrix.
     */
    void setT_MC(const Eigen::Matrix4f& T_MC)
    {
        const std::lock_guard<std::recursive_mutex> lock(mutex_);
        T_MC_ = T_MC;
    }

    /**
     * Set the current camera pose provided in world frame.
     *
     * @note T_MW_ is added to the pose to process the information further in map frame.
     *
     * \param[in] T_WC The desired camera pose encoded in a 4x4 matrix.
     */
    void setT_WC(const Eigen::Matrix4f& T_WC)
    {
        const std::lock_guard<std::recursive_mutex> lock(mutex_);
        T_MC_ = T_MW_ * T_WC;
    }

    /**
     * Set the initial camera pose provided in map frame.
     *
     * \param[in] init_T_MC The initial camera pose encoded in a 4x4 matrix.
     */
    void setInitT_MC(const Eigen::Matrix4f& init_T_MC)
    {
        init_T_MC_ = init_T_MC;
    }

    /**
     * Set the initial camera pose provided in world frame.
     *
     * @note T_MW_ is added to the pose to process the information further in map frame.
     *
     * \param[in] init_T_WC The initial camera pose encoded in a 4x4 matrix.
     */
    void setInitT_WC(const Eigen::Matrix4f& init_T_WC)
    {
        const std::lock_guard<std::recursive_mutex> lock(mutex_);
        init_T_MC_ = T_MW_ * init_T_WC;
    }

    /**
     * Set the camera pose used to render the 3D reconstruction.
     *
     * \param[in] T_WC The desired camera pose encoded in a 4x4 matrix.
     */
    void setRenderT_MC(Eigen::Matrix4f* render_T_MC = nullptr)
    {
        const std::lock_guard<std::recursive_mutex> lock(mutex_);
        if (render_T_MC == nullptr) {
            render_T_MC_ = &T_MC_;
            need_render_ = false;
        }
        else {
            render_T_MC_ = render_T_MC;
            need_render_ = true;
        }
    }

    /**
     * Get the camera pose used to render the 3D reconstruction. The default
     * is the current frame's camera pose.
     *
     * @note The view pose is currently only provided in map frame.
     *
     * \return The current rendering camera pose render_T_MC encoded in a 4x4 matrix.
     */
    Eigen::Matrix4f* renderT_MC_() const
    {
        const std::lock_guard<std::recursive_mutex> lock(mutex_);
        return render_T_MC_;
    }

    /**
     * Get the dimensions of the reconstructed volume in meters.
     *
     * \return A vector containing the x, y and z dimensions of the volume.
     */
    Eigen::Vector3f getMapDimension() const
    {
        const std::lock_guard<std::recursive_mutex> lock(mutex_);
        return (map_dim_);
    }

    /**
     * Get the size of the reconstructed volume in voxels.
     *
     * \return A vector containing the x, y and z resolution of the volume.
     */
    Eigen::Vector3i getMapSize() const
    {
        const std::lock_guard<std::recursive_mutex> lock(mutex_);
        return (map_size_);
    }

    /**
     * Get the resolution used when processing frames in the pipeline in
     * pixels.
     *
     * \return A vector containing the frame width and height.
     */
    Eigen::Vector2i getImageResolution() const
    {
        const std::lock_guard<std::recursive_mutex> lock(mutex_);
        return (image_res_);
    }

    const se::Image<float>& getDepth() const;



    // Semanticeight-only /////////////////////////////////////////////////////
    /**
     * Add the semantic masks for the RGB image to the pipeline. This is the
     * first stage of the pipeline.
     *
     * \param[in] segmentation The segmentation output for the RGB frame.
     * \return true (does not fail).
     */
    bool preprocessSegmentation(const se::SegmentationResult& segmentation);

    /**
     * Process the semantic masks to create masks suitable for merging the
     * detected objects. This is the third stage of the pipeline.
     *
     * \param[in] sensor
     */
    bool trackObjects(const SensorImpl& sensor, const int frame);

    /**
     * Integrate the 3D reconstruction resulting from the current frame to the
     * existing reconstruction. This integrates all objects detected in the
     * current frame. This is the fourth stage of the pipeline.
     *
     * \param[in] sensor
     * \param[in] frame The index of the current frame (starts from 0).
     * \return true if the current 3D reconstruction was added to the octree
     * and false if it wasn't.
     */
    bool integrateObjects(const SensorImpl& sensor, const size_t frame);

    /**
     * Raycast the 3D reconstruction after integration to generate the vertex
     * and normal maps for each object from the current pose. This is the fifth
     * stage of the pipeline.
     *
     * \note Raycast is not performed on the first 3 frames (those with an
     * index up to 2).
     *
     * \param[in] sensor
     * \return true if raycasting was performed and false if it wasn't.
     */
    bool raycastObjectsAndBg(const SensorImpl& sensor, const int frame);

    void renderInputSegmentation(uint32_t* image_data, const Eigen::Vector2i& image_res)
    {
        const std::lock_guard<std::recursive_mutex> lock(mutex_);
        renderMaskKernel<se::class_mask_elem_t>(
            image_data, image_res, rgba_image_, input_segmentation_.classMask());
    }

    /**
     * Render the current 3D reconstruction.
     *
     * \param[out] output_image_data A pointer to an array where the image will
     *                               be rendered. The array must be allocated
     *                               before calling this function, one uint32_t
     *                               per pixel.
     * \param[in] output_image_res   The dimensions of the output array (width
     *                               and height in pixels).
     * \param[in] sensor
     * details.
     */
    void renderObjects(uint32_t* output_image_data,
                       const Eigen::Vector2i& output_image_res,
                       const SensorImpl& sensor,
                       const RenderMode render_mode = RenderMode::InstanceID,
                       const bool render_bounding_volumes = true);

    /**
     * Render the predicted class of each object overlaid on the current RGB
     * frame with a different colour.
     *
     * \param[out] output_RGBA_image_data A pointer to an array where the image
     *                                    will be rendered. The array must be
     *                                    allocated before calling this
     *                                    function, one uint32_t per pixel.
     * \param[in] output_RGBA_image_res   The dimensions of the output image
     *                                    (width and height in pixels).
     */
    void renderObjectClasses(uint32_t* output_image_data,
                             const Eigen::Vector2i& output_image_res) const;

    /**
     * Render the object instances blended with the current RGB frame.
     *
     * \param[out] output_RGBA_image_data A pointer to an array where the image
     *                                    will be rendered. The array must be
     *                                    allocated before calling this
     *                                    function, one uint32_t per pixel.
     * \param[in] output_RGBA_image_res   The dimensions of the output image
     *                                    (width and height in pixels).
     */
    void renderObjectInstances(uint32_t* output_image_data,
                               const Eigen::Vector2i& output_image_res) const;

    /**
     * Render the object instance mask produced by raycasting blended with the
     * current RGB frame.
     *
     * \param[out] output_image_data Pointer to an array containing the
     * rendered frame, 4 channels, 8 bits per channel. The array must be
     * allocated before calling this function.
     * \param[in] output_image_res The dimensions of the output image (width
     * and height in pixels).
     */
    void renderRaycast(uint32_t* output_image_data, const Eigen::Vector2i& output_image_res);

    void saveObjectMeshes(const std::string& filename,
                          const Eigen::Matrix4f& T_FW = Eigen::Matrix4f::Identity()) const;

    std::vector<std::vector<se::Triangle>>
    objectTriangleMeshesV(const se::meshing::ScaleMode scale_mode = se::meshing::ScaleMode::Min);

    Objects& getObjectMaps()
    {
        const std::lock_guard<std::recursive_mutex> lock(mutex_);
        return objects_;
    }

    const Objects& getObjectMaps() const
    {
        const std::lock_guard<std::recursive_mutex> lock(mutex_);
        return objects_;
    }

    // Exploration only ///////////////////////////////////////////////////////
    /** Set cylinder whose curved face is just inside the camera frustum to free.
     * This is used to create some free space around the robot's starting position.
     * The frontier status of voxels in the cylinder is also set correctly.
     */
    void freeCylinder(const Eigen::Vector3f& centre_M,
                      const float radius_M,
                      const float height_M);

    void freeSphere(const Eigen::Vector3f& centre_M, const float radius_M);

    /**
     * Create some free space around the robot's starting position.
     * The frontier status of the free voxels is set correctly.
     */
    void freeInitialPosition(const SensorImpl& sensor, const std::string& type = "cylinder");

    std::set<se::key_t> getFrontiers() const
    {
        const std::lock_guard<std::recursive_mutex> lock(mutex_);
        return frontiers_;
    }

    const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>&
    environmentAABBEdgesM() const;

    std::vector<se::Volume<VoxelImpl::VoxelType>> frontierVolumes() const;

    float free_volume = 0.0f;
    float occupied_volume = 0.0f;
    float explored_volume = 0.0f;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif
