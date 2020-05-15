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

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>

#include <Eigen/Dense>

#include "se/commons.h"
#include "se/perfstats.h"
#include "se/timings.h"
#include "se/config.h"
#include "se/octree.hpp"
#include "se/image/image.hpp"
#include "se/continuous/volume_template.hpp"
#include "se/sensor_implementation.hpp"
#include "voxel_implementations.hpp"
#include "preprocessing.hpp"
#include "tracking.hpp"



template <typename T>
using Volume = VolumeTemplate<T, se::Octree>;

class DenseSLAMSystem {

  private:
    Eigen::Vector2i image_res_;
    Eigen::Matrix4f T_MW_;        // World to camera transformation; Convert in/output to map/world frame
    Eigen::Matrix4f T_MC_;        // Camera pose in map frame
    Eigen::Matrix4f* render_T_MC_; // Rendering camera pose in map frame
    Eigen::Matrix4f init_T_MC_;   // Initial camera pose in map frame
    Eigen::Vector3f map_dim_;
    Eigen::Vector3i map_size_;
    std::vector<int> iterations_;
    bool tracked_;
    bool integrated_;
    bool need_render_ = false;
    Configuration config_;

    // input once
    std::vector<float> gaussian_;

    // inter-frame
    se::Image<Eigen::Vector3f> surface_point_cloud_M_;
    se::Image<Eigen::Vector3f> surface_normals_M_;

    std::vector<se::key_t> allocation_list_;
    std::shared_ptr<se::Octree<VoxelImpl::VoxelType> > map_;
    Volume<VoxelImpl> volume_;

    // intra-frame
    std::vector<float> reduction_output_;
    std::vector<se::Image<float>  > scaled_depth_image_;
    std::vector<se::Image<Eigen::Vector3f> > input_point_cloud_C_;
    std::vector<se::Image<Eigen::Vector3f> > input_normals_C_;
    se::Image<float> depth_image_;
    se::Image<uint32_t> rgba_image_;
    std::vector<TrackData>  tracking_result_;
    Eigen::Matrix4f previous_T_MC_;
    Eigen::Matrix4f raycast_T_MC_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

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
     * \param[in] pyramid See ::Configuration.pyramid for more details.
     * \param[in] config_ The pipeline options.
     */
    DenseSLAMSystem(const Eigen::Vector2i& image_res,
                    const Eigen::Vector3i& map_size_,
                    const Eigen::Vector3f& map_dim_,
                    const Eigen::Vector3f& t_MW,
                    std::vector<int> &     pyramid,
                    const Configuration&   config_);
    /**
     * Constructor using the initial camera position.
     *
     * \param[in] input_res The size (width and height) of the input frames.
     * \param[in] map_size_ The x, y and z resolution of the
     * reconstructed volume in voxels.
     * \param[in] map_dim_ The x, y and z dimensions of the
     * reconstructed volume in meters.
     * \param[in] T_MW The world to map frame transformation encoded in a 4x4 matrix.
     * \param[in] pyramid See ::Configuration.pyramid for more details.
     * \param[in] config_ The pipeline options.
     */
    DenseSLAMSystem(const Eigen::Vector2i& image_res,
                    const Eigen::Vector3i& map_size_,
                    const Eigen::Vector3f& map_dim_,
                    const Eigen::Matrix4f& T_MW,
                    std::vector<int> &     pyramid,
                    const Configuration&   config_);

    ~DenseSLAMSystem() {
        delete render_T_MC_;
    }

    /**
     * Preprocess a single depth frame and add it to the pipeline.
     * This is the first stage of the pipeline.
     *
     * \param[in] input_depth Pointer to the depth frame data. Each pixel is
     * represented by a single uint16_t.
     * \param[in] input_res Size of the depth frame in pixels (width and
     * height).
     * \param[in] filter_depth Whether to filter the depth frame using a
     * bilateral filter to reduce the measurement noise.
     * \return true (does not fail).
     */
    bool preprocessDepth(const uint16_t*        input_depth_image_data,
                         const Eigen::Vector2i& input_depth_image_res,
                         const bool             filter_depth_image);

    /**
     * Preprocess an RGB frame and add it to the pipeline.
     * This is the first stage of the pipeline.
     *
     * \param[in] input_RGB Pointer to the RGB frame data, 3 channels, 8 bits
     * per channel.
     * \param[in] input_res Size of the depth and RGB frames in pixels (width
     * and height).
     * \param[in] filter_depth Whether to filter the depth frame using a
     * bilateral filter to reduce the measurement noise.
     * \return true (does not fail).
     */
    bool preprocessColor(const uint8_t*         input_RGB_image_data,
                         const Eigen::Vector2i& input_RGB_image_res);

    /**
     * Update the camera pose. Create a 3D reconstruction from the current
     * depth frame and compute a transformation using ICP. The 3D
     * reconstruction of the current frame is matched to the 3D reconstruction
     * obtained from all of the previous frames. This is the second stage of
     * the pipeline.
     *
     * \param[in] k The intrinsic camera parameters. See ::Configuration.camera
     * for details.
     * \param[in] icp_threshold The ICP convergence threshold.
     * \return true if the camera pose was updated and false if it wasn't.
     */
    bool track(const SensorImpl& sensor,
               const float       icp_threshold);

    /**
     * Integrate the 3D reconstruction resulting from the current frame to the
     * existing reconstruction. This is the third stage of the pipeline.
     *
     * \param[in] k The intrinsic camera parameters. See
     * ::Configuration.camera for details.
     * \param[in] mu TSDF truncation bound. See ::Configuration.mu for more
     * details.
     * \param[in] frame The index of the current frame (starts from 0).
     * \return true (does not fail).
     */
    bool integrate(const SensorImpl& sensor,
                   const unsigned    frame);

    /**
     * Raycast the map from the current pose to create a point cloud (point cloud
     * map) and respective normal vectors (normal map). The point cloud and normal
     * maps are then used to track the next frame in DenseSLAMSystem::tracking.
     * This is the fourth stage of the pipeline.
     *
     * @note Raycast is not performed on the first 3 frames (those with an
     * index up to 2).
     *
     * \param[in] k The intrinsic camera parameters. See
     * ::Configuration.camera for details.
     * \param[in] mu TSDF truncation bound. See ::Configuration.mu for more
     * details.
     * \return true (does not fail).
     */
    bool raycast(const SensorImpl& sensor);

    /*
     * TODO Implement this.
     */
    void dump_volume(const std::string filename);

    /*
     * TODO Document this.
     */
    void dump_mesh(const std::string filename);

    /**
     * Render the current 3D reconstruction. This function performs raycasting
     * if needed, otherwise it uses the point cloud and normal maps created in
     * DenseSLAMSystem::raycasting.
     *
     * \param[out] output_image_data A pointer to an array containing the rendered frame.
     * The array must be allocated before calling this function. The storage
     * layout is rgbwrgbwrgbw.
     * \param[in] output_image_res The dimensions of the output array (width and
     * height in pixels).
     */
    void renderVolume(unsigned char*         output_image_data,
                      const Eigen::Vector2i& output_image_res,
                      const SensorImpl&      sensor);

    /**
     * Render the output of the tracking algorithm. The meaning of the colors is as follows:
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
     * \param[out] output_image_data A pointer to an array containing the rendered frame.
     * The array must be allocated before calling this function. The x, y and
     * z members of each element of the array contain the R, G and B values of
     * the image respectively. The w member of each element of the array is
     * always 0 and is used for padding.
     * \param[in] output_res The dimensions of the output array (width and
     * height in pixels).
     */
    void renderTrack(unsigned char*         output_image_data,
                     const Eigen::Vector2i& output_image_res);

    /**
     * Render the current depth frame. The frame is rendered before
     * preprocessing while taking into account the values of ::nearPlane and
     * ::farPlane. Regions closer to the camera than ::nearPlane appear white
     * and regions further than ::farPlane appear black.
     *
     * \param[out] output_image_data A pointer to an array containing the rendered frame.
     * The array must be allocated before calling this function. The x, y and
     * z members of each element of the array contain the R, G and B values of
     * the image respectively. The w member of each element of the array is
     * always 0 and is used for padding.
     * \param[in] output_image_res The dimensions of the output array (width and
     * height in pixels).
     */
    void renderDepth(unsigned char*         output_image_data,
                     const Eigen::Vector2i& output_image_res,
                     const SensorImpl&      sensor);

    /**
     * Render the RGB frame currently in the pipeline.
     *
     * \param[out] output_RGBA_image_data Pointer to an array containing the rendered
     * frame, 4 channels, 8 bits per channel. The array must be allocated
     * before calling this function.
     * \param[in] output_RGBA_image_res The dimensions of the output image (width and
     * height in pixels).
     */
    void renderRGBA(uint8_t*               output_RGBA_image_data,
                    const Eigen::Vector2i& output_RGBA_image_res);

    //
    // Getters
    //

    /*
     * TODO Document this.
     */
    void getMap(std::shared_ptr<se::Octree<VoxelImpl::VoxelType> >& map) {
      map = map_;
    }

    /*
     * TODO Document this.
     */
    bool getTracked() {
      return (tracked_);
    }

    /*
     * TODO Document this.
     */
    bool getIntegrated() {
      return (integrated_);
    }

    /**
     * Get the translation of the world frame to the map frame.
     *
     * \return A vector containing the x, y and z coordinates of the translation.
     */
    Eigen::Vector3f t_MW() {
      return se::math::to_translation(T_MW_);
    }

    /**
     * Get the transformation of the world frame to map frame.
     *
     * \return The rotation (3x3 rotation matrix) and translation (3x1 vector) encoded in a 4x4 matrix.
     */
    Eigen::Matrix4f T_MW() {
      return T_MW_;
    }

    /**
     * Get the translation of the map frame to world frame.
     *
     * \return A vector containing the x, y and z coordinates of the translation.
     */
    Eigen::Vector3f t_WM() {
      Eigen::Vector3f t_WM = se::math::to_inverse_translation(T_MW_);
      return t_WM;
    }

    /**
     * Get the transformation of the map frame to world frame.
     *
     * \return The rotation (3x3 rotation matrix) and translation (3x1 vector) encoded in a 4x4 matrix.
     */
    Eigen::Matrix4f T_WM() {
      Eigen::Matrix4f T_WM = se::math::to_inverse_transformation(T_MW_);
      return T_WM;
    }

    /**
     * Get the current camera position in map frame.
     *
     * \return A vector containing the x, y and z coordinates t_MC.
     */
    Eigen::Vector3f t_MC() {
      return se::math::to_translation(T_MC_);
    }

    /**
     * Get the current camera position in world frame.
     *
     * \return A vector containing the x, y and z coordinates of t_WC.
     */
    Eigen::Vector3f t_WC() {
      Eigen::Matrix4f T_WC = se::math::to_inverse_transformation(T_MW_) * init_T_MC_;
      Eigen::Vector3f t_WC = se::math::to_translation(T_WC);
      return t_WC;
    }

    /**
     * Get the initial camera position in map frame.
     *
     * \return A vector containing the x, y and z coordinates of initt_MC.
     */
    Eigen::Vector3f initt_MC(){
      return se::math::to_translation(T_MC_);
    }

    /**
     * Get the initial camera position in world frame.
     *
     * \return A vector containing the x, y and z coordinates of initt_WC.
     */
    Eigen::Vector3f initt_WC(){
      Eigen::Matrix4f init_T_WC = se::math::to_inverse_transformation(T_MW_) * init_T_MC_;
      Eigen::Vector3f initt_WC = se::math::to_translation(init_T_WC);
      return initt_WC;
    }

    /**
     * Get the current camera pose in map frame.
     *
     * \return The current camera pose T_MC encoded in a 4x4 matrix.
     */
    Eigen::Matrix4f T_MC() {
      return T_MC_;
    }

    /**
     * Get the current camera pose in world frame.
     *
     * \return The current camera pose T_MC encoded in a 4x4 matrix.
     */
    Eigen::Matrix4f T_WC() {
      Eigen::Matrix4f T_WC = se::math::to_inverse_transformation(T_MW_) * T_MC_;
      return T_WC;
    }

    /**
     * Get the initial camera pose in map frame.
     *
     * \return The initial camera pose init_T_MC_ encoded in a 4x4 matrix.
     */
    Eigen::Matrix4f initT_MC() {
      return init_T_MC_;
    }

    /**
     * Get the inital camera pose in world frame.
     *
     * \return The initial camera pose T_MC encoded in a 4x4 matrix.
     */
    Eigen::Matrix4f initT_WC() {
      Eigen::Matrix4f init_T_WC = se::math::to_inverse_transformation(T_MW_) * init_T_MC_;
      return init_T_WC;
    }

    /**
     * Set the current camera pose provided in map frame.
     *
     * \param[in] T_MC The desired camera pose encoded in a 4x4 matrix.
     */
    void setT_MC(const Eigen::Matrix4f& T_MC) {
      T_MC_ = T_MC;
    }

    /**
     * Set the current camera pose provided in world frame.
     *
     * @note T_MW_ is added to the pose to process the information further in map frame.
     *
     * \param[in] T_WC The desired camera pose encoded in a 4x4 matrix.
     */
    void setT_WC(const Eigen::Matrix4f& T_WC) {
      T_MC_ = T_MW_ * T_WC;
    }

    /**
     * Set the initial camera pose provided in map frame.
     *
     * \param[in] init_T_MC The initial camera pose encoded in a 4x4 matrix.
     */
    void setInitT_MC(const Eigen::Matrix4f& init_T_MC) {
      init_T_MC_ = init_T_MC;
    }

    /**
     * Set the initial camera pose provided in world frame.
     *
     * @note T_MW_ is added to the pose to process the information further in map frame.
     *
     * \param[in] init_T_WC The initial camera pose encoded in a 4x4 matrix.
     */
    void setInitT_WC(const Eigen::Matrix4f& init_T_WC) {
      init_T_MC_ = T_MW_ * init_T_WC;
    }

    /**
     * Set the camera pose used to render the 3D reconstruction.
     *
     * \param[in] T_WC The desired camera pose encoded in a 4x4 matrix.
     */
    void setRenderT_MC(Eigen::Matrix4f* render_T_MC = nullptr) {
      if (render_T_MC == nullptr){
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
    Eigen::Matrix4f *renderT_MC_() {
      return render_T_MC_;
    }

    /**
     * Get the dimensions of the reconstructed volume in meters.
     *
     * \return A vector containing the x, y and z dimensions of the volume.
     */
    Eigen::Vector3f getMapDimension() {
      return (map_dim_);
    }

    /**
     * Get the size of the reconstructed volume in voxels.
     *
     * \return A vector containing the x, y and z resolution of the volume.
     */
    Eigen::Vector3i getMapSize() {
      return (map_size_);
    }

    /**
     * Get the resolution used when processing frames in the pipeline in
     * pixels.
     *
     * \return A vector containing the frame width and height.
     */
    Eigen::Vector2i getImageResolution() {
      return (image_res_);
    }
};

/**
 * Synchronize CPU and GPU.
 *
 * @note This function does nothing in the C++ implementation.
 */
void synchroniseDevices();

#endif
