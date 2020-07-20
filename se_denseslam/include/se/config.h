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

#ifndef CONFIG_H
#define CONFIG_H

#include <ostream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "se/utils/math_utils.h"



struct Configuration {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  //
  // Pipeline configuration parameters
  // Command line arguments are parsed in default_parameters.h
  //

  std::string sensor_type;
  std::string voxel_impl_type;
  std::string sequence_name;

  /**
   * The ratio of the input frame size over the frame size used internally.
   * Values greater than 1 result in the input frames being downsampled
   * before processing. Valid values are 1, 2, 4 and 8.
   * <br>\em Default: 1
   */
  int image_downsampling_factor;

  /**
   * Perform tracking on a frame every tracking_rate frames.
   * <br>\em Default: 1
   */
  int tracking_rate;

  /**
   * Integrate a 3D reconstruction every integration_rate frames. Should not
   * be less than tracking_rate.
   * <br>\em Default: 2
   */
  int integration_rate;

  /**
   * Render the 3D reconstruction every rendering_rate frames.
   * <br>\em Default: 4
   */
  int rendering_rate;

  /**
   * The x, y and z size of the reconstructed map in voxels.
   * <br>\em Default: (256, 256, 256)
   */
  Eigen::Vector3i map_size;

  /**
   * The x, y and z dimensions of the reconstructed map in meters.
   * <br>\em Default: (2, 2, 2)
   */
  Eigen::Vector3f map_dim;

  /**
   * The position of the first pose inside the volume. The coordinates are
   * expressed as fractions [0, 1] of the volume's extent. The default value of
   * (0.5, 0.5, 0) results in the first pose being placed halfway along the x
   * and y axes and at the beginning of the z axis.
   * <br>\em Default: (0.5, 0.5, 0)
   */
  Eigen::Vector3f t_MW_factor;

  /**
   * The number of pyramid levels and ICP iterations for the depth images. The
   * number of elements in the vector is equal to the number of pramid levels.
   * The values of the vector elements correspond to the number of ICP
   * iterations run at this particular pyramid level. The first vector element
   * corresponds to the first level etc. The first pyramid level contains the
   * original depth image. Each subsequent pyramid level contains the image of
   * the previous level downsampled to half the resolution. ICP starts from the
   * highest (lowest resolution) pyramid level. The default value of (10, 5, 4)
   * results in 10 ICP iterations for the initial depth frame, 5 iterations for
   * the initial depth frame downsampled once and 4 iterations for the initial
   * frame downsampled twice.
   * <br>\em Default: (10, 5, 4)
   */
  std::vector<int> pyramid;

  /*
   * TODO
   * <br>\em Default: ""
   */
  std::string dump_volume_file;

  /*
   * TODO
   * <br>\em Default: ""
   */
  std::string input_file;

  /*
   * TODO
   * <br>\em Default: ""
   */
  std::string log_file;

  /**
   * The path to a text file containing the ground truth poses T_WC. Each line
   * of the file should correspond to a single pose. The pose should be encoded
   * in the format `tx ty tz qx qy qz qw` where `tx`, `ty` and `tz` are the
   * position coordinates and `qx`, `qy`, `qz` and `qw` the orientation
   * quaternion. Each line in the file should be in the format `... tx ty tz qx
   * qy qz qw`, that is the pose is encoded in the last 7 columns of the line.
   * The other columns of the file are ignored. Lines beginning with # are
   * comments.
   * <br>\em Default: ""
   *
   * \note It is assumed that the ground truth poses are the camera frame C
   * expressed in the world frame W. The camera frame is assumed to be z
   * forward, x right with respect to the image. If the ground truth poses do
   * not adhere to this assumption then the ground truth transformation
   * Configuration::T_BC should be set appropriately.
   */
  std::string groundtruth_file;

  /**
   * A 4x4 transformation matrix post-multiplied with all poses read from the
   * ground truth file. It is used if the ground truth poses are in some frame
   * B other than the camera frame C.
   * <br>\em Default: Eigen::Matrix4f::Identity()
   */
  Eigen::Matrix4f T_BC;

  /**
   * The intrinsic camera parameters. camera.x, camera.y, camera.z and
   * camera.w are the x-axis focal length, y-axis focal length, horizontal
   * resolution (pixels) and vertical resolution (pixels) respectively.
   */
  Eigen::Vector4f camera;

  /**
   * Indicates if the camera uses a left hand coordinate system
   */
  bool left_hand_frame;

  /**
   * Whether the default intrinsic camera parameters have been overriden.
   */
  bool camera_overrided;

  /**
   * The TSDF truncation bound. Values of the TSDF are assumed to be in the
   * interval ±mu. See Section 3.3 of \cite NewcombeISMAR2011 for more
   * details.
   *  <br>\em Default: 0.1
   */
  float mu;

  /**
   * Nearest z-distance to the sensor along the sensor frame z-axis, that voxels are updated.
   */
  float near_plane;

  /**
   * Furthest z-distance to the sensor along the sensor frame z-axis, that voxels are updated.
   */
  float far_plane;

  /**
   * Read frames at the specified rate, waiting if the computation rate is
   * higher than se::Configuration::fps.
   *
   * @note Must be non-negative.
   *
   * <br>\em Default: 0
   */
  float fps;

  /*
   * TODO
   * <br>\em Default: false
   */
  bool blocking_read;

  /**
   * The ICP convergence threshold.
   * <br>\em Default: 1e-5
   */
  float icp_threshold;

  /**
   * Whether to hide the GUI. Hiding the GUI results in faster operation.
   * <br>\em Default: false
   */
  bool no_gui;

  /*
   * TODO
   * <br>\em Default: false
   */
  bool render_volume_fullsize;

  /**
   * Whether to filter the depth input frames using a bilateral filter.
   * Filtering using a bilateral filter helps to reduce the measurement
   * noise.
   * <br>\em Default: false
   */
  bool bilateral_filter;
};



static std::ostream& operator<<(std::ostream& out, const Configuration& config) {
  out << "Sequence name:                   " << config.sequence_name << "\n";
  out << "Sensor type:                     " << config.sensor_type << "\n";
  out << "Voxel impl type:                 " << config.voxel_impl_type << "\n";
  out << "Input file:                      " << config.input_file << "\n";
  out << "Map dim:                         " << config.map_dim.x() << "x"
                                             << config.map_dim.y() << "x"
                                             << config.map_dim.z() << " meters\n";
  out << "Map size:                        " << config.map_size.x() << "x"
                                             << config.map_size.y() << "x"
                                             << config.map_size.z() << " voxels\n";
  out << "World to map translation factor: " << config.t_MW_factor.x() << " "
                                             << config.t_MW_factor.y() << " "
                                             << config.t_MW_factor.z() << "\n";
  out << "Image downsampling-factor:       " << config.image_downsampling_factor << "\n";
  out << "Camera parameters:               " << config.camera.x() << " "
                                             << config.camera.y() << " "
                                             << config.camera.z() << " "
                                             << config.camera.w() << "\n";
  out << "Left hand frame:                 " << config.left_hand_frame << "\n";
  out << "Mu:                              " << config.mu << "\n";
  out << "Near plane:                      " << config.near_plane << "\n";
  out << "Far plane:                       " << config.far_plane << "\n";
  out << "Filter depth:                    " << (config.bilateral_filter
                                        ? "true" : "false") << "\n";
  out << "Tracking rate:                   " << config.tracking_rate << "\n";
  out << "Integration rate:                " << config.integration_rate << "\n";
  out << "Rendering rate:                  " << config.rendering_rate << "\n";
  out << "ICP pyramid levels:              ";
  for (const auto& level : config.pyramid) {
    out << " " << level;
  }
  out << "\n";
  out << "ICP threshold:                   " << config.icp_threshold << "\n";
  out << "Ground truth file:               " << config.groundtruth_file << "\n";
  out << "Ground truth T_BC:\n"              << config.T_BC << "\n";
  out << "Output mesh file:                " << config.dump_volume_file << "\n";
  out << "Log file:                        " << config.log_file << "\n";
  out << "Hide GUI:                        " << (config.no_gui
                                        ? "true" : "false") << "\n";
  out << "Blocking read:                   " << (config.blocking_read
                                        ? "true" : "false") << "\n";
  out << "FPS:                             " << config.fps << "\n";
  return out;
}

#endif
