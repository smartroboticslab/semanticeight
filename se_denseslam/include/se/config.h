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

#include <Eigen/Dense>
#include <ostream>
#include <string>
#include <vector>

#include "se/str_utils.hpp"
#include "se/utils/math_utils.h"



namespace se {
struct Configuration {
    //
    // Pipeline configuration parameters
    // Command line arguments are parsed in default_parameters.h
    //

    std::string sensor_type;
    std::string voxel_impl_type;
    std::string voxel_impl_yaml;
    std::string sequence_name;

    /**
     * The type of the sequence.
     * Valid types are (case insensitive):
     * - RAW (https://github.com/pamela-project/slambench)
     * - ICLNUIM (https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html)
     * - OpenNI (from file or live camera if sequence_path is empty)
     * - NewerCollege (https://ori-drs.github.io/newer-college-dataset/)
     *
     * <br>\em Default: RAW
     */
    std::string sequence_type;

    /**
     * The ratio of the input frame size over the frame size used internally.
     * Values greater than 1 result in the input frames being downsampled
     * before processing. Valid values are 1, 2, 4 and 8.
     *
     * <br>\em Default: 1
     */
    int sensor_downsampling_factor;

    /**
     * Perform tracking on a frame every tracking_rate frames.
     *
     * <br>\em Default: 1
     */
    int tracking_rate;

    /**
     * Integrate a 3D reconstruction every integration_rate frames. Should not
     * be less than tracking_rate.
     *
     * <br>\em Default: 1
     */
    int integration_rate;

    /**
     * Render the 3D reconstruction every rendering_rate frames
     * \note configuration::enable_render == true (default) required.
     *
     * Special cases:
     * If rendering_rate == 0 the volume is only rendered for configuration::max_frame.
     * If rendering_rate < 0  the volume is only rendered for frame abs(rendering_rate).
     *
     * <br>\em Default: 4
     */
    int rendering_rate;

    /**
     * Mesh the 3D reconstruction every meshing_rate frames.
     *
     * Special cases:
     * If meshing_rate == 0 the volume is only meshed for configuration::max_frame.
     * If meshing_rate < 0  the volume is only meshed for frame abs(meshing_rate).
     *
     * <br>\em Default: 100
     */
    int meshing_rate;

    /**
     * The x, y and z size of the reconstructed map in voxels.
     *
     * <br>\em Default: (256, 256, 256)
     */
    Eigen::Vector3i map_size;

    /**
     * The x, y and z dimensions of the reconstructed map in meters.
     *
     * <br>\em Default: (2, 2, 2)
     */
    Eigen::Vector3f map_dim;

    /**
     * The position of the first pose inside the volume. The coordinates are
     * expressed as fractions [0, 1] of the volume's extent. The default value
     * of (0.5, 0.5, 0) results in the first pose being placed halfway along
     * the x and y axes and at the beginning of the z axis.
     *
     * <br>\em Default: (0.5, 0.5, 0.5)
     */
    Eigen::Vector3f t_MW_factor;

    /**
     * The number of pyramid levels and ICP iterations for the depth images.
     * The number of elements in the vector is equal to the number of pramid
     * levels. The values of the vector elements correspond to the number of
     * ICP iterations run at this particular pyramid level. The first vector
     * element corresponds to the first level etc. The first pyramid level
     * contains the original depth image. Each subsequent pyramid level
     * contains the image of the previous level downsampled to half the
     * resolution. ICP starts from the highest (lowest resolution) pyramid
     * level. The default value of (10, 5, 4) results in 10 ICP iterations for
     * the initial depth frame, 5 iterations for the initial depth frame
     * downsampled once and 4 iterations for the initial frame downsampled
     * twice.
     *
     * <br>\em Default: (10, 5, 4)
     */
    std::vector<int> pyramid;

    /*
     * TODO
     * <br>\em Default: ""
     */
    std::string output_mesh_file;

    /*
     * TODO
     * <br>\em Default: ""
     */
    bool enable_structure;

    /*
     * TODO
     * <br>\em Default: ""
     */
    std::string output_structure_file;

    /*
     * TODO
     * <br>\em Default: ""
     */
    std::string sequence_path;

    /**
     * Whether to run the pipeline in benchmark mode. Hiding the GUI results in
     * faster operation.
     *
     * <br>\em Default: false
     */
    bool enable_benchmark;

    /**
     * The log file the timing results will be written to.
     *
     * <br>\em Default: std::cout if Configuration::enable_benchmark is blank
     * (--enable-benchmark) or not Configuration::enable_render
     * (--enable-render)
     * <br>\em Default: autogen log filename if the
     * Configuration::enable_benchmark argument is a directory
     * (--enable-benchmark=/PATH/TO/DIR)
     */
    std::string log_path;

    /**
     * The path to a text file containing the ground truth poses T_WC. Each
     * line of the file should correspond to a single pose. The pose should be
     * encoded in the format `tx ty tz qx qy qz qw` where `tx`, `ty` and `tz`
     * are the position coordinates and `qx`, `qy`, `qz` and `qw` the
     * orientation quaternion. Each line in the file should be in the format
     * `... tx ty tz qx qy qz qw`, that is the pose is encoded in the last 7
     * columns of the line. The other columns of the file are ignored. Lines
     * beginning with # are comments.
     *
     * <br>\em Default: ""
     *
     * \note It is assumed that the ground truth poses are the sensor frame C
     * expressed in the world frame W. The sensor frame is assumed to be z
     * forward, x right with respect to the image. If the ground truth poses do
     * not adhere to this assumption then the ground truth transformation
     * Configuration::T_BC should be set appropriately.
     */
    std::string ground_truth_file;

    /**
     * Whether to use the available ground truth camera pose.
     *
     * <br>\em Default: true
     */
    bool enable_ground_truth;


    /**
     * A 4x4 transformation matrix post-multiplied with all poses read from the
     * ground truth file. It is used if the ground truth poses are in some
     * frame B other than the sensor frame C.
     *
     * <br>\em Default: Eigen::Matrix4f::Identity()
     */
    Eigen::Matrix4f T_BC;

    /**
     * The initial pose of the body in world frame expressed in a 4x4
     * transformation matrix.
     *
     * \note If T_BC is the Idenity matrix init_T_WB equals init_T_WC
     *
     * <br>\em Default: Eigen::Matrix4f::Identity()
     */
    Eigen::Matrix4f init_T_WB;

    /**
     * The intrinsic sensor parameters. sensor_intrinsics.x,
     * sensor_intrinsics.y, sensor_intrinsics.z and sensor_intrinsics.w are the
     * x-axis focal length, y-axis focal length, horizontal resolution (pixels)
     * and vertical resolution (pixels) respectively.
     */
    Eigen::Vector4f sensor_intrinsics;

    /**
     * Indicates if the sensor uses a left hand coordinate system
     */
    bool left_hand_frame;

    /**
     * Nearest z-distance to the sensor along the sensor frame z-axis, that
     * voxels are updated.
     */
    float near_plane;

    /**
     * Furthest z-distance to the sensor along the sensor frame z-axis, that
     * voxels are updated.
     */
    float far_plane;

    /**
     * Read frames at the specified rate, waiting if the computation rate is
     * higher than se::Configuration::fps.
     *
     * \note Must be non-negative.
     *
     * <br>\em Default: 0
     */
    float fps;

    /**
     * Skip processing frames that could not be processed in time. Only has an
     * effect if se::Configuration::fps is greater than 0.
     * <br>\em Default: false
     */
    bool drop_frames;

    /**
     * Last frame to be integrated.
     * \note: se::Configuration::max_frame starts from 0.
     *
     * Special cases
     * If max_frame == -1 (default) the entire dataset will be integrated and the value will be overwritten by
     * number of frames in dataset - 1 (exception number of frames is unknwon e.g. OpenNI/live feed).
     *
     * If max_frame > number of frames in dataset - 1 the value will be overwritten by reader.numFrames()
     * (exception number of frames is unknwon e.g. OpenNI/live feed).
     *
     * If (max_frame == -1 (default) or max_frame > number of frames in dataset - 1) and the number of frames is unknown
     * (e.g. OpenNI/live feed) the frames are integrated until the pipeline is terminated and max_frame is kept at -1.
     *
     * max_frame in [0, number of frames in dataset - 1] or [0, inf] if number of frames in dataset is unknown.
     *
     * <br>\em Default: -1 (full dataset)
     */
    int max_frame;

    /**
     * The ICP convergence threshold.
     *
     * <br>\em Default: 1e-5
     */
    float icp_threshold;

    /**
     * Whether to mesh the octree.
     *
     * <br>\em Default: false
     */
    bool enable_meshing;

    /**
     * Whether to hide the GUI. Hiding the GUI results in faster operation.
     * <br>\em Default: true
     */
    bool enable_render;

    /*
     * TODO
     * <br>\em Default: ""
     */
    std::string output_render_file;

    /*
     * TODO
     * <br>\em Default: false
     */
    bool render_volume_fullsize;

    /**
     * Whether to filter the depth input frames using a bilateral filter.
     * Filtering using a bilateral filter helps to reduce the measurement
     * noise.
     *
     * <br>\em Default: false
     */
    bool bilateral_filter;

    // Exploration only ///////////////////////////////////////////////////////
    /** The minimum ratio of the frontier volume over the total volume of a Node/VoxelBlock for it
     * to be considered a frontier. Defaults to 0 for no minimum, i.e. all Nodes/VoxelBlocks with
     * some frontier volume will be considered frontiers.
     */
    int frontier_cluster_min_ratio;

    Eigen::Vector3f aabb_min_W;

    Eigen::Vector3f aabb_max_W;

    Eigen::Vector3f sampling_min_W;

    Eigen::Vector3f sampling_max_W;

    bool enable_exploration;

    int num_candidates;

    float exploration_weight;

    bool use_pose_history;

    int raycast_width;

    int raycast_height;

    float linear_velocity;

    float angular_velocity;

    float delta_t;

    /** The radius of the robot's bounding sphere in metres.
     * <br>\em Default: 0.3
     */
    float robot_radius;

    /** An extra safety radius in metres added to that of the robot's bounding sphere.
     * <br>\em Default: 0.0
     */
    float safety_radius;

    float min_control_point_radius;

    float skeleton_sample_precision;

    /** The maximum time in seconds that a path will be searched for by OMPL.
     * <br>\em Default: 0.1
     */
    float solving_time;

    Configuration() :
            sensor_type(""),
            voxel_impl_type(""),
            voxel_impl_yaml(""),
            sequence_name(""),
            sequence_type("raw"),
            sensor_downsampling_factor(1),
            tracking_rate(1),
            integration_rate(1),
            rendering_rate(4),
            meshing_rate(100),
            map_size(256, 256, 256),
            map_dim(2.0f, 2.0f, 2.0f),
            t_MW_factor(0.5f, 0.5f, 0.5f),
            pyramid({10, 5, 4}),
            output_mesh_file(""),
            enable_structure(false),
            output_structure_file(""),
            sequence_path(""),
            enable_benchmark(false),
            log_path(""),
            ground_truth_file(""),
            enable_ground_truth(true),
            T_BC(Eigen::Matrix4f::Identity()),
            init_T_WB(Eigen::Matrix4f::Identity()),
            sensor_intrinsics(0.0f, 0.0f, 0.0f, 0.0f),
            left_hand_frame(false),
            near_plane(0.4f),
            far_plane(4.0f),
            fps(0.0f),
            drop_frames(false),
            max_frame(-1),
            icp_threshold(1e-5),
            enable_meshing(false),
            enable_render(true),
            output_render_file(""),
            render_volume_fullsize(false),
            bilateral_filter(false),
            frontier_cluster_min_ratio(0.0f),
            aabb_min_W(0.0f, 0.0f, 0.0f),
            aabb_max_W(0.0f, 0.0f, 0.0f),
            // Use some large values because Infs/NaNs will make the limits difficult to convert to some
            // other frame.
            sampling_min_W(-10000, -10000, -10000),
            sampling_max_W(10000, 10000, 10000),
            enable_exploration(true),
            num_candidates(10),
            exploration_weight(0.5f),
            use_pose_history(true),
            raycast_width(36),
            raycast_height(10),
            linear_velocity(1.0f),
            angular_velocity(0.1f),
            delta_t(0.1f),
            robot_radius(0.3f),
            safety_radius(0.0f),
            min_control_point_radius(0.1f),
            skeleton_sample_precision(0.05f),
            solving_time(0.1f)
    {
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
} // namespace se



static std::ostream& operator<<(std::ostream& out, const se::Configuration& config)
{
    out << str_utils::header_to_pretty_str("GENERAL") << "\n";
    out << str_utils::str_to_pretty_str(config.voxel_impl_type, "Voxel impl type") << "\n";
    out << str_utils::str_to_pretty_str(config.sensor_type, "Sensor type") << "\n";
    out << "\n";

    out << str_utils::str_to_pretty_str(config.sequence_name, "Sequence name") << "\n";
    out << str_utils::str_to_pretty_str(config.sequence_type, "Sequence type") << "\n";
    out << str_utils::str_to_pretty_str(config.sequence_path, "Sequence path") << "\n";
    out << str_utils::str_to_pretty_str(config.ground_truth_file, "Ground truth file") << "\n";
    out << str_utils::str_to_pretty_str((config.log_path == "" ? "std::cout" : config.log_path),
                                        "Log file")
        << "\n";
    out << str_utils::bool_to_pretty_str(config.enable_benchmark, "Enable benchmark") << "\n";
    out << str_utils::bool_to_pretty_str(config.enable_ground_truth, "Enable ground truth") << "\n";
    out << str_utils::bool_to_pretty_str(config.enable_render, "Enable render") << "\n";
    if (config.output_render_file != "") {
        out << str_utils::str_to_pretty_str(config.output_render_file, "Output render file")
            << "\n";
    }
    out << str_utils::bool_to_pretty_str(config.enable_meshing, "Enable meshing") << "\n";
    if (config.output_mesh_file != "") {
        out << str_utils::str_to_pretty_str(config.output_mesh_file, "Output mesh file") << "\n";
    }
    out << str_utils::bool_to_pretty_str(config.enable_structure, "Enable structure") << "\n";
    if (config.output_structure_file != "") {
        out << str_utils::str_to_pretty_str(config.output_structure_file, "Output structure file")
            << "\n";
    }
    out << "\n";

    out << str_utils::value_to_pretty_str(config.integration_rate, "Integration rate") << "\n";
    out << str_utils::value_to_pretty_str(config.rendering_rate, "Rendering rate") << "\n";
    out << str_utils::value_to_pretty_str(config.meshing_rate, "Meshing rate") << "\n";
    out << str_utils::value_to_pretty_str(config.fps, "FPS") << "\n";
    out << str_utils::bool_to_pretty_str(config.drop_frames, "Drop frames") << "\n";
    out << str_utils::value_to_pretty_str(config.max_frame, "Max frame") << "\n";
    out << "\n";

    out << str_utils::vector_to_pretty_str(
        Eigen::VectorXi::Map(config.pyramid.data(), config.pyramid.size()), "ICP pyramid levels")
        << "\n";
    out << str_utils::value_to_pretty_str(config.icp_threshold, "ICP threshold") << "\n";
    out << str_utils::bool_to_pretty_str(config.render_volume_fullsize, "Render volume full-size")
        << "\n";
    out << "\n";

    out << str_utils::header_to_pretty_str("MAP") << "\n";
    out << str_utils::volume_to_pretty_str(config.map_size, "Map size", "voxel") << "\n";
    out << str_utils::volume_to_pretty_str(config.map_dim, "Map dim", "meter") << "\n";
    out << str_utils::value_to_pretty_str(
        config.map_dim.x() / config.map_size.x(), "Map res", "meter/voxel")
        << "\n";

    out << str_utils::vector_to_pretty_str(config.t_MW_factor, "t_MW_factor") << "\n";
    out << "\n";

    out << str_utils::header_to_pretty_str("SENSOR") << "\n";
    out << str_utils::vector_to_pretty_str(
        config.sensor_intrinsics, "Sensor intrinsics", {"fx", "fy", "cx", "cy"})
        << "\n";
    out << str_utils::bool_to_pretty_str(config.left_hand_frame, "Left-handed-coordinate system")
        << "\n";
    out << str_utils::value_to_pretty_str(config.sensor_downsampling_factor,
                                          "Sensor downsampling factor")
        << "\n";
    out << str_utils::bool_to_pretty_str(config.bilateral_filter, "Filter depth (bilateral filter)")
        << "\n";
    out << str_utils::value_to_pretty_str(config.near_plane, "Near plane", "meters") << "\n";
    out << str_utils::value_to_pretty_str(config.far_plane, "Far plane", "meters") << "\n";
    out << "\n";
    out << str_utils::matrix_to_pretty_str(config.T_BC, "T_BC") << "\n";
    out << "\n";
    out << str_utils::matrix_to_pretty_str(config.init_T_WB, "init_T_WB") << "\n";
    out << "\n";

    const Eigen::Vector3f t_MW = config.map_dim.x() * config.t_MW_factor;
    const Eigen::Matrix4f T_MW = se::math::to_transformation(t_MW);
    const Eigen::Matrix4f init_T_MB = T_MW * config.init_T_WB;
    const Eigen::Vector3f init_t_MB = se::math::to_translation(init_T_MB);
    out << str_utils::vector_to_pretty_str(init_t_MB, "init t_MB") << "\n";
    out << "\n";

    const Eigen::Vector3f init_t_MB_factor = init_t_MB / config.map_dim.x();
    out << str_utils::vector_to_pretty_str(init_t_MB_factor, "init t_MB_factor") << "\n";
    out << "\n";

    // Exploration only ///////////////////////////////////////////////////////
    out << str_utils::value_to_pretty_str(config.frontier_cluster_min_ratio,
                                          "Frontier cluster min volume")
        << "\n";
    out << str_utils::vector_to_pretty_str(config.aabb_min_W, "AABB min_w") << "\n";
    out << str_utils::vector_to_pretty_str(config.aabb_max_W, "AABB max_w") << "\n";
    out << str_utils::vector_to_pretty_str(config.sampling_min_W, "Sampling min_w") << "\n";
    out << str_utils::vector_to_pretty_str(config.sampling_max_W, "Sampling max_w") << "\n";
    out << str_utils::bool_to_pretty_str(config.enable_exploration, "Enable exploration") << "\n";
    out << str_utils::value_to_pretty_str(config.num_candidates, "Num candidates") << "\n";
    out << str_utils::value_to_pretty_str(config.exploration_weight, "Exploration weight") << "\n";
    out << str_utils::bool_to_pretty_str(config.use_pose_history, "Use pose history") << "\n";
    out << str_utils::value_to_pretty_str(config.raycast_width, "Raycast width") << "\n";
    out << str_utils::value_to_pretty_str(config.raycast_height, "Raycast height") << "\n";
    out << str_utils::value_to_pretty_str(config.linear_velocity, "Linear velocity") << "\n";
    out << str_utils::value_to_pretty_str(config.angular_velocity, "Angular velocity") << "\n";
    out << str_utils::value_to_pretty_str(config.delta_t, "Delta t") << "\n";
    out << str_utils::value_to_pretty_str(config.robot_radius, "Robot radius") << "\n";
    out << str_utils::value_to_pretty_str(config.safety_radius, "Safety radius") << "\n";
    out << str_utils::value_to_pretty_str(config.min_control_point_radius,
                                          "Min control point radius")
        << "\n";
    out << str_utils::value_to_pretty_str(config.skeleton_sample_precision,
                                          "Skeleton sample precision")
        << "\n";
    out << str_utils::value_to_pretty_str(config.solving_time, "Solving time") << "\n";

    return out;
}

#endif
