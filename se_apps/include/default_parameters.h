/*

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */

#ifndef __DEFAULT_PARAMETERS_H
#define __DEFAULT_PARAMETERS_H

#include <cstdlib>
#include <getopt.h>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include "filesystem.hpp"

#include "se/config.h"
#include "se/constant_parameters.h"
#include "se/str_utils.hpp"
#include "se/utils/math_utils.h"

// Default option values.
static constexpr bool         default_benchmark = false;
static const std::string      default_benchmark_path = "";
static constexpr bool         default_bilateral_filter = false;
static constexpr bool         default_drop_frames = false;
static constexpr float        default_far_plane = 4.0f;
static constexpr float        default_fps = 0.0f;
static const std::string      default_ground_truth_file = "";
static const Eigen::Matrix4f  default_gt_transform = Eigen::Matrix4f::Identity();
static constexpr float        default_icp_threshold = 1e-5;
static constexpr int          default_integration_rate = 2;
static constexpr int          default_iteration_count = 3;
static constexpr int          default_iterations[default_iteration_count] = { 10, 5, 4 };
static constexpr bool         default_left_hand_frame = false;
static const std::string      default_log_file = "";
static constexpr float        default_near_plane = 0.4f;
static const Eigen::Vector3i  default_map_size(256, 256, 256);
static const Eigen::Vector3f  default_map_dim(2.f, 2.f, 2.f);
static constexpr bool         default_no_gui = false;
static constexpr bool         default_render_volume_fullsize = false;
static constexpr int          default_rendering_rate = 4;
static const std::string      default_output_mesh_file = "";
static const std::string      default_sequence_name = "";
static constexpr int          default_sensor_downsampling_factor = 1;
static const Eigen::Vector4f  default_sensor_intrinsics = Eigen::Vector4f::Zero();
static const std::string      default_sequence_path = "";
static const Eigen::Vector3f  default_t_MW_factor(0.5f, 0.5f, 0.0f);
static constexpr int          default_tracking_rate = 1;



// Put colons after options with arguments
static std::string short_options = "b:B:c:d:f:F:g:G:h:i:k:l:n:N:o:p:q:r:s:S:t:v:y:Y:z:?";

static struct option long_options[] = {
  {"drop-frames",                no_argument,       0, 'b'},
  {"benchmark",                  optional_argument, 0, 'B'},
  {"sensor-downsampling-factor", required_argument, 0, 'c'},
  {"output-mesh-file",           required_argument, 0, 'd'},
  {"fps",                        required_argument, 0, 'f'},
  {"bilateral-filter",           no_argument,       0, 'F'},
  {"ground-truth",               required_argument, 0, 'g'},
  {"gt-transform",               required_argument, 0, 'G'},
  {"help",                       no_argument,       0, 'h'},
  {"sequence-path",              required_argument, 0, 'i'},
  {"sensor-intrinsics",          required_argument, 0, 'k'},
  {"icp-threshold",              required_argument, 0, 'l'},
  {"near-plane",                 required_argument, 0, 'n'},
  {"far-plane",                  required_argument, 0, 'N'},
  {"log-file",                   required_argument, 0, 'o'},
  {"init-pose",                  required_argument, 0, 'p'},
  {"no-gui",                     no_argument,       0, 'q'},
  {"integration-rate",           required_argument, 0, 'r'},
  {"map-dim",                    required_argument, 0, 's'},
  {"sequence-name",              required_argument, 0, 'S'},
  {"tracking-rate",              required_argument, 0, 't'},
  {"map-size",                   required_argument, 0, 'v'},
  {"pyramid-levels",             required_argument, 0, 'y'},
  {"yaml-file",                  required_argument, 0, 'Y'},
  {"rendering-rate",             required_argument, 0, 'z'},
  {"",                           no_argument,       0, '?'},
  {0, 0, 0, 0}
};



inline void print_arguments() {
  std::cerr << "-b  (--drop-frames)                       : default is false: don't drop frames\n";
  std::cerr << "-B  (--benchmark)                         : default is autogen benchmark filename\n";
  std::cerr << "-c  (--sensor-downsampling-factor)        : default is " << default_sensor_downsampling_factor << " (same size)\n";
  std::cerr << "-d  (--output-mesh-file) <filename>       : output mesh file\n";
  std::cerr << "-f  (--fps)                               : default is " << default_fps << "\n";
  std::cerr << "-F  (--bilateral-filter                   : default is disabled\n";
  std::cerr << "-g  (--ground-truth) <filename>           : ground truth file\n";
  std::cerr << "-G  (--gt-transform) tx,ty,tz,qx,qy,qz,qw : T_BC (translation and/or rotation)\n";
  std::cerr << "-h  (--help)                              : show this help message\n";
  std::cerr << "-i  (--sequence-path) <filename>          : sequence path\n";
  std::cerr << "-k  (--sensor-intrinsics)                 : default is defined by input\n";
  std::cerr << "-l  (--icp-threshold)                     : default is " << default_icp_threshold << "\n";
  std::cerr << "-o  (--log-file) <filename>               : default is stdout\n";
  std::cerr << "-n  (--near-plane)                        : default is " << default_near_plane << "\n";
  std::cerr << "-N  (--far-plane)                         : default is " << default_far_plane << "\n";
  std::cerr << "-p  (--init-pose)                         : default is " << default_t_MW_factor.x() << "," << default_t_MW_factor.y() << "," << default_t_MW_factor.z() << "\n";
  std::cerr << "-q  (--no-gui)                            : default is to display gui\n";
  std::cerr << "-r  (--integration-rate)                  : default is " << default_integration_rate << "\n";
  std::cerr << "-s  (--map-dim)                           : default is " << default_map_dim.x() << "," << default_map_dim.y() << "," << default_map_dim.z() << "\n";
  std::cerr << "-S  (--sequence-name)                     : name of sequence\n";
  std::cerr << "-t  (--tracking-rate)                     : default is " << default_tracking_rate << "\n";
  std::cerr << "-v  (--map-size)                          : default is " << default_map_size.x() << "," << default_map_size.y() << "," << default_map_size.z() << "\n";
  std::cerr << "-y  (--pyramid-levels)                    : default is 10,5,4\n";
  std::cerr << "-Y  (--yaml-file)                         : YAML file\n";
  std::cerr << "-z  (--rendering-rate)                    : default is " << default_rendering_rate << "\n";
}



inline Eigen::Vector3f atof3(char* arg) {
  Eigen::Vector3f res = Eigen::Vector3f::Zero();
  std::istringstream remaining_arg(arg);
  std::string s;
  if (std::getline(remaining_arg, s, ',')) {
    res.x() = atof(s.c_str());
  } else {
    // arg is empty
    return res;
  }
  if (std::getline(remaining_arg, s, ',')) {
    res.y() = atof(s.c_str());
  } else {
    // arg is x
    res.y() = res.x();
    res.z() = res.x();
    return res;
  }
  if (std::getline(remaining_arg, s, ',')) {
    res.z() = atof(s.c_str());
  } else {
    // arg is x,y
    res.z() = res.y();
  }
  return res;
}



inline Eigen::Vector3i atoi3(char* arg) {
  Eigen::Vector3i res = Eigen::Vector3i::Zero();
  std::istringstream remaining_arg(arg);
  std::string s;
  if (std::getline(remaining_arg, s, ',')) {
    res.x() = atoi(s.c_str());
  } else {
    // arg is empty
    return res;
  }
  if (std::getline(remaining_arg, s, ',')) {
    res.y() = atoi(s.c_str());
  } else {
    // arg is x
    res.y() = res.x();
    res.z() = res.x();
    return res;
  }
  if (std::getline(remaining_arg, s, ',')) {
    res.z() = atoi(s.c_str());
  } else {
    // arg is x,y
    res.z() = res.y();
  }
  return res;
}



inline Eigen::Vector4f atof4(char* arg) {
  Eigen::Vector4f res = Eigen::Vector4f::Zero();
  std::istringstream remaining_arg(arg);
  std::string s;
  if (std::getline(remaining_arg, s, ',')) {
    res.x() = atof(s.c_str());
  } else {
    // arg is empty
    return res;
  }
  if (std::getline(remaining_arg, s, ',')) {
    res.y() = atof(s.c_str());
  } else {
    // arg is x
    res.y() = res.x();
    res.z() = res.x();
    res.w() = res.x();
    return res;
  }
  if (std::getline(remaining_arg, s, ',')) {
    res.z() = atof(s.c_str());
  } else {
    // arg is x,y
    res.z() = res.y();
    res.w() = res.y();
    return res;
  }
  if (std::getline(remaining_arg, s, ',')) {
    res.w() = atof(s.c_str());
  } else {
    // arg is x,y,z
    res.w() = res.z();
  }
  return res;
}

// Transformation std::vector(16) to transformation Eigen::Matrix4f
Eigen::Matrix4f TvtoT(std::vector<float> T_v) {
  Eigen::Matrix4f T;
  T << T_v[0],  T_v[1],  T_v[2],  T_v[3],
       T_v[4],  T_v[5],  T_v[6],  T_v[7],
       T_v[8],  T_v[9],  T_v[10], T_v[11],
       T_v[12], T_v[13], T_v[14], T_v[15];
  return T;
}

std::string adjustString(std::string s) {
  std::replace(s.begin(), s.end(), '.', '_');
  std::replace(s.begin(), s.end(), '-', '_');
  std::replace(s.begin(), s.end(), ' ', '_');
  std::transform(s.begin(), s.end(),s.begin(), ::tolower);
  return s;
}

void generateBenchmarkFile(Configuration& config) {
  stdfs::path benchmark_path = config.benchmark_file;
  if (config.benchmark_file != "" && !stdfs::is_directory(benchmark_path)) {
    return;
  } else {
    if (config.sequence_name == "") {
      std::cout << "Please provide a sequence name to autogen benchmark file name.\n"
                   "Options: \n"
                   "  - Provide sequence name via terminal          (Type \"sequence_name\" + hit enter)\n"
                   "  - Leave blank                                 (Hit enter)\n"
                   "  - Provide full benchmark filename             (e.g. --benchmark=\"PATH/TO/result.txt\")\n"
                   "  - Set sequence name via command line argument (-S \"sequence_name\")\n"
                   "  - Set sequence name via YAML file             (sequence_name: \"sequence_name\") \n\n"
                   "Provide sequence name (e.g. icl-nuim-livingroom_traj_02):" << std::endl;
      std::getline(std::cin, config.sequence_name);
    }
    std::stringstream auto_benchmark_filename_ss;
    auto_benchmark_filename_ss                      << config.voxel_impl_type     <<
                                    "_"             << config.sensor_type         <<
    ((config.sequence_name != "") ? "_"              + config.sequence_name : "") <<
                                    "_size_"        << config.map_size.x()        <<
                                    "_dim_"         << config.map_dim.x()         <<
                                    "_down_sample_" << config.sensor_downsampling_factor;
    benchmark_path /= adjustString(auto_benchmark_filename_ss.str()) + ".txt";
    config.benchmark_file = benchmark_path;
  }
}

Configuration parseArgs(unsigned int argc, char** argv) {
  Configuration config;

  YAML::Node yaml_general_config = YAML::Load("");
  bool has_yaml_general_config = false;
  YAML::Node yaml_map_config = YAML::Load("");
  bool has_yaml_map_config = false;
  YAML::Node yaml_sensor_config = YAML::Load("");
  bool has_yaml_sensor_config = false;
  YAML::Node yaml_voxel_impl_config = YAML::Load("");
  bool has_yaml_voxel_impl_config = false;

  int c;
  int option_index = 0;
  while ((c = getopt_long(argc, argv, short_options.c_str(), long_options,
                          &option_index)) != -1) {
    if (c == 'Y')  {
      if (YAML::LoadFile(optarg)["general"]) {
        yaml_general_config = YAML::LoadFile(optarg)["general"];
        has_yaml_general_config = true;
      }
      if (YAML::LoadFile(optarg)["map"]) {
        yaml_map_config = YAML::LoadFile(optarg)["map"];
        has_yaml_map_config = true;
      }
      if (YAML::LoadFile(optarg)["sensor"]) {
        yaml_sensor_config = YAML::LoadFile(optarg)["sensor"];
        has_yaml_sensor_config = true;
      }
      if (YAML::LoadFile(optarg)["voxel_impl"]) {
        yaml_voxel_impl_config = YAML::LoadFile(optarg)["voxel_impl"];
        has_yaml_voxel_impl_config = true;
      }
    }
  }

  // CONFIGURE GENERAL
  // No GUI
  config.no_gui = (has_yaml_general_config && yaml_general_config["no_gui"])
      ? yaml_general_config["no_gui"].as<bool>() : default_no_gui;

  // Sequence name
  config.sequence_name = (has_yaml_general_config && yaml_general_config["sequence_name"])
      ? yaml_general_config["sequence_name"].as<std::string>() : default_sequence_name;

  // Sequence path file or directory path
  config.sequence_path = (has_yaml_general_config && yaml_general_config["sequence_path"])
      ? yaml_general_config["sequence_path"].as<std::string>() : default_sequence_path;

  // Ground truth file path
  config.ground_truth_file = (has_yaml_general_config && yaml_general_config["ground_truth_file"])
      ? yaml_general_config["ground_truth_file"].as<std::string>() : default_ground_truth_file;

  // Benchmark and result file or directory path
  config.benchmark = (has_yaml_general_config && yaml_general_config["benchmark"])
      ? yaml_general_config["benchmark"].as<bool>() : default_benchmark;
  config.benchmark_file = (has_yaml_general_config && yaml_general_config["benchmark_path"])
      ? yaml_general_config["benchmark_path"].as<std::string>() : default_benchmark_path;
  // Log file path
  config.log_file = (has_yaml_general_config && yaml_general_config["log_file"])
      ? yaml_general_config["log_file"].as<std::string>() : default_output_mesh_file;
  // Output mesh file path
  config.output_mesh_file = (has_yaml_general_config && yaml_general_config["output_mesh_file"])
                            ? yaml_general_config["output_mesh_file"].as<std::string>() : default_output_mesh_file;

  // Integration rate
  config.integration_rate = (has_yaml_general_config && yaml_general_config["integration_rate"])
      ? yaml_general_config["integration_rate"].as<int>() : default_integration_rate;
  // Tracking rate
  config.tracking_rate = (has_yaml_general_config && yaml_general_config["tracking_rate"])
      ? yaml_general_config["tracking_rate"].as<int>() : default_tracking_rate;
  // Rendering rate
  config.rendering_rate = (has_yaml_general_config && yaml_general_config["rendering_rate"])
      ? yaml_general_config["rendering_rate"].as<int>() : default_rendering_rate;
  // Frames per second
  config.fps = (has_yaml_general_config && yaml_general_config["fps"])
      ? yaml_general_config["fps"].as<float>() : default_fps;

  // Drop frames
  config.drop_frames = (has_yaml_general_config && yaml_general_config["drop_frames"])
      ? yaml_general_config["drop_frames"].as<bool>() : default_drop_frames;

  // ICP threshold
  config.icp_threshold = (has_yaml_general_config && yaml_general_config["icp_threshold"])
      ? yaml_general_config["icp_threshold"].as<float>() : default_icp_threshold;
  // Render volume fullsize
  config.render_volume_fullsize = (has_yaml_general_config && yaml_general_config["render_volume_fullsize"])
      ? yaml_general_config["render_volume_fullsize"].as<bool>() : default_render_volume_fullsize;
  // Bilateral filter
  config.bilateral_filter = (has_yaml_general_config && yaml_general_config["bilateral_filter"])
      ? yaml_general_config["bilateral_filter"].as<bool>() : default_bilateral_filter;

  config.pyramid.clear();
  if (has_yaml_general_config && yaml_general_config["pyramid"]) {
    config.pyramid = yaml_general_config["pyramid"].as<std::vector<int>>();
  } else {
    for (int i = 0; i < default_iteration_count; i++) {
      config.pyramid.push_back(default_iterations[i]);
    }
  }


  // CONFIGURE MAP
  // Map size
  config.map_size = (has_yaml_map_config && yaml_map_config["size"])
      ? Eigen::Vector3i::Constant(yaml_map_config["size"].as<int>()) : default_map_size;
  // Map dimension
  config.map_dim = (has_yaml_map_config && yaml_map_config["dim"])
      ? Eigen::Vector3f::Constant(yaml_map_config["dim"].as<float>()) : default_map_dim;
  // World to Map frame translation
  config.t_MW_factor = (has_yaml_map_config && yaml_map_config["t_MW_factor"])
      ? Eigen::Vector3f(yaml_map_config["t_MW_factor"].as<std::vector<float>>().data()) : default_t_MW_factor;


  // CONFIGURE SENSOR
  // Sensor type
  config.sensor_type = SensorImpl::type();
  // Sensor intrinsics
  if (has_yaml_sensor_config && yaml_sensor_config["intrinsics"]) {
    config.sensor_intrinsics = Eigen::Vector4f((yaml_sensor_config["intrinsics"].as<std::vector<float>>()).data());
  } else {
    config.sensor_intrinsics = default_sensor_intrinsics;
  }
  // Sensor overrided
  config.sensor_intrinsics_overrided = false;
  // Sensor downsamling factor
  config.sensor_downsampling_factor = (has_yaml_sensor_config && yaml_sensor_config["downsampling_factor"])
      ? yaml_sensor_config["downsampling_factor"].as<int>() : default_sensor_downsampling_factor;
  // Left hand coordinate frame
  config.left_hand_frame = (has_yaml_sensor_config && yaml_sensor_config["left_hand_frame"])
      ? yaml_sensor_config["left_hand_frame"].as<bool>() : default_left_hand_frame;
  // Camera to Body frame transformation
  config.T_BC = (has_yaml_sensor_config && yaml_sensor_config["T_BC"])
      ? Eigen::Matrix4f(TvtoT(yaml_sensor_config["T_BC"].as<std::vector<float>>())) : default_gt_transform;
  // Near plane
  config.near_plane = (has_yaml_sensor_config && yaml_sensor_config["near_plane"])
      ? yaml_sensor_config["near_plane"].as<float>() : default_near_plane;
  // Far plane
  config.far_plane = (has_yaml_sensor_config && yaml_sensor_config["far_plane"])
      ? yaml_sensor_config["far_plane"].as<float>() : default_far_plane;


  // CONFIGURE VOXEL IMPL
  config.voxel_impl_type = VoxelImpl::type();
  (has_yaml_voxel_impl_config) ? VoxelImpl::configure(yaml_voxel_impl_config) : VoxelImpl::configure();

  // Reset getopt_long state to start parsing from the beginning
  optind = 1;
  option_index = 0;
  std::vector<std::string> tokens;
  Eigen::Vector3f gt_transform_tran;
  Eigen::Quaternionf gt_transform_quat;
  while ((c = getopt_long(argc, argv, short_options.c_str(), long_options,
          &option_index)) != -1) {
    switch (c) {
      case 'b': // drop-frames
        config.drop_frames = true;
        break;

      case 'B': // benchmark
        config.benchmark = true;
        if (optarg) {
          config.benchmark_file = optarg;
        }
        break;

      case 'c': // sensor-downsampling-factor
        config.sensor_downsampling_factor = atoi(optarg);
        if (   (config.sensor_downsampling_factor != 1)
            && (config.sensor_downsampling_factor != 2)
            && (config.sensor_downsampling_factor != 4)
            && (config.sensor_downsampling_factor != 8)) {
          std::cerr << "Error: --sensor-downsampling-factor (-c) must be 1, 2 ,4 "
              << "or 8  (was " << optarg << ")\n";
          exit(EXIT_FAILURE);
        }
        break;

      case 'd': // output-mesh-file
        config.output_mesh_file = optarg;
        break;

      case 'f': // fps
        config.fps = atof(optarg);
        if (config.fps < 0) {
          std::cerr << "Error: --fps (-f) must be >= 0 (was " << optarg << ")\n";
          exit(EXIT_FAILURE);
        }
        break;

      case 'F': // bilateral-filter
        config.bilateral_filter = true;
        break;

      case 'g': // ground-truth
        config.ground_truth_file = optarg;
        break;

      case 'G': // gt-transform
        // Split argument into substrings
        tokens = split_string(optarg, ',');
        switch (tokens.size()) {
          case 3:
            // Translation
            gt_transform_tran = Eigen::Vector3f(std::stof(tokens[0]),
                std::stof(tokens[1]), std::stof(tokens[2]));
            config.T_BC.topRightCorner<3,1>() = gt_transform_tran;
            break;
          case 4:
            // Rotation
            // Create a quaternion and get the equivalent rotation matrix
            gt_transform_quat = Eigen::Quaternionf(std::stof(tokens[3]),
                std::stof(tokens[0]), std::stof(tokens[1]), std::stof(tokens[2]));
            config.T_BC.block<3,3>(0,0) = gt_transform_quat.toRotationMatrix();
            break;
          case 7:
            // Translation and rotation
            gt_transform_tran = Eigen::Vector3f(std::stof(tokens[0]),
                std::stof(tokens[1]), std::stof(tokens[2]));
            gt_transform_quat = Eigen::Quaternionf(std::stof(tokens[6]),
                std::stof(tokens[3]), std::stof(tokens[4]), std::stof(tokens[5]));
            config.T_BC.topRightCorner<3,1>() = gt_transform_tran;
            config.T_BC.block<3,3>(0,0) = gt_transform_quat.toRotationMatrix();
            break;
          default:
            std::cerr << "Error: Invalid number of parameters for argument gt-transform. Valid parameters are:\n"
                << "3 parameters (translation): tx,ty,tz\n"
                << "4 parameters (rotation in quaternion form): qx,qy,qz,qw\n"
                << "7 parameters (translation and rotation): tx,ty,tz,qx,qy,qz,qw"
                << std::endl;
            exit(EXIT_FAILURE);
        }
        break;

      case '?':
      case 'h': // help
        print_arguments();
        exit(EXIT_SUCCESS);

      case 'i': // sequence-path
        config.sequence_path = optarg;
        struct stat st;
        if (stat(config.sequence_path.c_str(), &st) != 0) {
          std::cerr << "Error: --sequence-path (-i) does not exist (was "
              << config.sequence_path << ")\n";
          exit(EXIT_FAILURE);
        }
        break;

      case 'k': // sensor-intrinsics
        config.sensor_intrinsics = atof4(optarg);
        config.sensor_intrinsics_overrided = true;
        if (config.sensor_intrinsics.y() < 0) {
          config.left_hand_frame = true;
          std::cerr << "update to left hand coordinate system" << std::endl;
        }
        break;

      case 'l': // icp-threshold
        config.icp_threshold = atof(optarg);
        break;

      case 'n': // near-plane
        config.near_plane = atof(optarg);
        break;

      case 'N': // far-plane
        config.far_plane = atof(optarg);
        break;

      case 'o': // log-file
        config.log_file = optarg;
        break;

      case 'p': // init-pose
        config.t_MW_factor = atof3(optarg);
        break;

      case 'q': // no-qui
        config.no_gui = true;
        break;

      case 'r': // integration-rate
        config.integration_rate = atoi(optarg);
        if (config.integration_rate < 1) {
          std::cerr << "Error: --integration-rate (-r) must >= 1 (was "
              << optarg << ")\n";
          exit(EXIT_FAILURE);
        }
        break;

      case 's': // map-size
        config.map_dim = atof3(optarg);
        if (   (config.map_dim.x() <= 0)
            || (config.map_dim.y() <= 0)
            || (config.map_dim.z() <= 0)) {
          std::cerr << "Error: --map-dim (-s) all dimensions must > 0 (was "
              << optarg << ")\n";
          exit(EXIT_FAILURE);
        }
        break;

      case 'S': // sequence-name
        {
          config.sequence_name = optarg;
        }
        break;

      case 't': // tracking-rate
        config.tracking_rate = atof(optarg);
        break;

      case 'z': // rendering-rate
        config.rendering_rate = atof(optarg);
        break;

      case 'v': // map-size
        config.map_size = atoi3(optarg);
        if (   (config.map_size.x() <= 0)
            || (config.map_size.y() <= 0)
            || (config.map_size.z() <= 0)) {
          std::cerr << "Error: --map-size (-s) all dimensions must > 0 (was "
              << optarg << ")\n";
          exit(EXIT_FAILURE);
        }

        break;

      case 'y': // pyramid-levels
        {
          std::istringstream remaining_arg(optarg);
          std::string s;
          config.pyramid.clear();
          while (std::getline(remaining_arg, s, ',')) {
            config.pyramid.push_back(atof(s.c_str()));
          }
        }
        break;

      case 'Y': // yaml-file
        break;

      default:
        print_arguments();
        exit(EXIT_FAILURE);
    }
  }

  // Ensure the parameter values are valid.
  if (config.near_plane >= config.far_plane) {
    std::cerr << "Error: Near plane must be smaller than far plane ("
              << config.near_plane << " >= " << config.far_plane << ")\n";
    exit(EXIT_FAILURE);
  }

  if (config.benchmark) {
    config.no_gui = false; // Turn log_file off
    generateBenchmarkFile(config);
  }

  return config;
}

#endif

