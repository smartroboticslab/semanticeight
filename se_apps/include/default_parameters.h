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

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

#include "se/config.h"
#include "se/constant_parameters.h"
#include "se/str_utils.hpp"
#include "se/utils/math_utils.h"

// Default option values.
static constexpr int default_iteration_count = 3;
static constexpr int default_iterations[default_iteration_count] = { 10, 5, 4 };
static constexpr float default_mu = 0.1f;
static constexpr float default_near_plane = 0.4f;
static constexpr float default_far_plane = 4.0f;
static constexpr bool default_blocking_read = false;
static constexpr float default_fps = 0.0f;
static constexpr bool default_left_hand_frame = false;
static constexpr float default_icp_threshold = 1e-5;
static constexpr int default_image_downsampling_factor = 1;
static constexpr int default_integration_rate = 2;
static constexpr int default_rendering_rate = 4;
static constexpr int default_tracking_rate = 1;
static const Eigen::Vector3i default_map_size(256, 256, 256);
static const Eigen::Vector3f default_map_dim(2.f, 2.f, 2.f);
static const Eigen::Vector3f default_t_MW_factor(0.5f, 0.5f, 0.0f);
static constexpr bool default_no_gui = false;
static constexpr bool default_render_volume_fullsize = false;
static constexpr bool default_bilateral_filter = false;
static const std::string default_dump_volume_file = "";
static const std::string default_sequence_name = "";
static const std::string default_input_file = "";
static const std::string default_log_file = "";
static const std::string default_groundtruth_file = "";
static const Eigen::Matrix4f default_gt_transform = Eigen::Matrix4f::Identity();
static const Eigen::Vector4f default_camera = Eigen::Vector4f::Zero();



// Put colons after options with arguments
static std::string short_options = "bc:d:f:Fg:G:hi:k:l:m:n:N:Y:o:p:qr:s:S:t:v:y:z:?";

static struct option long_options[] = {
  {"block-read",                no_argument,       0, 'b'},
  {"image-downsampling-factor", required_argument, 0, 'c'},
  {"dump-volume",               required_argument, 0, 'd'},
  {"fps",                       required_argument, 0, 'f'},
  {"bilateral-filter",          no_argument,       0, 'F'},
  {"ground-truth",              required_argument, 0, 'g'},
  {"gt-transform",              required_argument, 0, 'G'},
  {"help",                      no_argument,       0, 'h'},
  {"sequence-name",             required_argument, 0, 'S'},
  {"input-file",                required_argument, 0, 'i'},
  {"camera",                    required_argument, 0, 'k'},
  {"icp-threshold",             required_argument, 0, 'l'},
  {"mu",                        required_argument, 0, 'm'},
  {"near-plane",                required_argument, 0, 'n'},
  {"far-plane",                 required_argument, 0, 'N'},
  {"yaml-file",                 required_argument, 0, 'Y'},
  {"log-file",                  required_argument, 0, 'o'},
  {"init-pose",                 required_argument, 0, 'p'},
  {"no-gui",                    no_argument,       0, 'q'},
  {"integration-rate",          required_argument, 0, 'r'},
  {"map-dim",                   required_argument, 0, 's'},
  {"tracking-rate",             required_argument, 0, 't'},
  {"map-size",                  required_argument, 0, 'v'},
  {"pyramid-levels",            required_argument, 0, 'y'},
  {"rendering-rate",            required_argument, 0, 'z'},
  {"",                          no_argument,       0, '?'},
  {0, 0, 0, 0}
};



inline void print_arguments() {
  std::cerr << "-Y  (--yaml-file)                         : YAML file\n";
  std::cerr << "-b  (--block-read)                        : default is false: don't block reading\n";
  std::cerr << "-c  (--image-downsampling-factor)         : default is " << default_image_downsampling_factor << " (same size)\n";
  std::cerr << "-d  (--dump-volume) <filename>            : output mesh file\n";
  std::cerr << "-f  (--fps)                               : default is " << default_fps << "\n";
  std::cerr << "-F  (--bilateral-filter                   : default is disabled\n";
  std::cerr << "-S  (--sequence-name)                     : name of sequence\n";
  std::cerr << "-i  (--input-file) <filename>             : input file\n";
  std::cerr << "-k  (--camera)                            : default is defined by input\n";
  std::cerr << "-l  (--icp-threshold)                     : default is " << default_icp_threshold << "\n";
  std::cerr << "-o  (--log-file) <filename>               : default is stdout\n";
  std::cerr << "-m  (--mu)                                : default is " << default_mu << "\n";
  std::cerr << "-n  (--near-plane)                        : default is " << default_near_plane << "\n";
  std::cerr << "-N  (--far-plane)                         : default is " << default_far_plane << "\n";
  std::cerr << "-p  (--init-pose)                         : default is " << default_t_MW_factor.x() << "," << default_t_MW_factor.y() << "," << default_t_MW_factor.z() << "\n";
  std::cerr << "-q  (--no-gui)                            : default is to display gui\n";
  std::cerr << "-r  (--integration-rate)                  : default is " << default_integration_rate << "\n";
  std::cerr << "-s  (--map-dim)                           : default is " << default_map_dim.x() << "," << default_map_dim.y() << "," << default_map_dim.z() << "\n";
  std::cerr << "-t  (--tracking-rate)                     : default is " << default_tracking_rate << "\n";
  std::cerr << "-v  (--map-size)                          : default is " << default_map_size.x() << "," << default_map_size.y() << "," << default_map_size.z() << "\n";
  std::cerr << "-y  (--pyramid-levels)                    : default is 10,5,4\n";
  std::cerr << "-z  (--rendering-rate)                    : default is " << default_rendering_rate << "\n";
  std::cerr << "-g  (--ground-truth) <filename>           : ground truth file\n";
  std::cerr << "-G  (--gt-transform) tx,ty,tz,qx,qy,qz,qw : T_BC (translation and/or rotation)\n";
  std::cerr << "-h  (--help)                              : show this help message\n";
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

Configuration parseArgs(unsigned int argc, char** argv) {
  Configuration config;

  std::stringstream executable_ss;
  executable_ss << argv[0];
  std::string voxel_impl_type;
  std::string sensor_type;

  while(std::getline(executable_ss, voxel_impl_type, '-')) {
    if (voxel_impl_type == "denseslam") break;
  }
  std::getline(executable_ss, voxel_impl_type, '-');
  std::getline(executable_ss, sensor_type, '-');

  int c;
  int option_index = 0;
  YAML::Node yaml_general_config = YAML::Load("");
  YAML::Node yaml_map_config = YAML::Load("");
  YAML::Node yaml_sensor_config = YAML::Load("");
  YAML::Node yaml_voxel_impl_config = YAML::Load("");
  while ((c = getopt_long(argc, argv, short_options.c_str(), long_options,
                          &option_index)) != -1) {
    if (c == 'Y')  {
      yaml_general_config = YAML::LoadFile(optarg)["general"];
      yaml_map_config = YAML::LoadFile(optarg)["map"];
      yaml_sensor_config = YAML::LoadFile(optarg)["sensor"];
      yaml_voxel_impl_config = YAML::LoadFile(optarg)["voxel_impl"];
    }
  }

  // CONFIGURE GENERAL
  // No GUI
  config.no_gui = (yaml_general_config.Type() != YAML::NodeType::Null && yaml_general_config["no_gui"])
                  ? yaml_general_config["no_gui"].as<bool>() : default_no_gui;

  // Sequence name
  config.sequence_name = (yaml_general_config.Type() != YAML::NodeType::Null && yaml_general_config["sequence_name"])
                        ? yaml_general_config["sequence_name"].as<std::string>() : default_sequence_name;

  // Input file path
  config.input_file = (yaml_general_config.Type() != YAML::NodeType::Null && yaml_general_config["input_file"])
                      ? yaml_general_config["input_file"].as<std::string>() : default_input_file;

  // Ground truth file path
  config.groundtruth_file = (yaml_general_config.Type() != YAML::NodeType::Null && yaml_general_config["groundtruth_file"])
                      ? yaml_general_config["groundtruth_file"].as<std::string>() : default_groundtruth_file;

  // Mesh file path
  config.dump_volume_file = (yaml_general_config.Type() != YAML::NodeType::Null && yaml_general_config["dump_volume_file"])
                            ? yaml_general_config["dump_volume_file"].as<std::string>() : default_dump_volume_file;
  // Log file path
  config.log_file = (yaml_general_config.Type() != YAML::NodeType::Null && yaml_general_config["log_file"])
                    ? yaml_general_config["log_file"].as<std::string>() : default_dump_volume_file;

  // Integration rate
  config.integration_rate = (yaml_general_config.Type() != YAML::NodeType::Null && yaml_general_config["integration_rate"])
                            ? yaml_general_config["integration_rate"].as<int>() : default_integration_rate;
  // Tracking rate
  config.tracking_rate = (yaml_general_config.Type() != YAML::NodeType::Null && yaml_general_config["tracking_rate"])
                         ? yaml_general_config["tracking_rate"].as<int>() : default_tracking_rate;
  // Rendering rate
  config.rendering_rate = (yaml_general_config.Type() != YAML::NodeType::Null && yaml_general_config["rendering_rate"])
                          ? yaml_general_config["rendering_rate"].as<int>() : default_rendering_rate;
  // Frames per second
  config.fps = (yaml_general_config.Type() != YAML::NodeType::Null && yaml_general_config["fps"])
               ? yaml_general_config["fps"].as<float>() : default_fps;

  // Blocking read
  config.blocking_read = (yaml_general_config.Type() != YAML::NodeType::Null && yaml_general_config["blocking_read"])
                         ? yaml_general_config["blocking_read"].as<bool>() : default_blocking_read;

  // ICP threshold
  config.icp_threshold = (yaml_general_config.Type() != YAML::NodeType::Null && yaml_general_config["icp_threshold"])
                         ? yaml_general_config["icp_threshold"].as<float>() : default_icp_threshold;
  // Render volume fullsize
  config.render_volume_fullsize = (yaml_general_config.Type() != YAML::NodeType::Null && yaml_general_config["render_volume_fullsize"])
                                  ? yaml_general_config["render_volume_fullsize"].as<bool>() : default_render_volume_fullsize;
  // Bilateral filter
  config.bilateral_filter = (yaml_general_config.Type() != YAML::NodeType::Null && yaml_general_config["bilateral_filter"])
                            ? yaml_general_config["bilateral_filter"].as<bool>() : default_bilateral_filter;

  config.pyramid.clear();
  if (yaml_general_config.Type() != YAML::NodeType::Null && yaml_general_config["pyramid"]) {
    config.pyramid = yaml_general_config["pyramid"].as<std::vector<int>>();
  } else {
    for (int i = 0; i < default_iteration_count; i++) {
      config.pyramid.push_back(default_iterations[i]);
    }
  }


  // CONFIGURE MAP

  // Map size
  config.map_size = (yaml_map_config.Type() != YAML::NodeType::Null && yaml_map_config["map_size"])
                    ? Eigen::Vector3i::Constant(yaml_map_config["map_size"].as<int>()) : default_map_size;
  // Map dimension
  config.map_dim = (yaml_map_config.Type() != YAML::NodeType::Null && yaml_map_config["map_dim"])
                   ? Eigen::Vector3f::Constant(yaml_map_config["map_dim"].as<float>()) : default_map_dim;
  // World to Map frame translation
  config.t_MW_factor = (yaml_map_config.Type() != YAML::NodeType::Null && yaml_map_config["t_MW_factor"])
                       ? Eigen::Vector3f(yaml_map_config["t_MW_factor"].as<std::vector<float>>().data()) : default_t_MW_factor;


  // CONFIGURE SENSOR

  // Sensor type
  config.sensor_type = (yaml_sensor_config.Type() != YAML::NodeType::Null && yaml_sensor_config["type"])
                       ? yaml_sensor_config["type"].as<std::string>() : sensor_type;
  // Sensor intrinsics
  if (yaml_sensor_config.Type() != YAML::NodeType::Null && yaml_sensor_config["camera"]) {
    config.camera = Eigen::Vector4f((yaml_sensor_config["camera"].as<std::vector<float>>()).data());
  } else {
    config.camera = default_camera;
  }
  // Sensor overrided
  config.camera_overrided = false;
  // Image downsamling factor
  config.image_downsampling_factor = (yaml_sensor_config.Type() != YAML::NodeType::Null && yaml_sensor_config["image_downsampling_factor"])
                                     ? yaml_sensor_config["image_downsampling_factor"].as<int>() : default_image_downsampling_factor;
  // Left hand coordinate frame
  config.left_hand_frame = (yaml_sensor_config.Type() != YAML::NodeType::Null && yaml_sensor_config["left_hand_frame"])
                           ? yaml_sensor_config["left_hand_frame"].as<bool>() : default_left_hand_frame;
  // Camera to Body frame transformation
  config.T_BC = (yaml_sensor_config.Type() != YAML::NodeType::Null && yaml_sensor_config["T_BC"])
                ? Eigen::Matrix4f(TvtoT(yaml_sensor_config["T_BC"].as<std::vector<float>>())) : default_gt_transform;
  // Near plane
  config.near_plane = (yaml_sensor_config.Type() != YAML::NodeType::Null && yaml_sensor_config["near_plane"])
                      ? yaml_sensor_config["near_plane"].as<float>() : default_near_plane;
  // Far plane
  config.far_plane = (yaml_sensor_config.Type() != YAML::NodeType::Null && yaml_sensor_config["far_plane"])
                     ? yaml_sensor_config["far_plane"].as<float>() : default_far_plane;


  // CONFIGURE VOXEL IMPL
  // Voxel impl type
  config.voxel_impl_type = (yaml_voxel_impl_config.Type() != YAML::NodeType::Null && yaml_voxel_impl_config["type"])
                            ? yaml_voxel_impl_config["type"].as<std::string>() : voxel_impl_type;
  // Mu
  config.mu = (yaml_voxel_impl_config.Type() != YAML::NodeType::Null && yaml_voxel_impl_config["mu"])
              ? yaml_voxel_impl_config["mu"].as<float>() : default_mu;
  (yaml_voxel_impl_config.Type() != YAML::NodeType::Null) ? VoxelImpl::configure(yaml_voxel_impl_config) : VoxelImpl::configure();;

  // Reset getopt_long state to start parsing from the beginning
  optind = 1;
  option_index = 0;
  std::vector<std::string> tokens;
  Eigen::Vector3f gt_transform_tran;
  Eigen::Quaternionf gt_transform_quat;
  while ((c = getopt_long(argc, argv, short_options.c_str(), long_options,
          &option_index)) != -1) {
    switch (c) {
      case 'Y': // yaml-file
        break;

      case 'b': // blocking-read
        config.blocking_read = true;
        break;

      case 'c': // image-downsampling-factor
        config.image_downsampling_factor = atoi(optarg);
        if (   (config.image_downsampling_factor != 1)
            && (config.image_downsampling_factor != 2)
            && (config.image_downsampling_factor != 4)
            && (config.image_downsampling_factor != 8)) {
          std::cerr << "Error: --image-resolution-ratio (-c) must be 1, 2 ,4 "
              << "or 8  (was " << optarg << ")\n";
          exit(EXIT_FAILURE);
        }
        break;

      case 'd': // dump-volume
        config.dump_volume_file = optarg;
        break;

      case 'f': // fps
        config.fps = atof(optarg);
        if (config.fps < 0) {
          std::cerr << "Error: --fps (-f) must be >= 0 (was " << optarg << ")\n";
          exit(EXIT_FAILURE);
        }
        break;

      case 'g': // ground-truth
        config.groundtruth_file = optarg;
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

      case 'i': // input-file
        config.input_file = optarg;
        struct stat st;
        if (stat(config.input_file.c_str(), &st) != 0) {
          std::cerr << "Error: --input-file (-i) does not exist (was "
              << config.input_file << ")\n";
          exit(EXIT_FAILURE);
        }
        break;

      case 'k': // camera
        config.camera = atof4(optarg);
        config.camera_overrided = true;
        if (config.camera.y() < 0) {
          config.left_hand_frame = true;
          std::cerr << "update to left hand coordinate system" << std::endl;
        }
        break;

      case 'o': // log-file
        config.log_file = optarg;
        break;

      case 'l': // icp-threshold
        config.icp_threshold = atof(optarg);
        break;

      case 'm': // mu
        config.mu = atof(optarg);
        break;

      case 'n': // near-plane
        config.near_plane = atof(optarg);
        break;

      case 'N': // far-plane
        config.far_plane = atof(optarg);
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

      case 'F': // bilateral-filter
        config.bilateral_filter = true;
        break;

      default:
        print_arguments();
        exit(EXIT_FAILURE);
    }
  }

  std::replace(config.sequence_name.begin(), config.sequence_name.end(), ' ', '_');

  // Ensure the parameter values are valid.
  if (config.near_plane >= config.far_plane) {
    std::cerr << "Error: Near plane must be smaller than far plane ("
              << config.near_plane << " >= " << config.far_plane << ")\n";
    exit(EXIT_FAILURE);
  }

  std::cout << config;
  VoxelImpl::print_config(std::cout);
  return config;
}

#endif

