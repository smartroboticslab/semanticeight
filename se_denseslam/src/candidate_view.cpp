// SPDX-FileCopyrightText: 2019-2020 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2019 Anna Dai
// SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#include "se/candidate_view.hpp"

#include <se/image_utils.hpp>
#include <se/utils/math_utils.h>

/** Return the width of the window of a 360-degree image with width w that a sensor with horizontal
 * field of view hfov can view.
 */
int window_width(int w, float hfov) {
    const float window_percentage = hfov / (2.0f * M_PI_F);
    return window_percentage * w + 0.5f;
}



namespace se {
  CandidateView::CandidateView()
      : path_length_(-1.0f),
        path_time_(-1.0f),
        entropy_image_(1, 1),
        frustum_overlap_image_(1, 1),
        min_scale_image_(1, 1),
        entropy_(-1.0f),
        lod_gain_(-1.0f),
        utility_(-1.0f) {
  }



  CandidateView::CandidateView(const Eigen::Vector3f& t_MB)
      : CandidateView::CandidateView()
  {
    path_MB_.push_back(Eigen::Matrix4f::Identity());
    path_MB_.back().topRightCorner<3,1>() = t_MB;
  }



  CandidateView::CandidateView(const OctreePtr&              map,
                               ptp::OccupancyWorld&          ptp_map,
                               const std::vector<se::key_t>& /*frontiers*/,
                               const Objects&                objects,
                               const SensorImpl&             sensor,
                               const Eigen::Matrix4f&        T_MB,
                               const Eigen::Matrix4f&        T_BC,
                               const PoseHistory&            T_MC_history,
                               const CandidateConfig&        config)
      : path_length_(-1.0f),
        path_time_(-1.0f),
        entropy_image_(config.raycast_width, config.raycast_height),
        frustum_overlap_image_(config.raycast_width, 1, 0.0f),
        min_scale_image_(window_width(config.raycast_width, sensor.horizontal_fov), config.raycast_height),
        entropy_(-1.0f),
        lod_gain_(-1.0f),
        utility_(-1.0f),
        config_(config) {
    // Set-up the planner
    const ptp::PlanningParameter planner_config (config_.planner_config);
    ptp::ProbCollisionChecker planner_collision_checker (ptp_map, planner_config);
    ptp::SafeFlightCorridorGenerator planner (ptp_map, planner_collision_checker, planner_config);
    if (config_.planner_config.start_t_MB_.isApprox(config_.planner_config.goal_t_MB_)) {
      // No need to do path planning if start and goal positions are the same
      path_MB_.push_back(T_MB);
      path_MB_.push_back(T_MB);
    } else {
      // Plan a path to the goal
      if (planner.planPath(config_.planner_config.start_t_MB_, config_.planner_config.goal_t_MB_)
          == ptp::PlanningResult::Failed) {
        // Could not plan a path. Add the attempted goal point to the path for visualization
        path_MB_.push_back(Eigen::Matrix4f::Identity());
        path_MB_.back().topRightCorner<3,1>() = config_.planner_config.goal_t_MB_;
        return;
      }
      path_MB_ = convertPath(planner.getPath());
    }
    // Raycast to compute the optimal yaw angle.
    entropyRaycast(*map, sensor, T_BC, T_MC_history);
    path_MB_.back().topLeftCorner<3,3>() = yawToC_MB(yaw_M_);
    // Get the LoD gain of the objects.
    const SensorImpl raycasting_sensor (sensor, 0.5f);
    lod_gain_ = lod_gain_raycasting(objects, sensor, raycasting_sensor, path_MB_.back() * T_BC);
    // Compute the utility.
    path_time_ = pathTime(path_MB_, config_.velocity_linear, config_.velocity_angular);
    computeUtility();
  }



  bool CandidateView::isValid() const {
    return utility_ != -1.0f;
  }



  Path CandidateView::path() const {
    return path_MB_;
  }



  float CandidateView::utility() const {
    return utility_;
  }



  Eigen::Matrix4f CandidateView::goalT_MB() const {
    return path_MB_.back();
  }



  void CandidateView::computeIntermediateYaw(const Octree<VoxelImpl::VoxelType>& map,
                                             const SensorImpl&                   sensor,
                                             const Eigen::Matrix4f&              T_BC,
                                             const PoseHistory&                  T_MC_history) {
    // Raycast and optimize yaw at each intermediate path vertex
    for (size_t i = 1; i < path_MB_.size() - 1; i++) {
      const Eigen::Matrix4f T_MC = path_MB_[i] * T_BC;
      Image<float> entropy_image (entropy_image_.width(), entropy_image_.height());
      Image<float> frustum_overlap_image (entropy_image_.width(), 1);
      raycast_entropy(entropy_image, map, sensor, T_MC.topRightCorner<3,1>());
      frustum_overlap(frustum_overlap_image, sensor, T_MC, T_MC_history);
      const std::pair<float, float> r = optimal_yaw(entropy_image, frustum_overlap_image, sensor);
      path_MB_[i].topLeftCorner<3,3>() = yawToC_MB(r.first);
    }
  }



  Image<uint32_t> CandidateView::renderEntropy(const Octree<VoxelImpl::VoxelType>& map,
                                               const SensorImpl&                   sensor,
                                               const Eigen::Matrix4f&              T_BC,
                                               const bool                          visualize_yaw) const {
    const Eigen::Vector2i res (entropy_image_.width(), entropy_image_.height());
    Eigen::Matrix4f T_MB = Eigen::Matrix4f::Identity();
    T_MB.topRightCorner<3,1>() = config_.planner_config.goal_t_MB_;
    const Eigen::Matrix4f T_MC = T_MB * T_BC;
    Image<float> entropy (res.x(), res.y());
    raycast_entropy(entropy, map, sensor, T_MC.topRightCorner<3,1>());
    // Visualize the entropy in a colour image by reusing the depth colourmar
    Image<uint32_t> entropy_render (res.x(), res.y());
    // Decrease the max entropy somewhat otherwise the render is all black
    for (size_t i = 0; i < entropy.size(); ++i) {
      // Scale and clamp the entropy for visualization since it's values are typically too low.
      const uint8_t e = se::math::clamp(UINT8_MAX * (6.0f * entropy[i]) + 0.5f, 0.0f, static_cast<float>(UINT8_MAX));
      entropy_render[i] = se::pack_rgba(e, e, e, 0xFF);
    }
    // Using the depth colourmap does not result in a very intuitive visualization
    //se::depth_to_rgba(entropy_render.data(), entropy_image_.data(), res, 0.0f, max_e);
    // Visualize the optimal yaw
    if (visualize_yaw) {
      overlay_yaw(entropy_render, yaw_M_, sensor);
    }
    return entropy_render;
  }



  Image<uint32_t> CandidateView::renderDepth(const Octree<VoxelImpl::VoxelType>& map,
                                             const SensorImpl&                   sensor,
                                             const Eigen::Matrix4f&              T_BC,
                                             const bool                          visualize_yaw) const {
    const Eigen::Vector2i res (entropy_image_.width(), entropy_image_.height());
    Eigen::Matrix4f T_MB = Eigen::Matrix4f::Identity();
    T_MB.topRightCorner<3,1>() = config_.planner_config.goal_t_MB_;
    const Eigen::Matrix4f T_MC = T_MB * T_BC;
    // Raycast to get the depth
    Image<float> depth (res.x(), res.y());
    raycast_depth(depth, map, sensor, T_MC.topRightCorner<3,1>());
    // Render to a colour image
    Image<uint32_t> depth_render (res.x(), res.y());
    se::depth_to_rgba(depth_render.data(), depth.data(), res, sensor.near_plane, sensor.far_plane);
    // Visualize the optimal yaw
    if (visualize_yaw) {
      overlay_yaw(depth_render, yaw_M_, sensor);
    }
    return depth_render;
  }



  void CandidateView::entropyRaycast(const Octree<VoxelImpl::VoxelType>& map,
                                     const SensorImpl&                   sensor,
                                     const Eigen::Matrix4f&              T_BC,
                                     const PoseHistory&                  T_MC_history) {
    const Eigen::Matrix4f T_MC = path_MB_.back() * T_BC;
    // Raycast at the last path vertex
    raycast_entropy(entropy_image_, map, sensor, T_MC.topRightCorner<3,1>());
    if (config_.use_pose_history) {
      frustum_overlap(frustum_overlap_image_, sensor, T_MC, T_MC_history);
    }
    const std::pair<float, float> r = optimal_yaw(entropy_image_, frustum_overlap_image_, sensor);
    yaw_M_ = r.first;
    entropy_ = r.second;
  }



  void CandidateView::computeUtility() {
    utility_ = (config_.exploration_weight * entropy_ + (1.0f - config_.exploration_weight) * lod_gain_) / path_time_;
  }



  Eigen::Matrix3f CandidateView::yawToC_MB(const float yaw_M) {
    Eigen::Matrix3f C_MB = Eigen::Matrix3f::Zero();
    C_MB(0,0) =  cos(yaw_M);
    C_MB(0,1) = -sin(yaw_M);
    C_MB(1,0) =  sin(yaw_M);
    C_MB(1,1) =  cos(yaw_M);
    C_MB(2,2) =  1.0f;
    return C_MB;
  }



  Path CandidateView::convertPath(const ptp::Path<ptp::kDim>::Ptr ptp_path) {
    Path path (ptp_path->states.size(), Eigen::Matrix4f::Identity());
    for (size_t i = 0; i < ptp_path->states.size(); ++i) {
      const Eigen::Vector3f& t_MB = ptp_path->states[i].segment_end;
      path[i].topRightCorner<3,1>() = t_MB;
    }
    return path;
  }



  float CandidateView::pathTime(const Path& path, float velocity_linear, float velocity_angular) {
    // Compute the translation time
    float t_tran = 0.0f;
    for (size_t i = 0; i < path.size() - 1; ++i) {
      const Eigen::Matrix4f& start = path[i];
      const Eigen::Matrix4f& end = path[i + 1];
      const Eigen::Vector3f translation = end.topRightCorner<3,1>() - start.topRightCorner<3,1>();
      t_tran += translation.norm() / velocity_linear;
    }
    // Compute the rotation time
    float t_rot = 0.0f;
    for (size_t i = 0; i < path.size() - 1; ++i) {
      const float yaw_diff = math::yaw_error(path[i], path[i + 1]);
      t_rot += yaw_diff / velocity_angular;
    }
    const float t = std::max(t_tran, t_rot);
    return (std::fabs(t) > 10.0f * FLT_EPSILON) ? t : NAN;
  }
} // namespace se

