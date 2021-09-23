// SPDX-FileCopyrightText: 2021 Smart Robotics Lab
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#include "se/exploration_planner.hpp"
#include "se/exploration_utils.hpp"

namespace se {
  constexpr float ExplorationPlanner::goal_xy_threshold_;
  constexpr float ExplorationPlanner::goal_z_threshold_;
  constexpr float ExplorationPlanner::goal_yaw_threshold_;
  constexpr float ExplorationPlanner::goal_roll_pitch_threshold_;

  ExplorationPlanner::ExplorationPlanner(const OctreePtr          map,
                                         const Eigen::Matrix4f&   T_MW,
                                         const Eigen::Matrix4f&   T_BC,
                                         const ExplorationConfig& config)
      : map_(map), config_(config), T_MW_(T_MW), T_WM_(se::math::to_inverse_transformation(T_MW)),
        T_BC_(T_BC), T_CB_(se::math::to_inverse_transformation(T_BC))
  {
    ompl::msg::setLogLevel(ompl::msg::LOG_ERROR);
  }



  void ExplorationPlanner::setT_WB(const Eigen::Matrix4f& T_WB)
  {
    T_MB_history_.poses.push_back(T_MW_ * T_WB);
    T_MC_history_.poses.push_back(T_MB_history_.poses.back() * T_BC_);
  }



  Eigen::Matrix4f ExplorationPlanner::getT_WB() const
  {
    return T_WM_ * T_MB_history_.poses.back();
  }



  Path ExplorationPlanner::getT_WBHistory() const
  {
    const Path& T_MB_history = T_MB_history_.poses;
    Path T_WB_history (T_MB_history.size());
    std::transform(T_MB_history.begin(), T_MB_history.end(), T_WB_history.begin(),
        [&](const auto& T_MB) { return T_WM_ * T_MB; });
    return T_WB_history;
  }



  bool ExplorationPlanner::needsNewGoal() const
  {
    return goal_path_T_MB_.empty();
  }



  bool ExplorationPlanner::goalReached()
  {
    // If there is no goal, we have reached the goal.
    if (goal_path_T_MB_.empty()) {
      return true;
    }
    // Compute the error's between the current pose and the next vertex of the goal path.
    const Eigen::Matrix4f& current_T_MB = T_MB_history_.poses.back();
    const Eigen::Matrix4f& goal_T_MB = goal_path_T_MB_.front();
    const Eigen::Vector3f pos_error = math::position_error(goal_T_MB, current_T_MB);
    if (pos_error.head<2>().norm() > goal_xy_threshold_) {
      return false;
    }
    if (pos_error.tail<1>().norm() > goal_z_threshold_) {
      return false;
    }
    if (fabsf(math::yaw_error(goal_T_MB, current_T_MB)) > goal_yaw_threshold_) {
      return false;
    }
    if (fabsf(math::pitch_error(goal_T_MB, current_T_MB)) > goal_roll_pitch_threshold_) {
      return false;
    }
    if (fabsf(math::roll_error(goal_T_MB, current_T_MB)) > goal_roll_pitch_threshold_) {
      return false;
    }
    // Remove the reached pose from the goal path.
    goal_path_T_MB_.pop();
    return true;
  }



  bool ExplorationPlanner::goalT_WB(Eigen::Matrix4f& T_WB) const
  {
    if (goal_path_T_MB_.empty()) {
      return false;
    } else {
      T_WB = T_WM_ * goal_path_T_MB_.front();
      return true;
    }
  }



  Path ExplorationPlanner::computeNextPath_WB(const std::set<key_t>& frontiers,
                                              const Objects&         objects,
                                              const SensorImpl&      sensor)
  {
    const std::vector<key_t> frontier_vec(frontiers.begin(), frontiers.end());
    SinglePathExplorationPlanner planner (map_, frontier_vec, objects, sensor, T_BC_, T_MB_history_, T_MC_history_, config_);
    candidate_views_ = planner.views();
    rejected_candidate_views_ = planner.rejectedViews();
    goal_view_ = planner.bestView();
    const Path& goal_path_M = planner.bestPath();
    // Add it to the goal path queue.
    for (const auto& T_MB : goal_path_M) {
      goal_path_T_MB_.push(T_MB);
    }
    // Convert the goal path from the map to the world frame.
    Path goal_path_W (goal_path_M.size());
    std::transform(goal_path_M.begin(), goal_path_M.end(), goal_path_W.begin(),
        [&](const Eigen::Matrix4f& T_MB) { return T_WM_ * T_MB; });
    return goal_path_W;
  }



  const std::vector<CandidateView>& ExplorationPlanner::candidateViews() const
  {
    return candidate_views_;
  }



  const std::vector<CandidateView>& ExplorationPlanner::rejectedCandidateViews() const
  {
    return rejected_candidate_views_;
  }



  const CandidateView& ExplorationPlanner::goalView() const
  {
    return goal_view_;
  }



  Image<uint32_t> ExplorationPlanner::renderEntropy(const SensorImpl& sensor,
                                                    const bool        visualize_yaw)
  {
    return goal_view_.renderEntropy(*map_, sensor, T_BC_, visualize_yaw);
  }



  Image<uint32_t> ExplorationPlanner::renderEntropyDepth(const SensorImpl& sensor,
                                                         const bool        visualize_yaw)
  {
    return goal_view_.renderDepth(*map_, sensor, T_BC_, visualize_yaw);
  }



  Image<uint32_t> ExplorationPlanner::renderMinScale(const SensorImpl& sensor)
  {
    return goal_view_.renderMinScale(*map_, sensor, T_BC_);
  }



  int ExplorationPlanner::writePathPLY(const std::string& filename) const {
    std::ofstream f (filename);
    if (!f.is_open()) {
      std::cerr << "Unable to write file " << filename << "\n";
      return 1;
    }
    const size_t num_vertices = T_MB_history_.poses.size();
    const size_t num_edges = num_vertices - 1;
    f << "ply\n";
    f << "format ascii 1.0\n";
    f << "comment Generated by supereight\n";
    f << "element vertex " << num_vertices << "\n";
    f << "property float x\n";
    f << "property float y\n";
    f << "property float z\n";
    f << "property uchar red\n";
    f << "property uchar green\n";
    f << "property uchar blue\n";
    f << "element edge " << num_edges << "\n";
    f << "property int vertex1\n";
    f << "property int vertex2\n";
    f << "end_header\n";
    for (const auto& T_MB : T_MB_history_.poses) {
      const Eigen::Vector3f& t_WB = (T_WM_ * T_MB).topRightCorner<3,1>();
      f << t_WB.x() << " " << t_WB.y() << " " << t_WB.z() << " 255 0 0\n";
    }
    for (size_t i = 0; i < num_edges; i++) {
      f << i << " " << i + 1 << "\n";
    }
    f.close();
    return 0;
  }



  int ExplorationPlanner::writePathTSV(const std::string& filename) const {
    std::ofstream f (filename);
    if (!f.is_open()) {
      std::cerr << "Unable to write file " << filename << "\n";
      return 1;
    }
    f << "tx\tty\ttz\tqx\tqy\tqz\tqw\n";
    for (const auto& T_MB : T_MB_history_.poses) {
      const Eigen::Matrix4f& T_WB = T_WM_ * T_MB;
      const Eigen::Vector3f& t_WB = T_WB.topRightCorner<3,1>();
      const Eigen::Quaternionf q_WB (T_WB.topLeftCorner<3,3>());
      f << t_WB.x() << "\t" << t_WB.y() << "\t" << t_WB.z()
        << "\t" << q_WB.x() << "\t" << q_WB.y() << "\t" << q_WB.z() << "\t" << q_WB.w() << "\n";
    }
    f.close();
    return 0;
  }
} // namespace se

