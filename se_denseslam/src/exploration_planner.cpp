// SPDX-FileCopyrightText: 2021 Smart Robotics Lab
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#include "se/exploration_planner.hpp"

namespace se {
  ExplorationPlanner::ExplorationPlanner(const OctreePtr          map,
                                         const Eigen::Matrix4f&   T_MW,
                                         const ExplorationConfig& config)
      : map_(map), config_(config), T_MW_(T_MW), T_WM_(se::math::to_inverse_transformation(T_MW))
  {
  }

  void ExplorationPlanner::setT_WC(const Eigen::Matrix4f& T_WC)
  {
    T_MC_history_.poses.push_back(T_MW_ * T_WC);
  }



  Eigen::Matrix4f ExplorationPlanner::getT_WC()
  {
    return T_WM_ * T_MC_history_.poses.back();
  }



  Path ExplorationPlanner::getT_WCHistory() const
  {
    return T_MC_history_.poses;
  }



  bool ExplorationPlanner::goalReached() const
  {
    const Eigen::Matrix4f& current_T_MC = T_MC_history_.poses.back();
    const Eigen::Matrix4f& goal_T_MC = goal_view_.isValid() ? goal_view_.goal() : current_T_MC;
    if (math::position_error(goal_T_MC, current_T_MC).norm() > goal_position_threshold_) {
      return false;
    }
    if (fabsf(math::yaw_error(goal_T_MC, current_T_MC)) > goal_yaw_threshold_) {
      return false;
    }
    if (fabsf(math::pitch_error(goal_T_MC, current_T_MC)) > goal_roll_pitch_threshold_) {
      return false;
    }
    if (fabsf(math::roll_error(goal_T_MC, current_T_MC)) > goal_roll_pitch_threshold_) {
      return false;
    }
    return true;
  }



  Path ExplorationPlanner::computeNextPath_WC(const std::set<key_t>& frontiers,
                                              const Objects&         objects,
                                              const SensorImpl&      sensor)
  {
    //freeCurrentPosition(); // TODO call on map
    const std::vector<key_t> frontier_vec(frontiers.begin(), frontiers.end());
    SinglePathExplorationPlanner planner (map_, frontier_vec, objects, sensor, T_MC_history_, config_);
    candidate_views_ = planner.views();
    rejected_candidate_views_ = planner.rejectedViews();
    goal_view_ = planner.bestView();
    const Path& goal_path_M = planner.bestPath();
    // Convert the goal path from the map to the world frame.
    Path goal_path_W (goal_path_M.size());
    std::transform(goal_path_M.begin(), goal_path_M.end(), goal_path_W.begin(),
        [&](const Eigen::Matrix4f& T_MC) { return T_WM_ * T_MC; });
    return goal_path_W;
  }



  std::vector<CandidateView> ExplorationPlanner::candidateViews() const
  {
    return candidate_views_;
  }



  std::vector<CandidateView> ExplorationPlanner::rejectedCandidateViews() const
  {
    return rejected_candidate_views_;
  }



  CandidateView ExplorationPlanner::goalView() const
  {
    return goal_view_;
  }



  Image<uint32_t> ExplorationPlanner::renderEntropy(const SensorImpl& sensor,
                                                    const bool        visualize_yaw)
  {
    return goal_view_.renderEntropy(*map_, sensor, visualize_yaw);
  }



  Image<uint32_t> ExplorationPlanner::renderEntropyDepth(const SensorImpl& sensor,
                                                         const bool        visualize_yaw)
  {
    return goal_view_.renderDepth(*map_, sensor, visualize_yaw);
  }
} // namespace se

