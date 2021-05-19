// SPDX-FileCopyrightText: 2019-2020 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2019 Anna Dai
// SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#include "se/single_path_exploration_planner.hpp"

namespace se {
  SinglePathExplorationPlanner::SinglePathExplorationPlanner(const OctreePtr               map,
                                                             const std::vector<se::key_t>& frontiers,
                                                             const Objects&                objects,
                                                             const SensorImpl&             sensor,
                                                             const Eigen::Matrix4f&        T_MC,
                                                             const ExplorationConfig&      config)
      : config_(config),
        best_idx_(SIZE_MAX) {

    std::deque<se::key_t> remaining_frontiers(frontiers.begin(), frontiers.end());
    candidates_.reserve(config_.num_candidates);
    config_.candidate_config.planner_config.start_point_M_ = T_MC.topRightCorner<3,1>();
    // Add the current pose to the candidates
    CandidateConfig candidate_config = config_.candidate_config;
    candidate_config.planner_config.goal_point_M_ = T_MC.topRightCorner<3,1>();
    candidates_.emplace_back(map, frontiers, objects, sensor, T_MC, candidate_config);
    // Sample the candidate views aborting after a number of failed retries
    const size_t max_failed = 5 * config_.num_candidates;
    const int sampling_step = std::ceil(remaining_frontiers.size() / config_.num_candidates);
    while (candidates_.size() < static_cast<size_t>(config_.num_candidates)
        && rejected_candidates_.size() <= max_failed
        && !remaining_frontiers.empty()) {
      // Sample a point
      const Eigen::Vector3f candidate_t_MC = sampleCandidate(map, remaining_frontiers, objects, sampling_step);
      // Create the config for this particular candidate
      CandidateConfig candidate_config = config_.candidate_config;
      candidate_config.planner_config.goal_point_M_ = candidate_t_MC;
      // Create candidate and compute its utility
      candidates_.emplace_back(map, frontiers, objects, sensor, T_MC, candidate_config);
      // Remove the candidate if it's not valid
      if (!candidates_.back().isValid()) {
        rejected_candidates_.push_back(candidates_.back());
        candidates_.pop_back();
      }
    }
    // Find the best candidate
    float utility_max = 0.0f;
    for (size_t i = 0; i < candidates_.size(); i++) {
      if (candidates_[i].utility() > utility_max) {
        utility_max = candidates_[i].utility();
        best_idx_ = i;
      }
    }
    // Add an invalid candidate to return if no best candidate was found
    if (best_idx_ == SIZE_MAX) {
      best_idx_ = candidates_.size();
      candidates_.emplace_back();
      return;
    }
    // Compute the yaw angles at each path vertex of the best candidate
    candidates_[best_idx_].computeIntermediateYaw(*map, sensor);
  }



  CandidateView SinglePathExplorationPlanner::bestView() const {
    return candidates_[best_idx_];
  }



  Path SinglePathExplorationPlanner::bestPath() const {
    return candidates_[best_idx_].path();
  }



  float SinglePathExplorationPlanner::bestUtility() const {
    return candidates_[best_idx_].utility();
  }



  std::vector<CandidateView> SinglePathExplorationPlanner::views() const {
    return candidates_;
  }



  std::vector<CandidateView> SinglePathExplorationPlanner::rejectedViews() const {
    return rejected_candidates_;
  }



  Eigen::Vector3f SinglePathExplorationPlanner::sampleCandidate(const OctreePtr        map,
                                                                std::deque<se::key_t>& frontiers,
                                                                const Objects&         /*objects*/,
                                                                const int              sampling_step) {
    // TODO take objects into account
    if (frontiers.empty()) {
      return Eigen::Vector3f::Constant(NAN);
    }
    // Use the first element as the code of the sample
    const se::key_t code = frontiers.front();
    frontiers.pop_front();
    // Move the next sampling_step - 1 elements to the back
    if (!frontiers.empty()) {
      for (int i = 0; i < sampling_step - 1; ++i) {
        frontiers.push_back(frontiers.front());
        frontiers.pop_front();
      }
    }
    // Return the coordinates of the sampled Volume's centre
    const int size = map->depthToSize(keyops::depth(code));
    return map->voxelDim() * (keyops::decode(code).cast<float>() + Eigen::Vector3f::Constant(size / 2.0f));
  }
} // namespace se
