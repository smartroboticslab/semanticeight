// SPDX-FileCopyrightText: 2021 Smart Robotics Lab
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#ifndef __EXPLORATION_PLANNER_HPP
#define __EXPLORATION_PLANNER_HPP

#include "se/single_path_exploration_planner.hpp"

namespace se {
  class ExplorationPlanner {
    public:
      ExplorationPlanner(const OctreePtr          map,
                         const Eigen::Matrix4f&   T_MW,
                         const ExplorationConfig& config);

      void setT_WC(const Eigen::Matrix4f& T_WC);

      Eigen::Matrix4f getT_WC();

      Path getT_WCHistory() const;

      bool goalReached() const;

      /** Call the exploration planner and return the resulting camera path in the world frame.
       * The returned path is a series of T_WC.
       */
      Path computeNextPath_WC(const std::set<key_t>& frontiers,
                              const Objects&         objects,
                              const SensorImpl&      sensor);

      std::vector<CandidateView> candidateViews() const;

      std::vector<CandidateView> rejectedCandidateViews() const;

      CandidateView goalView() const;

      Image<uint32_t> renderEntropy(const SensorImpl& sensor,
                                    const bool        visualize_yaw = true);

      Image<uint32_t> renderEntropyDepth(const SensorImpl& sensor,
                                         const bool        visualize_yaw = true);

      /** Write the path vertices in the world frame as a PLY file.
       */
      int writePathPLY(const std::string& filename) const;

      /** Write the path vertices in the world frame as a TSV file.
       */
      int writePathTSV(const std::string& filename) const;

      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
      const OctreePtr map_;
      const ExplorationConfig config_;
      Eigen::Matrix4f T_MW_;
      Eigen::Matrix4f T_WM_;
      PoseHistory T_MC_history_;
      std::vector<CandidateView> candidate_views_;
      std::vector<CandidateView> rejected_candidate_views_;
      CandidateView goal_view_;

      static constexpr float goal_position_threshold_ = 0.2f;
      static constexpr float goal_yaw_threshold_ = math::deg_to_rad * 5.0f;
      static constexpr float goal_roll_pitch_threshold_ = math::deg_to_rad * 10.0f;
  };
} // namespace se

#endif // __EXPLORATION_PLANNER_HPP

