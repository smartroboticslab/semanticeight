// SPDX-FileCopyrightText: 2019-2020 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2019 Anna Dai
// SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#ifndef __SINGLE_PATH_EXPLORATION_PLANNER_HPP
#define __SINGLE_PATH_EXPLORATION_PLANNER_HPP

#include <deque>

#include <se/candidate_view.hpp>
#include <se/pose_history.hpp>

namespace se {
  struct ExplorationConfig {
    int num_candidates;
    CandidateConfig candidate_config;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };



  class SinglePathExplorationPlanner {
    public:
      /** \brief Generate the next-best-view on construction.
       */
      SinglePathExplorationPlanner(const OctreePtr               map,
                                   const std::vector<se::key_t>& frontiers,
                                   const Objects&                objects,
                                   const SensorImpl&             sensor,
                                   const PoseHistory&            T_MC_history,
                                   const ExplorationConfig&      config);

      CandidateView bestView() const;

      Path bestPath() const;

      float bestUtility() const;

      std::vector<CandidateView> views() const;

      std::vector<CandidateView> rejectedViews() const;

      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
      ExplorationConfig config_;
      std::vector<CandidateView> candidates_;
      std::vector<CandidateView> rejected_candidates_;
      size_t best_idx_;

      static Eigen::Vector3f sampleCandidate(const OctreePtr        map,
                                             std::deque<se::key_t>& frontiers,
                                             const Objects&         objects,
                                             const SensorImpl&      sensor,
                                             const PoseHistory&     T_MC_history,
                                             const int              sampling_step);
  };
} // namespace se

#endif // __EXPLORATION_PLANNER_HPP

