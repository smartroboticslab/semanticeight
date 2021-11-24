// SPDX-FileCopyrightText: 2019-2020 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2019 Anna Dai
// SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#ifndef __SINGLE_PATH_EXPLORATION_PLANNER_HPP
#define __SINGLE_PATH_EXPLORATION_PLANNER_HPP

#include <deque>

#include "se/candidate_view.hpp"
#include "se/morton_sampling_tree.hpp"
#include "se/pose_vector_history.hpp"

namespace se {
struct ExplorationConfig {
    int num_candidates;
    Eigen::Vector3f sampling_min_M;
    Eigen::Vector3f sampling_max_M;
    CandidateConfig candidate_config;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};



class SinglePathExplorationPlanner {
    public:
    /** \brief Generate the next-best-view on construction.
       */
    SinglePathExplorationPlanner(const OctreePtr map,
                                 const std::vector<se::key_t>& frontiers,
                                 const Objects& objects,
                                 const SensorImpl& sensor,
                                 const Eigen::Matrix4f& T_MB,
                                 const Eigen::Matrix4f& T_BC,
                                 const PoseVectorHistory& T_MB_history,
                                 const PoseVectorHistory& T_MC_history,
                                 const ExplorationConfig& config);

    CandidateView bestView() const;

    size_t bestViewIndex() const;

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

    /** Sample a candidate position mostly like ICRA 2020, iterate over the sorted frontiers list
     * every sampling_step elements. Instead of sampling a random voxel as in ICRA 2020, sample the
     * center of the volume since we might have node-level frontiers now.
     */
    static Eigen::Vector3f sampleCandidate(const OctreePtr map,
                                           std::deque<se::key_t>& frontiers,
                                           const Objects& objects,
                                           const SensorImpl& sensor,
                                           const PoseVectorHistory& T_MC_history,
                                           const int sampling_step,
                                           const Eigen::Vector3f& sampling_min_M,
                                           const Eigen::Vector3f& sampling_max_M);

    /** Use an se::MortonSamplingTree to sample frontiers more uniformly in space.
     */
    static Eigen::Vector3f sampleCandidate(const OctreePtr map,
                                           MortonSamplingTree& sampling_tree,
                                           const Eigen::Vector3f& sampling_min_M,
                                           const Eigen::Vector3f& sampling_max_M);

    /** Sample a candidate position by randomly selecting a frontier.
     */
    static Eigen::Vector3f sampleCandidate(const OctreePtr map,
                                           std::deque<se::key_t>& frontiers,
                                           const Eigen::Vector3f& sampling_min_M,
                                           const Eigen::Vector3f& sampling_max_M);
};
} // namespace se

#endif // __EXPLORATION_PLANNER_HPP
