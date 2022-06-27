// SPDX-FileCopyrightText: 2019-2020 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2019 Anna Dai
// SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#ifndef __SINGLE_PATH_EXPLORATION_PLANNER_HPP
#define __SINGLE_PATH_EXPLORATION_PLANNER_HPP

#include <deque>

#include "se/candidate_view.hpp"
#include "se/morton_sampling_tree.hpp"
#include "se/pose_grid_history.hpp"
#include "se/pose_mask_history.hpp"
#include "se/pose_vector_history.hpp"

namespace se {

typedef std::shared_ptr<const se::Octree<VoxelImpl::VoxelType>> OctreeConstPtr;

struct ExplorationConfig {
    // Sampling settings
    int num_candidates;
    float frontier_sampling_probability;
    CandidateConfig candidate_config;

    ExplorationConfig(const Configuration& c = Configuration(),
                      const Eigen::Matrix4f& T_MW = Eigen::Matrix4f::Identity());

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};



class SinglePathExplorationPlanner {
    public:
    /** \brief Generate the next-best-view on construction.
     * \warning The map frame (M) MUST be z-up.
     */
    SinglePathExplorationPlanner(const OctreeConstPtr map,
                                 const std::vector<se::key_t>& frontiers,
                                 const Objects& objects,
                                 const SensorImpl& sensor,
                                 const Eigen::Matrix4f& T_MB,
                                 const Eigen::Matrix4f& T_BC,
                                 const PoseGridHistory& T_MB_grid_history,
                                 const PoseMaskHistory& T_MB_mask_history,
                                 const PoseVectorHistory& T_MB_history,
                                 const ExplorationConfig& config);

    size_t bestViewIndex() const;

    Path bestPath() const;

    float bestUtility() const;

    std::vector<CandidateView> views() const;

    std::vector<CandidateView> rejectedViews() const;

    /** \brief Return true if the utility computation with and without the object gain resulted in
     * the same best candidate.
     */
    bool explorationDominant() const;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
    ExplorationConfig config_;
    std::vector<CandidateView> candidates_;
    std::vector<CandidateView> rejected_candidates_;
    size_t best_idx_;
    bool exploration_dominant_;

    /** Sample a candidate position mostly like ICRA 2020, iterate over the sorted frontiers list
     * every sampling_step elements. Instead of sampling a random voxel as in ICRA 2020, sample the
     * center of the volume since we might have node-level frontiers now.
     */
    static Eigen::Vector3f sampleCandidate(const se::Octree<VoxelImpl::VoxelType>& map,
                                           std::deque<se::key_t>& frontiers,
                                           const Objects& objects,
                                           const int sampling_step,
                                           const Eigen::Vector3f& sampling_min_M,
                                           const Eigen::Vector3f& sampling_max_M);

    /** Use an se::MortonSamplingTree to sample frontiers more uniformly in space.
     */
    static Eigen::Vector3f sampleCandidate(const se::Octree<VoxelImpl::VoxelType>& map,
                                           MortonSamplingTree& sampling_tree,
                                           const Eigen::Vector3f& sampling_min_M,
                                           const Eigen::Vector3f& sampling_max_M);

    /** Sample a candidate position by randomly selecting a frontier or object.
     */
    static Eigen::Vector3f sampleCandidate(const se::Octree<VoxelImpl::VoxelType>& map,
                                           std::deque<se::key_t>& frontiers,
                                           std::deque<ObjectPtr>& objects,
                                           const Eigen::Vector3f& sampling_min_M,
                                           const Eigen::Vector3f& sampling_max_M,
                                           const float frontier_sampling_prob = 0.5f);
};
} // namespace se

#endif // __EXPLORATION_PLANNER_HPP
