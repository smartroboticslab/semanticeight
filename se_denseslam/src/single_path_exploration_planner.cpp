// SPDX-FileCopyrightText: 2019-2020 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2019 Anna Dai
// SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#include "se/single_path_exploration_planner.hpp"

#include <algorithm>

namespace se {

template<typename FunctionT>
std::pair<size_t, float> best_candidate(const std::vector<CandidateView>& candidates,
                                        const FunctionT get_utility)
{
    const auto comp = [&](const auto& a, const auto& b) { return get_utility(a) < get_utility(b); };
    const auto best_it = std::max_element(candidates.begin(), candidates.end(), comp);
    if (best_it == candidates.end()) {
        return std::make_pair(SIZE_MAX, 0.0f);
    }
    else {
        return std::make_pair(std::distance(candidates.begin(), best_it), get_utility(*best_it));
    }
}



Eigen::Vector3f sample_random_frontier(const se::Octree<VoxelImpl::VoxelType>& map,
                                       std::deque<se::key_t>& frontiers,
                                       const Eigen::Vector3f& sampling_min_M,
                                       const Eigen::Vector3f& sampling_max_M)
{
    std::shuffle(frontiers.begin(), frontiers.end(), std::mt19937{std::random_device{}()});
    const key_t code = frontiers.back();
    frontiers.pop_back();
    // Return the coordinates of the sampled volume's centre.
    const int size = map.depthToSize(keyops::depth(code));
    Eigen::Vector3f pos_M = map.voxelDim()
        * (keyops::decode(code).cast<float>() + Eigen::Vector3f::Constant(size / 2.0f));
    math::clamp(pos_M, sampling_min_M, sampling_max_M);
    return pos_M;
}



Eigen::Vector3f sample_random_object(std::deque<ObjectPtr>& objects,
                                     const Eigen::Vector3f& sampling_min_M,
                                     const Eigen::Vector3f& sampling_max_M)
{
    std::shuffle(objects.begin(), objects.end(), std::mt19937{std::random_device{}()});
    ObjectPtr object = objects.back();
    objects.pop_back();
    // Return the coordinates of the sampled object's centre.
    Eigen::Vector3f pos_M =
        (object->T_MO_ * Eigen::Vector3f::Constant(object->mapDim() / 2.0f).homogeneous())
            .head<3>();
    math::clamp(pos_M, sampling_min_M, sampling_max_M);
    return pos_M;
}



ExplorationConfig::ExplorationConfig(const Configuration& c, const Eigen::Matrix4f& T_MW) :
        num_candidates(c.num_candidates),
        frontier_sampling_probability(c.frontier_sampling_probability),
        candidate_config(c, T_MW)
{
}



SinglePathExplorationPlanner::SinglePathExplorationPlanner(
    const OctreeConstPtr map,
    const std::vector<se::key_t>& frontiers,
    const Objects& objects,
    const SensorImpl& sensor,
    const Eigen::Matrix4f& T_MB,
    const Eigen::Matrix4f& T_BC,
    const PoseGridHistory& /* T_MB_grid_history */,
    const PoseMaskHistory& T_MB_mask_history,
    const PoseVectorHistory& /* T_MB_vector_history */,
    const ExplorationConfig& config) :
        config_(config), best_idx_(SIZE_MAX), exploration_dominant_(true)
{
    const PoseHistory* T_MB_history = &T_MB_mask_history;
    //MortonSamplingTree candidate_sampling_tree(frontiers, map->voxelDepth());
    std::deque<se::key_t> remaining_frontiers(frontiers.begin(), frontiers.end());
    // Get the incomplete objects.
    std::deque<ObjectPtr> remaining_objects;
    std::copy_if(objects.begin(),
                 objects.end(),
                 std::back_inserter(remaining_objects),
                 [](const auto o) { return !o->finished(); });
    candidates_.reserve(config_.num_candidates);
    config_.candidate_config.planner_config.start_t_MB_ = T_MB.topRightCorner<3, 1>();
    // Create a single planner for allcandidates.
    ptp::SafeFlightCorridorGenerator planner(map, config_.candidate_config.planner_config);
    // Add the current pose to the candidates
    CandidateConfig candidate_config = config_.candidate_config;
    candidate_config.planner_config.goal_t_MB_ = T_MB.topRightCorner<3, 1>();
    candidates_.emplace_back(
        *map, planner, frontiers, objects, sensor, T_MB, T_BC, T_MB_history, candidate_config);
    // Remove the candidate if it's not valid
    if (!candidates_.back().isValid()) {
        rejected_candidates_.push_back(candidates_.back());
        candidates_.pop_back();
    }
    const Eigen::Vector3f xyz_threshold(config_.candidate_config.goal_xy_threshold,
                                        config_.candidate_config.goal_xy_threshold,
                                        config_.candidate_config.goal_z_threshold);
    const Eigen::Vector3f sampling_min_M_reduced =
        config_.candidate_config.planner_config.sampling_min_M_ + xyz_threshold;
    const Eigen::Vector3f sampling_max_M_reduced =
        config_.candidate_config.planner_config.sampling_max_M_ - xyz_threshold;
    // Sample the candidate views aborting after a number of failed retries
    const size_t max_failed = 5 * config_.num_candidates;
    //const int sampling_step = std::ceil(remaining_frontiers.size() / config_.num_candidates);
    while (candidates_.size() < static_cast<size_t>(config_.num_candidates)
           //&& rejected_candidates_.size() <= max_failed && !candidate_sampling_tree.empty()) {
           && rejected_candidates_.size() <= max_failed && !remaining_frontiers.empty()) {
        // Sample a point
        //const Eigen::Vector3f candidate_t_MB =
        //    sampleCandidate(*map,
        //                    remaining_frontiers,
        //                    objects,
        //                    sampling_step,
        //                    sampling_min_M_reduced,
        //                    sampling_max_M_reduced);
        //const Eigen::Vector3f candidate_t_MB =
        //    sampleCandidate(*map,
        //                    candidate_sampling_tree,
        //                    sampling_min_M_reduced,
        //                    sampling_max_M_reduced);
        const Eigen::Vector3f candidate_t_MB =
            sampleCandidate(*map,
                            remaining_frontiers,
                            remaining_objects,
                            sampling_min_M_reduced,
                            sampling_max_M_reduced,
                            config_.frontier_sampling_probability);
        // Create the config for this particular candidate
        CandidateConfig candidate_config = config_.candidate_config;
        candidate_config.planner_config.goal_t_MB_ = candidate_t_MB;
        // Create candidate and compute its utility
        candidates_.emplace_back(
            *map, planner, frontiers, objects, sensor, T_MB, T_BC, T_MB_history, candidate_config);
        // Remove the candidate if it's not valid
        if (!candidates_.back().isValid()) {
            rejected_candidates_.push_back(candidates_.back());
            candidates_.pop_back();
        }
    }
    // Find the best candidate based of different utililties.
    const auto combined = best_candidate(candidates_, [](const auto& c) { return c.utility(); });
    const auto exploration =
        best_candidate(candidates_, [](const auto& c) { return c.entropyUtility(); });
    // Select the best candidate.
    best_idx_ = combined.first;
    // Add an invalid candidate to return if no best candidate was found
    if (best_idx_ == SIZE_MAX) {
        best_idx_ = candidates_.size();
        candidates_.emplace_back(*map, objects, sensor, T_BC);
        return;
    }
    // Compare the best candidate with the best candidate without taking objects into account.
    exploration_dominant_ = (best_idx_ == exploration.first);
    // Compute the yaw angles at each path vertex of the best candidate
    candidates_[best_idx_].computeIntermediateYaw(T_MB_history);
}



size_t SinglePathExplorationPlanner::bestViewIndex() const
{
    return best_idx_;
}



Path SinglePathExplorationPlanner::bestPath() const
{
    return candidates_[best_idx_].path();
}



float SinglePathExplorationPlanner::bestUtility() const
{
    return candidates_[best_idx_].utility();
}



std::vector<CandidateView> SinglePathExplorationPlanner::views() const
{
    return candidates_;
}



std::vector<CandidateView> SinglePathExplorationPlanner::rejectedViews() const
{
    return rejected_candidates_;
}



bool SinglePathExplorationPlanner::explorationDominant() const
{
    return exploration_dominant_;
}



Eigen::Vector3f
SinglePathExplorationPlanner::sampleCandidate(const se::Octree<VoxelImpl::VoxelType>& map,
                                              std::deque<se::key_t>& frontiers,
                                              const Objects& /*objects*/,
                                              const int sampling_step,
                                              const Eigen::Vector3f& sampling_min_M,
                                              const Eigen::Vector3f& sampling_max_M)
{
    // TODO take objects into account
    if (frontiers.empty()) {
        return Eigen::Vector3f::Constant(NAN);
    }
    Eigen::Vector3f pos;
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
    const int size = map.depthToSize(keyops::depth(code));
    pos = map.voxelDim()
        * (keyops::decode(code).cast<float>() + Eigen::Vector3f::Constant(size / 2.0f));
    se::math::clamp(pos, sampling_min_M, sampling_max_M);
    return pos;
}



Eigen::Vector3f
SinglePathExplorationPlanner::sampleCandidate(const se::Octree<VoxelImpl::VoxelType>& map,
                                              MortonSamplingTree& sampling_tree,
                                              const Eigen::Vector3f& sampling_min_M,
                                              const Eigen::Vector3f& sampling_max_M)
{
    key_t code;
    if (!sampling_tree.sampleCode(code)) {
        return Eigen::Vector3f::Constant(NAN);
    }
    // Return the coordinates of the sampled volume's centre
    const int size = map.depthToSize(keyops::depth(code));
    Eigen::Vector3f pos = map.voxelDim()
        * (keyops::decode(code).cast<float>() + Eigen::Vector3f::Constant(size / 2.0f));
    math::clamp(pos, sampling_min_M, sampling_max_M);
    return pos;
}



Eigen::Vector3f
SinglePathExplorationPlanner::sampleCandidate(const se::Octree<VoxelImpl::VoxelType>& map,
                                              std::deque<se::key_t>& frontiers,
                                              std::deque<ObjectPtr>& objects,
                                              const Eigen::Vector3f& sampling_min_M,
                                              const Eigen::Vector3f& sampling_max_M,
                                              const float frontier_sampling_prob)
{
    static std::mt19937 generator;
    static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    if (frontiers.empty() && objects.empty()) {
        return Eigen::Vector3f::Constant(NAN);
    }
    else if (objects.empty() || distribution(generator) < frontier_sampling_prob) {
        return sample_random_frontier(map, frontiers, sampling_min_M, sampling_max_M);
    }
    else {
        return sample_random_object(objects, sampling_min_M, sampling_max_M);
    }
}

} // namespace se
