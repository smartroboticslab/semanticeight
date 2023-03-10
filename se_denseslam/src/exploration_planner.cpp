// SPDX-FileCopyrightText: 2021 Smart Robotics Lab
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#include "se/exploration_planner.hpp"

#include "se/bounding_volume.hpp"
#include "se/exploration_utils.hpp"

namespace se {
ExplorationPlanner::ExplorationPlanner(const DenseSLAMSystem& pipeline,
                                       const SensorImpl& sensor,
                                       const se::Configuration& config) :
        config_(config, pipeline.T_MW()),
        map_(pipeline.getMap()),
        sampling_aabb_edges_M_(se::AABB(config_.candidate_config.planner_config.sampling_min_M_,
                                        config_.candidate_config.planner_config.sampling_max_M_)
                                   .edges()),
        T_MW_(pipeline.T_MW()),
        T_WM_(pipeline.T_WM()),
        T_BC_(config.T_BC),
        T_CB_(se::math::to_inverse_transformation(config.T_BC)),
        sensor_(sensor),
        T_MB_grid_history_(Eigen::Vector3f::Constant(map_->dim()),
                           config.isExperiment()
                               ? Eigen::Vector4f(2.0f, 2.0f, 2.0f, se::math::deg_to_rad(60.0f))
                               : Eigen::Vector4f(0.5f, 0.5f, 0.5f, se::math::deg_to_rad(30.0f))),
        T_MB_mask_history_(Eigen::Vector2i(config.raycast_width, config.raycast_height),
                           sensor,
                           config.T_BC,
                           Eigen::Vector3f::Constant(map_->dim()),
                           Eigen::Vector3f::Constant(0.5f)),
        candidate_views_(
            {CandidateView(*pipeline.getMap(), pipeline.getObjectMaps(), sensor, config.T_BC)}),
        goal_view_idx_(0),
        exploration_dominant_(true)
{
    ompl::msg::setLogLevel(ompl::msg::LOG_ERROR);
}



void ExplorationPlanner::recordT_WB(const Eigen::Matrix4f& T_WB, const se::Image<float>& depth)
{
    const Eigen::Matrix4f T_MB = T_MW_ * T_WB;
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    T_MB_grid_history_.record(T_MB);
    T_MB_mask_history_.record(T_MB, depth);
    T_MB_history_.record(T_MB);
}



Eigen::Matrix4f ExplorationPlanner::getT_WB() const
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    return T_WM_ * T_MB_history_.poses.back();
}



Path ExplorationPlanner::getT_WBHistory() const
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    const Path& T_MB_history = T_MB_history_.poses;
    Path T_WB_history(T_MB_history.size());
    std::transform(T_MB_history.begin(),
                   T_MB_history.end(),
                   T_WB_history.begin(),
                   [&](const auto& T_MB) { return T_WM_ * T_MB; });
    return T_WB_history;
}



bool ExplorationPlanner::needsNewGoal() const
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    return goal_path_T_MB_.empty();
}



math::PoseError ExplorationPlanner::goalCandidateError(const Eigen::Matrix4f& T_WB) const
{
    const Eigen::Matrix4f T_MB = T_MW_ * T_WB;
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    return math::pose_error(T_MB, goalView().goalT_MB());
}



bool ExplorationPlanner::inGoalCandidateThreshold(const Eigen::Matrix4f& T_WB) const
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    return withinThreshold(goalCandidateError(T_WB));
}



bool ExplorationPlanner::goalReached()
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    // If there is no goal, we have reached the goal.
    if (goal_path_T_MB_.empty()) {
        return true;
    }
    const Eigen::Matrix4f& current_T_MB = T_MB_history_.poses.back();
    const Eigen::Matrix4f& goal_T_MB = goal_path_T_MB_.front();
    const bool reached = withinThreshold(math::pose_error(goal_T_MB, current_T_MB));
    if (reached) {
        // Remove the reached pose from the goal path.
        goal_path_T_MB_.pop();
    }
    return reached;
}



bool ExplorationPlanner::goalT_WB(Eigen::Matrix4f& T_WB) const
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (goal_path_T_MB_.empty()) {
        return false;
    }
    else {
        T_WB = T_WM_ * goal_path_T_MB_.front();
        return true;
    }
}



void ExplorationPlanner::popGoalT_WB()
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    goal_path_T_MB_.pop();
}



Path ExplorationPlanner::computeNextPath_WB(const std::set<key_t>& frontiers,
                                            const Objects& objects,
                                            const Eigen::Matrix4f& start_T_WB)
{
    const std::vector<key_t> frontier_vec(frontiers.begin(), frontiers.end());
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    SinglePathExplorationPlanner planner(map_,
                                         frontier_vec,
                                         objects,
                                         sensor_,
                                         T_MW_ * start_T_WB,
                                         T_BC_,
                                         T_MB_grid_history_,
                                         T_MB_mask_history_,
                                         T_MB_history_,
                                         config_);
    candidate_views_ = planner.views();
    rejected_candidate_views_ = planner.rejectedViews();
    goal_view_idx_ = planner.bestViewIndex();
    const Path& goal_path_M = planner.bestPath();
    // Add it to the goal path queue.
    for (const auto& T_MB : goal_path_M) {
        goal_path_T_MB_.push(T_MB);
    }
    // Convert the goal path from the map to the world frame.
    Path goal_path_W(goal_path_M.size());
    std::transform(goal_path_M.begin(),
                   goal_path_M.end(),
                   goal_path_W.begin(),
                   [&](const Eigen::Matrix4f& T_MB) { return T_WM_ * T_MB; });
    return goal_path_W;
}



const std::vector<CandidateView>& ExplorationPlanner::candidateViews() const
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    return candidate_views_;
}



const std::vector<CandidateView>& ExplorationPlanner::rejectedCandidateViews() const
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    return rejected_candidate_views_;
}



const CandidateView& ExplorationPlanner::goalView() const
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    return candidate_views_[goalViewIndex()];
}



size_t ExplorationPlanner::goalViewIndex() const
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (goal_view_idx_ < candidate_views_.size()) {
        return goal_view_idx_;
    }
    else {
        throw std::runtime_error("goal_view_idx_ (" + std::to_string(goal_view_idx_)
                                 + ") >= candidate_views_.size() ("
                                 + std::to_string(candidate_views_.size()) + ")");
    }
}



void ExplorationPlanner::renderCurrentEntropyDepth(Image<uint32_t>& entropy,
                                                   Image<uint32_t>& depth,
                                                   const bool visualize_yaw)
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    goalView().renderCurrentEntropyDepth(entropy, depth, visualize_yaw);
}



const PoseGridHistory& ExplorationPlanner::getPoseGridHistory() const
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    return T_MB_grid_history_;
}



const PoseMaskHistory& ExplorationPlanner::getPoseMaskHistory() const
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    return T_MB_mask_history_;
}



const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>&
ExplorationPlanner::samplingAABBEdgesM() const
{
    // Only uses const members, no need to lock.
    return sampling_aabb_edges_M_;
}



int ExplorationPlanner::writePathPLY(const std::string& filename, const Eigen::Matrix4f& T_FW) const
{
    const Eigen::Matrix4f T_FM = T_FW * T_WM_;
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    Path history_W(T_MB_history_.poses.size());
    std::transform(T_MB_history_.poses.begin(),
                   T_MB_history_.poses.end(),
                   history_W.begin(),
                   [&](const Eigen::Matrix4f& T_MB) { return T_FM * T_MB; });
    return write_path_ply(filename, history_W);
}



int ExplorationPlanner::writePathTSV(const std::string& filename, const Eigen::Matrix4f& T_FW) const
{
    const Eigen::Matrix4f T_FM = T_FW * T_WM_;
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    Path history_W(T_MB_history_.poses.size());
    std::transform(T_MB_history_.poses.begin(),
                   T_MB_history_.poses.end(),
                   history_W.begin(),
                   [&](const Eigen::Matrix4f& T_MB) { return T_FM * T_MB; });
    return write_path_tsv(filename, history_W);
}



bool ExplorationPlanner::explorationDominant() const
{
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    return exploration_dominant_;
}



bool ExplorationPlanner::withinThreshold(const math::PoseError& error) const
{
    // Only uses const members, no need to lock.
    if (error.pos.head<2>().norm() > config_.candidate_config.goal_xy_threshold) {
        return false;
    }
    if (fabsf(error.pos.z()) > config_.candidate_config.goal_z_threshold) {
        return false;
    }
    if (fabsf(error.roll) > config_.candidate_config.goal_roll_pitch_threshold) {
        return false;
    }
    if (fabsf(error.pitch) > config_.candidate_config.goal_roll_pitch_threshold) {
        return false;
    }
    if (fabsf(error.yaw) > config_.candidate_config.goal_yaw_threshold) {
        return false;
    }
    return true;
}

} // namespace se
