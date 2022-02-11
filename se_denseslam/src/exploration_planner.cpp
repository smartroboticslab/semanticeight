// SPDX-FileCopyrightText: 2021 Smart Robotics Lab
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#include "se/exploration_planner.hpp"

#include "se/bounding_volume.hpp"
#include "se/exploration_utils.hpp"

namespace se {
ExplorationPlanner::ExplorationPlanner(const DenseSLAMSystem& pipeline,
                                       const se::Configuration& config) :
        config_({config.num_candidates,
                 (pipeline.T_MW() * config.sampling_min_W.homogeneous()).head<3>(),
                 (pipeline.T_MW() * config.sampling_max_W.homogeneous()).head<3>(),
                 config.goal_xy_threshold,
                 config.goal_z_threshold,
                 config.goal_roll_pitch_threshold,
                 config.goal_yaw_threshold,
                 {config.exploration_weight,
                  config.use_pose_history,
                  config.raycast_width,
                  config.raycast_height,
                  config.delta_t,
                  config.linear_velocity,
                  config.angular_velocity,
                  {"",
                   Eigen::Vector3f::Zero(),
                   Eigen::Vector3f::Zero(),
                   config.robot_radius,
                   config.safety_radius,
                   config.min_control_point_radius,
                   config.skeleton_sample_precision,
                   config.solving_time}}}),
        map_(pipeline.getMap()),
        sampling_aabb_edges_M_(se::AABB(config_.sampling_min_M, config_.sampling_max_M).edges()),
        T_MW_(pipeline.T_MW()),
        T_WM_(pipeline.T_WM()),
        T_BC_(config.T_BC),
        T_CB_(se::math::to_inverse_transformation(T_BC_)),
        T_MB_grid_history_(Eigen::Vector3f::Constant(map_->dim()))
{
    ompl::msg::setLogLevel(ompl::msg::LOG_ERROR);
}



void ExplorationPlanner::setT_WB(const Eigen::Matrix4f& T_WB)
{
    const Eigen::Matrix4f T_MB = T_MW_ * T_WB;
    T_MB_grid_history_.record(T_MB);
    T_MB_history_.record(T_MB);
}



void ExplorationPlanner::setPlanningT_WB(const Eigen::Matrix4f& T_WB)
{
    planning_T_MB_ = T_MW_ * T_WB;
}



Eigen::Matrix4f ExplorationPlanner::getT_WB() const
{
    return T_WM_ * T_MB_history_.poses.back();
}



Eigen::Matrix4f ExplorationPlanner::getPlanningT_WB() const
{
    return T_WM_ * planning_T_MB_;
}



Path ExplorationPlanner::getT_WBHistory() const
{
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
    if (pos_error.head<2>().norm() > config_.goal_xy_threshold) {
        return false;
    }
    if (pos_error.tail<1>().norm() > config_.goal_z_threshold) {
        return false;
    }
    if (fabsf(math::yaw_error(goal_T_MB, current_T_MB)) > config_.goal_yaw_threshold) {
        return false;
    }
    if (fabsf(math::pitch_error(goal_T_MB, current_T_MB)) > config_.goal_roll_pitch_threshold) {
        return false;
    }
    if (fabsf(math::roll_error(goal_T_MB, current_T_MB)) > config_.goal_roll_pitch_threshold) {
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
    }
    else {
        T_WB = T_WM_ * goal_path_T_MB_.front();
        return true;
    }
}



void ExplorationPlanner::popGoalT_WB()
{
    goal_path_T_MB_.pop();
}



Path ExplorationPlanner::computeNextPath_WB(const std::set<key_t>& frontiers,
                                            const Objects& objects,
                                            const SensorImpl& sensor)
{
    const std::vector<key_t> frontier_vec(frontiers.begin(), frontiers.end());
    SinglePathExplorationPlanner planner(map_,
                                         frontier_vec,
                                         objects,
                                         sensor,
                                         planning_T_MB_,
                                         T_BC_,
                                         T_MB_grid_history_,
                                         T_MB_history_,
                                         config_);
    candidate_views_ = planner.views();
    rejected_candidate_views_ = planner.rejectedViews();
    goal_view_ = planner.bestView();
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



size_t ExplorationPlanner::goalViewIndex() const
{
    return goal_view_idx_;
}



void ExplorationPlanner::renderCurrentEntropyDepth(Image<uint32_t>& entropy,
                                                   Image<uint32_t>& depth,
                                                   const SensorImpl& sensor,
                                                   const bool visualize_yaw)
{
    goal_view_.renderCurrentEntropyDepth(entropy, depth, *map_, sensor, T_BC_, visualize_yaw);
}



Image<uint32_t> ExplorationPlanner::renderMinScale(const SensorImpl& sensor)
{
    return goal_view_.renderMinScale(*map_, sensor, T_BC_);
}



const PoseGridHistory& ExplorationPlanner::getPoseGridHistory() const
{
    return T_MB_grid_history_;
}



const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>&
ExplorationPlanner::samplingAABBEdgesM() const
{
    return sampling_aabb_edges_M_;
}



int ExplorationPlanner::writePathPLY(const std::string& filename, const Eigen::Matrix4f& T_FW) const
{
    const Eigen::Matrix4f T_FM = T_FW * T_WM_;
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
    Path history_W(T_MB_history_.poses.size());
    std::transform(T_MB_history_.poses.begin(),
                   T_MB_history_.poses.end(),
                   history_W.begin(),
                   [&](const Eigen::Matrix4f& T_MB) { return T_FM * T_MB; });
    return write_path_tsv(filename, history_W);
}
} // namespace se
