// SPDX-FileCopyrightText: 2019-2020 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2019 Anna Dai
// SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#include "se/candidate_view.hpp"

#include <cassert>
#include <se/image_utils.hpp>
#include <se/utils/math_utils.h>

/** Return the width of the window of a 360-degree image with width w that a sensor with horizontal
 * field of view hfov can view.
 */
int window_width(int w, float hfov)
{
    const float window_percentage = hfov / (2.0f * M_PI_F);
    return window_percentage * w + 0.5f;
}



namespace se {
CandidateView::CandidateView() :
        path_length_(-1.0f),
        path_time_(-1.0f),
        entropy_(-1.0f),
        lod_gain_(-1.0f),
        entropy_image_(1, 1),
        frustum_overlap_image_(1, 1),
        min_scale_image_(1, 1),
        utility_(-1.0f)
{
}



CandidateView::CandidateView(const Eigen::Vector3f& t_MB) : CandidateView::CandidateView()
{
    path_MB_.push_back(Eigen::Matrix4f::Identity());
    path_MB_.back().topRightCorner<3, 1>() = t_MB;
}



CandidateView::CandidateView(const OctreePtr& map,
                             ptp::OccupancyWorld& ptp_map,
                             const std::vector<se::key_t>& /*frontiers*/,
                             const Objects& objects,
                             const SensorImpl& sensor,
                             const Eigen::Matrix4f& T_MB,
                             const Eigen::Matrix4f& T_BC,
                             const PoseVectorHistory& T_MB_history,
                             const CandidateConfig& config) :
        path_length_(-1.0f),
        path_time_(-1.0f),
        entropy_(-1.0f),
        lod_gain_(-1.0f),
        entropy_image_(config.raycast_width, config.raycast_height),
        frustum_overlap_image_(config.raycast_width, 1, 0.0f),
        min_scale_image_(1, 1),
        utility_(-1.0f),
        config_(config)
{
    // Set-up the planner
    const ptp::PlanningParameter planner_config(config_.planner_config);
    ptp::ProbCollisionChecker planner_collision_checker(ptp_map, planner_config);
    ptp::SafeFlightCorridorGenerator planner(ptp_map, planner_collision_checker, planner_config);
    if (config_.planner_config.start_t_MB_.isApprox(config_.planner_config.goal_t_MB_)) {
        // No need to do path planning if start and goal positions are the same
        path_MB_.push_back(T_MB);
        path_MB_.push_back(T_MB);
    }
    else {
        // Plan a path to the goal
        if (planner.planPath(config_.planner_config.start_t_MB_, config_.planner_config.goal_t_MB_)
            == ptp::PlanningResult::Failed) {
            // Could not plan a path. Add the attempted goal point to the path for visualization
            path_MB_.push_back(Eigen::Matrix4f::Identity());
            path_MB_.back().topRightCorner<3, 1>() = config_.planner_config.goal_t_MB_;
            return;
        }
        path_MB_ = convertPath(planner.getPath());
        // The first path vertex should have the same position as the current pose but a unit
        // orientation. Set it to exactly the current pose.
        assert(("The first path position is the current position",
                path_MB_.front().topRightCorner<3, 1>().isApprox(T_MB.topRightCorner<3, 1>())));
        path_MB_.front() = T_MB;
    }
    // Raycast to compute the optimal yaw angle.
    entropyRaycast(*map, sensor, T_BC, T_MB_history);
    path_MB_.back().topLeftCorner<3, 3>() = yawToC_MB(yaw_M_);
    // Get the LoD gain of the objects.
    const SensorImpl raycasting_sensor(sensor, 0.5f);
    lod_gain_ = lod_gain_raycasting(
        objects, sensor, raycasting_sensor, path_MB_.back() * T_BC, min_scale_image_);
    // Compute the utility.
    path_time_ = pathTime(path_MB_, config_.velocity_linear, config_.velocity_angular);
    computeUtility();
}



bool CandidateView::isValid() const
{
    return utility_ != -1.0f;
}



const Path& CandidateView::path() const
{
    return path_MB_;
}



float CandidateView::utility() const
{
    return utility_;
}



std::string CandidateView::utilityStr() const
{
    return utility_str_;
}



const Eigen::Matrix4f& CandidateView::goalT_MB() const
{
    return path_MB_.back();
}



void CandidateView::computeIntermediateYaw(const Octree<VoxelImpl::VoxelType>& map,
                                           const SensorImpl& sensor,
                                           const Eigen::Matrix4f& T_BC,
                                           const PoseVectorHistory& T_MB_history)
{
    // Raycast and optimize yaw at each intermediate path vertex
    for (size_t i = 1; i < path_MB_.size() - 1; i++) {
        const Eigen::Matrix4f T_MC = path_MB_[i] * T_BC;
        Image<float> entropy_image(entropy_image_.width(), entropy_image_.height());
        Image<float> frustum_overlap_image(entropy_image_.width(), 1);
        raycast_entropy(entropy_image, map, sensor, path_MB_[i], T_BC);
        frustum_overlap(frustum_overlap_image, sensor, T_MC, T_BC, T_MB_history);
        const std::pair<float, float> r = optimal_yaw(entropy_image, frustum_overlap_image, sensor);
        path_MB_[i].topLeftCorner<3, 3>() = yawToC_MB(r.first);
    }
    //yawBeforeMoving(path_MB_);
    //yawWhileMoving(path_MB_, config_.velocity_linear, config_.velocity_angular);
    //path_MB_ = getFinalPath(path_MB_, config_.delta_t, config_.velocity_linear, config_.velocity_angular, map.voxelDim());
}



Image<uint32_t> CandidateView::renderEntropy(const SensorImpl& sensor,
                                             const bool visualize_yaw) const
{
    return visualizeEntropy(entropy_image_, sensor, yaw_M_, visualize_yaw);
}



Image<uint32_t> CandidateView::renderCurrentEntropy(const Octree<VoxelImpl::VoxelType>& map,
                                                    const SensorImpl& sensor,
                                                    const Eigen::Matrix4f& T_BC,
                                                    const bool visualize_yaw) const
{
    Eigen::Matrix4f T_MB = Eigen::Matrix4f::Identity();
    T_MB.topRightCorner<3, 1>() = config_.planner_config.goal_t_MB_;
    Image<float> entropy(entropy_image_.width(), entropy_image_.height());
    raycast_entropy(entropy, map, sensor, T_MB, T_BC);
    return visualizeEntropy(entropy, sensor, yaw_M_, visualize_yaw);
}



Image<uint32_t> CandidateView::renderMinScale(const Octree<VoxelImpl::VoxelType>& /*map*/,
                                              const SensorImpl& /*sensor*/,
                                              const Eigen::Matrix4f& /*T_BC*/) const
{
    constexpr int max_scale_p1 = VoxelImpl::VoxelBlockType::max_scale + 1;
    Image<uint32_t> min_scale_render(min_scale_image_.width(), min_scale_image_.height());
#pragma omp parallel for
    for (size_t i = 0; i < min_scale_render.size(); ++i) {
        // Scale the minimum scale to the interval [0,255].
        const uint8_t s = UINT8_MAX * static_cast<float>(min_scale_image_[i] + 1) / max_scale_p1;
        min_scale_render[i] = se::pack_rgba(s, s, s, 0xFF);
    }
    return min_scale_render;
}



Image<uint32_t> CandidateView::renderDepth(const Octree<VoxelImpl::VoxelType>& map,
                                           const SensorImpl& sensor,
                                           const Eigen::Matrix4f& T_BC,
                                           const bool visualize_yaw) const
{
    const Eigen::Vector2i res(entropy_image_.width(), entropy_image_.height());
    Eigen::Matrix4f T_MB = Eigen::Matrix4f::Identity();
    T_MB.topRightCorner<3, 1>() = config_.planner_config.goal_t_MB_;
    // Raycast to get the depth
    Image<float> depth(res.x(), res.y());
    raycast_depth(depth, map, sensor, T_MB, T_BC);
    // Render to a colour image
    Image<uint32_t> depth_render(res.x(), res.y());
    se::depth_to_rgba(depth_render.data(), depth.data(), res, sensor.near_plane, sensor.far_plane);
    // Visualize the optimal yaw
    if (visualize_yaw) {
        overlay_yaw(depth_render, yaw_M_, sensor);
    }
    return depth_render;
}



Path CandidateView::addIntermediateTranslation(const Eigen::Matrix4f& segment_start_M,
                                               const Eigen::Matrix4f& segment_end_M,
                                               float delta_t,
                                               float velocity_linear,
                                               float resolution)
{
    Path path;
    path.emplace_back(segment_start_M);
    const Eigen::Vector3f diff = math::position_error(segment_start_M, segment_end_M);
    const Eigen::Vector3f dir = diff.normalized();
    const Eigen::Vector3f t_start = segment_start_M.topRightCorner<3, 1>();
    const float dist = diff.norm();
    // The dist_increment makes no sense.
    // ((m/s * s) / m/voxel) / m = (m / m/voxel) / m = (m * voxel/m) / m = voxel / m
    const float dist_increment = (velocity_linear * delta_t / resolution) / dist;
    for (float t = 0.0f; t <= 1.0f; t += dist_increment) {
        const Eigen::Vector3f t_intermediate = t_start + dir * t * dist;
        Eigen::Matrix4f intermediate_pose = Eigen::Matrix4f::Identity();
        intermediate_pose.topRightCorner<3, 1>() = t_intermediate;
        path.emplace_back(intermediate_pose);
    }
    path.emplace_back(segment_end_M);
    return path;
}



Path CandidateView::addIntermediateYaw(const Eigen::Matrix4f& segment_start_M,
                                       const Eigen::Matrix4f& segment_end_M,
                                       float delta_t,
                                       float velocity_angular)
{
    Path path;
    const float yaw_diff = math::yaw_error(segment_start_M, segment_end_M);
    const float yaw_increment = (velocity_angular * delta_t) / std::abs(yaw_diff);
    for (float t = 0.0f; t <= 1.0f; t += yaw_increment) {
        // Interpolate the quaternion.
        Eigen::Quaternionf q_start(segment_start_M.topLeftCorner<3, 3>());
        Eigen::Quaternionf q_end(segment_end_M.topLeftCorner<3, 3>());
        Eigen::Quaternionf q_intermediate = q_start.slerp(t, q_end).normalized();
        // Create the intermediate pose.
        Eigen::Matrix4f intermediate_pose = segment_end_M;
        intermediate_pose.topLeftCorner<3, 3>() = q_intermediate.toRotationMatrix();
        path.emplace_back(intermediate_pose);
    }
    path.emplace_back(segment_end_M);
    return path;
}



Path CandidateView::fuseIntermediatePaths(const Path& intermediate_translation,
                                          const Path& intermediate_yaw)
{
    Path path(std::max(intermediate_translation.size(), intermediate_yaw.size()));
    if (intermediate_yaw.size() >= intermediate_translation.size()) {
        for (size_t i = 0; i < intermediate_yaw.size(); i++) {
            path[i] = intermediate_yaw[i];
            if (i < intermediate_translation.size()) {
                path[i].topRightCorner<3, 1>() = intermediate_translation[i].topRightCorner<3, 1>();
            }
            else {
                path[i].topRightCorner<3, 1>() =
                    intermediate_translation.back().topRightCorner<3, 1>();
            }
        }
    }
    else {
        for (size_t i = 0; i < intermediate_translation.size(); i++) {
            path[i] = intermediate_translation[i];
            if (i < intermediate_yaw.size()) {
                path[i].topLeftCorner<3, 3>() = intermediate_yaw[i].topLeftCorner<3, 3>();
            }
            else {
                path[i].topLeftCorner<3, 3>() = intermediate_yaw.back().topLeftCorner<3, 3>();
            }
        }
    }
    return path;
}



Path CandidateView::getFinalPath(const Path& path_M,
                                 float delta_t,
                                 float velocity_linear,
                                 float velocity_angular,
                                 float resolution)
{
    Path final_path_M;
    for (size_t i = 1; i < path_M.size(); i++) {
        Path tran_path;
        if (math::position_error(path_M[i - 1], path_M[i]).norm() > velocity_linear * delta_t) {
            tran_path = addIntermediateTranslation(
                path_M[i - 1], path_M[i], delta_t, velocity_linear, resolution);
        }
        else {
            tran_path.emplace_back(path_M[i - 1]);
            tran_path.emplace_back(path_M[i]);
        }
        const Path yaw_path =
            addIntermediateYaw(path_M[i - 1], path_M[i], delta_t, velocity_angular);

        const Path fused_path = fuseIntermediatePaths(tran_path, yaw_path);
        final_path_M.insert(final_path_M.end(), fused_path.begin(), fused_path.end());
    }
    return final_path_M;
}



void CandidateView::entropyRaycast(const Octree<VoxelImpl::VoxelType>& map,
                                   const SensorImpl& sensor,
                                   const Eigen::Matrix4f& T_BC,
                                   const PoseVectorHistory& T_MB_history)
{
    const Eigen::Matrix4f T_MC = path_MB_.back() * T_BC;
    // Raycast at the last path vertex
    raycast_entropy(entropy_image_, map, sensor, path_MB_.back(), T_BC);
    if (config_.use_pose_history) {
        frustum_overlap(frustum_overlap_image_, sensor, T_MC, T_BC, T_MB_history);
    }
    const std::pair<float, float> r = optimal_yaw(entropy_image_, frustum_overlap_image_, sensor);
    yaw_M_ = r.first;
    entropy_ = r.second;
}



void CandidateView::computeUtility()
{
    utility_ =
        (config_.exploration_weight * entropy_ + (1.0f - config_.exploration_weight) * lod_gain_)
        / path_time_;
    constexpr char format[] = "(%4.2f * %6.4f + %4.2f * %6.4f) / %-7.3f = %f";
    // Resize the string with the appropriate number of characters to fit the output of snprintf().
    const int s = snprintf(nullptr,
                           0,
                           format,
                           config_.exploration_weight,
                           entropy_,
                           (1.0f - config_.exploration_weight),
                           lod_gain_,
                           path_time_,
                           utility_);
    utility_str_ = std::string(s + 1, '\0');
    snprintf(&utility_str_[0],
             s + 1,
             format,
             config_.exploration_weight,
             entropy_,
             (1.0f - config_.exploration_weight),
             lod_gain_,
             path_time_,
             utility_);
}



Eigen::Matrix3f CandidateView::yawToC_MB(const float yaw_M)
{
    Eigen::Matrix3f C_MB = Eigen::Matrix3f::Zero();
    C_MB(0, 0) = cos(yaw_M);
    C_MB(0, 1) = -sin(yaw_M);
    C_MB(1, 0) = sin(yaw_M);
    C_MB(1, 1) = cos(yaw_M);
    C_MB(2, 2) = 1.0f;
    return C_MB;
}



Path CandidateView::convertPath(const ptp::Path<ptp::kDim>::Ptr ptp_path)
{
    Path path(ptp_path->states.size(), Eigen::Matrix4f::Identity());
    for (size_t i = 0; i < ptp_path->states.size(); ++i) {
        const Eigen::Vector3f& t_MB = ptp_path->states[i].segment_end;
        path[i].topRightCorner<3, 1>() = t_MB;
    }
    return path;
}



void CandidateView::removeSmallMovements(Path& path, const float radius)
{
    if (!path.empty()) {
        const Eigen::Vector3f t_MB_0 = path[0].topRightCorner<3, 1>();
        for (size_t i = 1; i < path.size(); ++i) {
            const Eigen::Vector3f t_MB = path[i].topRightCorner<3, 1>();
            if ((t_MB_0 - t_MB).norm() <= radius) {
                path[i].topRightCorner<3, 1>() = t_MB_0;
            }
        }
    }
}



void CandidateView::yawBeforeMoving(Path& path)
{
    Path new_path;
    for (size_t i = 0; i < path.size() - 1; ++i) {
        // Add the current waypoint.
        new_path.push_back(path[i]);
        // Add an waypoint with the orientation of the next and the position of the current waypoint.
        new_path.push_back(path[i + 1]);
        new_path.back().topRightCorner<3, 1>() = path[i].topRightCorner<3, 1>();
    }
    new_path.push_back(path.back());
    path = new_path;
}



void CandidateView::yawWhileMoving(Path& path, float velocity_linear, float velocity_angular)
{
    constexpr float yaw_step = se::math::deg_to_rad * 45;
    Path new_path;
    for (size_t i = 0; i < path.size() - 1; ++i) {
        new_path.push_back(path[i]);
        const Eigen::Vector3f t_prev = path[i].topRightCorner<3, 1>();
        const Eigen::Vector3f t_next = path[i + 1].topRightCorner<3, 1>();
        const Eigen::Vector3f t_diff = t_next - t_prev;
        const Eigen::Quaternionf q_prev = Eigen::Quaternionf(path[i].topLeftCorner<3, 3>());
        const Eigen::Quaternionf q_next = Eigen::Quaternionf(path[i + 1].topLeftCorner<3, 3>());
        const float dist = (t_next - t_prev).norm();
        const float yaw_diff = math::yaw_error(path[i], path[i + 1]);
        const float time_tran = dist / velocity_linear;
        const float time_rot = yaw_diff / velocity_angular;
        const float time = std::max(time_tran, time_rot);
        const float dt = yaw_step / velocity_angular;
        for (float t = 0.0f; t < time; t += dt) {
            const float tt = t / time;
            new_path.emplace_back(Eigen::Matrix4f::Identity());
            new_path.back().topRightCorner<3, 1>() = t_prev + tt * t_diff;
            new_path.back().topLeftCorner<3, 3>() = q_prev.slerp(tt, q_next).toRotationMatrix();
        }
    }
    path = new_path;
}



float CandidateView::pathTime(const Path& path, float velocity_linear, float velocity_angular)
{
    // Compute the translation time
    float t_tran = 0.0f;
    for (size_t i = 0; i < path.size() - 1; ++i) {
        const Eigen::Matrix4f& start = path[i];
        const Eigen::Matrix4f& end = path[i + 1];
        const Eigen::Vector3f translation =
            end.topRightCorner<3, 1>() - start.topRightCorner<3, 1>();
        t_tran += translation.norm() / velocity_linear;
    }
    // Compute the rotation time
    //float t_rot = 0.0f;
    //for (size_t i = 0; i < path.size() - 1; ++i) {
    //  const float yaw_diff = fabsf(math::yaw_error(path[i], path[i + 1]));
    //  t_rot += yaw_diff / velocity_angular;
    //}
    const float yaw_diff = fabsf(math::yaw_error(path.front(), path.back()));
    const float t_rot = yaw_diff / velocity_angular;
    const float t = std::max(t_tran, t_rot);
    return (std::fabs(t) > 10.0f * FLT_EPSILON) ? t : NAN;
}



Image<uint32_t> CandidateView::visualizeEntropy(const Image<float>& entropy,
                                                const SensorImpl& sensor,
                                                const float yaw_M,
                                                const bool visualize_yaw)
{
    Image<uint32_t> entropy_render(entropy.width(), entropy.height());
    for (size_t i = 0; i < entropy.size(); ++i) {
        // Scale and clamp the entropy for visualization since its values are typically too low.
        const uint8_t e = se::math::clamp(
            UINT8_MAX * (6.0f * entropy[i]) + 0.5f, 0.0f, static_cast<float>(UINT8_MAX));
        entropy_render[i] = se::pack_rgba(e, e, e, 0xFF);
    }
    if (visualize_yaw) {
        overlay_yaw(entropy_render, yaw_M, sensor);
    }
    return entropy_render;
}
} // namespace se
