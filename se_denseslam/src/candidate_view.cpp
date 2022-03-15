// SPDX-FileCopyrightText: 2019-2020 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2019 Anna Dai
// SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#include "se/candidate_view.hpp"

#include <cassert>
#include <fstream>
#include <iomanip>
#include <se/image_utils.hpp>
#include <se/utils/math_utils.h>

namespace se {
CandidateView::CandidateView(const se::Octree<VoxelImpl::VoxelType>& map,
                             const SensorImpl& sensor,
                             const Eigen::Matrix4f& T_BC) :
        desired_t_MB_(NAN, NAN, NAN),
        yaw_M_(NAN),
        window_idx_(-1),
        window_width_(-1),
        path_time_(-1.0f),
        entropy_(-1.0f),
        lod_gain_(-1.0f),
        entropy_image_(1, 1),
        entropy_hits_M_(1, 1),
        frustum_overlap_mask_(1, 1),
        min_scale_image_(1, 1),
        sensor_(sensor),
        map_(map),
        T_BC_(T_BC),
        utility_(-1.0f),
        exploration_utility_(-1.0f),
        object_utility_(-1.0f)
{
}



CandidateView::CandidateView(const se::Octree<VoxelImpl::VoxelType>& map,
                             ptp::SafeFlightCorridorGenerator& planner,
                             const std::vector<se::key_t>& /*frontiers*/,
                             const Objects& objects,
                             const SensorImpl& sensor,
                             const Eigen::Matrix4f& T_MB,
                             const Eigen::Matrix4f& T_BC,
                             const PoseHistory* T_MB_history,
                             const CandidateConfig& config) :
        desired_t_MB_(config.planner_config.goal_t_MB_),
        yaw_M_(NAN),
        window_idx_(-1),
        window_width_(-1),
        path_time_(-1.0f),
        entropy_(-1.0f),
        lod_gain_(-1.0f),
        // TODO create if needed?
        entropy_image_(config.raycast_width, config.raycast_height),
        entropy_hits_M_(config.raycast_width, config.raycast_height),
        frustum_overlap_mask_(config.raycast_width, config.raycast_height, 0.0f),
        min_scale_image_(1, 1),
        sensor_(sensor),
        map_(map),
        T_BC_(T_BC),
        utility_(-1.0f),
        exploration_utility_(-1.0f),
        object_utility_(-1.0f),
        config_(config)
{
    // Reject based on the pose history.
    if (config_.use_pose_history && T_MB_history->rejectPosition(desired_t_MB_, sensor)) {
        path_MB_.push_back(Eigen::Matrix4f::Identity());
        path_MB_.back().topRightCorner<3, 1>() = desired_t_MB_;
        status_ = "Rejected from pose history";
        return;
    }
    if (config_.planner_config.start_t_MB_.isApprox(config_.planner_config.goal_t_MB_)) {
        // No need to do path planning if start and goal positions are the same
        path_MB_.push_back(T_MB);
        path_MB_.push_back(T_MB);
        status_ = "Same start/goal positions";
    }
    else {
        // Plan a path to the goal
        const auto status =
            planner.planPath(config_.planner_config.start_t_MB_, config_.planner_config.goal_t_MB_);
        status_ = ptp::to_string(status);
        if (status != ptp::PlanningResult::Success && status != ptp::PlanningResult::Partial) {
            // Could not plan a path. Add the attempted goal point to the path for visualization
            path_MB_.push_back(Eigen::Matrix4f::Identity());
            path_MB_.back().topRightCorner<3, 1>() = config_.planner_config.goal_t_MB_;
            return;
        }
        path_MB_ = convertPath(planner.getPath());
        // The first path vertex should have the same position as the current pose but a unit
        // orientation. Set it to exactly the current pose.
        assert((path_MB_.front().topRightCorner<3, 1>().isApprox(T_MB.topRightCorner<3, 1>())
                && "The first path position is the current position"));
        path_MB_.front() = T_MB;
    }
    // Raycast to compute the optimal yaw angle.
    entropyRaycast(T_MB_history);
    path_MB_.back().topLeftCorner<3, 3>() = yawToC_MB(yaw_M_);
    zeroRollPitch(path_MB_);
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



float CandidateView::explorationUtility() const
{
    return exploration_utility_;
}



float CandidateView::objectUtility() const
{
    return object_utility_;
}



std::string CandidateView::utilityStr() const
{
    return utility_str_;
}



const Eigen::Matrix4f& CandidateView::goalT_MB() const
{
    if (path_MB_.empty()) {
        static Eigen::Matrix4f invalid_T_MB = Eigen::Matrix4f::Identity();
        invalid_T_MB.topRightCorner<3, 1>() = Eigen::Vector3f::Constant(NAN);
        return invalid_T_MB;
    }
    else {
        return path_MB_.back();
    }
}



void CandidateView::computeIntermediateYaw(const PoseHistory* T_MB_history)
{
    // Raycast and optimize yaw at each intermediate path vertex
    for (size_t i = 1; i < path_MB_.size() - 1; i++) {
        Image<float> entropy_image(entropy_image_.width(), entropy_image_.height());
        Image<Eigen::Vector3f> entropy_hits(entropy_hits_M_.width(), entropy_hits_M_.height());
        raycast_entropy(entropy_image, entropy_hits, map_, sensor_, path_MB_[i], T_BC_);
        if (config_.use_pose_history) {
            Image<uint8_t> frustum_overlap_mask(
                entropy_image_.width(), entropy_image_.height(), 0u);
            const Eigen::Matrix4f T_MC = path_MB_[i] * T_BC_;
            T_MB_history->frustumOverlap(frustum_overlap_mask, sensor_, T_MC, T_BC_);
            entropy_image = mask_entropy_image(entropy_image, frustum_overlap_mask);
        }
        const auto r = optimal_yaw(entropy_image, entropy_hits, sensor_, path_MB_[i], T_BC_);
        path_MB_[i].topLeftCorner<3, 3>() = yawToC_MB(std::get<0>(r));
    }
    //yawBeforeMoving(path_MB_);
    //yawWhileMoving(path_MB_, config_.velocity_linear, config_.velocity_angular);
    //path_MB_ = getFinalPath(path_MB_, config_.delta_t, config_.velocity_linear, config_.velocity_angular, map_.voxelDim());
}



Image<uint32_t> CandidateView::renderEntropy(const bool visualize_yaw) const
{
    if (isValid()) {
        return visualize_entropy(entropy_image_, window_idx_, window_width_, visualize_yaw);
    }
    else {
        return Image<uint32_t>(entropy_image_.width(), entropy_image_.height(), 0xFF0000FF);
    }
}



Image<uint32_t> CandidateView::renderDepth(const bool visualize_yaw) const
{
    if (isValid()) {
        return visualize_depth(
            entropy_hits_M_, sensor_, goalT_MB(), window_idx_, window_width_, visualize_yaw);
    }
    else {
        return Image<uint32_t>(entropy_hits_M_.width(), entropy_hits_M_.height(), 0xFF000000);
    }
}



Image<uint32_t> CandidateView::renderMinScale() const
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



void CandidateView::renderCurrentEntropyDepth(Image<uint32_t>& entropy,
                                              Image<uint32_t>& depth,
                                              const bool visualize_yaw) const
{
    Eigen::Matrix4f T_MB = Eigen::Matrix4f::Identity();
    if (path_MB_.empty()) {
        T_MB.topRightCorner<3, 1>() = config_.planner_config.goal_t_MB_;
    }
    else {
        T_MB.topRightCorner<3, 1>() = path_MB_.back().topRightCorner<3, 1>();
    }
    Image<float> raw_entropy(entropy_image_.width(), entropy_image_.height());
    Image<Eigen::Vector3f> entropy_hits(entropy_image_.width(), entropy_image_.height());
    raycast_entropy(raw_entropy, entropy_hits, map_, sensor_, T_MB, T_BC_);
    entropy =
        visualize_entropy(raw_entropy, window_idx_, window_width_, visualize_yaw && isValid());
    depth = visualize_depth(
        entropy_hits, sensor_, T_MB, window_idx_, window_width_, visualize_yaw && isValid());
}



Image<Eigen::Vector3f> CandidateView::rays() const
{
    return entropy_hits_M_;
}



bool CandidateView::writeEntropyData(const std::string& filename) const
{
    std::ofstream f(filename);
    if (!f.good()) {
        return false;
    }
    f << std::fixed;

    f << std::setprecision(6);
    f << "Entropy\n";
    f << entropy_image_.width() << " " << entropy_image_.height() << "\n";
    for (int y = 0; y < entropy_image_.height(); y++) {
        for (int x = 0; x < entropy_image_.width(); x++) {
            f << std::setw(20) << entropy_image_(x, y);
            if (x != entropy_image_.width() - 1) {
                f << " ";
            }
        }
        f << "\n";
    }

    f << "\n";
    f << std::setprecision(6);
    f << "Frustum overlap\n";
    f << frustum_overlap_mask_.width() << " " << frustum_overlap_mask_.height() << "\n";
    for (int y = 0; y < frustum_overlap_mask_.height(); y++) {
        for (int x = 0; x < frustum_overlap_mask_.width(); x++) {
            f << std::setw(20) << frustum_overlap_mask_(x, y);
            if (x != frustum_overlap_mask_.width() - 1) {
                f << " ";
            }
        }
        f << "\n";
    }

    f << "\n";
    f << std::setprecision(3);
    f << "Entropy hits M\n";
    f << entropy_hits_M_.width() << " " << entropy_hits_M_.height() << "\n";
    for (int y = 0; y < entropy_hits_M_.height(); y++) {
        for (int x = 0; x < entropy_hits_M_.width(); x++) {
            f << std::setw(6) << entropy_hits_M_(x, y).x() << " " << std::setw(6)
              << entropy_hits_M_(x, y).y() << " " << std::setw(6) << entropy_hits_M_(x, y).z();
            if (x != entropy_hits_M_.width() - 1) {
                f << "   ";
            }
        }
        f << "\n";
    }

    f << "\n";
    f << std::setprecision(6);
    f << "Entropy: " << entropy_ << "\n";
    f << "Optimal yaw M: " << yaw_M_ << " rad   " << se::math::rad_to_deg(yaw_M_) << " degrees \n";
    f << "Window index: " << window_idx_ << " px\n";
    f << "Window width: " << window_width_ << " px\n";
    f << "Horizontal FoV: " << sensor_.horizontal_fov << " rad   "
      << se::math::rad_to_deg(sensor_.horizontal_fov) << " degrees \n";
    f << "Vertical FoV: " << sensor_.vertical_fov << " rad   "
      << se::math::rad_to_deg(sensor_.vertical_fov) << " degrees \n";

    return f.good();
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



void CandidateView::entropyRaycast(const PoseHistory* T_MB_history)
{
    // Raycast at the last path vertex
    raycast_entropy(entropy_image_, entropy_hits_M_, map_, sensor_, path_MB_.back(), T_BC_);
    if (config_.use_pose_history) {
        const Eigen::Matrix4f T_MC = path_MB_.back() * T_BC_;
        T_MB_history->frustumOverlap(frustum_overlap_mask_, sensor_, T_MC, T_BC_);
        entropy_image_ = mask_entropy_image(entropy_image_, frustum_overlap_mask_);
    }
    std::tie(yaw_M_, entropy_, window_idx_, window_width_) =
        optimal_yaw(entropy_image_, entropy_hits_M_, sensor_, path_MB_.back(), T_BC_);
}



void CandidateView::computeUtility()
{
    utility_ =
        (config_.exploration_weight * entropy_ + (1.0f - config_.exploration_weight) * lod_gain_)
        / path_time_;
    exploration_utility_ = entropy_ / path_time_;
    object_utility_ = lod_gain_ / path_time_;
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
    // Get rid of the trailing null byte.
    utility_str_.resize(s);
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
    constexpr float yaw_step = se::math::deg_to_rad(45);
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



void CandidateView::zeroRollPitch(Path& path_MB)
{
    for (auto& T_MB : path_MB) {
        Eigen::Quaternionf q_MB(T_MB.topLeftCorner<3, 3>());
        q_MB.x() = 0.0f;
        q_MB.y() = 0.0f;
        q_MB.normalize();
        T_MB.topLeftCorner<3, 3>() = q_MB.toRotationMatrix();
    }
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



std::ostream& operator<<(std::ostream& os, const CandidateView& c)
{
    os << "Valid:                 " << (c.isValid() ? "yes" : "no") << "\n";
    os << "Status:                " << c.status_ << "\n";
    os << "Utility:               " << c.utility() << "\n";
    os << "Exploration utility:   " << c.explorationUtility() << "\n";
    os << "Object utility:        " << c.objectUtility() << "\n";
    os << "Entropy:               " << c.entropy_ << "\n";
    os << "LoD gain:              " << c.lod_gain_ << "\n";
    os << "Path time:             " << c.path_time_ << "\n";
    os << "Path size:             " << c.path().size() << "\n";
    os << "Desired position M:    " << c.desired_t_MB_.x() << " " << c.desired_t_MB_.y() << " "
       << c.desired_t_MB_.z() << "\n";
    os << "Goal position M:       " << c.goalT_MB().topRightCorner<3, 1>().x() << " "
       << c.goalT_MB().topRightCorner<3, 1>().y() << " " << c.goalT_MB().topRightCorner<3, 1>().z()
       << "\n";
    os << "Goal yaw M:            " << c.yaw_M_ << "\n";
    os << "Window index:          " << c.window_idx_ << "\n";
    os << "Window width:          " << c.window_width_ << "\n";
    os << "Horizontal FoV:        " << se::math::rad_to_deg(c.sensor_.horizontal_fov) << "\n";
    os << "Vertical FoV:          " << se::math::rad_to_deg(c.sensor_.vertical_fov) << "\n";
    os << "Entropy image:         " << c.entropy_image_.width() << "x" << c.entropy_image_.height()
       << "\n";
    os << "Entropy hit image M:   " << c.entropy_hits_M_.width() << "x"
       << c.entropy_hits_M_.height() << "\n";
    os << "Frustum overlap mask:  " << c.frustum_overlap_mask_.width() << "x"
       << c.frustum_overlap_mask_.height() << "\n";
    os << "Min scale image:       " << c.min_scale_image_.width() << "x"
       << c.min_scale_image_.height() << "\n";
    os << "Utility computation:   " << c.utilityStr() << "\n";
    return os;
}

} // namespace se
