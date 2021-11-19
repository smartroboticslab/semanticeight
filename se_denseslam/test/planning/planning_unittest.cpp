// SPDX-FileCopyrightText: 2019-2021 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2019 Anna Dai
// SPDX-FileCopyrightText: 2020-2021 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include "../../src/candidate_view.cpp"



struct Pose3D {
    Eigen::Vector3f p;
    Eigen::Quaternionf q;

    Pose3D() : p(0.0, 0.0, 0.0), q(1.0, 0.0, 0.0, 0.0)
    {
    }

    Pose3D(Eigen::Vector3f point, Eigen::Quaternionf quat) : p(point), q(quat)
    {
    }

    int print(FILE* f = stdout) const
    {
        return fprintf(f,
                       "t: % 8.3f % 8.3f % 8.3f   q: % 6.3f % 6.3f % 6.3f % 6.3f\n",
                       p.x(),
                       p.y(),
                       p.z(),
                       q.x(),
                       q.y(),
                       q.z(),
                       q.w());
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct EulerAngles {
    float yaw, pitch, roll;
};



template<typename T>
using AlignedVector = std::vector<T, Eigen::aligned_allocator<T>>;

typedef AlignedVector<Eigen::Vector3i> VecVec3i;
typedef AlignedVector<Eigen::Vector3f> VecVec3f;
typedef AlignedVector<std::pair<Eigen::Vector3i, float>> VectorPair3iFloat;

typedef AlignedVector<Pose3D> VecPose;
typedef AlignedVector<VecPose> VecVecPose;
typedef AlignedVector<std::pair<Pose3D, float>> VecPairPoseFloat;



Eigen::Quaternionf toQuaternion(float yaw, float pitch, float roll)
{
    // Abbreviations for the various angular functions
    float cy = cos(yaw * 0.5);
    float sy = sin(yaw * 0.5);
    float cp = cos(pitch * 0.5);
    float sp = sin(pitch * 0.5);
    float cr = cos(roll * 0.5);
    float sr = sin(roll * 0.5);

    Eigen::Quaternionf q;
    q.w() = cy * cp * cr + sy * sp * sr;
    q.x() = cy * cp * sr - sy * sp * cr;
    q.y() = sy * cp * sr + cy * sp * cr;
    q.z() = sy * cp * cr - cy * sp * sr;
    return q;
}



void wrapYawRad(float& yaw_diff)
{
    if (yaw_diff < -M_PI) {
        yaw_diff += 2 * M_PI;
    }
    if (yaw_diff >= M_PI) {
        yaw_diff -= 2 * M_PI;
    }
}



EulerAngles toEulerAngles(Eigen::Quaternionf q)
{
    EulerAngles angles;
    // roll (x-axis rotation)
    float sinr_cosp = +2.0 * (q.w() * q.x() + q.y() * q.z());
    float cosr_cosp = +1.0 - 2.0 * (q.x() * q.x() + q.y() * q.y());
    angles.roll = atan2(sinr_cosp, cosr_cosp);
    // pitch (y-axis rotation)
    double sinp = +2.0 * (q.w() * q.y() - q.z() * q.x());
    if (fabs(sinp) >= 1) {
        angles.pitch = copysign(M_PI / 2, sinp); // use 90 degrees if out of range
    }
    else {
        angles.pitch = asin(sinp);
    }
    // yaw (z-axis rotation)
    float siny_cosp = +2.0 * (q.w() * q.z() + q.x() * q.y());
    float cosy_cosp = +1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z());
    angles.yaw = atan2(siny_cosp, cosy_cosp);
    return angles;
}



VecPose fusePath(VecPose& path_tmp, VecPose& yaw_path)
{
    VecPose path_out;
    if (yaw_path.size() >= path_tmp.size()) {
        for (size_t i = 0; i < yaw_path.size(); i++) {
            if (i < path_tmp.size()) {
                yaw_path[i].p = path_tmp[i].p;
            }
            else {
                yaw_path[i].p = path_tmp[path_tmp.size() - 1].p;
            }
        }
        path_out = yaw_path;
    }
    else {
        for (size_t i = 0; i < path_tmp.size(); i++) {
            if (i < yaw_path.size()) {
                path_tmp[i].q = yaw_path[i].q;
            }
            else {
                path_tmp[i].q = yaw_path[yaw_path.size() - 1].q;
            }
        }
        path_out = path_tmp;
    }
    return path_out;
}



VecPose getYawPath(const Pose3D& start, const Pose3D& goal, float dt, float max_yaw_rate)
{
    VecPose path;
    Pose3D pose_tmp;
    float yaw_diff = toEulerAngles(goal.q).yaw - toEulerAngles(start.q).yaw;
    wrapYawRad(yaw_diff);
    // interpolate yaw
    float yaw_increment = (max_yaw_rate * dt) / std::abs(yaw_diff);
    for (float i = 0.0f; i <= 1.0f; i += yaw_increment) {
        pose_tmp = goal;
        pose_tmp.q = start.q.slerp(i, goal.q);
        pose_tmp.q.normalize();
        path.push_back(pose_tmp);
    }
    path.push_back(goal);
    return path;
}



VecPose
addPathSegments(const Pose3D& start_in, const Pose3D& goal_in, float dt, float v_max, float res)
{
    VecPose path_out;
    path_out.push_back(start_in);
    float dist = (goal_in.p - start_in.p).norm();
    // The dist_increment makes no sense.
    // ((m/s * s) / m/voxel) / m = (m / m/voxel) / m = (m * voxel/m) / m = voxel / m
    float dist_increment = (v_max * dt / res) / dist;
    Eigen::Vector3f dir = (goal_in.p - start_in.p).normalized();
    for (float t = 0.0f; t <= 1.0f; t += dist_increment) {
        Eigen::Vector3f intermediate_point = start_in.p + dir * t * dist;
        Pose3D tmp(intermediate_point, {1.f, 0.f, 0.f, 0.f});
        path_out.push_back(tmp);
    }
    path_out.push_back(goal_in);
    return path_out;
}



VecPose getFinalPath(const Pose3D& current_pose,
                     const Pose3D& candidate_pose,
                     const VecPose& candidate_path,
                     float dt,
                     float v_max,
                     float max_yaw_rate,
                     float res)
{
    VecPose path;
    // first add points between paths
    if (candidate_path.size() > 2) {
        for (size_t i = 1; i < candidate_path.size(); i++) {
            VecPose path_tmp;
            if ((candidate_path[i].p - candidate_path[i - 1].p).norm() > v_max * dt) {
                path_tmp =
                    addPathSegments(candidate_path[i - 1], candidate_path[i], dt, v_max, res);
            }
            else {
                path_tmp.push_back(candidate_path[i - 1]);
                path_tmp.push_back(candidate_path[i]);
            }

            VecPose yaw_path;
            if (i == 1) {
                yaw_path = getYawPath(current_pose, candidate_path[i], dt, max_yaw_rate);
            }
            else {
                yaw_path = getYawPath(candidate_path[i - 1], candidate_path[i], dt, max_yaw_rate);
            }
            VecPose path_fused = fusePath(path_tmp, yaw_path);
            // push back the new path
            for (const auto& pose : path_fused) {
                path.push_back(pose);
            }
        }
    }
    else {
        VecPose yaw_path = getYawPath(current_pose, candidate_pose, dt, max_yaw_rate);
        VecPose path_tmp;
        if ((candidate_pose.p - current_pose.p).norm() > v_max * dt) {
            path_tmp = addPathSegments(current_pose, candidate_pose, dt, v_max, res);
        }
        else {
            path_tmp = candidate_path;
        }
        path = fusePath(path_tmp, yaw_path);
    }
    return path;
}



void assert_paths_eq(const VecPose& old_path, const se::Path& new_path)
{
    ASSERT_EQ(old_path.size(), new_path.size());
    for (size_t i = 0; i < old_path.size(); i++) {
        const Eigen::Vector3f t_old = old_path[i].p;
        const Eigen::Vector3f t_new = new_path[i].topRightCorner<3, 1>();
        EXPECT_TRUE(t_old.isApprox(t_new));
        //printf("% .3f % .3f % .3f == % .3f % .3f % .3f\n",
        //    t_old.x(), t_old.y(), t_old.z(), t_new.x(), t_new.y(), t_new.z());
        const Eigen::Quaternionf q_old = old_path[i].q;
        const Eigen::Quaternionf q_new(new_path[i].topLeftCorner<3, 3>());
        EXPECT_TRUE(q_old.isApprox(q_new));
        //printf("% .3f % .3f % .3f % .3f == % .3f % .3f % .3f % .3f\n",
        //    q_old.x(), q_old.y(), q_old.z(), q_old.w(), q_new.x(), q_new.y(), q_new.z(), q_new.w());
    }
}



class ICRA2020Planning : public ::testing::Test {
    public:
    // Planning parameters.
    static constexpr float delta_t_ = 0.50f;
    static constexpr float linear_velocity_ = 1.50f;
    static constexpr float angular_velocity_ = 0.75f;
    static constexpr float resolution_ = 0.01f;
    // Single segment start and end points.
    const Eigen::Vector3f t_start_ = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
    const Eigen::Vector3f t_end_ = Eigen::Vector3f(40.0f, 0.0f, 0.0f);
    const Eigen::Quaternionf q_start_ = toQuaternion(0.0f, 0.0f, 0.0f);
    const Eigen::Quaternionf q_end_ = toQuaternion(M_PI / 2.0f, 0.0f, 0.0f);
    // Single segment start and end points in the old and new formats.
    const Pose3D old_start_ = Pose3D(t_start_, q_start_);
    const Pose3D old_end_ = Pose3D(t_end_, q_end_);
    Eigen::Matrix4f new_start_ = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f new_end_ = Eigen::Matrix4f::Identity();
    VecPose old_path_;
    se::Path new_path_;

    ICRA2020Planning() : old_path_({old_start_, old_end_})
    {
        new_start_.topRightCorner<3, 1>() = t_start_;
        new_start_.topLeftCorner<3, 3>() = q_start_.toRotationMatrix();
        new_end_.topRightCorner<3, 1>() = t_end_;
        new_end_.topLeftCorner<3, 3>() = q_end_.toRotationMatrix();
        new_path_.emplace_back(new_start_);
        new_path_.emplace_back(new_end_);
    }
};

TEST_F(ICRA2020Planning, addIntermediateTranslation)
{
    VecPose old_tran_path =
        addPathSegments(old_start_, old_end_, delta_t_, linear_velocity_, resolution_);
    se::Path new_tran_path = se::CandidateView::addIntermediateTranslation(
        new_start_, new_end_, delta_t_, linear_velocity_, resolution_);
    assert_paths_eq(old_tran_path, new_tran_path);
}

TEST_F(ICRA2020Planning, addIntermediateYaw)
{
    VecPose old_yaw_path = getYawPath(old_start_, old_end_, delta_t_, angular_velocity_);
    se::Path new_yaw_path =
        se::CandidateView::addIntermediateYaw(new_start_, new_end_, delta_t_, angular_velocity_);
    assert_paths_eq(old_yaw_path, new_yaw_path);
}

TEST_F(ICRA2020Planning, fuseIntermediatePaths)
{
    VecPose old_tran_path =
        addPathSegments(old_start_, old_end_, delta_t_, linear_velocity_, resolution_);
    VecPose old_yaw_path = getYawPath(old_start_, old_end_, delta_t_, angular_velocity_);
    VecPose old_path = fusePath(old_tran_path, old_yaw_path);
    se::Path new_tran_path = se::CandidateView::addIntermediateTranslation(
        new_start_, new_end_, delta_t_, linear_velocity_, resolution_);
    se::Path new_yaw_path =
        se::CandidateView::addIntermediateYaw(new_start_, new_end_, delta_t_, angular_velocity_);
    se::Path new_path = se::CandidateView::fuseIntermediatePaths(new_tran_path, new_yaw_path);
    assert_paths_eq(old_path, new_path);
}

TEST_F(ICRA2020Planning, getFinalPath)
{
    const VecPose old_final_path = getFinalPath(old_start_,
                                                old_end_,
                                                old_path_,
                                                delta_t_,
                                                linear_velocity_,
                                                angular_velocity_,
                                                resolution_);
    const se::Path new_final_path = se::CandidateView::getFinalPath(
        new_path_, delta_t_, linear_velocity_, angular_velocity_, resolution_);
    assert_paths_eq(old_final_path, new_final_path);
}
