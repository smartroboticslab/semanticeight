// SPDX-FileCopyrightText: 2021 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#include "se/pose_grid_history.hpp"

#include <gtest/gtest.h>



class PoseGridTest : public ::testing::Test {
    public:
    const se::SensorConfig sensor_config_;
    const se::PinholeCamera sensor_;
    const Eigen::Vector3f dimensions_;
    const Eigen::Vector4f resolution_;
    se::PoseGridHistory grid_;

    PoseGridTest() :
            sensor_config_({640, 480, false, 0.4f, 4.0f, 525.0f, 525.0f, 319.5f, 239.5f}),
            sensor_(sensor_config_),
            dimensions_(2.0f, 2.0f, 1.0f),
            resolution_(0.5f, 1.0f, 0.5f, 36.0f * se::math::deg_to_rad),
            grid_(dimensions_, resolution_)
    {
    }
};



TEST_F(PoseGridTest, initialization)
{
    // Test the grid was created as expected.
    EXPECT_TRUE(grid_.dimensions().isApprox(dimensions_));
    EXPECT_TRUE(grid_.resolution().isApprox(resolution_));
    EXPECT_EQ(grid_.dimensionsCells().x(), 4);
    EXPECT_EQ(grid_.dimensionsCells().y(), 2);
    EXPECT_EQ(grid_.dimensionsCells().z(), 2);
    EXPECT_EQ(grid_.dimensionsCells().w(), 10);
    EXPECT_EQ(grid_.size(), 4 * 2 * 2 * 10);
    // Test that the grid contains no measurements.
    for (float x = resolution_.x() / 4.0f; x < dimensions_.x(); x += resolution_.x()) {
        for (float y = resolution_.y() / 4.0f; y < dimensions_.y(); y += resolution_.y()) {
            for (float z = resolution_.z() / 4.0f; z < dimensions_.z(); z += resolution_.z()) {
                for (float yaw = resolution_.w() / 4.0f; yaw < M_TAU_F; yaw += resolution_.w()) {
                    Eigen::Matrix4f T;
                    T << cos(yaw), -sin(yaw), 0, x, sin(yaw), cos(yaw), 0, y, 0, 0, 1, z, 0, 0, 0,
                        1;
                    EXPECT_EQ(grid_.get(T), 0);
                    EXPECT_EQ(grid_.get(Eigen::Vector4f(x, y, z, yaw)), 0);
                }
            }
        }
    }
}



TEST_F(PoseGridTest, recordAndGetAll)
{
    // Record two poses for each cell.
    for (float x = resolution_.x() / 4.0f; x < dimensions_.x(); x += resolution_.x()) {
        for (float y = resolution_.y() / 4.0f; y < dimensions_.y(); y += resolution_.y()) {
            for (float z = resolution_.z() / 4.0f; z < dimensions_.z(); z += resolution_.z()) {
                for (float yaw = resolution_.w() / 4.0f; yaw < M_TAU_F; yaw += resolution_.w()) {
                    Eigen::Matrix4f T;
                    T << cos(yaw), -sin(yaw), 0, x, sin(yaw), cos(yaw), 0, y, 0, 0, 1, z, 0, 0, 0,
                        1;
                    grid_.record(T);
                    grid_.record(Eigen::Vector4f(x, y, z, yaw));
                }
            }
        }
    }
    // Ensure that each cell contains two measurements.
    for (float x = resolution_.x() / 4.0f; x < dimensions_.x(); x += resolution_.x()) {
        for (float y = resolution_.y() / 4.0f; y < dimensions_.y(); y += resolution_.y()) {
            for (float z = resolution_.z() / 4.0f; z < dimensions_.z(); z += resolution_.z()) {
                for (float yaw = resolution_.w() / 4.0f; yaw < M_TAU_F; yaw += resolution_.w()) {
                    Eigen::Matrix4f T;
                    T << cos(yaw), -sin(yaw), 0, x, sin(yaw), cos(yaw), 0, y, 0, 0, 1, z, 0, 0, 0,
                        1;
                    EXPECT_EQ(grid_.get(T), 2);
                    EXPECT_EQ(grid_.get(Eigen::Vector4f(x, y, z, yaw)), 2);
                }
            }
        }
    }
    EXPECT_EQ(grid_.visitedPoses().size(), grid_.size());
}



TEST_F(PoseGridTest, recordOld)
{
    // Record various poses in the 0,0,0,0 cell.
    const Eigen::Vector4f base_pose(0, 0, 0, 0);
    EXPECT_EQ(grid_.get(base_pose), 0);

    grid_.record(base_pose);
    EXPECT_EQ(grid_.get(base_pose), 1);

    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T.topRightCorner<3, 1>() = grid_.resolution().head<3>() / 2.0f;
    grid_.record(T);
    EXPECT_EQ(grid_.get(base_pose), 2);

    grid_.record(Eigen::Vector4f(0, 0, 0, grid_.resolution().w() / 3.0f));
    EXPECT_EQ(grid_.get(base_pose), 3);

    // Record a pose in some other cell.
    T << 0, -1, 0, 2, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
    EXPECT_EQ(grid_.get(T), 0);
    grid_.record(T);
    EXPECT_EQ(grid_.get(T), 1);
}



TEST_F(PoseGridTest, recordYaw)
{
    int n = 0;
    for (float yaw = resolution_.w() / 4.0f; yaw < M_TAU_F; yaw += resolution_.w()) {
        n++;
        for (int i = 0; i < n; i++) {
            grid_.record(Eigen::Vector4f(0, 0, 0, yaw));
        }
    }
}



TEST_F(PoseGridTest, rejectionProbability)
{
    const Eigen::Vector4f pose(0.0f, 0.0f, 0.0f, 0.0f);
    EXPECT_FLOAT_EQ(grid_.rejectionProbability(pose.head<3>(), sensor_), 0.0f);
    for (int i = 0; i < 2; i++) {
        grid_.record(pose);
        EXPECT_FLOAT_EQ(grid_.rejectionProbability(pose.head<3>(), sensor_),
                        1.0f / grid_.dimensionsCells().w());
    }
}



TEST_F(PoseGridTest, visitedPoses)
{
    const Eigen::Vector4f pose(0.0f, 0.0f, 0.0f, 0.0f);
    grid_.record(pose);

    auto visited_poses = grid_.visitedPoses();
    ASSERT_EQ(visited_poses.size(), 1u);
    EXPECT_FLOAT_EQ(visited_poses.front().x(), pose.x());
    EXPECT_FLOAT_EQ(visited_poses.front().y(), pose.y());
    EXPECT_FLOAT_EQ(visited_poses.front().z(), pose.z());
    EXPECT_FLOAT_EQ(visited_poses.front().w(), pose.w());
}



TEST_F(PoseGridTest, singularity)
{
    // Exact identity matrix.
    Eigen::Matrix4f T0 = Eigen::Matrix4f::Identity();
    grid_.record(T0);
    EXPECT_EQ(grid_.get(T0), 1);

    // Less exact identity matrix but returns the correct yaw of 0.
    Eigen::Matrix4f T1;
    T1 << 1, 0, -4.47035e-08, 0, 0, 1, 0, 0, 4.47035e-08, 0, 1, 0, 0, 0, 0, 1;
    grid_.record(T1);
    EXPECT_EQ(grid_.get(T0), 2);

    // Less exact identity matrix but returns a wrong yaw of pi.
    Eigen::Matrix4f T2;
    T2 << 1, 0, -1.49012e-08, 0, -2.96534e-08, 1, 2.97527e-09, 0, 4.47035e-08, -2.98023e-08, 1, 0,
        0, 0, 0, 1;
    grid_.record(T2);
    EXPECT_EQ(grid_.get(T0), 3);

    EXPECT_TRUE(T2.isApprox(T0));
}