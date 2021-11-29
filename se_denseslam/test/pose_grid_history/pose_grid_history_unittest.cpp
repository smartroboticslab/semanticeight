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
            resolution_(0.5f, 1.0f, 0.5f, 72.0f * se::math::deg_to_rad),
            grid_(dimensions_, resolution_)
    {
    }
};



// Test that the grid was initialized to the expected number of cells.
TEST_F(PoseGridTest, initialization)
{
    const Eigen::Vector4i s = grid_.dimensionsCells();
    EXPECT_EQ(s.x(), 4);
    EXPECT_EQ(s.y(), 2);
    EXPECT_EQ(s.z(), 2);
    EXPECT_EQ(s.w(), 5);
    EXPECT_EQ(grid_.size(), 4 * 2 * 2 * 5);
}



TEST_F(PoseGridTest, increment)
{
    // Record various poses in the 0,0,0,0 cell.
    const Eigen::Vector4f base_pose(0.0f, 0.0f, 0.0f, 0.0f);
    EXPECT_EQ(grid_.get(base_pose), 0);
    grid_.record(base_pose);
    EXPECT_EQ(grid_.get(base_pose), 1);
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T.topRightCorner<3, 1>() = grid_.resolution().head<3>() / 2.0f;
    grid_.record(T);
    EXPECT_EQ(grid_.get(base_pose), 2);
    grid_.record(Eigen::Vector4f(0.0f, 0.0f, 0.0f, grid_.resolution().w() / 3.0f));
    EXPECT_EQ(grid_.get(base_pose), 3);
    grid_.record(Eigen::Vector4f(0.0f, 0.0f, 0.0f, -grid_.resolution().w() / 4.0f));
    EXPECT_EQ(grid_.get(base_pose), 4);

    // Record a pose in some other cell.
    Eigen::Matrix4f pose4;
    pose4 << 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
    EXPECT_EQ(grid_.get(pose4), 0);
    grid_.record(pose4);
    EXPECT_EQ(grid_.get(pose4), 1);
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
