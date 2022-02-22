// SPDX-FileCopyrightText: 2022 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2022 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include <se/filesystem.hpp>
#include <se/image_utils.hpp>
#include <se/pose_mask_history.hpp>



class PoseMaskTest : public ::testing::Test {
    public:
    const se::SensorConfig sensor_config_;
    const se::PinholeCamera sensor_;
    const Eigen::Vector2i raycast_res_;
    const Eigen::Matrix4f T_BC_;
    const Eigen::Vector3f dimensions_;
    const Eigen::Vector3f resolution_;
    se::PoseMaskHistory history_;
    const se::Image<se::PoseMaskHistory::MaskType> all_valid_mask_;
    const se::Image<se::PoseMaskHistory::MaskType> all_invalid_mask_;
    const se::Image<float> all_invalid_depth_;
    std::string tmp_;

    PoseMaskTest() :
            sensor_config_({640, 480, false, 0.4f, 4.0f, 525.0f, 525.0f, 319.5f, 239.5f}),
            sensor_(sensor_config_),
            raycast_res_(36, 10),
            T_BC_(
                (Eigen::Matrix4f() << 0, 0, 1, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 1).finished()),
            dimensions_(2.0f, 2.0f, 1.0f),
            resolution_(0.5f, 1.0f, 0.5f),
            history_(raycast_res_, sensor_, T_BC_, dimensions_, resolution_),
            all_valid_mask_(raycast_res_.x(), raycast_res_.y(), 0),
            all_invalid_mask_(raycast_res_.x(), raycast_res_.y(), UINT8_MAX),
            all_invalid_depth_(sensor_config_.width, sensor_config_.height, 0.0f),
            tmp_(stdfs::temp_directory_path() / stdfs::path("semanticeight_test_results"))
    {
        // Create a temporary directory for all the tests.
        stdfs::create_directories(tmp_);
    }
};



TEST_F(PoseMaskTest, initialization)
{
    for (float z = 0.0f; z < dimensions_.z(); z += resolution_.z()) {
        for (float y = 0.0f; y < dimensions_.y(); y += resolution_.y()) {
            for (float x = 0.0f; x < dimensions_.x(); x += resolution_.x()) {
                const auto& mask = history_.getMask(Eigen::Vector3f(x, y, z));
                ASSERT_EQ(mask.size(), all_valid_mask_.size());
                EXPECT_EQ(memcmp(mask.data(),
                                 all_valid_mask_.data(),
                                 sizeof(se::PoseMaskHistory::MaskType) * mask.size()),
                          0);
            }
        }
    }
}

TEST_F(PoseMaskTest, recordAndGet)
{
    for (float z = 0.0f; z < dimensions_.z(); z += resolution_.z()) {
        for (float y = 0.0f; y < dimensions_.y(); y += resolution_.y()) {
            for (float x = 0.0f; x < dimensions_.x(); x += resolution_.x()) {
                for (float yaw = 0.0f; yaw < 360.0f; yaw += 45.0f) {
                    Eigen::Matrix4f T_MB = Eigen::Matrix4f::Identity();
                    T_MB.topRightCorner<3, 1>() = Eigen::Vector3f(x, y, z);
                    T_MB.topLeftCorner<3, 3>() = se::math::yaw_to_rotm(se::math::deg_to_rad(yaw));
                    history_.record(T_MB, all_invalid_depth_);
                    const auto& mask = history_.getMask(T_MB);
                    se::save_pgm(mask,
                                 tmp_ + "/history_mask_" + std::to_string(x) + "_"
                                     + std::to_string(y) + "_" + std::to_string(z) + "_yaw_"
                                     + std::to_string(static_cast<int>(yaw)) + ".pgm");
                }
                EXPECT_EQ(memcmp(history_.getMask(Eigen::Vector3f(x, y, z)).data(),
                                 all_invalid_mask_.data(),
                                 sizeof(se::PoseMaskHistory::MaskType) * all_invalid_mask_.size()),
                          0);
            }
        }
    }
}

TEST_F(PoseMaskTest, writeMasks)
{
    history_.writeMasks(tmp_ + "/history_masks");
}
