// SPDX-FileCopyrightText: 2021 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#include <frustum_intersector.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <se/sensor.hpp>

class FrustumIntersection : public ::testing::Test {
    public:
    FrustumIntersection() :
            sensor_config_({640, 480, false, 0.4f, 4.0f, 525.0f, 525.0f, 319.5f, 239.5f}),
            sensor_(sensor_config_)
    {
    }

    const se::SensorConfig sensor_config_;
    const se::PinholeCamera sensor_;
};



TEST_F(FrustumIntersection, PolygonMeshProcessing)
{
    Eigen::Matrix4f T_C0C1 = Eigen::Matrix4f::Identity();
    float overlap = fi::frustum_intersection_pc(sensor_.frustum_vertices_, T_C0C1);
    EXPECT_FLOAT_EQ(overlap, 1.0f);

    T_C0C1.topRightCorner<3, 1>() += Eigen::Vector3f::Constant(0.5f);
    overlap = fi::frustum_intersection_pc(sensor_.frustum_vertices_, T_C0C1);
    EXPECT_LT(overlap, 1.0f);

    T_C0C1.topRightCorner<3, 1>().x() = 1000.0f;
    overlap = fi::frustum_intersection_pc(sensor_.frustum_vertices_, T_C0C1);
    EXPECT_FLOAT_EQ(overlap, 0.0f);
}
