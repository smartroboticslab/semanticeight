/*

Copyright 2016 Emanuele Vespa, Imperial College London

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include <gtest/gtest.h>
#include <se/utils/math_utils.h>

TEST(EigenUtils, ClampFixVec3)
{
    Eigen::Vector3i base{0, 20, -1};
    Eigen::Vector3i min{0, 0, 0};
    Eigen::Vector3i max{10, 10, 10};
    se::math::clamp(base, min, max);

    ASSERT_TRUE(base.x() >= 0 && base.x() <= 10);
    ASSERT_TRUE(base.y() >= 0 && base.y() <= 10);
    ASSERT_TRUE(base.z() >= 0 && base.z() <= 10);
}

TEST(EigenUtils, ClampFixVec2)
{
    Eigen::Vector2f base{-100.f, 34.f};
    Eigen::Vector2f min{0.f, 0.f};
    Eigen::Vector2f max{20.f, 10.f};
    se::math::clamp(base, min, max);

    ASSERT_TRUE(base.x() >= min.x() && base.x() <= max.x());
    ASSERT_TRUE(base.y() >= min.y() && base.y() <= max.y());
}

TEST(MathUtils, validTranformation)
{
    EXPECT_TRUE(se::math::is_valid_transformation(Eigen::Matrix4f::Identity()));
    {
        Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
        T(3, 3) += 0.01f;
        EXPECT_FALSE(se::math::is_valid_transformation(T));
    }
    {
        Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
        T(0, 0) -= 0.0001f;
        EXPECT_FALSE(se::math::is_valid_transformation(T));
    }
}

TEST(MathUtils, median)
{
    std::vector<int> v1{1, 2, 3, 4};
    EXPECT_EQ(se::math::median(v1), 2);

    std::vector<int> v2{2, 3, 4};
    EXPECT_EQ(se::math::median(v2), 3);

    const std::vector<short> v3{15, 0, 2, 4};
    EXPECT_EQ(se::math::median(v3), 3);

    std::vector<short> v4;
    EXPECT_EQ(se::math::median(v4), short());

    std::vector<float> v5{1.0f, 28.123f, 0.0f, 8.5f, 2.4f};
    EXPECT_FLOAT_EQ(se::math::median(v5), 2.4f);

    const std::vector<float> v6{1.0f, 0.0f, 8.5f, 2.4f};
    EXPECT_FLOAT_EQ(se::math::median(v6), 1.7f);
}

TEST(MathUtils, wrap_angle_2pi)
{
    constexpr float pi = M_PI;
    constexpr float tau = 2 * M_PI;
    constexpr float error = 0.00001;
    constexpr int n = 1 + 4 * 8;
    const float angles[n] = {
        // Positive, first circle
        0,
        pi / 4,
        pi / 2,
        3 * pi / 4,
        pi,
        5 * pi / 4,
        3 * pi / 2,
        7 * pi / 4,
        tau,
        // Positive, second circle
        tau + pi / 4,
        tau + pi / 2,
        tau + 3 * pi / 4,
        tau + pi,
        tau + 5 * pi / 4,
        tau + 3 * pi / 2,
        tau + 7 * pi / 4,
        2 * tau,
        // Negative, first circle
        -pi / 4,
        -pi / 2,
        -3 * pi / 4,
        -pi,
        -5 * pi / 4,
        -3 * pi / 2,
        -7 * pi / 4,
        -tau,
        // Negative, second circle
        -tau - pi / 4,
        -tau - pi / 2,
        -tau - 3 * pi / 4,
        -tau - pi,
        -tau - 5 * pi / 4,
        -tau - 3 * pi / 2,
        -tau - 7 * pi / 4,
        -2 * tau,
    };
    const float wrapped_angles[n] = {
        // Positive, first circle
        0,
        pi / 4,
        pi / 2,
        3 * pi / 4,
        pi,
        5 * pi / 4,
        3 * pi / 2,
        7 * pi / 4,
        0,
        // Positive, second circle
        pi / 4,
        pi / 2,
        3 * pi / 4,
        pi,
        5 * pi / 4,
        3 * pi / 2,
        7 * pi / 4,
        0,
        // Negative, first circle
        7 * pi / 4,
        3 * pi / 2,
        5 * pi / 4,
        pi,
        3 * pi / 4,
        pi / 2,
        pi / 4,
        0,
        // Negative, second circle
        7 * pi / 4,
        3 * pi / 2,
        5 * pi / 4,
        pi,
        3 * pi / 4,
        pi / 2,
        pi / 4,
        0,
    };

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(se::math::wrap_angle_2pi(angles[i]), wrapped_angles[i], error);
    }
}

TEST(MathUtils, wrap_angle_pi)
{
    constexpr float pi = M_PI;
    constexpr float tau = 2 * M_PI;
    constexpr float error = 0.00001;
    constexpr int n = 1 + 4 * 8;
    const float angles[n] = {
        // Positive, first circle
        0,
        pi / 4,
        pi / 2,
        3 * pi / 4,
        pi,
        5 * pi / 4,
        3 * pi / 2,
        7 * pi / 4,
        tau,
        // Positive, second circle
        tau + pi / 4,
        tau + pi / 2,
        tau + 3 * pi / 4,
        tau + pi,
        tau + 5 * pi / 4,
        tau + 3 * pi / 2,
        tau + 7 * pi / 4,
        2 * tau,
        // Negative, first circle
        -pi / 4,
        -pi / 2,
        -3 * pi / 4,
        -pi,
        -5 * pi / 4,
        -3 * pi / 2,
        -7 * pi / 4,
        -tau,
        // Negative, second circle
        -tau - pi / 4,
        -tau - pi / 2,
        -tau - 3 * pi / 4,
        -tau - pi,
        -tau - 5 * pi / 4,
        -tau - 3 * pi / 2,
        -tau - 7 * pi / 4,
        -2 * tau,
    };
    const float wrapped_angles[n] = {
        // Positive, first circle
        0,
        pi / 4,
        pi / 2,
        3 * pi / 4,
        -pi,
        -3 * pi / 4,
        -pi / 2,
        -pi / 4,
        0,
        // Positive, second circle
        pi / 4,
        pi / 2,
        3 * pi / 4,
        -pi,
        -3 * pi / 4,
        -pi / 2,
        -pi / 4,
        0,
        // Negative, first circle
        -pi / 4,
        -pi / 2,
        -3 * pi / 4,
        -pi,
        3 * pi / 4,
        pi / 2,
        pi / 4,
        0,
        // Negative, second circle
        -pi / 4,
        -pi / 2,
        -3 * pi / 4,
        -pi,
        3 * pi / 4,
        pi / 2,
        pi / 4,
        0,
    };

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(se::math::wrap_angle_pi(angles[i]), wrapped_angles[i], error);
    }
}

TEST(MathUtils, angle_diff)
{
    constexpr float pi = M_PI;
    constexpr float error = 0.00001;
    constexpr int n = 8;
    const float start_angles[n] = {0, 0, pi / 2, 0, -pi / 2, 0, 3 * pi / 2, pi / 4};
    const float end_angles[n] = {0, pi / 2, 0, -pi / 2, 0, 3 * pi / 2, 0, 7 * pi / 4};
    const float angle_diff[n] = {0, pi / 2, -pi / 2, -pi / 2, pi / 2, -pi / 2, pi / 2, -pi / 2};
    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(se::math::angle_diff(start_angles[i], end_angles[i]), angle_diff[i], error);
    }
}

TEST(MathUtils, is_between)
{
    EXPECT_TRUE(se::math::is_between(
        Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 0, 0)));
    EXPECT_TRUE(se::math::is_between(Eigen::Vector3f(-1, 2, 0),
                                     Eigen::Vector3f(-3, 0, -INFINITY),
                                     Eigen::Vector3f(1, INFINITY, 3)));
    EXPECT_TRUE(se::math::is_between(Eigen::Vector3f(2, 3, -8),
                                     Eigen::Vector3f(-INFINITY, -INFINITY, -INFINITY),
                                     Eigen::Vector3f(INFINITY, INFINITY, INFINITY)));
    EXPECT_TRUE(se::math::is_between(Eigen::Vector3f(2, 3, 1.2),
                                     Eigen::Vector3f(-INFINITY, -INFINITY, 0.5),
                                     Eigen::Vector3f(INFINITY, INFINITY, 2.5)));
    EXPECT_FALSE(se::math::is_between(
        Eigen::Vector3f(2, -1, 5.5), Eigen::Vector3f(-1, 0, 5), Eigen::Vector3f(3, 8, 6)));
}

TEST(MathUtils, rotm_and_yaw)
{
    for (float y = -M_TAU_F; y <= M_TAU_F; y += M_TAU_F / 16.0f) {
        EXPECT_FLOAT_EQ(se::math::wrap_angle_pi(y),
                        se::math::rotm_to_yaw(se::math::yaw_to_rotm(se::math::wrap_angle_pi(y))));
    }
}
