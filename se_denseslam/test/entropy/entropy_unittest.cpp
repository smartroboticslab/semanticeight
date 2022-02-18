// SPDX-FileCopyrightText: 2021 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#include "../../src/entropy.cpp"

#include <gtest/gtest.h>
#include <lodepng.h>
#include <se/filesystem.hpp>

#define EXPECT_FLOAT_EQ_NAN(x, y)   \
    if (std::isnan(x)) {            \
        EXPECT_TRUE(std::isnan(y)); \
    }                               \
    else {                          \
        EXPECT_FLOAT_EQ(x, y);      \
    }



TEST(Entropy, probToLogOdds)
{
    EXPECT_FLOAT_EQ(se::prob_to_log_odds(0.0f), -INFINITY);
    EXPECT_FLOAT_EQ(se::prob_to_log_odds(1.0f), INFINITY);
    EXPECT_FLOAT_EQ(se::prob_to_log_odds(0.5f), 0.0f);
    EXPECT_LT(se::prob_to_log_odds(0.25f), 0.0f);
    EXPECT_GT(se::prob_to_log_odds(0.75f), 0.0f);
}

TEST(Entropy, logOddsToProb)
{
    EXPECT_FLOAT_EQ(se::log_odds_to_prob(-INFINITY), 0.0f);
    EXPECT_FLOAT_EQ(se::log_odds_to_prob(INFINITY), 1.0f);
    EXPECT_FLOAT_EQ(se::log_odds_to_prob(-100.0f), 0.0f);
    EXPECT_FLOAT_EQ(se::log_odds_to_prob(100.0f), 1.0f);
    EXPECT_FLOAT_EQ(se::log_odds_to_prob(0.0f), 0.5f);
    EXPECT_GT(se::log_odds_to_prob(-1.0f), 0.0f);
    EXPECT_LT(se::log_odds_to_prob(-1.0f), 0.5f);
    EXPECT_GT(se::log_odds_to_prob(1.0f), 0.5f);
    EXPECT_LT(se::log_odds_to_prob(1.0f), 1.0f);
}

TEST(Entropy, probToLogOddsToProb)
{
    for (float p = 0.03f; p <= 0.97f; p += 0.1f) {
        EXPECT_FLOAT_EQ(se::log_odds_to_prob(se::prob_to_log_odds(p)), p);
    }
}

TEST(Entropy, entropy)
{
    EXPECT_FLOAT_EQ(se::entropy(0.0f), 0.0f);
    EXPECT_FLOAT_EQ(se::entropy(0.5f), 1.0f);
    EXPECT_FLOAT_EQ(se::entropy(1.0f), 0.0f);
    EXPECT_GT(se::entropy(0.25f), 0.0f);
    EXPECT_LT(se::entropy(0.25f), 1.0f);
    EXPECT_GT(se::entropy(0.75f), 0.0f);
    EXPECT_LT(se::entropy(0.75f), 1.0f);
    EXPECT_FLOAT_EQ(se::entropy(0.25f), se::entropy(0.75f));
    EXPECT_LT(se::entropy(0.25f), se::entropy(0.30f));
}

TEST(Entropy, indexToAzimuth)
{
    constexpr float hfov = M_TAU_F;
    for (int width : {10, 1024}) {
        // Compute the azimuth angle for all indices up to width - 1.
        std::vector<int> idx(width);
        std::iota(idx.begin(), idx.end(), 0);
        std::vector<float> azimuth(idx.size());
        std::transform(idx.begin(), idx.end(), azimuth.begin(), [width, hfov](auto i) {
            return se::index_to_azimuth(i, width, hfov);
        });
        for (size_t i = 0; i < azimuth.size(); ++i) {
            // Test bounds.
            EXPECT_GE(azimuth[i], -hfov / 2.0f);
            EXPECT_LT(azimuth[i], hfov / 2.0f);
            // Test for correct sign.
            if (i < azimuth.size() / 2) {
                EXPECT_GE(azimuth[i], 0.0f);
            }
            else {
                EXPECT_LE(azimuth[i], 0.0f);
            }
            // Test correct order (azimuth angles increase left-to-right).
            if (i > 0) {
                EXPECT_GT(azimuth[i - 1], azimuth[i]);
            }
        }
    }
}

TEST(Entropy, indexToPolar)
{
    constexpr int height = 64;
    constexpr float vfov = se::math::deg_to_rad(20.0f);
    constexpr float pitch_offset = se::math::deg_to_rad(10.0f);
    constexpr float polar_min = M_PI_F / 2.0f - vfov / 2.0f + pitch_offset;
    constexpr float polar_max = M_PI_F / 2.0f + vfov / 2.0f + pitch_offset;
    // Compute the polar angle for all indices up to height - 1.
    std::vector<int> idx(height);
    std::iota(idx.begin(), idx.end(), 0);
    std::vector<float> polar(idx.size());
    std::transform(idx.begin(), idx.end(), polar.begin(), [height, vfov, pitch_offset](auto i) {
        return se::index_to_polar(i, height, vfov, pitch_offset);
    });
    for (auto p : polar) {
        EXPECT_GT(p, polar_min);
        EXPECT_LT(p, polar_max);
    }
    for (size_t i = 0; i < polar.size() - 1; ++i) {
        EXPECT_LT(polar[i], polar[i + 1]);
    }
}

TEST(Entropy, azimuthToIndex)
{
    constexpr int width = 1024;
    constexpr float hfov = M_TAU_F;
    // Compute the indices for some azimuth angles in the interval (-pi, pi].
    std::vector<float> azimuth = {
        -hfov / 3.0f, -hfov / 4.0, -hfov / 8.0f, 0.0f, hfov / 8.0, hfov / 4.0, hfov / 3.0f};
    std::vector<int> idx(azimuth.size());
    std::transform(azimuth.begin(), azimuth.end(), idx.begin(), [width, hfov](auto a) {
        return se::azimuth_to_index(a, width, hfov);
    });
    for (auto i : idx) {
        EXPECT_GE(i, 0);
        EXPECT_LT(i, width);
    }
    for (size_t i = 0; i < idx.size() - 1; ++i) {
        EXPECT_GT(idx[i], idx[i + 1]);
    }
    // Test -pi separately because it doesn't follow the ordering.
    EXPECT_EQ(se::azimuth_to_index(-hfov / 2.0f, width, hfov), 0);
}

TEST(Entropy, indexToAzimuthToIndex)
{
    constexpr int width = 1024;
    constexpr float hfov = M_TAU_F;
    EXPECT_GT(se::index_to_azimuth(width / 4, width, hfov), 0.0f);
    EXPECT_LT(se::index_to_azimuth(3 * width / 4, width, hfov), 0.0f);

    for (int x = 0; x < width; x++) {
        const float theta = se::index_to_azimuth(x, width, hfov);
        EXPECT_EQ(se::azimuth_to_index(theta, width, hfov), x);
    }
}

TEST(Entropy, azimuthToIndexToAzimuth)
{
    constexpr int width = 1024;
    constexpr float hfov = M_TAU_F;
    constexpr float threshold = hfov / width / 2.0f;
    // Can't test -hfov/2 because it will wrap around the image (as intended).
    for (float a = -hfov / 2.0f + 0.001f; a < hfov / 2.0f; a += hfov / 8.0f) {
        const int idx = se::azimuth_to_index(a, width, hfov);
        EXPECT_LE(std::fabs(se::index_to_azimuth(idx, width, hfov) - a), threshold);
    }
}

TEST(Entropy, overlayYaw)
{
    for (auto yaw_M : {0.0f, M_PI_F / 3.0f, -M_PI_F}) {
        se::Image<uint32_t> image(36, 10, 0);
        const SensorImpl sensor({640, 480, false, 0.02, 5.0, 205.5, 205.5, 319.5, 239.5});
        const int window_idx = se::azimuth_to_index(
            se::math::wrap_angle_2pi(yaw_M + sensor.horizontal_fov / 2.0f), image.width(), M_TAU_F);
        const int window_width = se::compute_window_width(image.width(), sensor.horizontal_fov);
        overlay_yaw(image, window_idx, window_width);
        const std::string filename =
            stdfs::temp_directory_path() / std::string("overlay_" + std::to_string(yaw_M) + ".png");
        lodepng_encode32_file(filename.c_str(),
                              reinterpret_cast<const unsigned char*>(image.data()),
                              image.width(),
                              image.height());
    }
}
