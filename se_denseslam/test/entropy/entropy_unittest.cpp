// SPDX-FileCopyrightText: 2021 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#include "../../src/entropy.cpp"

#include <gtest/gtest.h>

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
}

TEST(Entropy, indexFromAzimuthFromIndex)
{
    constexpr int width = 1024;
    constexpr float hfov = 2.0f * M_PI_F;
    //EXPECT_FLOAT_EQ(se::azimuth_from_index(0, width, hfov), hfov / 2.0f);
    EXPECT_GT(se::azimuth_from_index(width / 4, width, hfov), 0.0f);
    //EXPECT_FLOAT_EQ(se::azimuth_from_index(width / 2, width, hfov), 0.0f);
    EXPECT_LT(se::azimuth_from_index(3 * width / 4, width, hfov), 0.0f);

    for (int x = 0; x < width; x++) {
        const float theta = se::azimuth_from_index(x, width, hfov);
        EXPECT_EQ(se::index_from_azimuth(theta, width, hfov), x);
    }
}

