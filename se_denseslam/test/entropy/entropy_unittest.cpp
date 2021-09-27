// SPDX-FileCopyrightText: 2021 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include "../../src/entropy.cpp"



TEST(Entropy, logOdds) {
  for (float p = 0.03f; p <= 0.97f; p += 0.1f) {
    EXPECT_FLOAT_EQ(se::log_odds_to_prob(se::prob_to_log_odds(p)), p);
  }
}

TEST(Entropy, azimuthFromIndex) {
  constexpr int width = 1024;
  constexpr float hfov = 2.0f * M_PI_F;
  EXPECT_FLOAT_EQ(se::azimuth_from_index(            0, width, hfov), hfov / 2.0f);
  EXPECT_GT      (se::azimuth_from_index(    width / 4, width, hfov),        0.0f);
  EXPECT_FLOAT_EQ(se::azimuth_from_index(    width / 2, width, hfov),        0.0f);
  EXPECT_LT      (se::azimuth_from_index(3 * width / 4, width, hfov),        0.0f);

  for (int x = 0; x < width; x++) {
    const float theta = se::azimuth_from_index(x, width, hfov);
    EXPECT_EQ(se::index_from_azimuth(theta, width, hfov), x);
  }
}

TEST(Entropy, windowWidth) {
  // Create an entropy image
  constexpr int w = 10;
  constexpr int h = 2;
  const float data[w * h] = {
      4, 3, 2, 1, 1, 0, 0, 0, 0, 5,
      2, 2, 1, 1, 1, 1, 1, 1, 1, 1};
  se::Image<float> entropy_image (w, h);
  memcpy(entropy_image.data(), data, w * h * sizeof(float));
  se::Image<float> frustum_overlap_image (w, h, 0.0f);

  // Compute max window
  constexpr int window_width = 3;
  const std::pair<int,float> r = se::max_window(entropy_image, frustum_overlap_image, window_width);
  EXPECT_EQ(r.first, 9);
  EXPECT_FLOAT_EQ(r.second, (5 + 1 + 4 + 2 + 3 + 2));
}

