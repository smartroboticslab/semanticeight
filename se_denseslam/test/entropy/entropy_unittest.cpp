// SPDX-FileCopyrightText: 2021 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include "../../src/entropy.cpp"

#define EXPECT_FLOAT_EQ_NAN(x, y) \
    if (std::isnan(x)) { \
      EXPECT_TRUE(std::isnan(y)); \
    } else { \
      EXPECT_FLOAT_EQ(x, y); \
    }



TEST(Entropy, probTologOddsToprob) {
  for (float p = 0.03f; p <= 0.97f; p += 0.1f) {
    EXPECT_FLOAT_EQ(se::log_odds_to_prob(se::prob_to_log_odds(p)), p);
  }
}

TEST(Entropy, indexFromAzimuthFromIndex) {
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

void test_sum_columns(const std::vector<float>& data,
                      const std::vector<float>& data_sum) {
  // The data must be in row-major order.
  const int w = data_sum.size();
  const int h = data.size() / data_sum.size();
  // Create the two images.
  se::Image<float> entropy_image (w, h);
  se::Image<float> frustum_overlap_image (w, h, 0.0f);
  memcpy(entropy_image.data(), data.data(), w * h * sizeof(float));

  const std::vector<float> column_sum = se::sum_columns(entropy_image, frustum_overlap_image);
  ASSERT_EQ(data_sum.size(), column_sum.size());
  for (size_t i = 0; i < data_sum.size(); ++i) {
    EXPECT_FLOAT_EQ_NAN(data_sum[i], column_sum[i]);
  }
}

TEST(Entropy, sumColumns) {
  test_sum_columns({1, 2,
                    3, 4},
                   {4, 6});
  test_sum_columns({4, 3, 2, 1, 1, 0, 0, 0, 0, 5,
                    2, 2, 1, 1, 1, 1, 1, 1, 1, 1},
                   {6, 5, 3, 2, 2, 1, 1, 1, 1, 6});
  test_sum_columns({0,  4,  8,
                    1,  5,  9,
                    2,  6, 10,
                    3,  7, 11},
                   {6, 22, 38});
  test_sum_columns({NAN, NAN,
                    NAN,   4},
                   {NAN, NAN});
}

void test_sum_windows(const std::vector<float>& data,
                      const std::vector<float>& data_sum,
                      int                       window_width) {
  const std::vector<float> window_sum = se::sum_windows(data, window_width);
  ASSERT_EQ(window_sum.size(), data_sum.size());
  for (size_t i = 0; i < data_sum.size(); ++i) {
    EXPECT_FLOAT_EQ_NAN(data_sum[i], window_sum[i]);
  }
}

TEST(Entropy, sumWindows) {
  test_sum_windows({0, 1, 2, 3, 4},
                   {0, 1, 2, 3, 4},
                   1);
  test_sum_windows({0, 1, 2, 3, 4},
                   {1, 3, 5, 7, 4},
                   2);
  test_sum_windows({0, 1, 2, 3, 4},
                   {3, 6, 9, 7, 5},
                   3);
  test_sum_windows({0,  1, 2, 3, 4},
                   {6, 10, 9, 8, 7},
                   4);
  test_sum_windows({ 0,  1,  2,  3,  4},
                   {10, 10, 10, 10, 10},
                   5);
  test_sum_windows({NAN, NAN,   2},
                   {NAN, NAN, NAN},
                   2);
  test_sum_windows({NAN, 1,   2},
                   {NAN, 3, NAN},
                   2);
}

void text_max_window(const std::vector<float>& data,
                     int                       w,
                     int                       window_width,
                     int                       max_window_idx,
                     float                     max_window_sum) {
  // The data must be in row-major order.
  const int h = data.size() / w;
  // Create the two images.
  se::Image<float> entropy_image (w, h);
  se::Image<float> frustum_overlap_image (w, h, 0.0f);
  memcpy(entropy_image.data(), data.data(), w * h * sizeof(float));

  const std::pair<int, float> r = se::max_window(entropy_image, frustum_overlap_image, window_width);
  EXPECT_EQ(max_window_idx, r.first);
  EXPECT_FLOAT_EQ_NAN(max_window_sum, r.second);
}

TEST(Entropy, maxWindow) {
  // Best window:  ####
  text_max_window({1, 1, 0, 1, 1,
                   1, 1, 0, 1, 1,
                   1, 1, 0, 1, 1},
                   5, 2,
                   0, 6 * 1);
  // Best window:        #######
  text_max_window({1, 1, 2, 2, 2,
                   1, 1, 2, 2, 2},
                   5, 3,
                   2, 6 * 2);
  // Best window:  ####     ####
  text_max_window({1, 1, 0, 2, 2,
                   1, 1, 0, 2, 2},
                   5, 4,
                   3, 4 * 1 + 4 * 2);
  // Best window:  #########
  text_max_window({NAN, 1, 0, 2, 2,
                     1, 1, 0, 2, 2},
                   5, 3,
                   0, NAN);
  // Best window:          #######
  //text_max_window({NAN, 1, 0, 2, 2,
  //                   1, 1, 0, 2, 2},
  //                 5, 3,
  //                 2, 4 * 2);
}

