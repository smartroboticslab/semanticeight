/*
 * SPDX-FileCopyrightText: 2019-2020 Smart Robotics Lab, Imperial College London
 * SPDX-FileCopyrightText: 2019-2020 Sotiris Papatheodorou, Imperial College London
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <memory>

#include <Eigen/Dense>

#include <se/image_utils.hpp>



class DepthSaveLoad : public ::testing::Test {
  protected:
    virtual void SetUp() {
      depth_image_data_ = new uint16_t[num_pixels_]();

      // Initialize the test image with a pattern.
      for (size_t x = 0; x < depth_image_width_; ++x) {
        for (size_t y = 0; y < depth_image_height_; ++y) {
          if (x > y) {
            depth_image_data_[x + depth_image_width_ * y] = UINT16_MAX / 4;
          } else if (x == y) {
            depth_image_data_[x + depth_image_width_ * y] = UINT16_MAX / 2;
          } else {
            depth_image_data_[x + depth_image_width_ * y] = UINT16_MAX;
          }
        }
      }
    }

    uint16_t* depth_image_data_;
    const size_t depth_image_width_  = 64;
    const size_t depth_image_height_ = 64;
    const size_t num_pixels_ = depth_image_width_ * depth_image_height_;
    const size_t depth_size_bytes_ = sizeof(uint16_t) * num_pixels_;
    const Eigen::Vector2i depth_image_res_
        = Eigen::Vector2i(depth_image_width_, depth_image_height_);
};



TEST_F(DepthSaveLoad, SaveThenLoadPNG) {
  // Save the image.
  const int save_ok = se::save_depth_png(depth_image_data_, depth_image_res_, "/tmp/depth.png");
  EXPECT_EQ(save_ok, 0);

  // Load the image.
  uint16_t* loaded_depth_image_data;
  Eigen::Vector2i loaded_depth_image_res (0, 0);
  const int load_ok = se::load_depth_png(&loaded_depth_image_data, loaded_depth_image_res, "/tmp/depth.png");
  EXPECT_EQ(load_ok, 0);

  // Compare the loaded image with the saved one.
  EXPECT_EQ(static_cast<unsigned>(loaded_depth_image_res.x()), depth_image_width_);
  EXPECT_EQ(static_cast<unsigned>(loaded_depth_image_res.y()), depth_image_height_);
  EXPECT_EQ(memcmp(loaded_depth_image_data, depth_image_data_, depth_size_bytes_), 0);

  free(loaded_depth_image_data);
}



TEST_F(DepthSaveLoad, SaveThenLoadPGM) {
  // Save the image.
  const int save_ok = se::save_depth_pgm(depth_image_data_, depth_image_res_, "/tmp/depth.pgm");
  EXPECT_EQ(save_ok, 0);

  // Load the image.
  uint16_t* loaded_depth_image_data;
  Eigen::Vector2i loaded_depth_image_res (0, 0);
  const int load_ok = se::load_depth_pgm(&loaded_depth_image_data, loaded_depth_image_res, "/tmp/depth.pgm");
  EXPECT_EQ(load_ok, 0);

  // Compare the loaded image with the saved one.
  EXPECT_EQ(static_cast<unsigned>(loaded_depth_image_res.x()), depth_image_width_);
  EXPECT_EQ(static_cast<unsigned>(loaded_depth_image_res.y()), depth_image_height_);
  EXPECT_EQ(memcmp(loaded_depth_image_data, depth_image_data_, depth_size_bytes_), 0);

  free(loaded_depth_image_data);
}

