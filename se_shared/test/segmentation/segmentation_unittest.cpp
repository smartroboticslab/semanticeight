/*
 * Copyright 2019 Sotiris Papatheodorou, Imperial College London
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "se/segmentation.hpp"



class FreeFunctionTesting : public ::testing::Test {
  protected:
    virtual void SetUp() {
      mask_1 = cv::Mat::zeros(mask_size, se::mask_t);
      mask_2 = cv::Mat::zeros(mask_size, se::mask_t);

      // Initialize an instance mask to:
      // 0 1
      // 2 3
      instance_mask = cv::Mat(mask_size, se::instance_mask_t, se::instance_bg);
      for (int i = 0; i < instance_mask.cols * instance_mask.rows; i++) {
        instance_mask.at<se::instance_mask_elem_t>(i) = i;
      }
    }

    const int mask_w = 2;
    const int mask_h = 2;
    const cv::Size mask_size = cv::Size(mask_w, mask_h);
    cv::Mat mask_1;
    cv::Mat mask_2;
    cv::Mat instance_mask;
};



TEST_F(FreeFunctionTesting, ExtractInstance) {
  for (int i = 0; i < instance_mask.cols * instance_mask.rows; i++) {
    // Extract the individual instance mask for each instance ID.
    const cv::Mat individual_mask = se::extract_instance(instance_mask, i);

    // Test each pixel of the individual instance mask has the correct value.
    for (int j = 0; j < individual_mask.cols * individual_mask.rows; j++) {
      if (j == i) {
        EXPECT_EQ(individual_mask.at<se::mask_elem_t>(j), 255);
      } else {
        EXPECT_EQ(individual_mask.at<se::mask_elem_t>(j), 0);
      }
    }
  }
}

