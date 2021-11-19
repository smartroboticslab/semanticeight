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

#include "se/depth_utils.hpp"

#include <Eigen/Dense>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>



class DepthImageErrorTest : public ::testing::Test {
    protected:
    void SetUp() override
    {
        // Initialize the depth images.
        const float depth_1_vals[img_size] = {0.0f, 0.0f, 1.0f, 1.5f};
        const float depth_2_vals[img_size] = {0.0f, 1.0f, 0.0f, 1.0f};
        for (size_t p = 0; p < depth_1.size(); ++p) {
            depth_1[p] = depth_1_vals[p];
            depth_2[p] = depth_2_vals[p];
        }

        // Initialize the masks.
        mask_all = cv::Mat::ones(cv::Size(width, height), CV_8UC1);
        mask_subset = cv::Mat::zeros(cv::Size(width, height), CV_8UC1);
        mask_invalid = cv::Mat::ones(cv::Size(width, height), CV_8UC1);
        mask_subset.at<uint8_t>(0) = 1;
        mask_subset.at<uint8_t>(1) = 1;
        mask_subset.at<uint8_t>(3) = 1;
        mask_invalid.at<uint8_t>(3) = 0;
    }

    static constexpr int width = 2;
    static constexpr int height = 2;
    static constexpr int img_size = width * height;
    se::Image<float> depth_1 = se::Image<float>(width, height, 0.f);
    se::Image<float> depth_2 = se::Image<float>(width, height, 0.f);
    cv::Mat mask_all;
    cv::Mat mask_subset;
    cv::Mat mask_invalid;
};



TEST_F(DepthImageErrorTest, ErrorAll)
{
    const float e = depth_image_error(depth_1, depth_2, mask_all);

    EXPECT_FLOAT_EQ(e, 0.5f * 0.5f);
}



TEST_F(DepthImageErrorTest, ErrorSubset)
{
    const float e = depth_image_error(depth_1, depth_2, mask_subset);

    EXPECT_FLOAT_EQ(e, 0.5f * 0.5f);
}



TEST_F(DepthImageErrorTest, ErrorInvalid)
{
    const float e = depth_image_error(depth_1, depth_2, mask_invalid);

    EXPECT_FLOAT_EQ(e, -1.f);
}
