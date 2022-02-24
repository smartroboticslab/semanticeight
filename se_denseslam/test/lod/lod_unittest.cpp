// SPDX-FileCopyrightText: 2022 Smart Robotics Lab
// SPDX-FileCopyrightText: 2022 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include <iostream>
#include <se/lod.hpp>
#include <vector>



struct TestData {
    int8_t min_scale;
    int8_t expected_scale;
    int8_t desired_scale;
    int8_t gain;
};

std::ostream& operator<<(std::ostream& os, const TestData& d)
{
    os << "block_min_scale: " << static_cast<int>(d.min_scale)
       << "  block_expected_scale: " << static_cast<int>(d.expected_scale)
       << "  desired_scale: " << static_cast<int>(d.desired_scale)
       << " -> gain: " << static_cast<int>(d.gain);
    return os;
}



TEST(LoD, zeroScaleGain)
{
    constexpr int8_t max_scale = VoxelImpl::VoxelBlockType::max_scale;
    // Generate all possible scale values.
    std::vector<int8_t> scales(max_scale + 1);
    std::iota(scales.begin(), scales.end(), 0);
    // Generate and test all input values that give a gain of 0.
    std::vector<TestData> data;
    for (auto expected_scale : scales) {
        for (auto desired_scale : scales) {
            for (int8_t min_scale = 0; min_scale <= desired_scale; ++min_scale) {
                data.push_back({min_scale, expected_scale, desired_scale, 0});
            }
        }
    }
    for (const auto& d : data) {
        EXPECT_FLOAT_EQ(se::scale_gain(d.min_scale, d.expected_scale, d.desired_scale), d.gain);
    }
}

TEST(LoD, scaleGain)
{
    constexpr int8_t max_scale = VoxelImpl::VoxelBlockType::max_scale;
    const std::vector<TestData> data{
        // Desired scale 0
        {1, 0, 0, 1},
        {2, 0, 0, 2},
        {2, 1, 0, 1},
        {max_scale, 0, 0, max_scale},
        {max_scale, 1, 0, 2},
        {max_scale, 2, 0, 1},
        // Desired scale 1
        {2, 0, 1, 1},
        {2, 1, 1, 1},
        {max_scale, 0, 1, 2},
        {max_scale, 1, 1, 2},
        {max_scale, 2, 1, 1},
        // Desired scale 2
        {max_scale, 0, 2, 1},
        {max_scale, 1, 2, 1},
        {max_scale, 2, 2, 1},
    };
    for (const auto& d : data) {
        EXPECT_FLOAT_EQ(se::scale_gain(d.min_scale, d.expected_scale, d.desired_scale), d.gain);
    }
}
