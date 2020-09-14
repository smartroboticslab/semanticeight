// SPDX-FileCopyrightText: 2020 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include <se/node.hpp>



struct TestVoxelT {
  typedef struct VoxelData {
    int x;
    int y;

    bool operator==(const VoxelData& other) const {
      return (x == other.x) && (y == other.y);
    }

    bool operator!=(const VoxelData& other) const {
      return !(*this == other);
    }
  } VoxelData;

  static inline VoxelData invalid(){ return {0, 0}; }
  static inline VoxelData initData(){ return {1, 0}; }
};



class VoxelBlockCommonTest : public ::testing::Test {
  protected:
    virtual void SetUp() {
      // Initialize the VoxelBlocks
      block_full_.size(se::VoxelBlock<TestVoxelT>::size_li);
      block_single_.size(se::VoxelBlock<TestVoxelT>::size_li);
      // Compute the number of voxels per scale
      for (int scale = 0; scale < num_scales_; ++scale) {
        const int scale_size_li = se::VoxelBlock<TestVoxelT>::size_li / (1 << scale);
        voxels_per_scale_.push_back(scale_size_li * scale_size_li * scale_size_li);
      }
    }

    se::VoxelBlockFull<TestVoxelT> block_full_;
    se::VoxelBlockSingle<TestVoxelT> block_single_;
    static constexpr int num_scales_ = se::VoxelBlock<TestVoxelT>::max_scale + 1;
    std::vector<int> voxels_per_scale_;
};



TEST_F(VoxelBlockCommonTest, setAndGetData) {
  // Set data at all scales
  for (int scale = 0; scale < num_scales_; ++scale) {
    for (int voxel_idx = 0; voxel_idx < voxels_per_scale_[scale]; ++voxel_idx) {
      const TestVoxelT::VoxelData data {scale, 1};
      block_full_.setData(voxel_idx, scale, data);
      block_single_.setDataSafe(voxel_idx, scale, data);
    }
  }
  // Read the data using the VoxelBlock method
  for (int scale = 0; scale < num_scales_; ++scale) {
    for (int voxel_idx = 0; voxel_idx < voxels_per_scale_[scale]; ++voxel_idx) {
      const TestVoxelT::VoxelData data {scale, 1};
      EXPECT_EQ(block_full_.data(voxel_idx, scale), data);
      EXPECT_EQ(block_single_.data(voxel_idx, scale), data);
    }
  }
  // Read the data using a linear index
  int scale_offset = 0;
  for (int scale = 0; scale < num_scales_; ++scale) {
    for (int voxel_idx = 0; voxel_idx < voxels_per_scale_[scale]; ++voxel_idx) {
      const TestVoxelT::VoxelData data {scale, 1};
      EXPECT_EQ(block_full_.data(scale_offset + voxel_idx), data);
      EXPECT_EQ(block_single_.data(scale_offset + voxel_idx), data);
    }
    scale_offset += voxels_per_scale_[scale];
  }
}

