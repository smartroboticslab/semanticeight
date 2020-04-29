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
#include <random>
#include "octree.hpp"
#include "utils/math_utils.h"
#include "utils/morton_utils.hpp"
#include "gtest/gtest.h"

struct TestVoxelT {
  typedef float VoxelData;
  static inline VoxelData empty(){ return 0.f; }
  static inline VoxelData initValue(){ return 0.f; }

  template <typename T>
  using MemoryPoolType = se::PagedMemoryPool<T>;
  template <typename BufferT>
  using MemoryBufferType = se::PagedMemoryBuffer<BufferT>;
};

TEST(AllocationTest, EmptySingleVoxel) {
  typedef se::Octree<TestVoxelT> OctreeF;
  OctreeF octree;
  octree.init(256, 5);
  const Eigen::Vector3i vox = {25, 65, 127};
  const se::key_t code = octree.hash(vox(0), vox(1), vox(2));
  se::key_t allocation_list[1] = {code};
  const TestVoxelT::VoxelData val = octree.get(vox(0), vox(1), vox(2));
  EXPECT_EQ(val, TestVoxelT::empty());
}

TEST(AllocationTest, SetSingleVoxel) {
  typedef se::Octree<TestVoxelT> OctreeF;
  OctreeF octree;
  octree.init(256, 5);
  const Eigen::Vector3i vox = {25, 65, 127};
  const se::key_t code = octree.hash(vox(0), vox(1), vox(2));
  se::key_t allocation_list[1] = {code};
  octree.allocate(allocation_list, 1);

  se::VoxelBlock<TestVoxelT> * block = octree.fetch(vox(0), vox(1), vox(2));
  TestVoxelT::VoxelData written_val = 2.f;
  block->data(vox, written_val);

  const TestVoxelT::VoxelData read_val = octree.get(vox(0), vox(1), vox(2));
  EXPECT_EQ(written_val, read_val);
}

TEST(AllocationTest, FetchOctant) {
  typedef se::Octree<TestVoxelT> OctreeF;
  OctreeF octree;
  const int voxel_depth = 8;
  const unsigned int block_side = 8;
  const unsigned int size = std::pow(2, voxel_depth);
  octree.init(size, 5);
  const Eigen::Vector3i vox = {25, 65, 127};
  const se::key_t code = octree.hash(vox(0), vox(1), vox(2));
  se::key_t allocation_list[1] = {code};
  octree.allocate(allocation_list, 1);

  const int level = 3; /* 32 voxels per side */
  se::Node<TestVoxelT> * node = octree.fetch_octant(vox(0), vox(1), vox(2), level);
  se::key_t fetched_code = node->code_;

  const se::key_t gt_code = octree.hash(vox(0), vox(1), vox(2), level);
  ASSERT_EQ(fetched_code, gt_code);
}

TEST(AllocationTest, MortonPrefixMask) {

  const unsigned int max_bits = 21;
  const unsigned int block_side = 8;
  const unsigned int size = std::pow(2, max_bits);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(0, size);

  constexpr int num_samples = 10;
  se::key_t keys[num_samples];
  se::key_t tempkeys[num_samples];
  Eigen::Vector3i coordinates[num_samples];

  for(int i = 0; i < num_samples; ++i) {
    const Eigen::Vector3i vox = {dis(gen), dis(gen), dis(gen)};
    coordinates[i] = Eigen::Vector3i(vox);
    const se::key_t code = compute_morton(vox(0), vox(1), vox(2));
    keys[i] = code;
  }

  const int voxel_depth = log2(size);
  const int block_depth = voxel_depth - log2(block_side);
  const unsigned int shift = max_bits - voxel_depth;
  int edge = size/2;
  for (int level = 0; level <= block_depth; level++){
    const se::key_t mask = MASK[level + shift];
    compute_prefix(keys, tempkeys, num_samples, mask);
    for(int i = 0; i < num_samples; ++i) {
      const Eigen::Vector3i masked_vox = unpack_morton(tempkeys[i]);
      ASSERT_EQ(masked_vox(0) % edge, 0);
      ASSERT_EQ(masked_vox(1) % edge, 0);
      ASSERT_EQ(masked_vox(2) % edge, 0);
      const Eigen::Vector3i vox = coordinates[i];
      // printf("vox: %d, %d, %d\n", vox(0), vox(1), vox(2));
      // printf("masked level %d: %d, %d, %d\n", level, masked_vox(0), masked_vox(1), masked_vox(2) );
    }
    edge = edge/2;
  }
}

TEST(AllocationTest, ParentInsert) {
  se::Octree<TestVoxelT> octree;
  const unsigned int block_side = 8;
  const int voxel_depth = 8;
  const unsigned int size = std::pow(2, voxel_depth);
  octree.init(size, 5);
  const int block_depth = octree.blockDepth();
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(0, size);
  const Eigen::Vector3i vox = {dis(gen), dis(gen), dis(gen)};
  octree.insert(vox(0), vox(1), vox(2));
  se::VoxelBlock<TestVoxelT> * block = octree.fetch(vox(0), vox(1), vox(2));
  EXPECT_NE(block, nullptr);
  se::Node<TestVoxelT> * parent_node = block->parent();
  for(int level = block_depth - 1; level >= 0; level--){
    se::Node<TestVoxelT> * node = octree.fetch_octant(vox(0), vox(1), vox(2), level);
    ASSERT_EQ(parent_node, node);
    parent_node = parent_node->parent();
  }
}

TEST(AllocationTest, ParentAllocation) {
  se::Octree<TestVoxelT> octree;
  const unsigned int block_side = 8;
  const int voxel_depth = 8;
  const unsigned int size = std::pow(2, voxel_depth);
  octree.init(size, 5);  std::random_device rd;
  const int block_depth = octree.blockDepth();
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(0, size);
  const Eigen::Vector3i vox = {dis(gen), dis(gen), dis(gen)};
  const unsigned code = octree.hash(vox(0), vox(1), vox(2));
  se::key_t allocation_list[1] = {code};
  octree.allocate(allocation_list, 1);

  se::VoxelBlock<TestVoxelT> * block = octree.fetch(vox(0), vox(1), vox(2));
  EXPECT_NE(block, nullptr);
  se::Node<TestVoxelT> * parent_node = block->parent();
  for(int level = block_depth - 1; level >= 0; level--){
    se::Node<TestVoxelT> * node = octree.fetch_octant(vox(0), vox(1), vox(2), level);
    ASSERT_EQ(parent_node, node);
    parent_node = parent_node->parent();
  }
}
