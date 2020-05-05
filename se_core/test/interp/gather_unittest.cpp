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
#include "octree.hpp"
#include "utils/math_utils.h"
#include "interpolation/interp_gather.hpp"
#include "gtest/gtest.h"
#include "node_iterator.hpp"

struct TestVoxelT {
  typedef float VoxelData;
  static inline VoxelData empty(){ return 0.f; }
  static inline VoxelData initData(){ return 1.f; }

  template <typename T>
  using MemoryPoolType = se::PagedMemoryPool<T>;
  template <typename BufferT>
  using MemoryBufferType = se::PagedMemoryBuffer<BufferT>;
};

class GatherTest : public ::testing::Test {
  protected:
    virtual void SetUp() {
      octree_.init(512, 5);
      const Eigen::Vector3i blocks_coord[10] = {{56, 12, 254}, {87, 32, 423}, {128, 128, 128},
      {136, 128, 128}, {128, 136, 128}, {136, 136, 128},
      {128, 128, 136}, {136, 128, 136}, {128, 136, 136}, {136, 136, 136}};
      se::key_t allocation_list[10];
      for(int i = 0; i < 10; ++i) {
        allocation_list[i] = octree_.hash(blocks_coord[i].x(), blocks_coord[i].y(), blocks_coord[i].z());
      }
      octree_.allocate(allocation_list, 10);
    }

  typedef se::Octree<TestVoxelT> OctreeF;
  OctreeF octree_;
};

TEST_F(GatherTest, Init) {
  EXPECT_EQ(octree_.get(137, 138, 130), TestVoxelT::initData());
}

TEST_F(GatherTest, GatherLocal) {
  TestVoxelT::VoxelData points[8];
  const Eigen::Vector3i base = {136, 128, 136};
  se::internal::gather_values(octree_, base, 1, [](const auto& data){ return data; }, points);

  for(int i = 0; i < 8; ++i) {
    EXPECT_EQ(points[i], TestVoxelT::initData());
  }
}

TEST_F(GatherTest, ZCrosses) {
  TestVoxelT::VoxelData points[8];
  const unsigned block_size = se::VoxelBlock<TestVoxelT>::side;
  const Eigen::Vector3i base = {132, 128, 135};
  unsigned int crossmask = ((base.x() % block_size) == block_size - 1 << 2) |
                           ((base.y() % block_size) == block_size - 1 << 1) |
                            (base.z() % block_size) == block_size - 1;
  ASSERT_EQ(crossmask, 1);
  se::VoxelBlock<TestVoxelT>* block = octree_.fetch(base.x(), base.y(), base.z());
  se::internal::gather_values(octree_, base, 0, [](const auto& data){ return data; }, points);

  for(int i = 0; i < 8; ++i) {
    EXPECT_EQ(points[i], TestVoxelT::initData());
  }
}

TEST_F(GatherTest, YCrosses) {
  TestVoxelT::VoxelData points[8];
  const unsigned block_size = se::VoxelBlock<TestVoxelT>::side;
  const Eigen::Vector3i base = {132, 135, 132};
  unsigned int crossmask = ((base.x() % block_size == block_size - 1) << 2) |
                           ((base.y() % block_size == block_size - 1) << 1) |
                            ((base.z() % block_size) == block_size - 1);
  ASSERT_EQ(crossmask, 2);
  se::VoxelBlock<TestVoxelT>* block = octree_.fetch(base.x(), base.y(), base.z());
  se::internal::gather_values(octree_, base, 0, [](const auto& data){ return data; }, points);

  for(int i = 0; i < 8; ++i) {
    EXPECT_EQ(points[i], TestVoxelT::initData());
  }
}

TEST_F(GatherTest, XCrosses) {
  TestVoxelT::VoxelData points[8];
  const unsigned block_size = se::VoxelBlock<TestVoxelT>::side;
  const Eigen::Vector3i base = {135, 132, 132};
  unsigned int crossmask = ((base.x() % block_size == block_size - 1) << 2) |
                           ((base.y() % block_size == block_size - 1) << 1) |
                            ((base.z() % block_size) == block_size - 1);
  ASSERT_EQ(crossmask, 4);
  se::VoxelBlock<TestVoxelT>* block = octree_.fetch(base.x(), base.y(), base.z());
  se::internal::gather_values(octree_, base, 0, [](const auto& data){ return data; }, points);

  for(int i = 0; i < 8; ++i) {
    EXPECT_EQ(points[i], TestVoxelT::initData());
  }
}

TEST_F(GatherTest, YZCross) {
  TestVoxelT::VoxelData points[8];
  const unsigned block_size = se::VoxelBlock<TestVoxelT>::side;
  const Eigen::Vector3i base = {129, 135, 135};
  unsigned int crossmask = ((base.x() % block_size == block_size - 1) << 2) |
                           ((base.y() % block_size == block_size - 1) << 1) |
                            ((base.z() % block_size) == block_size - 1);
  ASSERT_EQ(crossmask, 3);
  se::VoxelBlock<TestVoxelT>* block = octree_.fetch(base.x(), base.y(), base.z());
  se::internal::gather_values(octree_, base, 0, [](const auto& data){ return data; }, points);

  for(int i = 0; i < 8; ++i) {
    EXPECT_EQ(points[i], TestVoxelT::initData());
  }
}

TEST_F(GatherTest, XZCross) {
  TestVoxelT::VoxelData points[8];
  const unsigned block_size = se::VoxelBlock<TestVoxelT>::side;
  const Eigen::Vector3i base = {135, 131, 135};
  unsigned int crossmask = ((base.x() % block_size == block_size - 1) << 2) |
                           ((base.y() % block_size == block_size - 1) << 1) |
                            ((base.z() % block_size) == block_size - 1);
  ASSERT_EQ(crossmask, 5);
  se::VoxelBlock<TestVoxelT>* block = octree_.fetch(base.x(), base.y(), base.z());
  se::internal::gather_values(octree_, base, 0, [](const auto& data){ return data; }, points);

  for(int i = 0; i < 8; ++i) {
    EXPECT_EQ(points[i], TestVoxelT::initData());
  }
}

TEST_F(GatherTest, XYCross) {
  TestVoxelT::VoxelData points[8];
  const unsigned block_size = se::VoxelBlock<TestVoxelT>::side;
  const Eigen::Vector3i base = {135, 135, 138};
  unsigned int crossmask = ((base.x() % block_size == block_size - 1) << 2) |
                           ((base.y() % block_size == block_size - 1) << 1) |
                            ((base.z() % block_size) == block_size - 1);
  ASSERT_EQ(crossmask, 6);
  se::VoxelBlock<TestVoxelT>* block = octree_.fetch(base.x(), base.y(), base.z());
  se::internal::gather_values(octree_, base, 0, [](const auto& data){ return data; }, points);

  for(int i = 0; i < 8; ++i) {
    EXPECT_EQ(points[i], TestVoxelT::initData());
  }
}

TEST_F(GatherTest, AllCross) {
  TestVoxelT::VoxelData points[8];
  const unsigned block_size = se::VoxelBlock<TestVoxelT>::side;
  const Eigen::Vector3i base = {135, 135, 135};
  unsigned int crossmask = ((base.x() % block_size == block_size - 1) << 2) |
                           ((base.y() % block_size == block_size - 1) << 1) |
                            ((base.z() % block_size) == block_size - 1);
  ASSERT_EQ(crossmask, 7);
  se::VoxelBlock<TestVoxelT>* block = octree_.fetch(base.x(), base.y(), base.z());
  se::internal::gather_values(octree_, base, 0, [](const auto& data){ return data; }, points);

  for(int i = 0; i < 8; ++i) {
    EXPECT_EQ(points[i], TestVoxelT::initData());
  }
}

TEST_F(GatherTest, FetchWithStack) {
  se::Node<TestVoxelT>* stack[CAST_STACK_DEPTH] = {nullptr};
  int voxel_depth = octree_.voxelDepth();
  se::internal::fetch(stack, octree_.root(), voxel_depth,
      Eigen::Vector3i(128, 136, 128));
  auto res = octree_.fetch(128, 136, 128);
  int l = 0;
  while(stack[l] && stack[l + 1] != nullptr) {
    ASSERT_TRUE(se::parent(stack[l + 1]->code_, voxel_depth) == stack[l]->code_);
    ++l;
  }
}

TEST_F(GatherTest, FetchNeighbourhood) {
  Eigen::Vector3i voxel_coord_0(128,   0, 0); // (1, 0, 0)
  Eigen::Vector3i voxel_coord_1(0,   128, 0); // (0, 1, 0)
  Eigen::Vector3i voxel_coord_2(128, 128, 0); // (1, 1, 0)
  Eigen::Vector3i base(64,   64, 0);
  octree_.insert(voxel_coord_0.x(), voxel_coord_0.y(), voxel_coord_0.z(), 2);
  octree_.insert(voxel_coord_1.x(), voxel_coord_1.y(), voxel_coord_1.z(), 2);
  octree_.insert(voxel_coord_2.x(), voxel_coord_2.y(), voxel_coord_2.z(), 2);
  octree_.insert(base.x(), base.y(), base.z(), 3);

  se::Node<TestVoxelT>* stack[CAST_STACK_DEPTH] = {nullptr};
  int voxel_depth = octree_.voxelDepth();
  auto base_ptr = se::internal::fetch(stack, octree_.root(), voxel_depth, base);
  auto n1 = se::internal::fetch_neighbour(stack, base_ptr, voxel_depth, 1);
  auto n2 = se::internal::fetch_neighbour(stack, base_ptr, voxel_depth, 2);
  auto n3 = se::internal::fetch_neighbour(stack, base_ptr, voxel_depth, 3);
  ASSERT_TRUE(se::keyops::decode(base_ptr->code_) == base);
  ASSERT_TRUE(se::keyops::decode(n1->code_) == voxel_coord_0);
  ASSERT_TRUE(se::keyops::decode(n2->code_) == voxel_coord_1);
  ASSERT_TRUE(se::keyops::decode(n3->code_) == voxel_coord_2);

  std::fill(std::begin(stack), std::end(stack), nullptr);
  base_ptr = se::internal::fetch(stack, octree_.root(), voxel_depth, voxel_coord_0);
  auto failed = se::internal::fetch_neighbour(stack, base_ptr, voxel_depth, 1);
  ASSERT_TRUE(failed == nullptr);
}
