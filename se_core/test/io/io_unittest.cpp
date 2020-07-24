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
#include <fstream>
#include <string>
#include <random>
#include "io/se_serialise.hpp"
#include "node.hpp"
#include "octree.hpp"
#include "gtest/gtest.h"

struct TestVoxelFullT {
  typedef float VoxelData;
  static inline VoxelData invalid(){ return 0.f; }
  static inline VoxelData initData(){ return 1.f; }

  using VoxelBlockType = se::VoxelBlockFull<TestVoxelFullT>;

  template <typename T>
  using MemoryPoolType = se::PagedMemoryPool<T>;
  template <typename BufferT>
  using MemoryBufferType = se::PagedMemoryBuffer<BufferT>;
};

struct OccupancyVoxelFullT {
  struct VoxelData {
    float x;
    double y;
  };
  static inline VoxelData invalid(){ return {0.f, 0.}; }
  static inline VoxelData initData(){ return {1.f, 0.}; }

  using VoxelBlockType = se::VoxelBlockFull<OccupancyVoxelFullT>;

  template <typename T>
  using MemoryPoolType = se::PagedMemoryPool<T>;
  template <typename BufferT>
  using MemoryBufferType = se::PagedMemoryBuffer<BufferT>;
};

TEST(SerialiseUnitTestFull, WriteReadNode) {
  std::string filename = "test.bin";
  {
    std::ofstream os (filename, std::ios::binary);
    se::Node<TestVoxelFullT> node;
    node.code(24);
    node.size_ = 256;
    for(int child_idx = 0; child_idx < 8; ++child_idx)
      node.data_[child_idx] =  5.f;
    se::internal::serialise(os, node);
  }

  {
    std::ifstream is(filename, std::ios::binary);
    se::Node<TestVoxelFullT> node;
    se::internal::deserialise(node, is);
    ASSERT_EQ(node.code(), 24);
    ASSERT_EQ(node.size_, 256);
    for(int child_idx = 0; child_idx < 8; ++child_idx)
      ASSERT_EQ(node.data_[child_idx], 5.f);
  }
}

TEST(SerialiseUnitTestFull, WriteReadBlock) {
  std::string filename = "test.bin";
  {
    std::ofstream os (filename, std::ios::binary);
    TestVoxelFullT::VoxelBlockType block;
    block.code(24);
    block.coordinates(Eigen::Vector3i(40, 48, 52));
    for(int voxel_idx = 0; voxel_idx < 512; ++voxel_idx)
      block.setData(voxel_idx, 5.f);
    se::internal::serialise(os, block);
  }

  {
    std::ifstream is(filename, std::ios::binary);
    TestVoxelFullT::VoxelBlockType block;
    se::internal::deserialise(block, is);
    ASSERT_EQ(block.code(), 24);
    ASSERT_TRUE(block.coordinates() == Eigen::Vector3i(40, 48, 52));
    for(int voxel_idx = 0; voxel_idx < 512; ++voxel_idx)
      ASSERT_EQ(block.data(voxel_idx), 5.f);
  }
}

TEST(SerialiseUnitTestFull, WriteReadBlockStruct) {
  std::string filename = "test.bin";
  {
    std::ofstream os (filename, std::ios::binary);
    OccupancyVoxelFullT::VoxelBlockType block;
    block.code(24);
    block.coordinates(Eigen::Vector3i(40, 48, 52));
    for(int voxel_idx = 0; voxel_idx < 512; ++voxel_idx)
      block.setData(voxel_idx, {5.f, 2.});
    se::internal::serialise(os, block);
  }

  {
    std::ifstream is(filename, std::ios::binary);
    OccupancyVoxelFullT::VoxelBlockType block;
    se::internal::deserialise(block, is);
    ASSERT_EQ(block.code(), 24);
    ASSERT_TRUE(block.coordinates() == Eigen::Vector3i(40, 48, 52));
    for(int voxel_idx = 0; voxel_idx < 512; ++voxel_idx) {
      auto data = block.data(voxel_idx);
      ASSERT_EQ(data.x, 5.f);
      ASSERT_EQ(data.y, 2.);
    }
  }
}

TEST(SerialiseUnitTestFull, SerialiseTree) {
  se::Octree<TestVoxelFullT> octree;
  const int block_depth = octree.blockDepth();
  int   size = 1024;
  float dim  = 10.f;
  octree.init(size, dim);
  std::mt19937 gen(1); //Standard mersenne_twister_engine seeded with constant
  std::uniform_int_distribution<> dis(0, size - 1);

  int num_tested = 0;
  for(int i = 1, size = octree.size() / 2; i <= block_depth; ++i, size = size / 2) {
    for(int j = 0; j < 20; ++j) {
      Eigen::Vector3i voxel_coord(dis(gen), dis(gen), dis(gen));
      octree.insert(voxel_coord.x(), voxel_coord.y(), voxel_coord.z(), i);
    }
  }
  std::string filename = "octree-test.bin";
  octree.save(filename);

  se::Octree<TestVoxelFullT> octree_copy;
  octree_copy.load(filename);

  ASSERT_EQ(octree.size(), octree_copy.size());
  ASSERT_EQ(octree.dim(), octree_copy.dim());

  auto& node_buffer_base = octree.pool().nodeBuffer();
  auto& node_buffer_copy = octree_copy.pool().nodeBuffer();
  ASSERT_EQ(node_buffer_base.size(), node_buffer_copy.size());
  for(int i = 0; i < node_buffer_base.size(); ++i) {
    se::Node<TestVoxelFullT> * node_base  = node_buffer_base[i];
    se::Node<TestVoxelFullT> * node_copy = node_buffer_copy[i];
    ASSERT_EQ(node_base->code(), node_copy->code());
    ASSERT_EQ(node_base->children_mask_, node_copy->children_mask_);
  }

  auto& block_buffer_base = octree.pool().blockBuffer();
  auto& block_buffer_copy = octree_copy.pool().blockBuffer();
  ASSERT_EQ(block_buffer_base.size(), block_buffer_copy.size());
}

TEST(SerialiseUnitTestFull, SerialiseBlock) {
  se::Octree<TestVoxelFullT> octree;
  int   size = 1024;
  float dim  = 10.f;
  octree.init(size, dim);
  std::mt19937 gen(1); //Standard mersenne_twister_engine seeded with constant
  std::uniform_int_distribution<> dis(0, size - 1);

  int num_tested = 0;
  for (int j = 0; j < 20; ++j) {
    Eigen::Vector3i voxel_coord(dis(gen), dis(gen), dis(gen));
    octree.insert(voxel_coord.x(), voxel_coord.y(), voxel_coord.z(), octree.blockDepth());
    auto block = octree.fetch(voxel_coord.x(), voxel_coord.y(), voxel_coord.z());
    for (int voxel_idx = 0; voxel_idx < TestVoxelFullT::VoxelBlockType::size_cu; ++voxel_idx)
      block->setData(voxel_idx, dis(gen));
  }

  std::string filename = "block-test.bin";
  octree.save(filename);

  se::Octree<TestVoxelFullT> octree_copy;
  octree_copy.load(filename);

  auto &block_buffer_base = octree.pool().blockBuffer();
  auto &block_buffer_copy = octree_copy.pool().blockBuffer();
  for (int i = 0; i < block_buffer_base.size(); i++) {
    for (int voxel_idx = 0; voxel_idx < TestVoxelFullT::VoxelBlockType::size_cu; voxel_idx++) {
      ASSERT_EQ(block_buffer_base[i]->data(voxel_idx), block_buffer_copy[i]->data(voxel_idx));
    }
  }
}

struct TestVoxelSingleT {
  typedef float VoxelData;
  static inline VoxelData invalid(){ return 0.f; }
  static inline VoxelData initData(){ return 1.f; }

  using VoxelBlockType = se::VoxelBlockSingle<TestVoxelSingleT>;

  template <typename T>
  using MemoryPoolType = se::PagedMemoryPool<T>;
  template <typename BufferT>
  using MemoryBufferType = se::PagedMemoryBuffer<BufferT>;
};

struct OccupancyVoxelSingleT {
  struct VoxelData {
    float x;
    double y;
  };
  static inline VoxelData invalid(){ return {0.f, 0.}; }
  static inline VoxelData initData(){ return {1.f, 0.}; }

  using VoxelBlockType = se::VoxelBlockSingle<OccupancyVoxelSingleT>;

  template <typename T>
  using MemoryPoolType = se::PagedMemoryPool<T>;
  template <typename BufferT>
  using MemoryBufferType = se::PagedMemoryBuffer<BufferT>;
};

TEST(SerialiseUnitTestSingle, WriteReadNode) {
  std::string filename = "test.bin";
  {
    std::ofstream os (filename, std::ios::binary);
    se::Node<TestVoxelSingleT> node;
    node.code(24);
    node.size_ = 256;
    for(int child_idx = 0; child_idx < 8; ++child_idx)
      node.data_[child_idx] =  5.f;
    se::internal::serialise(os, node);
  }

  {
    std::ifstream is(filename, std::ios::binary);
    se::Node<TestVoxelSingleT> node;
    se::internal::deserialise(node, is);
    ASSERT_EQ(node.code(), 24);
    ASSERT_EQ(node.size_, 256);
    for(int child_idx = 0; child_idx < 8; ++child_idx)
      ASSERT_EQ(node.data_[child_idx], 5.f);
  }
}

TEST(SerialiseUnitTestSingle, WriteReadBlock) {
  std::string filename = "test.bin";
  {
    std::ofstream os (filename, std::ios::binary);
    TestVoxelSingleT::VoxelBlockType block;
    block.code(24);
    block.coordinates(Eigen::Vector3i(40, 48, 52));
    block.allocateDownTo(0);
    for(int voxel_idx = 0; voxel_idx < 512; ++voxel_idx)
      block.setData(voxel_idx, 5.f);
    se::internal::serialise(os, block);
  }

  {
    std::ifstream is(filename, std::ios::binary);
    TestVoxelSingleT::VoxelBlockType block;
    se::internal::deserialise(block, is);
    ASSERT_EQ(block.code(), 24);
    ASSERT_TRUE(block.coordinates() == Eigen::Vector3i(40, 48, 52));
    for(int voxel_idx = 0; voxel_idx < 512; ++voxel_idx)
      ASSERT_EQ(block.data(voxel_idx), 5.f);
  }
}

TEST(SerialiseUnitTestSingle, WriteReadBlockStruct) {
  std::string filename = "test.bin";
  {
    std::ofstream os (filename, std::ios::binary);
    OccupancyVoxelSingleT::VoxelBlockType block;
    block.code(24);
    block.coordinates(Eigen::Vector3i(40, 48, 52));
    block.allocateDownTo(0);
    for(int voxel_idx = 0; voxel_idx < 512; ++voxel_idx)
      block.setData(voxel_idx, {5.f, 2.});
    se::internal::serialise(os, block);
  }

  {
    std::ifstream is(filename, std::ios::binary);
    OccupancyVoxelSingleT::VoxelBlockType block;
    se::internal::deserialise(block, is);
    ASSERT_EQ(block.code(), 24);
    ASSERT_TRUE(block.coordinates() == Eigen::Vector3i(40, 48, 52));
    for(int voxel_idx = 0; voxel_idx < 512; ++voxel_idx) {
      auto data = block.data(voxel_idx);
      ASSERT_EQ(data.x, 5.f);
      ASSERT_EQ(data.y, 2.);
    }
  }
}

TEST(SerialiseUnitTestSingle, SerialiseTree) {
  se::Octree<TestVoxelSingleT> octree;
  const int block_depth = octree.blockDepth();
  int   size = 1024;
  float dim  = 10.f;
  octree.init(size, dim);
  std::mt19937 gen(1); //Standard mersenne_twister_engine seeded with constant
  std::uniform_int_distribution<> dis(0, size - 1);
  
  int num_tested = 0;
  for(int i = 1, size = octree.size() / 2; i <= block_depth; ++i, size = size / 2) {
    for(int j = 0; j < 20; ++j) {
      Eigen::Vector3i voxel_coord(dis(gen), dis(gen), dis(gen));
      octree.insert(voxel_coord.x(), voxel_coord.y(), voxel_coord.z(), i);
    }
  }
  std::string filename = "octree-test.bin";
  octree.save(filename);

  se::Octree<TestVoxelSingleT> octree_copy;
  octree_copy.load(filename);

  ASSERT_EQ(octree.size(), octree_copy.size());
  ASSERT_EQ(octree.dim(), octree_copy.dim());

  auto& node_buffer_base = octree.pool().nodeBuffer();
  auto& node_buffer_copy = octree_copy.pool().nodeBuffer();
  ASSERT_EQ(node_buffer_base.size(), node_buffer_copy.size());
  for(int i = 0; i < node_buffer_base.size(); ++i) {
    se::Node<TestVoxelSingleT> * node_base  = node_buffer_base[i];
    se::Node<TestVoxelSingleT> * node_copy = node_buffer_copy[i];
    ASSERT_EQ(node_base->code(), node_copy->code());
    ASSERT_EQ(node_base->children_mask_, node_copy->children_mask_);
  }

  auto& block_buffer_base = octree.pool().blockBuffer();
  auto& block_buffer_copy = octree_copy.pool().blockBuffer();
  ASSERT_EQ(block_buffer_base.size(), block_buffer_copy.size());
}

TEST(SerialiseUnitTestSingle, SerialiseBlock) {
  se::Octree<TestVoxelSingleT> octree;
  int   size = 1024;
  float dim  = 10.f;
  octree.init(size, dim);
  std::mt19937 gen(1); //Standard mersenne_twister_engine seeded with constant
  std::uniform_int_distribution<> dis(0, size - 1);

  int num_tested = 0;
  for (int j = 0; j < 20; ++j) {
    Eigen::Vector3i voxel_coord(dis(gen), dis(gen), dis(gen));
    octree.insert(voxel_coord.x(), voxel_coord.y(), voxel_coord.z(), octree.blockDepth());
    auto block = octree.fetch(voxel_coord.x(), voxel_coord.y(), voxel_coord.z());
    block->allocateDownTo();
    for (int voxel_idx = 0; voxel_idx < TestVoxelSingleT::VoxelBlockType::size_cu; ++voxel_idx)
      block->setData(voxel_idx, dis(gen));
  }
  
  std::string filename = "block-test.bin";
  octree.save(filename);

  se::Octree<TestVoxelSingleT> octree_copy;
  octree_copy.load(filename);

  auto &block_buffer_base = octree.pool().blockBuffer();
  auto &block_buffer_copy = octree_copy.pool().blockBuffer();
  for (int i = 0; i < block_buffer_base.size(); i++) {
    for (int voxel_idx = 0; voxel_idx < TestVoxelSingleT::VoxelBlockType::size_cu; voxel_idx++) {
      ASSERT_EQ(block_buffer_base[i]->data(voxel_idx), block_buffer_copy[i]->data(voxel_idx));
    }
  }
}
