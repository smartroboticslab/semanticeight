/*
  Copyright 2016 Nils Funk, Imperial College London
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
#include "se/algorithms/meshing.hpp"
#include "se/io/meshing_io.hpp"
#include "gtest/gtest.h"

struct TestVoxelT {
  struct VoxelData {
    float x;
    float y;
  };
  static inline VoxelData invalid(){ return {0.f, 0.f}; }
  static inline VoxelData initData(){ return {-1.f, 1.f}; }

  using VoxelBlockType = se::VoxelBlockFull<TestVoxelT>;

  using MemoryPoolType = se::PagedMemoryPool<TestVoxelT>;
  template <typename BufferT>
  using MemoryBufferType = se::PagedMemoryBuffer<BufferT>;
};

using VoxelBlockType = TestVoxelT::VoxelBlockType;

TEST(MeshingTest, EqualScaleNeighbour) {
  typedef se::Octree<TestVoxelT> OctreeF;
  OctreeF octree;
  octree.init(32, 32);
  se::key_t allocation_list[27];
  int list_idx = 0;
  std::vector<Eigen::Vector3i> block_coords;
  for (int x = 0; x <= 16; x += VoxelBlockType::size_li) {
    for (int y = 0; y <= 16; y += VoxelBlockType::size_li) {
      for (int z = 0; z <= 16; z += VoxelBlockType::size_li) {
        allocation_list[list_idx] = octree.hash(x, y, z);
        block_coords.push_back(Eigen::Vector3i(x, y, z));
        list_idx++;
      }
    }
  }
  octree.allocate(allocation_list, 27);
  
  Eigen::Vector3i block_coord_c = Eigen::Vector3i(8, 8, 8);
  int block_scale_c = 0;
  int block_stride_c = 1 << block_scale_c;
  int block_scale_n = 0;

  for (auto block_coord : block_coords) {
    VoxelBlockType* block = octree.fetch(block_coord.x(), block_coord.y(), block_coord.z());
    block->current_scale(block_scale_n);
    block->min_scale(block_scale_n);
  }

  VoxelBlockType* block_c = octree.fetch(block_coord_c.x(), block_coord_c.y(), block_coord_c.z());
  block_c->current_scale(block_scale_c);
  block_c->min_scale(block_scale_c);
  for (int x = 0; x < VoxelBlockType::size_li - 1; x += block_stride_c) {
    for (int y = 0; y < VoxelBlockType::size_li - 1; y += block_stride_c) {
      for (int z = 0; z < VoxelBlockType::size_li - 1; z += block_stride_c) {
        Eigen::Vector3i voxel_coord = block_coord_c + Eigen::Vector3i(x, y, z);
        block_c->setData(voxel_coord, block_scale_c, {1.f, 1.f});
      }
    }
  }

  std::string filename = "../../out/multires-mesh-equal-neighbour-unittest.vtk";
  std::cout << "Saving triangle mesh to file :" << filename  << std::endl;

  std::vector<se::Triangle> mesh;
  auto inside = [](const TestVoxelT::VoxelData& data) {
    return data.x < 0.f;
  };

  auto select_value = [](const TestVoxelT::VoxelData& data) {
    return data.x;
  };

  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  Eigen::Matrix4f T_MW = Eigen::Matrix4f::Identity();
  writeVtkMesh(filename.c_str(), mesh, se::math::to_inverse_transformation(T_MW));
}

TEST(MeshingTest, CoarserScaleNeighbour) {
  typedef se::Octree<TestVoxelT> OctreeF;
  OctreeF octree;
  octree.init(32, 32);
  se::key_t allocation_list[27];
  int list_idx = 0;
  std::vector<Eigen::Vector3i> block_coords;
  for (int x = 0; x <= 16; x += VoxelBlockType::size_li) {
    for (int y = 0; y <= 16; y += VoxelBlockType::size_li) {
      for (int z = 0; z <= 16; z += VoxelBlockType::size_li) {
        allocation_list[list_idx] = octree.hash(x, y, z);
        block_coords.push_back(Eigen::Vector3i(x, y, z));
        list_idx++;
      }
    }
  }
  octree.allocate(allocation_list, 27);

  Eigen::Vector3i block_coord_c = Eigen::Vector3i(8, 8, 8);
  int block_scale_c = 0;
  int block_stride_c = 1 << block_scale_c;
  int block_scale_n = 1;

  for (auto block_coord : block_coords) {
    VoxelBlockType* block = octree.fetch(block_coord.x(), block_coord.y(), block_coord.z());
    block->current_scale(block_scale_n);
    block->min_scale(block_scale_n);
  }

  VoxelBlockType* block_c = octree.fetch(block_coord_c.x(), block_coord_c.y(), block_coord_c.z());
  block_c->current_scale(block_scale_c);
  block_c->min_scale(block_scale_c);
  for (int x = 0; x < VoxelBlockType::size_li - 1; x += block_stride_c) {
    for (int y = 0; y < VoxelBlockType::size_li - 1; y += block_stride_c) {
      for (int z = 0; z < VoxelBlockType::size_li - 1; z += block_stride_c) {
        Eigen::Vector3i voxel_coord = block_coord_c + Eigen::Vector3i(x, y, z);
        block_c->setData(voxel_coord, block_scale_c, {1.f, 1.f});
      }
    }
  }

  std::string filename = "../../out/multires-mesh-coarser-neighbour-unittest.vtk";
  std::cout << "Saving triangle mesh to file :" << filename  << std::endl;

  std::vector<se::Triangle> mesh;
  auto inside = [](const TestVoxelT::VoxelData& data) {
    return data.x < 0.f;
  };

  auto select_value = [](const TestVoxelT::VoxelData& data) {
    return data.x;
  };

  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  Eigen::Matrix4f T_MW = Eigen::Matrix4f::Identity();
  writeVtkMesh(filename.c_str(), mesh, se::math::to_inverse_transformation(T_MW));
}

TEST(MeshingTest, FinerScaleNeighbour) {
  typedef se::Octree<TestVoxelT> OctreeF;
  OctreeF octree;
  octree.init(32, 32);
  se::key_t allocation_list[27];
  int list_idx = 0;
  std::vector<Eigen::Vector3i> block_coords;
  for (int x = 0; x <= 16; x += VoxelBlockType::size_li) {
    for (int y = 0; y <= 16; y += VoxelBlockType::size_li) {
      for (int z = 0; z <= 16; z += VoxelBlockType::size_li) {
        allocation_list[list_idx] = octree.hash(x, y, z);
        block_coords.push_back(Eigen::Vector3i(x, y, z));
        list_idx++;
      }
    }
  }
  octree.allocate(allocation_list, 27);

  Eigen::Vector3i block_coord_c = Eigen::Vector3i(8, 8, 8);
  int block_scale_c = 1;
  int block_stride_c = 1 << block_scale_c;
  int block_scale_n = 0;

  for (auto block_coord : block_coords) {
    VoxelBlockType* block = octree.fetch(block_coord.x(), block_coord.y(), block_coord.z());
    block->current_scale(block_scale_n);
    block->min_scale(block_scale_n);
  }

  VoxelBlockType* block_c = octree.fetch(block_coord_c.x(), block_coord_c.y(), block_coord_c.z());
  block_c->current_scale(block_scale_c);
  block_c->min_scale(block_scale_c);
  for (int x = 0; x < VoxelBlockType::size_li - 1; x += block_stride_c) {
    for (int y = 0; y < VoxelBlockType::size_li - 1; y += block_stride_c) {
      for (int z = 0; z < VoxelBlockType::size_li - 1; z += block_stride_c) {
        Eigen::Vector3i voxel_coord = block_coord_c + Eigen::Vector3i(x, y, z);
        block_c->setData(voxel_coord, block_scale_c, {1.f, 1.f});
      }
    }
  }

  std::string filename = "../../out/multires-mesh-finer-neighbour-unittest.vtk";
  std::cout << "Saving triangle mesh to file :" << filename  << std::endl;

  std::vector<se::Triangle> mesh;
  auto inside = [](const TestVoxelT::VoxelData& data) {
    return data.x < 0.f;
  };

  auto select_value = [](const TestVoxelT::VoxelData& data) {
    return data.x;
  };

  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  Eigen::Matrix4f T_MW = Eigen::Matrix4f::Identity();
  writeVtkMesh(filename.c_str(), mesh, se::math::to_inverse_transformation(T_MW));
}