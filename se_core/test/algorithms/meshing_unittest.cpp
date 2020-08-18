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
  for (unsigned x = 0; x <= 16; x += VoxelBlockType::size_li) {
    for (unsigned y = 0; y <= 16; y += VoxelBlockType::size_li) {
      for (unsigned z = 0; z <= 16; z += VoxelBlockType::size_li) {
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
  for (unsigned x = 0; x < VoxelBlockType::size_li - 1; x += block_stride_c) {
    for (unsigned y = 0; y < VoxelBlockType::size_li - 1; y += block_stride_c) {
      for (unsigned z = 0; z < VoxelBlockType::size_li - 1; z += block_stride_c) {
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
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));
}

TEST(MeshingTest, CoarserScaleNeighbour) {
  typedef se::Octree<TestVoxelT> OctreeF;
  OctreeF octree;
  octree.init(32, 32);
  se::key_t allocation_list[27];
  int list_idx = 0;
  std::vector<Eigen::Vector3i> block_coords;
  for (unsigned x = 0; x <= 16; x += VoxelBlockType::size_li) {
    for (unsigned y = 0; y <= 16; y += VoxelBlockType::size_li) {
      for (unsigned z = 0; z <= 16; z += VoxelBlockType::size_li) {
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
  for (unsigned x = 0; x < VoxelBlockType::size_li - 1; x += block_stride_c) {
    for (unsigned y = 0; y < VoxelBlockType::size_li - 1; y += block_stride_c) {
      for (unsigned z = 0; z < VoxelBlockType::size_li - 1; z += block_stride_c) {
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
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));
}

TEST(MeshingTest, FinerScaleNeighbour) {
  typedef se::Octree<TestVoxelT> OctreeF;
  OctreeF octree;
  octree.init(32, 32);
  se::key_t allocation_list[27];
  int list_idx = 0;
  std::vector<Eigen::Vector3i> block_coords;
  for (unsigned x = 0; x <= 16; x += VoxelBlockType::size_li) {
    for (unsigned y = 0; y <= 16; y += VoxelBlockType::size_li) {
      for (unsigned z = 0; z <= 16; z += VoxelBlockType::size_li) {
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
  for (unsigned x = 0; x < VoxelBlockType::size_li - 1; x += block_stride_c) {
    for (unsigned y = 0; y < VoxelBlockType::size_li - 1; y += block_stride_c) {
      for (unsigned z = 0; z < VoxelBlockType::size_li - 1; z += block_stride_c) {
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
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));
}

TEST(MeshingTest, Wall) {
  typedef se::Octree<TestVoxelT> OctreeF;
  OctreeF octree;
  octree.init(32, 32);
  se::key_t allocation_list[32];
  int list_idx = 0;
  std::vector<Eigen::Vector3i> block_coords_1;
  std::vector<Eigen::Vector3i> block_coords_2;
  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      allocation_list[list_idx] = octree.hash(8, y, z);
      block_coords_1.push_back(Eigen::Vector3i(8, y, z));
      list_idx++;
      allocation_list[list_idx] = octree.hash(16, y, z);
      block_coords_2.push_back(Eigen::Vector3i(16, y, z));
      list_idx++;
    }
  }
  octree.allocate(allocation_list, 32);

  int   block_scale_1  = 0;
  int   block_stride_1 = 1 << block_scale_1;
  float block_value_1  = 0.5f;

  int   block_scale_2  = 0;
  int   block_stride_2 = 1 << block_scale_1;
  float block_value_2  = -0.5f;

  for (auto block_coord : block_coords_1) {
    VoxelBlockType* block = octree.fetch(block_coord.x(), block_coord.y(), block_coord.z());
    block->current_scale(block_scale_1);
    block->min_scale(block_scale_1);
    for (unsigned x = 0; x < VoxelBlockType::size_li; x += block_stride_1) {
      for (unsigned y = 0; y < VoxelBlockType::size_li; y += block_stride_1) {
        for (unsigned z = 0; z < VoxelBlockType::size_li; z += block_stride_1) {
          Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x, y, z);
          block->setData(voxel_coord, block_scale_1, {block_value_1, 1.f});
        }
      }
    }
  }

  for (auto block_coord : block_coords_2) {
    VoxelBlockType* block = octree.fetch(block_coord.x(), block_coord.y(), block_coord.z());
    block->current_scale(block_scale_2);
    block->min_scale(block_scale_2);
    for (unsigned x = 0; x < VoxelBlockType::size_li; x += block_stride_2) {
      for (unsigned y = 0; y < VoxelBlockType::size_li; y += block_stride_2) {
        for (unsigned z = 0; z < VoxelBlockType::size_li; z += block_stride_2) {
          Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x, y, z);
          block->setData(voxel_coord, block_scale_2, {block_value_2, 1.f});
        }
      }
    }
  }

  std::string filename = "../../out/multires-mesh-wall1-unittest.vtk";
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
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));
}

TEST(MeshingTest, WallCrossesXFineToCoarseAlongY) {
  typedef se::Octree<TestVoxelT> OctreeF;
  OctreeF octree;
  octree.init(32, 32);
  se::key_t allocation_list[32];
  int list_idx = 0;
  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 8; x < 24; x += VoxelBlockType::size_li) {
        allocation_list[list_idx] = octree.hash(x, y, z);
        list_idx++;
      }
    }
  }
  octree.allocate(allocation_list, 32);


  int   block_scale_0     = 0;
  int   block_stride_0    = 1 << block_scale_0;
  float block_value_0_out = 0.5f;
  float block_value_0_in  = -0.5f;

  int   block_scale_1     = 1;
  int   block_stride_1    = 1 << block_scale_0;
  float block_value_1_out = 1.f;
  float block_value_1_in  = -1.f;

  int   block_scale_2     = 2;
  int   block_stride_2    = 1 << block_scale_0;
  float block_value_2_out = 2.f;
  float block_value_2_in  = -2.f;

  int   block_scale_3     = 3;
  int   block_stride_3    = 1 << block_scale_0;
  float block_value_3_out = 4.f;
  float block_value_3_in  = -4.f;

  auto inside = [](const TestVoxelT::VoxelData& data) {
    return data.x < 0.f;
  };

  auto select_value = [](const TestVoxelT::VoxelData& data) {
    return data.x;
  };

  Eigen::Matrix4f T_MW = Eigen::Matrix4f::Identity();

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 8; x < 24; x += VoxelBlockType::size_li) {
        if (y < 16) { // Scale 0
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_0);
          block->min_scale(block_scale_0);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_0) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_0) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_0) {
                if (x < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 1
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_1);
          block->min_scale(block_scale_1);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_1) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_1) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_1) {
                if (x < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  std::string filename = "../../out/multires-mesh-wall-crosses-x-axis-fine-to-coarse-along-y-scale-0-1-unittest.vtk";
  std::vector<se::Triangle> mesh;
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 8; x < 24; x += VoxelBlockType::size_li) {
        if (y < 16) { // Scale 0
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_0);
          block->min_scale(block_scale_0);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_0) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_0) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_0) {
                if (x < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 2
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_2);
          block->min_scale(block_scale_2);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_2) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_2) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_2) {
                if (x < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-x-axis-fine-to-coarse-along-y-scale-0-2-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 8; x < 24; x += VoxelBlockType::size_li) {
        if (y < 16) { // Scale 0
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_0);
          block->min_scale(block_scale_0);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_0) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_0) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_0) {
                if (x < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 3
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_3);
          block->min_scale(block_scale_3);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_3) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_3) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_3) {
                if (x < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-x-axis-fine-to-coarse-along-y-scale-0-3-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 8; x < 24; x += VoxelBlockType::size_li) {
        if (y < 16) { // Scale 1
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_1);
          block->min_scale(block_scale_1);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_1) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_1) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_1) {
                if (x < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 2
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_2);
          block->min_scale(block_scale_2);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_2) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_2) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_2) {
                if (x < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_1_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-x-axis-fine-to-coarse-along-y-scale-1-2-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 8; x < 24; x += VoxelBlockType::size_li) {
        if (y < 16) { // Scale 1
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_1);
          block->min_scale(block_scale_1);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_1) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_1) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_1) {
                if (x < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 3
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_3);
          block->min_scale(block_scale_3);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_3) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_3) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_3) {
                if (x < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-x-axis-fine-to-coarse-along-y-scale-1-3-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 8; x < 24; x += VoxelBlockType::size_li) {
        if (y < 16) { // Scale 2
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_2);
          block->min_scale(block_scale_2);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_2) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_2) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_2) {
                if (x < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 3
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_3);
          block->min_scale(block_scale_3);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_3) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_3) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_3) {
                if (x < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-x-axis-fine-to-coarse-along-y-scale-2-3-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));
}

TEST(MeshingTest, WallCrossesXFineToCoarseAlongZ) {
  typedef se::Octree<TestVoxelT> OctreeF;
  OctreeF octree;
  octree.init(32, 32);
  se::key_t allocation_list[32];
  int list_idx = 0;
  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 8; x < 24; x += VoxelBlockType::size_li) {
        allocation_list[list_idx] = octree.hash(x, y, z);
        list_idx++;
      }
    }
  }
  octree.allocate(allocation_list, 32);


  int   block_scale_0     = 0;
  int   block_stride_0    = 1 << block_scale_0;
  float block_value_0_out = 0.5f;
  float block_value_0_in  = -0.5f;

  int   block_scale_1     = 1;
  int   block_stride_1    = 1 << block_scale_0;
  float block_value_1_out = 1.f;
  float block_value_1_in  = -1.f;

  int   block_scale_2     = 2;
  int   block_stride_2    = 1 << block_scale_0;
  float block_value_2_out = 2.f;
  float block_value_2_in  = -2.f;

  int   block_scale_3     = 3;
  int   block_stride_3    = 1 << block_scale_0;
  float block_value_3_out = 4.f;
  float block_value_3_in  = -4.f;

  auto inside = [](const TestVoxelT::VoxelData& data) {
    return data.x < 0.f;
  };

  auto select_value = [](const TestVoxelT::VoxelData& data) {
    return data.x;
  };

  Eigen::Matrix4f T_MW = Eigen::Matrix4f::Identity();

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 8; x < 24; x += VoxelBlockType::size_li) {
        if (z < 16) { // Scale 0
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_0);
          block->min_scale(block_scale_0);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_0) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_0) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_0) {
                if (x < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 1
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_1);
          block->min_scale(block_scale_1);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_1) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_1) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_1) {
                if (x < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  std::string filename = "../../out/multires-mesh-wall-crosses-x-axis-fine-to-coarse-along-z-scale-0-1-unittest.vtk";
  std::vector<se::Triangle> mesh;
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 8; x < 24; x += VoxelBlockType::size_li) {
        if (z < 16) { // Scale 0
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_0);
          block->min_scale(block_scale_0);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_0) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_0) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_0) {
                if (x < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 2
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_2);
          block->min_scale(block_scale_2);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_2) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_2) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_2) {
                if (x < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-x-axis-fine-to-coarse-along-z-scale-0-2-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 8; x < 24; x += VoxelBlockType::size_li) {
        if (z < 16) { // Scale 0
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_0);
          block->min_scale(block_scale_0);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_0) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_0) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_0) {
                if (x < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 3
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_3);
          block->min_scale(block_scale_3);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_3) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_3) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_3) {
                if (x < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-x-axis-fine-to-coarse-along-z-scale-0-3-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 8; x < 24; x += VoxelBlockType::size_li) {
        if (z < 16) { // Scale 1
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_1);
          block->min_scale(block_scale_1);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_1) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_1) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_1) {
                if (x < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 2
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_2);
          block->min_scale(block_scale_2);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_2) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_2) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_2) {
                if (x < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_1_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-x-axis-fine-to-coarse-along-z-scale-1-2-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 8; x < 24; x += VoxelBlockType::size_li) {
        if (z < 16) { // Scale 1
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_1);
          block->min_scale(block_scale_1);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_1) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_1) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_1) {
                if (x < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 3
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_3);
          block->min_scale(block_scale_3);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_3) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_3) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_3) {
                if (x < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-x-axis-fine-to-coarse-along-z-scale-1-3-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 8; x < 24; x += VoxelBlockType::size_li) {
        if (z < 16) { // Scale 2
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_2);
          block->min_scale(block_scale_2);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_2) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_2) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_2) {
                if (x < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 3
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_3);
          block->min_scale(block_scale_3);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_3) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_3) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_3) {
                if (x < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-x-axis-fine-to-coarse-along-z-scale-2-3-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));
}

TEST(MeshingTest, WallCrossesXCoarseToFineAlongY) {
  typedef se::Octree<TestVoxelT> OctreeF;
  OctreeF octree;
  octree.init(32, 32);
  se::key_t allocation_list[32];
  int list_idx = 0;
  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 8; x < 24; x += VoxelBlockType::size_li) {
        allocation_list[list_idx] = octree.hash(x, y, z);
        list_idx++;
      }
    }
  }
  octree.allocate(allocation_list, 32);


  int   block_scale_0     = 0;
  int   block_stride_0    = 1 << block_scale_0;
  float block_value_0_out = 0.5f;
  float block_value_0_in  = -0.5f;

  int   block_scale_1     = 1;
  int   block_stride_1    = 1 << block_scale_0;
  float block_value_1_out = 1.f;
  float block_value_1_in  = -1.f;

  int   block_scale_2     = 2;
  int   block_stride_2    = 1 << block_scale_0;
  float block_value_2_out = 2.f;
  float block_value_2_in  = -2.f;

  int   block_scale_3     = 3;
  int   block_stride_3    = 1 << block_scale_0;
  float block_value_3_out = 4.f;
  float block_value_3_in  = -4.f;

  auto inside = [](const TestVoxelT::VoxelData& data) {
    return data.x < 0.f;
  };

  auto select_value = [](const TestVoxelT::VoxelData& data) {
    return data.x;
  };

  Eigen::Matrix4f T_MW = Eigen::Matrix4f::Identity();

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 8; x < 24; x += VoxelBlockType::size_li) {
        if (y >= 16) { // Scale 0
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_0);
          block->min_scale(block_scale_0);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_0) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_0) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_0) {
                if (x >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 1
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_1);
          block->min_scale(block_scale_1);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_1) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_1) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_1) {
                if (x >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  std::string filename = "../../out/multires-mesh-wall-crosses-x-axis-coarse-to-fine-along-y-scale-1-0-unittest.vtk";
  std::vector<se::Triangle> mesh;
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 8; x < 24; x += VoxelBlockType::size_li) {
        if (y >= 16) { // Scale 0
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_0);
          block->min_scale(block_scale_0);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_0) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_0) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_0) {
                if (x >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 2
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_2);
          block->min_scale(block_scale_2);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_2) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_2) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_2) {
                if (x >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-x-axis-coarse-to-fine-along-y-scale-2-0-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 8; x < 24; x += VoxelBlockType::size_li) {
        if (y >= 16) { // Scale 0
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_0);
          block->min_scale(block_scale_0);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_0) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_0) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_0) {
                if (x >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 3
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_3);
          block->min_scale(block_scale_3);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_3) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_3) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_3) {
                if (x >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-x-axis-coarse-to-fine-along-y-scale-3-0-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 8; x < 24; x += VoxelBlockType::size_li) {
        if (y >= 16) { // Scale 1
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_1);
          block->min_scale(block_scale_1);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_1) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_1) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_1) {
                if (x >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 2
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_2);
          block->min_scale(block_scale_2);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_2) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_2) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_2) {
                if (x >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_1_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-x-axis-coarse-to-fine-along-y-scale-2-1-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 8; x < 24; x += VoxelBlockType::size_li) {
        if (y >= 16) { // Scale 1
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_1);
          block->min_scale(block_scale_1);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_1) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_1) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_1) {
                if (x >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 3
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_3);
          block->min_scale(block_scale_3);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_3) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_3) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_3) {
                if (x >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-x-axis-coarse-to-fine-along-y-scale-3-1-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 8; x < 24; x += VoxelBlockType::size_li) {
        if (y >= 16) { // Scale 2
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_2);
          block->min_scale(block_scale_2);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_2) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_2) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_2) {
                if (x >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 3
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_3);
          block->min_scale(block_scale_3);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_3) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_3) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_3) {
                if (x >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-x-axis-coarse-to-fine-along-y-scale-3-2-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));
}

TEST(MeshingTest, WallCrossesXCoarseToFineAlongZ) {
  typedef se::Octree<TestVoxelT> OctreeF;
  OctreeF octree;
  octree.init(32, 32);
  se::key_t allocation_list[32];
  int list_idx = 0;
  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 8; x < 24; x += VoxelBlockType::size_li) {
        allocation_list[list_idx] = octree.hash(x, y, z);
        list_idx++;
      }
    }
  }
  octree.allocate(allocation_list, 32);


  int   block_scale_0     = 0;
  int   block_stride_0    = 1 << block_scale_0;
  float block_value_0_out = 0.5f;
  float block_value_0_in  = -0.5f;

  int   block_scale_1     = 1;
  int   block_stride_1    = 1 << block_scale_0;
  float block_value_1_out = 1.f;
  float block_value_1_in  = -1.f;

  int   block_scale_2     = 2;
  int   block_stride_2    = 1 << block_scale_0;
  float block_value_2_out = 2.f;
  float block_value_2_in  = -2.f;

  int   block_scale_3     = 3;
  int   block_stride_3    = 1 << block_scale_0;
  float block_value_3_out = 4.f;
  float block_value_3_in  = -4.f;

  auto inside = [](const TestVoxelT::VoxelData& data) {
    return data.x < 0.f;
  };

  auto select_value = [](const TestVoxelT::VoxelData& data) {
    return data.x;
  };

  Eigen::Matrix4f T_MW = Eigen::Matrix4f::Identity();

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 8; x < 24; x += VoxelBlockType::size_li) {
        if (z >= 16) { // Scale 0
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_0);
          block->min_scale(block_scale_0);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_0) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_0) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_0) {
                if (x >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 1
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_1);
          block->min_scale(block_scale_1);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_1) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_1) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_1) {
                if (x >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  std::string filename = "../../out/multires-mesh-wall-crosses-x-axis-coarse-to-fine-along-z-scale-1-0-unittest.vtk";
  std::vector<se::Triangle> mesh;
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 8; x < 24; x += VoxelBlockType::size_li) {
        if (z >= 16) { // Scale 0
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_0);
          block->min_scale(block_scale_0);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_0) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_0) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_0) {
                if (x >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 2
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_2);
          block->min_scale(block_scale_2);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_2) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_2) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_2) {
                if (x >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-x-axis-coarse-to-fine-along-z-scale-2-0-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 8; x < 24; x += VoxelBlockType::size_li) {
        if (z >= 16) { // Scale 0
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_0);
          block->min_scale(block_scale_0);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_0) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_0) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_0) {
                if (x >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 3
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_3);
          block->min_scale(block_scale_3);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_3) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_3) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_3) {
                if (x >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-x-axis-coarse-to-fine-along-z-scale-3-0-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 8; x < 24; x += VoxelBlockType::size_li) {
        if (z >= 16) { // Scale 1
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_1);
          block->min_scale(block_scale_1);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_1) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_1) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_1) {
                if (x >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 2
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_2);
          block->min_scale(block_scale_2);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_2) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_2) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_2) {
                if (x >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_1_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-x-axis-coarse-to-fine-along-z-scale-2-1-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 8; x < 24; x += VoxelBlockType::size_li) {
        if (z >= 16) { // Scale 1
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_1);
          block->min_scale(block_scale_1);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_1) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_1) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_1) {
                if (x >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 3
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_3);
          block->min_scale(block_scale_3);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_3) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_3) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_3) {
                if (x >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-x-axis-coarse-to-fine-along-z-scale-3-1-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 8; x < 24; x += VoxelBlockType::size_li) {
        if (z >= 16) { // Scale 2
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_2);
          block->min_scale(block_scale_2);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_2) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_2) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_2) {
                if (x >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 3
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_3);
          block->min_scale(block_scale_3);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_3) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_3) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_3) {
                if (x >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-x-axis-coarse-to-fine-along-z-scale-3-2-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));
}

TEST(MeshingTest, WallCrossesYFineToCoarseAlongX) {
  typedef se::Octree<TestVoxelT> OctreeF;
  OctreeF octree;
  octree.init(32, 32);
  se::key_t allocation_list[32];
  int list_idx = 0;
  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned y = 8; y < 24; y += VoxelBlockType::size_li) {
      for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
        allocation_list[list_idx] = octree.hash(x, y, z);
        list_idx++;
      }
    }
  }
  octree.allocate(allocation_list, 32);


  int   block_scale_0     = 0;
  int   block_stride_0    = 1 << block_scale_0;
  float block_value_0_out = 0.5f;
  float block_value_0_in  = -0.5f;

  int   block_scale_1     = 1;
  int   block_stride_1    = 1 << block_scale_0;
  float block_value_1_out = 1.f;
  float block_value_1_in  = -1.f;

  int   block_scale_2     = 2;
  int   block_stride_2    = 1 << block_scale_0;
  float block_value_2_out = 2.f;
  float block_value_2_in  = -2.f;

  int   block_scale_3     = 3;
  int   block_stride_3    = 1 << block_scale_0;
  float block_value_3_out = 4.f;
  float block_value_3_in  = -4.f;

  auto inside = [](const TestVoxelT::VoxelData& data) {
    return data.x < 0.f;
  };

  auto select_value = [](const TestVoxelT::VoxelData& data) {
    return data.x;
  };

  Eigen::Matrix4f T_MW = Eigen::Matrix4f::Identity();

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
      for (unsigned y = 8; y < 24; y += VoxelBlockType::size_li) {
        if (x < 16) { // Scale 0
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_0);
          block->min_scale(block_scale_0);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_0) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_0) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_0) {
                if (y < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 1
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_1);
          block->min_scale(block_scale_1);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_1) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_1) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_1) {
                if (y < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  std::string filename = "../../out/multires-mesh-wall-crosses-y-axis-fine-to-coarse-along-x-scale-0-1-unittest.vtk";
  std::vector<se::Triangle> mesh;
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
      for (unsigned y = 8; y < 24; y += VoxelBlockType::size_li) {
        if (x < 16) { // Scale 0
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_0);
          block->min_scale(block_scale_0);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_0) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_0) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_0) {
                if (y < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 2
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_2);
          block->min_scale(block_scale_2);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_2) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_2) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_2) {
                if (y < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-y-axis-fine-to-coarse-along-x-scale-0-2-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
      for (unsigned y = 8; y < 24; y += VoxelBlockType::size_li) {
        if (x < 16) { // Scale 0
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_0);
          block->min_scale(block_scale_0);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_0) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_0) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_0) {
                if (y < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 3
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_3);
          block->min_scale(block_scale_3);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_3) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_3) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_3) {
                if (y < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-y-axis-fine-to-coarse-along-x-scale-0-3-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
      for (unsigned y = 8; y < 24; y += VoxelBlockType::size_li) {
        if (x < 16) { // Scale 1
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_1);
          block->min_scale(block_scale_1);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_1) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_1) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_1) {
                if (y < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 2
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_2);
          block->min_scale(block_scale_2);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_2) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_2) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_2) {
                if (y < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_1_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-y-axis-fine-to-coarse-along-x-scale-1-2-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
      for (unsigned y = 8; y < 24; y += VoxelBlockType::size_li) {
        if (x < 16) { // Scale 1
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_1);
          block->min_scale(block_scale_1);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_1) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_1) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_1) {
                if (y < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 3
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_3);
          block->min_scale(block_scale_3);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_3) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_3) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_3) {
                if (y < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-y-axis-fine-to-coarse-along-x-scale-1-3-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
      for (unsigned y = 8; y < 24; y += VoxelBlockType::size_li) {
        if (x < 16) { // Scale 2
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_2);
          block->min_scale(block_scale_2);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_2) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_2) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_2) {
                if (y < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 3
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_3);
          block->min_scale(block_scale_3);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_3) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_3) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_3) {
                if (y < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-y-axis-fine-to-coarse-along-x-scale-2-3-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));
}

TEST(MeshingTest, WallCrossesYFineToCoarseAlongZ) {
  typedef se::Octree<TestVoxelT> OctreeF;
  OctreeF octree;
  octree.init(32, 32);
  se::key_t allocation_list[32];
  int list_idx = 0;
  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned y = 8; y < 24; y += VoxelBlockType::size_li) {
      for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
        allocation_list[list_idx] = octree.hash(x, y, z);
        list_idx++;
      }
    }
  }
  octree.allocate(allocation_list, 32);


  int   block_scale_0     = 0;
  int   block_stride_0    = 1 << block_scale_0;
  float block_value_0_out = 0.5f;
  float block_value_0_in  = -0.5f;

  int   block_scale_1     = 1;
  int   block_stride_1    = 1 << block_scale_0;
  float block_value_1_out = 1.f;
  float block_value_1_in  = -1.f;

  int   block_scale_2     = 2;
  int   block_stride_2    = 1 << block_scale_0;
  float block_value_2_out = 2.f;
  float block_value_2_in  = -2.f;

  int   block_scale_3     = 3;
  int   block_stride_3    = 1 << block_scale_0;
  float block_value_3_out = 4.f;
  float block_value_3_in  = -4.f;

  auto inside = [](const TestVoxelT::VoxelData& data) {
    return data.x < 0.f;
  };

  auto select_value = [](const TestVoxelT::VoxelData& data) {
    return data.x;
  };

  Eigen::Matrix4f T_MW = Eigen::Matrix4f::Identity();

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
      for (unsigned y = 8; y < 24; y += VoxelBlockType::size_li) {
        if (z < 16) { // Scale 0
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_0);
          block->min_scale(block_scale_0);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_0) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_0) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_0) {
                if (y < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 1
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_1);
          block->min_scale(block_scale_1);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_1) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_1) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_1) {
                if (y < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  std::string filename = "../../out/multires-mesh-wall-crosses-y-axis-fine-to-coarse-along-z-scale-0-1-unittest.vtk";
  std::vector<se::Triangle> mesh;
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
      for (unsigned y = 8; y < 24; y += VoxelBlockType::size_li) {
        if (z < 16) { // Scale 0
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_0);
          block->min_scale(block_scale_0);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_0) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_0) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_0) {
                if (y < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 2
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_2);
          block->min_scale(block_scale_2);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_2) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_2) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_2) {
                if (y < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-y-axis-fine-to-coarse-along-z-scale-0-2-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
      for (unsigned y = 8; y < 24; y += VoxelBlockType::size_li) {
        if (z < 16) { // Scale 0
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_0);
          block->min_scale(block_scale_0);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_0) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_0) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_0) {
                if (y < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 3
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_3);
          block->min_scale(block_scale_3);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_3) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_3) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_3) {
                if (y < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-y-axis-fine-to-coarse-along-z-scale-0-3-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
      for (unsigned y = 8; y < 24; y += VoxelBlockType::size_li) {
        if (z < 16) { // Scale 1
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_1);
          block->min_scale(block_scale_1);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_1) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_1) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_1) {
                if (y < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 2
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_2);
          block->min_scale(block_scale_2);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_2) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_2) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_2) {
                if (y < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_1_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-y-axis-fine-to-coarse-along-z-scale-1-2-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
      for (unsigned y = 8; y < 24; y += VoxelBlockType::size_li) {
        if (z < 16) { // Scale 1
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_1);
          block->min_scale(block_scale_1);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_1) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_1) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_1) {
                if (y < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 3
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_3);
          block->min_scale(block_scale_3);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_3) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_3) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_3) {
                if (y < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-y-axis-fine-to-coarse-along-z-scale-1-3-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
      for (unsigned y = 8; y < 24; y += VoxelBlockType::size_li) {
        if (z < 16) { // Scale 2
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_2);
          block->min_scale(block_scale_2);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_2) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_2) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_2) {
                if (y < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 3
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_3);
          block->min_scale(block_scale_3);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_3) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_3) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_3) {
                if (y < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-y-axis-fine-to-coarse-along-z-scale-2-3-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));
}

TEST(MeshingTest, WallCrossesYCoarseToFineAlongX) {
  typedef se::Octree<TestVoxelT> OctreeF;
  OctreeF octree;
  octree.init(32, 32);
  se::key_t allocation_list[32];
  int list_idx = 0;
  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned y = 8; y < 24; y += VoxelBlockType::size_li) {
      for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
        allocation_list[list_idx] = octree.hash(x, y, z);
        list_idx++;
      }
    }
  }
  octree.allocate(allocation_list, 32);


  int   block_scale_0     = 0;
  int   block_stride_0    = 1 << block_scale_0;
  float block_value_0_out = 0.5f;
  float block_value_0_in  = -0.5f;

  int   block_scale_1     = 1;
  int   block_stride_1    = 1 << block_scale_0;
  float block_value_1_out = 1.f;
  float block_value_1_in  = -1.f;

  int   block_scale_2     = 2;
  int   block_stride_2    = 1 << block_scale_0;
  float block_value_2_out = 2.f;
  float block_value_2_in  = -2.f;

  int   block_scale_3     = 3;
  int   block_stride_3    = 1 << block_scale_0;
  float block_value_3_out = 4.f;
  float block_value_3_in  = -4.f;

  auto inside = [](const TestVoxelT::VoxelData& data) {
    return data.x < 0.f;
  };

  auto select_value = [](const TestVoxelT::VoxelData& data) {
    return data.x;
  };

  Eigen::Matrix4f T_MW = Eigen::Matrix4f::Identity();

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
      for (unsigned y = 8; y < 24; y += VoxelBlockType::size_li) {
        if (x >= 16) { // Scale 0
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_0);
          block->min_scale(block_scale_0);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_0) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_0) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_0) {
                if (y >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 1
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_1);
          block->min_scale(block_scale_1);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_1) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_1) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_1) {
                if (y >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  std::string filename = "../../out/multires-mesh-wall-crosses-y-axis-coarse-to-fine-along-x-scale-1-0-unittest.vtk";
  std::vector<se::Triangle> mesh;
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
      for (unsigned y = 8; y < 24; y += VoxelBlockType::size_li) {
        if (x >= 16) { // Scale 0
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_0);
          block->min_scale(block_scale_0);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_0) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_0) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_0) {
                if (y >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 2
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_2);
          block->min_scale(block_scale_2);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_2) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_2) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_2) {
                if (y >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-y-axis-coarse-to-fine-along-x-scale-2-0-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
      for (unsigned y = 8; y < 24; y += VoxelBlockType::size_li) {
        if (x >= 16) { // Scale 0
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_0);
          block->min_scale(block_scale_0);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_0) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_0) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_0) {
                if (y >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 3
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_3);
          block->min_scale(block_scale_3);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_3) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_3) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_3) {
                if (y >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-y-axis-coarse-to-fine-along-x-scale-3-0-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
      for (unsigned y = 8; y < 24; y += VoxelBlockType::size_li) {
        if (x >= 16) { // Scale 1
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_1);
          block->min_scale(block_scale_1);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_1) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_1) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_1) {
                if (y >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 2
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_2);
          block->min_scale(block_scale_2);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_2) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_2) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_2) {
                if (y >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_1_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-y-axis-coarse-to-fine-along-x-scale-2-1-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
      for (unsigned y = 8; y < 24; y += VoxelBlockType::size_li) {
        if (x >= 16) { // Scale 1
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_1);
          block->min_scale(block_scale_1);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_1) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_1) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_1) {
                if (y >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 3
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_3);
          block->min_scale(block_scale_3);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_3) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_3) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_3) {
                if (y >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-y-axis-coarse-to-fine-along-x-scale-3-1-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
      for (unsigned y = 8; y < 24; y += VoxelBlockType::size_li) {
        if (x >= 16) { // Scale 2
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_2);
          block->min_scale(block_scale_2);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_2) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_2) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_2) {
                if (y >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 3
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_3);
          block->min_scale(block_scale_3);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_3) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_3) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_3) {
                if (y >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-y-axis-coarse-to-fine-along-x-scale-3-2-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));
}

TEST(MeshingTest, WallCrossesYCoarseToFineAlongZ) {
  typedef se::Octree<TestVoxelT> OctreeF;
  OctreeF octree;
  octree.init(32, 32);
  se::key_t allocation_list[32];
  int list_idx = 0;
  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned y = 8; y < 24; y += VoxelBlockType::size_li) {
      for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
        allocation_list[list_idx] = octree.hash(x, y, z);
        list_idx++;
      }
    }
  }
  octree.allocate(allocation_list, 32);


  int   block_scale_0     = 0;
  int   block_stride_0    = 1 << block_scale_0;
  float block_value_0_out = 0.5f;
  float block_value_0_in  = -0.5f;

  int   block_scale_1     = 1;
  int   block_stride_1    = 1 << block_scale_0;
  float block_value_1_out = 1.f;
  float block_value_1_in  = -1.f;

  int   block_scale_2     = 2;
  int   block_stride_2    = 1 << block_scale_0;
  float block_value_2_out = 2.f;
  float block_value_2_in  = -2.f;

  int   block_scale_3     = 3;
  int   block_stride_3    = 1 << block_scale_0;
  float block_value_3_out = 4.f;
  float block_value_3_in  = -4.f;

  auto inside = [](const TestVoxelT::VoxelData& data) {
    return data.x < 0.f;
  };

  auto select_value = [](const TestVoxelT::VoxelData& data) {
    return data.x;
  };

  Eigen::Matrix4f T_MW = Eigen::Matrix4f::Identity();

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
      for (unsigned y = 8; y < 24; y += VoxelBlockType::size_li) {
        if (x >= 16) { // Scale 0
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_0);
          block->min_scale(block_scale_0);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_0) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_0) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_0) {
                if (y >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 1
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_1);
          block->min_scale(block_scale_1);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_1) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_1) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_1) {
                if (y >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  std::string filename = "../../out/multires-mesh-wall-crosses-y-axis-coarse-to-fine-along-z-scale-1-0-unittest.vtk";
  std::vector<se::Triangle> mesh;
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
      for (unsigned y = 8; y < 24; y += VoxelBlockType::size_li) {
        if (x >= 16) { // Scale 0
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_0);
          block->min_scale(block_scale_0);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_0) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_0) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_0) {
                if (y >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 2
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_2);
          block->min_scale(block_scale_2);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_2) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_2) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_2) {
                if (y >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-y-axis-coarse-to-fine-along-z-scale-2-0-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
      for (unsigned y = 8; y < 24; y += VoxelBlockType::size_li) {
        if (x >= 16) { // Scale 0
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_0);
          block->min_scale(block_scale_0);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_0) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_0) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_0) {
                if (y >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 3
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_3);
          block->min_scale(block_scale_3);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_3) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_3) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_3) {
                if (y >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-y-axis-coarse-to-fine-along-z-scale-3-0-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
      for (unsigned y = 8; y < 24; y += VoxelBlockType::size_li) {
        if (x >= 16) { // Scale 1
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_1);
          block->min_scale(block_scale_1);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_1) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_1) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_1) {
                if (y >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 2
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_2);
          block->min_scale(block_scale_2);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_2) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_2) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_2) {
                if (y >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_1_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-y-axis-coarse-to-fine-along-z-scale-2-1-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
      for (unsigned y = 8; y < 24; y += VoxelBlockType::size_li) {
        if (x >= 16) { // Scale 1
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_1);
          block->min_scale(block_scale_1);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_1) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_1) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_1) {
                if (y >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 3
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_3);
          block->min_scale(block_scale_3);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_3) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_3) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_3) {
                if (y >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-y-axis-coarse-to-fine-along-z-scale-3-1-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 0; z < 32; z += VoxelBlockType::size_li) {
    for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
      for (unsigned y = 8; y < 24; y += VoxelBlockType::size_li) {
        if (x >= 16) { // Scale 2
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_2);
          block->min_scale(block_scale_2);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_2) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_2) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_2) {
                if (y >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 3
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_3);
          block->min_scale(block_scale_3);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_3) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_3) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_3) {
                if (y >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-y-axis-coarse-to-fine-along-z-scale-3-2-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));
}

TEST(MeshingTest, WallCrossesZFineToCoarseAlongX) {
  typedef se::Octree<TestVoxelT> OctreeF;
  OctreeF octree;
  octree.init(32, 32);
  se::key_t allocation_list[32];
  int list_idx = 0;
  for (unsigned z = 8; z < 24; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
        allocation_list[list_idx] = octree.hash(x, y, z);
        list_idx++;
      }
    }
  }
  octree.allocate(allocation_list, 32);


  int   block_scale_0     = 0;
  int   block_stride_0    = 1 << block_scale_0;
  float block_value_0_out = 0.5f;
  float block_value_0_in  = -0.5f;

  int   block_scale_1     = 1;
  int   block_stride_1    = 1 << block_scale_0;
  float block_value_1_out = 1.f;
  float block_value_1_in  = -1.f;

  int   block_scale_2     = 2;
  int   block_stride_2    = 1 << block_scale_0;
  float block_value_2_out = 2.f;
  float block_value_2_in  = -2.f;

  int   block_scale_3     = 3;
  int   block_stride_3    = 1 << block_scale_0;
  float block_value_3_out = 4.f;
  float block_value_3_in  = -4.f;

  auto inside = [](const TestVoxelT::VoxelData& data) {
    return data.x < 0.f;
  };

  auto select_value = [](const TestVoxelT::VoxelData& data) {
    return data.x;
  };

  Eigen::Matrix4f T_MW = Eigen::Matrix4f::Identity();

  for (unsigned z = 8; z < 24; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
        if (x < 16) { // Scale 0
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_0);
          block->min_scale(block_scale_0);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_0) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_0) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_0) {
                if (z < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 1
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_1);
          block->min_scale(block_scale_1);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_1) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_1) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_1) {
                if (z < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  std::string filename = "../../out/multires-mesh-wall-crosses-z-axis-fine-to-coarse-along-x-scale-0-1-unittest.vtk";
  std::vector<se::Triangle> mesh;
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 8; z < 24; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
        if (x < 16) { // Scale 0
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_0);
          block->min_scale(block_scale_0);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_0) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_0) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_0) {
                if (z < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 2
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_2);
          block->min_scale(block_scale_2);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_2) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_2) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_2) {
                if (z < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-z-axis-fine-to-coarse-along-x-scale-0-2-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 8; z < 24; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
        if (x < 16) { // Scale 0
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_0);
          block->min_scale(block_scale_0);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_0) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_0) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_0) {
                if (z < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 3
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_3);
          block->min_scale(block_scale_3);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_3) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_3) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_3) {
                if (z < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-z-axis-fine-to-coarse-along-x-scale-0-3-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 8; z < 24; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
        if (x < 16) { // Scale 1
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_1);
          block->min_scale(block_scale_1);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_1) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_1) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_1) {
                if (z < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 2
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_2);
          block->min_scale(block_scale_2);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_2) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_2) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_2) {
                if (z < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_1_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-z-axis-fine-to-coarse-along-x-scale-1-2-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 8; z < 24; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
        if (x < 16) { // Scale 1
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_1);
          block->min_scale(block_scale_1);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_1) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_1) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_1) {
                if (z < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 3
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_3);
          block->min_scale(block_scale_3);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_3) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_3) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_3) {
                if (z < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-z-axis-fine-to-coarse-along-x-scale-1-3-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 8; z < 24; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
        if (x < 16) { // Scale 2
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_2);
          block->min_scale(block_scale_2);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_2) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_2) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_2) {
                if (z < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 3
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_3);
          block->min_scale(block_scale_3);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_3) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_3) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_3) {
                if (z < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-z-axis-fine-to-coarse-along-x-scale-2-3-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));
}

TEST(MeshingTest, WallCrossesZFineToCoarseAlongY) {
  typedef se::Octree<TestVoxelT> OctreeF;
  OctreeF octree;
  octree.init(32, 32);
  se::key_t allocation_list[32];
  int list_idx = 0;
  for (unsigned z = 8; z < 24; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
        allocation_list[list_idx] = octree.hash(x, y, z);
        list_idx++;
      }
    }
  }
  octree.allocate(allocation_list, 32);


  int   block_scale_0     = 0;
  int   block_stride_0    = 1 << block_scale_0;
  float block_value_0_out = 0.5f;
  float block_value_0_in  = -0.5f;

  int   block_scale_1     = 1;
  int   block_stride_1    = 1 << block_scale_0;
  float block_value_1_out = 1.f;
  float block_value_1_in  = -1.f;

  int   block_scale_2     = 2;
  int   block_stride_2    = 1 << block_scale_0;
  float block_value_2_out = 2.f;
  float block_value_2_in  = -2.f;

  int   block_scale_3     = 3;
  int   block_stride_3    = 1 << block_scale_0;
  float block_value_3_out = 4.f;
  float block_value_3_in  = -4.f;

  auto inside = [](const TestVoxelT::VoxelData& data) {
    return data.x < 0.f;
  };

  auto select_value = [](const TestVoxelT::VoxelData& data) {
    return data.x;
  };

  Eigen::Matrix4f T_MW = Eigen::Matrix4f::Identity();

  for (unsigned z = 8; z < 24; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
        if (y < 16) { // Scale 0
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_0);
          block->min_scale(block_scale_0);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_0) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_0) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_0) {
                if (z < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 1
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_1);
          block->min_scale(block_scale_1);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_1) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_1) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_1) {
                if (z < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  std::string filename = "../../out/multires-mesh-wall-crosses-z-axis-fine-to-coarse-along-y-scale-0-1-unittest.vtk";
  std::vector<se::Triangle> mesh;
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 8; z < 24; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
        if (y < 16) { // Scale 0
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_0);
          block->min_scale(block_scale_0);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_0) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_0) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_0) {
                if (z < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 2
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_2);
          block->min_scale(block_scale_2);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_2) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_2) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_2) {
                if (z < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-z-axis-fine-to-coarse-along-y-scale-0-2-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 8; z < 24; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
        if (y < 16) { // Scale 0
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_0);
          block->min_scale(block_scale_0);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_0) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_0) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_0) {
                if (z < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 3
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_3);
          block->min_scale(block_scale_3);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_3) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_3) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_3) {
                if (z < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-z-axis-fine-to-coarse-along-y-scale-0-3-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 8; z < 24; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
        if (y < 16) { // Scale 1
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_1);
          block->min_scale(block_scale_1);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_1) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_1) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_1) {
                if (z < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 2
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_2);
          block->min_scale(block_scale_2);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_2) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_2) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_2) {
                if (z < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_1_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-z-axis-fine-to-coarse-along-y-scale-1-2-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 8; z < 24; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
        if (y < 16) { // Scale 1
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_1);
          block->min_scale(block_scale_1);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_1) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_1) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_1) {
                if (z < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 3
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_3);
          block->min_scale(block_scale_3);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_3) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_3) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_3) {
                if (z < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-z-axis-fine-to-coarse-along-y-scale-1-3-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 8; z < 24; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
        if (y < 16) { // Scale 2
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_2);
          block->min_scale(block_scale_2);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_2) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_2) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_2) {
                if (z < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 3
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_3);
          block->min_scale(block_scale_3);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_3) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_3) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_3) {
                if (z < 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-z-axis-fine-to-coarse-along-y-scale-2-3-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));
}

TEST(MeshingTest, WallCrossesZCoarseToFineAlongX) {
  typedef se::Octree<TestVoxelT> OctreeF;
  OctreeF octree;
  octree.init(32, 32);
  se::key_t allocation_list[32];
  int list_idx = 0;
  for (unsigned z = 8; z < 24; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
        allocation_list[list_idx] = octree.hash(x, y, z);
        list_idx++;
      }
    }
  }
  octree.allocate(allocation_list, 32);


  int   block_scale_0     = 0;
  int   block_stride_0    = 1 << block_scale_0;
  float block_value_0_out = 0.5f;
  float block_value_0_in  = -0.5f;

  int   block_scale_1     = 1;
  int   block_stride_1    = 1 << block_scale_0;
  float block_value_1_out = 1.f;
  float block_value_1_in  = -1.f;

  int   block_scale_2     = 2;
  int   block_stride_2    = 1 << block_scale_0;
  float block_value_2_out = 2.f;
  float block_value_2_in  = -2.f;

  int   block_scale_3     = 3;
  int   block_stride_3    = 1 << block_scale_0;
  float block_value_3_out = 4.f;
  float block_value_3_in  = -4.f;

  auto inside = [](const TestVoxelT::VoxelData& data) {
    return data.x < 0.f;
  };

  auto select_value = [](const TestVoxelT::VoxelData& data) {
    return data.x;
  };

  Eigen::Matrix4f T_MW = Eigen::Matrix4f::Identity();

  for (unsigned z = 8; z < 24; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
        if (x >= 16) { // Scale 0
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_0);
          block->min_scale(block_scale_0);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_0) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_0) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_0) {
                if (z >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 1
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_1);
          block->min_scale(block_scale_1);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_1) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_1) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_1) {
                if (z >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  std::string filename = "../../out/multires-mesh-wall-crosses-z-axis-coarse-to-fine-along-x-scale-1-0-unittest.vtk";
  std::vector<se::Triangle> mesh;
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 8; z < 24; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
        if (x >= 16) { // Scale 0
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_0);
          block->min_scale(block_scale_0);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_0) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_0) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_0) {
                if (z >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 2
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_2);
          block->min_scale(block_scale_2);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_2) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_2) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_2) {
                if (z >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-z-axis-coarse-to-fine-along-x-scale-2-0-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 8; z < 24; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
        if (x >= 16) { // Scale 0
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_0);
          block->min_scale(block_scale_0);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_0) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_0) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_0) {
                if (z >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 3
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_3);
          block->min_scale(block_scale_3);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_3) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_3) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_3) {
                if (z >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-z-axis-coarse-to-fine-along-x-scale-3-0-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 8; z < 24; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
        if (x >= 16) { // Scale 1
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_1);
          block->min_scale(block_scale_1);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_1) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_1) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_1) {
                if (z >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 2
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_2);
          block->min_scale(block_scale_2);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_2) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_2) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_2) {
                if (z >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_1_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-z-axis-coarse-to-fine-along-x-scale-2-1-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 8; z < 24; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
        if (x >= 16) { // Scale 1
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_1);
          block->min_scale(block_scale_1);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_1) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_1) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_1) {
                if (z >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 3
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_3);
          block->min_scale(block_scale_3);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_3) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_3) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_3) {
                if (z >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-z-axis-coarse-to-fine-along-x-scale-3-1-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 8; z < 24; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
        if (x >= 16) { // Scale 2
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_2);
          block->min_scale(block_scale_2);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_2) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_2) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_2) {
                if (z >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 3
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_3);
          block->min_scale(block_scale_3);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_3) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_3) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_3) {
                if (z >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-z-axis-coarse-to-fine-along-x-scale-3-2-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));
}

TEST(MeshingTest, WallCrossesZCoarseToFineAlongY) {
  typedef se::Octree<TestVoxelT> OctreeF;
  OctreeF octree;
  octree.init(32, 32);
  se::key_t allocation_list[32];
  int list_idx = 0;
  for (unsigned z = 8; z < 24; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
        allocation_list[list_idx] = octree.hash(x, y, z);
        list_idx++;
      }
    }
  }
  octree.allocate(allocation_list, 32);


  int   block_scale_0     = 0;
  int   block_stride_0    = 1 << block_scale_0;
  float block_value_0_out = 0.5f;
  float block_value_0_in  = -0.5f;

  int   block_scale_1     = 1;
  int   block_stride_1    = 1 << block_scale_0;
  float block_value_1_out = 1.f;
  float block_value_1_in  = -1.f;

  int   block_scale_2     = 2;
  int   block_stride_2    = 1 << block_scale_0;
  float block_value_2_out = 2.f;
  float block_value_2_in  = -2.f;

  int   block_scale_3     = 3;
  int   block_stride_3    = 1 << block_scale_0;
  float block_value_3_out = 4.f;
  float block_value_3_in  = -4.f;

  auto inside = [](const TestVoxelT::VoxelData& data) {
    return data.x < 0.f;
  };

  auto select_value = [](const TestVoxelT::VoxelData& data) {
    return data.x;
  };

  Eigen::Matrix4f T_MW = Eigen::Matrix4f::Identity();

  for (unsigned z = 8; z < 24; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
        if (y >= 16) { // Scale 0
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_0);
          block->min_scale(block_scale_0);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_0) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_0) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_0) {
                if (z >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 1
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_1);
          block->min_scale(block_scale_1);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_1) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_1) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_1) {
                if (z >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  std::string filename = "../../out/multires-mesh-wall-crosses-z-axis-coarse-to-fine-along-y-scale-1-0-unittest.vtk";
  std::vector<se::Triangle> mesh;
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 8; z < 24; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
        if (y >= 16) { // Scale 0
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_0);
          block->min_scale(block_scale_0);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_0) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_0) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_0) {
                if (z >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 2
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_2);
          block->min_scale(block_scale_2);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_2) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_2) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_2) {
                if (z >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-z-axis-coarse-to-fine-along-y-scale-2-0-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 8; z < 24; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
        if (y >= 16) { // Scale 0
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_0);
          block->min_scale(block_scale_0);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_0) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_0) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_0) {
                if (z >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_0, {block_value_0_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 3
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_3);
          block->min_scale(block_scale_3);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_3) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_3) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_3) {
                if (z >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-z-axis-coarse-to-fine-along-y-scale-3-0-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 8; z < 24; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
        if (y >= 16) { // Scale 1
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_1);
          block->min_scale(block_scale_1);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_1) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_1) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_1) {
                if (z >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 2
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_2);
          block->min_scale(block_scale_2);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_2) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_2) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_2) {
                if (z >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_1_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-z-axis-coarse-to-fine-along-y-scale-2-1-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 8; z < 24; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
        if (y >= 16) { // Scale 1
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_1);
          block->min_scale(block_scale_1);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_1) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_1) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_1) {
                if (z >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_1, {block_value_1_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 3
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_3);
          block->min_scale(block_scale_3);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_3) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_3) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_3) {
                if (z >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-z-axis-coarse-to-fine-along-y-scale-3-1-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));

  for (unsigned z = 8; z < 24; z += VoxelBlockType::size_li) {
    for (unsigned y = 0; y < 32; y += VoxelBlockType::size_li) {
      for (unsigned x = 0; x < 32; x += VoxelBlockType::size_li) {
        if (y >= 16) { // Scale 2
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_2);
          block->min_scale(block_scale_2);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_2) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_2) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_2) {
                if (z >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_2, {block_value_2_out, 1.f});
                }
              }
            }
          }
        } else {      // Scale 3
          VoxelBlockType *block = octree.fetch(x, y, z);
          block->current_scale(block_scale_3);
          block->min_scale(block_scale_3);
          for (unsigned z_rel = 0; z_rel < VoxelBlockType::size_li; z_rel += block_stride_3) {
            for (unsigned y_rel = 0; y_rel < VoxelBlockType::size_li; y_rel += block_stride_3) {
              for (unsigned x_rel = 0; x_rel < VoxelBlockType::size_li; x_rel += block_stride_3) {
                if (z >= 16) { // Inside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_in, 1.f});
                } else {      // Outside
                  Eigen::Vector3i voxel_coord = block->coordinates() + Eigen::Vector3i(x_rel, y_rel, z_rel);
                  block->setData(voxel_coord, block_scale_3, {block_value_3_out, 1.f});
                }
              }
            }
          }
        }
      }
    }
  }

  filename = "../../out/multires-mesh-wall-crosses-z-axis-coarse-to-fine-along-y-scale-3-2-unittest.vtk";
  mesh.clear();
  se::algorithms::dual_marching_cube(octree, select_value, inside, mesh);
  save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(T_MW));
}