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

#include <gtest/gtest.h>
#include <random>
#include <se/octree.hpp>
#include <se/utils/math_utils.h>
#include <se/utils/morton_utils.hpp>

struct TestVoxelT {
    typedef float VoxelData;
    static inline VoxelData invalid()
    {
        return 0.f;
    }
    static inline VoxelData initData()
    {
        return 0.f;
    }

    using VoxelBlockType = se::VoxelBlockFull<TestVoxelT>;

    using MemoryPoolType = se::PagedMemoryPool<TestVoxelT>;
    template<typename BufferT>
    using MemoryBufferType = se::PagedMemoryBuffer<BufferT>;
};

TEST(AllocationTest, EmptySingleVoxel)
{
    typedef se::Octree<TestVoxelT> OctreeF;
    OctreeF octree;
    octree.init(256, 5);
    const Eigen::Vector3i voxel_coord = {25, 65, 127};
    TestVoxelT::VoxelData data;
    octree.get(voxel_coord.x(), voxel_coord.y(), voxel_coord.z(), data);
    EXPECT_EQ(data, TestVoxelT::invalid());
}

TEST(AllocationTest, SetSingleVoxel)
{
    typedef se::Octree<TestVoxelT> OctreeF;
    OctreeF octree;
    octree.init(256, 5);
    const Eigen::Vector3i voxel_coord = {25, 65, 127};
    const se::key_t code = octree.hash(voxel_coord.x(), voxel_coord.y(), voxel_coord.z());
    se::key_t allocation_list[1] = {code};
    octree.allocate(allocation_list, 1);

    TestVoxelT::VoxelBlockType* block =
        octree.fetch(voxel_coord.x(), voxel_coord.y(), voxel_coord.z());
    TestVoxelT::VoxelData written_data = 2.f;
    block->setData(voxel_coord, written_data);

    TestVoxelT::VoxelData read_data;
    octree.get(voxel_coord.x(), voxel_coord.y(), voxel_coord.z(), read_data);
    EXPECT_EQ(written_data, read_data);
}

TEST(AllocationTest, FetchOctant)
{
    typedef se::Octree<TestVoxelT> OctreeF;
    OctreeF octree;
    const int voxel_depth = 8;
    const unsigned int size = std::pow(2, voxel_depth);
    octree.init(size, 5);
    const Eigen::Vector3i voxel_coord = {25, 65, 127};
    const se::key_t code = octree.hash(voxel_coord.x(), voxel_coord.y(), voxel_coord.z());
    se::key_t allocation_list[1] = {code};
    octree.allocate(allocation_list, 1);

    const int depth = 3; /* 32 voxels per side */
    se::Node<TestVoxelT>* node =
        octree.fetchNode(voxel_coord.x(), voxel_coord.y(), voxel_coord.z(), depth);
    se::key_t fetched_code = node->code();

    const se::key_t gt_code = octree.hash(voxel_coord.x(), voxel_coord.y(), voxel_coord.z(), depth);
    ASSERT_EQ(fetched_code, gt_code);
}

TEST(AllocationTest, MortonPrefixMask)
{
    const unsigned int max_bits = 21;
    const unsigned int block_size = 8;
    const unsigned int map_size = std::pow(2, max_bits);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, map_size);

    constexpr int num_samples = 10;
    se::key_t keys[num_samples];
    se::key_t tempkeys[num_samples];
    Eigen::Vector3i voxels_coord[num_samples];

    for (int i = 0; i < num_samples; ++i) {
        const Eigen::Vector3i voxel_coord = {dis(gen), dis(gen), dis(gen)};
        voxels_coord[i] = Eigen::Vector3i(voxel_coord);
        const se::key_t code = compute_morton(voxel_coord.x(), voxel_coord.y(), voxel_coord.z());
        keys[i] = code;
    }

    const int voxel_depth = log2(map_size);
    const int block_depth = voxel_depth - log2(block_size);
    const unsigned int shift = max_bits - voxel_depth;
    int node_size = map_size / 2;
    for (int depth = 0; depth <= block_depth; depth++) {
        const se::key_t mask = MASK[depth + shift];
        compute_prefix(keys, tempkeys, num_samples, mask);
        for (int i = 0; i < num_samples; ++i) {
            const Eigen::Vector3i masked_voxel_coord = unpack_morton(tempkeys[i]);
            ASSERT_EQ(masked_voxel_coord.x() % node_size, 0);
            ASSERT_EQ(masked_voxel_coord.y() % node_size, 0);
            ASSERT_EQ(masked_voxel_coord.z() % node_size, 0);
            // printf("voxel_coord: %d, %d, %d\n", voxel_coord.x(), voxel_coord.y(), voxel_coord.z());
            // printf("masked depth %d: %d, %d, %d\n", depth, masked_voxel_coord.x(), masked_voxel_coord.y(), masked_voxel_coord.z() );
        }
        node_size = node_size / 2;
    }
}

TEST(AllocationTest, ParentInsert)
{
    se::Octree<TestVoxelT> octree;
    const int voxel_depth = 8;
    const unsigned int map_size = std::pow(2, voxel_depth);
    octree.init(map_size, 5);
    const int block_depth = octree.blockDepth();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, map_size);
    const Eigen::Vector3i voxel_coord = {dis(gen), dis(gen), dis(gen)};
    octree.insert(voxel_coord.x(), voxel_coord.y(), voxel_coord.z());
    TestVoxelT::VoxelBlockType* block =
        octree.fetch(voxel_coord.x(), voxel_coord.y(), voxel_coord.z());
    EXPECT_NE(block, nullptr);
    se::Node<TestVoxelT>* parent = block->parent();
    for (int depth = block_depth - 1; depth >= 0; depth--) {
        se::Node<TestVoxelT>* node =
            octree.fetchNode(voxel_coord.x(), voxel_coord.y(), voxel_coord.z(), depth);
        ASSERT_EQ(parent, node);
        parent = parent->parent();
    }
}

TEST(AllocationTest, ParentAllocation)
{
    se::Octree<TestVoxelT> octree;
    const int voxel_depth = 8;
    const unsigned int map_size = std::pow(2, voxel_depth);
    octree.init(map_size, 5);
    std::random_device rd;
    const int block_depth = octree.blockDepth();
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, map_size);
    const Eigen::Vector3i voxel_coord = {dis(gen), dis(gen), dis(gen)};
    const unsigned code = octree.hash(voxel_coord.x(), voxel_coord.y(), voxel_coord.z());
    se::key_t allocation_list[1] = {code};
    octree.allocate(allocation_list, 1);

    TestVoxelT::VoxelBlockType* block =
        octree.fetch(voxel_coord.x(), voxel_coord.y(), voxel_coord.z());
    EXPECT_NE(block, nullptr);
    se::Node<TestVoxelT>* parent = block->parent();
    for (int depth = block_depth - 1; depth >= 0; depth--) {
        se::Node<TestVoxelT>* node =
            octree.fetchNode(voxel_coord.x(), voxel_coord.y(), voxel_coord.z(), depth);
        ASSERT_EQ(parent, node);
        parent = parent->parent();
    }
}



struct ThresholdVoxelT {
    typedef float VoxelData;
    static inline VoxelData invalid()
    {
        return 0.f;
    }
    static inline VoxelData initData()
    {
        return 1.f;
    }

    using VoxelBlockType = se::VoxelBlockSingleMax<ThresholdVoxelT>;

    static float selectVoxelValue(const VoxelData& data)
    {
        return data;
    };

    static float isValid(const VoxelData& /* data */)
    {
        return true;
    };

    static float threshold(const VoxelData& data)
    {
        return data;
    };

    using MemoryPoolType = se::MemoryPool<ThresholdVoxelT>;
    template<typename ElemT>
    using MemoryBufferType = std::vector<ElemT>;
};

TEST(Octree, Threshold)
{
    const int voxel_depth = 6;

    se::Octree<ThresholdVoxelT> octree;
    octree.init(1 << voxel_depth, 5);
    Eigen::Vector3i voxel_coord(0, 0, 0);
    octree.insert(voxel_coord.x(), voxel_coord.y(), voxel_coord.z());

    se::Node<ThresholdVoxelT>* node = octree.root();

    int value = 10;
    while (!node->isBlock()) {
        for (int child_idx = 0; child_idx < 8; child_idx++) {
            node->childData(child_idx) = value;
        }

        node = node->child(0);
        value--;
    }

    ThresholdVoxelT::VoxelBlockType* block = static_cast<ThresholdVoxelT::VoxelBlockType*>(node);
    int min_scale = 1;
    value++;
    block->allocateDownTo(min_scale);
    block->current_scale(min_scale);
    for (int s = ThresholdVoxelT::VoxelBlockType::max_scale; s >= min_scale; s--) {
        block->setMaxData(voxel_coord, s, value);
        value--;
    }

    /// Start testing

    // depth = 0 | scale = 6 | size = 64 | <= root
    // depth = 1 | scale = 5 | size = 32 | max_value = 10
    // depth = 2 | scale = 4 | size = 16 | max_value =  9
    // depth = 3 | scale = 3 | size =  8 | max_value =  8
    // depth = 4 | scale = 2 | size =  4 | max_value =  7
    // depth = 5 | scale = 1 | size =  2 | max_value =  6 <= current scale
    // depth = 6 | scale = 0 | size =  1 | max_value =  5 <= voxel depth

    /// Test 8

    ThresholdVoxelT::VoxelData v_data = ThresholdVoxelT::initData();
    int v_size = -1;
    Eigen::Vector3i v_corner = Eigen::Vector3i::Constant(-1);
    bool is_finest = false;

    float t_value = 5.1;
    float t_size = 1;
    octree.getThreshold(voxel_coord
                            + Eigen::Vector3i::Constant(ThresholdVoxelT::VoxelBlockType::size_li),
                        t_value,
                        t_size,
                        v_data,
                        v_size,
                        v_corner,
                        is_finest);
    ASSERT_EQ(v_data, 8);
    ASSERT_EQ(v_size, 8);
    ASSERT_EQ(is_finest, true);

    /// Test 7

    v_data = ThresholdVoxelT::initData();
    v_size = -1;
    v_corner = Eigen::Vector3i::Constant(-1);
    is_finest = true;

    t_value = 5.1;
    t_size = 16;

    octree.getThreshold(voxel_coord
                            + Eigen::Vector3i::Constant(ThresholdVoxelT::VoxelBlockType::size_li),
                        t_value,
                        t_size,
                        v_data,
                        v_size,
                        v_corner,
                        is_finest);
    ASSERT_EQ(v_data, 9);
    ASSERT_EQ(v_size, 16);
    ASSERT_EQ(is_finest, false);

    /// Test 6

    v_data = ThresholdVoxelT::initData();
    v_size = -1;
    v_corner = Eigen::Vector3i::Constant(-1);
    is_finest = false;

    t_value = 5.1;
    t_size = 8;

    octree.getThreshold(voxel_coord
                            + Eigen::Vector3i::Constant(ThresholdVoxelT::VoxelBlockType::size_li),
                        t_value,
                        t_size,
                        v_data,
                        v_size,
                        v_corner,
                        is_finest);
    ASSERT_EQ(v_data, 8);
    ASSERT_EQ(v_size, 8);
    ASSERT_EQ(is_finest, true);

    /// Test 5

    v_data = ThresholdVoxelT::initData();
    v_size = -1;
    v_corner = Eigen::Vector3i::Constant(-1);
    is_finest = false;

    t_value = 6.1;
    t_size = 2;

    octree.getThreshold(voxel_coord, t_value, t_size, v_data, v_size, v_corner, is_finest);
    ASSERT_EQ(v_data, 6);
    ASSERT_EQ(v_size, 2);
    ASSERT_EQ(is_finest, true);

    /// Test 4

    v_data = ThresholdVoxelT::initData();
    v_size = -1;
    v_corner = Eigen::Vector3i::Constant(-1);
    is_finest = false;

    t_value = 6.1;
    t_size = 2;

    octree.getThreshold(voxel_coord, t_value, t_size, v_data, v_size, v_corner, is_finest);
    ASSERT_EQ(v_data, 6);
    ASSERT_EQ(v_size, 2);
    ASSERT_EQ(is_finest, true);

    /// Test 3

    v_data = ThresholdVoxelT::initData();
    v_size = -1;
    v_corner = Eigen::Vector3i::Constant(-1);
    is_finest = true;

    t_value = 6.1;
    t_size = 1;

    octree.getThreshold(voxel_coord, t_value, t_size, v_data, v_size, v_corner, is_finest);
    ASSERT_EQ(v_data, 6);
    ASSERT_EQ(v_size, 2);
    ASSERT_EQ(is_finest, true);

    /// Test 2

    v_data = ThresholdVoxelT::initData();
    v_size = -1;
    v_corner = Eigen::Vector3i::Constant(-1);
    is_finest = true;

    t_value = 8.1;
    t_size = 16;
    octree.getThreshold(voxel_coord, t_value, t_size, v_data, v_size, v_corner, is_finest);
    ASSERT_EQ(v_data, 9);
    ASSERT_EQ(v_size, 16);
    ASSERT_EQ(is_finest, false);

    /// Test 1

    v_data = ThresholdVoxelT::initData();
    v_size = -1;
    v_corner = Eigen::Vector3i::Constant(-1);
    is_finest = true;

    t_value = 9.1;
    t_size = 8;
    octree.getThreshold(voxel_coord, t_value, t_size, v_data, v_size, v_corner, is_finest);
    ASSERT_EQ(v_data, 9);
    ASSERT_EQ(v_size, 16);
    ASSERT_EQ(is_finest, false);
}
