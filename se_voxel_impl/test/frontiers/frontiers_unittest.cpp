// SPDX-FileCopyrightText: 2021 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <gtest/gtest.h>

#include "se/io/graphviz_io.hpp"
#include "se/voxel_implementations/MultiresOFusion/MultiresOFusion.hpp"

// Include .cpp files to get access to internal functions
#include "MultiresOFusion/MultiresOFusion.cpp"
#include "MultiresOFusion/MultiresOFusion_mapping.cpp"
#include "MultiresOFusion/MultiresOFusion_rendering.cpp"



template<typename T>
constexpr T pow_const(const T base, const T exponent)
{
    return (exponent > 0 ? base * pow_const(base, exponent - 1) : 1);
}

constexpr int block_coords_to_scale(const int, const int, const int z)
{
    return std::min(std::max(z / BLOCK_SIZE, 0), 3);
}



/** Create an se::Octree with the following 3 levels:
 * 1. Root Node
 * 2. 8 Nodes
 * 3. 64 VoxelBlocks
 */
class FrontierTest : public ::testing::Test {
    public:
    // Initialize the udpate functor
    FrontierTest() :
            node_list_(block_depth_),
            depth_image_(w_, h_),
            rgba_image_(w_, h_),
            sensor_(sensor_config_),
            functor_(octree_,
                     block_list_,
                     node_list_,
                     free_list_,
                     low_variance_list_,
                     projects_inside_list_,
                     depth_image_,
                     rgba_image_,
                     fg_image_,
                     pooling_depth_image_,
                     sensor_,
                     Eigen::Matrix4f::Identity(),
                     voxel_dim_,
                     voxel_depth_,
                     sensor_config_.far_plane,
                     1)
    {
    }

    protected:
    virtual void SetUp()
    {
        // Configure the voxel implementation
        MultiresOFusion::configure(voxel_dim_);
        // Initialize the octree
        octree_.init(size_, dim_);
        // Compute the Morton codes of all VoxelBlocks
        std::vector<se::key_t> alloc_list;
        for (int z = 0; z < size_; z += BLOCK_SIZE) {
            for (int y = 0; y < size_; y += BLOCK_SIZE) {
                for (int x = 0; x < size_; x += BLOCK_SIZE) {
                    alloc_list.push_back(octree_.hash(x, y, z));
                }
            }
        }
        // Allocate all Nodes/VoxelBlocks in the octree
        octree_.allocate(alloc_list.data(), alloc_list.size());
        // Allocate VoxelBlocks to different scales depending on their z coordinate
        for (int z = 0; z < size_; z += BLOCK_SIZE) {
            for (int y = 0; y < size_; y += BLOCK_SIZE) {
                for (int x = 0; x < size_; x += BLOCK_SIZE) {
                    auto* block = octree_.fetch(x, y, z);
                    // This will allocate voxels at all scales because the octree is 4x4x4 VoxelBlocks
                    block->allocateDownTo(block_coords_to_scale(x, y, z));
                    block_list_.push_back(block);
                }
            }
        }
        // Keep track of frontier VoxelBlocks
        candidate_frontiers_.insert(alloc_list.begin(), alloc_list.end());

        // Create some frontier voxels
        data_ = MultiresOFusion::VoxelType::initData();
        data_.x = MultiresOFusion::surface_boundary - 1.0f;
        data_.y = 1;
        data_.observed = true;
        data_.frontier = true;
        for (int z = 0; z < size_; z += BLOCK_SIZE) {
            octree_.set(0, 0, z, data_);
            num_frontier_voxels_++;
            // Keep track of the Morton codes of the expected frontier VoxelBlocks
            expected_frontiers_.insert(octree_.hash(0, 0, z));
        }

        se::to_graphviz(octree_,
                        std::string("/tmp/frontier_unittest_octree.gv"),
                        MultiresOFusion::VoxelType::to_graphviz_label);
    }

    // Octree
    static constexpr int size_ = 4 * BLOCK_SIZE;
    static constexpr float dim_ = 1.0f;
    static constexpr float voxel_dim_ = dim_ / size_;
    static constexpr int voxel_depth_ = se::math::log2_const(size_);
    static constexpr int block_depth_ = voxel_depth_ - se::math::log2_const(BLOCK_SIZE);
    se::Octree<MultiresOFusion::VoxelType> octree_;
    std::vector<MultiresOFusion::VoxelBlockType*> block_list_;
    std::vector<std::set<se::Node<MultiresOFusion::VoxelType>*>> node_list_;
    std::set<se::key_t> candidate_frontiers_;
    std::set<se::key_t> expected_frontiers_;
    size_t num_frontier_voxels_ = 0;
    MultiresOFusion::VoxelType::VoxelData data_;
    // Update functor
    static constexpr int w_ = 320;
    static constexpr int h_ = 240;
    std::vector<bool> free_list_;
    std::vector<bool> low_variance_list_;
    std::vector<bool> projects_inside_list_;
    se::Image<float> depth_image_;
    se::Image<uint32_t> rgba_image_;
    cv::Mat fg_image_;
    se::DensePoolingImage<SensorImpl>* pooling_depth_image_ = nullptr;
    se::SensorConfig sensor_config_ =
        {w_, h_, false, 0.1f, 10.0f, 525.0f, 525.0f, w_ / 2.0f, h_ / 2.0f};
    SensorImpl sensor_;
    MultiresOFusionUpdate<se::PinholeCamera> functor_;
};



TEST_F(FrontierTest, initialization)
{
    // Ensure that the VoxelBlocks were allocated only up to the desired scale.
    for (int z = 0; z < size_; z += BLOCK_SIZE) {
        for (int y = 0; y < size_; y += BLOCK_SIZE) {
            for (int x = 0; x < size_; x += BLOCK_SIZE) {
                auto* block = octree_.fetch(x, y, z);
                ASSERT_NE(block, nullptr);
                EXPECT_EQ(block->min_scale(), block_coords_to_scale(x, y, z));
            }
        }
    }
    // Test that data at higher scales of the updated VoxelBlocks is uninitialized
    for (int z = 0; z < size_; z += BLOCK_SIZE) {
        auto* block = octree_.fetch(0, 0, z);
        ASSERT_NE(block, nullptr);
        // Check all scales above the minimum allocated scale
        for (int scale = block->min_scale() + 1; scale < block->max_scale; scale++) {
            const auto data = block->data(Eigen::Vector3i(0, 0, z), scale);
            EXPECT_FALSE(MultiresOFusion::VoxelType::isValid(data));
            EXPECT_EQ(data, MultiresOFusion::VoxelType::initData());
        }
    }
    // Test that data at the parent Nodes of the updated VoxelBlocks is uninitialized
    for (int depth = 0; depth < octree_.blockDepth() - 1; depth++) {
        const se::Node<MultiresOFusion::VoxelType>* node = octree_.fetchNode(0, 0, 0, depth);
        ASSERT_NE(node, nullptr);
        // Get the data of the 2 children with x = y = 0
        const auto data_0 = node->childData(0 + 0 * 2 + 0 * 4);
        const auto data_1 = node->childData(0 + 0 * 2 + 1 * 4);
        EXPECT_EQ(data_0, MultiresOFusion::VoxelType::initData());
        EXPECT_EQ(data_1, MultiresOFusion::VoxelType::initData());
    }
}



TEST_F(FrontierTest, frontierVolumes)
{
    std::vector<se::Volume<MultiresOFusion::VoxelType>> volumes =
        se::frontier_volumes(octree_, candidate_frontiers_);
    EXPECT_EQ(volumes.size(), num_frontier_voxels_);
    for (const auto& volume : volumes) {
        // Convert to voxel coordinates
        const Eigen::Vector3f coord_M =
            volume.centre_M - Eigen::Vector3f::Constant(volume.dim / 2.0f);
        const Eigen::Vector3i coord = octree_.pointToVoxel(coord_M);
        EXPECT_EQ(coord.x(), 0);
        EXPECT_EQ(coord.y(), 0);
        EXPECT_EQ(coord.z() % BLOCK_SIZE, 0);
        EXPECT_EQ(volume.data, data_);
        // Due to the way the frontiers were initialized their size depends on their z coordinate
        const int expected_size = pow_const(2, (coord.z() / BLOCK_SIZE));
        EXPECT_EQ(volume.size, expected_size);
    }
}



TEST_F(FrontierTest, upPropagation)
{
    // Up-propagation of data only inside VoxelBlocks
    for (auto block : block_list_) {
        updating_model::propagateBlockToCoarsestScale(block);
    }
    // Test that data at higher scales is now valid
    for (int z = 0; z < size_; z += BLOCK_SIZE) {
        auto* block = octree_.fetch(0, 0, z);
        ASSERT_NE(block, nullptr);
        // Check all scales above the minimum allocated scale
        for (int scale = block->min_scale() + 1; scale <= block->max_scale; scale++) {
            const auto data = block->data(Eigen::Vector3i(0, 0, z), scale);
            EXPECT_TRUE(MultiresOFusion::VoxelType::isValid(data));
            EXPECT_FLOAT_EQ(data.x, data_.x);
            EXPECT_FLOAT_EQ(data.fg, data_.fg);
            EXPECT_EQ(data.y, data_.y);
            EXPECT_EQ(data.observed, false);
            EXPECT_EQ(data.frontier, data_.frontier);
        }
    }

    // Up-propagate of data along Nodes all the way to the root
    functor_.propagateToRoot();
    se::to_graphviz(octree_,
                    std::string("/tmp/frontier_unittest_octree_prop.gv"),
                    MultiresOFusion::VoxelType::to_graphviz_label);
    // Test that data at the parent Nodes of the updated VoxelBlocks is valid
    for (int depth = 0; depth < octree_.blockDepth() - 1; depth++) {
        const se::Node<MultiresOFusion::VoxelType>* node = octree_.fetchNode(0, 0, 0, depth);
        ASSERT_NE(node, nullptr);
        // Get the data of the 2 children with x = y = 0
        const auto data_0 = node->childData(0 + 0 * 2 + 0 * 4);
        const auto data_1 = node->childData(0 + 0 * 2 + 1 * 4);
        EXPECT_TRUE(MultiresOFusion::VoxelType::isValid(data_0));
        EXPECT_FLOAT_EQ(data_0.x, data_.x);
        EXPECT_FLOAT_EQ(data_0.fg, data_.fg);
        EXPECT_EQ(data_0.y, data_.y);
        EXPECT_EQ(data_0.observed, false);
        EXPECT_EQ(data_0.frontier, data_.frontier);
        EXPECT_TRUE(MultiresOFusion::VoxelType::isValid(data_1));
        EXPECT_FLOAT_EQ(data_1.x, data_.x);
        EXPECT_FLOAT_EQ(data_1.fg, data_.fg);
        EXPECT_EQ(data_1.y, data_.y);
        EXPECT_EQ(data_1.observed, false);
        EXPECT_EQ(data_1.frontier, data_.frontier);
    }
}

TEST_F(FrontierTest, isFrontier)
{
    // Test for frontiers at the finest allocated scale
    for (int scale = 0; scale <= MultiresOFusion::VoxelBlockType::max_scale; scale++) {
        const int z = BLOCK_SIZE * scale;
        const Eigen::Vector3i coord(0, 0, z);
        const auto* block = octree_.fetch(coord);
        EXPECT_TRUE(is_frontier(coord, scale, *block, octree_));
    }
}

TEST_F(FrontierTest, updateFrontierData)
{
    // Update all VoxelBlocks with frontiers and test the new frontier volume
    for (int scale = 0; scale <= MultiresOFusion::VoxelBlockType::max_scale; scale++) {
        const int z = BLOCK_SIZE * scale;
        const Eigen::Vector3i coord(0, 0, z);
        se::Node<MultiresOFusion::VoxelType>* node = octree_.fetch(coord);
        ASSERT_NE(node, nullptr);
        const int frontier_volume = se::update_frontier_data(*node, octree_);
        // We expect to have only a single frontier voxel at each VoxelBlock
        const int expected_frontier_volume =
            se::math::cu(MultiresOFusion::VoxelBlockType::scaleVoxelSize(scale));
        EXPECT_EQ(frontier_volume, expected_frontier_volume);
    }
}

TEST_F(FrontierTest, updateFrontiers)
{
    // Update the frontiers
    constexpr int min_frontier_volume = 0;
    std::set<se::key_t> updated_frontiers(candidate_frontiers_);
    se::update_frontiers(octree_, updated_frontiers, min_frontier_volume);
    // Test that only the actual frontier VoxelBlocks remain
    EXPECT_EQ(updated_frontiers.size(), expected_frontiers_.size());
    for (const auto& code : expected_frontiers_) {
        EXPECT_EQ(updated_frontiers.count(code), 1lu);
    }

    // Test that the frontier voxels remained the same
    std::vector<se::Volume<MultiresOFusion::VoxelType>> volumes =
        se::frontier_volumes(octree_, updated_frontiers);
    EXPECT_EQ(volumes.size(), num_frontier_voxels_);
    for (const auto& volume : volumes) {
        // Convert to voxel coordinates
        const Eigen::Vector3f coord_M =
            volume.centre_M - Eigen::Vector3f::Constant(volume.dim / 2.0f);
        const Eigen::Vector3i coord = octree_.pointToVoxel(coord_M);
        EXPECT_EQ(coord.x(), 0);
        EXPECT_EQ(coord.y(), 0);
        EXPECT_EQ(coord.z() % BLOCK_SIZE, 0);
        EXPECT_EQ(volume.data, data_);
        // Due to the way the frontiers were initialized their size depends on their z coordinate
        const int expected_size = pow_const(2, (coord.z() / BLOCK_SIZE));
        EXPECT_EQ(volume.size, expected_size);
    }
}
