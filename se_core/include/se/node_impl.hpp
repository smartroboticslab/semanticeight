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

#ifndef NODE_IMPL_HPP
#define NODE_IMPL_HPP

#include <algorithm>
#include <cstdlib>

#include "se/octant_ops.hpp"

namespace se {
// Node implementation

template<typename T>
Node<T>::Node(const typename T::VoxelData init_data) :
        code_(0), size_(0), children_mask_(0), timestamp_(0)
{
    for (unsigned int child_idx = 0; child_idx < 8; child_idx++) {
        children_data_[child_idx] = init_data;
        children_ptr_[child_idx] = nullptr;
        parent_ptr_ = nullptr;
    }
}

template<typename T>
Node<T>::Node(const Node<T>& node)
{
    initFromNode(node);
}

template<typename T>
void Node<T>::operator=(const Node<T>& node)
{
    initFromNode(node);
}

template<typename T>
const typename T::VoxelData& Node<T>::data() const
{
    // Used so we can return a reference to invalid data
    static const typename T::VoxelData invalid = T::invalid();
    // Return invalid data if this is the root node
    if (parent_ptr_ == nullptr) {
        return invalid;
    }
    // Find the child index of this node in its parent
    // TODO this is not the most efficient way but computing the child index directly requires the
    // octree dimensions
    int child_idx = 0;
    for (; child_idx < 8; child_idx++) {
        if (parent_ptr_->child(child_idx) == this) {
            break;
        }
    }
    if (child_idx < 8) {
        return parent_ptr_->childData(child_idx);
    }
    else {
        // The parent does not contain a pointer to this node, broken octree
        return invalid;
    }
}

template<typename T>
void Node<T>::initFromNode(const se::Node<T>& node)
{
    code_ = node.code();
    size_ = node.size_;
    children_mask_ = node.children_mask();
    timestamp_ = node.timestamp();
    active_ = node.active();
    std::copy(node.childrenData(), node.childrenData() + 8, children_data_);
}

template<typename T>
key_t Node<T>::childCode(const int child_idx, const int voxel_depth) const
{
    assert(0 <= child_idx && child_idx < 8);

    const Eigen::Vector3i child_coord = childCoord(child_idx);
    const int child_depth = voxel_depth - log2(size_) + 1;
    return keyops::encode(
        child_coord.x(), child_coord.y(), child_coord.z(), child_depth, voxel_depth);
}

template<typename T>
Eigen::Vector3i Node<T>::coordinates() const
{
    return se::keyops::decode(code_);
}

template<typename T>
Eigen::Vector3i Node<T>::centreCoordinates() const
{
    return coordinates() + Eigen::Vector3i::Constant(size_ / 2);
}

template<typename T>
Eigen::Vector3i Node<T>::childCoord(const int child_idx) const
{
    assert(0 <= child_idx && child_idx < 8);

    const std::div_t d1 = std::div(child_idx, 4);
    const std::div_t d2 = std::div(d1.rem, 2);
    const int rel_z = d1.quot;
    const int rel_y = d2.quot;
    const int rel_x = d2.rem;
    const int child_size = size_ / 2;
    return coordinates() + child_size * Eigen::Vector3i(rel_x, rel_y, rel_z);
}

template<typename T>
Eigen::Vector3i Node<T>::childCentreCoord(const int child_idx) const
{
    assert(0 <= child_idx && child_idx < 8);

    const int child_size = size_ / 2;
    return childCoord(child_idx) + Eigen::Vector3i::Constant(child_size / 2);
}



// Voxel block base implementation

template<typename T>
VoxelBlock<T>::VoxelBlock(const int current_scale, const int min_scale) :
        coordinates_(Eigen::Vector3i::Constant(0)),
        current_scale_(current_scale),
        min_scale_(min_scale),
        min_scale_reached_(min_scale)
{
    this->size_ = BLOCK_SIZE;
}

template<typename T>
VoxelBlock<T>::VoxelBlock(const VoxelBlock<T>& block)
{
    initFromBlock(block);
}

template<typename T>
void VoxelBlock<T>::operator=(const VoxelBlock<T>& block)
{
    initFromBlock(block);
}

template<typename T>
Eigen::Vector3i VoxelBlock<T>::voxelCoordinates(const int voxel_idx) const
{
    assert(voxel_idx >= 0);

    int remaining_voxel_idx = voxel_idx;
    int scale = 0;
    int size_at_scale_cu = this->size_cu;
    while (remaining_voxel_idx / size_at_scale_cu >= 1) {
        scale += 1;
        remaining_voxel_idx -= size_at_scale_cu;
        size_at_scale_cu = scaleNumVoxels(scale);
    }
    return voxelCoordinates(remaining_voxel_idx, scale);
}

template<typename T>
Eigen::Vector3i VoxelBlock<T>::voxelCoordinates(const int voxel_idx, const int scale) const
{
    assert(0 <= scale && scale <= VoxelBlock<T>::max_scale);
    assert(voxel_idx >= 0);

    const std::div_t d1 = std::div(voxel_idx, se::math::sq(scaleSize(scale)));
    const std::div_t d2 = std::div(d1.rem, scaleSize(scale));
    const int z = d1.quot;
    const int y = d2.quot;
    const int x = d2.rem;
    return this->coordinates_ + scaleVoxelSize(scale) * Eigen::Vector3i(x, y, z);
}

template<typename T>
bool VoxelBlock<T>::contains(const Eigen::Vector3i& voxel_coord) const
{
    const Eigen::Vector3i voxel_offset = voxel_coord - coordinates_;
    return 0 <= voxel_offset.x() && voxel_offset.x() < static_cast<int>(size_li)
        && 0 <= voxel_offset.y() && voxel_offset.y() < static_cast<int>(size_li)
        && 0 <= voxel_offset.z() && voxel_offset.z() < static_cast<int>(size_li);
}

template<typename T>
constexpr int VoxelBlock<T>::scaleSize(const int scale)
{
    assert(0 <= scale && scale <= VoxelBlock<T>::max_scale);

    return size_li >> scale;
}

template<typename T>
constexpr int VoxelBlock<T>::scaleVoxelSize(const int scale)
{
    assert(0 <= scale && scale <= VoxelBlock<T>::max_scale);

    return 1 << scale;
}

template<typename T>
constexpr int VoxelBlock<T>::scaleNumVoxels(const int scale)
{
    assert(0 <= scale && scale <= VoxelBlock<T>::max_scale);

    return se::math::cu(scaleSize(scale));
}

template<typename T>
constexpr int VoxelBlock<T>::scaleOffset(const int scale)
{
    assert(0 <= scale && scale <= VoxelBlock<T>::max_scale);

    int scale_offset = 0;
    for (int s = 0; s < scale; ++s) {
        scale_offset += scaleNumVoxels(s);
    }
    return scale_offset;
}

template<typename T>
void VoxelBlock<T>::updateMinScaleReached()
{
    if (min_scale_reached_ < 0 || (min_scale_ >= 0 && min_scale_ < min_scale_reached_)) {
        min_scale_reached_ = min_scale_;
    }
}

template<typename T>
void VoxelBlock<T>::initFromBlock(const VoxelBlock<T>& block)
{
    this->code_ = block.code();
    this->size_ = block.size_;
    this->children_mask_ = block.children_mask();
    this->timestamp_ = block.timestamp();
    this->active_ = block.active();
    coordinates_ = block.coordinates();
    min_scale_ = block.min_scale();
    min_scale_reached_ = block.minScaleReached();
    current_scale_ = block.current_scale();
    std::copy(block.childrenData(), block.childrenData() + 8, this->children_data_);
}



// Voxel block finest scale allocation implementation

template<typename T>
VoxelBlockFinest<T>::VoxelBlockFinest(const typename T::VoxelData init_data) : VoxelBlock<T>(0, 0)
{
    for (unsigned int voxel_idx = 0; voxel_idx < num_voxels_in_block; voxel_idx++) {
        block_data_[voxel_idx] = init_data;
    }
}

template<typename T>
VoxelBlockFinest<T>::VoxelBlockFinest(const VoxelBlockFinest<T>& block)
{
    initFromBlock(block);
}

template<typename T>
void VoxelBlockFinest<T>::operator=(const VoxelBlockFinest<T>& block)
{
    initFromBlock(block);
}

template<typename T>
inline typename VoxelBlock<T>::VoxelData
VoxelBlockFinest<T>::data(const Eigen::Vector3i& voxel_coord) const
{
    Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
    return block_data_[voxel_offset.x() + voxel_offset.y() * this->size_li
                       + voxel_offset.z() * this->size_sq];
}

template<typename T>
inline typename VoxelBlock<T>::VoxelData
VoxelBlockFinest<T>::data(const Eigen::Vector3i& voxel_coord, const int /* scale_0 */) const
{
    return data(voxel_coord);
}

template<typename T>
inline void VoxelBlockFinest<T>::setData(const Eigen::Vector3i& voxel_coord,
                                         const VoxelData& voxel_data)
{
    Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
    block_data_[voxel_offset.x() + voxel_offset.y() * this->size_li
                + voxel_offset.z() * this->size_sq] = voxel_data;
}

template<typename T>
inline void VoxelBlockFinest<T>::setData(const Eigen::Vector3i& voxel_coord,
                                         const int /* scale_0 */,
                                         const VoxelData& voxel_data)
{
    setData(voxel_coord, voxel_data);
}

template<typename T>
inline typename VoxelBlock<T>::VoxelData VoxelBlockFinest<T>::data(const int voxel_idx) const
{
    return block_data_[voxel_idx];
}

template<typename T>
inline void VoxelBlockFinest<T>::setData(const int voxel_idx, const VoxelData& voxel_data)
{
    block_data_[voxel_idx] = voxel_data;
}

template<typename T>
inline typename VoxelBlock<T>::VoxelData VoxelBlockFinest<T>::data(const int voxel_idx_at_scale_0,
                                                                   const int /* scale_0 */) const
{
    return block_data_[voxel_idx_at_scale_0];
}

template<typename T>
inline void VoxelBlockFinest<T>::setData(const int voxel_idx,
                                         const int /* scale_0 */,
                                         const VoxelData& voxel_data)
{
    block_data_[voxel_idx] = voxel_data;
}

template<typename T>
void VoxelBlockFinest<T>::initFromBlock(const VoxelBlockFinest<T>& block)
{
    this->code_ = block.code();
    this->size_ = block.size_;
    this->children_mask_ = block.children_mask();
    this->timestamp_ = block.timestamp();
    this->active_ = block.active();
    this->coordinates_ = block.coordinates();
    this->min_scale_ = block.min_scale();
    this->min_scale_reached_ = block.minScaleReached();
    this->current_scale_ = block.current_scale();
    std::copy(block.childrenData(), block.childrenData() + 8, this->children_data_);
    std::copy(block.blockData(), block.blockData() + num_voxels_in_block, blockData());
}



// Voxel block full scale allocation implementation

template<typename T>
VoxelBlockFull<T>::VoxelBlockFull(const typename T::VoxelData init_data) : VoxelBlock<T>(0, -1)
{
    for (unsigned int voxel_idx = 0; voxel_idx < num_voxels_in_block; voxel_idx++) {
        block_data_[voxel_idx] = init_data;
    }
}

template<typename T>
VoxelBlockFull<T>::VoxelBlockFull(const VoxelBlockFull<T>& block)
{
    initFromBlock(block);
}

template<typename T>
void VoxelBlockFull<T>::operator=(const VoxelBlockFull<T>& block)
{
    initFromBlock(block);
}

template<typename T>
inline typename VoxelBlock<T>::VoxelData
VoxelBlockFull<T>::data(const Eigen::Vector3i& voxel_coord) const
{
    Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
    return block_data_[voxel_offset.x() + voxel_offset.y() * this->size_li
                       + voxel_offset.z() * this->size_sq];
}

template<typename T>
inline typename VoxelBlock<T>::VoxelData VoxelBlockFull<T>::data(const Eigen::Vector3i& voxel_coord,
                                                                 const int scale) const
{
    Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
    int scale_offset = 0;
    int scale_tmp = 0;
    int num_voxels = this->size_cu;
    while (scale_tmp < scale) {
        scale_offset += num_voxels;
        num_voxels /= 8;
        ++scale_tmp;
    }
    const int local_size = this->size_li / (1 << scale);
    voxel_offset = voxel_offset / (1 << scale);
    return block_data_[scale_offset + voxel_offset.x() + voxel_offset.y() * local_size
                       + voxel_offset.z() * se::math::sq(local_size)];
}

template<typename T>
inline void VoxelBlockFull<T>::setData(const Eigen::Vector3i& voxel_coord,
                                       const VoxelData& voxel_data)
{
    Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
    block_data_[voxel_offset.x() + voxel_offset.y() * this->size_li
                + voxel_offset.z() * this->size_sq] = voxel_data;
}

template<typename T>
inline void VoxelBlockFull<T>::setData(const Eigen::Vector3i& voxel_coord,
                                       const int scale,
                                       const VoxelData& voxel_data)
{
    Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
    int scale_offset = 0;
    int scale_tmp = 0;
    int num_voxels = this->size_cu;
    while (scale_tmp < scale) {
        scale_offset += num_voxels;
        num_voxels /= 8;
        ++scale_tmp;
    }

    const int size_at_scale = this->size_li / (1 << scale);
    voxel_offset = voxel_offset / (1 << scale);
    block_data_[scale_offset + voxel_offset.x() + voxel_offset.y() * size_at_scale
                + voxel_offset.z() * se::math::sq(size_at_scale)] = voxel_data;
}

template<typename T>
inline typename VoxelBlock<T>::VoxelData VoxelBlockFull<T>::data(const int voxel_idx) const
{
    return block_data_[voxel_idx];
}

template<typename T>
inline void VoxelBlockFull<T>::setData(const int voxel_idx, const VoxelData& voxel_data)
{
    block_data_[voxel_idx] = voxel_data;
}

template<typename T>
inline typename VoxelBlock<T>::VoxelData VoxelBlockFull<T>::data(const int voxel_idx_at_scale,
                                                                 const int scale) const
{
    return block_data_[this->scaleOffset(scale) + voxel_idx_at_scale];
}

template<typename T>
inline void VoxelBlockFull<T>::setData(const int voxel_idx_at_scale,
                                       const int scale,
                                       const VoxelData& voxel_data)
{
    block_data_[this->scaleOffset(scale) + voxel_idx_at_scale] = voxel_data;
}

template<typename T>
void VoxelBlockFull<T>::initFromBlock(const VoxelBlockFull<T>& block)
{
    this->code_ = block.code();
    this->size_ = block.size_;
    this->children_mask_ = block.children_mask();
    this->timestamp_ = block.timestamp();
    this->active_ = block.active();
    this->coordinates_ = block.coordinates();
    this->min_scale_ = block.min_scale();
    this->min_scale_reached_ = block.minScaleReached();
    this->current_scale_ = block.current_scale();
    std::copy(block.childrenData(), block.childrenData() + 8, this->children_data_);
    std::copy(block.blockData(), block.blockData() + num_voxels_in_block, blockData());
}



// Voxel block single scale allocation implementation

template<typename T>
VoxelBlockSingle<T>::VoxelBlockSingle(const typename T::VoxelData init_data) :
        VoxelBlock<T>(0, -1), init_data_(init_data)
{
}

template<typename T>
VoxelBlockSingle<T>::VoxelBlockSingle(const VoxelBlockSingle<T>& block)
{
    initFromBlock(block);
}

template<typename T>
void VoxelBlockSingle<T>::operator=(const VoxelBlockSingle<T>& block)
{
    initFromBlock(block);
}

template<typename T>
VoxelBlockSingle<T>::~VoxelBlockSingle()
{
    for (auto data_at_scale : block_data_) {
        delete[] data_at_scale;
    }
}

template<typename T>
inline typename VoxelBlock<T>::VoxelData VoxelBlockSingle<T>::initData() const
{
    return init_data_;
}

template<typename T>
inline void VoxelBlockSingle<T>::setInitData(const VoxelData& init_data)
{
    init_data_ = init_data;
}

template<typename T>
inline typename VoxelBlock<T>::VoxelData
VoxelBlockSingle<T>::data(const Eigen::Vector3i& voxel_coord) const
{
    if (VoxelBlock<T>::max_scale - (block_data_.size() - 1) != 0) {
        return init_data_;
    }
    else {
        Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
        return block_data_[VoxelBlock<T>::max_scale][voxel_offset.x()
                                                     + voxel_offset.y() * this->size_li
                                                     + voxel_offset.z() * this->size_sq];
    }
}

template<typename T>
inline void VoxelBlockSingle<T>::setData(const Eigen::Vector3i& voxel_coord,
                                         const VoxelData& voxel_data)
{
    Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
    block_data_[VoxelBlock<T>::max_scale][voxel_offset.x() + voxel_offset.y() * this->size_li
                                          + voxel_offset.z() * this->size_sq] = voxel_data;
}

template<typename T>
inline void VoxelBlockSingle<T>::setDataSafe(const Eigen::Vector3i& voxel_coord,
                                             const VoxelData& voxel_data)
{
    allocateDownTo(0);
    Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
    block_data_[VoxelBlock<T>::max_scale][voxel_offset.x() + voxel_offset.y() * this->size_li
                                          + voxel_offset.z() * this->size_sq] = voxel_data;
}

template<typename T>
inline typename VoxelBlock<T>::VoxelData
VoxelBlockSingle<T>::data(const Eigen::Vector3i& voxel_coord, const int scale) const
{
    if (VoxelBlock<T>::max_scale - (block_data_.size() - 1) > static_cast<size_t>(scale)) {
        return init_data_;
    }
    else {
        Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
        voxel_offset = voxel_offset / (1 << scale);
        const int size_at_scale = this->size_li >> scale;
        return block_data_[VoxelBlock<T>::max_scale - scale]
                          [voxel_offset.x() + voxel_offset.y() * size_at_scale
                           + voxel_offset.z() * se::math::sq(size_at_scale)];
    }
}

template<typename T>
inline void VoxelBlockSingle<T>::setData(const Eigen::Vector3i& voxel_coord,
                                         const int scale,
                                         const VoxelData& voxel_data)
{
    int size_at_scale = this->size_li >> scale;
    Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
    voxel_offset = voxel_offset / (1 << scale);
    block_data_[VoxelBlock<T>::max_scale - scale]
               [voxel_offset.x() + voxel_offset.y() * size_at_scale
                + voxel_offset.z() * se::math::sq(size_at_scale)] = voxel_data;
}

template<typename T>
inline void VoxelBlockSingle<T>::setDataSafe(const Eigen::Vector3i& voxel_coord,
                                             const int scale,
                                             const VoxelData& voxel_data)
{
    allocateDownTo(scale);
    int size_at_scale = this->size_li >> scale;
    Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
    voxel_offset = voxel_offset / (1 << scale);
    block_data_[VoxelBlock<T>::max_scale - scale]
               [voxel_offset.x() + voxel_offset.y() * size_at_scale
                + voxel_offset.z() * se::math::sq(size_at_scale)] = voxel_data;
}

template<typename T>
inline typename VoxelBlock<T>::VoxelData VoxelBlockSingle<T>::data(const int voxel_idx) const
{
    int remaining_voxel_idx = voxel_idx;
    int scale = 0;
    int size_at_scale_cu = this->size_cu;
    while (remaining_voxel_idx / size_at_scale_cu >= 1) {
        scale += 1;
        remaining_voxel_idx -= size_at_scale_cu;
        size_at_scale_cu = se::math::cu(this->size_li >> scale);
    }
    if (VoxelBlock<T>::max_scale - (block_data_.size() - 1) > static_cast<size_t>(scale)) {
        return init_data_;
    }
    else {
        return block_data_[VoxelBlock<T>::max_scale - scale][remaining_voxel_idx];
    }
}

template<typename T>
inline void VoxelBlockSingle<T>::setData(const int voxel_idx, const VoxelData& voxel_data)
{
    int remaining_voxel_idx = voxel_idx;
    int scale = 0;
    int size_at_scale_cu = this->size_cu;
    while (remaining_voxel_idx / size_at_scale_cu >= 1) {
        scale += 1;
        remaining_voxel_idx -= size_at_scale_cu;
        size_at_scale_cu = se::math::cu(this->size_li >> scale);
    }
    block_data_[VoxelBlock<T>::max_scale - scale][remaining_voxel_idx] = voxel_data;
}

template<typename T>
inline void VoxelBlockSingle<T>::setDataSafe(const int voxel_idx, const VoxelData& voxel_data)
{
    int remaining_voxel_idx = voxel_idx;
    int scale = 0;
    int size_at_scale_cu = this->size_cu;
    while (remaining_voxel_idx / size_at_scale_cu >= 1) {
        scale += 1;
        remaining_voxel_idx -= size_at_scale_cu;
        size_at_scale_cu = se::math::cu(this->size_li >> scale);
    }
    allocateDownTo(scale);
    block_data_[VoxelBlock<T>::max_scale - scale][remaining_voxel_idx] = voxel_data;
}

template<typename T>
inline typename VoxelBlock<T>::VoxelData VoxelBlockSingle<T>::data(const int voxel_idx_at_scale,
                                                                   const int scale) const
{
    const size_t scale_idx = VoxelBlock<T>::max_scale - scale;
    if (scale_idx < block_data_.size()) {
        return block_data_[scale_idx][voxel_idx_at_scale];
    }
    else {
        return init_data_;
    }
}

template<typename T>
inline void VoxelBlockSingle<T>::setData(const int voxel_idx_at_scale,
                                         const int scale,
                                         const VoxelData& voxel_data)
{
    const size_t scale_idx = VoxelBlock<T>::max_scale - scale;
    block_data_[scale_idx][voxel_idx_at_scale] = voxel_data;
}

template<typename T>
inline void
VoxelBlockSingle<T>::setDataSafe(const int voxel_idx, const int scale, const VoxelData& voxel_data)
{
    allocateDownTo(scale);
    setData(voxel_idx, scale, voxel_data);
}

template<typename T>
void VoxelBlockSingle<T>::allocateDownTo()
{
    if (VoxelBlock<T>::max_scale - (block_data_.size() - 1) != 0) {
        for (int scale = VoxelBlock<T>::max_scale - block_data_.size(); scale >= 0; scale--) {
            int size_at_scale = this->size_li >> scale;
            int num_voxels_at_scale = se::math::cu(size_at_scale);
            VoxelData* voxel_data = new VoxelData[num_voxels_at_scale];
            initialiseData(voxel_data, num_voxels_at_scale);
            block_data_.push_back(voxel_data);
        }
        this->min_scale_ = 0;
        this->min_scale_reached_ = 0;
    }
}

template<typename T>
void VoxelBlockSingle<T>::allocateDownTo(const int scale)
{
    if (VoxelBlock<T>::max_scale - (block_data_.size() - 1) > static_cast<size_t>(scale)) {
        for (int scale_tmp = VoxelBlock<T>::max_scale - block_data_.size(); scale_tmp >= scale;
             scale_tmp--) {
            int size_at_scale_tmp = this->size_li >> scale_tmp;
            int num_voxels_at_scale_tmp = se::math::cu(size_at_scale_tmp);
            VoxelData* voxel_data = new VoxelData[num_voxels_at_scale_tmp];
            initialiseData(voxel_data, num_voxels_at_scale_tmp);
            block_data_.push_back(voxel_data);
        }
        this->min_scale_ = scale;
        this->updateMinScaleReached();
    }
}

template<typename T>
void VoxelBlockSingle<T>::deleteUpTo(const int scale)
{
    if (this->min_scale_ == -1 || this->min_scale_ > scale)
        return;
    for (int scale_tmp = this->min_scale_; scale_tmp < scale; scale_tmp++) {
        auto data_at_scale = block_data_[this->max_scale - scale_tmp];
        delete[] data_at_scale;
        block_data_.pop_back();
    }
    this->min_scale_ = scale;
    this->updateMinScaleReached();
}

template<typename T>
void VoxelBlockSingle<T>::initFromBlock(const VoxelBlockSingle<T>& block)
{
    this->code_ = block.code();
    this->size_ = block.size_;
    this->children_mask_ = block.children_mask();
    this->timestamp_ = block.timestamp();
    this->active_ = block.active();
    this->coordinates_ = block.coordinates();
    this->min_scale_ = block.min_scale();
    this->min_scale_reached_ = block.minScaleReached();
    this->current_scale_ = block.current_scale();
    std::copy(block.childrenData(), block.childrenData() + 8, this->children_data_);
    if (block.min_scale()
        != -1) { // Verify that at least some mip-mapped level has been initialised.
        for (int scale = this->max_scale; scale >= block.min_scale(); scale--) {
            int size_at_scale = this->size_li >> scale;
            int num_voxels_at_scale = se::math::cu(size_at_scale);
            blockData().push_back(new typename T::VoxelData[num_voxels_at_scale]);
            std::copy(block.blockData()[VoxelBlock<T>::max_scale - scale],
                      block.blockData()[VoxelBlock<T>::max_scale - scale] + num_voxels_at_scale,
                      blockData()[VoxelBlock<T>::max_scale - scale]);
        }
    }
}

template<typename T>
void VoxelBlockSingle<T>::initialiseData(VoxelData* voxel_data, const int num_voxels)
{
    for (int voxel_idx = 0; voxel_idx < num_voxels; voxel_idx++) {
        voxel_data[voxel_idx] = init_data_;
    }
}



// Voxel block single scale allocation implementation

template<typename T>
VoxelBlockSingleMax<T>::VoxelBlockSingleMax(const typename T::VoxelData init_data) :
        VoxelBlock<T>(0, -1), buffer_scale_(-1), init_data_(init_data)
{
}



template<typename T>
VoxelBlockSingleMax<T>::VoxelBlockSingleMax(const VoxelBlockSingleMax<T>& block)
{
    initFromBlock(block);
}



template<typename T>
void VoxelBlockSingleMax<T>::operator=(const VoxelBlockSingleMax<T>& block)
{
    initFromBlock(block);
}



template<typename T>
VoxelBlockSingleMax<T>::~VoxelBlockSingleMax()
{
    for (auto& data_at_scale : block_data_) {
        delete[] data_at_scale;
    }

    block_max_data_
        .pop_back(); ///<< Avoid double free as the min scale data points to the same data.
    for (auto& max_data_at_scale : block_max_data_) {
        delete[] max_data_at_scale;
    }

    if (buffer_data_ && buffer_scale_ < this->min_scale_) {
        delete[] buffer_data_;
    }
}



template<typename T>
inline typename VoxelBlock<T>::VoxelData VoxelBlockSingleMax<T>::initData() const
{
    return init_data_;
}



template<typename T>
inline void VoxelBlockSingleMax<T>::setInitData(const VoxelData& init_data)
{
    init_data_ = init_data;
}



template<typename T>
inline typename VoxelBlock<T>::VoxelData
VoxelBlockSingleMax<T>::data(const Eigen::Vector3i& voxel_coord) const
{
    if (VoxelBlock<T>::max_scale - (block_data_.size() - 1) != 0) {
        return init_data_;
    }
    else {
        const Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
        assert(0 <= voxel_offset.x() && voxel_offset.x() <= VoxelBlock<T>::scaleSize(0));
        assert(0 <= voxel_offset.y() && voxel_offset.y() <= VoxelBlock<T>::scaleSize(0));
        assert(0 <= voxel_offset.z() && voxel_offset.z() <= VoxelBlock<T>::scaleSize(0));
        return block_data_[VoxelBlock<T>::max_scale][voxel_offset.x()
                                                     + voxel_offset.y() * this->size_li
                                                     + voxel_offset.z() * this->size_sq];
    }
}



template<typename T>
inline void VoxelBlockSingleMax<T>::setData(const Eigen::Vector3i& voxel_coord,
                                            const VoxelData& voxel_data)
{
    const Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
    assert(0 <= voxel_offset.x() && voxel_offset.x() <= VoxelBlock<T>::scaleSize(0));
    assert(0 <= voxel_offset.y() && voxel_offset.y() <= VoxelBlock<T>::scaleSize(0));
    assert(0 <= voxel_offset.z() && voxel_offset.z() <= VoxelBlock<T>::scaleSize(0));
    block_data_[VoxelBlock<T>::max_scale][voxel_offset.x() + voxel_offset.y() * this->size_li
                                          + voxel_offset.z() * this->size_sq] = voxel_data;
}



template<typename T>
inline void VoxelBlockSingleMax<T>::setDataSafe(const Eigen::Vector3i& voxel_coord,
                                                const VoxelData& voxel_data)
{
    const Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
    assert(0 <= voxel_offset.x() && voxel_offset.x() <= VoxelBlock<T>::scaleSize(0));
    assert(0 <= voxel_offset.y() && voxel_offset.y() <= VoxelBlock<T>::scaleSize(0));
    assert(0 <= voxel_offset.z() && voxel_offset.z() <= VoxelBlock<T>::scaleSize(0));
    allocateDownTo(0);
    block_data_[VoxelBlock<T>::max_scale][voxel_offset.x() + voxel_offset.y() * this->size_li
                                          + voxel_offset.z() * this->size_sq] = voxel_data;
}



template<typename T>
inline typename VoxelBlock<T>::VoxelData
VoxelBlockSingleMax<T>::data(const Eigen::Vector3i& voxel_coord, const int scale) const
{
    assert(0 <= scale && scale <= VoxelBlock<T>::max_scale);

    if (VoxelBlock<T>::max_scale - (block_data_.size() - 1) > static_cast<size_t>(scale)) {
        return init_data_;
    }
    else {
        Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
        voxel_offset = voxel_offset / (1 << scale);
        assert(0 <= voxel_offset.x() && voxel_offset.x() <= VoxelBlock<T>::scaleSize(scale));
        assert(0 <= voxel_offset.y() && voxel_offset.y() <= VoxelBlock<T>::scaleSize(scale));
        assert(0 <= voxel_offset.z() && voxel_offset.z() <= VoxelBlock<T>::scaleSize(scale));
        const int size_at_scale = this->size_li >> scale;
        return block_data_[VoxelBlock<T>::max_scale - scale]
                          [voxel_offset.x() + voxel_offset.y() * size_at_scale
                           + voxel_offset.z() * se::math::sq(size_at_scale)];
    }
}



template<typename T>
inline void VoxelBlockSingleMax<T>::setData(const Eigen::Vector3i& voxel_coord,
                                            const int scale,
                                            const VoxelData& voxel_data)
{
    assert(0 <= scale && scale <= VoxelBlock<T>::max_scale);

    int size_at_scale = this->size_li >> scale;
    Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
    voxel_offset = voxel_offset / (1 << scale);
    assert(0 <= voxel_offset.x() && voxel_offset.x() <= VoxelBlock<T>::scaleSize(scale));
    assert(0 <= voxel_offset.y() && voxel_offset.y() <= VoxelBlock<T>::scaleSize(scale));
    assert(0 <= voxel_offset.z() && voxel_offset.z() <= VoxelBlock<T>::scaleSize(scale));
    block_data_[VoxelBlock<T>::max_scale - scale]
               [voxel_offset.x() + voxel_offset.y() * size_at_scale
                + voxel_offset.z() * se::math::sq(size_at_scale)] = voxel_data;
}

template<typename T>
inline void VoxelBlockSingleMax<T>::setDataSafe(const Eigen::Vector3i& voxel_coord,
                                                const int scale,
                                                const VoxelData& voxel_data)
{
    assert(0 <= scale && scale <= VoxelBlock<T>::max_scale);

    allocateDownTo(scale);
    int size_at_scale = this->size_li >> scale;
    Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
    voxel_offset = voxel_offset / (1 << scale);
    assert(0 <= voxel_offset.x() && voxel_offset.x() <= VoxelBlock<T>::scaleSize(scale));
    assert(0 <= voxel_offset.y() && voxel_offset.y() <= VoxelBlock<T>::scaleSize(scale));
    assert(0 <= voxel_offset.z() && voxel_offset.z() <= VoxelBlock<T>::scaleSize(scale));
    block_data_[VoxelBlock<T>::max_scale - scale]
               [voxel_offset.x() + voxel_offset.y() * size_at_scale
                + voxel_offset.z() * se::math::sq(size_at_scale)] = voxel_data;
}



template<typename T>
inline typename VoxelBlock<T>::VoxelData VoxelBlockSingleMax<T>::data(const int voxel_idx) const
{
    assert(voxel_idx >= 0);

    int remaining_voxel_idx = voxel_idx;
    int scale = 0;
    int size_at_scale_cu = this->size_cu;
    while (voxel_idx / size_at_scale_cu >= 1) {
        scale += 1;
        remaining_voxel_idx -= size_at_scale_cu;
        size_at_scale_cu = se::math::cu(this->size_li >> scale);
    }
    if (VoxelBlock<T>::max_scale - (block_data_.size() - 1) > static_cast<size_t>(scale)) {
        return init_data_;
    }
    else {
        return block_data_[VoxelBlock<T>::max_scale - scale][remaining_voxel_idx];
    }
}



template<typename T>
inline void VoxelBlockSingleMax<T>::setData(const int voxel_idx, const VoxelData& voxel_data)
{
    assert(voxel_idx >= 0);

    int remaining_voxel_idx = voxel_idx;
    int scale = 0;
    int size_at_scale_cu = this->size_cu;
    while (remaining_voxel_idx / size_at_scale_cu >= 1) {
        scale += 1;
        remaining_voxel_idx -= size_at_scale_cu;
        size_at_scale_cu = se::math::cu(this->size_li >> scale);
    }
    block_data_[VoxelBlock<T>::max_scale - scale][remaining_voxel_idx] = voxel_data;
}



template<typename T>
inline void VoxelBlockSingleMax<T>::setDataSafe(const int voxel_idx, const VoxelData& voxel_data)
{
    assert(voxel_idx >= 0);

    int remaining_voxel_idx = voxel_idx;
    int scale = 0;
    int size_at_scale_cu = this->size_cu;
    while (remaining_voxel_idx / size_at_scale_cu >= 1) {
        scale += 1;
        remaining_voxel_idx -= size_at_scale_cu;
        size_at_scale_cu = se::math::cu(this->size_li >> scale);
    }
    allocateDownTo(scale);
    block_data_[VoxelBlock<T>::max_scale - scale][remaining_voxel_idx] = voxel_data;
}



template<typename T>
inline typename VoxelBlock<T>::VoxelData VoxelBlockSingleMax<T>::data(const int voxel_idx_at_scale,
                                                                      const int scale) const
{
    assert(0 <= scale && scale <= VoxelBlock<T>::max_scale);
    assert(0 <= voxel_idx_at_scale && VoxelBlock<T>::scaleNumVoxels(scale));

    const size_t scale_idx = VoxelBlock<T>::max_scale - scale;
    if (scale_idx < block_data_.size()) {
        return block_data_[scale_idx][voxel_idx_at_scale];
    }
    else {
        return init_data_;
    }
}



template<typename T>
inline void VoxelBlockSingleMax<T>::setData(const int voxel_idx_at_scale,
                                            const int scale,
                                            const VoxelData& voxel_data)
{
    assert(0 <= scale && scale <= VoxelBlock<T>::max_scale);
    assert(0 <= voxel_idx_at_scale && VoxelBlock<T>::scaleNumVoxels(scale));

    const size_t scale_idx = VoxelBlock<T>::max_scale - scale;
    block_data_[scale_idx][voxel_idx_at_scale] = voxel_data;
}



template<typename T>
inline void VoxelBlockSingleMax<T>::setDataSafe(const int voxel_idx_at_scale,
                                                const int scale,
                                                const VoxelData& voxel_data)
{
    allocateDownTo(scale);
    setData(voxel_idx_at_scale, scale, voxel_data);
}



template<typename T>
inline typename VoxelBlock<T>::VoxelData
VoxelBlockSingleMax<T>::maxData(const Eigen::Vector3i& voxel_coord) const
{
    if (VoxelBlock<T>::max_scale - (block_data_.size() - 1) != 0) {
        return init_data_;
    }
    else {
        const Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
        assert(0 <= voxel_offset.x() && voxel_offset.x() <= VoxelBlock<T>::scaleSize(0));
        assert(0 <= voxel_offset.y() && voxel_offset.y() <= VoxelBlock<T>::scaleSize(0));
        assert(0 <= voxel_offset.z() && voxel_offset.z() <= VoxelBlock<T>::scaleSize(0));
        return block_max_data_[VoxelBlock<T>::max_scale][voxel_offset.x()
                                                         + voxel_offset.y() * this->size_li
                                                         + voxel_offset.z() * this->size_sq];
    }
}



template<typename T>
inline void VoxelBlockSingleMax<T>::setMaxData(const Eigen::Vector3i& voxel_coord,
                                               const VoxelData& voxel_data)
{
    const Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
    assert(0 <= voxel_offset.x() && voxel_offset.x() <= VoxelBlock<T>::scaleSize(0));
    assert(0 <= voxel_offset.y() && voxel_offset.y() <= VoxelBlock<T>::scaleSize(0));
    assert(0 <= voxel_offset.z() && voxel_offset.z() <= VoxelBlock<T>::scaleSize(0));
    block_max_data_[VoxelBlock<T>::max_scale][voxel_offset.x() + voxel_offset.y() * this->size_li
                                              + voxel_offset.z() * this->size_sq] = voxel_data;
}



template<typename T>
inline void VoxelBlockSingleMax<T>::setMaxDataSafe(const Eigen::Vector3i& voxel_coord,
                                                   const VoxelData& voxel_data)
{
    allocateDownTo(0);
    setMaxData(voxel_coord, voxel_data);
}



template<typename T>
inline typename VoxelBlock<T>::VoxelData
VoxelBlockSingleMax<T>::maxData(const Eigen::Vector3i& voxel_coord, const int scale) const
{
    assert(0 <= scale && scale <= VoxelBlock<T>::max_scale);

    if (VoxelBlock<T>::max_scale - (block_data_.size() - 1) > static_cast<size_t>(scale)) {
        return init_data_;
    }
    else {
        Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
        voxel_offset = voxel_offset / (1 << scale);
        assert(0 <= voxel_offset.x() && voxel_offset.x() <= VoxelBlock<T>::scaleSize(scale));
        assert(0 <= voxel_offset.y() && voxel_offset.y() <= VoxelBlock<T>::scaleSize(scale));
        assert(0 <= voxel_offset.z() && voxel_offset.z() <= VoxelBlock<T>::scaleSize(scale));
        const int size_at_scale = this->size_li >> scale;
        return block_max_data_[VoxelBlock<T>::max_scale - scale]
                              [voxel_offset.x() + voxel_offset.y() * size_at_scale
                               + voxel_offset.z() * se::math::sq(size_at_scale)];
    }
}



template<typename T>
inline void VoxelBlockSingleMax<T>::setMaxData(const Eigen::Vector3i& voxel_coord,
                                               const int scale,
                                               const VoxelData& voxel_data)
{
    assert(0 <= scale && scale <= VoxelBlock<T>::max_scale);

    int size_at_scale = this->size_li >> scale;
    Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
    voxel_offset = voxel_offset / (1 << scale);
    assert(0 <= voxel_offset.x() && voxel_offset.x() <= VoxelBlock<T>::scaleSize(scale));
    assert(0 <= voxel_offset.y() && voxel_offset.y() <= VoxelBlock<T>::scaleSize(scale));
    assert(0 <= voxel_offset.z() && voxel_offset.z() <= VoxelBlock<T>::scaleSize(scale));
    block_max_data_[VoxelBlock<T>::max_scale - scale]
                   [voxel_offset.x() + voxel_offset.y() * size_at_scale
                    + voxel_offset.z() * se::math::sq(size_at_scale)] = voxel_data;
}

template<typename T>
inline void VoxelBlockSingleMax<T>::setMaxDataSafe(const Eigen::Vector3i& voxel_coord,
                                                   const int scale,
                                                   const VoxelData& voxel_data)
{
    allocateDownTo(scale);
    setMaxData(voxel_coord, scale, voxel_data);
}


template<typename T>
inline typename VoxelBlock<T>::VoxelData VoxelBlockSingleMax<T>::maxData(const int voxel_idx) const
{
    assert(voxel_idx >= 0);

    int remaining_voxel_idx = voxel_idx;
    int scale = 0;
    int size_at_scale_cu = this->size_cu;
    while (voxel_idx / size_at_scale_cu >= 1) {
        scale += 1;
        remaining_voxel_idx -= size_at_scale_cu;
        size_at_scale_cu = se::math::cu(this->size_li >> scale);
    }
    if (VoxelBlock<T>::max_scale - (block_data_.size() - 1) > static_cast<size_t>(scale)) {
        return init_data_;
    }
    else {
        return block_max_data_[VoxelBlock<T>::max_scale - scale][remaining_voxel_idx];
    }
}



template<typename T>
inline void VoxelBlockSingleMax<T>::setMaxData(const int voxel_idx, const VoxelData& voxel_data)
{
    assert(voxel_idx >= 0);

    int remaining_voxel_idx = voxel_idx;
    int scale = 0;
    int size_at_scale_cu = this->size_cu;
    while (remaining_voxel_idx / size_at_scale_cu >= 1) {
        scale += 1;
        remaining_voxel_idx -= size_at_scale_cu;
        size_at_scale_cu = se::math::cu(this->size_li >> scale);
    }
    block_max_data_[VoxelBlock<T>::max_scale - scale][remaining_voxel_idx] = voxel_data;
}



template<typename T>
inline void VoxelBlockSingleMax<T>::setMaxDataSafe(const int voxel_idx, const VoxelData& voxel_data)
{
    allocateDownTo(0);
    setMaxData(voxel_idx, voxel_data);
}



template<typename T>
inline typename VoxelBlock<T>::VoxelData
VoxelBlockSingleMax<T>::maxData(const int voxel_idx_at_scale, const int scale) const
{
    assert(0 <= scale && scale <= VoxelBlock<T>::max_scale);
    assert(0 <= voxel_idx_at_scale && voxel_idx_at_scale <= VoxelBlock<T>::scaleNumVoxels(scale));

    const size_t scale_idx = VoxelBlock<T>::max_scale - scale;
    if (scale_idx < block_max_data_.size()) {
        return block_max_data_[scale_idx][voxel_idx_at_scale];
    }
    else {
        return init_data_;
    }
}



template<typename T>
inline void VoxelBlockSingleMax<T>::setMaxData(const int voxel_idx_at_scale,
                                               const int scale,
                                               const VoxelData& voxel_data)
{
    assert(0 <= scale && scale <= VoxelBlock<T>::max_scale);
    assert(0 <= voxel_idx_at_scale && voxel_idx_at_scale <= VoxelBlock<T>::scaleNumVoxels(scale));

    const size_t scale_idx = VoxelBlock<T>::max_scale - scale;
    block_max_data_[scale_idx][voxel_idx_at_scale] = voxel_data;
}



template<typename T>
inline void VoxelBlockSingleMax<T>::setMaxDataSafe(const int voxel_idx_at_scale,
                                                   const int scale,
                                                   const VoxelData& voxel_data)
{
    allocateDownTo(scale);
    setMaxData(voxel_idx_at_scale, scale, voxel_data);
}



template<typename T>
void VoxelBlockSingleMax<T>::allocateDownTo()
{
    if (VoxelBlock<T>::max_scale - (block_data_.size() - 1) != 0) {
        for (int scale = VoxelBlock<T>::max_scale - block_data_.size(); scale >= 0; scale--) {
            int size_at_scale = this->size_li >> scale;
            int num_voxels_at_scale = se::math::cu(size_at_scale);

            if (scale == 0) {
                VoxelData* voxel_data = new VoxelData[num_voxels_at_scale];
                initialiseData(voxel_data, num_voxels_at_scale);
                block_data_.push_back(voxel_data);
                block_max_data_.push_back(
                    voxel_data); ///<< Mean and max data are the same at the min scale.
            }
            else {
                VoxelData* voxel_data = new VoxelData[num_voxels_at_scale];
                VoxelData* voxel_max_data = new VoxelData[num_voxels_at_scale];
                initialiseData(voxel_data, num_voxels_at_scale);
                block_data_.push_back(voxel_data);
                std::copy(voxel_data,
                          voxel_data + num_voxels_at_scale,
                          voxel_max_data); ///<< Copy init content.
                block_max_data_.push_back(voxel_max_data);
            }
        }

        this->current_scale_ = 0;
        this->min_scale_ = 0;
        this->min_scale_reached_ = 0;
        curr_data_ = block_data_[this->max_scale];
    }
}



template<typename T>
void VoxelBlockSingleMax<T>::allocateDownTo(const int min_scale)
{
    assert(0 <= min_scale && min_scale <= VoxelBlock<T>::max_scale);

    if (VoxelBlock<T>::max_scale - (block_data_.size() - 1) > static_cast<size_t>(min_scale)) {
        for (int scale = VoxelBlock<T>::max_scale - block_data_.size(); scale >= min_scale;
             scale--) {
            int size_at_scale = this->size_li >> scale;
            int num_voxels_at_scale = se::math::cu(size_at_scale);

            if (scale == min_scale) {
                VoxelData* voxel_data = new VoxelData[num_voxels_at_scale];
                initialiseData(voxel_data, num_voxels_at_scale);
                block_data_.push_back(voxel_data);
                block_max_data_.push_back(
                    voxel_data); ///<< Mean and max data are the same at the min scale.
            }
            else {
                VoxelData* voxel_data = new VoxelData[num_voxels_at_scale];
                VoxelData* voxel_max_data = new VoxelData[num_voxels_at_scale];
                initialiseData(voxel_data, num_voxels_at_scale);
                block_data_.push_back(voxel_data);
                std::copy(voxel_data,
                          voxel_data + num_voxels_at_scale,
                          voxel_max_data); ///<< Copy init content.
                block_max_data_.push_back(voxel_max_data);
            }
        }


        this->current_scale_ = min_scale;
        this->min_scale_ = min_scale;
        this->updateMinScaleReached();
        curr_data_ = block_data_[this->max_scale - min_scale];
    }
}



template<typename T>
void VoxelBlockSingleMax<T>::deleteUpTo(const int min_scale)
{
    assert(0 <= min_scale && min_scale <= VoxelBlock<T>::max_scale);

    if (this->min_scale_ == -1 || this->min_scale_ >= min_scale)
        return;

    auto& data_at_scale = block_data_[this->max_scale - this->min_scale_];
    delete[] data_at_scale;
    block_data_.pop_back();
    block_max_data_
        .pop_back(); ///<< Avoid double free as the min scale data points to the same data.

    for (int scale = this->min_scale_ + 1; scale < min_scale; scale++) {
        // Delete mean data
        data_at_scale = block_data_[this->max_scale - scale];
        delete[] data_at_scale;
        block_data_.pop_back();

        // Delete max data
        auto& max_data_at_scale = block_max_data_[this->max_scale - scale];
        delete[] max_data_at_scale;
        block_max_data_.pop_back();
    }

    // Replace max data at min scale with same as mean data.
    auto& max_data_at_scale = block_max_data_[this->max_scale - min_scale];
    delete[] max_data_at_scale;
    block_max_data_.pop_back();
    block_max_data_.push_back(block_data_[this->max_scale - min_scale]);

    this->min_scale_ = min_scale;
}



template<typename T>
void VoxelBlockSingleMax<T>::incrCurrObservedCount(bool do_increment)
{
    if (do_increment) {
        curr_observed_count_++;
    }
}



template<typename T>
void VoxelBlockSingleMax<T>::resetCurrCount()
{
    curr_integr_count_ = 0;
    curr_observed_count_ = 0;
}



template<typename T>
void VoxelBlockSingleMax<T>::initCurrCout()
{
    if (init_data_.observed) {
        int size_at_scale = this->size_li >> this->current_scale_;
        int num_voxels_at_scale = se::math::cu(size_at_scale);
        curr_integr_count_ = init_data_.y;
        curr_observed_count_ = num_voxels_at_scale;
    }
    else {
        resetCurrCount();
    }
}



template<typename T>
void VoxelBlockSingleMax<T>::incrBufferIntegrCount(const bool do_increment)
{
    if (do_increment
        || buffer_observed_count_ * se::math::cu(1 << buffer_scale_)
            >= 0.90 * curr_observed_count_ * se::math::cu(1 << this->current_scale_)) {
        buffer_integr_count_++;
    }
}



template<typename T>
void VoxelBlockSingleMax<T>::incrBufferObservedCount(const bool do_increment)
{
    if (do_increment) {
        buffer_observed_count_++;
    }
}



template<typename T>
void VoxelBlockSingleMax<T>::resetBufferCount()
{
    buffer_integr_count_ = 0;
    buffer_observed_count_ = 0;
}



template<typename T>
void VoxelBlockSingleMax<T>::resetBuffer()
{
    if (buffer_scale_ < this->current_scale_) {
        delete[] buffer_data_;
    }
    buffer_data_ = nullptr;
    buffer_scale_ = -1;
    resetBufferCount();
}



template<typename T>
void VoxelBlockSingleMax<T>::initBuffer(const int buffer_scale)
{
    assert(0 <= buffer_scale && buffer_scale <= VoxelBlock<T>::max_scale);

    resetBuffer();

    buffer_scale_ = buffer_scale;

    if (buffer_scale < this->current_scale_) {
        // Initialise all data to init data.
        const int size_at_scale = this->size_li >> buffer_scale;
        const int num_voxels_at_scale = se::math::cu(size_at_scale);
        buffer_data_ = new VoxelData[num_voxels_at_scale]; ///<< Data must still be initialised.
    }
    else {
        buffer_data_ = block_data_[VoxelBlock<T>::max_scale - buffer_scale_];
    }
}



template<typename T>
bool VoxelBlockSingleMax<T>::switchData()
{
    if (buffer_integr_count_ >= 20
        && buffer_observed_count_ * se::math::cu(1 << buffer_scale_) >= 0.9 * curr_observed_count_
                * se::math::cu(1 << this->current_scale_)) { // TODO: Find threshold

        /// !!! We'll switch !!!
        if (buffer_scale_ < this->current_scale_) { ///<< Switch to finer scale.
            block_data_.push_back(buffer_data_);
            block_max_data_.push_back(buffer_data_); ///< Share data at finest scale.

            /// Add allocate data for the scale that mean and max data shared before.
            const int size_at_scale = this->size_li >> (buffer_scale_ + 1);
            const int num_voxels_at_scale = se::math::cu(size_at_scale);
            block_max_data_[this->max_scale - (buffer_scale_ + 1)] =
                new VoxelData[num_voxels_at_scale]; ///<< Data must still be initialised.
        }
        else { ///<< Switch to coarser scale.
            deleteUpTo(buffer_scale_);
        }

        /// Update observed state
        const int size_at_buffer_scale = this->size_li >> buffer_scale_;
        const int num_voxels_at_buffer_scale = se::math::cu(size_at_buffer_scale);

        int missed_observed_count = 0;
        for (int voxel_idx = 0; voxel_idx < num_voxels_at_buffer_scale; voxel_idx++) {
            VoxelData& data = buffer_data_[voxel_idx];
            if (data.y > 0 && !data.observed) {
                data.observed = true;
                buffer_observed_count_++;
                missed_observed_count++;
            }
        }

        this->current_scale_ = buffer_scale_;
        this->min_scale_ = buffer_scale_;
        this->updateMinScaleReached();

        curr_data_ = buffer_data_;
        curr_integr_count_ = buffer_integr_count_;
        curr_observed_count_ = buffer_observed_count_;
        buffer_data_ = nullptr;
        buffer_scale_ = -1;
        resetBufferCount();
        return true;
    }
    return false;
}



template<typename T>
typename VoxelBlock<T>::VoxelData&
VoxelBlockSingleMax<T>::bufferData(const Eigen::Vector3i& voxel_coord) const
{
    Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
    voxel_offset = voxel_offset / (1 << buffer_scale_);
    const int size_at_scale = this->size_li >> buffer_scale_;
    return buffer_data_[voxel_offset.x() + voxel_offset.y() * size_at_scale
                        + voxel_offset.z() * se::math::sq(size_at_scale)];
}



template<typename T>
typename VoxelBlock<T>::VoxelData&
VoxelBlockSingleMax<T>::bufferData(const Eigen::Vector3i& voxel_coord)
{
    Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
    voxel_offset = voxel_offset / (1 << buffer_scale_);
    const int size_at_scale = this->size_li >> buffer_scale_;
    return buffer_data_[voxel_offset.x() + voxel_offset.y() * size_at_scale
                        + voxel_offset.z() * se::math::sq(size_at_scale)];
}



template<typename T>
typename VoxelBlock<T>::VoxelData* VoxelBlockSingleMax<T>::blockDataAtScale(const int scale)
{
    assert(0 <= scale && scale <= VoxelBlock<T>::max_scale);

    if (scale < this->min_scale_) {
        return nullptr;
    }
    else {
        return block_data_[this->max_scale - scale];
    }
}



template<typename T>
typename VoxelBlock<T>::VoxelData* VoxelBlockSingleMax<T>::blockMaxDataAtScale(const int scale)
{
    assert(0 <= scale && scale <= VoxelBlock<T>::max_scale);

    if (scale < this->min_scale_) {
        return nullptr;
    }
    else {
        return block_max_data_[this->max_scale - scale];
    }
}



template<typename T>
void VoxelBlockSingleMax<T>::initFromBlock(const VoxelBlockSingleMax<T>& block)
{
    this->code_ = block.code();
    this->size_ = block.size_;
    this->children_mask_ = block.children_mask();
    this->timestamp_ = block.timestamp();
    this->active_ = block.active();
    this->coordinates_ = block.coordinates();
    this->min_scale_ = block.min_scale();
    this->min_scale_reached_ = block.minScaleReached();
    this->current_scale_ = block.current_scale();
    init_data_ = block.initData();
    std::copy(block.childrenData(), block.childrenData() + 8, this->children_data_);
    if (block.min_scale()
        != -1) { // Verify that at least some mip-mapped level has been initialised.
        for (int scale = this->max_scale; scale >= block.min_scale(); scale--) {
            int size_at_scale = this->size_li >> scale;
            int num_voxels_at_scale = se::math::cu(size_at_scale);
            blockData().push_back(new typename T::VoxelData[num_voxels_at_scale]);
            std::copy(block.blockData()[VoxelBlock<T>::max_scale - scale],
                      block.blockData()[VoxelBlock<T>::max_scale - scale] + num_voxels_at_scale,
                      blockData()[VoxelBlock<T>::max_scale - scale]);
        }
        for (int scale = this->max_scale; scale >= block.min_scale() + 1; scale--) {
            int size_at_scale = this->size_li >> scale;
            int num_voxels_at_scale = se::math::cu(size_at_scale);
            blockMaxData().push_back(new typename T::VoxelData[num_voxels_at_scale]);
            std::copy(block.blockMaxData()[VoxelBlock<T>::max_scale - scale],
                      block.blockMaxData()[VoxelBlock<T>::max_scale - scale] + num_voxels_at_scale,
                      blockMaxData()[VoxelBlock<T>::max_scale - scale]);
        }
        blockMaxData()[VoxelBlock<T>::max_scale - this->min_scale_] =
            blockData()[VoxelBlock<T>::max_scale - this->min_scale_];
    }
}

template<typename T>
void VoxelBlockSingleMax<T>::initialiseData(VoxelData* voxel_data, const int num_voxels)
{
    std::fill(voxel_data, voxel_data + num_voxels, init_data_);
}


} // namespace se

#endif // OCTREE_IMPL_HPP
