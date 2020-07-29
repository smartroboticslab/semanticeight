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

namespace se {
// Node implementation

template <typename T>
Node<T>::Node(typename T::VoxelData init_data) {
  code_ = 0;
  size_ = 0;
  children_mask_ = 0;
  timestamp_ = 0;
  for (unsigned int child_idx = 0; child_idx < 8; child_idx++) {
    data_[child_idx]      = init_data;
    parent_ptr_           = nullptr;
    child_ptr_[child_idx] = nullptr;
  }
}

template <typename T>
Node<T>::Node(Node<T>& node) {
  initFromNode(node);
}

template <typename T>
void Node<T>::operator=(Node<T>& node) {
  initFromNode(node);
}

template <typename T>
void Node<T>::initFromNode(se::Node<T>& node) {
  code_           = node.code_;
  size_           = node.size_;
  children_mask_  = node.children_mask_;
  timestamp_      = node.timestamp();
  active_         = node.active();
  std::memcpy(data_, node.data_, 8 * sizeof(VoxelData));
}

// Voxel block base implementation

template <typename T>
VoxelBlock<T>::VoxelBlock() :
    coordinates_(Eigen::Vector3i::Constant(0)),
    current_scale_(0),
    min_scale_(-1) { };

template <typename T>
VoxelBlock<T>::VoxelBlock(VoxelBlock<T>& block) {
  initFromBlock(block);
}

template <typename T>
void VoxelBlock<T>::operator=(VoxelBlock<T>& block) {
  initFromBlock(block);
}

template <typename T>
void VoxelBlock<T>::initFromBlock(VoxelBlock<T>& block) {
  this->code_  = block.code_;
  this->active(block.active());
  coordinates(block.coordinates());
  min_scale(block.min_scale());
  current_scale(block.current_scale());
}

// Voxel block full scale allocation implementation

template <typename T>
VoxelBlockFull<T>::VoxelBlockFull(typename T::VoxelData init_data) {
  for (unsigned int voxel_idx = 0; voxel_idx < num_voxels; voxel_idx++) {
    voxel_block_[voxel_idx] = init_data;
  }
}

template <typename T>
VoxelBlockFull<T>::VoxelBlockFull(VoxelBlockFull<T>& block) {
  initFromBlock(block);
}

template <typename T>
void VoxelBlockFull<T>::operator=(VoxelBlockFull<T>& block) {
  initFromBlock(block);
}

template <typename T>
inline typename VoxelBlock<T>::VoxelData
VoxelBlockFull<T>::data(const Eigen::Vector3i& voxel_coord) const {
  Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
  return voxel_block_[voxel_offset.x() +
                      voxel_offset.y() * this->size +
                      voxel_offset.z() * this->size_sq];
}

template <typename T>
inline typename VoxelBlock<T>::VoxelData
VoxelBlockFull<T>::data(const Eigen::Vector3i& voxel_coord, const int scale) const {
  Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
  int scale_offset = 0;
  int scale_tmp = 0;
  int num_voxels = this->size_cu;
  while(scale_tmp < scale) {
    scale_offset += num_voxels;
    num_voxels /= 8;
    ++scale_tmp;
  }
  const int local_size = this->size / (1 << scale);
  voxel_offset = voxel_offset / (1 << scale);
  return voxel_block_[scale_offset + voxel_offset.x() +
                      voxel_offset.y() * local_size +
                      voxel_offset.z() * se::math::sq(local_size)];
}

template <typename T>
inline void VoxelBlockFull<T>::setData(const Eigen::Vector3i& voxel_coord,
                                       const VoxelData& voxel_data){
  Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
  voxel_block_[voxel_offset.x() + voxel_offset.y() * this->size + voxel_offset.z() * this->size_sq] = voxel_data;
}

template <typename T>
inline void VoxelBlockFull<T>::setData(const Eigen::Vector3i& voxel_coord, const int scale,
                                   const VoxelData& voxel_data){
  Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
  int scale_offset = 0;
  int scale_tmp = 0;
  int num_voxels = this->size_cu;
  while(scale_tmp < scale) {
    scale_offset += num_voxels;
    num_voxels /= 8;
    ++scale_tmp;
  }

  const int size_at_scale = this->size / (1 << scale);
  voxel_offset = voxel_offset / (1 << scale);
  voxel_block_[scale_offset + voxel_offset.x() +
               voxel_offset.y() * size_at_scale +
               voxel_offset.z() * se::math::sq(size_at_scale)] = voxel_data;
}

template <typename T>
inline typename VoxelBlock<T>::VoxelData
VoxelBlockFull<T>::data(const int voxel_idx) const {
  return voxel_block_[voxel_idx];
}

template <typename T>
inline void VoxelBlockFull<T>::setData(const int voxel_idx, const VoxelData& voxel_data){
  voxel_block_[voxel_idx] = voxel_data;
}

template <typename T>
void VoxelBlockFull<T>::initFromBlock(VoxelBlockFull<T>& block) {
  this->coordinates(block.coordinates());
  this->code_  = block.code_;
  this->active(block.active());
  this->min_scale(block.min_scale());
  this->current_scale(block.current_scale());
  std::memcpy(getBlockRawPtr(), block.getBlockRawPtr(), (num_voxels) * sizeof(*(block.getBlockRawPtr())));
}

// Voxel block single scale allocation implementation

template <typename T>
VoxelBlockSingle<T>::VoxelBlockSingle(VoxelBlockSingle<T>& block) {
  initFromBlock(block);
};

template <typename T>
void VoxelBlockSingle<T>::operator=(VoxelBlockSingle<T>& block) {
  initFromBlock(block);
}


template <typename T>
VoxelBlockSingle<T>::~VoxelBlockSingle() {
  for (auto data_at_scale : block_data_) {
    delete[] data_at_scale;
  }
}

template <typename T>
inline typename VoxelBlock<T>::VoxelData
VoxelBlockSingle<T>::initData() const { return init_data_; }

template <typename T>
inline void VoxelBlockSingle<T>::setInitData(const VoxelData& init_data) { init_data_ = init_data;}

template <typename T>
inline typename VoxelBlock<T>::VoxelData
VoxelBlockSingle<T>::data(const Eigen::Vector3i& voxel_coord) const {
  if (VoxelBlock<T>::max_scale - (block_data_.size() - 1) != 0) return init_data_;
  else {
    Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
    return block_data_[VoxelBlock<T>::max_scale][voxel_offset.x() +
                                                 voxel_offset.y() * this->size +
                                                 voxel_offset.z() * this->size_sq];
  }
}

template <typename T>
inline void VoxelBlockSingle<T>::setData(const Eigen::Vector3i& voxel_coord,
                                         const VoxelData&       voxel_data){
  Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
  block_data_[VoxelBlock<T>::max_scale][voxel_offset.x() +
                                        voxel_offset.y() * this->size +
                                        voxel_offset.z() * this->size_sq] = voxel_data;
}

template <typename T>
inline void VoxelBlockSingle<T>::setDataSafe(const Eigen::Vector3i& voxel_coord,
                                             const VoxelData&       voxel_data){
  allocateDownTo(0);
  Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
  block_data_[VoxelBlock<T>::max_scale][voxel_offset.x() +
                                        voxel_offset.y() * this->size +
                                        voxel_offset.z() * this->size_sq] = voxel_data;
}

template <typename T>
inline typename VoxelBlock<T>::VoxelData
VoxelBlockSingle<T>::data(const Eigen::Vector3i& voxel_coord,
                          const int              scale) const {
  if (VoxelBlock<T>::max_scale - (block_data_.size() - 1) > scale) return init_data_;
  else {
    Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
    voxel_offset = voxel_offset / (1 << scale);
    const int size_at_scale = this->size >> scale;
    return block_data_[VoxelBlock<T>::max_scale - scale][voxel_offset.x() +
                                                         voxel_offset.y() * size_at_scale +
                                                         voxel_offset.z() * se::math::sq(size_at_scale)];
  }
}

template <typename T>
inline void VoxelBlockSingle<T>::setData(const Eigen::Vector3i& voxel_coord,
                                         const int              scale,
                                         const VoxelData&       voxel_data) {
  int size_at_scale = this->size >> scale;
  Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
  voxel_offset = voxel_offset / (1 << scale);
  block_data_[VoxelBlock<T>::max_scale - scale][voxel_offset.x() +
                                 voxel_offset.y() * size_at_scale +
                                 voxel_offset.z() * se::math::sq(size_at_scale)] = voxel_data;
}

template <typename T>
inline void VoxelBlockSingle<T>::setDataSafe(const Eigen::Vector3i& voxel_coord,
                                             const int              scale,
                                             const VoxelData&       voxel_data) {
  allocateDownTo(scale);
  int size_at_scale = this->size >> scale;
  Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
  voxel_offset = voxel_offset / (1 << scale);
  block_data_[VoxelBlock<T>::max_scale - scale][voxel_offset.x() +
                                                voxel_offset.y() * size_at_scale +
                                                voxel_offset.z() * se::math::sq(size_at_scale)] = voxel_data;
}

template <typename T>
inline typename VoxelBlock<T>::VoxelData
VoxelBlockSingle<T>::data(const int voxel_idx) const {
  int remaining_voxel_idx = voxel_idx;
  int scale = 0;
  int size_at_scale_cu = this->size_cu;
  while (voxel_idx / size_at_scale_cu >= 1) {
    scale += 1;
    remaining_voxel_idx -= size_at_scale_cu;
    size_at_scale_cu = se::math::cu(this->size >> scale);
  }
  if (VoxelBlock<T>::max_scale - (block_data_.size() - 1) > scale) return init_data_;
  else return block_data_[VoxelBlock<T>::max_scale - scale][remaining_voxel_idx];
}

template <typename T>
inline void VoxelBlockSingle<T>::setData(const int        voxel_idx,
                                         const VoxelData& voxel_data) {
  int remaining_voxel_idx = voxel_idx;
  int scale = 0;
  int size_at_scale_cu = this->size_cu;
  while (remaining_voxel_idx / size_at_scale_cu >= 1) {
    scale += 1;
    remaining_voxel_idx -= size_at_scale_cu;
    size_at_scale_cu = se::math::cu(this->size >> scale);
  }
  block_data_[VoxelBlock<T>::max_scale - scale][remaining_voxel_idx] = voxel_data;
}

template <typename T>
inline void VoxelBlockSingle<T>::setDataSafe(const int        voxel_idx,
                                             const VoxelData& voxel_data) {
  int remaining_voxel_idx = voxel_idx;
  int scale = 0;
  int size_at_scale_cu = this->size_cu;
  while (remaining_voxel_idx / size_at_scale_cu >= 1) {
    scale += 1;
    remaining_voxel_idx -= size_at_scale_cu;
    size_at_scale_cu = se::math::cu(this->size >> scale);
  }
  allocateDownTo(scale);
  block_data_[VoxelBlock<T>::max_scale - scale][remaining_voxel_idx] = voxel_data;
}

template <typename T>
void VoxelBlockSingle<T>::allocateDownTo() {
  if (VoxelBlock<T>::max_scale - (block_data_.size() - 1) != 0) {
    for (int scale = VoxelBlock<T>::max_scale - block_data_.size(); scale >= 0; scale --) {
      int size_at_scale = this->size >> scale;
      int num_voxels_at_scale = se::math::cu(size_at_scale);
      VoxelData* voxel_data = new VoxelData[num_voxels_at_scale];
      initaliseData(voxel_data, num_voxels_at_scale);
      block_data_.push_back(voxel_data);
    }
    this->min_scale_ = 0;
  }
}

template <typename T>
void VoxelBlockSingle<T>::allocateDownTo(const int scale) {
  if (VoxelBlock<T>::max_scale - (block_data_.size() - 1) > scale) {
    for (int scale_tmp = VoxelBlock<T>::max_scale - block_data_.size(); scale_tmp >= scale; scale_tmp --) {
      int size_at_scale_tmp = this->size >> scale_tmp;
      int num_voxels_at_scale_tmp = se::math::cu(size_at_scale_tmp);
      VoxelData* voxel_data = new VoxelData[num_voxels_at_scale_tmp];
      initaliseData(voxel_data, num_voxels_at_scale_tmp);
      block_data_.push_back(voxel_data);
    }
    this->min_scale_ = scale;
  }
}

template <typename T>
void VoxelBlockSingle<T>::deleteUpTo(const int scale) {
  if (this->min_scale_ == -1 || this->min_scale_ > scale) return;
  for (int scale_tmp = this->min_scale_; scale_tmp < scale; scale_tmp++) {
    auto data_at_scale = block_data_[this->max_scale - scale_tmp];
    delete[] data_at_scale;
    block_data_.pop_back();
  }
  this->min_scale_ = scale;
}

template <typename T>
void VoxelBlockSingle<T>::initFromBlock(VoxelBlockSingle<T>& block) {
  this->coordinates(block.coordinates());
  this->code_ = block.code_;
  this->active(block.active());
  this->min_scale(block.min_scale());
  this->current_scale(block.current_scale());
  if (block.min_scale() != -1) { // Verify that at least some mip-mapped level has been initalised.
    for (int scale = VoxelBlock<T>::max_scale; scale >= block.min_scale(); scale--) {
      int size_at_scale = VoxelBlock<T>::size >> scale;
      int num_voxels_at_scale = se::math::cu(size_at_scale);
      blockData().push_back(new typename T::VoxelData[num_voxels_at_scale]);
      std::memcpy(blockData()[VoxelBlock<T>::max_scale - scale],
          block.blockData()[VoxelBlock<T>::max_scale - scale],
          (num_voxels_at_scale) * sizeof(*(block.blockData()[VoxelBlock<T>::max_scale - scale])));
    }
  }
}

template <typename T>
void VoxelBlockSingle<T>::initaliseData(VoxelData* voxel_data, int num_voxels) {
  for (int voxel_idx = 0; voxel_idx < num_voxels; voxel_idx++) {
    voxel_data[voxel_idx] = init_data_;
  }
}

} // namespace se

#endif // OCTREE_IMPL_HPP

