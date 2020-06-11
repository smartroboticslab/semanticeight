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

template <typename T>
inline typename VoxelBlock<T>::VoxelData
VoxelBlockFull<T>::data(const Eigen::Vector3i& voxel_coord) const {
  Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
  return voxel_block_[voxel_offset.x() +
                      voxel_offset.y() * VoxelBlock<T>::size +
                      voxel_offset.z() * VoxelBlock<T>::size_sq];
}

template <typename T>
inline typename VoxelBlock<T>::VoxelData
VoxelBlockFull<T>::data(const Eigen::Vector3i& voxel_coord, const int scale) const {
  Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
  int scale_offset = 0;
  int scale_tmp = 0;
  int num_voxels = VoxelBlock<T>::size_cu;
  while(scale_tmp < scale) {
    scale_offset += num_voxels;
    num_voxels /= 8;
    ++scale_tmp;
  }
  const int local_size = VoxelBlock<T>::size / (1 << scale);
  voxel_offset = voxel_offset / (1 << scale);
  return voxel_block_[scale_offset + voxel_offset.x() +
                      voxel_offset.y() * local_size +
                      voxel_offset.z() * se::math::sq(local_size)];
}

template <typename T>
inline void VoxelBlockFull<T>::setData(const Eigen::Vector3i& voxel_coord,
                                       const VoxelData& voxel_data){
  Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
  voxel_block_[voxel_offset.x() + voxel_offset.y() * VoxelBlock<T>::size + voxel_offset.z() * VoxelBlock<T>::size_sq] = voxel_data;
}

template <typename T>
inline void VoxelBlockFull<T>::setData(const Eigen::Vector3i& voxel_coord, const int scale,
                                   const VoxelData& voxel_data){
  Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
  int scale_offset = 0;
  int scale_tmp = 0;
  int num_voxels = VoxelBlock<T>::size_cu;
  while(scale_tmp < scale) {
    scale_offset += num_voxels;
    num_voxels /= 8;
    ++scale_tmp;
  }

  const int size_at_scale = VoxelBlock<T>::size / (1 << scale);
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

} // namespace se

#endif // OCTREE_IMPL_HPP

