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

#ifndef NODE_H
#define NODE_H

#include <time.h>
#include <atomic>
#include "octree_defines.h"
#include "utils/math_utils.h"
#include "io/se_serialise.hpp"

namespace se {
/*! \brief A non-leaf node of the Octree. Each Node has 8 children.
 */

static inline Eigen::Vector3f getSampleCoord(const Eigen::Vector3i& octant_coord,
                                             const int              octant_size,
                                             const Eigen::Vector3f& sample_offset_frac) {
  return octant_coord.cast<float>() + sample_offset_frac * octant_size;
}

template <typename T>
class Node {

public:
  typedef typename T::VoxelData VoxelData;

  VoxelData data_[8];
  key_t code_;
  unsigned int size_;
  unsigned char children_mask_;
  unsigned int timestamp_;
  bool active_;

  Node(typename T::VoxelData init_data = T::initData()) {
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

  virtual ~Node(){};

  Node*& child(const int x, const int y, const int z) {
    return child_ptr_[x + y * 2 + z * 4];
  };

  const Node* child(const int x, const int y, const int z) const {
    return child_ptr_[x + y * 2 + z * 4];
  };

  Node*& child(const int child_idx ) {
    return child_ptr_[child_idx];
  }

  const Node* child(const int child_idx ) const {
    return child_ptr_[child_idx];
  }

  Node*& parent() {
    return parent_ptr_;
  }

  const Node* parent() const {
    return parent_ptr_;
  }

  unsigned int timestamp() { return timestamp_; }
  unsigned int timestamp(unsigned int t) { return timestamp_ = t; }

  void active(const bool a){ active_ = a; }
  bool active() const { return active_; }

  virtual bool isBlock() const { return false; }

protected:
    Node* parent_ptr_;
    Node* child_ptr_[8];
private:
    friend std::ofstream& internal::serialise <> (std::ofstream& out, Node& node);
    friend void internal::deserialise <> (Node& node, std::ifstream& in);
};

/*! \brief A leaf node of the Octree. Each VoxelBlock contains compute_num_voxels() voxels
 * voxels.
 */
template <typename T>
class VoxelBlock: public Node<T> {

public:
  using VoxelData = typename T::VoxelData;

  VoxelBlock() {
    coordinates_ = Eigen::Vector3i::Constant(0);
    current_scale_ = 0;
    min_scale_ = -1;
  };

  static constexpr unsigned int size      = BLOCK_SIZE;
  static constexpr unsigned int size_sq   = se::math::sq(size);
  static constexpr unsigned int size_cu   = se::math::cu(size);
  static constexpr unsigned int max_scale = se::math::log2_const(size);

  bool isBlock() const { return true; }

  Eigen::Vector3i coordinates() const { return coordinates_; }
  void coordinates(const Eigen::Vector3i& block_coord){ coordinates_ = block_coord; }

  int current_scale() const { return current_scale_; }
  void current_scale(const int s) { current_scale_ = s; }

  int min_scale() const { return min_scale_; }
  void min_scale(const int s) { min_scale_ = s; }

  virtual VoxelData data(const Eigen::Vector3i& voxel_coord) const {};
  virtual void setData(const Eigen::Vector3i& voxel_coord, const VoxelData& voxel_data) {};

  virtual VoxelData data(const Eigen::Vector3i& voxel_coord, const int scale) const {};
  virtual void setData(const Eigen::Vector3i& voxel_coord, const int scale, const VoxelData& voxel_data) {};

  virtual VoxelData data(const int voxel_idx) const {};
  virtual void setData(const int voxel_idx, const VoxelData& voxel_data) {};

protected:
  Eigen::Vector3i coordinates_;
  int current_scale_;
  int min_scale_;
};


/*! \brief A leaf node of the Octree. Each VoxelBlock contains compute_num_voxels() voxels
 * voxels.
 */
template <typename T>
class VoxelBlockFull: public VoxelBlock<T> {

public:
  using VoxelData = typename VoxelBlock<T>::VoxelData;

  VoxelBlockFull(typename T::VoxelData init_data = T::initData()) {
    for (unsigned int voxel_idx = 0; voxel_idx < num_voxels; voxel_idx++) {
      voxel_block_[voxel_idx] = init_data;
    }
  }

  VoxelData data(const Eigen::Vector3i& voxel_coord) const;
  void setData(const Eigen::Vector3i& voxel_coord, const VoxelData& voxel_data);

  VoxelData data(const Eigen::Vector3i& voxel_coord, const int scale) const;
  void setData(const Eigen::Vector3i& voxel_coord, const int scale, const VoxelData& voxel_data);

  VoxelData data(const int voxel_idx) const;
  void setData(const int voxel_idx, const VoxelData& voxel_data);

  VoxelData* getBlockRawPtr() { return voxel_block_; }
  static constexpr int data_size() { return sizeof(VoxelBlockFull<T>); }

  VoxelBlockFull(const VoxelBlockFull&) = delete;

private:
  static constexpr size_t compute_num_voxels() {
    size_t voxel_count = 0;
    unsigned int size_at_scale = VoxelBlock<T>::size;
    while(size_at_scale >= 1) {
      voxel_count += size_at_scale * size_at_scale * size_at_scale;
      size_at_scale = size_at_scale >> 1;
    }
    return voxel_count;
  }
  static constexpr size_t num_voxels = compute_num_voxels();
  VoxelData voxel_block_[num_voxels]; // Brick of data.

  friend std::ofstream& internal::serialise <> (std::ofstream& out,
                                                VoxelBlockFull& node);
  friend void internal::deserialise <> (VoxelBlockFull& node, std::ifstream& in);
};

} // namespace se

#include "node_impl.hpp"

#endif
