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
#include <vector>
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
  unsigned int size_;
  unsigned char children_mask_;
  unsigned int timestamp_;
  bool active_;

  Node(typename T::VoxelData init_data = T::initData());

  Node(Node<T>& node);

  void operator=(Node<T>& node);

  virtual ~Node(){};

  void code(key_t code) {
    code_ = code;
  }

  key_t code() {
    return code_;
  }

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
  key_t code_;
  Node* parent_ptr_;
  Node* child_ptr_[8];

private:
  // Internal copy helper function
  void initFromNode(Node<T>& node);
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

  static constexpr unsigned int size      = BLOCK_SIZE;
  static constexpr unsigned int size_sq   = se::math::sq(size);
  static constexpr unsigned int size_cu   = se::math::cu(size);
  static constexpr unsigned int max_scale = se::math::log2_const(size);

  VoxelBlock();

  VoxelBlock(VoxelBlock<T>& block);

  void operator=(VoxelBlock<T>& block);

  virtual ~VoxelBlock() {};

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

private:
  // Internal copy helper function
  void initFromBlock(VoxelBlock<T>& block);
};


/*! \brief A leaf node of the Octree. Each VoxelBlock contains compute_num_voxels() voxels
 * voxels.
 */
template <typename T>
class VoxelBlockFull: public VoxelBlock<T> {

public:
  using VoxelData = typename VoxelBlock<T>::VoxelData;

  VoxelBlockFull(typename T::VoxelData init_data = T::initData());

  VoxelBlockFull(VoxelBlockFull<T>& block);

  void operator=(VoxelBlockFull<T>& block);

  virtual ~VoxelBlockFull() {};

  VoxelData data(const Eigen::Vector3i& voxel_coord) const;
  void setData(const Eigen::Vector3i& voxel_coord, const VoxelData& voxel_data);

  VoxelData data(const Eigen::Vector3i& voxel_coord, const int scale) const;
  void setData(const Eigen::Vector3i& voxel_coord, const int scale, const VoxelData& voxel_data);

  VoxelData data(const int voxel_idx) const;
  void setData(const int voxel_idx, const VoxelData& voxel_data);

  VoxelData* blockData() { return block_data_; }
  static constexpr int data_size() { return sizeof(VoxelBlockFull<T>); }

private:
  // Internal copy helper function
  void initFromBlock(VoxelBlockFull<T>& block);

  static constexpr size_t compute_num_voxels() {
    size_t voxel_count = 0;
    unsigned int size_at_scale = VoxelBlock<T>::size;
    while(size_at_scale >= 1) {
      voxel_count += size_at_scale * size_at_scale * size_at_scale;
      size_at_scale = size_at_scale >> 1;
    }
    return voxel_count;
  }

  static constexpr size_t num_voxels_in_block = compute_num_voxels();
  VoxelData block_data_[num_voxels_in_block]; // Brick of data.

  friend std::ofstream& internal::serialise <> (std::ofstream& out,
                                                VoxelBlockFull& node);
  friend void internal::deserialise <> (VoxelBlockFull& node, std::ifstream& in);
};

/*! \brief A leaf node of the Octree. Each VoxelBlock contains compute_num_voxels_in_block() voxels
 * voxels.
 */
template <typename T>
class VoxelBlockSingle: public VoxelBlock<T> {

public:
  using VoxelData = typename VoxelBlock<T>::VoxelData;

  VoxelBlockSingle(typename T::VoxelData init_data = T::initData()) : init_data_(init_data) { };

  VoxelBlockSingle(VoxelBlockSingle<T>& block);

  void operator=(VoxelBlockSingle<T>& block);

  ~VoxelBlockSingle();

  VoxelData initData() const;
  void setInitData(const VoxelData& init_data);

  VoxelData data(const Eigen::Vector3i& voxel_coord) const;
  void setData(const Eigen::Vector3i& voxel_coord, const VoxelData& voxel_data);
  void setDataSafe(const Eigen::Vector3i& voxel_coord, const VoxelData& voxel_data);

  VoxelData data(const Eigen::Vector3i& voxel_coord, const int scale) const;
  void setData(const Eigen::Vector3i& voxel_coord, const int scale, const VoxelData& voxel_data);
  void setDataSafe(const Eigen::Vector3i& voxel_coord, const int scale, const VoxelData& voxel_data);

  VoxelData data(const int voxel_idx) const;
  void setData(const int voxel_idx, const VoxelData& voxel_data);
  void setDataSafe(const int voxel_idx, const VoxelData& voxel_data);

  void allocateDownTo();
  void allocateDownTo(const int scale);

  void deleteUpTo(const int scale);

  std::vector<VoxelData*>& blockData() { return block_data_; }
  static constexpr int data_size() { return sizeof(VoxelBlock<T>); }

private:
  // Internal copy helper function
  void initFromBlock(VoxelBlockSingle<T>& block);
  void initaliseData(VoxelData* voxel_data, int num_voxels);
  std::vector<VoxelData*> block_data_; // block_data_[0] returns the data at scale = max_scale and not scale = 0
  VoxelData init_data_;

  friend std::ofstream& internal::serialise <> (std::ofstream& out, VoxelBlockSingle& node);
  friend void internal::deserialise <> (VoxelBlockSingle& node, std::ifstream& in);
};

} // namespace se

#include "node_impl.hpp"

#endif
