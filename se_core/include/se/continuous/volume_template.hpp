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
#ifndef VOLUME_TEMPLATE_H
#define VOLUME_TEMPLATE_H

#include <iostream>
#include <memory>
#include "se/utils/memory_pool.hpp"
#include "se/octree.hpp"
#include <type_traits>
#include <cstring>
#include <Eigen/Dense>



/**
 * Continuous volume abstraction
 * Sparse, dynamically allocated storage accessed through the
 * appropriate indexer (octree/hash table).
 * */
template <typename VoxelImpl, template<typename> class DiscreteOctreeT>
class VolumeTemplate {

  public:
    typedef typename VoxelImpl::VoxelType VoxelType;
    typedef typename VoxelImpl::VoxelType::VoxelData VoxelData;

    VolumeTemplate(){};
    VolumeTemplate(unsigned int size, float dim, DiscreteOctreeT<VoxelType>* octree) :
      octree_(octree) {
        size_ = size;
        dim_ = dim;
      };

    VoxelData get(const Eigen::Vector3f& point_M, const int scale = 0) const {
      const float inverse_voxel_dim = size_ / dim_;
      const Eigen::Vector4i voxel_coord = (inverse_voxel_dim * point_M.homogeneous()).cast<int>();
        return octree_->get_fine(voxel_coord.x(),
                                 voxel_coord.y(),
                                 voxel_coord.z(),
                                 scale);
    }

    /*! \brief Interp voxel value at metric position  (x,y,z)
     * \param point_M three-dimensional coordinates in which each component belongs
     * to the interval [0, _extent]
     * \param stride distance between neighbouring sampling point, in voxels.
     * Must be >= 1
     * \return signed distance function value at voxel position (x,y,z)
     */
    template <typename NodeValueSelector,
              typename VoxelValueSelector>
    std::pair<float, int> interp(const Eigen::Vector3f& point_M,
                                 NodeValueSelector      select_node_value,
                                 VoxelValueSelector     select_voxel_value) const {
      const float inverse_voxel_dim = size_ / dim_;
      Eigen::Vector3f voxel_coord_f = (inverse_voxel_dim * point_M);
      return octree_->interp(voxel_coord_f, 0, select_node_value, select_voxel_value);
    }

    template <typename NodeValueSelector,
              typename VoxelValueSelector>
    std::pair<float, int> interp(const Eigen::Vector3f& point_M,
                                 const int              min_scale,
                                 NodeValueSelector      select_node_value,
                                 VoxelValueSelector     select_voxel_value) const {
      const float inverse_voxel_dim = size_ / dim_;
      Eigen::Vector3f voxel_coord_f = (inverse_voxel_dim * point_M);
      return octree_->interp(voxel_coord_f, min_scale, select_node_value, select_voxel_value);
    }

    /*! \brief Compute gradient at metric position  (x,y,z)
     * \param point_M three-dimensional coordinates in which each component belongs
     * to the interval [0, dim_]
     * \param stride distance between neighbouring sampling point, in voxels.
     * Must be >= 1
     * \return signed distance function value at voxel position (x,y,z)
     */
    template <typename FieldSelector>
    Eigen::Vector3f grad(const Eigen::Vector3f& point_M,
        const int h,
        FieldSelector select) const {
      const float inverse_voxel_dim = size_ / dim_;
      Eigen::Vector3f voxel_coord_f = inverse_voxel_dim * point_M;
      return octree_->grad(voxel_coord_f, select, h);
    }

    unsigned int size() const { return size_; }
    float dim() const { return dim_; }

    DiscreteOctreeT<VoxelType> * octree_;

  private:
    unsigned int size_;
    float dim_;
};
#endif
