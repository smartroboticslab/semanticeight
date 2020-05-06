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

#ifndef AA_FUNCTOR_HPP
#define AA_FUNCTOR_HPP
#include <functional>
#include <vector>

#include "../utils/math_utils.h"
#include "../node.hpp"
#include "../functors/data_handler.hpp"
#include "../geometry/aabb_collision.hpp"

namespace se {
  namespace functor {

    template <typename FieldType, template <typename FieldT> class OctreeT,
              typename UpdateF>

      class axis_aligned {
        public:
        axis_aligned(OctreeT<FieldType>& octree, UpdateF f) : octree_(octree), function_(f),
        min_(Eigen::Vector3i::Constant(0)),
        max_(Eigen::Vector3i::Constant(octree.size())){ }

        axis_aligned(OctreeT<FieldType>& octree, UpdateF f, const Eigen::Vector3i min,
            const Eigen::Vector3i max) : octree_(octree), function_(f),
        min_(min), max_(max){ }

        void update_block(se::VoxelBlock<FieldType> * block) {
          Eigen::Vector3i block_coord = block->coordinates();
          unsigned int y, z, x;
          Eigen::Vector3i block_size = Eigen::Vector3i::Constant(se::VoxelBlock<FieldType>::size);
          Eigen::Vector3i start = block_coord.cwiseMax(min_);
          Eigen::Vector3i last = (block_coord + block_size).cwiseMin(max_);

          for(z = start(2); z < last(2); ++z) {
            for (y = start(1); y < last(1); ++y) {
              for (x = start(0); x < last(0); ++x) {
                Eigen::Vector3i voxel_coord = Eigen::Vector3i(x, y, z);
                VoxelBlockHandler<FieldType> handler = {block, voxel_coord};
                function_(handler, voxel_coord);
              }
            }
          }
        }

        void update_node(se::Node<FieldType> * node) {
          Eigen::Vector3i node_coord = Eigen::Vector3i(unpack_morton(node->code_));
#pragma omp simd
          for(int child_idx = 0; child_idx < 8; ++child_idx) {
            const Eigen::Vector3i dir =  Eigen::Vector3i((child_idx & 1) > 0, (child_idx & 2) > 0, (child_idx & 4) > 0);
            const Eigen::Vector3i child_coord = node_coord + (dir * (node->size_ / 2));
            if(!(se::math::in(child_coord.x(), min_.x(), max_.x()) &&
                 se::math::in(child_coord.y(), min_.y(), max_.y()) &&
                 se::math::in(child_coord.z(), min_.z(), max_.z()))) continue;
            NodeHandler<FieldType> handler = {node, child_idx};
            function_(handler, child_coord);
          }
        }

        void apply() {

          auto& block_buffer = octree_.pool().blockBuffer();
#pragma omp parallel for
          for (unsigned int i = 0; i < block_buffer.size(); ++i) {
            update_block(block_buffer[i]);
          }

          auto& node_buffer = octree_.pool().nodeBuffer();
#pragma omp parallel for
          for (unsigned int i = 0; i < node_buffer.size(); ++i) {
            update_node(node_buffer[i]);
          }
        }

      private:
        OctreeT<FieldType>& octree_;
        UpdateF function_;
        Eigen::Vector3i min_;
        Eigen::Vector3i max_;
      };

    /*!
     * \brief Applies a function object to each voxel/octant in the octree.
     * \param octree Octree on which the function is going to be applied.
     * \param funct Update function to be applied.
     */
    template <typename FieldType, template <typename FieldT> class OctreeT,
              typename UpdateF>
    void axis_aligned_map(OctreeT<FieldType>& octree, UpdateF funct) {
    axis_aligned<FieldType, OctreeT, UpdateF> aa_functor(octree, funct);
    aa_functor.apply();
    }

    template <typename FieldType, template <typename FieldT> class OctreeT,
              typename UpdateF>
    void axis_aligned_map(OctreeT<FieldType>& octree, UpdateF funct,
        const Eigen::Vector3i& min, const Eigen::Vector3i& max) {
    axis_aligned<FieldType, OctreeT, UpdateF> aa_functor(octree, funct, min,  max);
    aa_functor.apply();
    }
  }
}
#endif
