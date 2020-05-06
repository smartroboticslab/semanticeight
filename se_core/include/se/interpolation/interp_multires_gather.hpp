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

#ifndef INTERP_MULTIRES_GATHER_H
#define INTERP_MULTIRES_GATHER_H

#include "../octree_defines.h"
#include "../node.hpp"
#include "../octant_ops.hpp"

namespace se {
  namespace internal_multires {
    constexpr int INVALID_SAMPLE = -2;

    /*
     * Interpolation's value gather offsets
     */
    static const Eigen::Vector3i interp_offsets[8] =
        {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
         {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};



    template <typename FieldType, typename FieldSelector,
              template<typename FieldT> class MapIndex,
              typename ValueT>
    inline void gather_local(
        const se::VoxelBlock<FieldType>*         block,
        const MapIndex<FieldType>&               fetcher,
        const Eigen::Vector3i&                   base,
        const int                                scale,
        const int                                stride,
        FieldSelector                            select_value,
        ValueT                                   values[8]) {

      if (!block) {
        if (std::is_same<ValueT, bool>::value) {
          Eigen::Vector3i voxel_coord = base + stride * interp_offsets[0];
          ValueT value = select_value(fetcher.get(voxel_coord.x(), voxel_coord.y(), voxel_coord.z()));
          values[0] = value;
          values[1] = value;
          values[2] = value;
          values[3] = value;
          values[4] = value;
          values[5] = value;
          values[6] = value;
          values[7] = value;
        } else {
          Eigen::Vector3i voxel_coord = base + stride * interp_offsets[0];
          const auto neighbour = fetcher.get(voxel_coord.x(), voxel_coord.y(), voxel_coord.z());
          ValueT value = neighbour.x_max / neighbour.y;
          values[0] = value;
          values[1] = value;
          values[2] = value;
          values[3] = value;
          values[4] = value;
          values[5] = value;
          values[6] = value;
          values[7] = value;
        }
        return;
      }
      values[0] = select_value(block->data(base + stride*interp_offsets[0], scale));
      values[1] = select_value(block->data(base + stride*interp_offsets[1], scale));
      values[2] = select_value(block->data(base + stride*interp_offsets[2], scale));
      values[3] = select_value(block->data(base + stride*interp_offsets[3], scale));
      values[4] = select_value(block->data(base + stride*interp_offsets[4], scale));
      values[5] = select_value(block->data(base + stride*interp_offsets[5], scale));
      values[6] = select_value(block->data(base + stride*interp_offsets[6], scale));
      values[7] = select_value(block->data(base + stride*interp_offsets[7], scale));
      return;
    }



    template <typename FieldType, typename FieldSelector,
              template<typename FieldT> class MapIndex,
              typename ValueT>
    inline void gather_4(const se::VoxelBlock<FieldType>*         block,
                         const MapIndex<FieldType>&               fetcher,
                         const Eigen::Vector3i&                   base,
                         const int                                scale,
                         const int                                stride,
                         FieldSelector                            select_value,
                         const unsigned int                       offsets[4],
                         ValueT                                   values[8]) {

      if (!block) {
        if (std::is_same<ValueT, bool>::value) {
          Eigen::Vector3i voxel_coord = base + stride*interp_offsets[offsets[0]];
          ValueT value = select_value(fetcher.get(voxel_coord.x(), voxel_coord.y(), voxel_coord.z()));
          values[offsets[0]] = value;
          values[offsets[1]] = value;
          values[offsets[2]] = value;
          values[offsets[3]] = value;
        } else {
          Eigen::Vector3i voxel_coord = base + stride*interp_offsets[offsets[0]];
          const auto neighbour = fetcher.get(voxel_coord.x(), voxel_coord.y(), voxel_coord.z());
          ValueT value = neighbour.x_max / neighbour.y;
          values[offsets[0]] = value;
          values[offsets[1]] = value;
          values[offsets[2]] = value;
          values[offsets[3]] = value;
        }
        return;
      }
      values[offsets[0]] = select_value(block->data(base + stride*interp_offsets[offsets[0]], scale));
      values[offsets[1]] = select_value(block->data(base + stride*interp_offsets[offsets[1]], scale));
      values[offsets[2]] = select_value(block->data(base + stride*interp_offsets[offsets[2]], scale));
      values[offsets[3]] = select_value(block->data(base + stride*interp_offsets[offsets[3]], scale));
      return;
    }



    template <typename FieldType, typename FieldSelector,
              template<typename FieldT> class MapIndex,
              typename ValueT>
    inline void gather_2(const se::VoxelBlock<FieldType>*         block,
                         const MapIndex<FieldType>&               fetcher,
                         const Eigen::Vector3i&                   base,
                         const int                                scale,
                         const int                                stride,
                         FieldSelector                            select_value,
                         const unsigned int                       offsets[2],
                         ValueT                                   values[8]) {

      if (!block) {
        if (std::is_same<ValueT, bool>::value) {
          Eigen::Vector3i voxel_coord = base + stride*interp_offsets[offsets[0]];
          ValueT value = select_value(fetcher.get(voxel_coord.x(), voxel_coord.y(), voxel_coord.z()));
          values[offsets[0]] = value;
          values[offsets[1]] = value;
        } else {
          Eigen::Vector3i voxel_coord = base + stride*interp_offsets[offsets[0]];
          const auto neighbour = fetcher.get(voxel_coord.x(), voxel_coord.y(), voxel_coord.z());
          ValueT value = neighbour.x_max / neighbour.y;
          values[offsets[0]] = value;
          values[offsets[1]] = value;
        }
        return;
      }

      values[offsets[0]] = select_value(block->data(base + stride*interp_offsets[offsets[0]], scale));
      values[offsets[1]] = select_value(block->data(base + stride*interp_offsets[offsets[1]], scale));
      return;
    }

    template <typename FieldType,
              template<typename FieldT> class MapIndex,
              class FieldSelector,
              typename ValueT>
    inline int gather_values(
        const MapIndex<FieldType>&               fetcher,
        const Eigen::Vector3i&                   base,
        const int                                scale,
        FieldSelector                            select_value,
        ValueT                                   values[8]) {

      const int stride = 1 << scale;
      unsigned int block_size = se::VoxelBlock<FieldType>::size;
      unsigned int crossmask
          = (((base.x() & (block_size - 1)) == block_size - stride) << 2)
          | (((base.y() & (block_size - 1)) == block_size - stride) << 1)
          |  ((base.z() & (block_size - 1)) == block_size - stride);

      switch(crossmask) {
        case 0: /* all local */
          {
            se::VoxelBlock<FieldType> * block = fetcher.fetch(base(0), base(1), base(2));
            if (block && block->current_scale() > scale)
              return block->current_scale();
            gather_local(block, fetcher, base, scale, stride, select_value, values);
          }
          break;
        case 1: /* z crosses */
          {
            const unsigned int offs1[4] = {0, 1, 2, 3};
            const unsigned int offs2[4] = {4, 5, 6, 7};
            se::VoxelBlock<FieldType>* block = fetcher.fetch(base(0), base(1), base(2));
            if (block && block->current_scale() > scale)
              return block->current_scale();
            gather_4(block, fetcher, base, scale, stride, select_value, offs1, values);
            const Eigen::Vector3i base1 = base + stride * interp_offsets[offs2[0]];
            block = fetcher.fetch(base1(0), base1(1), base1(2));
            if (block && block->current_scale() > scale)
              return block->current_scale();
            gather_4(block, fetcher, base, scale, stride, select_value, offs2, values);
          }
          break;
        case 2: /* y crosses */
          {
            const unsigned int offs1[4] = {0, 1, 4, 5};
            const unsigned int offs2[4] = {2, 3, 6, 7};
            se::VoxelBlock<FieldType>* block = fetcher.fetch(base(0), base(1), base(2));
            gather_4(block, fetcher, base, scale, stride, select_value, offs1, values);
            if (block && block->current_scale() > scale)
              return block->current_scale();
            const Eigen::Vector3i base1 = base + stride * interp_offsets[offs2[0]];
            block = fetcher.fetch(base1(0), base1(1), base1(2));
            if (block && block->current_scale() > scale)
              return block->current_scale();
            gather_4(block, fetcher, base, scale, stride, select_value, offs2, values);
          }
          break;
        case 3: /* y, z cross */
          {
            const unsigned int offs1[2] = {0, 1};
            const unsigned int offs2[2] = {2, 3};
            const unsigned int offs3[2] = {4, 5};
            const unsigned int offs4[2] = {6, 7};
            const Eigen::Vector3i base2 = base + stride * interp_offsets[offs2[0]];
            const Eigen::Vector3i base3 = base + stride * interp_offsets[offs3[0]];
            const Eigen::Vector3i base4 = base + stride * interp_offsets[offs4[0]];
            se::VoxelBlock<FieldType>* block = fetcher.fetch(base(0), base(1), base(2));
            if (block && block->current_scale() > scale)
              return block->current_scale();
            gather_2(block, fetcher, base, scale, stride, select_value, offs1, values);
            block = fetcher.fetch(base2(0), base2(1), base2(2));
            if (block && block->current_scale() > scale)
              return block->current_scale();
            gather_2(block, fetcher, base, scale, stride, select_value, offs2, values);
            block = fetcher.fetch(base3(0), base3(1), base3(2));
            if (block && block->current_scale() > scale)
              return block->current_scale();
            gather_2(block, fetcher, base, scale, stride, select_value, offs3, values);
            block = fetcher.fetch(base4(0), base4(1), base4(2));
            if (block && block->current_scale() > scale)
              return block->current_scale();
            gather_2(block, fetcher, base, scale, stride, select_value, offs4, values);
          }
          break;
        case 4: /* x crosses */
          {
            const unsigned int offs1[4] = {0, 2, 4, 6};
            const unsigned int offs2[4] = {1, 3, 5, 7};
            se::VoxelBlock<FieldType>* block = fetcher.fetch(base(0), base(1), base(2));
            if (block && block->current_scale() > scale)
              return block->current_scale();
            gather_4(block, fetcher, base, scale, stride, select_value, offs1, values);
            const Eigen::Vector3i base1 = base + stride * interp_offsets[offs2[0]];
            block = fetcher.fetch(base1(0), base1(1), base1(2));
            if (block && block->current_scale() > scale)
              return block->current_scale();
            gather_4(block, fetcher, base, scale, stride, select_value, offs2, values);
          }
          break;
        case 5: /* x,z cross */
          {
            const unsigned int offs1[2] = {0, 2};
            const unsigned int offs2[2] = {1, 3};
            const unsigned int offs3[2] = {4, 6};
            const unsigned int offs4[2] = {5, 7};
            const Eigen::Vector3i base2 = base + stride * interp_offsets[offs2[0]];
            const Eigen::Vector3i base3 = base + stride * interp_offsets[offs3[0]];
            const Eigen::Vector3i base4 = base + stride * interp_offsets[offs4[0]];
            se::VoxelBlock<FieldType>* block = fetcher.fetch(base(0), base(1), base(2));
            if (block && block->current_scale() > scale)
              return block->current_scale();
            gather_2(block, fetcher, base, scale, stride, select_value, offs1, values);
            block = fetcher.fetch(base2(0), base2(1), base2(2));
            if (block && block->current_scale() > scale)
              return block->current_scale();
            gather_2(block, fetcher, base, scale, stride, select_value, offs2, values);
            block = fetcher.fetch(base3(0), base3(1), base3(2));
            if (block && block->current_scale() > scale)
              return block->current_scale();
            gather_2(block, fetcher, base, scale, stride, select_value, offs3, values);
            block = fetcher.fetch(base4(0), base4(1), base4(2));
            if (block && block->current_scale() > scale)
              return block->current_scale();
            gather_2(block, fetcher, base, scale, stride, select_value, offs4, values);
          }
          break;
        case 6: /* x,y cross */
          {
            const unsigned int offs1[2] = {0, 4};
            const unsigned int offs2[2] = {1, 5};
            const unsigned int offs3[2] = {2, 6};
            const unsigned int offs4[2] = {3, 7};
            const Eigen::Vector3i base2 = base + stride * interp_offsets[offs2[0]];
            const Eigen::Vector3i base3 = base + stride * interp_offsets[offs3[0]];
            const Eigen::Vector3i base4 = base + stride * interp_offsets[offs4[0]];
            se::VoxelBlock<FieldType> * block = fetcher.fetch(base(0), base(1), base(2));
            if (block && block->current_scale() > scale)
              return block->current_scale();
            gather_2(block, fetcher, base, scale, stride, select_value, offs1, values);
            block = fetcher.fetch(base2(0), base2(1), base2(2));
            if (block && block->current_scale() > scale)
              return block->current_scale();
            gather_2(block, fetcher, base, scale, stride, select_value, offs2, values);
            block = fetcher.fetch(base3(0), base3(1), base3(2));
            if (block && block->current_scale() > scale)
              return block->current_scale();
            gather_2(block, fetcher, base, scale, stride, select_value, offs3, values);
            block = fetcher.fetch(base4(0), base4(1), base4(2));
            if (block && block->current_scale() > scale)
              return block->current_scale();
            gather_2(block, fetcher, base, scale, stride, select_value, offs4, values);
          }
          break;

        case 7:
          {
            Eigen::Vector3i voxels_coord[8];
            voxels_coord[0] = base + stride * interp_offsets[0];
            voxels_coord[1] = base + stride * interp_offsets[1];
            voxels_coord[2] = base + stride * interp_offsets[2];
            voxels_coord[3] = base + stride * interp_offsets[3];
            voxels_coord[4] = base + stride * interp_offsets[4];
            voxels_coord[5] = base + stride * interp_offsets[5];
            voxels_coord[6] = base + stride * interp_offsets[6];
            voxels_coord[7] = base + stride * interp_offsets[7];

            for (int i = 0; i < 8; ++i) {
              auto block = fetcher.fetch(voxels_coord[i].x(), voxels_coord[i].y(), voxels_coord[i].z());
              if (block && block->current_scale() > scale)
                return block->current_scale();

              if (block) {
                values[i] = select_value(block->data(voxels_coord[i], scale));
              } else {
                if (!block) {
                  if (std::is_same<ValueT, bool>::value) {
                    ValueT value = select_value(fetcher.get(voxels_coord[i].x(), voxels_coord[i].y(), voxels_coord[i].z()));
                    values[i] = value;
                  } else {
                    const auto neighbour = fetcher.get(voxels_coord[i].x(), voxels_coord[i].y(), voxels_coord[i].z());
                    ValueT value = neighbour.x_max / neighbour.y;
                    values[i] = value;
                  }
                }
              }
            }
          }
          break;
      }
      return scale;
    }



    /*! \brief Fetch the field sample corresponding to the octant neighbour along the
     * specified direction. If the search fails the second element of the returned
     * is set to false.
     * \param stack stack of ancestor nodes of octant
     * \param octant base octant.
     * \param max_depth maximum depth of the tree.
     * \param dir direction along which to fetch the neighbou. Only positive
     * search directions are allowed along any axes.
     */
    template <typename Precision, typename FieldType, typename FieldSelector>
    static inline std::pair<Precision, Eigen::Vector3i> fetch_neighbour_sample(
        Node<FieldType>* stack[],
        Node<FieldType>* octant,
        const int        max_depth,
        const int        dir,
        FieldSelector    select_value) {

      int depth = se::keyops::depth(octant->code_);
      while (depth > 0) {
        int child_idx = se::child_idx(stack[depth]->code_, max_depth);
        int sibling = child_idx ^ dir;
        if ((sibling & dir) == dir) { // if sibling still in octant's family
          const int child_size = 1 << (max_depth - depth);
          const Eigen::Vector3i coords = se::keyops::decode(stack[depth-1]->code_)
              + child_size * Eigen::Vector3i((sibling & 1), (sibling & 2) >> 1, (sibling & 4) >> 2);
          return {select_value(stack[depth - 1]->data_[sibling]), coords};
        }
        depth--;
      }
      return {Precision(), Eigen::Vector3i::Constant(INVALID_SAMPLE)};
    }



    /*! \brief Fetch the neighbour of octant in the desired direction which is at
     * most refined as the starting octant.
     * \param stack stack of ancestor nodes of octant
     * \param octant base octant.
     * \param max_depth maximum depth of the tree.
     * \param dir direction along which to fetch the neighbou. Only positive
     * search directions are allowed along any axes.
     */
    template <typename FieldType>
    static inline Node<FieldType>* fetch_neighbour(Node<FieldType>* stack[],
                                                   Node<FieldType>* octant,
                                                   const int        max_depth,
                                                   const int        dir) {

      int depth = se::keyops::depth(octant->code_);
      while (depth > 0) {
        int child_idx = se::child_idx(stack[depth]->code_, max_depth);
        int sibling = child_idx ^ dir;
        if ((sibling & dir) == dir) { // if sibling still in octant's family
          return stack[depth - 1]->child(sibling);
        }
        depth--;
      }
      return nullptr;
    }



    /*! \brief Fetch the finest octant containing (x,y,z) starting from root node.
     * It is required that pos is contained withing the root node, i.e. pos is
     * within the interval [root.pos, root.pos + root.size].
     * \param stack stack of traversed nodes
     * \param root Root node from where the search starts.
     * \param pos integer position of searched octant
     */
    template <typename T>
    static inline Node<T>* fetch(Node<T>*               stack[],
                                 Node<T>*               root,
                                 const int              max_depth,
                                 const Eigen::Vector3i& pos) {

      unsigned node_size = (1 << (max_depth - se::keyops::depth(root->code_))) / 2;
      constexpr unsigned int block_size = BLOCK_SIZE;
      Node<T>* node = root;
      int d = 0;
      for (; node_size >= block_size; ++d, node_size = node_size >> 1) {
        stack[d] = node;
        auto next = node->child(
            (pos.x() & node_size) > 0u,
            (pos.y() & node_size) > 0u,
            (pos.z() & node_size) > 0u);
        if (!next)
          break;
        node = next;
      }
      stack[d] = node;
      return node;
    }
  } // end namespace internal_multires
} // end namespace se

#endif

