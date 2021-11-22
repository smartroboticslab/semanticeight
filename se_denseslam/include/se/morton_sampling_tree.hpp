// SPDX-FileCopyrightText: 2021 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MORTON_SAMPLING_TREE_HPP
#define MORTON_SAMPLING_TREE_HPP

#include <memory>
#include <string>

#include "se/octree_defines.h"

// TODO can it have node-level frontiers?

namespace se {

struct MortonSamplingTreeNode;

/** An octree used to sample Morton codes in a more spatially-uniform manner.
 */
class MortonSamplingTree {
    public:
    /** Generate a sampling tree from the supplied Morton codes. The voxel_depth is the one returned
     * from se::Octree::voxelDepth().
     */
    MortonSamplingTree(const std::vector<key_t>& codes, const int voxel_depth);

    /** Uniformly sample  in space (in a non-mathematical sense)a Morton code corresponding to an
     * se::Node. Return true on success and false if there are no more codes in the tree. The
     * sampled code is removed from the tree.
     */
    bool sampleCode(key_t& code);

    /** Return the number of Morton codes contained in the tree.
     */
    size_t size() const;

    /** Return whether the tree is empty.
     */
    bool empty() const;

    /** Save the current state of the sampling tree in the Graphviz dot format.
     */
    int toGraphviz(const std::string& filename) const;



    private:
    const int voxel_depth_;
    std::unique_ptr<MortonSamplingTreeNode> root_;

    /** Add a Morton code to the tree, allocating nodes as needed. This is used to incrementally
     * build the tree.
     */
    void addCode(const key_t code);

    /** Return the Morton code of the supplied node and remove it from the tree.
     */
    key_t removeCode(MortonSamplingTreeNode* node);
};

} // namespace se

#include "morton_sampling_tree_impl.hpp"

#endif // MORTON_SAMPLING_TREE_HPP
