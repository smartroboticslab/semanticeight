// SPDX-FileCopyrightText: 2021 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MORTON_SAMPLING_TREE_IMPL_HPP
#define MORTON_SAMPLING_TREE_IMPL_HPP

#include <array>
#include <vector>

namespace se {

struct MortonSamplingTreeNode {
    const key_t code;
    MortonSamplingTreeNode* const parent;
    size_t descendants;
    size_t initial_descendants;
    std::array<std::unique_ptr<MortonSamplingTreeNode>, 8> children;

    /** Construct the tree root node.
     */
    MortonSamplingTreeNode();

    /** Construct an intermediate tree node.
     */
    MortonSamplingTreeNode(const key_t code, MortonSamplingTreeNode* const parent);

    /** Return a pointer to the child node the next sample should be taken from. Returns nullptr if
     * the node has no children.
     */
    MortonSamplingTreeNode* childToSample() const;

    /** Return the Morton code of the node and propagate the descendant decrement to the root.
     */
    key_t sample();

    /** A node is a leaf if its children array contains only nullptrs.
     */
    bool isLeaf() const;

    /** Return the percentage of descendants over initial_descendants rounded to the nearest
     * integer.
     */
    int percentFull() const;
};

} // namespace se

#endif // MORTON_SAMPLING_TREE_IMPL_HPP
