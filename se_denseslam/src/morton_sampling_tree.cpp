// SPDX-FileCopyrightText: 2021 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#include "se/morton_sampling_tree.hpp"

#include <cstdio>
#include <exception>
#include <queue>
#include <random>
#include <stack>

#include "se/octant_ops.hpp"

namespace se {

MortonSamplingTree::MortonSamplingTree(const std::vector<key_t>& codes, const int voxel_depth) :
        voxel_depth_(voxel_depth), root_(new MortonSamplingTreeNode)
{
    assert((voxel_depth_ >= 0) && "The voxel depth must be non-negative");
    for (const auto code : codes) {
        addCode(code);
    }
}



bool MortonSamplingTree::sampleCode(key_t& code)
{
    // The constructor always initializes root_ so we skip the nullptr test except in debug.
    assert((root_) && "The root has been initialized");
    if (empty()) {
        return false;
    }
    // Find the node to sample.
    MortonSamplingTreeNode* node = root_.get();
    while (!node->isLeaf()) {
        node = node->childToSample();
        // childToSample() should never return a nullptr since it is only called on nodes with
        // children.
        assert((node) && "childToSample() shouldn't return a nullptr");
    }
    code = removeCode(node);
    return true;
}



size_t MortonSamplingTree::size() const
{
    return root_->descendants;
}



bool MortonSamplingTree::empty() const
{
    return root_->descendants == 0;
}



int MortonSamplingTree::toGraphviz(const std::string& filename) const
{
    FILE* f = fopen(filename.c_str(), "w");
    if (!f) {
        return 1;
    }
    fprintf(f, "strict digraph morton_sampling_tree {\n");
    fprintf(f, "\tnode [shape=box, fontname=\"Monospace\"];\n");
    fprintf(f, "\tedge [arrowhead=none];\n");
    // Iterate over all octree nodes.
    std::queue<const MortonSamplingTreeNode*> remaining_nodes;
    remaining_nodes.push(root_.get());
    while (!remaining_nodes.empty()) {
        const MortonSamplingTreeNode* const node = remaining_nodes.front();
        remaining_nodes.pop();
        fprintf(f,
                "\tN%016lX [label=\"%016lX\\nD:%d %zu/%zu\"]\n",
                node->code,
                node->code,
                keyops::depth(node->code),
                node->descendants,
                node->initial_descendants);
        fprintf(f, "\tN%016lX -> {", node->code);
        // Iterate over the children.
        for (int i = 0; i < 8; i++) {
            if (node->children[i]) {
                const MortonSamplingTreeNode* const child_node = node->children[i].get();
                fprintf(f, " N%016lX", child_node->code);
                remaining_nodes.push(child_node);
            }
        }
        fprintf(f, "}\n");
        // Color the root node.
        if (node->code == 0) {
            fprintf(f, "\tN%016lX [style=filled, fillcolor=lightblue]\n", node->code);
        }
    }
    fprintf(f, "}\n");
    return fclose(f);
}



void MortonSamplingTree::addCode(const key_t code)
{
    std::stack<key_t> code_per_depth;
    std::stack<int> child_idx_per_depth;
    key_t code_at_depth = code;
    do {
        code_per_depth.push(code_at_depth);
        child_idx_per_depth.push(child_idx(code_at_depth, voxel_depth_));
        code_at_depth = parent(code_per_depth.top(), voxel_depth_);
    } while (code_at_depth != 0);

    // Descend into the tree and update the nodes as required.
    MortonSamplingTreeNode* node = root_.get();
    node->descendants++;
    node->initial_descendants++;
    while (!code_per_depth.empty()) {
        const int child_idx = child_idx_per_depth.top();
        // Allocate the child if needed.
        if (!node->children[child_idx]) {
            const key_t child_code = code_per_depth.top();
            node->children[child_idx] = std::make_unique<MortonSamplingTreeNode>(child_code, node);
        }
        // Update the child node descendant counts.
        node = node->children[child_idx].get();
        node->descendants++;
        node->initial_descendants++;
        code_per_depth.pop();
        child_idx_per_depth.pop();
    }
}



key_t MortonSamplingTree::removeCode(MortonSamplingTreeNode* node)
{
    assert((node) && "The node isn't null");
    assert((node->isLeaf()) && "A leaf node is being removed");
    const key_t code = node->sample();
    // Deallocate the sampled node and any leaf parent nodes resulting from the deallocation.
    for (MortonSamplingTreeNode* n = node; n && n->isLeaf();) {
        const int idx = child_idx(n->code, voxel_depth_);
        n = n->parent;
        if (n) {
            n->children[idx].reset();
        }
    }
    return code;
}



MortonSamplingTreeNode::MortonSamplingTreeNode() :
        code(0), parent(nullptr), descendants(0), initial_descendants(0)
{
}



MortonSamplingTreeNode::MortonSamplingTreeNode(const key_t _code,
                                               MortonSamplingTreeNode* const _parent) :
        code(_code), parent(_parent), descendants(0), initial_descendants(0)
{
    assert((parent) && "The parent pointer isn't null");
}



MortonSamplingTreeNode* MortonSamplingTreeNode::childToSample() const
{
    // Get raw pointers to all the existing children.
    std::vector<MortonSamplingTreeNode*> cp;
    for (const auto& p : children) {
        if (p) {
            cp.push_back(p.get());
        }
    }

    // Find the maximum number of descendants among the children.
    const int max_percent_full =
        (*std::max_element(cp.begin(), cp.end(), [](const auto* a, const auto* b) {
            return a->percentFull() < b->percentFull();
        }))->percentFull();

    // Get the children with the maximum number of descendants.
    std::vector<MortonSamplingTreeNode*> max_children;
    std::copy_if(cp.begin(), cp.end(), std::back_inserter(max_children), [=](const auto* n) {
        return n->percentFull() == max_percent_full;
    });

    if (max_children.size() > 1) {
        // Find the maximum number of initial descendants among the children.
        const size_t max_initial_descendants =
            (*std::max_element(cp.begin(), cp.end(), [](const auto* a, const auto* b) {
                return a->initial_descendants < b->initial_descendants;
            }))->initial_descendants;
        // Keep the children with the maximum number of initial descendants.
        std::remove_if(max_children.begin(), max_children.end(), [=](const auto* n) {
            return n->initial_descendants < max_initial_descendants;
        });
    }

    MortonSamplingTreeNode* max_child = nullptr;
    if (max_children.size() >= 1) {
        // Randomly keep one child.
        std::vector<MortonSamplingTreeNode*> child;
        std::sample(max_children.begin(),
                    max_children.end(),
                    std::back_inserter(child),
                    1,
                    std::mt19937{std::random_device{}()});
        max_child = child.front();
    }

    return max_child;
}



key_t MortonSamplingTreeNode::sample()
{
    assert((isLeaf()) && "Sampling should be performed on leaves");
    // Decrement the descendant count of this and all parent nodes.
    for (MortonSamplingTreeNode* node = this; node; node = node->parent) {
        node->descendants--;
    }
    return code;
}



bool MortonSamplingTreeNode::isLeaf() const
{
    return std::all_of(
        children.begin(), children.end(), [](const auto& n) { return n == nullptr; });
}



int MortonSamplingTreeNode::percentFull() const
{
    return std::round(100.0f * descendants / initial_descendants);
}

} // namespace se
