// SPDX-FileCopyrightText: 2021 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#ifndef __GRAPHVIZ_IO_IMPL_HPP
#define __GRAPHVIZ_IO_IMPL_HPP

namespace se {
std::string _scale_to_graphviz_colour(const int scale)
{
    switch (scale) {
    case 0:
        return std::string("aquamarine");
    case 1:
        return std::string("coral");
    case 2:
        return std::string("lightblue");
    case 3:
        return std::string("hotpink");
    default:
        return std::string("");
    }
}



template<typename VoxelT, typename FunctionT>
int to_graphviz(const se::Octree<VoxelT>& octree,
                const std::string& filename,
                const FunctionT data_to_str)
{
    // Node IDs in graphviz were chosen to be in the form N<Morton code> since Morton codes
    // uniquely identify se::Octree Nodes

    // Open the file and write the Graphviz header
    FILE* f = fopen(filename.c_str(), "w");
    if (!f) {
        return 1;
    }
    // Set most of the graph style options
    fprintf(f, "strict digraph octree {\n");
    fprintf(
        f,
        "\tnode [shape=box, style=\"filled\", fontname=\"Monospace\", fontsize=12, label=\"\"];\n");
    fprintf(f, "\tedge [arrowhead=none];\n");

    // Traverse the octree breadth-first visualizing each Node
    std::queue<se::Node<VoxelT>*> remaining_nodes;
    if (octree.root()) {
        remaining_nodes.push(octree.root());
        // Write the root node label
        fprintf(f, "\tN%ld [label=\"N 0 0 0  S:%d\"]", octree.root()->code(), octree.size());
    }
    // Store the child data string representations because the have to be written in the file after
    // the connectivity specification
    std::vector<std::string> child_data;
    child_data.reserve(8);
    while (!remaining_nodes.empty()) {
        // Write the next node
        se::Node<VoxelT>* const node = remaining_nodes.front();
        remaining_nodes.pop();
        // Start writing the child node list
        fprintf(f, "\tN%ld -> {", node->code());

        if (!node->isBlock()) {
            child_data.clear();
            // Process each Node child
            for (int i = 0; i < 8; i++) {
                const se::key_t child_code = node->childCode(i, octree.voxelDepth());
                // Write the child ID in the child node list
                fprintf(f, " N%ld", child_code);
                // Format the child coordinates and child Node size as a string starting with B or N
                // depending on whether the child is a VoxelBlock or Node
                const Eigen::Vector3i child_coords = node->childCoord(i);
                const bool child_is_block = (node->size() / 2 == VoxelBlock<VoxelT>::size_li);
                const std::string coord_str = (child_is_block ? "B " : "N ")
                    + std::to_string(child_coords.x()) + " " + std::to_string(child_coords.y())
                    + " " + std::to_string(child_coords.z())
                    + "  S:" + std::to_string(node->size() / 2) + "\\n";
                // Get the child data string representation
                const std::string data_str = data_to_str(node->childData(i));
                // Create the graphviz node label
                const std::string label_str = "N" + std::to_string(child_code) + " [label=\""
                    + coord_str + data_str + "\"];\n";
                child_data.emplace_back(label_str);
                // Add the child to the remaining nodes if it's allocated
                se::Node<VoxelT>* const child = node->child(i);
                if (child) {
                    remaining_nodes.push(child);
                }
            }
        }
        // Close the child node list
        fprintf(f, "}\n");

        // Colour VoxelBlocks differently
        if (node->isBlock()) {
            fprintf(f, "\tN%ld [fillcolor=lightblue]\n", node->code());
        }

        // Visualize the child data (must be done outside the edge declaration)
        for (const auto& data_str : child_data) {
            fprintf(f, "\t%s\n", data_str.c_str());
        }
    }

    // Write the footer and close the file
    fprintf(f, "}\n");
    return fclose(f);
}



template<typename VoxelT, typename FunctionT>
int to_graphviz(const se::Node<VoxelT>& node,
                const std::string& filename,
                const FunctionT data_to_str)
{
    using BlockT = VoxelBlockSingleMax<VoxelT>;
    // Open the file and write the Graphviz header
    FILE* f = fopen(filename.c_str(), "w");
    if (!f) {
        return 1;
    }
    // Set most of the graph style options
    fprintf(f, "strict digraph octree_node {\n");
    fprintf(f, "\tnode [shape=box, style=\"filled\", fontname=\"Monospace\", fontsize=12];\n");
    fprintf(f, "\tedge [arrowhead=none];\n");

    if (node.isBlock()) {
        // TODO SEM This will throw an std::bad_cast for non-MultiresOFusion
        const BlockT& block = dynamic_cast<const BlockT&>(node);
        const int current_scale = block.current_scale();
        // Visualize all scales
        for (int scale = current_scale; scale <= BlockT::max_scale; scale++) {
            const int scale_size = BlockT::scaleSize(scale);
            // Iterate over all voxels at this scale
            for (int z = 0; z < scale_size; z++) {
                for (int y = 0; y < scale_size; y++) {
                    for (int x = 0; x < scale_size; x++) {
                        // Generate the voxel label
                        const int voxel_idx = x + y * scale_size + z * scale_size * scale_size;
                        const int voxel_size = BlockT::scaleVoxelSize(scale);
                        const auto& data = block.data(voxel_idx, scale);
                        const std::string coord_str("V " + std::to_string(x) + " "
                                                    + std::to_string(y) + " " + std::to_string(z)
                                                    + " (" + std::to_string(voxel_idx)
                                                    + ")  S:" + std::to_string(voxel_size) + "\\n");
                        const std::string label = coord_str + data_to_str(data);
                        // Colour depending on the scale
                        const std::string colour = _scale_to_graphviz_colour(scale);
                        // Write the voxel info
                        const std::string id =
                            std::to_string(scale) + "_" + std::to_string(voxel_idx);
                        fprintf(f,
                                "\tV%s [fillcolor=%s,label=\"%s\"];\n",
                                id.c_str(),
                                colour.c_str(),
                                label.c_str());
                        // Connect this voxel to the voxels at lower scales
                        if (scale > current_scale) {
                            const int child_scale_size = BlockT::scaleSize(scale - 1);
                            fprintf(f, "\tV%s -> {", id.c_str());
                            for (int zz = 0; zz < 2; zz++) {
                                for (int yy = 0; yy < 2; yy++) {
                                    for (int xx = 0; xx < 2; xx++) {
                                        const int child_idx = (2 * x + xx)
                                            + (2 * y + yy) * child_scale_size
                                            + (2 * z + zz) * child_scale_size * child_scale_size;
                                        const std::string child_id = std::to_string(scale - 1) + "_"
                                            + std::to_string(child_idx);
                                        fprintf(f, " V%s", child_id.c_str());
                                    }
                                }
                            }
                            fprintf(f, "}\n");
                        }
                    }
                }
            }
        }
    }
    else {
        // TODO SEM implement for nodes
        throw std::runtime_error("se::to_graphviz() not implemented for Nodes");
    }

    // Write the footer and close the file
    fprintf(f, "}\n");
    return fclose(f);
}
} // namespace se

#endif // __GRAPHVIZ_IO_IMPL_HPP
