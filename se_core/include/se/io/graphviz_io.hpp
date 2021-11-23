// SPDX-FileCopyrightText: 2021 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#ifndef __GRAPHVIZ_IO_HPP
#define __GRAPHVIZ_IO_HPP

#include <queue>

#include "se/octree.hpp"

namespace se {
/** Generate a Graphviz diagram in filename showing the structure of the octree.
   * Nodes are shown in gray and VoxelBlocks in light blue.
   * A lambda function data_to_str can be used to specify how Node data
   * is converted to an std::string for visualization. The signature of the lambda
   * function should be
   * ```
   * [](const typename VoxelT::VoxelData& d)->std::string
   * ```
   * If you don't want to visualize Node data just use
   * ```
   * [](const auto&)->std::string{ return std::string(); }
   * ```
   *
   * \note To visualize the resulting diagram you can use
   * `dot -Tpdf /path/to/file.gv > /tmp/file.pdf && xdg-open /tmp/file.pdf`.
   *
   * \param[in] octree      The octree to visualize.
   * \param[in] filename    The filename of the resulting Graphviz file. Graphviz
   *                        files have a .gv extension by convention.
   * \param[in] data_to_str The function used to format VoxelT::VoxelData to std::string.
   * \return             0 on success, non-zero on IO error.
   */
template<typename VoxelT, typename FunctionT>
int to_graphviz(const se::Octree<VoxelT>& octree,
                const std::string& filename,
                const FunctionT data_to_str);

/** Generate a Graphviz diagram in filename showing the data the node.
   * All allocated scales are shown for VoxelBlocks.
   * A lambda function data_to_str can be used to specify how Node data
   * is converted to an std::string for visualization similarly to se::to_graphviz.
   *
   * \todo Implement Node visualization.
   * \todo Implement visualization for other kinds of VoxelBlocks.
   *
   * \param[in] node        The node to visualize.
   * \param[in] filename    The filename of the resulting Graphviz file. Graphviz
   *                        files have a .gv extension by convention.
   * \param[in] data_to_str The function used to format VoxelT::VoxelData to std::string.
   * \return 0 on success, non-zero on IO error.
   */
template<typename VoxelT, typename FunctionT>
int to_graphviz(const se::Node<VoxelT>& node,
                const std::string& filename,
                const FunctionT data_to_str);
} // namespace se

#include "graphviz_io_impl.hpp"

#endif // __GRAPHVIZ_IO_HPP
