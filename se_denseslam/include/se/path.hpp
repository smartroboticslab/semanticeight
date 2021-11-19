// SPDX-FileCopyrightText: 2021 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PATH_HPP
#define PATH_HPP

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <string>
#include <vector>

namespace se {

typedef std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> Path;

int write_path_tsv(const std::string& filename, const Path& path);

int write_path_ply(const std::string& filename, const Path& path);

} // namespace se

#endif // PATH_HPP
