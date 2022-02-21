// SPDX-FileCopyrightText: 2021 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#include "se/exploration_utils.hpp"

#include "se/utils/math_utils.h"

/** Return the volume of the intersection of two axis-aligned rectangular
 * cuboids.
 *
 * \note Assuming a_min < a_max and b_min < b_max.
 *
 * \note https://studiofreya.com/3d-math-and-physics/simple-aabb-vs-aabb-collision-detection/
 */
float rect_cuboid_intersection_volume(const Eigen::Vector3f& a_min,
                                      const Eigen::Vector3f& a_max,
                                      const Eigen::Vector3f& b_min,
                                      const Eigen::Vector3f& b_max)
{
    const Eigen::Vector3f a_center = (a_max + a_min) / 2.0f;
    const Eigen::Vector3f b_center = (b_max + b_min) / 2.0f;
    const Eigen::Vector3f a_sides = a_max - a_min;
    const Eigen::Vector3f b_sides = b_max - b_min;
    const Eigen::Vector3f center_dists = (a_center - b_center).array().abs().matrix();
    const Eigen::Vector3f limit_dists = (a_sides + b_sides) / 2.0f;

    if ((a_min.array() <= b_min.array()).all() && (b_max.array() <= a_max.array()).all()) {
        // a contains b.
        return b_sides.prod();
    }
    else if ((b_min.array() <= a_min.array()).all() && (a_max.array() <= b_max.array()).all()) {
        // b contains a.
        return a_sides.prod();
    }
    else if ((center_dists.array() < limit_dists.array()).all()) {
        // a intersects b if they overlap in all 3 axes.
        Eigen::Vector3f inters_sides = limit_dists - center_dists;
        // The intersection side cannot be larger than the side of the smallest cuboid.
        const Eigen::Vector3f min_sides = (a_sides.array().min(b_sides.array())).matrix();
        if (inters_sides.x() > min_sides.x()) {
            inters_sides.x() = min_sides.x();
        }
        if (inters_sides.y() > min_sides.y()) {
            inters_sides.y() = min_sides.y();
        }
        if (inters_sides.z() > min_sides.z()) {
            inters_sides.z() = min_sides.z();
        }
        return inters_sides.x() * inters_sides.y() * inters_sides.z();
    }
    else {
        // a and b do not intersect.
        return 0.0f;
    }
}



namespace se {
ExploredVolume::ExploredVolume(se::Octree<VoxelImpl::VoxelType>& map,
                               const Eigen::Vector3f& aabb_min_M,
                               const Eigen::Vector3f& aabb_max_M)
{
    const bool intersect = aabb_min_M.x() < aabb_max_M.x() && aabb_min_M.y() < aabb_max_M.y()
        && aabb_min_M.z() < aabb_max_M.z();
    for (const auto& volume : map) {
        // The iterator will not return invalid (uninitialized) data so just focus on free and
        // occupied.
        if (se::math::cu(volume.dim) == 0.0f) {
            std::cout << "WAT?\n";
        }
        float v = 0.0f;
        if (intersect) {
            // Compute the voxel AABB.
            const Eigen::Vector3f voxel_min_M =
                volume.centre_M - Eigen::Vector3f::Constant(volume.dim / 2.0f);
            const Eigen::Vector3f voxel_max_M =
                volume.centre_M + Eigen::Vector3f::Constant(volume.dim / 2.0f);
            // Intersect with the map AABB.
            v = rect_cuboid_intersection_volume(aabb_min_M, aabb_max_M, voxel_min_M, voxel_max_M);
        }
        else {
            v = se::math::cu(volume.dim);
        }
        if (VoxelImpl::VoxelType::isFree(volume.data)) {
            free_volume += v;
        }
        else {
            occupied_volume += v;
        }
    }
    explored_volume = free_volume + occupied_volume;
}



void freeSphere(se::Octree<VoxelImpl::VoxelType>& map,
                const Eigen::Vector3f& centre_M,
                float radius)
{
    if (!std::is_same<VoxelImpl, MultiresOFusion>::value) {
        throw std::domain_error("Only MultiresOFusion is supported");
    }
    // Compute the sphere's AABB corners in metres and voxels
    const Eigen::Vector3f aabb_min_M = centre_M - Eigen::Vector3f::Constant(radius);
    const Eigen::Vector3f aabb_max_M = centre_M + Eigen::Vector3f::Constant(radius);
    // Compute the coordinates of all the points corresponding to voxels in the AABB
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> aabb_points_M;
    for (float z = aabb_min_M.z(); z <= aabb_max_M.z(); z += map.voxelDim()) {
        for (float y = aabb_min_M.y(); y <= aabb_max_M.y(); y += map.voxelDim()) {
            for (float x = aabb_min_M.x(); x <= aabb_max_M.x(); x += map.voxelDim()) {
                aabb_points_M.push_back(Eigen::Vector3f(x, y, z));
            }
        }
    }
    // Allocate the required VoxelBlocks
    std::set<se::key_t> code_set;
    for (const auto& point_M : aabb_points_M) {
        const Eigen::Vector3i voxel = map.pointToVoxel(point_M);
        if (map.contains(voxel)) {
            code_set.insert(map.hash(voxel.x(), voxel.y(), voxel.z(), map.blockDepth()));
        }
    }
    std::vector<se::key_t> codes(code_set.begin(), code_set.end());
    map.allocate(codes.data(), codes.size());
    // The data to store in the free voxels
    auto data = VoxelImpl::VoxelType::initData();
    data.x = -21; // The path planning threshold is -20.
    data.y = 1;
    data.observed = true;
    // Allocate the VoxelBlocks up to some scale
    constexpr int scale = 3;
    std::vector<VoxelImpl::VoxelBlockType*> blocks;
    map.getBlockList(blocks, false);
    for (auto& block : blocks) {
        block->active(true);
        block->allocateDownTo(scale);
    }
    // Set the sphere voxels to free
    for (const auto& point_M : aabb_points_M) {
        const Eigen::Vector3f voxel_dist_M = (centre_M - point_M).array().abs().matrix();
        if (voxel_dist_M.norm() <= radius) {
            map.setAtPoint(point_M, data, scale);
        }
    }
}
} // namespace se
