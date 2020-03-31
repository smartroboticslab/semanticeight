#include "se/voxel_implementations/MultiresOFusion/MultiresOFusion.hpp"



// Initialize static data members.
constexpr bool  MultiresOFusion::invert_normals;
constexpr float MultiresOFusion::surface_boundary;
constexpr float MultiresOFusion::min_occupancy;
constexpr float MultiresOFusion::max_occupancy;
constexpr float MultiresOFusion::max_weight;
constexpr int   MultiresOFusion::fs_integr_scale;

// Implement static member functions.
size_t MultiresOFusion::buildAllocationList(
    se::key_t*                               allocation_list,
    size_t                                   reserved,
    se::Octree<MultiresOFusion::VoxelType>&    map,
    const Eigen::Matrix4f&                   T_wc,
    const Eigen::Matrix4f&                   K,
    const float*                             depth_image,
    const Eigen::Vector2i&                   image_size,
    const float                              mu) {

  return 0;

}