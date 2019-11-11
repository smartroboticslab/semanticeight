/*
 * Created by binbin on 25/04/18.
 * Modified by Sotiris Papatheodorou
 */

#include <iomanip>
#include <numeric>

#include <se/object.hpp>

#include "se/semanticeight_definitions.hpp"



  Object::Object(const std::shared_ptr<se::Octree<VoxelImpl::VoxelType>> map,
                 const Eigen::Vector2i&                                  image_res,
                 const Eigen::Matrix4f&                                  T_OM,
                 const Eigen::Matrix4f&                                  T_MC,
                 const int                                               instance_id)
    : instance_id(instance_id),
      num_blocks_per_min_scale({}),
      detected_integrations(0),
      undetected_integrations(0),
      image_res_(image_res),
      prev_surface_point_cloud_M_(image_res.x(), image_res.y()),
      prev_surface_normals_M_(image_res.x(), image_res.y()),
      raycast_T_MC_(T_MC),
      surface_point_cloud_M_(image_res.x(), image_res.y()),
      surface_normals_M_(image_res.x(), image_res.y()),
      T_OM_(T_OM),
      T_MO_(se::math::to_inverse_transformation(T_OM_)) {
    map_ = map;
}



Object::Object(const Eigen::Vector2i&         image_res,
               const Eigen::Vector3i&         map_size,
               const Eigen::Vector3f&         map_dim,
               const Eigen::Matrix4f&         T_OM,
               const Eigen::Matrix4f&         T_MC,
               const int                      instance_id)
    : instance_id(instance_id),
      num_blocks_per_min_scale({}),
      detected_integrations(0),
      undetected_integrations(0),
      image_res_(image_res),
      prev_surface_point_cloud_M_(image_res.x(), image_res.y()),
      prev_surface_normals_M_(image_res.x(), image_res.y()),
      raycast_T_MC_(T_MC),
      surface_point_cloud_M_(image_res.x(), image_res.y()),
      surface_normals_M_(image_res.x(), image_res.y()),
      T_OM_(T_OM),
      T_MO_(se::math::to_inverse_transformation(T_OM_)) {
  map_ = std::shared_ptr<se::Octree<VoxelImpl::VoxelType>>(new se::Octree<VoxelImpl::VoxelType>());
  map_->init(map_size.x(), map_dim.x());
}



int Object::minScale() const {
  for (size_t i = 0; i < num_blocks_per_min_scale.size(); i++) {
    if (num_blocks_per_min_scale[i] > 0) {
      return i;
    }
  }
  // No blocks at any scale.
  return -1;
}



int Object::maxScale() const {
  for (size_t i = num_blocks_per_min_scale.size(); i-- > 0;) {
    if (num_blocks_per_min_scale[i] > 0) {
      return i;
    }
  }
  // No blocks at any scale.
  return -1;
}



bool Object::finished() const {
  const float num_blocks = std::accumulate(num_blocks_per_min_scale.begin(), num_blocks_per_min_scale.end(), 0);
  const float pc_scale_0 = num_blocks_per_min_scale[0] / num_blocks;
  return pc_scale_0 > 0.9f;
}



void Object::integrate(const se::Image<float>&         depth_image,
                       const se::Image<uint32_t>&      rgba_image,
                       const se::InstanceSegmentation& segmentation,
                       const Eigen::Matrix4f&          T_MC,
                       const SensorImpl&               sensor,
                       const size_t                    frame) {
    // Create the integration mask.
    cv::Mat mask = segmentation.generateIntegrationMask();
#if SE_VERBOSE >= SE_VERBOSE_DETAILED
  std::cout << __func__ << " in: "
      << "   RGB " << rgba_image.width() << "x" << rgba_image.height()
      << "   Depth " << depth_image.width() << "x" << depth_image.height()
      << "   Masks " << mask.cols << "x" << mask.rows << "\n";
#endif
  // Over-allocate memory for octant list
  const float voxel_size = map_->voxelDim();
  const int num_vox_per_pix = map_->dim()
      / ((se::VoxelBlock<VoxelImpl::VoxelType>::size_li) * voxel_size);
  const size_t total = num_vox_per_pix
      * depth_image.width() * depth_image.height();
  allocation_list_.reserve(total);


#if SE_VERBOSE >= SE_VERBOSE_DETAILED
  std::cout << __func__ << ":"
      << "   voxel_size: " << voxel_size
      << "   num_vox_per_pix: " << num_vox_per_pix
      << "   total: " << total << "\n";
#endif

  const Eigen::Matrix4f& T_OC = T_OM_ * T_MC;
  const size_t allocated = VoxelImpl::buildAllocationList(
      *map_,
      depth_image,
      T_OC,
      sensor,
      allocation_list_.data(),
      allocation_list_.capacity());


#if SE_VERBOSE >= SE_VERBOSE_DETAILED
  std::cout << __func__ << ":" << "   allocated: " << allocated << "\n";
#endif

  // Allocate the required octants.
  map_->allocate(allocation_list_.data(), allocated);

  // Update the map
  VoxelImpl::integrate(
      *map_,
      depth_image,
      rgba_image,
      mask,
      se::math::to_inverse_transformation(T_OC),
      sensor,
      frame);

  // Update the integration count.
  if (segmentation.detected()) {
    detected_integrations++;
  } else {
    undetected_integrations++;
  }

  // Update the object class confidence.
  conf.merge(segmentation.conf);

  // Count the number of VoxelBlocks at each scale.
  num_blocks_per_min_scale.fill(0);
  std::vector<ObjVoxelImpl::VoxelBlockType*> block_list;
  // TODO SEM Are the active blocks correct for MultiresOFusion?
  // TODO SEM How to distinguish between free and occupied blocks in MultiresOFusion?
  map_->getBlockList(block_list, false);
  for (const auto* block : block_list) {
    if (block) {
      num_blocks_per_min_scale[block->min_scale()]++;
    }
  }
#if SE_VERBOSE >= SE_VERBOSE_DETAILED
  std::cout << __func__ << " out\n";
#endif
}



void Object::raycast(const Eigen::Matrix4f& T_MC,
                     const SensorImpl&      sensor) {
  raycast_T_MC_ = T_MC;
  raycastKernel<VoxelImpl>(*map_, surface_point_cloud_M_, surface_normals_M_,
      raycast_T_MC_, sensor);
}



void Object::renderObjectVolume(uint32_t*              output_image_data,
                                const Eigen::Vector2i& output_image_res,
                                const SensorImpl&      sensor,
                                const Eigen::Matrix4f& render_T_MC,
                                const bool             /*render_color*/) {
  se::Image<Eigen::Vector3f> render_surface_point_cloud_M (image_res_.x(), image_res_.y());
  se::Image<Eigen::Vector3f> render_surface_normals_M (image_res_.x(), image_res_.y());
  if (render_T_MC.isApprox(raycast_T_MC_)) {
    // Copy the raycast from the camera viewpoint. Can't safely use memcpy with
    // Eigen objects it seems.
    for (size_t i = 0; i < surface_point_cloud_M_.size(); ++i) {
      render_surface_point_cloud_M[i] = surface_point_cloud_M_[i];
      render_surface_normals_M[i] = surface_normals_M_[i];
    }
  } else {
    // Raycast the map from the render viewpoint.
    raycastKernel<VoxelImpl>(*map_, render_surface_point_cloud_M,
        render_surface_normals_M, render_T_MC, sensor);
  }
  renderVolumeKernel<VoxelImpl>(output_image_data, output_image_res,
      se::math::to_translation(render_T_MC), ambient,
      render_surface_point_cloud_M, render_surface_normals_M);
}



std::ostream& operator<<(std::ostream& os, const Object& o) {
  const std::ios_base::fmtflags f (os.flags());
  // Assuming up to 999 objects.
  os << "Object " << std::setw(3) << o.instance_id << ", "
    // The longest class name has 16 characters (chest_of_drawers) but we'll likely not need larger
    // than 8 characters (backpack).
    << std::setw(8) << se::class_id_to_str(o.classId()) << " (" << o.classId() << ") "
    << std::setw(3) << std::fixed << std::setprecision(0) << 100.0f * o.conf.confidence() << "%, scales";
  // Show detailed information about the minimum scales of the allocated VoxelBlocks.
  const int num_blocks = std::accumulate(o.num_blocks_per_min_scale.begin(), o.num_blocks_per_min_scale.end(), 0);
  for (int s = 0; s <= ObjVoxelImpl::VoxelBlockType::max_scale; s++) {
    const float pc = 100.0f * o.num_blocks_per_min_scale[s] / num_blocks;
    os << " " << std::setw(3) << std::fixed << std::setprecision(0) << pc << "%";
  }
  os.flags(f);
  return os;
}

