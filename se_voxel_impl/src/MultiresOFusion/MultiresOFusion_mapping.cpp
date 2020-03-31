#include <se/voxel_implementations/MultiresOFusion/MultiresOFusion.hpp>
#include <se/voxel_implementations/MultiresOFusion/sensor_model.hpp>
#include "../../../se_denseslam/include/se/constant_parameters.h"
#include <set>
#include <se/utils/math_utils.h>
#include <Eigen/Core>

struct AllocateAndUpdateRecurse {

AllocateAndUpdateRecurse(se::Octree<MultiresOFusion::VoxelType>&                        map,
                         std::vector<se::VoxelBlock<MultiresOFusion::VoxelType>*>&      voxel_block_list,
                         std::vector<std::set<se::Node<MultiresOFusion::VoxelType>*>>&  node_list,
                         const se::Image<float>&                                      depth_image,
                         const se::KernelImage* const                                 kernel_depth_image,
                         const Eigen::Matrix4f&                                       K,
                         const Eigen::Matrix4f&                                       P,
                         const Sophus::SE3f&                                          T_cw,
                         const float                                                  mu,
                         const float                                                  map_res,
                         const Eigen::Vector3f&                                       offset,
                         const size_t                                                 max_level,
                         const float                                                  max_depth,
                         const unsigned                                               frame) :
                         map_(map),
                         pool_(map.pool()),
                         voxel_block_list_(voxel_block_list),
                         node_list_(node_list),
                         depth_image_(depth_image),
                         kernel_depth_image_(kernel_depth_image),
                         K_(K),
                         T_cw_(T_cw),
                         P_(P),
                         mu_(mu),
                         map_res_(map_res),
                         offset_(offset),
                         max_level_(max_level),
                         max_depth_(max_depth),
                         frame_(frame),
                         scaled_pix_((K.inverse() *
                                      (Eigen::Vector3f(1, 0 ,1) - Eigen::Vector3f(0, 0, 1)).homogeneous()).x()),
                         dist_thresh_0_(1.5 * map_res / scaled_pix_),
                         dist_thresh_1_(3   * map_res / scaled_pix_),
                         dist_thresh_2_(6   * map_res / scaled_pix_),
                         zero_depth_band_(1.0e-6f),
                         size_2_radius_(std::sqrt(3.0f) / 2.0f){

  point_offset_ << 0, 1, 0, 1, 0, 1, 0, 1,
      0, 0, 1, 1, 0, 0, 1, 1,
      0, 0, 0, 0, 1, 1, 1, 1;


};

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  se::Octree<MultiresOFusion::VoxelType>& map_;
  se::DummyMemoryPool<MultiresOFusion::VoxelType>& pool_;
  std::vector<se::VoxelBlock<MultiresOFusion::VoxelType>*>& voxel_block_list_;
  std::vector<std::set<se::Node<MultiresOFusion::VoxelType>*>>& node_list_;
  const se::Image<float>&      depth_image_;
  const se::KernelImage* const kernel_depth_image_;
  const Eigen::Matrix4f& K_;
  const Eigen::Matrix4f& P_;
  const Sophus::SE3f& T_cw_;
  const float mu_;
  const float map_res_;
  const Eigen::Vector3f& offset_;
  const size_t max_level_;
  const float max_depth_;
  const int frame_;
  const float scaled_pix_;
  const float dist_thresh_0_;
  const float dist_thresh_1_;
  const float dist_thresh_2_;
  Eigen::Matrix<float, 3, 8> point_offset_;
  const float zero_depth_band_;
  const float size_2_radius_;

  static inline constexpr int max_scale_ = se::math::log2_const(se::VoxelBlock<MultiresOFusion::VoxelType>::side);

  float getDepthSample(float center_u, float center_v, float center_d)
  {
    int32_t cUi = static_cast<int32_t>(center_u + 0.5f);
    int32_t cVi = static_cast<int32_t>(center_v + 0.5f);

    if(cUi < 0 || cUi >= depth_image_.width() || cVi < 0 || cVi >= depth_image_.height() || center_d <= 0)
    {
      return 0.0f;
    }

    auto depth = (depth_image_.data())[cUi + cVi * depth_image_.width()];
    return depth;
  }

  /**
   * \brief Computes the scale corresponding to the back-projected pixel size
   * in voxel space
   * \param[in] vox centroid coordinates of the test voxel
   * \param[in] twc translational component of the camera position
   * \param[in] scale_pix unitary pixel side after application of inverse
   * calibration matrixfre
   * \param[out] scale scale from which propagate up voxel values
   */
  inline float computeScale(const Eigen::Vector3f& vox,
                            const Eigen::Vector3f& twc,
                            const Eigen::Matrix3f& Rcw,
                            const int              last_scale,
                            const int              min_scale) {

    const float dist = (Rcw * (map_res_ * vox - twc)).z();

    int scale = 0;
    if (dist < dist_thresh_0_)
      scale = 0;
    else if (dist < dist_thresh_1_)
      scale = 1;
    else if (dist < dist_thresh_2_)
      scale = 2;
    else
      scale = 3;
    scale = std::min(scale, max_scale_);

    float dist_hyst = dist;
    bool recompute = false;
    if (scale > last_scale && min_scale != -1) {
      dist_hyst = dist - 0.25;
      recompute = true;
    } else if (scale < last_scale && min_scale != -1) {
      dist_hyst = dist + 0.25;
      recompute = true;
    }

    if (recompute) {
      if (dist_hyst < dist_thresh_0_)
        scale = 0;
      else if (dist_hyst < dist_thresh_1_)
        scale = 1;
      else if (dist_hyst < dist_thresh_2_)
        scale = 2;
      else
        scale = 3;
      scale = std::min(scale, max_scale_);
    }

    return scale;
  }

  static MultiresOFusionData propagateUp(se::Node<MultiresOFusion::VoxelType>* node, const int max_level,
                           const unsigned frame) {
    if(!node->parent()) {
      node->timestamp(frame);
      return MultiresOFusion::VoxelType::empty();
    }

    int num_data = 0;
    int y_max    = 0;
    int num_observed = 0;
    unsigned last_frame = 0;
    float x_max = 2 * MultiresOFusion::min_occupancy;
    for(int i = 0; i < 8; ++i) {
      const auto& tmp = node->value_[i];
      if (tmp.y > 0) { // At least 1 integration
        num_data++;
        if (tmp.x_max > x_max) {
          x_max = tmp.x_max;
          y_max = tmp.y;
        }
        if (tmp.frame > last_frame)
          last_frame = tmp.frame;
      }
      if (tmp.observed == true)
      {
        num_observed++;
      }
    }

    const unsigned int id = se::child_id(node->code_,
                                         se::keyops::level(node->code_), max_level);
    auto& data = node->parent()->value_[id];
    if(num_data != 0) {
      data.x_max  = x_max;
      data.y      = y_max;
      data.frame  = last_frame;
      if (num_observed == 8)
        data.observed = true;
    }
    
    node->timestamp(frame);
    return data;
  }

  /**
   * Update the subgrids of a voxel block starting from a given scale up
   * to a maximum scale.
   *
   * \param[in] block VoxelBlock to be updated
   * \param[in] scale scale from which propagate up voxel values
  */
  static void propagateUp(se::VoxelBlock<MultiresOFusion::VoxelType>* block, const int scale) {
    const Eigen::Vector3i base = block->coordinates();
    const int side = se::VoxelBlock<MultiresOFusion::VoxelType>::side;
    for(int curr_scale = scale; curr_scale < max_scale_; ++curr_scale) {
      const int stride = 1 << (curr_scale + 1);
      for(int z = 0; z < side; z += stride)
        for(int y = 0; y < side; y += stride)
          for(int x = 0; x < side; x += stride) {
            const Eigen::Vector3i curr = base + Eigen::Vector3i(x, y, z);
            float mean_x = 0;
            float mean_y = 0;
            int num_data = 0;
            int num_observed = 0;
            unsigned last_frame = 0;
            float x_max = MultiresOFusion::min_occupancy;
            for(int k = 0; k < stride; k += stride/2)
              for(int j = 0; j < stride; j += stride/2 )
                for(int i = 0; i < stride; i += stride/2) {
                  auto tmp = block->data(curr + Eigen::Vector3i(i, j , k), curr_scale);
                  if (tmp.y > 0) {
                    mean_x += tmp.x;
                    mean_y += tmp.y;
                    num_data++;
                    if (tmp.x_max > x_max)
                      x_max = tmp.x_max;
                    if (tmp.frame > last_frame)
                      last_frame = tmp.frame;
                  }
                  if (tmp.observed)
                  {
                    num_observed++;
                  }
                }
            auto data = block->data(curr, curr_scale + 1);

            if(num_data != 0) {
              // TODO: This needs to be discussed.
              //  Should be devided by num_data or num_children for probability voxel type
              mean_x       /= num_data;
              mean_y       /= num_data;
              data.x        = mean_x;
              data.x_last   = mean_x;
              data.y        = mean_y;
              data.y_last   = mean_y;
              data.x_max    = x_max;
              data.frame    = last_frame;
              if (num_observed == 8)
                data.observed = true;
            } else { // TODO: SEEMS REDUNDANT
              data = MultiresOFusion::VoxelType::initValue();
            }
            block->data(curr, curr_scale + 1, data);
          }
    }
  }

  static float interp(const se::Octree<MultiresOFusion::VoxelType>& map,
                      const se::VoxelBlock<MultiresOFusion::VoxelType>* block,
                      const Eigen::Vector3i& vox, const int scale, bool& valid) {
    Eigen::IOFormat CleanFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", " << ", ";");

    // Compute base point in parent block
    const int side   = se::VoxelBlock<MultiresOFusion::VoxelType>::side >> (scale + 1);
    const int stride = 1 << (scale + 1);

    const Eigen::Vector3f& offset = se::Octree<MultiresOFusion::VoxelType>::_offset;
    Eigen::Vector3i base = stride * (vox.cast<float>()/stride - offset).cast<int>().cwiseMax(Eigen::Vector3i::Constant(0));

    const Eigen::Vector3f vox_f  = vox.cast<float>() + offset * (stride/2);
    Eigen::Vector3f base_f = base.cast<float>() + offset*(stride);
    Eigen::Vector3f factor = (vox_f - base_f) / stride;

    for (int i = 0; i < 3; i++) {
      if (factor[i] < 0) {
        factor[i]  = 1 + factor[i];
        base[i]   -= stride;
        base_f[i] -= stride;
      }
    }

    float points[8];
    se::internal_multires::gather_points(map, block->coordinates() + base, scale + 1,
                                         [](const auto& val) { return val.x; }, points);

    bool observed[8];
    se::internal_multires::gather_points(map, block->coordinates() + base, scale + 1,
                            [](const auto& val) { return val.observed; }, observed);
    for(int i = 0; i < 8; ++i) {
      if(observed[i] == 0) {
        valid = false;
        return MultiresOFusion::VoxelType::initValue().x;
      }
    }
    valid = true;

    const float v_000 = points[0] * (1 - factor.x()) + points[1] * factor.x();
    const float v_001 = points[2] * (1 - factor.x()) + points[3] * factor.x();
    const float v_010 = points[4] * (1 - factor.x()) + points[5] * factor.x();
    const float v_011 = points[6] * (1 - factor.x()) + points[7] * factor.x();

    const float v_0 = v_000 * (1 - factor.y()) + v_001 * (factor.y());
    const float v_1 = v_010 * (1 - factor.y()) + v_011 * (factor.y());

    const float val = v_0 * (1 - factor.z()) + v_1 * factor.z();

    return val;
  }

  /**
   * Update a voxel block at a given scale by first propagating down the parent
   * values and then integrating the new measurement;
  */
  void propagateDowndate(se::VoxelBlock<MultiresOFusion::VoxelType>* block,
                         const int scale) {

    const int side = se::VoxelBlock<MultiresOFusion::VoxelType>::side;
    const int parent_scale = scale + 1;
    const int stride = 1 << parent_scale;
    const int half_stride = stride >> 1;

    const Eigen::Vector3i base = block->coordinates();
    const Eigen::Matrix3f delta = T_cw_.rotationMatrix() * (Eigen::Matrix3f() << map_res_, 0, 0,
        0, map_res_, 0,
        0, 0, map_res_).finished();
    const Eigen::Matrix3f camera_delta = K_.topLeftCorner<3,3>() * delta;
    const Eigen::Vector3f base_scaled =
        T_cw_ * (map_res_ * (block->coordinates().template cast<float>() + half_stride * offset_));
    const Eigen::Vector3f base_camera = K_.topLeftCorner<3, 3>() * base_scaled;

    for(int z = 0; z < side; z += stride) {
      for(int y = 0; y < side; y += stride) {
        for(int x = 0; x < side; x += stride) {
          const Eigen::Vector3i parent = base + Eigen::Vector3i(x, y, z);
          auto data = block->data(parent, parent_scale);
          float delta_x = data.x - data.x_last;
          float delta_y = data.y - data.y_last;
          for(int k = 0; k < stride; k += half_stride) {
            for(int j = 0; j < stride; j += half_stride) {
              for(int i = 0; i < stride; i += half_stride) {
                const Eigen::Vector3i vox = parent + Eigen::Vector3i(i, j , k);
                auto curr = block->data(vox, scale);

                bool is_valid = true;
                auto occupancy = interp(map_, block, vox - base, scale, is_valid);

                if (curr.observed == false) {
                  if (is_valid) {
                    curr.x        = occupancy;
                    curr.x_max    = curr.x * data.y;
                    curr.y        = data.y;
                    curr.y_last   = data.y;
                    curr.frame    = data.frame;
                    curr.observed = true;
                    block->data(vox, scale, curr);
                  }
                } else {
                  float delta_y_curr = (delta_y + curr.y > MultiresOFusion::max_weight) ? MultiresOFusion::max_weight - curr.y : delta_y;
                  float delta_x_curr = (curr.x + delta_x < -5.015) ? -5.015 - curr.x : delta_x;
                  curr.x = curr.x + delta_x_curr;
                  curr.x = se::math::clamp(curr.x + delta_x, -5.015f, 5.015f);
                  curr.frame = data.frame;
                  curr.observed = true;
                  if (is_valid) { // TODO: occupancy is not scaled correctly
                    curr.x = (curr.x * curr.y + occupancy * delta_y) / (curr.y + delta_y);
                  }
                  curr.y = curr.y + delta_y_curr;
                  curr.x_max = se::math::clamp(curr.x * curr.y, MultiresOFusion::min_occupancy, MultiresOFusion::max_occupancy);
                  block->data(vox, scale, curr);
                }

                const Eigen::Vector3f camera_voxel = base_camera + camera_delta * Eigen::Vector3f(x + i, y + j, z + k);
                const Eigen::Vector3f pos = base_scaled + delta * Eigen::Vector3f(x + i, y + j, z + k);

                if (pos.z() < 0.0001f || pos.z() > max_depth_) {
                  block->data(vox, scale, curr);
                  continue;
                }
                const float inverse_depth = 1.f / camera_voxel.z();
                const Eigen::Vector2f pixel = Eigen::Vector2f(
                    camera_voxel.x() * inverse_depth + 0.5f,
                    camera_voxel.y() * inverse_depth + 0.5f);
                if (pixel.x() < 0.5f || pixel.x() > depth_image_.width() - 1.5f ||
                    pixel.y() < 0.5f || pixel.y() > depth_image_.height() - 1.5f) {
                  block->data(vox, scale, curr);
                  continue;
                }
                const Eigen::Vector2i px = pixel.cast<int>();
                const float depth_sample = depth_image_[px.x() + depth_image_.width()*px.y()];
                // Continue on invalid depth measurement
                if (depth_sample <=  0) {
                  block->data(vox, scale, curr);
                  continue;
                }

                float proj_scale =  std::sqrt( 1 + se::math::sq(pos.x() / pos.z())
                                                                         + se::math::sq(pos.y() / pos.z()));
                // Update the LogOdd
                sensor_model<OFusionModel<MultiresOFusion::VoxelType::VoxelData>>::updateBlock(pos[2], depth_sample, mu_, map_res_, curr, frame_, scale, proj_scale);
                block->data(vox, scale, curr);
              }
            }
          }
        }
      }
    }
    block->current_scale(scale);
  }

  /**
   * Update a voxel block at a given scale by first updating the observad state of all voxels at the
   * scale to true if the have been partially observed (y > 0);
  */
  void propagateUpdate(se::VoxelBlock<MultiresOFusion::VoxelType>* block,
                       const int scale) {
    const Eigen::Vector3i base = block->coordinates();
    constexpr int side = se::VoxelBlock<MultiresOFusion::VoxelType>::side;
    block->current_scale(scale);
    const int stride = 1 << scale;

    const Eigen::Vector3f delta = T_cw_.rotationMatrix() * Eigen::Vector3f(map_res_, 0, 0);
    const Eigen::Vector3f cameraDelta = K_.topLeftCorner<3,3>() * delta;
    for(int z = 0; z < side; z += stride) {
      for (int y = 0; y < side; y += stride) {
        Eigen::Vector3i pix = base + Eigen::Vector3i(0, y, z);
        Eigen::Vector3f start = T_cw_ * (map_res_ * (pix.cast<float>() + stride * offset_));
        Eigen::Vector3f camerastart = K_.topLeftCorner<3, 3>() * start;
        for (int x = 0; x < side; x += stride, pix.x() += stride) {
          const Eigen::Vector3f camera_voxel = camerastart + (x * cameraDelta);
          const Eigen::Vector3f pos = start + (x * delta);
          auto data = block->data(pix, scale);
          if (data.y > 0)
            data.observed = true;
          if (pos.z() < 0.0001f || pos.z() > max_depth_) {
            block->data(pix, scale, data);
            continue;
          }
          const float inverse_depth = 1.f / camera_voxel.z();
          const Eigen::Vector2f pixel = Eigen::Vector2f(
              camera_voxel.x() * inverse_depth + 0.5f,
              camera_voxel.y() * inverse_depth + 0.5f);

          if (pixel.x() < 0.5f || pixel.x() > depth_image_.width() - 1.5f ||
              pixel.y() < 0.5f || pixel.y() > depth_image_.height() - 1.5f) {
            block->data(pix, scale, data);
            continue;
          }

          const Eigen::Vector2i px = pixel.cast<int>();
          const float depth_sample = depth_image_[px.x() + depth_image_.width() * px.y()];
          // Continue on invalid depth measurement
          if (depth_sample <= 0) {
            block->data(pix, scale, data);
            continue;
          }

          float proj_scale = std::sqrt(1 + se::math::sq(pos.x() / pos.z())
                                       + se::math::sq(pos.y() / pos.z()));
          // Update the LogOdd
          sensor_model<OFusionModel<MultiresOFusion::VoxelType::VoxelData>>::updateBlock(pos[2], depth_sample, mu_,
                                                                                         map_res_, data, frame_, scale,
                                                                                         proj_scale);
          block->data(pix, scale, data);
        }
      }
    }
  }

  void updateBlock(se::VoxelBlock<MultiresOFusion::VoxelType>* block, int min_scale = 0) {
    constexpr int side = se::VoxelBlock<MultiresOFusion::VoxelType>::side;
    const Eigen::Vector3i base = block->coordinates();
    const int last_scale = block->current_scale();
    int scale = computeScale((base + Eigen::Vector3i::Constant(side/2)).cast<float>(),
                              T_cw_.inverse().translation(), T_cw_.rotationMatrix(), last_scale, block->min_scale());
    scale = std::max(last_scale - 1, std::max(scale, min_scale));
    block->min_scale(block->min_scale() < 0 ? scale : std::min(block->min_scale(), scale));

    if(last_scale > scale) {
      // Down propagate values first
      propagateDowndate(block, scale);
      return;
    } else if (last_scale < scale) {
      // Update observed state at new scale
      propagateUpdate(block, scale);
      return;
    }

    block->current_scale(scale);
    const int stride = 1 << scale;

    const Eigen::Vector3f delta = T_cw_.rotationMatrix() * Eigen::Vector3f(map_res_, 0, 0);
    const Eigen::Vector3f cameraDelta = K_.topLeftCorner<3,3>() * delta;
    for(int z = 0; z < side; z += stride)
      for(int y = 0; y < side; y += stride) {
        Eigen::Vector3i pix = base + Eigen::Vector3i(0, y, z);
        Eigen::Vector3f start = T_cw_ * (map_res_ * (pix.cast<float>() + stride*offset_));
        Eigen::Vector3f camerastart = K_.topLeftCorner<3,3>() * start;
        for(int x = 0; x < side; x += stride, pix.x() += stride) {
          const Eigen::Vector3f camera_voxel = camerastart + (x*cameraDelta);
          const Eigen::Vector3f pos = start + (x*delta);
          if (pos.z() < 0.0001f || pos.z() > max_depth_) continue;
          const float inverse_depth = 1.f / camera_voxel.z();
          const Eigen::Vector2f pixel = Eigen::Vector2f(
              camera_voxel.x() * inverse_depth + 0.5f,
              camera_voxel.y() * inverse_depth + 0.5f);

          if (pixel.x() < 0.5f || pixel.x() > depth_image_.width() - 1.5f ||
              pixel.y() < 0.5f || pixel.y() > depth_image_.height() - 1.5f) continue;

          const Eigen::Vector2i px = pixel.cast<int>();
          const float depth_sample = depth_image_[px.x() + depth_image_.width()*px.y()];
          // Continue on invalid depth measurement
          if (depth_sample <=  0) continue;
          auto data = block->data(pix, scale);

          float proj_scale =  std::sqrt( 1 + se::math::sq(pos.x() / pos.z())
                                         + se::math::sq(pos.y() / pos.z()));
          // Update the LogOdd
          sensor_model<OFusionModel<MultiresOFusion::VoxelType::VoxelData>>::updateBlock(pos[2], depth_sample, mu_, map_res_, data, frame_, scale, proj_scale);
          block->data(pix, scale, data);
        }
      }
  }

  void freeNodeRecurs(se::Node<MultiresOFusion::VoxelType>* node, int level) {
    for (int i = 0; i < 8; i++) {
      auto child_node = node->child(i);
      if (child_node) {
        if (child_node->isLeaf()) {
          se::VoxelBlock<MultiresOFusion::VoxelType>* block = dynamic_cast<se::VoxelBlock<MultiresOFusion::VoxelType>*>(child_node);
          // Voxel block has a low variance (unknown data and frustum crossing allowed). Update data at a minimum
          // free space integration scale or coarser (depending on later scale selection).
          updateBlock(block, MultiresOFusion::fs_integr_scale);
#pragma omp critical (voxel_lock)
          { // Add voxel block to voxel block list for later up propagation
            voxel_block_list_.push_back(dynamic_cast<se::VoxelBlock<MultiresOFusion::VoxelType>*>(block));
          }
        } else {
          freeNodeRecurs(child_node, level + 1);
        }
      } else {
        sensor_model<OFusionModel<MultiresOFusion::VoxelType::VoxelData>>::freeNode(node->value_[i], frame_);
#pragma omp critical (node_lock)
        { // Add node to node list for later up propagation (finest node for this branch)
          node_list_[level].insert(node);
        }
      }
    }
  }

  void freeNode(MultiresOFusion::VoxelType::VoxelData& data) {
    sensor_model<OFusionModel<MultiresOFusion::VoxelType::VoxelData>>::freeNode(data, frame_);
  }

  /**
   * @brief Get reference to a child value given a father node.
   * @param parent_node Ptr to father node.
   * @param rel_step Child relative position with respect to father.
   * @return Reference to value for update.
   */
  MultiresOFusion::VoxelType::VoxelData& getChildValue(se::Node<MultiresOFusion::VoxelType>* parent_node, const Eigen::Vector3i& rel_step)
  {
    assert(!parent_node->isLeaf() && "Father node must not be a leaf");

    return parent_node->value_[rel_step[2] * 4 + rel_step[1] * 2 + rel_step[0]];
  }

  /**
   * @brief Allocate a child node given a father node
   * @param parent_node Ptr to father node.
   * @param childblock_size Size in integer coordinates of the node to allocate
   * @param rel_step Child relative position with respect to father.
   * @return Ptr to allocated node
   */
  auto allocateChild(se::Node<MultiresOFusion::VoxelType>* parent_node,
                     const Eigen::Vector3i& corner,
                     const Eigen::Vector3i& rel_step,
                     const int node_size,
                     const unsigned level) -> se::Node<MultiresOFusion::VoxelType>* {
    int id = rel_step[0] + rel_step[1] * 2 + rel_step[2] * 4;
    se::Node<MultiresOFusion::VoxelType>*& node = parent_node->child(id);
    if(node) {
      // Return node if already allocated
      return node;
    }

    /*
     * Otherwise allocate and initalise child
     */

    // Update child mask
    parent_node->children_mask_ = parent_node->children_mask_ | (1 << id);

    if(node_size <= static_cast<int>(se::VoxelBlock<MultiresOFusion::VoxelType>::side)) {
      // Allocate voxel block
#pragma omp critical (voxel_alloc_lock)
      {
        auto init_value = parent_node->value_[id]; // Initalise child with parent value
        // Initialise remaining values that are not kept track of at node level
        init_value.x      = (init_value.y > 0) ? init_value.x_max / init_value.y : 0;
        init_value.x_last = init_value.x;
        init_value.y_last = init_value.y;
        node = map_.pool().acquireBlock(init_value);
        se::VoxelBlock<MultiresOFusion::VoxelType>* block = dynamic_cast<se::VoxelBlock<MultiresOFusion::VoxelType>*>(node);
        block->coordinates(corner);
        block->side_ = node_size;
        block->code_ = se::keyops::encode(corner.x(), corner.y(), corner.z(),
                                          max_level_ - log2(se::VoxelBlock<MultiresOFusion::VoxelType>::side), max_level_);
        block->parent() = parent_node;
      }
    } else {
      // Allocate node
#pragma omp critical (node_alloc_lock)
      {
        node = map_.pool().acquireNode(parent_node->value_[id]);
        node->side_ = node_size;
        node->code_ = se::keyops::encode(corner.x(), corner.y(), corner.z(), level, max_level_);
        node->parent() = parent_node;
        // Initalise all children with parent value if it differs from the default value
        if (node->parent()->value_[id].x_max != MultiresOFusion::VoxelType::initValue().x_max) {
          for (int i = 0; i < 8; i++) {
            node->value_[i] = node->parent()->value_[id];
          }
        }
      }
    }

    assert(node && "Voxel alloc failure");
    return node;
  }

  bool cameraInNode(const Eigen::Vector3i corner, const int node_size, Eigen::Vector3f t_wc) {
    Eigen::Vector3f voxel_pos = t_wc / map_res_;
    if (   voxel_pos.x() >= corner.x() && voxel_pos.x() <= corner.x() + node_size
        && voxel_pos.y() >= corner.y() && voxel_pos.y() <= corner.y() + node_size
        && voxel_pos.z() >= corner.z() && voxel_pos.z() <= corner.z() + node_size) {
      return true;
    }
    return false;
  }

  bool crossesFrustum(const Eigen::Matrix<float, 4, 8> corners_cam_hom, const Eigen::VectorXi corners_infront, const int num_corners_infront) {
    Eigen::MatrixXf corners_cam_hom_infront(3, num_corners_infront);
    int corner_id_infront = 0;
    for (int corner_id = 0; corner_id < 8; corner_id++) {
      if (corners_infront(corner_id) == 1) {
        corners_cam_hom_infront.col(corner_id_infront) = corners_cam_hom.col(corner_id).head(3);
        corner_id_infront++;
      }
    }
    Eigen::MatrixXf corner_projections(2, num_corners_infront);
    corner_projections = corners_cam_hom_infront.colwise().hnormalized().topRows(2);
    for (int projection_id = 0; projection_id < num_corners_infront; projection_id++) {
      int u = corner_projections(0, projection_id);
      int v = corner_projections(1, projection_id);
      if (kernel_depth_image_->inImage(u, v)) {
        return true;
      }
    }
    return false;
  }

  /**
   * @brief Update and allocate a given node and all its children recursively
   * @param corner      Corner of the node to be processed (bottom, left, front coordinates)
   * @param node_size   Size in [voxel] of the node to be processed
   * @param level       Level of the node to be processed (root = 0)
   * @param rel_step    Relative step of node within parent node (e.g. [1, 0, 1], [0, 1, 1])
   * @param parent_node Pointer to the nodes parent
   */
  void operator()(const Eigen::Vector3i&               corner,
                  const int                            node_size,
                  const int                            level,
                  const                                Eigen::Vector3i& rel_step,
                  se::Node <MultiresOFusion::VoxelType>* parent_node) {
//    std::cout << corner.x() << "," << corner.y() << "," << corner.z() << "|" << node_size << std::endl;
//    if (corner.x() == 384 && corner.y() == 256 && corner.z() == 128) {

    /*
     * Approximate max and min depth to quickly check if the node is behind the camera or maximum depth
     */

    // Compute the node centre's depth in camera frame
    const float half_node_size_m = node_size * map_res_ / 2.0f;
    const Eigen::Vector3f center_m = corner.cast<float>() * map_res_ + Eigen::Vector3f(half_node_size_m, half_node_size_m, half_node_size_m);
    const Eigen::Vector4f center_cam_hom = P_ * center_m.homogeneous();
    assert(std::fabs(center_cam_hom[3] - 1.0f) < zero_depth_band_);

    // Extend and reduce the depth by the sphere radius covering the entire cube
    const float approx_depth_max = center_cam_hom[2] + node_size * size_2_radius_ * map_res_;
    const float approx_depth_min = center_cam_hom[2] - node_size * size_2_radius_ * map_res_;

    // CASE 0.1 (OUT OF BOUNDS): Block is behind the camera or behind the maximum depth value
    if (   approx_depth_min > max_depth_
           || approx_depth_max <= 0) {
      return;
    }

    /*
     * Approximate a 2D bounding box covering the projected node in the image plane
     */

    // Compute the 8 corners of the node to be evaluated
    Eigen::Matrix<float, 3, 8> corners_v = ((node_size * point_offset_).colwise() + corner.cast<float>());
    Eigen::Matrix<float, 3, 8> corners_m = map_res_ * corners_v;

    // Project the 8 corners into the image plane
    Eigen::Matrix<float, 4, 8> corners_cam_hom = P_ * corners_m.colwise().homogeneous();
    // TODO: There is no validation yet that non of the depth is within the zero_depth_band_
//    Eigen::IOFormat OctaveFmt(Eigen::StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");
//    std::cout << P_.format(OctaveFmt) << std::endl;

    bool should_split = false;

    Eigen::VectorXi corners_infront(8);
    corners_infront << 1, 1, 1, 1, 1, 1, 1, 1;
    for (int corner_id = 0; corner_id < 8; corner_id++) {
      if (corners_cam_hom(2, corner_id) < zero_depth_band_) {
        corners_infront(corner_id) = 0;
      }
    }
    int num_corners_infront = corners_infront.sum();

    // CASE 0.2 (OUT OF BOUNDS): Block is behind the camera
    if (num_corners_infront == 0) {
      return;
    }

    int low_variance; // -1 := low variance infront of the surface, 0 := high variance, 1 = low_variance behind the surface
    se::KernelImage::PixelT pix; // min, max pixel batch depth + crossing frustum state + contains unknown values state

    if(level < max_level_ - log2(se::VoxelBlock<MultiresOFusion::VoxelType>::side) + 1) {
      if (num_corners_infront < 8) {
        // CASE 1 (CAMERA IN NODE):
        if (cameraInNode(corner, node_size, T_cw_.inverse().translation())) {
          should_split = true;
        } else if (crossesFrustum(corners_cam_hom, corners_infront, num_corners_infront)) {
//          // EXCEPTION: The node is crossing the frustum boundary, but already reached the min allocation scale for crossing nodes
//          if (node_size == se::VoxelBlock<MultiresOFusion::VoxelType>::side) {
//            return;
//          }
          should_split = true;
        } else {
          return;
        }
        // CASE 2 (FRUSTUM BOUNDARY): Node partly behind the camera and crosses the the frustum boundary
      } else {
        Eigen::Matrix<float, 2, 8> corner_projections = corners_cam_hom.topRows(3).colwise().hnormalized().topRows(2);
        Eigen::Matrix<float, 1, 8> corner_depth       = corners_cam_hom.row(2);

        // Compute the minimum and maximum pixel values to generate the bounding box
        Eigen::Vector2i bb_min = corner_projections.rowwise().minCoeff().cast<int>();
        Eigen::Vector2i bb_max = corner_projections.rowwise().maxCoeff().cast<int>();
        float node_dist_min = corner_depth.minCoeff();
        float node_dist_max = corner_depth.maxCoeff();

        pix = kernel_depth_image_->conservativeQuery(bb_min, bb_max);

        // CASE 0.3 (OUT OF BOUNDS): The node is behind surface
        if (approx_depth_min > pix.max + sensor_model<OFusionModel<MultiresOFusion::VoxelType::VoxelData>>::computeSigma()) {
          return;
        }

        // CASE 0.4 (OUT OF BOUNDS): The node is outside frustum (i.e left, right, below, above) or all pixel values are unknown -> return intermediately
        if(pix.status_2 == 2) {
          return;
        }

        low_variance = sensor_model<OFusionModel<MultiresOFusion::VoxelType>>::lowVariance(
            pix.min, pix.max, node_dist_min, node_dist_max, mu_);

        const se::key_t code = se::keyops::encode(corner.x(), corner.y(), corner.z(), level, max_level_);
        const unsigned int id = se::child_id(code, level, map_.maxLevel());

        // CASE 1 (REDUNDANT DATA): Depth values in the bounding box are far away from the node or unknown (1).
        //                          The node to be evaluated is free (2) and fully observed (3),
        if(low_variance != 0 && parent_node->value_[id].x_max <= MultiresOFusion::min_occupancy && parent_node->value_[id].observed == true) {
          return;
        }

        // CASE 2 (FRUSTUM BOUNDARY): The node is crossing the frustum boundary
        if(pix.status_1 == 1) {
          // EXCEPTION: The node is crossing the frustum boundary, but already reached the min allocation scale for crossing nodes
//          if (node_size == se::VoxelBlock<MultiresOFusion::VoxelType>::side) {
//            return;
//          }
          should_split = true;
        }

        // CASE 3: The node is inside the frustum, but projects into partially known pixel
        else if(pix.status_2 == 1) {
          should_split = true;
        }

        // CASE 4: The node is inside the frustum with only known data + node has a potential high variance
        else if (low_variance == 0){
          should_split = true;
        }
      }
    }

    if(should_split) {
      // Returns a pointer to the according node if it has previously been allocated.
      se::Node<MultiresOFusion::VoxelType>* node = allocateChild(parent_node, corner, rel_step, node_size, level);
      if(node->isLeaf()) { // Evaluate the node directly if it is a voxel block
        node->active(true);
        // Cast from node to voxel block
        se::VoxelBlock<MultiresOFusion::VoxelType>* block = dynamic_cast<se::VoxelBlock<MultiresOFusion::VoxelType>*>(node);
        if (low_variance != 0) {
          // Voxel block has a low variance (unknown data and frustum crossing allowed). Update data at a minimum
          // free space integration scale or coarser (depending on later scale selection).
          updateBlock(block, MultiresOFusion::fs_integr_scale);
        } else {
          // Otherwise update values at the finest integration scale or coarser (depending on later scale selection).
          updateBlock(block, 0);
        }
#pragma omp critical (voxel_lock)
        { // Add voxel block to voxel block list for later up propagation
          voxel_block_list_.push_back(dynamic_cast<se::VoxelBlock<MultiresOFusion::VoxelType>*>(node));
        }
      }
      else {
        // Split! Start recursive process
#pragma omp parallel for
        for(int id = 0; id < 8; ++id) {
          int child_size = node_size / 2;
          Eigen::Vector3i child_rel_step = Eigen::Vector3i((id & 1) > 0, (id & 2) > 0, (id & 4) > 0);
          Eigen::Vector3i child_corner = corner + child_rel_step * child_size;
          (*this)(child_corner, child_size, level + 1, child_rel_step, node);
        }
      }
    } else {
      assert(level);
      int id = rel_step[0] + rel_step[1] * 2 + rel_step[2] * 4;
      se::Node<MultiresOFusion::VoxelType>* node = parent_node->child(id);
      if (!node) {
        // Node does not exist -> Does NOT have children that need to be updated
        if (low_variance == -1) {
          // Free node
          MultiresOFusion::VoxelType::VoxelData& value = getChildValue(parent_node, rel_step);
          freeNode(value);
#pragma omp critical (node_lock)
          { // Add node to node list for later up propagation (finest node for this branch)
            node_list_[level - 1].insert(parent_node);
          }
        } // else node has low variance behind surface (ignore)
      } else {
        // Node does exist -> Does POTENTIALLY have children that that need to be updated
          if (node->isLeaf()) {
            // Node is a voxel block -> Does NOT have children that need to be updated
            se::VoxelBlock<MultiresOFusion::VoxelType>* block = dynamic_cast<se::VoxelBlock<MultiresOFusion::VoxelType>*>(node);
            // Node has a low variance (unknown data and frustum crossing allowed). Update data at a minimum
            // free space integration scale or coarser (depending on later scale selection).
            updateBlock(block, MultiresOFusion::fs_integr_scale);
#pragma omp critical (voxel_lock)
            { // Add voxel block to voxel block list for later up propagation
              voxel_block_list_.push_back(dynamic_cast<se::VoxelBlock<MultiresOFusion::VoxelType>*>(node));
            }
          } else {
            // Node has children
            if (low_variance == -1) {
              //Free node recursively
              freeNodeRecurs(node, level);
            } // else node has low variance behind surface (ignore)
          }
      }
    }
  };

  void propagateNodes() {
    for(const auto& b : voxel_block_list_) {
      if(b->parent()) {
        propagateUp(b, b->current_scale());
        node_list_[map_.leavesLevel() - 1].insert(b->parent());
        const unsigned int id = se::child_id(b->code_,
                                             se::keyops::level(b->code_), map_.maxLevel());
        auto data = b->data(b->coordinates(), se::math::log2_const(se::VoxelBlock<MultiresOFusion::VoxelType>::side));
        auto& parent_data = b->parent()->value_[id];
        parent_data = data;

        auto test = b->parent()->value_[id];
        if (data.observed && data.x_max <= MultiresOFusion::min_occupancy + 0.01) { //&& data.x * data.y <= MultiresOFusion::min_occupancy
          pool_.deleteBlock(b, max_level_);
        }
      }
    }

    for (int l = map_.leavesLevel() - 1; l > 0; l--) {
      std::set<se::Node<MultiresOFusion::VoxelType>*>::iterator it;
      for (it = node_list_[l].begin(); it != node_list_[l].end(); ++it) {
        se::Node<MultiresOFusion::VoxelType>* n = *it;
        if(n->timestamp() == frame_) continue;
        if(n->parent()) {
          auto data = AllocateAndUpdateRecurse::propagateUp(n, max_level_, frame_);
          node_list_[l-1].insert(n->parent());
          if (data.observed && data.x_max <= MultiresOFusion::min_occupancy + 0.01) {
            pool_.deleteNode(n, max_level_);
          }
        }
      }
    }
  }
};

void MultiresOFusion::integrate(se::Octree<MultiresOFusion::VoxelType>& map,
                              const Sophus::SE3f&                       T_cw,
                              const Eigen::Matrix4f&                    K,
                              const se::Image<float>&                   depth_image,
                              const float                               mu,
                              const unsigned                            frame) {

  // Create min/map depth pooling image for different bounding box sizes
  const se::KernelImage* const kernel_depth_image = new se::KernelImage(depth_image); // TODO: Should be changed to unique pointer?

  float max_depth = std::min(farPlane, kernel_depth_image->maxValue() + sensor_model<OFusionModel<MultiresOFusion::VoxelType::VoxelData>>::computeSigma());
  const Eigen::Matrix4f P      = K * T_cw.matrix();
  const float map_res          = map.dim() / map.size();
  const Eigen::Vector3f offset = se::Octree<MultiresOFusion::VoxelType>::_offset;

  std::vector<se::VoxelBlock<MultiresOFusion::VoxelType>*> voxel_block_list;
  std::vector<std::set<se::Node<MultiresOFusion::VoxelType>*>> node_list(map.leavesLevel());
  AllocateAndUpdateRecurse funct(map, voxel_block_list, node_list, depth_image, kernel_depth_image, K, P, T_cw, mu,
      map_res, offset, map.maxLevel(), max_depth, frame);

  // Launch on the 8 voxels of the first level
#pragma omp parallel for
  for(int i = 0; i < 8; ++i)
  {
    int child_size = map.size() / 2;
    Eigen::Vector3i child_rel_step = Eigen::Vector3i((i & 1) > 0, (i & 2) > 0, (i & 4) > 0);
    Eigen::Vector3i child_corner = child_rel_step * child_size; // Because, + corner is (0, 0, 0) at root level
    funct(child_corner, child_size, 1, child_rel_step, map.root());
  }

  funct.propagateNodes();
  delete(kernel_depth_image);
}