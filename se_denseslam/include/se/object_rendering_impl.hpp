#ifndef OBJECT_RAYCASTING_IMPL_HPP
#define OBJECT_RAYCASTING_IMPL_HPP

template<typename RaycastF>
ObjectHit raycast_objects(const Objects& objects,
                          const std::map<int, cv::Mat>& raycasting_masks,
                          const Eigen::Vector2f pixel,
                          const Eigen::Vector3f& ray_origin_MC,
                          const Eigen::Vector3f& ray_dir_M,
                          const float near_dist,
                          const float far_dist,
                          RaycastF raycast)
{
    ObjectHit hit;
    // Raycast each object.
    float nearest_hit_dist = INFINITY;
    float nearest_hit_prob = 0.0f;
    for (const auto& o : objects) {
        // Skip pixels outside the bounding volume mask.
        if (!raycasting_masks.empty()
            && !raycasting_masks.at(o->instance_id).at<se::mask_elem_t>(pixel.y(), pixel.x())) {
            continue;
        }
        const Eigen::Vector3f t_OC = (o->T_OM_ * ray_origin_MC.homogeneous()).head(3);
        const Eigen::Vector3f ray_dir_O = se::math::to_rotation(o->T_OM_) * ray_dir_M;
        const Eigen::Vector4f hit_O = raycast(*(o->map_), t_OC, ray_dir_O, near_dist, far_dist);
        if (hit_O.w() >= 0.0f) {
            const float hit_distance = (t_OC - hit_O.head<3>()).norm();
            if (hit_distance > nearest_hit_dist) {
                continue;
            }
            const float fg_prob =
                o->map_
                    ->interpAtPoint(hit_O.head<3>(), [](const auto& data) { return data.getFg(); })
                    .first;
            if (fg_prob <= 0.5f
                || (hit_distance == nearest_hit_dist && fg_prob <= nearest_hit_prob)) {
                continue;
            }
            // Good hit found.
            nearest_hit_prob = fg_prob;
            nearest_hit_dist = hit_distance;
            hit.instance_id = o->instance_id;
            hit.scale = static_cast<int8_t>(hit_O.w());
            // Fetch the VoxelBlock containing the hit and get its minimum updated scale. Not sure
            // why block would ever be nullptr if we got a valid hit but it happens.
            const auto* block = o->map_->fetch(o->map_->pointToVoxel(hit_O.head<3>()));
            if (block) {
                hit.min_scale = block->minScaleReached();
            }
            hit.hit_M = (o->T_MO_ * hit_O.head(3).homogeneous()).head(3);
            const Eigen::Vector3f normal_O =
                o->map_->gradAtPoint(hit_O.head<3>(),
                                     ObjVoxelImpl::VoxelType::selectNodeValue,
                                     ObjVoxelImpl::VoxelType::selectVoxelValue,
                                     static_cast<int>(hit_O.w() + 0.5f));
            if (normal_O.norm() == 0.0f) {
                hit.normal_M = Eigen::Vector3f(INVALID, 0.f, 0.f);
            }
            else {
                // Invert normals for TSDF representations.
                hit.normal_M = se::math::to_rotation(o->T_MO_)
                    * (ObjVoxelImpl::invert_normals ? -normal_O : normal_O).normalized();
            }
        }
    }
    return hit;
}

#endif // OBJECT_RAYCASTING_IMPL_HPP
