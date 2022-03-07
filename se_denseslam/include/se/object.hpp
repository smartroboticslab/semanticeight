/*
 * Created by binbin on 25/04/18.
 * Modified by Sotiris Papatheodorou
 */

#ifndef __OBJECT_HPP
#define __OBJECT_HPP

#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

#include "se/bounding_volume.hpp"
#include "se/commons.h"
#include "se/config.h"
#include "se/image/image.hpp"
#include "se/octree.hpp"
#include "se/perfstats.h"
#include "se/preprocessing.hpp"
#include "se/rendering.hpp"
#include "se/segmentation_result.hpp"
#include "se/semanticeight_definitions.hpp"
#include "se/sensor_implementation.hpp"
#include "se/timings.h"
#include "se/tracking.hpp"
#include "se/voxel_implementations.hpp"



/**
 * Contains the storage and all relevant parameters for a detected object.
 */
struct Object {
    template<typename T>
    using ScaleArray = std::array<T, ObjVoxelImpl::VoxelBlockType::num_scales>;

    /**
     * The instance ID of the object. Used to distinguish between different
     * objects with the same semantic class label.
     */
    int instance_id;

    /** The confidence of all semantic classes for this object. Also used to store the object's
     * semantic class ID.
     */
    se::DetectionConfidence conf;

    /** The number of VoxelBlocks with minimum data at each scale. It is updated in
     * Object::integrate().
     */
    ScaleArray<int> num_blocks_per_min_scale;

    /** The number of measurements integrated into the object where the object was detected by the
     * semantic segmentation network.
     */
    size_t detected_integrations;

    /** The number of measurements integrated into the object where the object was not detected by
     * the semantic segmentation network. In that case the measurement was integrated based on the
     * object's raycasting mask.
     */
    size_t undetected_integrations;

    // vertex and normal maps before intergration
    Eigen::Vector2i image_res_;
    se::Image<Eigen::Vector3f> prev_surface_point_cloud_M_;
    se::Image<Eigen::Vector3f> prev_surface_normals_M_;

    // Raycasting
    Eigen::Matrix4f raycast_T_MC_;
    se::Image<Eigen::Vector3f> surface_point_cloud_M_;
    se::Image<Eigen::Vector3f> surface_normals_M_;
    se::Image<int8_t> scale_image_;
    se::Image<int8_t> min_scale_image_;

    // Map
    Eigen::Matrix4f T_OM_;
    Eigen::Matrix4f T_MO_;
    std::shared_ptr<se::Octree<ObjVoxelImpl::VoxelType>> map_;

    /**
     * A solid that contains all the vertices of this volume. Used for efficient raycasting. The
     * bounding solid is created in the map frame.
     */
#if SE_BOUNDING_VOLUME == SE_BV_SPHERE
    se::BoundingSphere bounding_volume_M_;
#elif SE_BOUNDING_VOLUME == SE_BV_BOX
    se::AABB bounding_volume_M_;
#endif



    Object(const std::shared_ptr<se::Octree<ObjVoxelImpl::VoxelType>> map,
           const Eigen::Vector2i& image_res,
           const Eigen::Matrix4f& T_OM,
           const Eigen::Matrix4f& T_MC,
           const int instance_id);

    Object(const Eigen::Vector2i& image_res,
           const Eigen::Vector3i& map_size,
           const Eigen::Vector3f& map_dim,
           const Eigen::Matrix4f& T_OM,
           const Eigen::Matrix4f& T_MC,
           const int instance_id);

    int classId() const
    {
        return conf.classId();
    }

    float mapDim() const
    {
        return map_->dim();
    }

    int mapSize() const
    {
        return map_->size();
    }

    float voxelDim() const
    {
        return map_->voxelDim();
    }

    int minScale() const;

    int maxScale() const;

    bool finished() const;

    /**
     * Integrate a depth image taken from T_MC into the volume.
     *
     * \param[in] depth_image Input depth image
     * \param[in] rgb_image Input rgb image
     * \param[in] mask Segmentation mask corresponding this volume instance
     * \param[in] T_MC Camera pose
     * \param[in] frame Frame number
     *
     * \note This function should be very similar to the original supereight
     * DenseSLAMSystem::integrate() function.
     */
    void integrate(const se::Image<float>& depth_image,
                   const se::Image<uint32_t>& rgba_image,
                   const se::InstanceSegmentation& segmentation,
                   const cv::Mat& raycasted_object_mask,
                   const Eigen::Matrix4f& T_MC,
                   const SensorImpl& sensor,
                   const size_t frame);

    /**
     * Raycast the volume from view in order to create the vertex and normal
     * maps surface_point_cloud_M_ and surface_normals_M_ respectively.
     */
    void raycast(const Eigen::Matrix4f& T_MC, const SensorImpl& sensor);

    /**
     * Render the volume of the Object as viewed from render_T_MC.
     */
    void renderObjectVolume(uint32_t* output_image_data,
                            const Eigen::Vector2i& output_image_res,
                            const SensorImpl& sensor,
                            const Eigen::Matrix4f& render_T_MC,
                            const bool render_color);

    Object::ScaleArray<float> percentageAtScale() const;

    void print(FILE* f = stdout) const;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};



std::ostream& operator<<(std::ostream& os, const Object& o);

typedef std::shared_ptr<Object> ObjectPtr;
typedef std::vector<ObjectPtr> Objects;

#endif
