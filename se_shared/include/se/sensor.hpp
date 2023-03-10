// SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#ifndef __SENSOR_HPP
#define __SENSOR_HPP

#include <Eigen/Dense>
#include <cmath>

#include "se/image/image.hpp"
#include "se/image_utils.hpp"
#include "se/projection.hpp"
#include "se/segmentation.hpp"



namespace se {

struct SensorConfig {
    // General
    int width = 0;
    int height = 0;
    bool left_hand_frame = false;
    float near_plane = 0.f;
    float far_plane = INFINITY;
    // Pinhole camera
    float fx = nan("");
    float fy = nan("");
    float cx = nan("");
    float cy = nan("");
    // LIDAR
    Eigen::VectorXf beam_azimuth_angles = Eigen::VectorXf(1);
    Eigen::VectorXf beam_elevation_angles = Eigen::VectorXf(1);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};



struct PinholeCamera {
    PinholeCamera(const SensorConfig& c);

    PinholeCamera(const PinholeCamera& pinhole_camera, const float scaling_factor);

    /**
     * \brief Determine the corresponding image value of the projected pixel for a point_C in camera frame.
     *
     * \param sensor          Reference to the used sensor used for the projection.
     * \param point_C         3D coordinates of the point to be projected in camera frame.
     * \param depth_image     Image
     * \param depth_value     Reference to the depth value to be determined.
     * \param rgba_image      Corresponding RGBA image.
     * \param rgba_value      Reference to the corresponding RGBA value.
     * \param mask            Always returns false where the mask value is
     *                        negative, even if the projection is valid and the
     *                        predicate is satisfied.
     * \param mask_value      Reference to the corresponding mask value.
     * \param valid_predicate Functor indicating if the fetched pixel value is valid.
     *
     * \return is_valid   Returns true if the projection is successful and false if the projection is unsuccessful
     *                    or the pixel value is invalid.
     */
    template<typename ValidPredicate>
    bool projectToPixelValue(const Eigen::Vector3f& point_C,
                             const se::Image<float>& image,
                             float& image_value,
                             const se::Image<uint32_t>& rgba_image,
                             uint32_t& rgba_value,
                             const cv::Mat& mask,
                             se::integration_mask_elem_t& mask_value,
                             ValidPredicate valid_predicate) const
    {
        Eigen::Vector2f pixel_f;
        if (model.project(point_C, &pixel_f) != srl::projection::ProjectionStatus::Successful) {
            return false;
        }
        const Eigen::Vector2i pixel = se::round_pixel(pixel_f);
        mask_value = mask.at<se::integration_mask_elem_t>(pixel.y(), pixel.x());
        image_value = image(pixel.x(), pixel.y());
        rgba_value = rgba_image(pixel.x(), pixel.y());
        // Return false for invalid depth measurement
        if (!valid_predicate(image_value, rgba_value, mask_value)) {
            return false;
        }
        return true;
    }



    template<typename ValidPredicate>
    bool getPixelValue(const Eigen::Vector2f& pixel_f,
                       const se::Image<float>& image,
                       float& image_value,
                       ValidPredicate valid_predicate) const
    {
        if (!model.isInImage(pixel_f)) {
            return false;
        }
        Eigen::Vector2i pixel = se::round_pixel(pixel_f);
        image_value = image(pixel.x(), pixel.y());
        // Return false for invalid depth measurement
        if (!valid_predicate(image_value)) {
            return false;
        }
        return true;
    }



    /**
     * \brief Computes the scale corresponding to the back-projected pixel size
     * in voxel space
     * \param[in] block_centre    The coordinates of the VoxelBlock
     *                            centre in the camera frame.
     * \param[in] voxel_dim       The voxel edge length in meters.
     * \param[in] last_scale      Scale from which propagate up voxel
     *                            values.
     * \param[in] min_scale       Finest scale at which data has been
     *                            integrated into the voxel block (-1 if no
     *                            data has been integrated yet).
     * \param[in] max_block_scale The maximum allowed scale within a
     *                            VoxelBlock.
     * \return The scale that should be used for the integration.
     */
    int computeIntegrationScale(const Eigen::Vector3f& block_centre,
                                const float voxel_dim,
                                const int last_scale,
                                const int min_scale,
                                const int max_block_scale) const;

    int targetIntegrationScale(const Eigen::Vector3f& block_centre,
                               const float voxel_dim,
                               const int max_block_scale) const;

    /**
     * \brief Return the minimum distance at which measurements are available
     * along the ray passing through pixels x and y.
     *
     * This differs from the PinholeCamera::near_plane since the near_plane is
     * a z-value while nearDist is a distance along a ray.
     *
     * \param[in] ray_C The ray starting from the camera center and expressed
     *                  in the camera frame along which nearDist
     *                  will be computed.
     * \return The minimum distance along the ray through the pixel at which
     *         valid measurements may be encountered.
     */
    float nearDist(const Eigen::Vector3f& ray_C) const;

    /**
     * \brief Return the maximum distance at which measurements are available
     * along the ray passing through pixels x and y.
     *
     * This differs from the PinholeCamera::far_plane since the far_plane is a
     * z-value while farDist is a distance along a ray.
     *
     * \param[in] ray_C The ray starting from the camera center and expressed
     *                  in the camera frame along which nearDist
     *                  will be computed.
     * \return The maximum distance along the ray through the pixel at which
     *         valid measurements may be encountered.
     */
    float farDist(const Eigen::Vector3f& ray_C) const;

    /**
     * \brief Convert a point in the sensor frame into a depth measurement.
     * For the PinholeCamera this means returning the z-coordinate
     * of the point.
     *
     * \param[in] point_C A point observed by the sensor expressed in the
     *                    sensor frame.
     * \return The depth value that the sensor would get from this point.
     */
    float measurementFromPoint(const Eigen::Vector3f& point_C) const;

    /**
     * \brief Test whether a 3D point in camera coordinates is inside the
     * camera frustum.
     */
    bool pointInFrustum(const Eigen::Vector3f& point_C) const;

    /**
     * \brief Test whether a 3D point in camera coordinates is inside the
     * camera frustum.
     *
     * The difference from PinholeCamera::pointInFrustum is that it is assumed
     * that the far plane is at infinity.
     */
    bool pointInFrustumInf(const Eigen::Vector3f& point_C) const;

    /**
     * \brief Test whether a sphere in camera coordinates is inside the camera
     * frustum.
     *
     * It is tested whether the sphere's center is inside the camera frustum
     * offest outwards by the sphere's radius. This is a quick test that in
     * some rare cases may return a sphere as being visible although it isn't.
     */
    bool sphereInFrustum(const Eigen::Vector3f& center_C, const float radius) const;

    /**
     * \brief Test whether a sphere in camera coordinates is inside the camera
     * frustum.
     *
     * The difference from PinholeCamera::sphereInFrustum is that it is assumed
     * that the far plane is at infinity.
     */
    bool sphereInFrustumInf(const Eigen::Vector3f& center_C, const float radius) const;

    /**
     * \brief Test whether a 3D AABB in camera coordinates is inside the camera frustum.
     *
     * This implements the method described here
     * http://www.lighthouse3d.com/tutorials/view-frustum-culling/geometric-approach-testing-boxes/
     * which can result in some false positives.
     */
    bool aabbInFrustum(const Eigen::Matrix<float, 3, 8>& vertices_C) const;

    /**
     * \brief Test whether a 3D AABB in camera coordinates is inside the camera frustum.
     *
     * The difference from PinholeCamera::aabbInFrustum is that it is assumed that the far plane is
     * at infinity.
     */
    bool aabbInFrustumInf(const Eigen::Matrix<float, 3, 8>& vertices_C) const;

    /**
     * \brief Test whether a 3D ray in camera coordinates is inside the
     * camera frustum.
     *
     * The difference from PinholeCamera::pointInFrustum is that it is assumed
     * that the near plane is at 0 and the far plane is at infinity.
     */
    bool rayInFrustum(const Eigen::Vector3f& ray_C) const;

    static std::string type()
    {
        return "pinholecamera";
    }



    srl::projection::PinholeCamera<srl::projection::NoDistortion> model;
    bool left_hand_frame;
    float near_plane;
    float far_plane;
    float scaled_pixel;
    /** \brief The horizontal field of view in radians. */
    float horizontal_fov;
    /** \brief The vertical field of view in radians. */
    float vertical_fov;

    static constexpr int num_frustum_vertices_ = 8;
    static constexpr int num_frustum_normals_ = 6;
    Eigen::Matrix<float, 4, num_frustum_vertices_> frustum_vertices_;
    /** \brief The frustum normals point towards the interior of the frustum. */
    Eigen::Matrix<float, 4, num_frustum_normals_> frustum_normals_;

    /** The radius of the sphere centered on the camera frame and containing the frustum. */
    float radius;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    private:
    void computeFrustumVertices();
    void computeFrustumNormals();
};



struct OusterLidar {
    OusterLidar(const SensorConfig& c);

    OusterLidar(const OusterLidar& ouster_lidar, const float scaling_factor);

    Eigen::Matrix3f K() const
    {
        return Eigen::Matrix3f::Zero();
    }

    /**
     * \brief Determine the corresponding image value of the projected pixel for a point_C in camera frame.
     *
     * \param sensor          Reference to the used sensor used for the projection.
     * \param point_C         3D coordinates of the point to be projected in camera frame.
     * \param depth_image     Image
     * \param depth_value     Reference to the depth value to be determined.
     * \param valid_predicate Functor indicating if the fetched pixel value is valid.
     *
     * \return is_valid   Returns true if the projection is successful and false if the projection is unsuccessful
     *                    or the depth value is invalid.
     */
    template<typename ValidPredicate>
    bool projectToPixelValue(const Eigen::Vector3f& point_C,
                             const se::Image<float>& image,
                             float& image_value,
                             const se::Image<uint32_t>& /* rgba_image */,
                             uint32_t& rgba_value,
                             const cv::Mat& /* mask */,
                             se::integration_mask_elem_t& mask_value,
                             ValidPredicate valid_predicate) const
    {
        Eigen::Vector2f pixel_f;
        if (model.project(point_C, &pixel_f) != srl::projection::ProjectionStatus::Successful) {
            return false;
        }
        const Eigen::Vector2i pixel = se::round_pixel(pixel_f);
        image_value = image(pixel.x(), pixel.y());
        // Return false for invalid depth measurement
        if (!valid_predicate(image_value, rgba_value, mask_value)) {
            return false;
        }
        return true;
    }



    template<typename ValidPredicate>
    bool getPixelValue(const Eigen::Vector2f& pixel_f,
                       const se::Image<float>& image,
                       float& image_value,
                       ValidPredicate valid_predicate) const
    {
        if (!model.isInImage(pixel_f)) {
            return false;
        }
        Eigen::Vector2i pixel = se::round_pixel(pixel_f);
        image_value = image(pixel.x(), pixel.y());
        // Return false for invalid depth measurement
        if (!valid_predicate(image_value)) {
            return false;
        }
        return true;
    }



    /**
     * \brief Computes the scale corresponding to the back-projected pixel size
     * in voxel space
     * \param[in] block_centre    The coordinates of the VoxelBlock
     *                            centre in the camera frame.
     * \param[in] voxel_dim       The voxel edge length in meters.
     * \param[in] last_scale      Scale from which propagate up voxel
     *                            values.
     * \param[in] min_scale       Finest scale at which data has been
     *                            integrated into the voxel block (-1 if no
     *                            data has been integrated yet).
     * \param[in] max_block_scale The maximum allowed scale within a
     *                            VoxelBlock.
     * \return The scale that should be used for the integration.
     */
    int computeIntegrationScale(const Eigen::Vector3f& block_centre,
                                const float voxel_dim,
                                const int last_scale,
                                const int min_scale,
                                const int max_block_scale) const;

    int targetIntegrationScale(const Eigen::Vector3f& block_centre,
                               const float voxel_dim,
                               const int max_block_scale) const;

    /**
     * \brief Return the minimum distance at which measurements are available
     * along the ray passing through pixels x and y.
     *
     * This function just returns OusterLidar::near_plane.
     *
     * \param[in] ray_C The ray starting from the camera center and expressed
     *                  in the camera frame along which nearDist
     *                  will be computed.
     * \return The minimum distance along the ray through the pixel at which
     *         valid measurements may be encountered.
     */
    float nearDist(const Eigen::Vector3f& ray_C) const;

    /**
     * \brief Return the maximum distance at which measurements are available
     * along the ray passing through pixels x and y.
     *
     * This function just returns OusterLidar::far_plane.
     *
     * \param[in] ray_C The ray starting from the camera center and expressed
     *                  in the camera frame along which nearDist
     *                  will be computed.
     * \return The maximum distance along the ray through the pixel at which
     *         valid measurements may be encountered.
     */
    float farDist(const Eigen::Vector3f& ray_C) const;

    /**
     * \brief Convert a point in the sensor frame into a depth measurement.
     * For the OusterLidar this means returning the norm of the point.
     *
     * \param[in] point_C A point observed by the sensor expressed in the
     *                    sensor frame.
     * \return The depth value that the sensor would get from this point.
     */
    float measurementFromPoint(const Eigen::Vector3f& point_C) const;

    /**
     * \brief Test whether a 3D point in camera coordinates is inside the
     * sensor frustum.
     *
     * \todo Implement
     */
    bool pointInFrustum(const Eigen::Vector3f& point_C) const;

    /**
     * \brief Test whether a 3D point in camera coordinates is inside the
     * sensor frustum.
     *
     * \todo Implement
     *
     * The difference from OusterLidar::pointInFrustum is that it is assumed
     * that the far plane is at infinity.
     */
    bool pointInFrustumInf(const Eigen::Vector3f& point_C) const;

    /**
     * \brief Test whether a sphere in camera coordinates is inside the sensor
     * frustum.
     *
     * \todo Implement
     *
     * \todo Describe any issues/assumptions/approximations once implemented.
     */
    bool sphereInFrustum(const Eigen::Vector3f& center_C, const float radius) const;

    /**
     * \brief Test whether a sphere in camera coordinates is inside the sensor
     * frustum.
     *
     * \todo Implement
     *
     * The difference from OusterLidar::sphereInFrustum is that it is assumed
     * that the far plane is at infinity.
     */
    bool sphereInFrustumInf(const Eigen::Vector3f& center_C, const float radius) const;

    static std::string type()
    {
        return "ousterlidar";
    }

    srl::projection::OusterLidar model;
    bool left_hand_frame;
    float near_plane;
    float far_plane;
    float min_ray_angle;
    /** \brief The horizontal field of view in radians. */
    float horizontal_fov;
    /** \brief The vertical field of view in radians. */
    float vertical_fov;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

} // namespace se

#endif
