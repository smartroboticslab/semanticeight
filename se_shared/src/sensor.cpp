// SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#include "se/sensor.hpp"

#include <cassert>

#include "se/utils/math_utils.h"

// Explicit template class instantiation
template class srl::projection::PinholeCamera<srl::projection::NoDistortion>;

// Used for initializing a PinholeCamera.
const srl::projection::NoDistortion _distortion;

// Static variables
constexpr int se::PinholeCamera::num_frustum_vertices_;
constexpr int se::PinholeCamera::num_frustum_normals_;



se::PinholeCamera::PinholeCamera(const SensorConfig& c) :
        model(c.width, c.height, c.fx, c.fy, c.cx, c.cy, _distortion),
        left_hand_frame(c.left_hand_frame),
        near_plane(c.near_plane),
        far_plane(c.far_plane),
        scaled_pixel(1 / c.fx)
{
    computeFrustumVertices();
    computeFrustumNormals();

    assert(c.width > 0);
    assert(c.height > 0);
    assert(c.near_plane >= 0.f);
    assert(c.far_plane > c.near_plane);
    assert(!std::isnan(c.fx));
    assert(!std::isnan(c.fy));
    assert(!std::isnan(c.cx));
    assert(!std::isnan(c.cy));

    horizontal_fov = 2.0f * atanf(c.width / (2.0f * c.fx));
    vertical_fov = 2.0f * atanf(c.height / (2.0f * c.fy));

    radius = frustum_vertices_.col(4).norm();
}

se::PinholeCamera::PinholeCamera(const PinholeCamera& pc, const float sf) :
        model(pc.model.imageWidth() * sf,
              pc.model.imageHeight() * sf,
              pc.model.focalLengthU() * sf,
              pc.model.focalLengthV() * sf,
              ((pc.model.imageCenterU() + 0.5f) * sf - 0.5f),
              ((pc.model.imageCenterV() + 0.5f) * sf - 0.5f),
              _distortion),
        left_hand_frame(pc.left_hand_frame),
        near_plane(pc.near_plane),
        far_plane(pc.far_plane)
{
    computeFrustumVertices();
    computeFrustumNormals();
}

int se::PinholeCamera::computeIntegrationScale(const Eigen::Vector3f& block_centre,
                                               const float voxel_dim,
                                               const int last_scale,
                                               const int min_scale,
                                               const int max_block_scale) const
{
    const float dist = block_centre.z();

    const float pv_ratio = dist * scaled_pixel / voxel_dim;

    // Negative scales may arise if the voxel_radius is greater than the
    // tangent_radius.

    if (min_scale == -1) {
        int scale = 0;
        if (pv_ratio < 1.5) {
            scale = 0;
        }
        else if (pv_ratio < 3) {
            scale = 1;
        }
        else if (pv_ratio < 6) {
            scale = 2;
        }
        else {
            scale = 3;
        }
        return scale;

        /// EQUIVALENT
        //    return std::min(std::max(0, static_cast<int>(ceil(std::log2(pv_ratio)))), max_block_scale);
    }

    float lower_thresh = 0;
    float upper_thresh = 0;
    if (last_scale == 0) {
        lower_thresh = 0;
        upper_thresh = 1.75f;
    }
    else if (last_scale == 1) {
        lower_thresh = 1.25f;
        upper_thresh = 3.25f;
    }
    else if (last_scale == 2) {
        lower_thresh = 2.75f;
        upper_thresh = 6.25f;
    }
    else {
        lower_thresh = 5.75f;
        upper_thresh = std::numeric_limits<float>::max();
    }

    /// EQUIVALENT
    //  float lower_thresh = 0;                       ///<< 0 -> 0;   1 -> 0.5f, 2 -> 1.5f, 3-> 3.f, ...
    //  if (last_scale == 1) {
    //    lower_thresh = 0.5f;
    //  } else if  (last_scale > 1) {
    //    lower_thresh = (1 << (last_scale -  2)) * 1.5;
    //  }
    //
    //  float upper_thresh = (1 << last_scale) * 1.5; ///<< 0 -> 1.5; 1 -> 3f,   2 -> 6.f,  3-> 12.f, ..., std::numeric_limits<float>max()
    //  if (last_scale == max_block_scale) {
    //    upper_thresh    = std::numeric_limits<float>::max();
    //  }

    if (pv_ratio < lower_thresh) {
        return std::max(last_scale - 1, 0);
    }
    else if (pv_ratio > upper_thresh) {
        return std::min(last_scale + 1, max_block_scale);
    }
    else {
        return last_scale;
    }
}

int se::PinholeCamera::targetIntegrationScale(const Eigen::Vector3f& block_centre,
                                              const float voxel_dim,
                                              const int max_block_scale) const
{
    const float dist = block_centre.z();
    const float pv_ratio = dist * scaled_pixel / voxel_dim;
    int scale = 0;
    if (pv_ratio < 1.5) {
        scale = 0;
    }
    else if (pv_ratio < 3) {
        scale = 1;
    }
    else if (pv_ratio < 6) {
        scale = 2;
    }
    else {
        scale = 3;
    }
    scale = std::min(scale, max_block_scale);
    return scale;
}

float se::PinholeCamera::nearDist(const Eigen::Vector3f& ray_C) const
{
    return near_plane / ray_C.normalized().z();
}

float se::PinholeCamera::farDist(const Eigen::Vector3f& ray_C) const
{
    return far_plane / ray_C.normalized().z();
}

float se::PinholeCamera::measurementFromPoint(const Eigen::Vector3f& point_C) const
{
    return point_C.z();
}

bool se::PinholeCamera::pointInFrustum(const Eigen::Vector3f& point_C) const
{
    for (size_t i = 0; i < num_frustum_normals_; ++i) {
        // Compute the signed distance between the point and the plane
        const float distance = point_C.homogeneous().dot(frustum_normals_.col(i));
        if (distance < 0.0f) {
            // A negative distance means that the point is located on the opposite
            // halfspace than the one the plane normal is pointing towards
            return false;
        }
    }
    return true;
}

bool se::PinholeCamera::pointInFrustumInf(const Eigen::Vector3f& point_C) const
{
    // Skip the far plane normal
    for (size_t i = 0; i < num_frustum_normals_ - 1; ++i) {
        // Compute the signed distance between the point and the plane
        const float distance = point_C.homogeneous().dot(frustum_normals_.col(i));
        if (distance < 0.0f) {
            // A negative distance means that the point is located on the opposite
            // halfspace than the one the plane normal is pointing towards
            return false;
        }
    }
    return true;
}

bool se::PinholeCamera::sphereInFrustum(const Eigen::Vector3f& center_C, const float radius) const
{
    for (size_t i = 0; i < num_frustum_normals_; ++i) {
        // Compute the signed distance between the point and the plane
        const float distance = center_C.homogeneous().dot(frustum_normals_.col(i));
        if (distance < -radius) {
            // Instead of testing for negative distance as in
            // se::PinholeCamera::pointInFrustum, test for distance smaller than
            // -radius so that the test is essentially performed on the plane offset
            // by radius.
            return false;
        }
    }
    return true;
}

bool se::PinholeCamera::sphereInFrustumInf(const Eigen::Vector3f& center_C,
                                           const float radius) const
{
    // Skip the far plane normal
    for (size_t i = 0; i < num_frustum_normals_ - 1; ++i) {
        // Compute the signed distance between the point and the plane
        const float distance = center_C.homogeneous().dot(frustum_normals_.col(i));
        if (distance < -radius) {
            // Instead of testing for negative distance as in
            // se::PinholeCamera::pointInFrustum, test for distance smaller than
            // -radius so that the test is essentially performed on the plane offset
            // by radius.
            return false;
        }
    }
    return true;
}

bool se::PinholeCamera::aabbInFrustum(const Eigen::Matrix<float, 3, 8>& vertices_C) const
{
    // The AABB is outside the frustum if all of its vertices are outside the same plane.
    for (size_t i = 0; i < num_frustum_normals_; ++i) {
        for (int v = 0; v < vertices_C.cols(); ++v) {
            // Compute the signed distance between the vertex and the plane.
            const float distance = vertices_C.col(v).homogeneous().dot(frustum_normals_.col(i));
            if (distance >= 0.0f) {
                // If at least one vertex is inside this plane we don't need to keep testing it.
                break;
            }
            else if (v == vertices_C.cols() - 1) {
                // All vertices were outside this plane, the AABB is guaranteed to be outside the
                // frustum.
                return false;
            }
        }
    }
    // The vertices aren't all on the outside of any single plane.
    return true;
}

bool se::PinholeCamera::aabbInFrustumInf(const Eigen::Matrix<float, 3, 8>& vertices_C) const
{
    // The AABB is outside the frustum if all of its vertices are outside the same plane.
    for (size_t i = 0; i < num_frustum_normals_ - 1; ++i) {
        for (int v = 0; v < vertices_C.cols(); ++v) {
            // Compute the signed distance between the vertex and the plane.
            const float distance = vertices_C.col(v).homogeneous().dot(frustum_normals_.col(i));
            if (distance >= 0.0f) {
                // If at least one vertex is inside this plane we don't need to keep testing it.
                break;
            }
            else if (v == vertices_C.cols() - 1) {
                // All vertices were outside this plane, the AABB is guaranteed to be outside the
                // frustum.
                return false;
            }
        }
    }
    // The vertices aren't all on the outside of any single plane.
    return true;
}

bool se::PinholeCamera::rayInFrustum(const Eigen::Vector3f& ray_C) const
{
    // Skip the near and far plane normals
    for (size_t i = 0; i < num_frustum_normals_ - 2; ++i) {
        // Compute the signed distance between the point and the plane
        const float distance = ray_C.homogeneous().dot(frustum_normals_.col(i));
        if (distance < 0.0f) {
            // A negative distance means that the point is located on the opposite
            // halfspace than the one the plane normal is pointing towards
            return false;
        }
    }
    return true;
}

void se::PinholeCamera::computeFrustumVertices()
{
    Eigen::Vector3f point_C;
    // Back-project the frame corners to get the frustum vertices
    // Top left
    model.backProject(Eigen::Vector2f(0.0f, 0.0f), &point_C);
    frustum_vertices_.col(0) = point_C.homogeneous();
    frustum_vertices_.col(4) = point_C.homogeneous();
    // Top right
    model.backProject(Eigen::Vector2f(model.imageWidth(), 0.0f), &point_C);
    frustum_vertices_.col(1) = point_C.homogeneous();
    frustum_vertices_.col(5) = point_C.homogeneous();
    // Bottom right
    model.backProject(Eigen::Vector2f(model.imageWidth(), model.imageHeight()), &point_C);
    frustum_vertices_.col(2) = point_C.homogeneous();
    frustum_vertices_.col(6) = point_C.homogeneous();
    // Bottom left
    model.backProject(Eigen::Vector2f(0.0f, model.imageHeight()), &point_C);
    frustum_vertices_.col(3) = point_C.homogeneous();
    frustum_vertices_.col(7) = point_C.homogeneous();
    // Scale the frustum vertices with the appropriate depth for near and far
    // plane vertices
    for (int i = 0; i < num_frustum_vertices_ / 2; ++i) {
        frustum_vertices_.col(i).head<3>() *= near_plane;
        frustum_vertices_.col(num_frustum_vertices_ / 2 + i).head<3>() *= far_plane;
    }
}

void se::PinholeCamera::computeFrustumNormals()
{
    // The w vector component corresponds to the distance of the plane from the
    // origin. It should be 0 for all planes other than the near and far planes.
    // Left plane vector.
    frustum_normals_.col(0) = se::math::plane_normal(
        frustum_vertices_.col(4), frustum_vertices_.col(0), frustum_vertices_.col(3));
    frustum_normals_.col(0).w() = 0.0f;
    // Right plane vector.
    frustum_normals_.col(1) = se::math::plane_normal(
        frustum_vertices_.col(1), frustum_vertices_.col(5), frustum_vertices_.col(6));
    frustum_normals_.col(1).w() = 0.0f;
    // Bottom plane vector.
    frustum_normals_.col(2) = se::math::plane_normal(
        frustum_vertices_.col(7), frustum_vertices_.col(3), frustum_vertices_.col(2));
    frustum_normals_.col(2).w() = 0.0f;
    // Top plane vector.
    frustum_normals_.col(3) = se::math::plane_normal(
        frustum_vertices_.col(5), frustum_vertices_.col(1), frustum_vertices_.col(0));
    frustum_normals_.col(3).w() = 0.0f;
    // Near plane vector.
    frustum_normals_.col(4) = Eigen::Vector4f(0.f, 0.f, 1.f, -near_plane);
    // Far plane vector.
    frustum_normals_.col(5) = Eigen::Vector4f(0.f, 0.f, -1.f, far_plane);
}



se::OusterLidar::OusterLidar(const SensorConfig& c) :
        model(c.width, c.height, c.beam_azimuth_angles, c.beam_elevation_angles),
        left_hand_frame(c.left_hand_frame),
        near_plane(c.near_plane),
        far_plane(c.far_plane)
{
    assert(c.width > 0);
    assert(c.height > 0);
    assert(c.near_plane >= 0.f);
    assert(c.far_plane > c.near_plane);
    assert(c.beam_azimuth_angles.size() > 0);
    assert(c.beam_elevation_angles.size() > 0);
    float min_elevation_angle = fabsf(c.beam_elevation_angles[1] - c.beam_elevation_angles[0]);
    for (int i = 2; i < c.beam_elevation_angles.size(); i++) {
        const float diff = fabsf(c.beam_elevation_angles[i - 1] - c.beam_elevation_angles[i]);
        if (diff < min_elevation_angle) {
            min_elevation_angle = diff;
        }
    }
    const float azimuth_angle = 360.0f / c.width;
    min_ray_angle = std::min(min_elevation_angle, azimuth_angle);
    horizontal_fov = 2.0f * M_PI;
    const float max_elevation = c.beam_elevation_angles.maxCoeff();
    const float min_elevation = c.beam_elevation_angles.minCoeff();
    vertical_fov = se::math::deg_to_rad(max_elevation - min_elevation);
}

se::OusterLidar::OusterLidar(const OusterLidar& ol, const float sf) :
        model(ol.model.imageWidth() * sf,
              ol.model.imageHeight() * sf,
              ol.model.beamAzimuthAngles(),
              ol.model.beamElevationAngles()), // TODO: Does the beam need to be scaled too?
        left_hand_frame(ol.left_hand_frame),
        near_plane(ol.near_plane),
        far_plane(ol.far_plane)
{
}

int se::OusterLidar::computeIntegrationScale(const Eigen::Vector3f& block_centre,
                                             const float voxel_dim,
                                             const int last_scale,
                                             const int min_scale,
                                             const int max_block_scale) const
{
    const float dist = block_centre.norm();
    // Compute the side length in metres of a pixel projected dist metres from
    // the camera. This computes the chord length corresponding to the ray angle
    // at distance dist.
    const float pixel_dim = 2.0f * dist * std::tan(se::math::deg_to_rad(min_ray_angle / 2.0f));
    // Compute the ratio using the worst case voxel_dim (space diagonal)
    const float pv_ratio = pixel_dim / (std::sqrt(3) * voxel_dim);
    int scale = 0;
    if (pv_ratio < 1.5f) {
        scale = 0;
    }
    else if (pv_ratio < 3.0f) {
        scale = 1;
    }
    else if (pv_ratio < 6.0f) {
        scale = 2;
    }
    else {
        scale = 3;
    }
    scale = std::min(scale, max_block_scale);

    Eigen::Vector3f block_centre_hyst = block_centre;
    bool recompute = false;
    if (scale > last_scale && min_scale != -1) {
        block_centre_hyst -= 0.25 * block_centre_hyst.normalized();
        recompute = true;
    }
    else if (scale < last_scale && min_scale != -1) {
        block_centre_hyst += 0.25 * block_centre_hyst.normalized();
        recompute = true;
    }

    if (recompute) {
        return computeIntegrationScale(
            block_centre_hyst, voxel_dim, last_scale, -1, max_block_scale);
    }
    else {
        return scale;
    }
}

int se::OusterLidar::targetIntegrationScale(const Eigen::Vector3f& block_centre,
                                            const float voxel_dim,
                                            const int max_block_scale) const
{
    const float dist = block_centre.norm();
    // Compute the side length in metres of a pixel projected dist metres from
    // the camera. This computes the chord length corresponding to the ray angle
    // at distance dist.
    const float pixel_dim = 2.0f * dist * std::tan(se::math::deg_to_rad(min_ray_angle / 2.0f));
    // Compute the ratio using the worst case voxel_dim (space diagonal)
    const float pv_ratio = pixel_dim / (std::sqrt(3) * voxel_dim);
    int scale = 0;
    if (pv_ratio < 1.5f) {
        scale = 0;
    }
    else if (pv_ratio < 3.0f) {
        scale = 1;
    }
    else if (pv_ratio < 6.0f) {
        scale = 2;
    }
    else {
        scale = 3;
    }
    scale = std::min(scale, max_block_scale);
    return scale;
}

float se::OusterLidar::nearDist(const Eigen::Vector3f&) const
{
    return near_plane;
}

float se::OusterLidar::farDist(const Eigen::Vector3f&) const
{
    return far_plane;
}

float se::OusterLidar::measurementFromPoint(const Eigen::Vector3f& point_C) const
{
    return point_C.norm();
}

bool se::OusterLidar::pointInFrustum(const Eigen::Vector3f& /*point_C*/) const
{
    // TODO Implement
    return false;
}

bool se::OusterLidar::pointInFrustumInf(const Eigen::Vector3f& /*point_C*/) const
{
    // TODO Implement
    return false;
}

bool se::OusterLidar::sphereInFrustum(const Eigen::Vector3f& /*center_C*/,
                                      const float /*radius*/) const
{
    // TODO Implement
    return false;
}

bool se::OusterLidar::sphereInFrustumInf(const Eigen::Vector3f& /*center_C*/,
                                         const float /*radius*/) const
{
    // TODO Implement
    return false;
}
