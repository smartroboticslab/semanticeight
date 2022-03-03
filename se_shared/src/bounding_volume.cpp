// SPDX-FileCopyrightText: 2019-2020 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2019-2020 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#include "se/bounding_volume.hpp"

#include "se/utils/math_utils.h"



/**
 * Given a point [x y]^T on the plane, return a vector containing [x^2 x*y y^2
 * x y].
 */
inline Eigen::Matrix<float, 1, 5> point_to_conic_powers(const Eigen::Vector2f& point)
{
    Eigen::Matrix<float, 1, 5> tmp_vec;
    tmp_vec << point.x() * point.x(), point.x() * point.y(), point.y() * point.y(), point.x(),
        point.y();
    return tmp_vec;
}



se::BoundingSphere::BoundingSphere() :
        center_(Eigen::Vector3f::Zero()), radius_(0.f), is_ellipse_(0)
{
}



se::BoundingSphere::BoundingSphere(const Eigen::Vector3f& center, float radius) :
        center_(center), radius_(radius), is_ellipse_(0)
{
}



se::BoundingSphere::BoundingSphere(const Eigen::Vector3f& vertex_min,
                                   const Eigen::Vector3f& vertex_max) :
        is_ellipse_(0)
{
    // Set the center as the midpoint.
    center_ = (vertex_max + vertex_min) / 2.f;

    // Set the radius as half the maximum range.
    radius_ = (vertex_max - vertex_min).maxCoeff();
}



se::BoundingSphere::BoundingSphere(const se::Image<Eigen::Vector3f>& vertex_map,
                                   const Eigen::Matrix4f& T_vm,
                                   const cv::Mat& mask) :
        is_ellipse_(0)
{
    Eigen::Vector3f vertex_min;
    Eigen::Vector3f vertex_max;

    vertexMapMinMax(vertex_map, mask, T_vm, vertex_min, vertex_max);

    // Set the center as the midpoint.
    center_ = (vertex_max + vertex_min) / 2.f;

    // Set the radius as half the maximum range.
    radius_ = (vertex_max - vertex_min).maxCoeff();
}



bool se::BoundingSphere::contains(const Eigen::Vector3f& point_V) const
{
    return (center_ - point_V).norm() <= radius_;
}



bool se::BoundingSphere::isVisible(const Eigen::Matrix4f& T_VC,
                                   const se::PinholeCamera& camera) const
{
    // Convert the sphere center from world to camera coordinates.
    const Eigen::Vector3f center_c = (T_VC.inverse() * center_.homogeneous()).head<3>();

    return camera.sphereInFrustumInf(center_c, radius_);
}



void se::BoundingSphere::merge(const BoundingSphere& other)
{
    const Eigen::Vector3f ct = center_;
    const Eigen::Vector3f co = other.center_;
    const float rt = radius_;
    const float ro = other.radius_;
    const float center_dist = (co - ct).norm();

    if ((center_dist + ro <= rt) or (ro == 0.f)) {
        // This sphere contains the other sphere.
        // Nothing to do.
    }
    else if ((center_dist + rt <= ro) or (rt == 0.f)) {
        // The other sphere contains this sphere.
        radius_ = ro;
        center_ = co;
    }
    else {
        // Merge the spheres.
        radius_ = (rt + ro + center_dist) / 2.f;
        center_ = ct + (radius_ - rt) * (co - ct).normalized();
    }
}



void se::BoundingSphere::merge(const se::Image<Eigen::Vector3f>& vertex_map,
                               const Eigen::Matrix4f& T_vm,
                               const cv::Mat& mask)
{
    const se::BoundingSphere tmp_bounding_volume(vertex_map, T_vm, mask);
    merge(tmp_bounding_volume);
}



void se::BoundingSphere::computeProjection(const se::PinholeCamera& camera,
                                           const Eigen::Matrix4f& T_cs)
{
    // Transform the center to camera coordinates. center_c is also the direction
    // vector since the camera center is [0 0 0]^T.
    const Eigen::Vector3f center_c = (T_cs * center_.homogeneous()).head<3>();

    // Test whether the sphere is visible.
    if (not camera.sphereInFrustumInf(center_c, radius_)) {
        is_ellipse_ = 0;
        return;
    }

    // Compute the distance from the camera center to the sphere center.
    const float dist_to_sphere_center = center_c.norm();

    // If the distance from the camera center to the sphere center is smaller
    // than the sphere radius then the camera is inside the sphere and it should
    // not be rendered.
    if (dist_to_sphere_center >= radius_) {
        // Compute the unit direction vector from the camera center to the sphere
        // center.
        const Eigen::Vector3f dir_c = center_c.normalized();

        // The sphere has a tangent cone whose apex is at the camera center and axis
        // is along dir_c. The following angle is half the cone's aperture.
        const float half_aperture = asin(radius_ / dist_to_sphere_center);

        // The cone is tangent to the sphere at a circle. That circle outlines the
        // visible part of the sphere. That visible circle's radius is computed
        // below.
        const float visible_circle_radius = cos(half_aperture) * radius_;

        // Compute the distance from the camera center to the visible circle center.
        const float dist_to_visible_center = dist_to_sphere_center
            - sqrt(radius_ * radius_ - visible_circle_radius * visible_circle_radius);

        // Compute the visible circle center.
        const Eigen::Vector3f visible_circle_center = dist_to_visible_center * dir_c;

        // Compute the intersection point of the camera z axis with the visible
        // circle plane. First compute the negative plane constant coefficient
        // neg_d.
        const float neg_d = visible_circle_center.dot(dir_c);
        const Eigen::Vector3f ray_plane_intersection(0.f, 0.f, neg_d / dir_c.z());

        // Compute the transformation from the visible circle to the camera frame
        // by computing the three basis vectors.
        // z is in the direction of the ray from the camera center through the
        // visible circle's center.
        const Eigen::Vector3f basis_z = dir_c;
        // x is arbitrarily set from the visible circle's center towards the
        // intersection of the visible circle plane and the camera's z axis.
        const Eigen::Vector3f basis_x =
            (ray_plane_intersection - visible_circle_center).normalized();
        // y is the cross product of the other two to get a right handed cordinate
        // system.
        const Eigen::Vector3f basis_y = basis_z.cross(basis_x);
        // Compute the homogeneous transformation matrix from the basis vectors and
        // translation vector.
        Eigen::Matrix4f T_CV = Eigen::Matrix4f::Identity();
        T_CV.block<3, 1>(0, 0) = basis_x;
        T_CV.block<3, 1>(0, 1) = basis_y;
        T_CV.block<3, 1>(0, 2) = basis_z;
        T_CV.block<3, 1>(0, 3) = visible_circle_center;

        // Generate 5 distinct points on the circumference of the visible circle.
        // First construct them on the visible circle frame xy plane and then
        // transform them to the camera frame. Use the circle parametric equation
        // but evaluate the cos() and sin() functions offline for speed.
        // 0 degrees.
        const Eigen::Vector3f p0 =
            (T_CV * Eigen::Vector4f(visible_circle_radius, 0.f, 0.f, 1.f)).head<3>();
        // 90 degrees.
        const Eigen::Vector3f p1 =
            (T_CV * Eigen::Vector4f(0.f, visible_circle_radius, 0.f, 1.f)).head<3>();
        // 180 degrees.
        const Eigen::Vector3f p2 =
            (T_CV * Eigen::Vector4f(-visible_circle_radius, 0.f, 0.f, 1.f)).head<3>();
        // 270 degrees.
        const Eigen::Vector3f p3 =
            (T_CV * Eigen::Vector4f(0.f, -visible_circle_radius, 0.f, 1.f)).head<3>();
        // 45 degrees.
        const Eigen::Vector3f p4 = (T_CV
                                    * Eigen::Vector4f(0.707106781186548f * visible_circle_radius,
                                                      0.707106781186548f * visible_circle_radius,
                                                      0.f,
                                                      1.f))
                                       .head<3>();

        // Project the 5 points on the image plane.
        Eigen::Vector2f p0_px = Eigen::Vector2f::Zero();
        Eigen::Vector2f p1_px = Eigen::Vector2f::Zero();
        Eigen::Vector2f p2_px = Eigen::Vector2f::Zero();
        Eigen::Vector2f p3_px = Eigen::Vector2f::Zero();
        Eigen::Vector2f p4_px = Eigen::Vector2f::Zero();
        camera.model.project(p0, &p0_px);
        camera.model.project(p1, &p1_px);
        camera.model.project(p2, &p2_px);
        camera.model.project(p3, &p3_px);
        camera.model.project(p4, &p4_px);

        // Compute the ellipse parameters from the 5 projected points. The general
        // ellipse equation is Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0.
        // F can be selected arbitrarily without loss of generality. Select a large
        // value so that the other parameters' values are not too small.
        const float F = camera.model.imageWidth() * camera.model.imageHeight();

        // Create the matrices for the system Hw = h, where w are the unknown
        // general ellipse coefficients.
        // Each line of the H matrix is in the form: [x^2 x*y y^2 x y]
        Eigen::Matrix<float, 5, 5> H;
        H.row(0) = point_to_conic_powers(p0_px);
        H.row(1) = point_to_conic_powers(p1_px);
        H.row(2) = point_to_conic_powers(p2_px);
        H.row(3) = point_to_conic_powers(p3_px);
        H.row(4) = point_to_conic_powers(p4_px);
        const Eigen::Matrix<float, 5, 1> h = Eigen::Matrix<float, 5, 1>::Constant(-F);
        // Solve the linear system.
        const Eigen::Matrix<float, 5, 1> ellipse_coefficients = H.colPivHouseholderQr().solve(h);
        const float A = ellipse_coefficients[0];
        const float B = ellipse_coefficients[1];
        const float C = ellipse_coefficients[2];
        const float D = ellipse_coefficients[3];
        const float E = ellipse_coefficients[4];

        // Compute the canonical ellipse parameters.
        const float tmp1 = B * B - 4 * A * C;
        const float tmp2 = sqrt((A - C) * (A - C) + B * B);
        // Ellipse axes.
        a_ = -sqrt(2 * (A * E * E + C * D * D - B * D * E + F * tmp1) * (A + C + tmp2)) / tmp1;
        b_ = -sqrt(2 * (A * E * E + C * D * D - B * D * E + F * tmp1) * (A + C - tmp2)) / tmp1;
        // Ellipse center.
        x_c_ = (2 * C * D - B * E) / tmp1;
        y_c_ = (2 * A * E - B * D) / tmp1;
        // Ellipse angle in degrees.
        if (B == 0) {
            if (A < C) {
                theta_ = 0;
            }
            else {
                theta_ = 90.f;
            }
        }
        else {
            theta_ = 180.f / 3.1415926f * atan((C - A - tmp2) / B);
        }

        // Test if the parameters are nan or if the shape is not an ellipse.
        const float delta =
            (A * C - B * B / 4.f) * F + B * E * D / 4.f - C * D * D / 4.f - A * E * E / 4.f;
        if (isnan(a_) or isnan(b_) or isnan(C * delta) or (C * delta >= 0.f)) {
            is_ellipse_ = 2;
        }
        else {
            is_ellipse_ = 1;
        }
    }
    else {
        is_ellipse_ = 2;
    }
}



cv::Mat se::BoundingSphere::raycastingMask(const Eigen::Vector2i& mask_size,
                                           const Eigen::Matrix4f& T_VC,
                                           const se::PinholeCamera& camera)
{
    computeProjection(camera, T_VC.inverse());

    // Initialize the destination Mat.
    cv::Mat raycasting_mask = cv::Mat::zeros(cv::Size(mask_size.x(), mask_size.y()), se::mask_t);

    if (is_ellipse_ == 1) {
        // Draw a filled ellipse on the mask.
        cv::ellipse(
            raycasting_mask, cv::Point(x_c_, y_c_), cv::Size(a_, b_), theta_, 0, 360, 255, -1);
    }
    else if (is_ellipse_ == 2) {
        // The camera is inside the sphere.
        raycasting_mask.setTo(255);
    }
    return raycasting_mask;
}



void se::BoundingSphere::overlay(uint32_t* out,
                                 const Eigen::Vector2i& output_size,
                                 const Eigen::Matrix4f& T_VC,
                                 const se::PinholeCamera& camera,
                                 const cv::Scalar& colour,
                                 const float) const
{
    const Eigen::Matrix4f T_CV = T_VC.inverse();

    // TODO: Use opacity parameter

    // Create a cv::Mat header for the output image so that OpenCV drawing
    // functions can be used. No data is copied or deallocated by OpenCV so this
    // operation is fast.
    cv::Mat out_mat(cv::Size(output_size.x(), output_size.y()), CV_8UC4, out);

    // Transform the center to camera coordinates. center_c is also the direction
    // vector since the camera center is [0 0 0]^T.
    const Eigen::Vector3f center_c = (T_CV * center_.homogeneous()).head<3>();

    // Project the sphere center on the image plane.
    Eigen::Vector2f center_px = Eigen::Vector2f::Zero();
    camera.model.project(center_c, &center_px);

    // Draw the projected sphere center as a small circle.
    constexpr int point_radius_px = 2;
    cv::circle(out_mat, cv::Point(center_px.x(), center_px.y()), point_radius_px, colour, -1);

    if (is_ellipse_ == 1) {
        // Draw the ellipse outline.
        cv::ellipse(out_mat, cv::Point(x_c_, y_c_), cv::Size(a_, b_), theta_, 0, 360, colour);
    }
}



std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>
se::BoundingSphere::edges() const
{
    return std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>();
}



se::AABB::AABB() :
        min_(Eigen::Vector3f::Constant(INFINITY)), max_(Eigen::Vector3f::Constant(-INFINITY))
{
    updateVertices();
}



se::AABB::AABB(const Eigen::Vector3f& min, const Eigen::Vector3f& max) : min_(min), max_(max)
{
    updateVertices();
}



se::AABB::AABB(const se::Image<Eigen::Vector3f>& vertex_map,
               const Eigen::Matrix4f& T_vm,
               const cv::Mat& mask)
{
    vertexMapMinMax(vertex_map, mask, T_vm, min_, max_);
    updateVertices();
}



bool se::AABB::contains(const Eigen::Vector3f& point_V) const
{
    return (min_.array() <= point_V.array()).all() && (point_V.array() <= max_.array()).all();
}



bool se::AABB::isVisible(const Eigen::Matrix4f& T_VC, const se::PinholeCamera& camera) const
{
    // Convert the AABB vertices from the bounding volume to the camera frame.
    const Eigen::Matrix4f T_CV = se::math::to_inverse_transformation(T_VC);
    Eigen::Matrix<float, 3, 8> vertices_C;
    for (int i = 0; i < vertices_.cols(); ++i) {
        vertices_C.col(i) = (T_CV * vertices_.col(i).homogeneous()).head<3>();
    }
    return camera.aabbInFrustum(vertices_C);
}



void se::AABB::merge(const AABB& other)
{
    // Min.
    if (other.min_.x() < min_.x()) {
        min_.x() = other.min_.x();
    }
    if (other.min_.y() < min_.y()) {
        min_.y() = other.min_.y();
    }
    if (other.min_.z() < min_.z()) {
        min_.z() = other.min_.z();
    }
    // Max.
    if (other.max_.x() > max_.x()) {
        max_.x() = other.max_.x();
    }
    if (other.max_.y() > max_.y()) {
        max_.y() = other.max_.y();
    }
    if (other.max_.z() > max_.z()) {
        max_.z() = other.max_.z();
    }

    updateVertices();
}



void se::AABB::merge(const se::Image<Eigen::Vector3f>& vertex_map,
                     const Eigen::Matrix4f& T_vm,
                     const cv::Mat& mask)
{
    const se::AABB tmp_bounding_volume(vertex_map, T_vm, mask);
    merge(tmp_bounding_volume);
    updateVertices();
}



cv::Mat se::AABB::raycastingMask(const Eigen::Vector2i& mask_size,
                                 const Eigen::Matrix4f& T_VC,
                                 const se::PinholeCamera& camera)
{
    // Return early if the AABB isn't visible.
    if (!isVisible(T_VC, camera)) {
        return cv::Mat::zeros(cv::Size(mask_size.x(), mask_size.y()), se::mask_t);
    }
    // Project the AABB vertices on the image plane.
    std::vector<cv::Point2f> projected_vertices = projectAABB(T_VC, camera);
    // Return a full mask if some vertices can't be projected.
    auto result = std::find_if(projected_vertices.begin(),
                               projected_vertices.end(),
                               [](const auto& p) { return isnan(p.x) || isnan(p.y); });
    if (result != projected_vertices.end()) {
        return cv::Mat(cv::Size(mask_size.x(), mask_size.y()), se::mask_t, cv::Scalar(255));
    }
    // Compute the convex hull of the projected vertices.
    std::vector<cv::Point2f> convex_hull_vertices;
    cv::convexHull(projected_vertices, convex_hull_vertices);
    // Fill the convex hull on the mask. The convex hull points need to be converted from float to
    // int coordinates.
    std::vector<cv::Point2i> convex_hull_vertices_i(convex_hull_vertices.size());
    std::transform(convex_hull_vertices.begin(),
                   convex_hull_vertices.end(),
                   convex_hull_vertices_i.begin(),
                   [](const auto& p) { return cv::Point2i(p.x + 0.5f, p.y + 0.5f); });
    cv::Mat raycasting_mask = cv::Mat::zeros(cv::Size(mask_size.x(), mask_size.y()), se::mask_t);
    cv::fillConvexPoly(raycasting_mask, convex_hull_vertices_i, 255);
    return raycasting_mask;
}



void se::AABB::overlay(uint32_t* out,
                       const Eigen::Vector2i& output_size,
                       const Eigen::Matrix4f& T_VC,
                       const se::PinholeCamera& camera,
                       const cv::Scalar& colour,
                       const float) const
{
    // TODO: Use opacity parameter
    // TODO: Erroneous edges when vertices behind the camera project inside the image.
    // TODO: Erroneous edges when vertices behind the camera project in an x or y coordinate (but
    // not both) inside the image.

    // Return early if the AABB isn't visible.
    if (!isVisible(T_VC, camera)) {
        return;
    }

    // Project the AABB vertices on the image plane.
    std::vector<cv::Point2f> projected_vertices = projectAABB(T_VC, camera);

    // Create a cv::Mat header for the output image so that OpenCV drawing
    // functions can be used. No data is copied or deallocated by OpenCV so this
    // operation is fast.
    cv::Mat out_mat(cv::Size(output_size.x(), output_size.y()), CV_8UC4, out);

    std::vector<std::pair<int, int>> vertex_pairs = {
        // Bottom box edges.
        {0, 1},
        {1, 3},
        {3, 2},
        {2, 0},
        // Top box edges.
        {4, 5},
        {5, 7},
        {7, 6},
        {6, 4},
        // Vertical box edges.
        {0, 4},
        {1, 5},
        {2, 6},
        {3, 7},
    };

    for (const auto& v : vertex_pairs) {
        const auto& v1 = projected_vertices[v.first];
        const auto& v2 = projected_vertices[v.second];
        if (!isnan(v1.x) && !isnan(v1.y) && !isnan(v2.x) && !isnan(v2.y)) {
            cv::line(out_mat, v1, v2, colour);
        }
    }
}



std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> se::AABB::edges() const
{
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> e;
    e.reserve(24);
    // Add the bottom AABB edges.
    e.push_back(vertices_.col(0));
    e.push_back(vertices_.col(1));
    e.push_back(vertices_.col(1));
    e.push_back(vertices_.col(3));
    e.push_back(vertices_.col(3));
    e.push_back(vertices_.col(2));
    e.push_back(vertices_.col(2));
    e.push_back(vertices_.col(0));
    // Add the top AABB edges.
    e.push_back(vertices_.col(4));
    e.push_back(vertices_.col(5));
    e.push_back(vertices_.col(5));
    e.push_back(vertices_.col(7));
    e.push_back(vertices_.col(7));
    e.push_back(vertices_.col(6));
    e.push_back(vertices_.col(6));
    e.push_back(vertices_.col(4));
    // Add the vertical AABB edges.
    e.push_back(vertices_.col(0));
    e.push_back(vertices_.col(4));
    e.push_back(vertices_.col(1));
    e.push_back(vertices_.col(5));
    e.push_back(vertices_.col(2));
    e.push_back(vertices_.col(6));
    e.push_back(vertices_.col(3));
    e.push_back(vertices_.col(7));
    return e;
}



void se::AABB::updateVertices()
{
    // Recompute the AABB vertices.
    vertices_.col(0) = Eigen::Vector3f(min_.x(), min_.y(), min_.z());
    vertices_.col(1) = Eigen::Vector3f(max_.x(), min_.y(), min_.z());
    vertices_.col(2) = Eigen::Vector3f(min_.x(), max_.y(), min_.z());
    vertices_.col(3) = Eigen::Vector3f(max_.x(), max_.y(), min_.z());
    vertices_.col(4) = Eigen::Vector3f(min_.x(), min_.y(), max_.z());
    vertices_.col(5) = Eigen::Vector3f(max_.x(), min_.y(), max_.z());
    vertices_.col(6) = Eigen::Vector3f(min_.x(), max_.y(), max_.z());
    vertices_.col(7) = Eigen::Vector3f(max_.x(), max_.y(), max_.z());
}



std::vector<cv::Point2f> se::AABB::projectAABB(const Eigen::Matrix4f& T_VC,
                                               const se::PinholeCamera& camera) const
{
    std::vector<cv::Point2f> projected_vertices(8, cv::Point2f(NAN, NAN));
    const Eigen::Matrix4f T_CV = se::math::to_inverse_transformation(T_VC);
    for (size_t i = 0; i < projected_vertices.size(); ++i) {
        Eigen::Vector3f vertex_C = (T_CV * vertices_.col(i).homogeneous()).head<3>();
        // We still need to draw lines with vertices behind the camera, project them as if they
        // are in front.
        vertex_C.z() = std::fabs(vertex_C.z());
        // Project the vertex on the image.
        Eigen::Vector2f pixel;
        switch (camera.model.project(vertex_C, &pixel)) {
        // Relying on the fact that srl::PinholeCamera::project() modifies pixel before testing
        // whether it's outside the image, masked or behing the camera. We keep the
        // out-of-bounds coordinates to allow creating the correct convex hull and drawing the
        // correct lines.
        case srl::projection::ProjectionStatus::Behind:
        case srl::projection::ProjectionStatus::Masked:
        case srl::projection::ProjectionStatus::OutsideImage:
        case srl::projection::ProjectionStatus::Successful:
            projected_vertices[i] = cv::Point2f(pixel.x(), pixel.y());
            break;
        case srl::projection::ProjectionStatus::Invalid:
        default:
            // Keep NANs for invalid projections.
            break;
        }
    }
    return projected_vertices;
}



// Utility functions //////////////////////////////////////////////////////////
int se::vertexMapStats(const se::Image<Eigen::Vector3f>& point_cloud_C,
                       const cv::Mat& mask,
                       const Eigen::Matrix4f& T_MC,
                       Eigen::Vector3f& vertex_min,
                       Eigen::Vector3f& vertex_max,
                       Eigen::Vector3f& vertex_mean)
{
    // Initialize min, max and mean vertex elements.
    vertex_min = Eigen::Vector3f::Constant(INFINITY);
    vertex_max = Eigen::Vector3f::Constant(-INFINITY);
    vertex_mean = Eigen::Vector3f::Zero();
    int count = 0;

    //TODO: parallelize
    for (int pixely = 0; pixely < point_cloud_C.height(); ++pixely) {
        for (int pixelx = 0; pixelx < point_cloud_C.width(); ++pixelx) {
            if (mask.at<se::mask_elem_t>(pixely, pixelx) != 0) {
                const int pixel_ind = pixelx + pixely * point_cloud_C.width();

                // Skip vertices whose coordinates are all zero as they are invalid.
                if (point_cloud_C[pixel_ind].isApproxToConstant(0.f)) {
                    continue;
                }

                const Eigen::Vector4f vertex = T_MC * point_cloud_C[pixel_ind].homogeneous();

                if (vertex.x() > vertex_max.x())
                    vertex_max.x() = vertex.x();
                if (vertex.x() < vertex_min.x())
                    vertex_min.x() = vertex.x();
                if (vertex.y() > vertex_max.y())
                    vertex_max.y() = vertex.y();
                if (vertex.y() < vertex_min.y())
                    vertex_min.y() = vertex.y();
                if (vertex.z() > vertex_max.z())
                    vertex_max.z() = vertex.z();
                if (vertex.z() < vertex_min.z())
                    vertex_min.z() = vertex.z();

                vertex_mean.x() += vertex.x();
                vertex_mean.y() += vertex.y();
                vertex_mean.z() += vertex.z();
                count++;
            }
        }
    }
    // Is the average needed for the center or will the midpoint do?
    vertex_mean.x() /= count;
    vertex_mean.y() /= count;
    vertex_mean.z() /= count;

    return count;
}



void se::vertexMapMinMax(const se::Image<Eigen::Vector3f>& point_cloud_C,
                         const cv::Mat& mask,
                         const Eigen::Matrix4f& T_MC,
                         Eigen::Vector3f& vertex_min,
                         Eigen::Vector3f& vertex_max)
{
    // Initialize min and max vertex elements.
    vertex_min = Eigen::Vector3f::Constant(INFINITY);
    vertex_max = Eigen::Vector3f::Constant(-INFINITY);

    //TODO: parallelize
    for (int pixely = 0; pixely < point_cloud_C.height(); ++pixely) {
        for (int pixelx = 0; pixelx < point_cloud_C.width(); ++pixelx) {
            if (mask.at<se::mask_elem_t>(pixely, pixelx) != 0) {
                const int pixel_ind = pixelx + pixely * point_cloud_C.width();

                // Skip vertices whose coordinates are all zero as they are invalid.
                if (point_cloud_C[pixel_ind].isApproxToConstant(0.f)) {
                    continue;
                }

                const Eigen::Vector4f vertex = T_MC * point_cloud_C[pixel_ind].homogeneous();

                if (vertex.x() > vertex_max.x())
                    vertex_max.x() = vertex.x();
                if (vertex.x() < vertex_min.x())
                    vertex_min.x() = vertex.x();
                if (vertex.y() > vertex_max.y())
                    vertex_max.y() = vertex.y();
                if (vertex.y() < vertex_min.y())
                    vertex_min.y() = vertex.y();
                if (vertex.z() > vertex_max.z())
                    vertex_max.z() = vertex.z();
                if (vertex.z() < vertex_min.z())
                    vertex_min.z() = vertex.z();
            }
        }
    }
}
