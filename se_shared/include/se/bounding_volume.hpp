// SPDX-FileCopyrightText: 2019-2020 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2019-2020 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#ifndef __BOUNDING_VOLUME_HPP
#define __BOUNDING_VOLUME_HPP

#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <opencv2/opencv.hpp>

#include "se/image/image.hpp"
#include "se/segmentation.hpp"
#include "se/semanticeight_definitions.hpp"
#include "se/sensor.hpp"



namespace se {

class BoundingVolume {
    public:
    virtual ~BoundingVolume()
    {
    }

    virtual bool contains(const Eigen::Vector3f& point_V) const = 0;

    virtual bool isVisible(const Eigen::Matrix4f& T_VC, const PinholeCamera& camera) const = 0;

    virtual void merge(const se::Image<Eigen::Vector3f>& vertex_map,
                       const Eigen::Matrix4f& T_vm,
                       const cv::Mat& mask) = 0;

    virtual cv::Mat raycastingMask(const Eigen::Vector2i& mask_size,
                                   const Eigen::Matrix4f& T_VC,
                                   const PinholeCamera& camera) = 0;

    virtual void overlay(uint32_t* out,
                         const Eigen::Vector2i& output_size,
                         const Eigen::Matrix4f& T_VC,
                         const PinholeCamera& camera,
                         const cv::Scalar& colour,
                         const float opacity) const = 0;

    virtual std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>
    edges() const = 0;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};



class BoundingSphere : public BoundingVolume {
    public:
    Eigen::Vector3f center_;
    float radius_;

    // Ellipse parameters.
    float a_;
    float b_;
    float x_c_;
    float y_c_;
    float theta_;
    uint8_t is_ellipse_;

    BoundingSphere();

    BoundingSphere(const Eigen::Vector3f& center, float radius = 0.f);

    BoundingSphere(const Eigen::Vector3f& vertex_min, const Eigen::Vector3f& vertex_max);

    BoundingSphere(const se::Image<Eigen::Vector3f>& vertex_map,
                   const Eigen::Matrix4f& T_vm,
                   const cv::Mat& mask);

    bool contains(const Eigen::Vector3f& point_V) const;

    bool isVisible(const Eigen::Matrix4f& T_VC, const PinholeCamera& camera) const;

    void merge(const BoundingSphere& other);

    void merge(const se::Image<Eigen::Vector3f>& vertex_map,
               const Eigen::Matrix4f& T_vm,
               const cv::Mat& mask);

    /**
       * The projection of the bounding sphere on an image is an ellipse. This
       * function computes the parameters of this ellipse given the camera
       * model and the transformation from bounding sphere to camera frame.
       */
    void computeProjection(const PinholeCamera& camera, const Eigen::Matrix4f& T_cs);

    cv::Mat raycastingMask(const Eigen::Vector2i& mask_size,
                           const Eigen::Matrix4f& T_VC,
                           const PinholeCamera& camera);

    void overlay(uint32_t* out,
                 const Eigen::Vector2i& output_size,
                 const Eigen::Matrix4f& T_VC,
                 const PinholeCamera& camera,
                 const cv::Scalar& colour,
                 const float opacity) const;

    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> edges() const;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};



class AABB : public BoundingVolume {
    public:
    Eigen::Vector3f min_;
    Eigen::Vector3f max_;

    AABB();

    AABB(const Eigen::Vector3f& min, const Eigen::Vector3f& max);

    AABB(const se::Image<Eigen::Vector3f>& vertex_map,
         const Eigen::Matrix4f& T_vm,
         const cv::Mat& mask);

    bool contains(const Eigen::Vector3f& point_V) const;

    bool isVisible(const Eigen::Matrix4f& T_VC, const PinholeCamera& camera) const;

    void merge(const AABB& other);

    void merge(const se::Image<Eigen::Vector3f>& vertex_map,
               const Eigen::Matrix4f& T_vm,
               const cv::Mat& mask);

    cv::Mat raycastingMask(const Eigen::Vector2i& mask_size,
                           const Eigen::Matrix4f& T_VC,
                           const PinholeCamera& camera);

    void overlay(uint32_t* out,
                 const Eigen::Vector2i& output_size,
                 const Eigen::Matrix4f& T_VC,
                 const PinholeCamera& camera,
                 const cv::Scalar& colour,
                 const float opacity) const;

    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> edges() const;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    private:
    Eigen::Matrix<float, 3, 8> vertices_;

    void updateVertices();

    /** Points corresponding the vertices that couldn't be projected contain NANs.
     */
    std::vector<cv::Point2f> projectAABB(const Eigen::Matrix4f& T_VC,
                                         const PinholeCamera& camera) const;
};



/**
   * Given a vertex map and a binary mask, compute the minimum, maximum and
   * mean vertex coordinates after transforming them to the Map frame using
   * T_MC. Returns the number of vertices processed.
   */
int vertexMapStats(const se::Image<Eigen::Vector3f>& point_cloud_C,
                   const cv::Mat& mask,
                   const Eigen::Matrix4f& T_MC,
                   Eigen::Vector3f& vertex_min,
                   Eigen::Vector3f& vertex_max,
                   Eigen::Vector3f& vertex_mean);



/**
   * Given a vertex map and a binary mask, compute the minimum and maximum
   * vertex coordinates after transforming them to the Map frame using T_MC.
   */
void vertexMapMinMax(const se::Image<Eigen::Vector3f>& point_cloud_C,
                     const cv::Mat& mask,
                     const Eigen::Matrix4f& T_MC,
                     Eigen::Vector3f& vertex_min,
                     Eigen::Vector3f& vertex_max);

} // namespace se

#endif
