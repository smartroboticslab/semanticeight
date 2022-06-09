// SPDX-FileCopyrightText: 2022 Smart Robotics Lab
// SPDX-FileCopyrightText: 2022 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include <memory>

#include "se/filesystem.hpp"
#include "se/io/meshing_io.hpp"
#include "se/io/octree_io.hpp"
#include "se/object.hpp"
#include "se/point_cloud_utils.hpp"

/* Create an se::Object containing two perpendicular walls created by integrating synthetic depth
 * images from two camera poses (C1 and C2) as shown in the diagram below:
 *   y
 *   ^
 *   │     ║
 *   │C2   ║ in
 *  0┤ <   ╚═════
 *   │
 *- 1┤     v C1
 *   └─┬───┬─────> x
 *    -1   0
 */
class GainRaycastingTest : public ::testing::Test {
    public:
    GainRaycastingTest() :
            T_MW_((Eigen::Matrix4f() << 1,
                   0,
                   0,
                   map_dim_ / 2,
                   0,
                   1,
                   0,
                   map_dim_ / 2,
                   0,
                   0,
                   1,
                   map_dim_ / 2,
                   0,
                   0,
                   0,
                   1)
                      .finished()),
            T_WM_(se::math::to_inverse_transformation(T_MW_)),
            T_BC_(
                (Eigen::Matrix4f() << 0, 0, 1, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 1).finished()),
            T_CB_(se::math::to_inverse_transformation(T_BC_)),
            image_res_(640, 480),
            sensor_({image_res_.x(),
                     image_res_.y(),
                     false,
                     0.1f,
                     10.0f,
                     525.0f,
                     525.0f,
                     image_res_.x() / 2 - 0.5f,
                     image_res_.y() / 2 - 0.5f}),
            object_(std::make_unique<Object>(image_res_,
                                             Eigen::Vector3i::Constant(map_size_),
                                             Eigen::Vector3f::Constant(map_dim_),
                                             Eigen::Matrix4f::Identity(),
                                             Eigen::Matrix4f::Identity(),
                                             0)),
            tmp_(stdfs::temp_directory_path()
                 / stdfs::path("semanticeight_test_results/gain_raycasting_unittest"))
    {
        ObjVoxelImpl::configure(se::SemanticClass::default_res);
        se::semantic_classes = se::SemanticClasses::matterport3d_classes();
        se::semantic_classes.setEnabled("wall");

        stdfs::create_directories(tmp_);

        const Eigen::Matrix4f T_MC1 = T_MW_ * get_T_WB1(dist_) * T_BC_;
        const Eigen::Matrix4f T_MC2 = T_MW_ * get_T_WB2(dist_) * T_BC_;
        const se::Image<float> depth_1 = half_valid_image(image_res_, dist_, false);
        const se::Image<float> depth_2 = half_valid_image(image_res_, dist_, true);
        const se::Image<uint32_t> color_1 = half_valid_image(image_res_, 0xFFFFFFFF, false);
        const se::Image<uint32_t> color_2 = half_valid_image(image_res_, 0xFFFFFFFF, true);
        const cv::Mat mask(image_res_.y(), image_res_.x(), se::mask_t, cv::Scalar(UINT8_MAX));
        const se::InstanceSegmentation segmentation(
            object_->instance_id, se::semantic_classes.id("wall"), mask);
        for (int i = 0; i < 10; ++i) {
            object_->integrate(depth_1, color_1, segmentation, mask, T_MC1, sensor_, 2 * i);
            object_->integrate(depth_2, color_2, segmentation, mask, T_MC2, sensor_, 2 * i + 1);
        }

        // Save the generated mesh and TSDF slice.
        std::vector<se::Triangle> mesh;
        ObjVoxelImpl::dumpMesh(*(object_->map_), mesh, se::meshing::ScaleMode::Min);
        se::io::save_mesh_vtk(mesh, tmp_ + "/mesh_V.vtk");
        Eigen::Matrix4f T_WO = T_WM_;
        T_WO.topLeftCorner<3, 3>() *= object_->map_->voxelDim();
        se::io::save_mesh_ply(mesh, tmp_ + "/mesh_W.ply", T_WO);
        save_3d_value_slice_vtk(*(object_->map_),
                                tmp_ + "/slice_V.vtk",
                                Eigen::Vector3i(0, 0, map_size_ / 2),
                                Eigen::Vector3i(map_size_, map_size_, map_size_ / 2 + 1),
                                ObjVoxelImpl::VoxelType::selectNodeValue,
                                ObjVoxelImpl::VoxelType::selectVoxelValue);
    }

    protected:
    static constexpr int map_size_ = 512;
    static constexpr float map_dim_ = 5.0f;
    static constexpr float dist_ = 1.0f;
    const Eigen::Matrix4f T_MW_;
    const Eigen::Matrix4f T_WM_;
    const Eigen::Matrix4f T_BC_;
    const Eigen::Matrix4f T_CB_;
    const Eigen::Vector2i image_res_;
    const SensorImpl sensor_;
    std::unique_ptr<Object> object_;
    const std::string tmp_;



    private:
    static Eigen::Matrix4f get_T_WB1(float dist)
    {
        Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
        T.topLeftCorner<3, 3>() << 0, -1, 0, 1, 0, 0, 0, 0, 1;
        T.topRightCorner<3, 1>().y() = -dist;
        return T;
    }

    static Eigen::Matrix4f get_T_WB2(float dist)
    {
        Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
        T.topRightCorner<3, 1>().x() = -dist;
        return T;
    }

    /** Return an image whose left or right half is filled with value and the other half filled with
     * zeros. Set left to true to fill the left half with value.
     */
    template<typename T>
    static se::Image<T> half_valid_image(const Eigen::Vector2i& res, T value, bool left)
    {
        se::Image<T> image(res.x(), res.y(), T(0));
        for (int y = 0; y < image.height(); ++y) {
            for (int x = 0; x < image.width(); ++x) {
                if ((left && x < image.width() / 2) || (!left && x >= image.width() / 2)) {
                    image(x, y) = value;
                }
            }
        }
        return image;
    }
};

TEST_F(GainRaycastingTest, backFaceRaycasting)
{
    struct RayTestCase {
        Eigen::Vector3f origin_W;
        Eigen::Vector3f dir_W;
        bool back_hit;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    const std::array<RayTestCase, 4> ray_test_cases{
        {// No hit from outside.
         {Eigen::Vector3f(-dist_, -dist_, 0), Eigen::Vector3f::UnitX(), false},
         // No hit from inside.
         {Eigen::Vector3f(dist_ / 40, dist_ / 40, 0), Eigen::Vector3f::UnitX(), false},
         // Front-face hit.
         {Eigen::Vector3f(-dist_, dist_ / 2, 0), Eigen::Vector3f::UnitX(), false},
         // Back-face hit.
         {Eigen::Vector3f(dist_ / 2, dist_ / 2, 0), -Eigen::Vector3f::UnitX(), true}}};

    const Eigen::Matrix3f C_MW = T_MW_.topLeftCorner<3, 3>();
    const auto& map = *(object_->map_);
    for (size_t i = 0; i < ray_test_cases.size(); ++i) {
        const auto& ray = ray_test_cases[i];
        const Eigen::Vector3f ray_origin_M = (T_MW_ * ray.origin_W.homogeneous()).head<3>();
        const Eigen::Vector3f ray_dir_M = C_MW * ray.dir_W;
        // Assume the ray is at the center of the camera so nearDist()/farDist() are
        // near_plane/far_plane.
        const Eigen::Vector4f hit_M = ObjVoxelImpl::raycastBackFace(
            map, ray_origin_M, ray_dir_M, sensor_.near_plane, sensor_.far_plane);
        EXPECT_FLOAT_EQ(hit_M.w(), ray.back_hit ? 1.0f : 0.0f);

        // Save the rays.
        const Eigen::Matrix4f T =
            (Eigen::Matrix4f() << Eigen::Matrix3f::Identity(), ray.origin_W, 0, 0, 0, 1).finished();
        se::save_rays_ply({ray.dir_W}, tmp_ + "/ray_W_" + std::to_string(i) + ".ply", T);
    }
}
