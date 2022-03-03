/*
 * Copyright (C) 2019 Sotiris Papatheodorou
 */

// On Eigen before 3.3.6 GCC shows this warning:
// warning: argument 1 value ‘X’ exceeds maximum object size Y [-Walloc-size-larger-than=]

// TODO
// Find out why the tolerances are needed.
// Some tests fail, the error is probably in the octave test scripts, not here.
//   (octave sometimes gets an imaginary value for a)

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "se/bounding_volume.hpp"
#include "se/str_utils.hpp"
#include "se/utils/math_utils.h"


// Tolerance in pixels when comparing projected point coordinates.
#define TOLERANCE_PX 0.05
// Tolerance in degrees when comparing ellipse angles.
#define TOLERANCE_DEG 0.05


typedef std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> Vector3fVector;
typedef std::vector<se::BoundingSphere, Eigen::aligned_allocator<se::BoundingSphere>>
    BoundingSphereVector;



class BoundingSphereTest : public ::testing::Test {
    protected:
    int readTestResults(const std::string& filename,
                        se::PinholeCamera& camera,
                        Vector3fVector& centers_c,
                        std::vector<float>& radii,
                        std::vector<float>& a,
                        std::vector<float>& b,
                        std::vector<float>& x_c,
                        std::vector<float>& y_c,
                        std::vector<float>& theta,
                        std::vector<uint8_t>& valid_ellipse)
    {
        // Open file for reading.
        std::ifstream infile(filename.c_str(), std::ifstream::in);

        // Read file line by line.
        std::string line;
        while (true) {
            std::getline(infile, line);

            // EOF reached
            if (not infile.good()) {
                return 1;
            }

            // Ignore comment lines
            if ((line[0] == '#') and isalpha(line[2])) {
                continue;
            }

            // Read camera parameters.
            if ((line[0] == '#') and not isalpha(line[2])) {
                const std::vector<std::string> camera_data = str_utils::split_str(line, ' ');
                const size_t num_camera_cols = camera_data.size();
                constexpr size_t desired_camera_cols = 9;
                if (num_camera_cols != desired_camera_cols) {
                    std::cerr
                        << "Invalid test point file format. Expected " << desired_camera_cols
                        << " columns: "
                        << "width(px) height(px) fx(px) fy(px) cx(px) cy(px) near_plane(m) far_plane(m)"
                        << "\n but got " << num_camera_cols << ": " << line << std::endl;
                    return 1;
                }
                const int width = std::stoi(camera_data[1]);
                const int height = std::stoi(camera_data[2]);
                const float fx = std::stof(camera_data[3]);
                const float fy = std::stof(camera_data[4]);
                const float cx = std::stof(camera_data[5]);
                const float cy = std::stof(camera_data[6]);
                const float near_plane = std::stof(camera_data[7]);
                const float far_plane = std::stof(camera_data[8]);
                se::SensorConfig c = {width, height, false, near_plane, far_plane, fx, fy, cx, cy};
                camera = se::PinholeCamera(c);
                continue;
            }

            // Data line read, split on spaces
            const std::vector<std::string> line_data = str_utils::split_str(line, ' ');
            const size_t num_cols = line_data.size();
            constexpr size_t required_cols = 15;
            if (num_cols < required_cols) {
                std::cerr
                    << "Invalid test point file format. Expected " << required_cols << " columns: "
                    << "x_camera(m) y_camera(m) z_camera(m) x_image(px) y_image(px) valid_projection point_in_frustum sphere_radius(m) sphere_in_frustum a(px) b(px) x_c(px) y_c(px) theta(deg) valid_ellipse ..."
                    << "\n but got " << num_cols << ": " << line << std::endl;
                return 1;
            }

            // Read the point coordinates in the camera frame.
            const Eigen::Vector3f tmp_point_c(
                std::stof(line_data[0]), std::stof(line_data[1]), std::stof(line_data[2]));
            centers_c.push_back(tmp_point_c);

            // Read the sphere radius.
            radii.push_back(std::stof(line_data[7]));

            // Read the ellipse parameters.
            a.push_back(std::stof(line_data[9]));
            b.push_back(std::stof(line_data[10]));
            x_c.push_back(std::stof(line_data[11]));
            y_c.push_back(std::stof(line_data[12]));
            theta.push_back(std::stof(line_data[13]));

            // Read the valid ellipse test result.
            valid_ellipse.push_back(std::stoi(line_data[14]));
        }

        return 0;
    }



    void SetUp() override
    {
        const std::string test_data_directory = TEST_DATA_DIR;

        // Read the test points for the HD camera.
        readTestResults(test_data_directory + "/camera_hd.txt",
                        camera_hd,
                        centers_c_hd,
                        radii_hd,
                        a_hd,
                        b_hd,
                        x_c_hd,
                        y_c_hd,
                        theta_hd,
                        valid_ellipse_hd);
        // Initialize the bounding spheres for the HD camera.
        bounding_spheres_hd.resize(centers_c_hd.size());
        for (size_t i = 0; i < bounding_spheres_hd.size(); ++i) {
            bounding_spheres_hd[i] = se::BoundingSphere(centers_c_hd[i], radii_hd[i]);
        }

        // Read the test points for the ROS camera.
        readTestResults(test_data_directory + "/camera_ros.txt",
                        camera_ros,
                        centers_c_ros,
                        radii_ros,
                        a_ros,
                        b_ros,
                        x_c_ros,
                        y_c_ros,
                        theta_ros,
                        valid_ellipse_ros);
        // Initialize the bounding spheres for the ROS camera.
        bounding_spheres_ros.resize(centers_c_ros.size());
        for (size_t i = 0; i < bounding_spheres_ros.size(); ++i) {
            bounding_spheres_ros[i] = se::BoundingSphere(centers_c_ros[i], radii_ros[i]);
        }
    }



    const se::SensorConfig default_config =
        {640, 480, false, 0.1f, 4.0f, 525.0f, 525.0f, 319.5f, 239.5f};

    // HD camera.
    se::PinholeCamera camera_hd = se::PinholeCamera(default_config);
    Vector3fVector centers_c_hd;
    std::vector<float> radii_hd;
    std::vector<float> a_hd;
    std::vector<float> b_hd;
    std::vector<float> x_c_hd;
    std::vector<float> y_c_hd;
    std::vector<float> theta_hd;
    std::vector<uint8_t> valid_ellipse_hd;
    BoundingSphereVector bounding_spheres_hd;

    // ROS camera.
    se::PinholeCamera camera_ros = se::PinholeCamera(default_config);
    Vector3fVector centers_c_ros;
    std::vector<float> radii_ros;
    std::vector<float> a_ros;
    std::vector<float> b_ros;
    std::vector<float> x_c_ros;
    std::vector<float> y_c_ros;
    std::vector<float> theta_ros;
    std::vector<uint8_t> valid_ellipse_ros;
    BoundingSphereVector bounding_spheres_ros;
};



TEST_F(BoundingSphereTest, Constructor)
{
    // HD camera.
    for (size_t i = 0; i < bounding_spheres_hd.size(); ++i) {
        EXPECT_FLOAT_EQ(bounding_spheres_hd[i].center_.x(), centers_c_hd[i].x());
        EXPECT_FLOAT_EQ(bounding_spheres_hd[i].center_.y(), centers_c_hd[i].y());
        EXPECT_FLOAT_EQ(bounding_spheres_hd[i].center_.z(), centers_c_hd[i].z());
        EXPECT_FLOAT_EQ(bounding_spheres_hd[i].radius_, radii_hd[i]);
    }

    // ROS camera.
    for (size_t i = 0; i < bounding_spheres_ros.size(); ++i) {
        EXPECT_FLOAT_EQ(bounding_spheres_ros[i].center_.x(), centers_c_ros[i].x());
        EXPECT_FLOAT_EQ(bounding_spheres_ros[i].center_.y(), centers_c_ros[i].y());
        EXPECT_FLOAT_EQ(bounding_spheres_ros[i].center_.z(), centers_c_ros[i].z());
        EXPECT_FLOAT_EQ(bounding_spheres_ros[i].radius_, radii_ros[i]);
    }
}



TEST_F(BoundingSphereTest, computeProjection)
{
    // HD camera.
    for (size_t i = 0; i < bounding_spheres_hd.size(); ++i) {
        bounding_spheres_hd[i].computeProjection(camera_hd, Eigen::Matrix4f::Identity());

        EXPECT_EQ(bounding_spheres_hd[i].is_ellipse_, valid_ellipse_hd[i]);
        if (bounding_spheres_hd[i].is_ellipse_ and valid_ellipse_hd[i]) {
            EXPECT_NEAR(bounding_spheres_hd[i].a_, a_hd[i], TOLERANCE_PX);
            EXPECT_NEAR(bounding_spheres_hd[i].b_, b_hd[i], TOLERANCE_PX);
            EXPECT_NEAR(bounding_spheres_hd[i].x_c_, x_c_hd[i], TOLERANCE_PX);
            EXPECT_NEAR(bounding_spheres_hd[i].y_c_, y_c_hd[i], TOLERANCE_PX);
            EXPECT_NEAR(bounding_spheres_hd[i].theta_, theta_hd[i], TOLERANCE_DEG);
        }
    }

    // ROS camera.
    for (size_t i = 0; i < bounding_spheres_ros.size(); ++i) {
    }
}



TEST(BoundingSphere, contains)
{
    const Eigen::Vector3f zeros = Eigen::Vector3f::Zero();
    const Eigen::Vector3f ones = Eigen::Vector3f::Ones();
    // Point
    {
        const se::BoundingSphere bv(zeros, 0);
        EXPECT_TRUE(bv.contains(zeros));
        EXPECT_FALSE(bv.contains(ones));
    }
    // Sphere
    {
        const se::BoundingSphere bv(ones, 1);
        EXPECT_FALSE(bv.contains(zeros));
        EXPECT_TRUE(bv.contains(ones));
        EXPECT_TRUE(bv.contains(Eigen::Vector3f(1.1, 1.1, 0.8)));
    }
}



TEST(AABB, contains)
{
    const Eigen::Vector3f zeros = Eigen::Vector3f::Zero();
    const Eigen::Vector3f ones = Eigen::Vector3f::Ones();
    // Point
    {
        const se::AABB bv(zeros, zeros);
        EXPECT_TRUE(bv.contains(zeros));
        EXPECT_FALSE(bv.contains(ones));
    }
    // Sphere
    {
        const se::AABB bv(-ones, ones);
        EXPECT_TRUE(bv.contains(zeros));
        EXPECT_TRUE(bv.contains(ones));
        EXPECT_FALSE(bv.contains(Eigen::Vector3f(1.1, 1.1, 0.8)));
    }
}



// Test that cv::line() draws using out-of-bounds points.
TEST(BoundingVolumePrerequisites, line)
{
    constexpr int w = 64;
    constexpr int h = 64;
    const cv::Scalar colour(UINT8_MAX);
    cv::Mat image(cv::Size(w, h), CV_8UC1, cv::Scalar(0));
    // Horizontal line.
    cv::line(image, cv::Point2i(0, 0), cv::Point2i(w + 10, 0), colour);
    for (int x = 0; x < image.cols; ++x) {
        EXPECT_EQ(image.at<uint8_t>(0, x), UINT8_MAX);
    }
    // Vertical line.
    cv::line(image, cv::Point2i(0, -10), cv::Point2i(0, h + 10), colour);
    for (int y = 0; y < image.rows; ++y) {
        EXPECT_EQ(image.at<uint8_t>(y, 0), UINT8_MAX);
    }
    // Diagonal line.
    cv::line(image, cv::Point2i(-10, -10), cv::Point2i(w + 10, h + 10), colour);
    ASSERT_EQ(image.rows, image.cols);
    for (int x = 0; x < image.cols; ++x) {
        EXPECT_EQ(image.at<uint8_t>(x, x), UINT8_MAX);
    }
}



// Test that cv::fillConvexPoly() draws using out-of-bounds points.
TEST(BoundingVolumePrerequisites, fillConvexPoly)
{
    constexpr int w = 64;
    constexpr int h = 64;
    cv::Mat image(cv::Size(w, h), CV_8UC1, cv::Scalar(0));
    const std::vector<cv::Point2i> vertices{{-10, -10}, {w + 10, -10}, {w + 10, h + 10}};
    cv::fillConvexPoly(image, vertices, cv::Scalar(UINT8_MAX));

    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            const uint8_t desired = x >= y ? UINT8_MAX : 0;
            EXPECT_EQ(image.at<uint8_t>(y, x), desired);
        }
    }
}



TEST(AABB, project)
{
    if constexpr (false) {
        const se::PinholeCamera sensor(
            {640, 480, false, 0.001f, 10.0f, 325.0f, 325.0f, 319.5f, 239.5f});
        const int w = sensor.model.imageWidth();
        const int h = sensor.model.imageHeight();
        const cv::Scalar bg_color(0x00, 0x00, 0x00, 0xFF);
        const cv::Scalar aabb_color(0xFF, 0xFF, 0xFF, 0xFF);

        constexpr float dim_M = 1.0f;
        constexpr float inc = dim_M / 10.0f;
        se::AABB aabb_M(Eigen::Vector3f::Constant(-dim_M / 2.0f),
                        Eigen::Vector3f::Constant(dim_M / 2.0f));

        Eigen::Matrix4f T_MC = Eigen::Matrix4f::Identity();
        // Invalid projection -> smaller mask.
        T_MC.topRightCorner<3, 1>() = Eigen::Vector3f(0, 0, -dim_M / 2);
        // Vertices behind the camera -> correct mask.
        //T_MC.topRightCorner<3, 1>() = Eigen::Vector3f(0, 0, -dim_M / 2 + inc);
        // Vertices behind the camera -> bigger mask.
        //T_MC.topRightCorner<3, 1>() = Eigen::Vector3f(-dim_M / 2 - inc, 0, -dim_M / 2 + inc);
        // Vertices behind the camera -> wrong overlay.
        //T_MC.topRightCorner<3, 1>() = Eigen::Vector3f(-dim_M / 2 - inc, 0, -dim_M / 2 + 2 * inc);

        cv::namedWindow("test", cv::WINDOW_AUTOSIZE | cv::WINDOW_KEEPRATIO);
        bool quit = false;
        while (!quit) {
            std::cout << "T_MC\n" << T_MC << "\n";


            auto start_time = std::chrono::steady_clock::now();
            cv::Mat overlay(cv::Size(w, h), CV_8UC4, bg_color);
            aabb_M.overlay(reinterpret_cast<uint32_t*>(overlay.data),
                           Eigen::Vector2i(w, h),
                           T_MC,
                           sensor,
                           aabb_color,
                           1.0f);
            cv::Mat mask_binary = aabb_M.raycastingMask(Eigen::Vector2i(w, h), T_MC, sensor);
            double duration_ms = 1000.0f
                * std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time)
                      .count();
            std::cout << "Overlay: " << duration_ms << " ms\n";

            start_time = std::chrono::steady_clock::now();
            cv::Mat mask(cv::Size(w, h), CV_8UC4, bg_color);
            cv::cvtColor(mask_binary, mask, cv::COLOR_GRAY2RGBA);
            duration_ms = 1000.0f
                * std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time)
                      .count();
            std::cout << "Mask:    " << duration_ms << " ms\n";

            cv::Mat composite(cv::Size(w, 2 * h), CV_8UC4);
            cv::vconcat(overlay, mask, composite);
            cv::imshow("test", composite);
            switch (cv::waitKey(0)) {
            case 'q':
                quit = true;
                break;
            case 'w':
                T_MC.topRightCorner<3, 1>().z() += inc;
                break;
            case 's':
                T_MC.topRightCorner<3, 1>().z() -= inc;
                break;
            case 'd':
                T_MC.topRightCorner<3, 1>().x() += inc;
                break;
            case 'a':
                T_MC.topRightCorner<3, 1>().x() -= inc;
                break;
            case 'f':
                T_MC.topRightCorner<3, 1>().y() += inc;
                break;
            case 'r':
                T_MC.topRightCorner<3, 1>().y() -= inc;
                break;
            default:
                break;
            }
        }
    }
}
