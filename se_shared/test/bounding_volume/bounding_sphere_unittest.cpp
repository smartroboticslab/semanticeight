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
