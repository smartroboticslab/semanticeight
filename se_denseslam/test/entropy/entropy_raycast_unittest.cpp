// SPDX-FileCopyrightText: 2021 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include <se/filesystem.hpp>
#include <se/point_cloud_utils.hpp>

#include "../../src/entropy.cpp"


class EntropyRaycast : public ::testing::Test {
    protected:
    std::string tmp_;

    EntropyRaycast()
    {
        // Create a temporary directory for all the tests.
        tmp_ = stdfs::temp_directory_path() / stdfs::path("semanticeight_test_results");
        stdfs::create_directories(tmp_);
    }
};


TEST_F(EntropyRaycast, rayImage)
{
    const Eigen::VectorXf elevation_angles = (Eigen::VectorXf(64) << 17.744,
                                              17.12,
                                              16.536,
                                              15.982,
                                              15.53,
                                              14.936,
                                              14.373,
                                              13.823,
                                              13.373,
                                              12.786,
                                              12.23,
                                              11.687,
                                              11.241,
                                              10.67,
                                              10.132,
                                              9.574,
                                              9.138,
                                              8.577,
                                              8.023,
                                              7.479,
                                              7.046,
                                              6.481,
                                              5.944,
                                              5.395,
                                              4.963,
                                              4.401,
                                              3.859,
                                              3.319,
                                              2.871,
                                              2.324,
                                              1.783,
                                              1.238,
                                              0.786,
                                              0.245,
                                              -0.299,
                                              -0.849,
                                              -1.288,
                                              -1.841,
                                              -2.275,
                                              -2.926,
                                              -3.378,
                                              -3.91,
                                              -4.457,
                                              -5.004,
                                              -5.46,
                                              -6.002,
                                              -6.537,
                                              -7.096,
                                              -7.552,
                                              -8.09,
                                              -8.629,
                                              -9.196,
                                              -9.657,
                                              -10.183,
                                              -10.732,
                                              -11.289,
                                              -11.77,
                                              -12.297,
                                              -12.854,
                                              -13.415,
                                              -13.916,
                                              -14.442,
                                              -14.997,
                                              -15.595)
                                                 .finished();
    const Eigen::VectorXf azimuth_angles = (Eigen::VectorXf(64) << 3.102,
                                            3.0383750000000003,
                                            2.98175,
                                            2.950125,
                                            3.063,
                                            3.021375,
                                            3.00175,
                                            2.996125,
                                            3.045,
                                            3.031375,
                                            3.03375,
                                            3.043125,
                                            3.042,
                                            3.043375,
                                            3.05175,
                                            3.074125,
                                            3.03,
                                            3.051375,
                                            3.0797499999999998,
                                            3.101125,
                                            3.034,
                                            3.067375,
                                            3.09775,
                                            3.142125,
                                            3.048,
                                            3.093375,
                                            3.13475,
                                            3.170125,
                                            3.059,
                                            3.107375,
                                            3.15275,
                                            3.194125,
                                            3.085,
                                            3.136375,
                                            3.17675,
                                            3.217125,
                                            3.117,
                                            3.159375,
                                            3.15275,
                                            3.257125,
                                            3.149,
                                            3.189375,
                                            3.22975,
                                            3.270125,
                                            3.19,
                                            3.222375,
                                            3.26075,
                                            3.291125,
                                            3.23,
                                            3.253375,
                                            3.28775,
                                            3.301125,
                                            3.274,
                                            3.299375,
                                            3.31975,
                                            3.306125,
                                            3.327,
                                            3.3453749999999998,
                                            3.3377499999999998,
                                            3.322125,
                                            3.393,
                                            3.384375,
                                            3.35875,
                                            3.324125)
                                               .finished();
    const SensorImpl sensor({640,
                             480,
                             false,
                             0.02,
                             5.0,
                             205.46963709898583,
                             205.46963709898583,
                             320.5,
                             240.5,
                             azimuth_angles,
                             elevation_angles});

    constexpr int raycast_width = 72;
    constexpr int raycast_height = 20;
    const Eigen::Matrix4f T_MB = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f T_BC;
    T_BC << 0, 0, 1, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 1;

    // T_BC from Firefly.
    Eigen::Matrix4f T_BC_pitch;
    T_BC_pitch << 0, -0.09983341664682817, 0.9950041652780257, 0.1155739796873748, -1, 0, 0, 0.055,
        0, -0.9950041652780257, -0.09983341664682817, -0.02502997417539525, 0, 0, 0, 1;

    se::Image<Eigen::Vector3f> rays_M =
        se::ray_M_360_image(raycast_width, raycast_height, sensor, T_BC);
    se::Image<Eigen::Vector3f> rays_Mc =
        se::ray_M_360_image(raycast_width, raycast_height, sensor, T_BC_pitch);

    se::save_point_cloud_pcd(rays_M, tmp_ + "/rays_M.pcd", T_MB);
    se::save_point_cloud_pcd(rays_Mc, tmp_ + "/rays_M_corrected_pitch.pcd", T_MB);

    // pcl_viewer -ax 1 /tmp/semanticeight_test_results/rays_*.pcd
    std::cout << "pcl_viewer -ax 1 " + tmp_ + "/rays_*.pcd\n";
}
