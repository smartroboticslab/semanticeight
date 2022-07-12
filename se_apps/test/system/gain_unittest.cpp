// SPDX-FileCopyrightText: 2022 Smart Robotics Lab
// SPDX-FileCopyrightText: 2022 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include "reader_raw.hpp"
#include "se/DenseSLAMSystem.h"
#include "se/completion.hpp"
#include "se/dist.hpp"
#include "se/exploration_planner.hpp"
#include "se/filesystem.hpp"
#include "se/point_cloud_utils.hpp"

#ifndef SEQUENCE_PATH
#    define SEQUENCE_PATH "."
#endif

TEST(System, gainRaycasting)
{
    se::Configuration config;
    config.sequence_type = "raw";
    config.sequence_name = "experiment"; // Disable depth noise.
    config.sequence_path = SEQUENCE_PATH "/scene.raw";
    config.ground_truth_file = SEQUENCE_PATH "/association.txt";
    config.map_size = Eigen::Vector3i::Constant(256);
    config.map_dim = Eigen::Vector3f::Constant(10.24f);
    config.sensor_intrinsics = Eigen::Vector4f(525.0f, 525.0f, 319.5f, 239.5f);
    config.sensor_downsampling_factor = 4;
    config.near_plane = 0.4f;
    config.far_plane = 4.0f;
    config.T_BC << 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    config.raycast_width *= 20;
    config.raycast_height *= 20;

    // Initialize the reader.
    const se::ReaderConfig reader_config{se::string_to_reader_type(config.sequence_type),
                                         config.sequence_path,
                                         config.ground_truth_file,
                                         0.0f,
                                         config.fps,
                                         config.drop_frames,
                                         0};
    std::unique_ptr<se::Reader> reader(new se::RAWReader(reader_config));
    ASSERT_TRUE(reader);
    ASSERT_TRUE(reader->good());
    const Eigen::Vector2i input_res = reader->depthImageRes();
    ASSERT_EQ(input_res, Eigen::Vector2i(640, 480));
    ASSERT_EQ(input_res, reader->RGBAImageRes());
    EXPECT_EQ(reader->numFrames(), 2u);

    // Initialize the pipeline.
    se::semantic_classes = se::SemanticClasses::coco_classes();
    se::semantic_classes.setEnabled("book");
    se::semantic_classes.setRes("book", 0.01f);
    const Eigen::Vector2i image_res = input_res / config.sensor_downsampling_factor;
    const SensorImpl sensor(
        {image_res.x(),
         image_res.y(),
         config.left_hand_frame,
         config.near_plane,
         config.far_plane,
         config.sensor_intrinsics[0] / config.sensor_downsampling_factor,
         config.sensor_intrinsics[1] / config.sensor_downsampling_factor,
         ((config.sensor_intrinsics[2] + 0.5f) / config.sensor_downsampling_factor - 0.5f),
         ((config.sensor_intrinsics[3] + 0.5f) / config.sensor_downsampling_factor - 0.5f),
         (Eigen::VectorXf(1) << 0.0f).finished(),
         (Eigen::VectorXf(1) << 0.0f).finished()});
    std::unique_ptr<DenseSLAMSystem> pipeline(
        new DenseSLAMSystem(image_res,
                            config.map_size,
                            config.map_dim,
                            Eigen::Vector3f(config.t_MW_factor.cwiseProduct(config.map_dim)),
                            config.pyramid,
                            config));
    ASSERT_TRUE(pipeline);

    // Read the data for a single frame.
    int frame = 0;
    // In the candidate code it is assumed that the map frame (M) is z-up and since the TUM dataset
    // contains camera poses we read T_WC even though the function argument is named T_WB.
    Eigen::Matrix4f T_WC;
    se::Image<float> input_depth(input_res.x(), input_res.y());
    se::Image<uint32_t> input_rgba(input_res.x(), input_res.y());
    se::SegmentationResult input_segmentation(input_res.x(), input_res.y());
    while (reader->nextData(input_depth, input_rgba, T_WC, input_segmentation)
           == se::ReaderStatus::ok) {
        // Process and integrate the frame.
        pipeline->preprocessDepth(input_depth.data(), input_res, config.bilateral_filter);
        pipeline->preprocessColor(input_rgba.data(), input_res);
        pipeline->preprocessSegmentation(input_segmentation);
        pipeline->setT_WC(T_WC);
        pipeline->trackObjects(sensor, frame);
        pipeline->integrate(sensor, frame);
        pipeline->integrateObjects(sensor, frame);
        pipeline->raycastObjectsAndBg(sensor, frame);
        const auto& objects = pipeline->getObjectMaps();
        EXPECT_EQ(objects.size(), 1u);

        // Candidate pose.
        Eigen::Matrix4f candidate_T_WB = Eigen::Matrix4f::Identity();
        candidate_T_WB.topRightCorner<3, 1>() = Eigen::Vector3f(0, 0, 1);
        const Eigen::Matrix4f candidate_T_MB = pipeline->T_MW() * candidate_T_WB;

        // Initialize stuff required for sampling candidates.
        se::CandidateConfig candidate_config(config);
        candidate_config.planner_config.start_t_MB_ = candidate_T_MB.topRightCorner<3, 1>();
        candidate_config.planner_config.goal_t_MB_ = candidate_config.planner_config.start_t_MB_;
        se::PoseMaskHistory T_MB_history(
            Eigen::Vector2i(candidate_config.raycast_width, candidate_config.raycast_height),
            sensor,
            config.T_BC,
            Eigen::Vector3f::Constant(pipeline->getMap()->dim()));
        ompl::msg::setLogLevel(ompl::msg::LOG_ERROR);
        ptp::SafeFlightCorridorGenerator planner(pipeline->getMap(),
                                                 candidate_config.planner_config);
        const auto frontiers = pipeline->getFrontiers();

        // Sample candidate.
        se::CandidateView candidate(*pipeline->getMap(),
                                    planner,
                                    std::vector<se::key_t>(frontiers.begin(), frontiers.end()),
                                    objects,
                                    sensor,
                                    candidate_T_MB,
                                    config.T_BC,
                                    &T_MB_history,
                                    candidate_config);
        ASSERT_TRUE(candidate.isValid());
        EXPECT_EQ(candidate.status_, "Same start/goal positions");
        EXPECT_EQ(candidate.path().size(), 2u);
        const Eigen::Vector3f goal_t_MB = candidate.goalT_MB().topRightCorner<3, 1>();
        EXPECT_TRUE(goal_t_MB.isApprox(candidate_T_MB.topRightCorner<3, 1>()));

        // Create renders.
        se::Image<uint32_t> volume_render(image_res.x(), image_res.y());
        pipeline->renderObjects(volume_render.data(), image_res, sensor, RenderMode::Scale, false);
        const se::Image<uint32_t> entropy_render = candidate.renderEntropy(false);
        const se::Image<uint32_t> depth_render = candidate.renderDepth(false);
        const se::Image<uint32_t> object_dist_render = candidate.renderObjectDistGain(false);

        // Save renders.
        const std::string tmp(stdfs::temp_directory_path()
                              / stdfs::path("semanticeight_test_results/gain_unittest"));
        stdfs::create_directories(tmp);
        std::stringstream suffix;
        suffix << std::setw(5) << std::setfill('0') << frame << ".png";
        lodepng_encode32_file((tmp + "/volume_" + suffix.str()).c_str(),
                              reinterpret_cast<const unsigned char*>(volume_render.data()),
                              volume_render.width(),
                              volume_render.height());
        lodepng_encode32_file((tmp + "/entropy_" + suffix.str()).c_str(),
                              reinterpret_cast<const unsigned char*>(entropy_render.data()),
                              entropy_render.width(),
                              entropy_render.height());
        lodepng_encode32_file((tmp + "/depth_" + suffix.str()).c_str(),
                              reinterpret_cast<const unsigned char*>(depth_render.data()),
                              depth_render.width(),
                              depth_render.height());
        lodepng_encode32_file((tmp + "/object_dist_" + suffix.str()).c_str(),
                              reinterpret_cast<const unsigned char*>(object_dist_render.data()),
                              object_dist_render.width(),
                              object_dist_render.height());

        // Create more renders
        Eigen::Matrix4f T_CCr;
        T_CCr << -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 2.0f, 0, 0, 0, 1;
        Eigen::Matrix4f T_MB =
            pipeline->T_MC() * T_CCr * se::math::to_inverse_transformation(config.T_BC);
        se::Image<Eigen::Vector3f> rays_M = se::ray_M_image(sensor, T_MB * config.T_BC);
        const se::Image<Eigen::Vector3f> no_hits_M(
            rays_M.width(), rays_M.height(), Eigen::Vector3f::Constant(NAN));

        const se::Image<float> compl_gain =
            se::object_completion_gain(rays_M, no_hits_M, objects, sensor, T_MB, config.T_BC);
        const se::Image<uint32_t> compl_gain_render =
            se::visualize_entropy(compl_gain, 0, 0, false);
        lodepng_encode32_file((tmp + "/compl_gain_" + suffix.str()).c_str(),
                              reinterpret_cast<const unsigned char*>(compl_gain_render.data()),
                              compl_gain_render.width(),
                              compl_gain_render.height());

        const se::Image<float> bg_gain = se::bg_dist_gain(
            candidate.entropy_hits_M_, *pipeline->getMap(), sensor, T_MB, config.T_BC);
        const se::Image<uint32_t> bg_gain_render = se::visualize_entropy(bg_gain, 0, 0, false);
        lodepng_encode32_file((tmp + "/bg_gain_" + suffix.str()).c_str(),
                              reinterpret_cast<const unsigned char*>(bg_gain_render.data()),
                              bg_gain_render.width(),
                              bg_gain_render.height());


        pipeline->saveMesh(tmp + "/mesh.ply");
        se::save_point_cloud_ply(rays_M, tmp + "/rays.ply", Eigen::Matrix4f::Identity());

        frame++;
    }
}
