/*
 * Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 * Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1
 *
 * This code is licensed under the MIT License.
 */

#include "default_parameters.h"
#include "draw.hpp"
#include "montage.hpp"
#include "reader.hpp"
#include "se/DenseSLAMSystem.h"
#include "se/dist.hpp"
#include "se/exploration_planner.hpp"
#include "se/perfstats.h"
#include "se/system_info.hpp"



void storeStats(const size_t frame,
                const Eigen::Vector3f& t_WC,
                const bool tracked,
                const bool integrated)
{
    se::perfstats.sample("frame", frame, PerfStats::FRAME);
    se::perfstats.sample("t_WC.x", t_WC.x(), PerfStats::POSITION);
    se::perfstats.sample("t_WC.y", t_WC.y(), PerfStats::POSITION);
    se::perfstats.sample("t_WC.z", t_WC.z(), PerfStats::POSITION);
    se::perfstats.sample("RAM", se::ram_usage_self() / 1024.0 / 1024.0, PerfStats::MEMORY);
    se::perfstats.sample("tracked", tracked, PerfStats::BOOL);
    se::perfstats.sample("integrated", integrated, PerfStats::BOOL);
}



int main(int argc, char** argv)
{
    try {
        se::Configuration config = parseArgs(argc, argv);

        const se::ReaderConfig reader_config{se::string_to_reader_type(config.sequence_type),
                                             config.sequence_path,
                                             config.ground_truth_file,
                                             0.0f,
                                             config.fps,
                                             config.drop_frames,
                                             0};
        std::unique_ptr<se::Reader> reader(se::create_reader(reader_config));
        if (!reader || !reader->good()) {
            std::cerr << "No valid input file specified\n";
            return EXIT_FAILURE;
        }
        const size_t frame_offset = reader->frame();

        if (config.max_frame == -1
            || (reader->numFrames() != 0
                && config.max_frame > static_cast<long int>(reader->numFrames()) - 1)) {
            config.max_frame = reader->numFrames() - 1;
        }

        // Construct image buffers
        const Eigen::Vector2i input_image_res = reader->depthImageRes();
        const Eigen::Vector2i image_res = input_image_res / config.sensor_downsampling_factor;
        se::Image<float> input_depth_image(input_image_res.x(), input_image_res.y());
        se::Image<uint32_t> input_rgba_image(input_image_res.x(), input_image_res.y());
        se::SegmentationResult input_segmentation(input_image_res.x(), input_image_res.y());
        se::Image<uint32_t> rgba_render(image_res.x(), image_res.y());
        se::Image<uint32_t> depth_render(image_res.x(), image_res.y());
        se::Image<uint32_t> track_render(image_res.x(), image_res.y());
        se::Image<uint32_t> volume_render(image_res.x(), image_res.y());
        se::Image<uint32_t> volume_render_color(image_res.x(), image_res.y());
        se::Image<uint32_t> volume_render_scale(image_res.x(), image_res.y());
        se::Image<uint32_t> volume_render_min_scale(image_res.x(), image_res.y());
        se::Image<uint32_t> class_render(image_res.x(), image_res.y());
        se::Image<uint32_t> instance_render(image_res.x(), image_res.y());
        se::Image<uint32_t> raycast_render(image_res.x(), image_res.y());
        se::Image<uint32_t> segmentation_render(image_res.x(), image_res.y());
        se::Image<uint32_t> volume_aabb_render(image_res.x(), image_res.y());

        se::Image<float> entropy_gain(image_res.x(), image_res.y());
        se::Image<float> obj_dist_gain(image_res.x(), image_res.y());
        se::Image<float> bg_dist_gain(image_res.x(), image_res.y());
        se::Image<Eigen::Vector3f> entropy_hits_M(image_res.x(), image_res.y());
        se::Image<uint32_t> entropy_gain_render(image_res.x(), image_res.y());
        se::Image<uint32_t> obj_dist_gain_render(image_res.x(), image_res.y());
        se::Image<uint32_t> bg_dist_gain_render(image_res.x(), image_res.y());

        // Setup semantic classes
        se::semantic_classes = se::SemanticClasses::coco_classes();
        se::semantic_classes.setEnabled("book");

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
             azimuth_angles,
             elevation_angles});

        const Eigen::Vector3f t_MW = config.t_MW_factor.cwiseProduct(config.map_dim);
        std::unique_ptr<DenseSLAMSystem> pipeline(
            new DenseSLAMSystem(image_res,
                                Eigen::Vector3i::Constant(config.map_size.x()),
                                Eigen::Vector3f::Constant(config.map_dim.x()),
                                t_MW,
                                config.pyramid,
                                config,
                                config.voxel_impl_yaml));

        size_t num_planning_iterations = 0u;
        std::unique_ptr<se::ExplorationPlanner> planner(
            new se::ExplorationPlanner(*pipeline, sensor, config));

        // Update the initial pose
        if (!config.ground_truth_file.empty()) {
            Eigen::Matrix4f init_T_WB;
            if (reader->getPose(init_T_WB, 0) != se::ReaderStatus::ok) {
                std::cerr << "Couldn't read initial pose\n";
                return EXIT_FAILURE;
            }
            config.init_T_WB = init_T_WB;
        }
        pipeline->setInitT_WC(config.init_T_WB * config.T_BC);
        pipeline->setT_WC(config.init_T_WB * config.T_BC);
        planner->recordT_WB(config.init_T_WB, se::Image<float>(image_res.x(), image_res.y(), 1.0f));

        // Setup logging and stats
        std::ostream* log_stream = &std::cout;
        std::ofstream log_file_stream;
        generate_log_file(config);
        if (config.log_path != "") {
            log_file_stream.open(config.log_path.c_str());
            log_stream = &log_file_stream;
        }
        log_stream->setf(std::ios::fixed, std::ios::floatfield);
        *log_stream << config;
        *log_stream << reader.get();
        *log_stream << VoxelImpl::printConfig() << std::endl;
        se::perfstats.includeDetailed(true);
        se::perfstats.setFilestream(&log_file_stream);

        const bool create_renders = config.enable_render && config.rendering_rate != 0;
        const bool save_renders = create_renders && !config.output_render_file.empty();
        const std::string render_prefix = config.output_render_file + "/";
        std::string render_suffix;

        Eigen::Matrix4f T_WB;
        const bool track = !config.enable_ground_truth;
        while (true) {
            TICK("TOTAL");
            se::perfstats.setIter(reader->frame());
            const size_t frame = reader->frame() - frame_offset;
#if SE_VERBOSE >= SE_VERBOSE_MINIMAL
            std::cout << "Frame " << frame << " ----------------------------------------\n";
#endif
            const bool render_frame = create_renders && frame % config.rendering_rate == 0;
            if (save_renders && render_frame) {
                stdfs::create_directories(config.output_render_file);
                std::stringstream path_suffix_ss;
                path_suffix_ss << std::setw(5) << std::setfill('0') << frame << ".png";
                render_suffix = path_suffix_ss.str();
            }

            TICK("COMPUTATION")
            const std::chrono::time_point<std::chrono::steady_clock> now =
                std::chrono::steady_clock::now();
            std::vector<std::chrono::time_point<std::chrono::steady_clock>> timings(7, now);

            // Read frames and ground truth data if set
            TICK("ACQUISITION")
            const se::ReaderStatus read_status = config.enable_ground_truth
                ? reader->nextData(input_depth_image, input_rgba_image, T_WB, input_segmentation)
                : reader->nextData(input_depth_image, input_rgba_image);
            TOCK("ACQUISITION")
            if (read_status == se::ReaderStatus::skip) {
                // Skip this frame
                TOCK("COMPUTATION")
                TOCK("TOTAL")
                continue;
            }
            else if (read_status == se::ReaderStatus::eof
                     || read_status == se::ReaderStatus::error) {
                // Finish processing if the next frame could not be read
                timings[0] = std::chrono::steady_clock::now();
                TOCK("COMPUTATION")
                TOCK("TOTAL")
                break;
            }

            if (config.max_frame != -1 && frame > static_cast<size_t>(config.max_frame)) {
                // Reached the frame limit
                timings[0] = std::chrono::steady_clock::now();
                TOCK("COMPUTATION")
                TOCK("TOTAL")
                break;
            }

            if (save_renders && render_frame) {
                const Eigen::Matrix4f T_MB = pipeline->T_MW() * T_WB;
                se::raycast_entropy(
                    entropy_gain, entropy_hits_M, *(pipeline->getMap()), sensor, T_MB, config.T_BC);
                obj_dist_gain = se::object_dist_gain(
                    entropy_hits_M, pipeline->getObjectMaps(), sensor, T_MB, config.T_BC);
                bg_dist_gain = se::bg_dist_gain(
                    entropy_hits_M, *(pipeline->getMap()), sensor, T_MB, config.T_BC);
                // TODO SEM compute weighted gain

                entropy_gain_render = se::visualize_entropy(entropy_gain, 0, 0, false);
                obj_dist_gain_render = se::visualize_entropy(obj_dist_gain, 0, 0, false);
                bg_dist_gain_render = se::visualize_entropy(bg_dist_gain, 0, 0, false);
                lodepng_encode32_file((render_prefix + "entropy_pre_" + render_suffix).c_str(),
                                      reinterpret_cast<unsigned char*>(entropy_gain_render.data()),
                                      image_res.x(),
                                      image_res.y());
                lodepng_encode32_file((render_prefix + "obj_dist_pre_" + render_suffix).c_str(),
                                      reinterpret_cast<unsigned char*>(obj_dist_gain_render.data()),
                                      image_res.x(),
                                      image_res.y());
                lodepng_encode32_file((render_prefix + "bg_dist_pre_" + render_suffix).c_str(),
                                      reinterpret_cast<unsigned char*>(bg_dist_gain_render.data()),
                                      image_res.x(),
                                      image_res.y());
            }

            TICK("PREPROCESSING")
            pipeline->preprocessDepth(
                input_depth_image.data(), input_image_res, config.bilateral_filter);
            pipeline->preprocessColor(input_rgba_image.data(), input_image_res);
            if (config.enable_ground_truth) {
                pipeline->preprocessSegmentation(input_segmentation);
            }
            TOCK("PREPROCESSING")

            bool tracked = false;
            if (track) {
                // No ground truth used, call track every tracking_rate frames.
                if (frame % config.tracking_rate == 0) {
                    tracked = pipeline->track(sensor, config.icp_threshold);
                }
            }
            else {
                // Set the pose to the ground truth.
                pipeline->setT_WC(T_WB * config.T_BC);
                planner->recordT_WB(T_WB, pipeline->getDepth());
                tracked = true;
            }
            // Call object tracking.
            pipeline->trackObjects(sensor, frame);

            bool integrated = false;
            // Integrate only if tracking was successful every integration_rate frames
            // or it is one of the first 4 frames.
            if ((tracked && (frame % config.integration_rate == 0)) || frame <= 3) {
                integrated =
                    pipeline->integrate(sensor, frame) && pipeline->integrateObjects(sensor, frame);
            }

            storeStats(frame, pipeline->t_WC(), tracked, integrated);

#if SE_VERBOSE >= SE_VERBOSE_MINIMAL
            printf("Free volume:     %10.3f m³\n", pipeline->free_volume);
            printf("Occupied volume: %10.3f m³\n", pipeline->occupied_volume);
            printf("Explored volume: %10.3f m³\n", pipeline->explored_volume);
#endif

            // Planning TMP
            if (planner->goalReached() || num_planning_iterations == 0) {
                std::cout << "Planning " << num_planning_iterations << "\n";
                const se::Path path_WB = planner->computeNextPath_WB(
                    pipeline->getFrontiers(), pipeline->getObjectMaps(), T_WB);
                num_planning_iterations++;
            }

            if (render_frame) {
                TICK("RENDERING")
                pipeline->renderObjectClasses(class_render.data(), image_res);
                pipeline->renderObjectInstances(instance_render.data(), image_res);
                pipeline->raycastObjectsAndBg(sensor, frame);
                pipeline->renderRGBA(rgba_render.data(), image_res);
                pipeline->renderDepth(depth_render.data(), image_res, sensor);
                pipeline->renderTrack(track_render.data(), image_res);
                pipeline->renderObjects(
                    volume_render.data(), image_res, sensor, RenderMode::InstanceID, false);
                pipeline->renderObjects(
                    volume_render_color.data(), image_res, sensor, RenderMode::Color, false);
                pipeline->renderObjects(
                    volume_render_scale.data(), image_res, sensor, RenderMode::Scale, false);
                pipeline->renderObjects(
                    volume_render_min_scale.data(), image_res, sensor, RenderMode::MinScale, false);
                pipeline->renderRaycast(raycast_render.data(), image_res);
                TOCK("RENDERING")
            }
            TOCK("COMPUTATION")

            if (save_renders && render_frame) {
                pipeline->renderInputSegmentation(segmentation_render.data(), image_res);
                pipeline->renderObjects(
                    volume_aabb_render.data(), image_res, sensor, RenderMode::InstanceID);

                lodepng_encode32_file((render_prefix + "rgba_" + render_suffix).c_str(),
                                      reinterpret_cast<unsigned char*>(rgba_render.data()),
                                      image_res.x(),
                                      image_res.y());
                lodepng_encode32_file((render_prefix + "depth_" + render_suffix).c_str(),
                                      reinterpret_cast<unsigned char*>(depth_render.data()),
                                      image_res.x(),
                                      image_res.y());
                lodepng_encode32_file((render_prefix + "segm_" + render_suffix).c_str(),
                                      reinterpret_cast<unsigned char*>(segmentation_render.data()),
                                      image_res.x(),
                                      image_res.y());
                lodepng_encode32_file((render_prefix + "volume_" + render_suffix).c_str(),
                                      reinterpret_cast<unsigned char*>(volume_render.data()),
                                      image_res.x(),
                                      image_res.y());
                lodepng_encode32_file((render_prefix + "volume_color_" + render_suffix).c_str(),
                                      reinterpret_cast<unsigned char*>(volume_render_color.data()),
                                      image_res.x(),
                                      image_res.y());
                lodepng_encode32_file((render_prefix + "volume_scale_" + render_suffix).c_str(),
                                      reinterpret_cast<unsigned char*>(volume_render_scale.data()),
                                      image_res.x(),
                                      image_res.y());
                lodepng_encode32_file(
                    (render_prefix + "volume_min_scale_" + render_suffix).c_str(),
                    reinterpret_cast<unsigned char*>(volume_render_min_scale.data()),
                    image_res.x(),
                    image_res.y());
                lodepng_encode32_file((render_prefix + "volume_aabb_" + render_suffix).c_str(),
                                      reinterpret_cast<unsigned char*>(volume_aabb_render.data()),
                                      image_res.x(),
                                      image_res.y());
                lodepng_encode32_file((render_prefix + "raycast_" + render_suffix).c_str(),
                                      reinterpret_cast<unsigned char*>(raycast_render.data()),
                                      image_res.x(),
                                      image_res.y());
                lodepng_encode32_file((render_prefix + "instance_" + render_suffix).c_str(),
                                      reinterpret_cast<unsigned char*>(instance_render.data()),
                                      image_res.x(),
                                      image_res.y());
                lodepng_encode32_file((render_prefix + "class_" + render_suffix).c_str(),
                                      reinterpret_cast<unsigned char*>(class_render.data()),
                                      image_res.x(),
                                      image_res.y());

                const Eigen::Matrix4f T_MB = pipeline->T_MW() * T_WB;
                se::raycast_entropy(
                    entropy_gain, entropy_hits_M, *(pipeline->getMap()), sensor, T_MB, config.T_BC);
                obj_dist_gain = se::object_dist_gain(
                    entropy_hits_M, pipeline->getObjectMaps(), sensor, T_MB, config.T_BC);
                bg_dist_gain = se::bg_dist_gain(
                    entropy_hits_M, *(pipeline->getMap()), sensor, T_MB, config.T_BC);
                // TODO SEM compute weighted gain

                entropy_gain_render = se::visualize_entropy(entropy_gain, 0, 0, false);
                obj_dist_gain_render = se::visualize_entropy(obj_dist_gain, 0, 0, false);
                bg_dist_gain_render = se::visualize_entropy(bg_dist_gain, 0, 0, false);
                lodepng_encode32_file((render_prefix + "entropy_post_" + render_suffix).c_str(),
                                      reinterpret_cast<unsigned char*>(entropy_gain_render.data()),
                                      image_res.x(),
                                      image_res.y());
                lodepng_encode32_file((render_prefix + "obj_dist_post_" + render_suffix).c_str(),
                                      reinterpret_cast<unsigned char*>(obj_dist_gain_render.data()),
                                      image_res.x(),
                                      image_res.y());
                lodepng_encode32_file((render_prefix + "bg_dist_post_" + render_suffix).c_str(),
                                      reinterpret_cast<unsigned char*>(bg_dist_gain_render.data()),
                                      image_res.x(),
                                      image_res.y());
            }

            // Save meshes
            const bool mesh_volume = frame == static_cast<size_t>(config.max_frame)
                || (config.meshing_rate != 0 && frame % config.meshing_rate == 0);
            if (mesh_volume && config.output_mesh_file != "") {
                stdfs::create_directories(config.output_mesh_file);
                std::stringstream output_mesh_meter_file_ss;
                output_mesh_meter_file_ss << config.output_mesh_file << "/mesh_" << std::setw(5)
                                          << std::setfill('0') << frame << ".ply";
                pipeline->saveMesh(output_mesh_meter_file_ss.str());
                std::stringstream output_mesh_object_file_ss;
                output_mesh_object_file_ss << config.output_mesh_file << "/mesh_" << std::setw(5)
                                           << std::setfill('0') << frame << "_object";
                pipeline->saveObjectMeshes(output_mesh_object_file_ss.str());
            }

            // Save the octree structure and slices
            const bool save_structure =
                config.enable_structure && config.output_structure_file != "";
            if (save_structure && (mesh_volume || frame == static_cast<size_t>(config.max_frame))) {
                std::stringstream output_structure_file_ss;
                output_structure_file_ss << config.output_structure_file << "_frame_"
                                         << std::setw(4) << std::setfill('0') << frame;
                pipeline->saveStructure(output_structure_file_ss.str().c_str());
            }

            const bool add_structure_stats = true;
            if constexpr (add_structure_stats) {
                size_t num_nodes = 0u;
                size_t num_blocks = 0u;
                std::vector<size_t> num_blocks_per_scale(VoxelImpl::VoxelBlockType::max_scale + 1,
                                                         0);
                pipeline->structureStats(num_nodes, num_blocks, num_blocks_per_scale);
                se::perfstats.sample("num_nodes", num_nodes, PerfStats::COUNT);
                se::perfstats.sample("num_blocks", num_blocks, PerfStats::COUNT);
                se::perfstats.sample("num_blocks_s0", num_blocks_per_scale[0], PerfStats::COUNT);
                se::perfstats.sample("num_blocks_s1", num_blocks_per_scale[1], PerfStats::COUNT);
                se::perfstats.sample("num_blocks_s2", num_blocks_per_scale[2], PerfStats::COUNT);
                se::perfstats.sample("num_blocks_s3", num_blocks_per_scale[3], PerfStats::COUNT);
            }

            if (config.enable_benchmark || !config.enable_render) {
                if (config.log_path != "") {
                    se::perfstats.writeToFilestream();
                }
                else {
                    se::perfstats.writeToOStream(*log_stream);
                }
            }

#ifdef SE_GLUT
            // Create vectors of images and labels.
            const cv::Size res(image_res.x(), image_res.y());
            std::vector<cv::Mat> images;
            std::vector<std::string> labels;
            labels.emplace_back("Input RGB");
            images.emplace_back(res, CV_8UC4, rgba_render.data());
            labels.emplace_back("Input depth");
            images.emplace_back(res, CV_8UC4, depth_render.data());
            labels.emplace_back("Object instances");
            images.emplace_back(res, CV_8UC4, instance_render.data());
            labels.emplace_back("Object raycasting");
            images.emplace_back(res, CV_8UC4, raycast_render.data());
            labels.emplace_back("Objects");
            images.emplace_back(res, CV_8UC4, volume_render_color.data());
            labels.emplace_back("Object instances");
            images.emplace_back(res, CV_8UC4, volume_render.data());
            // Combine all the images into one, overlay the labels and show it.
            const cv::Mat render = se::montage(2, 3, images, labels);
            drawit(reinterpret_cast<uint32_t*>(render.data),
                   Eigen::Vector2i(render.cols, render.rows));
#endif
            TOCK("TOTAL")
        }
        if (!config.enable_benchmark) {
            std::cout << "{";
            se::perfstats.writeSummaryToOStream(std::cout, false);
            std::cout << "}\n";
        }
        return EXIT_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}

