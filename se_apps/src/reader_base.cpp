/*
 * SPDX-FileCopyrightText: 2014 University of Edinburgh, Imperial College London, University of Manchester
 * SPDX-FileCopyrightText: 2016-2019 Emanuele Vespa
 * SPDX-FileCopyrightText: 2020-2022 Smart Robotics Lab, Imperial College London, Technical University of Munich
 * SPDX-FileCopyrightText: 2020-2022 Nils Funk
 * SPDX-FileCopyrightText: 2020-2022 Sotiris Papatheodorou
 * SPDX-License-Identifier: MIT
 */

#include "reader_base.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <sstream>
#include <thread>

#include "se/filesystem.hpp"
#include "se/str_utils.hpp"



se::ReaderType se::string_to_reader_type(const std::string& s)
{
    std::string s_lowered(s);
    str_utils::to_lower(s_lowered);
    if (s_lowered == "openni") {
        return se::ReaderType::OPENNI;
    }
    else if (s_lowered == "raw") {
        return se::ReaderType::RAW;
    }
    else if (s_lowered == "newercollege") {
        return se::ReaderType::NEWERCOLLEGE;
    }
    else if (s_lowered == "tum") {
        return se::ReaderType::TUM;
    }
    else if (s_lowered == "interiornet") {
        return se::ReaderType::INTERIORNET;
    }
    else {
        return se::ReaderType::UNKNOWN;
    }
}



std::string se::reader_type_to_string(se::ReaderType t)
{
    if (t == se::ReaderType::OPENNI) {
        return "OpenNI";
    }
    else if (t == se::ReaderType::RAW) {
        return "raw";
    }
    else if (t == se::ReaderType::NEWERCOLLEGE) {
        return "NewerCollege";
    }
    else if (t == se::ReaderType::TUM) {
        return "TUM";
    }
    else if (t == se::ReaderType::INTERIORNET) {
        return "InteriorNet";
    }
    else {
        return "unknown";
    }
}



std::ostream& se::operator<<(std::ostream& os, const se::ReaderConfig& c)
{
    os << str_utils::str_to_pretty_str(se::reader_type_to_string(c.reader_type), "reader_type")
       << "\n";
    os << str_utils::str_to_pretty_str(c.sequence_path, "sequence_path") << "\n";
    os << str_utils::str_to_pretty_str(c.ground_truth_file, "ground_truth_file") << "\n";
    os << str_utils::value_to_pretty_str(c.fps, "fps") << "\n";
    os << str_utils::bool_to_pretty_str(c.drop_frames, "drop_frames") << "\n";
    os << str_utils::value_to_pretty_str(c.verbose, "verbose") << "\n";
    return os;
}



std::ostream& se::operator<<(std::ostream& os, const ReaderStatus& s)
{
    switch (s) {
    case ReaderStatus::ok:
        os << "OK";
        break;
    case ReaderStatus::skip:
        os << "skip";
        break;
    case ReaderStatus::eof:
        os << "EOF";
        break;
    case ReaderStatus::error:
        os << "error";
        break;
    default:
        os << "unknown status";
    }
    return os;
}



se::Reader::Reader(const se::ReaderConfig& c) :
        sequence_path_(c.sequence_path),
        ground_truth_file_(c.ground_truth_file),
        depth_image_res_(1, 1),
        rgba_image_res_(1, 1),
        fps_(c.fps),
        spf_(1.0 / c.fps),
        drop_frames_(c.drop_frames),
        verbose_(c.verbose),
        is_live_reader_(false),
        status_(se::ReaderStatus::ok),
        frame_(SIZE_MAX),
        num_frames_(0),
        ground_truth_frame_(SIZE_MAX),
        ground_truth_delimiter_(' '),
        segmentation_frame_(SIZE_MAX)
{
    // Trim trailing slashes from sequence_path_
    sequence_path_.erase(sequence_path_.find_last_not_of("/") + 1);
    // Open the ground truth file if supplied
    if (!ground_truth_file_.empty()) {
        ground_truth_fs_.open(ground_truth_file_, std::ios::in);
        if (!ground_truth_fs_.good()) {
            std::cerr << "Error: Could not read ground truth file " << ground_truth_file_ << "\n";
            status_ = se::ReaderStatus::error;
        }
        if (str_utils::ends_with(ground_truth_file_, ".csv")) {
            ground_truth_delimiter_ = ',';
        }
        segmentation_fs_.open(ground_truth_file_, std::ios::in);
        if (!segmentation_fs_.good()) {
            std::cerr << "Error: Could not read segmentation index file " << ground_truth_file_
                      << "\n";
            status_ = se::ReaderStatus::error;
        }
        segmentation_base_dir_ = str_utils::dirname(ground_truth_file_) + "/segmentation";
    }
    // Ensure the available clock has enough accuracy to measure the requested
    // inter-frame time intervals. Compare the clock tick interval with the
    // seconds per frame.
    using se_clock = std::chrono::steady_clock;
    constexpr double clock_tick_period =
        se_clock::period::num / static_cast<double>(se_clock::period::den);
    if (clock_tick_period > spf_) {
        std::stringstream s;
        s << "The inter-frame time interval (" << std::fixed << std::setprecision(15) << spf_
          << " s) must be greater than the clock tick period (" << clock_tick_period
          << " s). Use a lower FPS value.";
        throw std::invalid_argument(s.str());
    }
}



se::ReaderStatus se::Reader::nextData(se::Image<float>& depth_image)
{
    if (!good()) {
        if (verbose_ >= 1) {
            std::clog << "Stopping reading due to reader status: " << status_ << "\n";
        }
        return status_;
    }
    nextFrame();
    status_ = nextDepth(depth_image);
    if (!good()) {
        if (verbose_ >= 1) {
            std::clog << "Stopping reading due to nextDepth() status: " << status_ << "\n";
        }
    }
    return status_;
}



se::ReaderStatus se::Reader::nextData(se::Image<float>& depth_image,
                                      se::Image<uint32_t>& rgba_image)
{
    if (!good()) {
        if (verbose_ >= 1) {
            std::clog << "Stopping reading due to reader status: " << status_ << "\n";
        }
        return status_;
    }
    nextFrame();
    status_ = nextDepth(depth_image);
    if (!good()) {
        if (verbose_ >= 1) {
            std::clog << "Stopping reading due to nextDepth() status: " << status_ << "\n";
        }
        return status_;
    }
    status_ = mergeStatus(nextRGBA(rgba_image), status_);
    if (!good()) {
        if (verbose_ >= 1) {
            std::clog << "Stopping reading due to nextRGBA() status: " << status_ << "\n";
        }
    }
    return status_;
}



se::ReaderStatus se::Reader::nextData(se::Image<float>& depth_image,
                                      se::Image<uint32_t>& rgba_image,
                                      Eigen::Matrix4f& T_WB)
{
    if (!good()) {
        if (verbose_ >= 1) {
            std::clog << "Stopping reading due to reader status: " << status_ << "\n";
        }
        return status_;
    }
    nextFrame();
    status_ = nextDepth(depth_image);
    if (!good()) {
        if (verbose_ >= 1) {
            std::clog << "Stopping reading due to nextDepth() status: " << status_ << "\n";
        }
        return status_;
    }
    status_ = mergeStatus(nextRGBA(rgba_image), status_);
    if (!good()) {
        if (verbose_ >= 1) {
            std::clog << "Stopping reading due to nextRGBA() status: " << status_ << "\n";
        }
        return status_;
    }
    status_ = mergeStatus(nextPose(T_WB), status_);
    if (!good()) {
        if (verbose_ >= 1) {
            std::clog << "Stopping reading due to nextPose() status: " << status_ << "\n";
        }
    }
    return status_;
}



se::ReaderStatus se::Reader::nextData(se::Image<float>& depth_image,
                                      se::Image<uint32_t>& rgba_image,
                                      Eigen::Matrix4f& T_WB,
                                      se::SegmentationResult& segmentation)
{
    if (!good()) {
        if (verbose_ >= 1) {
            std::clog << "Stopping reading due to reader status: " << status_ << "\n";
        }
        return status_;
    }
    nextFrame();
    status_ = nextDepth(depth_image);
    if (!good()) {
        if (verbose_ >= 1) {
            std::clog << "Stopping reading due to nextDepth() status: " << status_ << "\n";
        }
        return status_;
    }
    status_ = mergeStatus(nextRGBA(rgba_image), status_);
    if (!good()) {
        if (verbose_ >= 1) {
            std::clog << "Stopping reading due to nextRGBA() status: " << status_ << "\n";
        }
        return status_;
    }
    status_ = mergeStatus(nextPose(T_WB), status_);
    if (!good()) {
        if (verbose_ >= 1) {
            std::clog << "Stopping reading due to nextPose() status: " << status_ << "\n";
        }
        return status_;
    }
    status_ = mergeStatus(nextSegmentation(segmentation), status_);
    if (!good()) {
        if (verbose_ >= 1) {
            std::clog << "Stopping reading due to nextSegmentation() status: " << status_ << "\n";
        }
    }
    return status_;
}



void se::Reader::restart()
{
    frame_ = SIZE_MAX;
    ground_truth_frame_ = SIZE_MAX;
    prev_frame_timestamp_ = std::chrono::steady_clock::time_point();
    if (ground_truth_fs_.good() || ground_truth_fs_.eof()) {
        ground_truth_fs_.clear();
        ground_truth_fs_.seekg(0);
    }
    if (segmentation_fs_.good() || segmentation_fs_.eof()) {
        segmentation_fs_.clear();
        segmentation_fs_.seekg(0);
    }
}



bool se::Reader::good() const
{
    return ((status_ == se::ReaderStatus::ok) || (status_ == se::ReaderStatus::skip));
}



size_t se::Reader::frame() const
{
    return frame_;
}



size_t se::Reader::numFrames() const
{
    return num_frames_;
}



Eigen::Vector2i se::Reader::depthImageRes() const
{
    return depth_image_res_;
}



Eigen::Vector2i se::Reader::RGBAImageRes() const
{
    return rgba_image_res_;
}



bool se::Reader::isLiveReader() const
{
    return is_live_reader_;
}



se::ReaderStatus se::Reader::mergeStatus(se::ReaderStatus status_1, se::ReaderStatus status_2)
{
    const int istatus_1 = static_cast<int>(status_1);
    const int istatus_2 = static_cast<int>(status_2);
    return static_cast<se::ReaderStatus>(std::max(istatus_1, istatus_2));
}



se::ReaderStatus se::Reader::nextPose(Eigen::Matrix4f& T_WB)
{
    return readPose(T_WB, frame_, ground_truth_delimiter_);
}

se::ReaderStatus se::Reader::getPose(Eigen::Matrix4f& T_WB, const size_t frame)
{
    // Store and reset current ground truth frame
    size_t ground_truth_frame_curr = ground_truth_frame_;
    ground_truth_frame_ = SIZE_MAX;

    // Store and reset current ground truth file stream
    std::fpos ground_truth_fs_pos_curr = ground_truth_fs_.tellg();
    std::ios_base::iostate ground_truth_fs_state_curr = ground_truth_fs_.rdstate();
    ground_truth_fs_.clear();
    ground_truth_fs_.seekg(0);

    auto status = readPose(T_WB, frame, ground_truth_delimiter_);

    // Restore current state of ground truth variables
    ground_truth_frame_ = ground_truth_frame_curr;
    ground_truth_fs_.setstate(ground_truth_fs_state_curr);
    ground_truth_fs_.seekg(ground_truth_fs_pos_curr);
    return status;
}


se::ReaderStatus
se::Reader::readPose(Eigen::Matrix4f& T_WB, const size_t frame, const char delimiter)
{
    std::string line;
    while (true) {
        std::getline(ground_truth_fs_, line);
        // EOF reached
        if (!ground_truth_fs_.good()) {
            return se::ReaderStatus::eof;
        }
        // Ignore comment lines
        if (line[0] == '#') {
            continue;
        }
        // Skip ground truth data until the ones corresponding to the current frame
        // are found. This only happens when frames are dropped.
        ground_truth_frame_++;
        if (ground_truth_frame_ < frame) {
            continue;
        }
        // Data line read, split on spaces
        const std::vector<std::string> line_data = str_utils::split_str(line, delimiter);
        const size_t num_cols = line_data.size();
        if (num_cols < 7) {
            std::cerr << "Error: Invalid ground truth file format. "
                      << "Expected line format: ... tx ty tz qx qy qz qw\n";
            return se::ReaderStatus::error;
        }
        // Convert the last 7 columns to float
        float pose_data[7];
        bool pose_data_valid = true;
        for (uint8_t i = 0; i < 7; ++i) {
            pose_data[i] = std::stof(line_data[num_cols + i - 7]);
            if (!std::isfinite(pose_data[i])) {
                pose_data_valid = false;
            }
        }
        // Ensure all the pose elements are finite
        if (!pose_data_valid) {
            if (verbose_ >= 1) {
                std::cerr << "Warning: Expected finite ground truth pose but got";
                for (uint8_t i = 0; i < 7; ++i) {
                    std::cerr << " " << pose_data[i];
                }
                std::cerr << "\n";
            }
            return se::ReaderStatus::skip;
        }
        // Convert to position and orientation
        const Eigen::Vector3f position(pose_data[0], pose_data[1], pose_data[2]);
        const Eigen::Quaternionf orientation(
            pose_data[6], pose_data[3], pose_data[4], pose_data[5]);
        // Ensure the quaternion represents a valid orientation
        if (std::abs(orientation.norm() - 1.0f) > 1e-3) {
            if (verbose_ >= 1) {
                std::cerr << "Warning: Expected unit quaternion but got " << orientation.x() << " "
                          << orientation.y() << " " << orientation.z() << " " << orientation.w()
                          << " (x,y,z,w) with norm " << orientation.norm() << "\n";
            }
            return se::ReaderStatus::skip;
        }
        // Combine into the pose
        T_WB = Eigen::Matrix4f::Identity();
        T_WB.block<3, 1>(0, 3) = position;
        T_WB.block<3, 3>(0, 0) = orientation.toRotationMatrix();

        return se::ReaderStatus::ok;
    }
}



se::ReaderStatus se::Reader::nextSegmentation(se::SegmentationResult& segmentation)
{
    std::string line;
    while (true) {
        std::getline(segmentation_fs_, line);
        // EOF reached.
        if (!segmentation_fs_.good()) {
            return se::ReaderStatus::eof;
        }
        // Ignore comment lines.
        if (line[0] == '#') {
            continue;
        }
        // Skip segmentation data until the ones corresponding to the current frame
        // are found. This only happens when frames are dropped.
        segmentation_frame_++;
        if (segmentation_frame_ < frame_) {
            continue;
        }
        // Data line read, split on spaces.
        const std::vector<std::string> line_data = str_utils::split_str(line, ' ');
        const size_t num_cols = line_data.size();
        if (num_cols < 2) {
            std::cerr << "Invalid ground truth file format. "
                      << "Expected at least 2 columns but got " << num_cols << ".\n"
                      << "  Expected line format: ID timestamp ...\n"
                      << "  but got: " << line << std::endl;
            return se::ReaderStatus::error;
        }
        // Read the basename of the RGB image into the file_basename string
        const std::string file_basename = line_data[1];
        // Read the segmentation result
        if (segmentation.read(segmentation_base_dir_, file_basename)) {
            return se::ReaderStatus::ok;
        }
        else {
            return se::ReaderStatus::error;
        }
    }
}



void se::Reader::nextFrame()
{
    // Just increment the frame number when no FPS was specified or a live camera
    // is being used.
    if ((fps_ == 0.0f) || is_live_reader_) {
        frame_++;
        return;
    }

    // Avoid huge names in this function.
    namespace chr = std::chrono;

    // Compute the seconds since the previous frame was read.
    const chr::steady_clock::time_point curr_frame_timestamp = chr::steady_clock::now();
    const double delta_t =
        chr::duration<double>(curr_frame_timestamp - prev_frame_timestamp_).count();

    if (delta_t <= spf_) {
        // The previous frame was processed quickly, wait until it's time to get
        // the next one.
        const double seconds_to_sleep = spf_ - delta_t;
        std::this_thread::sleep_for(chr::duration<double>(seconds_to_sleep));
        frame_++;
    }
    else {
        // The previous frame was processed too slowly. Ensure that no frames are
        // dropped if no frames have been read yet (frame_ == SIZE_MAX).
        if (drop_frames_ && (frame_ != SIZE_MAX)) {
            const size_t delta_frame = std::ceil(fps_ * delta_t);
            frame_ += delta_frame;
        }
        else {
            frame_++;
        }
    }

    // Record the time right after frame_ was incremented. Due to the call to
    // std::this_thread::sleep_for it might be drastically different than
    // curr_frame_timestamp.
    prev_frame_timestamp_ = chr::steady_clock::now();
}
