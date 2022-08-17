// SPDX-FileCopyrightText: 2019-2020 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2019 Anna Dai
// SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#include "se/entropy.hpp"

#include <algorithm>

namespace se {
/** \brief Convert a probability to log-odds form.
 */
float prob_to_log_odds(float p)
{
    assert((0.0f <= p && "The probability mustn't be negative."));
    assert((p <= 1.0f && "The probability mustn't be greater than 1."));
    if (p == 0.0f) {
        return -INFINITY;
    }
    else if (p == 1.0f) {
        return INFINITY;
    }
    else {
        return log2f(p / (1.0f - p));
    }
}



/** \brief Convert a probability in log-odds form to a normal probability.
 */
float log_odds_to_prob(float l)
{
    // Too small/large values will produce NaNs, set an arbitrary limit that works on 32-bit floats.
    if (l <= -100.0f) {
        return 0.0f;
    }
    else if (l >= 100.0f) {
        return 1.0f;
    }
    else {
        return exp2f(l) / (1.0f + exp2f(l));
    }
}



/** \brief Compute the Shannon entropy of an occupancy probability.
 * H = -p log2(p) - (1-p) log2(1-p)
 * \return The entropy in the interval [0, 1].
 */
float entropy(float p)
{
    assert((0.0f <= p && "The probability mustn't be negative."));
    assert((p <= 1.0f && "The probability mustn't be greater than 1."));
    if (p == 0.0f || p == 1.0f) {
        return 0.0f;
    }
    else {
        const float p_compl = 1.0f - p;
        return -p * log2f(p) - p_compl * log2f(p_compl);
    }
}



float index_to_azimuth(const int x_idx, const int width, const float hfov)
{
    assert(0 <= x_idx);
    assert(x_idx < width);
    assert(0.0f < hfov);
    assert(hfov <= M_TAU_F);
    const float delta_theta = hfov / width;
    const float theta_max = hfov / 2.0f;
    // Image column coordinates increase towards the right but azimuth angle increases towards the
    // left.
    return se::math::wrap_angle_pi(theta_max - delta_theta * (x_idx + 0.5f));
}



float index_to_polar(int y_idx, int height, float vfov, float pitch_offset)
{
    assert(0 <= y_idx);
    assert(y_idx < height);
    assert(0.0f < vfov);
    assert(vfov <= M_PI_F);
    const float delta_phi = vfov / height;
    const float phi_min = M_PI_F / 2.0f - vfov / 2.0f + pitch_offset;
    // Both image row coordinates and polar angles increase downwards.
    return se::math::wrap_angle_pi(phi_min + delta_phi * (y_idx + 0.5f));
}



int azimuth_to_index(const float theta, const int width, const float hfov)
{
    assert(0.0f < hfov);
    assert(hfov <= M_TAU_F);
    const float delta_theta = hfov / width;
    const float theta_max = hfov / 2.0f;
    // Image column coordinates increase towards the right but azimuth angle increases towards the
    // left. Modulo with width because angles of ±pi result in an index of width otherwise.
    return static_cast<int>(
               roundf((theta_max - se::math::wrap_angle_pi(theta)) / delta_theta - 0.5f))
        % width;
}



/** \brief Compute the ray direction in the body frame B (x-forward, z-up) for a pixel x,y of an
 * image width x height when performing a 360-degree raycast with the supplied verical_fov.
 */
Eigen::Vector3f
ray_dir_M(int x, int y, int width, int height, float vertical_fov, float pitch_offset)
{
    // Compute the spherical coordinates of the ray.
    const float theta = index_to_azimuth(x, width, M_TAU_F);
    const float phi = index_to_polar(y, height, vertical_fov, pitch_offset);
    // Convert spherical coordinates to cartesian coordinates assuming a radius of 1.
    return Eigen::Vector3f(sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi));
}



Image<Eigen::Vector3f> ray_M_image(const SensorImpl& sensor, const Eigen::Matrix4f& T_MC)
{
    const int w = sensor.model.imageWidth();
    const int h = sensor.model.imageHeight();
    Image<Eigen::Vector3f> rays_M(w, h);
#pragma omp parallel for
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            // Compute the ray in the camera frame C.
            if (!sensor.model.backProject(Eigen::Vector2f(x, y), &rays_M(x, y))) {
                throw std::runtime_error("Invalid backprojection");
            }
            // Transform the ray to the map frame M.
            rays_M(x, y) = (T_MC.topLeftCorner<3, 3>() * rays_M(x, y)).normalized();
        }
    }
    return rays_M;
}



Image<Eigen::Vector3f> ray_M_360_image(const int width,
                                       const int height,
                                       const SensorImpl& sensor,
                                       const Eigen::Matrix4f& T_BC,
                                       const float roll_pitch_threshold)
{
    // Transformation from the camera body frame Bc (x-forward, z-up) to the camera frame C
    // (z-forward, x-right).
    Eigen::Matrix4f T_CBc;
    T_CBc << 0, -1, -0, -0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 1;
    const Eigen::Matrix4f T_MB = Eigen::Matrix4f::Identity();
    const Eigen::Matrix4f T_MBc = T_MB * T_BC * T_CBc;
    // Slightly reduce the camera's vertical FoV to account for the pitch threshold used to
    // determine whether the goal has been reached. Without this workaround the top or bottom of the
    // gain image will continue to have a gain since the candidate wasn't observed from the exact
    // pose expected.
    const float reduced_vfov = sensor.vertical_fov - 2.0f * roll_pitch_threshold;
    // The pitch angle of the camera relative to the body frame. Take the vertical FoV reduction
    // into account.
    const float pitch = math::rotm_to_pitch(T_MBc.topLeftCorner<3, 3>());
    Image<Eigen::Vector3f> rays_M(width, height);
#pragma omp parallel for
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            rays_M(x, y) = ray_dir_M(x, y, width, height, reduced_vfov, pitch);
        }
    }
    return rays_M;
}



/** Return the entropy and the point at which raycasting stopped.
 */
std::pair<float, Eigen::Vector3f> entropy_along_ray(const Octree<VoxelImpl::VoxelType>& map,
                                                    const Eigen::Vector3f& ray_origin_M,
                                                    const Eigen::Vector3f& ray_dir_M,
                                                    const float t_near,
                                                    const float t_far)
{
    static_assert(std::is_same<VoxelImpl, MultiresOFusion>::value,
                  "360 raycasting implemented only for MultiresOFusion");
    float ray_entropy = 0.0f;
    Eigen::Vector3f point_M = Eigen::Vector3f::Constant(NAN);
    const float t_step = map.voxelDim();
    for (float t = t_near; t <= t_far; t += t_step) {
        point_M = ray_origin_M + t * ray_dir_M;
        if (map.containsPoint(point_M)) {
            VoxelImpl::VoxelType::VoxelData data;
            map.getAtPoint(point_M, data);
            const float l = VoxelImpl::VoxelType::threshold(data);
            ray_entropy += entropy(log_odds_to_prob(l));
            if (l > VoxelImpl::surface_boundary) {
                // We have reached occupied space, stop raycasting.
                break;
            }
        }
        else {
            // We ended up outside the map, stop raycasting.
            break;
        }
    }
    return std::make_pair(ray_entropy, point_M);
}



int compute_window_width(const int image_width, const float hfov)
{
    const float window_percentage = hfov / M_TAU_F;
    // Even if the window width takes an extra column into account, the rayInFrustum() test will
    // reject it.
    return window_percentage * image_width + 0.5f;
}



std::vector<float> sum_windows(const Image<float>& entropy_image,
                               const Image<Eigen::Vector3f>& entropy_hits_M,
                               const SensorImpl& sensor,
                               const Eigen::Matrix4f& T_MB,
                               const Eigen::Matrix4f& T_BC)
{
    const int window_width = compute_window_width(entropy_image.width(), sensor.horizontal_fov);
    std::vector<float> window_sums(entropy_image.width(), 0.0f);
    for (size_t w = 0; w < window_sums.size(); w++) {
        int rays_in_frustum = 0;
        // The window's yaw is the azimuth angle of its middle column.
        const float yaw_M_left_edge = index_to_azimuth(w, entropy_image.width(), M_TAU_F);
        const float yaw_M = se::math::wrap_angle_pi(yaw_M_left_edge - sensor.horizontal_fov / 2.0f);
        Eigen::Matrix4f tmp_T_MB = T_MB;
        tmp_T_MB.topLeftCorner<3, 3>() = se::math::yaw_to_rotm(yaw_M);
        const Eigen::Matrix4f T_CM = se::math::to_inverse_transformation(tmp_T_MB * T_BC);
        for (int y = 0; y < entropy_image.height(); y++) {
            for (int i = 0; i < window_width; i++) {
                const int x = (w + i) % entropy_image.width();
                const Eigen::Vector3f ray_C = (T_CM * entropy_hits_M(x, y).homogeneous()).head<3>();
                if (sensor.rayInFrustum(ray_C)) {
                    window_sums[w] += entropy_image(x, y);
                    rays_in_frustum++;
                }
            }
        }
        if (rays_in_frustum) {
            // Normalize the entropy in the interval [0-1] using the number of rays in the window.
            window_sums[w] /= rays_in_frustum;
        }
    }
    return window_sums;
}



/** Find the window that contains the maximum sum of pixel values from image.
 * \return The column index of the leftmost edge of the window and its sum or -1 and -INFINITY if
 *         the image width is 0.
 */
std::pair<int, float> max_window(const Image<float>& entropy_image,
                                 const Image<Eigen::Vector3f>& entropy_hits_M,
                                 const SensorImpl& sensor,
                                 const Eigen::Matrix4f& T_MB,
                                 const Eigen::Matrix4f& T_BC)
{
    const std::vector<float> window_sums =
        sum_windows(entropy_image, entropy_hits_M, sensor, T_MB, T_BC);
    // Find the window with the maximum sum
    const auto max_it = std::max_element(window_sums.begin(), window_sums.end());
    if (max_it != window_sums.end()) {
        const float max_sum = *max_it;
        const int max_idx = std::distance(window_sums.begin(), max_it);
        return std::make_pair(max_idx, max_sum);
    }
    else {
        return std::make_pair(-1, -INFINITY);
    }
}



float max_ray_entropy(const float voxel_dim, const float near_plane, const float far_plane)
{
    const float ray_length = far_plane - near_plane;
    const float max_voxels_along_ray = ray_length / voxel_dim;
    constexpr float max_entropy_per_voxel = 1.0f;
    return max_entropy_per_voxel * max_voxels_along_ray;
}



void raycast_entropy(Image<float>& entropy_image,
                     Image<Eigen::Vector3f>& entropy_hits_M,
                     const Octree<VoxelImpl::VoxelType>& map,
                     const SensorImpl& sensor,
                     const Eigen::Matrix4f& T_MB,
                     const Eigen::Matrix4f& T_BC)
{
    if (entropy_image.width() != sensor.model.imageWidth()
        || entropy_image.height() != sensor.model.imageHeight()) {
        entropy_image = Image<float>(sensor.model.imageWidth(), sensor.model.imageHeight());
    }
    if (entropy_hits_M.width() != entropy_image.width()
        || entropy_hits_M.height() != entropy_image.height()) {
        entropy_hits_M = Image<Eigen::Vector3f>(entropy_image.width(), entropy_image.height());
    }
    const float ray_entropy_max =
        max_ray_entropy(map.voxelDim(), sensor.near_plane, sensor.far_plane);
    const Eigen::Vector3f& t_MB = T_MB.topRightCorner<3, 1>();
    const Image<Eigen::Vector3f> rays_M = ray_M_image(sensor, T_MB * T_BC);
#pragma omp parallel for
    for (int y = 0; y < entropy_image.height(); y++) {
#pragma omp simd
        for (int x = 0; x < entropy_image.width(); x++) {
            // Accumulate the entropy along the ray
            const auto r =
                entropy_along_ray(map, t_MB, rays_M(x, y), sensor.near_plane, sensor.far_plane);
            // Normalize the per-ray entropy in the interval [0-1].
            entropy_image(x, y) = r.first / ray_entropy_max;
            entropy_hits_M(x, y) = r.second;
        }
    }
}



void raycast_entropy_360(Image<float>& entropy_image,
                         Image<Eigen::Vector3f>& entropy_hits_M,
                         const Octree<VoxelImpl::VoxelType>& map,
                         const SensorImpl& sensor,
                         const Eigen::Matrix4f& T_MB,
                         const Eigen::Matrix4f& T_BC,
                         const float roll_pitch_threshold)
{
    if (entropy_hits_M.width() != entropy_image.width()
        || entropy_hits_M.height() != entropy_image.height()) {
        entropy_hits_M = Image<Eigen::Vector3f>(entropy_image.width(), entropy_image.height());
    }
    const float ray_entropy_max =
        max_ray_entropy(map.voxelDim(), sensor.near_plane, sensor.far_plane);
    const Eigen::Vector3f& t_MB = T_MB.topRightCorner<3, 1>();
    const Image<Eigen::Vector3f> rays_M = ray_M_360_image(
        entropy_image.width(), entropy_image.height(), sensor, T_BC, roll_pitch_threshold);
#pragma omp parallel for
    for (int y = 0; y < entropy_image.height(); y++) {
#pragma omp simd
        for (int x = 0; x < entropy_image.width(); x++) {
            // Accumulate the entropy along the ray
            const auto r =
                entropy_along_ray(map, t_MB, rays_M(x, y), sensor.near_plane, sensor.far_plane);
            // Normalize the per-ray entropy in the interval [0-1].
            entropy_image(x, y) = r.first / ray_entropy_max;
            entropy_hits_M(x, y) = r.second;
        }
    }
}



Image<float> mask_entropy_image(const Image<float>& entropy_image,
                                const Image<uint8_t>& frustum_overlap_mask)
{
    Image<float> masked_entropy(entropy_image.width(), entropy_image.height());
#pragma omp parallel for
    for (int y = 0; y < masked_entropy.height(); ++y) {
#pragma omp simd
        for (int x = 0; x < masked_entropy.width(); ++x) {
            masked_entropy(x, y) = frustum_overlap_mask(x, y) ? 0.0f : entropy_image(x, y);
        }
    }
    return masked_entropy;
}



std::tuple<float, float, int, int> optimal_yaw(const Image<float>& entropy_image,
                                               const Image<Eigen::Vector3f>& entropy_hits_M,
                                               const SensorImpl& sensor,
                                               const Eigen::Matrix4f& T_MB,
                                               const Eigen::Matrix4f& T_BC)
{
    // Use a sliding window to compute the yaw angle that results in the maximum entropy
    const std::pair<int, float> r = max_window(entropy_image, entropy_hits_M, sensor, T_MB, T_BC);
    // Azimuth angle of the left edge of the window
    const int best_idx = r.first;
    const float yaw_M_left_edge = index_to_azimuth(best_idx, entropy_image.width(), M_TAU_F);
    // The window's yaw is the azimuth angle of its middle column. Make sure to wrap the angle since
    // it can become < -pi if the window's middle wraps around.
    const float best_yaw_M =
        se::math::wrap_angle_pi(yaw_M_left_edge - sensor.horizontal_fov / 2.0f);
    const float best_entropy = r.second;
    return std::make_tuple(best_yaw_M,
                           best_entropy,
                           best_idx,
                           compute_window_width(entropy_image.width(), sensor.horizontal_fov));
}



Image<uint32_t> visualize_entropy(const Image<float>& entropy,
                                  const int window_idx,
                                  const int window_width,
                                  const bool visualize_yaw)
{
    Image<uint32_t> entropy_render(entropy.width(), entropy.height());
    for (size_t i = 0; i < entropy.size(); ++i) {
        // Halve the entropy when visualizing yaw to allow having white overlays.
        const uint8_t e = UINT8_MAX * (entropy[i] / (visualize_yaw ? 2.0f : 1.0f));
        entropy_render[i] = se::pack_rgba(e, e, e, 0xFF);
    }
    if (visualize_yaw) {
        overlay_yaw(entropy_render, window_idx, window_width);
    }
    return entropy_render;
}



Image<uint32_t> visualize_depth(const Image<Eigen::Vector3f>& entropy_hits_M,
                                const SensorImpl& sensor,
                                const Eigen::Matrix4f& T_MB,
                                const int window_idx,
                                const int window_width,
                                const bool visualize_yaw)
{
    const Eigen::Vector2i res(entropy_hits_M.width(), entropy_hits_M.height());
    const Eigen::Matrix4f T_BM = se::math::to_inverse_transformation(T_MB);
    // Convert the point cloud to depth along the ray
    Image<float> depth(res.x(), res.y(), 0.0f);
#pragma omp parallel for
    for (size_t i = 0; i < depth.size(); i++) {
        // Decrease the depth by a bit so that hits at the far plane are shown as invalid.
        // TODO SEM make this offset as big as the voxel dimensions.
        depth[i] = (T_BM * entropy_hits_M[i].homogeneous()).head<3>().norm() + 0.1f;
    }
    // Render to a colour image
    Image<uint32_t> depth_render(res.x(), res.y());
    se::depth_to_rgba(depth_render.data(), depth.data(), res, sensor.near_plane, sensor.far_plane);
    // Visualize the optimal yaw
    if (visualize_yaw) {
        overlay_yaw(depth_render, window_idx, window_width);
    }
    return depth_render;
}



void overlay_yaw(Image<uint32_t>& image, const int window_idx, const int window_width)
{
    // Show the FOV rectangle blended with the original image.
    {
        const int w = image.width();
        const int h = image.height();
        cv::Mat fov(cv::Size(w, h), CV_8UC4, cv::Scalar(255, 0, 0, 255));
        cv::Mat fov_alpha(cv::Size(w, h), CV_32FC1, cv::Scalar(0.0f));
        const cv::Scalar fov_on(0.5f);
        const int thickness = 1;
        // Compute minimum and maximum horizontal pixel coordinates of the FOV rectangle.
        const int x_min = window_idx;
        const int x_max = (window_idx + window_width - 1) % w;
        // Draw the vertical lines.
        cv::line(fov_alpha, cv::Point(x_min, 0), cv::Point(x_min, h - 1), fov_on, thickness);
        cv::line(fov_alpha, cv::Point(x_max, 0), cv::Point(x_max, h - 1), fov_on, thickness);
        // Draw the horizontal lines.
        if (x_min < x_max) {
            // No wrapping of the FOV rectangle.
            cv::line(fov_alpha, cv::Point(x_min, 0), cv::Point(x_max, 0), fov_on, thickness);
            cv::line(
                fov_alpha, cv::Point(x_min, h - 1), cv::Point(x_max, h - 1), fov_on, thickness);
        }
        else {
            // The FOV rectangle wraps around the left-right edges of the image.
            cv::line(fov_alpha, cv::Point(x_min, 0), cv::Point(w - 1, 0), fov_on, thickness);
            cv::line(fov_alpha, cv::Point(0, 0), cv::Point(x_max, 0), fov_on, thickness);
            cv::line(
                fov_alpha, cv::Point(x_min, h - 1), cv::Point(w - 1, h - 1), fov_on, thickness);
            cv::line(fov_alpha, cv::Point(0, h - 1), cv::Point(x_max, h - 1), fov_on, thickness);
        }
        // Blend with the entropy render.
        cv::Mat image_cv(cv::Size(w, h), CV_8UC4, image.data());
        cv::Mat image_alpha(cv::Size(w, h), CV_32FC1, cv::Scalar(0.5f));
        cv::blendLinear(image_cv, fov, image_alpha, fov_alpha, image_cv);
    }

    // Resize the image to 720xSOMETHING to allow having nicer visualizations.
    const int w = 720;
    const int h = w * static_cast<float>(image.height()) / image.width() + 0.5f;
    {
        cv::Mat image_cv(cv::Size(image.width(), image.height()), CV_8UC4, image.data());
        Image<uint32_t> out_image(w, h);
        cv::Mat out_image_cv(cv::Size(w, h), CV_8UC4, out_image.data());
        cv::resize(image_cv, out_image_cv, out_image_cv.size(), 0.0, 0.0, cv::INTER_NEAREST);
        image = out_image;
    }

    // Show the yaw angle major and minor tick marks.
    cv::Mat image_cv(cv::Size(w, h), CV_8UC4, image.data());
    const cv::Scalar tick_color = cv::Scalar(255, 255, 255, 255);
    const int major_tick_thickness = 2 * w / 360;
    const int minor_tick_thickness = 1 * w / 360;
    for (float t = 0.0f; t <= 1.0f; t += 0.25f) {
        const int x = t * (w - 1);
        cv::line(image_cv,
                 cv::Point(x, h - 1),
                 cv::Point(x, h - 1 - 0.04 * h),
                 tick_color,
                 major_tick_thickness);
    }
    for (float t = 0.125f; t < 1.0f; t += 0.125f) {
        const int x = t * (w - 1);
        cv::line(image_cv,
                 cv::Point(x, h - 1),
                 cv::Point(x, h - 1 - 0.02 * h),
                 tick_color,
                 minor_tick_thickness);
    }

    // Draw the angle labels.
    constexpr auto font = cv::FONT_HERSHEY_SIMPLEX;
    const int thickness = 1 * w / 360;
    std::map<float, std::string> labels{
        {M_PI_F / 2.0f, "90"}, {0.0f, "0"}, {-M_PI_F / 2.0f, "-90 "}};
    for (const auto& [angle, label] : labels) {
        // Get the dimensions of the resulting text box for a scale of 1.
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, font, 1.0, thickness, &baseline);
        // Compute the scale so that the text height is 2% of the image height.
        const double scale = h / 10.0 / text_size.height;
        // Scale the baseline.
        baseline = scale * (baseline + thickness);
        // Get the actual text size.
        text_size = cv::getTextSize(label, font, scale, thickness, &baseline);
        // Center the text above the angle tick mark.
        const int x = azimuth_to_index(angle, w, M_TAU_F);
        cv::Point text_pos(x - text_size.width / 2, h - 1 - baseline);
        cv::putText(
            image_cv, label, text_pos, font, scale, cv::Scalar(255, 255, 255, 255), thickness);
    }
}



void render_pose_entropy_depth(Image<uint32_t>& entropy,
                               Image<uint32_t>& depth,
                               const Octree<VoxelImpl::VoxelType>& map,
                               const SensorImpl& sensor,
                               const Eigen::Matrix4f& T_MB,
                               const Eigen::Matrix4f& T_BC,
                               const bool visualize_yaw,
                               const float roll_pitch_threshold)
{
    Image<float> raw_entropy(entropy.width(), entropy.height());
    Image<Eigen::Vector3f> entropy_hits(entropy.width(), entropy.height());
    raycast_entropy_360(raw_entropy, entropy_hits, map, sensor, T_MB, T_BC, roll_pitch_threshold);
    const float yaw_M = se::math::rotm_to_yaw(T_MB.topLeftCorner<3, 3>());
    const int window_idx = se::azimuth_to_index(
        se::math::wrap_angle_2pi(yaw_M + sensor.horizontal_fov / 2.0f), entropy.width(), M_TAU_F);
    const int window_width = compute_window_width(entropy.width(), sensor.horizontal_fov);
    entropy = visualize_entropy(raw_entropy, window_idx, window_width, visualize_yaw);
    depth = visualize_depth(entropy_hits, sensor, T_MB, window_idx, window_width, visualize_yaw);
}

} // namespace se
