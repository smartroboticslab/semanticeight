// SPDX-FileCopyrightText: 2019-2020 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2019 Anna Dai
// SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#include "se/entropy.hpp"

#include <algorithm>
#include <frustum_intersector.hpp>

namespace se {
/** \brief Convert a probability to log-odds form.
   */
float prob_to_log_odds(float p)
{
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
    if (p == 0.0f || p == 1.0f) {
        return 0.0f;
    }
    else {
        const float p_compl = 1.0f - p;
        return -p * log2f(p) - p_compl * log2f(p_compl);
    }
}



/** \brief Compute the azimuth angle given a column index, the image width and the horizontal
   * sensor FOV. It is assumed that the image spans an azimuth angle range of hfov and that azimuth
   * angle 0 corresponds to the middle column of the image.
   * \return The azimuth angle in the interval [-pi,pi].
   */
float azimuth_from_index(const int x_idx, const int width, const float hfov)
{
    assert(0 <= x_idx);
    assert(x_idx < width);
    assert(0 < hfov);
    assert(hfov <= M_TAU_F);
    const float delta_theta = hfov / width;
    const float theta_max = hfov / 2.0f;
    // Image column coordinates increase towards the right but azimuth angle increases towards the
    // left.
    return theta_max - delta_theta * (x_idx + 0.5f);
}



/** \brief Compute the polar angle given a row index, the image height and the vertical sensor
   * FOV. It is assumed that the image spans a polar angle range of vfov and that polar angle pi/2
   * corresponds to the middle row of the image.
   * \return The polar angle in the interval [0,pi].
   */
float polar_from_index(int y_idx, int height, float vfov, float pitch_offset)
{
    assert(0 <= y_idx);
    assert(y_idx < height);
    assert(0 < vfov);
    assert(vfov <= M_PI_F);
    const float delta_phi = vfov / height;
    const float phi_min = M_PI_F / 2.0f - vfov / 2.0f + pitch_offset;
    // Both image row coordinates and polar angles increase downwards.
    return phi_min + delta_phi * (y_idx + 0.5f);
}



/** \brief Compute the column index given an azimuth angle, the image width and the horizontal
   * sensor FOV. It is assumed that the image spans an azimuth angle  range of hfov and that azimuth
   * angle 0 corresponds to the middle column of the image.
   * \return The column index in the interval [0,width-1].
   */
int index_from_azimuth(const float theta, const int width, const float hfov)
{
    assert(theta >= -hfov / 2.0f);
    assert(theta <= hfov / 2.0f);
    const float delta_theta = hfov / width;
    const float theta_max = hfov / 2.0f;
    // Image column coordinates increase towards the right but azimuth angle increases towards the
    // left.
    return roundf((theta_max - theta) / delta_theta - 0.5f);
}



/** \brieaf Compute the ray direction in the body frame B (x-forward, z-up) for a pixel x,y of an
 * image width x height when performing a 360-degree raycast with the supplied verical_fov.
 */
Eigen::Vector3f
ray_dir_from_pixel(int x, int y, int width, int height, float vertical_fov, float pitch_offset)
{
    // Compute the spherical coordinates of the ray.
    const float theta = azimuth_from_index(x, width, M_TAU_F);
    const float phi = polar_from_index(y, height, vertical_fov, pitch_offset);
    // Convert spherical coordinates to cartesian coordinates assuming a radius of 1.
    return Eigen::Vector3f(sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi));
}



/** \brieaf Compute the ray directions in the body frame B (x-forward, z-up) for all pixels of an
 * image width x height.
 */
Image<Eigen::Vector3f>
ray_image(const int width, const int height, const SensorImpl& sensor, const Eigen::Matrix4f& T_BC)
{
    // Transformation from the camera body frame Bc (x-forward, z-up) to the camera frame C
    // (z-forward, x-right).
    Eigen::Matrix4f T_CBc;
    T_CBc << 0, -1, -0, -0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 1;
    const Eigen::Matrix4f T_BBc = T_BC * T_CBc;
    // The pitch angle of the camera relative to the body frame.
    const float pitch = math::wrap_angle_pi(T_BBc.topLeftCorner<3, 3>().eulerAngles(2, 1, 0).y());
    Image<Eigen::Vector3f> rays(width, height);
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            rays(x, y) = ray_dir_from_pixel(x, y, width, height, sensor.vertical_fov, pitch);
        }
    }
    return rays;
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
        //const float l = map.interpAtPoint(point_M, VoxelImpl::VoxelType::threshold).first;
        VoxelImpl::VoxelType::VoxelData data;
        map.getAtPoint(point_M, data);
        const float l = VoxelImpl::VoxelType::threshold(data);
        ray_entropy += entropy(log_odds_to_prob(l));
        if (l > VoxelImpl::surface_boundary) {
            // We have reached occupied space, stop raycasting.
            break;
        }
    }
    return std::make_pair(ray_entropy, point_M);
}



std::vector<float> sum_windows(const Image<float>& entropy_image,
                               const Image<Eigen::Vector3f>& /* entropy_hits_M */,
                               const Image<float>& frustum_overlap_image,
                               const SensorImpl& sensor,
                               const Eigen::Matrix4f& T_MB,
                               const Eigen::Matrix4f& T_BC)
{
    const Image<Eigen::Vector3f> rays_B =
        ray_image(entropy_image.width(), entropy_image.height(), sensor, T_BC);
    const float window_percentage = sensor.horizontal_fov / M_TAU_F;
    // Even if the window width takes an extra column into account, the rayInFrustum() test will
    // reject it.
    const int window_width = window_percentage * entropy_image.width() + 0.5f;
    std::vector<float> window_sums(entropy_image.width(), 0.0f);
    for (size_t w = 0; w < window_sums.size(); w++) {
        int n = 0;
        // The window's yaw is the azimuth angle of its middle column
        const float theta = azimuth_from_index(w, entropy_image.width(), M_TAU_F);
        const float yaw_M = theta - sensor.horizontal_fov / 2.0f;
        Eigen::Matrix4f tmp_T_MB = T_MB;
        tmp_T_MB.topLeftCorner<3, 3>() = se::math::yaw_to_rotm(yaw_M);
        const Eigen::Matrix4f T_CM = se::math::to_inverse_transformation(tmp_T_MB * T_BC);
        for (int y = 0; y < entropy_image.height(); y++) {
            for (int i = 0; i < window_width; i++) {
                const int x = (w + i) % entropy_image.width();
                const Eigen::Vector3f ray_C = (T_CM * T_MB * rays_B(x, y).homogeneous()).head<3>();
                if (sensor.rayInFrustum(ray_C)) {
                    if (frustum_overlap_image[x] < 1e-6f) {
                        window_sums[w] += entropy_image(x, y);
                    }
                    n++;
                }
            }
        }
        // Normalize the entropy in the interval [0-1] using the number of rays in the window.
        window_sums[w] /= n;
    }
    return window_sums;
}



/** Find the window that contains the maximum sum of pixel values from image.
   * \return The column index of the leftmost edge of the window and its sum or -1 and -INFINITY if
   *         the image width is 0.
   */
std::pair<int, float> max_window(const Image<float>& entropy_image,
                                 const Image<Eigen::Vector3f>& entropy_hits_M,
                                 const Image<float>& frustum_overlap_image,
                                 const SensorImpl& sensor,
                                 const Eigen::Matrix4f& T_MB,
                                 const Eigen::Matrix4f& T_BC)
{
    const std::vector<float> window_sums =
        sum_windows(entropy_image, entropy_hits_M, frustum_overlap_image, sensor, T_MB, T_BC);
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
    // This is the number of voxels and the max entropy since the max entropy per-voxel is 1.
    return BLOCK_SIZE * sqrtf(3.0f) * (far_plane - near_plane) / voxel_dim;
}



void raycast_entropy(Image<float>& entropy_image,
                     Image<Eigen::Vector3f>& entropy_hits_M,
                     const Octree<VoxelImpl::VoxelType>& map,
                     const SensorImpl& sensor,
                     const Eigen::Matrix4f& T_MB,
                     const Eigen::Matrix4f& T_BC)
{
    if (entropy_hits_M.width() != entropy_image.width()
        || entropy_hits_M.height() != entropy_image.height()) {
        entropy_hits_M = Image<Eigen::Vector3f>(entropy_image.width(), entropy_image.height());
    }
    const Eigen::Vector3f& t_MB = T_MB.topRightCorner<3, 1>();
    const Image<Eigen::Vector3f> rays =
        ray_image(entropy_image.width(), entropy_image.height(), sensor, T_BC);
#pragma omp parallel for
    for (int y = 0; y < entropy_image.height(); y++) {
#pragma omp simd
        for (int x = 0; x < entropy_image.width(); x++) {
            // Accumulate the entropy along the ray
            const auto r =
                entropy_along_ray(map, t_MB, rays(x, y), sensor.near_plane, sensor.far_plane);
            // Normalize the per-ray entropy in the interval [0-1].
            entropy_image(x, y) =
                r.first / max_ray_entropy(map.voxelDim(), sensor.near_plane, sensor.far_plane);
            entropy_hits_M(x, y) = r.second;
        }
    }
}



void frustum_overlap(Image<float>& frustum_overlap_image,
                     const SensorImpl& sensor,
                     const Eigen::Matrix4f& T_MC,
                     const Eigen::Matrix4f& T_BC,
                     const PoseHistory* T_MB_history)
{
    const Eigen::Matrix4f T_CM = se::math::to_inverse_transformation(T_MC);
    const se::PoseVector neighbors = T_MB_history->neighbourPoses(T_MC, sensor);
#pragma omp parallel for
    for (int x = 0; x < frustum_overlap_image.width(); x++) {
        std::vector<float> overlap;
        overlap.reserve(neighbors.size());
        for (const auto& n_T_MB : neighbors) {
            // Convert the neighbor pose to the candidate frame.
            const Eigen::Matrix4f T_CCn = T_CM * n_T_MB * T_BC;
            overlap.push_back(fi::frustum_intersection_pc(sensor.frustum_vertices_, T_CCn));
        }
        frustum_overlap_image[x] =
            overlap.empty() ? 0.0f : *std::max_element(overlap.begin(), overlap.end());
    }
}



std::pair<float, float> optimal_yaw(const Image<float>& entropy_image,
                                    const Image<Eigen::Vector3f>& entropy_hits_M,
                                    const Image<float>& frustum_overlap_image,
                                    const SensorImpl& sensor,
                                    const Eigen::Matrix4f& T_MB,
                                    const Eigen::Matrix4f& T_BC)
{
    // Use a sliding window to compute the yaw angle that results in the maximum entropy
    const std::pair<int, float> r =
        max_window(entropy_image, entropy_hits_M, frustum_overlap_image, sensor, T_MB, T_BC);
    // Azimuth angle of the left edge of the window
    const int best_idx = r.first;
    const float theta = azimuth_from_index(best_idx, entropy_image.width(), M_TAU_F);
    // The window's yaw is the azimuth angle of its middle column
    const float best_yaw_M = theta - sensor.horizontal_fov / 2.0f;
    const float best_entropy = r.second;
    return std::make_pair(best_yaw_M, best_entropy);
}



Image<uint32_t> visualize_entropy(const Image<float>& entropy,
                                  const SensorImpl& sensor,
                                  const float yaw_M,
                                  const bool visualize_yaw)
{
    Image<uint32_t> entropy_render(entropy.width(), entropy.height());
    for (size_t i = 0; i < entropy.size(); ++i) {
        // Scale and clamp the entropy for visualization since its values are typically too low.
        const uint8_t e = se::math::clamp(
            UINT8_MAX * (6.0f * entropy[i]) + 0.5f, 0.0f, static_cast<float>(UINT8_MAX));
        entropy_render[i] = se::pack_rgba(e, e, e, 0xFF);
    }
    if (visualize_yaw) {
        overlay_yaw(entropy_render, yaw_M, sensor);
    }
    return entropy_render;
}



Image<uint32_t> visualize_depth(const Image<Eigen::Vector3f>& entropy_hits_M,
                                const SensorImpl& sensor,
                                const Eigen::Matrix4f& T_MB,
                                const float yaw_M,
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
        overlay_yaw(depth_render, yaw_M, sensor);
    }
    return depth_render;
}



void overlay_yaw(Image<uint32_t>& image, const float yaw_M, const SensorImpl& sensor)
{
    // Resize the image to 720xSOMETHING to allow having nicer visualizations.
    {
        cv::Mat image_cv(cv::Size(image.width(), image.height()), CV_8UC4, image.data());
        const int w = 720;
        const int h = w * static_cast<float>(image.height()) / image.width() + 0.5f;
        Image<uint32_t> out_image(w, h);
        cv::Mat out_image_cv(cv::Size(w, h), CV_8UC4, out_image.data());
        cv::resize(image_cv, out_image_cv, out_image_cv.size(), 0.0, 0.0, cv::INTER_NEAREST);
        image = out_image;
    }

    // Visualize the FOV rectangle.
    const int w = image.width();
    const int h = image.height();
    cv::Mat image_cv(cv::Size(w, h), CV_8UC4, image.data());
    const cv::Scalar fov_color = cv::Scalar(255, 0, 0, 128);
    const int line_thickness = 2 * w / 360;
    // Compute minimum and maximum horizontal pixel coordinates of the FOV rectangle.
    const int x_min =
        index_from_azimuth(yaw_M + sensor.horizontal_fov / 2.0f, image.width(), M_TAU_F);
    const int x_max =
        index_from_azimuth(yaw_M - sensor.horizontal_fov / 2.0f, image.width(), M_TAU_F);
    // Draw the vertical lines.
    cv::line(image_cv, cv::Point(x_min % w, 0), cv::Point(x_min % w, h), fov_color, line_thickness);
    cv::line(image_cv, cv::Point(x_max % w, 0), cv::Point(x_max % w, h), fov_color, line_thickness);
    // Draw the horizontal lines.
    if (0 <= x_min && x_max < w) {
        cv::line(
            image_cv, cv::Point(x_min % w, 0), cv::Point(x_max % w, 0), fov_color, line_thickness);
        cv::line(
            image_cv, cv::Point(x_min % w, h), cv::Point(x_max % w, h), fov_color, line_thickness);
    }
    else {
        cv::line(image_cv, cv::Point(x_min % w, 0), cv::Point(w - 1, 0), fov_color, line_thickness);
        cv::line(image_cv, cv::Point(0, 0), cv::Point(x_max % w, 0), fov_color, line_thickness);
        cv::line(image_cv, cv::Point(x_min % w, h), cv::Point(w - 1, h), fov_color, line_thickness);
        cv::line(image_cv, cv::Point(0, h), cv::Point(x_max % w, h), fov_color, line_thickness);
    }

    // Show the yaw angle major and minor tick marks.
    const cv::Scalar tick_color = cv::Scalar(255, 255, 255, 128);
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
        const int x = index_from_azimuth(angle, w, M_TAU_F);
        cv::Point text_pos(x - text_size.width / 2, h - 1 - baseline);
        cv::putText(
            image_cv, label, text_pos, font, scale, cv::Scalar(255, 255, 255, 128), thickness);
    }
}

} // namespace se
