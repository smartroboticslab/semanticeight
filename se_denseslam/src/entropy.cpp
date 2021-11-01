// SPDX-FileCopyrightText: 2019-2020 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2019 Anna Dai
// SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <frustum_intersector.hpp>

#include "se/entropy.hpp"

namespace se {
  /** \brief Convert a probability to log-odds form.
   */
  float prob_to_log_odds(float p) {
    if (p == 0.0f) {
      return -INFINITY;
    } else if (p == 1.0f) {
      return INFINITY;
    } else {
      return log2f(p / (1.0f - p));
    }
  }



  /** \brief Convert a probability in log-odds form to a normal probability.
   */
  float log_odds_to_prob(float l) {
    // Too small/large values will produce NaNs, set an arbitrary limit that works on 32-bit floats.
    if (l <= -100.0f) {
      return 0.0f;
    } else if (l >= 100.0f) {
      return 1.0f;
    } else {
      return exp2f(l) / (1.0f + exp2f(l));
    }
  }



  /** \brief Compute the Shannon entropy of an occupancy probability.
   * H = -p log2(p) - (1-p) log2(1-p)
   * \return The entropy in the interval [0, 1].
   */
  float entropy(float p) {
    if (p == 0.0f || p == 1.0f) {
      return 0.0f;
    } else {
      const float p_compl = 1.0f - p;
      return -p * log2f(p) - p_compl * log2f(p_compl);
    }
  }



  /** \brief Compute the azimuth angle given a column index, the image width and the horizontal
   * sensor FOV. It is assumed that the image spans an azimuth angle range of hfov and that azimuth
   * angle 0 corresponds to the middle column of the image.
   * \return The azimuth angle in the interval [-pi,pi].
   */
  float azimuth_from_index(const int x_idx, const int width, const float hfov) {
    assert(0 <= x_idx);
    assert(x_idx < width);
    assert(0 < hfov);
    assert(hfov <= 2.0f * M_PI_F);
    const float delta_theta = hfov / width;
    const float theta_max = hfov / 2.0f;
    // Image column coordinates increase towards the right but azimuth angle increases towards the
    // left.
    return theta_max - delta_theta * x_idx;
  }



  /** \brief Compute the polar angle given a row index, the image height and the vertical sensor
   * FOV. It is assumed that the image spans a polar angle range of vfov and that polar angle pi/2
   * corresponds to the middle row of the image.
   * \return The polar angle in the interval [0,pi].
   */
  float polar_from_index(int y_idx, int height, float vfov, float pitch_offset) {
    assert(0 <= y_idx);
    assert(y_idx < height);
    assert(0 < vfov);
    assert(vfov <= M_PI_F);
    const float delta_phi = vfov / height;
    const float phi_min = M_PI_F / 2.0f - vfov / 2.0f + pitch_offset;
    // Both image row coordinates and polar angles increase downwards.
    return phi_min + delta_phi * y_idx;
  }



  /** \brief Compute the column index given an azimuth angle, the image width and the horizontal
   * sensor FOV. It is assumed that the image spans an azimuth angle  range of hfov and that azimuth
   * angle 0 corresponds to the middle column of the image.
   * \return The column index in the interval [0,width-1].
   */
  int index_from_azimuth(const float theta, const int width, const float hfov) {
    assert(theta >= -hfov / 2.0f);
    assert(theta <=  hfov / 2.0f);
    const float delta_theta = hfov / width;
    const float theta_max = hfov / 2.0f;
    // Image column coordinates increase towards the right but azimuth angle increases towards the
    // left.
    return roundf((theta_max - theta) / delta_theta);
  }



  /** \brieaf Compute the ray direction in the map frame (x-forward, z-up) for a pixel x,y of an
   * image width x height when performing a 360-degree raycast with the supplied verical_fov.
   */
  Eigen::Vector3f ray_dir_from_pixel(int   x,
                                     int   y,
                                     int   width,
                                     int   height,
                                     float vertical_fov,
                                     float pitch_offset) {
    // Compute the spherical coordinates of the ray.
    const float theta = azimuth_from_index(x, width, 2.0f * M_PI_F);
    const float phi = polar_from_index(y, height, vertical_fov, pitch_offset);
    // Convert spherical coordinates to cartesian coordinates assuming a radius of 1.
    const Eigen::Vector3f ray_dir_M (sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi));
    return ray_dir_M;
  }



  float information_gain_along_ray(const Octree<VoxelImpl::VoxelType>& map,
                                   const Eigen::Vector3f&              ray_origin_M,
                                   const Eigen::Vector3f&              ray_dir_M,
                                   const float                         t_near,
                                   const float                         t_far) {
    static_assert(std::is_same<VoxelImpl, MultiresOFusion>::value,
        "360 raycasting implemented only for MultiresOFusion");
    float ray_entropy = 0.0f;
    const float t_step = map.voxelDim() / 2.0f;
    for (float t = t_near; t <= t_far; t += t_step) {
      const Eigen::Vector3f point_M = ray_origin_M + t * ray_dir_M;
      const float l = map.interpAtPoint(point_M, VoxelImpl::VoxelType::threshold).first;
      ray_entropy += entropy(log_odds_to_prob(l));
      if (l > VoxelImpl::surface_boundary) {
        // We have reached occupied space, stop raycasting.
        break;
      }
    }
    return ray_entropy;
  }


  std::vector<float> sum_columns(const Image<float>& entropy_image,
                                 const Image<float>& frustum_overlap_image) {
    std::vector<float> column_sums (entropy_image.width(), 0.0f);
    // Don't add an omp parallel for here
    for (int y = 0; y < entropy_image.height(); y++) {
      for (int x = 0; x < entropy_image.width(); x++) {
        column_sums[x] += entropy_image(x, y) * (1.0f - frustum_overlap_image[x]);
      }
    }
    return column_sums;
  }



  std::vector<float> sum_windows(const std::vector<float>& column_sums, const int window_width) {
    std::vector<float> window_sums (column_sums.size(), 0.0f);
    for (size_t x = 0; x < column_sums.size(); x++) {
      // Accumulate the values inside the window with wrap-around
      for (int i = 0; i < window_width; i++) {
        window_sums[x] += column_sums[(x + i) % column_sums.size()];
      }
    }
    return window_sums;
  }



  /** Find the window that contains the maximum sum of pixel values from image.
   * \return The column index of the leftmost edge of the window and its sum or -1 and -INFINITY if
   *         the image width is 0.
   */
  std::pair<int, float> max_window(const Image<float>& image,
                                   const Image<float>& frustum_overlap_image,
                                   const int           window_width) {
    float max_sum = -INFINITY;
    int max_idx = -1;
    // Sum all columns in the image to speed up subsequent computations. The window is applied to
    // all rows so this way we'll only do window_width - 1 additions for each window instead of
    // (window_width - 1) * entropy_image.height().
    const std::vector<float> column_sums = sum_columns(image, frustum_overlap_image);
    const std::vector<float> window_sums = sum_windows(column_sums, window_width);
    // Find the window with the maximum sum
    const auto max_it = std::max_element(window_sums.begin(), window_sums.end());
    if (max_it != window_sums.end()) {
      max_sum = *max_it;
      max_idx = std::distance(window_sums.begin(), max_it);;
    }
    return std::make_pair(max_idx, max_sum);
  }



  Image<Eigen::Vector3f> ray_image(const int              width,
                                   const int              height,
                                   const SensorImpl&      sensor,
                                   const float            pitch_offset) {
    Image<Eigen::Vector3f> rays(width, height);
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        const Eigen::Vector3f ray_M = ray_dir_from_pixel(x, y, width, height, sensor.vertical_fov, pitch_offset);
        rays(x, y) = ray_M;
      }
    }
    return rays;
  }



  float max_ray_entropy(const float voxel_dim,
                        const float near_plane,
                        const float far_plane) {
    return 8.0f * sqrtf(3.0f) * (far_plane - near_plane) / voxel_dim;
  }



  void raycast_entropy(Image<float>&                       entropy_image,
                       const Octree<VoxelImpl::VoxelType>& map,
                       const SensorImpl&                   sensor,
                       const Eigen::Matrix4f&              T_MB,
                       const Eigen::Matrix4f&              T_BC) {
    const Eigen::Vector3f& t_MB = T_MB.topRightCorner<3,1>();
    // Transformation from the camera body frame Bc (x-forward, z-up) to the camera frame C
    // (z-forward, x-right).
    Eigen::Matrix4f T_CBc;
    T_CBc << 0, -1, -0, -0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 1;
    const Eigen::Matrix4f T_BBc = T_BC * T_CBc;
    // The pitch angle of the camera relative to the body frame.
    const float pitch = math::wrap_angle_pi(T_BBc.topLeftCorner<3,3>().eulerAngles(2, 1, 0).y());
#pragma omp parallel for
    for (int y = 0; y < entropy_image.height(); y++) {
#pragma omp simd
      for (int x = 0; x < entropy_image.width(); x++) {
        const Eigen::Vector3f ray_dir_M = ray_dir_from_pixel(x, y,
            entropy_image.width(), entropy_image.height(), sensor.vertical_fov, pitch);
        // Accumulate the entropy along the ray
        entropy_image(x, y) = information_gain_along_ray(map, t_MB, ray_dir_M,
            sensor.near_plane, sensor.far_plane);
        // Normalize the per-ray entropy in the interval [0-1].
        entropy_image(x, y) /= max_ray_entropy(map.voxelDim(), sensor.near_plane, sensor.far_plane);
      }
    }
  }



  void raycast_depth(Image<float>&                       depth_image,
                     const Octree<VoxelImpl::VoxelType>& map,
                     const SensorImpl&                   sensor,
                     const Eigen::Matrix4f&              T_MB,
                     const Eigen::Matrix4f&              T_BC) {
    const Eigen::Vector3f& t_MB = T_MB.topRightCorner<3,1>();
    // Transformation from the camera body frame Bc (x-forward, z-up) to the camera frame C
    // (z-forward, x-right).
    Eigen::Matrix4f T_CBc;
    T_CBc << 0, -1, -0, -0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 1;
    const Eigen::Matrix4f T_BBc = T_BC * T_CBc;
    // The pitch angle of the camera relative to the body frame.
    const float pitch = math::wrap_angle_pi(T_BBc.topLeftCorner<3,3>().eulerAngles(2, 1, 0).y());
#pragma omp parallel for
    for (int y = 0; y < depth_image.height(); y++) {
#pragma omp simd
      for (int x = 0; x < depth_image.width(); x++) {
        const Eigen::Vector3f ray_dir_M = ray_dir_from_pixel(x, y,
            depth_image.width(), depth_image.height(), sensor.vertical_fov, pitch);
        // Compute the hit along the ray
        const Eigen::Vector4f hit_M = VoxelImpl::raycast(map, t_MB, ray_dir_M,
            sensor.near_plane, sensor.far_plane);
        if (hit_M == Eigen::Vector4f::Zero()) {
          depth_image(x, y) = 0.0f;
        } else {
          depth_image(x, y) = (hit_M.head<3>() - t_MB).norm();
        }
      }
    }
  }



  void frustum_overlap(Image<float>&          frustum_overlap_image,
                       const SensorImpl&      sensor,
                       const Eigen::Matrix4f& T_MC,
                       const PoseHistory&     T_MC_history) {
    const Eigen::Matrix4f T_CM = se::math::to_inverse_transformation(T_MC);
    const se::PoseVector neighbors = T_MC_history.neighbourPoses(T_MC, sensor);
#pragma omp parallel for
    for (int x = 0; x < frustum_overlap_image.width(); x++) {
      std::vector<float> overlap;
      overlap.reserve(neighbors.size());
      for (const auto& n_T_MC : neighbors) {
        // Convert the neighbor pose to the candidate frame.
        const Eigen::Matrix4f T_CCn = T_CM * n_T_MC;
        overlap.push_back(fi::frustum_intersection_pc(sensor.frustum_vertices_, T_CCn));
      }
      frustum_overlap_image[x] = *std::max_element(overlap.begin(), overlap.end());
    }
  }



  std::pair<float, float> optimal_yaw(const Image<float>& entropy_image,
                                      const Image<float>& frustum_overlap_image,
                                      const SensorImpl&   sensor) {
    // Use a sliding window to compute the yaw angle that results in the maximum entropy
    const float window_percentage = sensor.horizontal_fov / (2.0f * M_PI_F);
    const int window_width = window_percentage * entropy_image.width() + 0.5f;
    const std::pair<int, float> r = max_window(entropy_image, frustum_overlap_image, window_width);
    // Azimuth angle of the left edge of the window
    const int best_idx = r.first;
    const float theta = azimuth_from_index(best_idx, entropy_image.width(), 2.0f * M_PI_F);
    // The window's yaw is the azimuth angle of its middle column
    const float best_yaw_M = theta - sensor.horizontal_fov / 2.0f;
    // Normalize the entropy in the interval [0-1] using the window size.
    const float best_entropy = r.second / (window_width * entropy_image.height());
    return std::make_pair(best_yaw_M, best_entropy);
  }



  void overlay_yaw(Image<uint32_t>&  image,
                   const float       yaw_M,
                   const SensorImpl& sensor) {
    // Compute minimum and maximum horizontal pixel coordinates of the FOV rectangle
    const int x_min = index_from_azimuth(yaw_M + sensor.horizontal_fov / 2.0f, image.width(), 2.0f * M_PI_F);
    const int x_max = index_from_azimuth(yaw_M - sensor.horizontal_fov / 2.0f, image.width(), 2.0f * M_PI_F);
    // Draw the FOV rectangle on the image
    const uint32_t fov_color = 0xFF0000FF;
#pragma omp parallel for
    for (int x = x_min; x <= x_max; x++) {
      image(x % image.width(),                  0) = se::blend(image(x % image.width(),                  0), fov_color, 0.5f);
      image(x % image.width(), image.height() - 1) = se::blend(image(x % image.width(), image.height() - 1), fov_color, 0.5f);
    }
#pragma omp parallel for
    for (int y = 1; y < image.height() - 1; y++) {
      image(x_min % image.width(), y) = se::blend(image(x_min % image.width(), y), fov_color, 0.5f);
      image(x_max % image.width(), y) = se::blend(image(x_max % image.width(), y), fov_color, 0.5f);
    }
  }



  int write_entropy(const std::string&  filename,
                    const Image<float>& entropy_image,
                    const SensorImpl&   sensor) {
    const Image<float> frustum_overlap_image (entropy_image.width(), entropy_image.height(), 0.0f);
    const float window_percentage = sensor.horizontal_fov / (2.0f * M_PI_F);
    const int window_width = window_percentage * entropy_image.width() + 0.5f;
    const std::vector<float> column_sums = sum_columns(entropy_image, frustum_overlap_image);
    const std::vector<float> window_sums = sum_windows(column_sums, window_width);
    const std::pair<int, float> r = max_window(entropy_image, frustum_overlap_image, window_width);

    // Open the file for writing.
    FILE* fp;
    if (!(fp = fopen(filename.c_str(), "w"))) {
      std::cerr << "Unable to write file " << filename << "\n";
      return 1;
    }
    // Write the image data
    for (int y = 0; y < entropy_image.height(); y++) {
      for (int x = 0; x < entropy_image.width(); x++) {
        fprintf(fp, "%9.3f", entropy_image(x, y));
        // Do not add a whitespace after the last element of a row.
        if (x < entropy_image.width() - 1) {
          fprintf(fp, " ");
        }
      }
      // Add a newline at the end of each row.
      fprintf(fp, "\n");
    }
    // Write the column sums
    fprintf(fp, "column-sums\n");
    for (int x = 0; x < entropy_image.width(); x++) {
      fprintf(fp, "%9.3f", column_sums[x]);
      if (x < entropy_image.width() - 1) {
        fprintf(fp, " ");
      }
    }
    fprintf(fp, "\n");
    // Write the window sums
    fprintf(fp, "window-sums-%d\n", window_width);
    for (int x = 0; x < entropy_image.width(); x++) {
      fprintf(fp, "%9.3f", window_sums[x]);
      if (x < entropy_image.width() - 1) {
        fprintf(fp, " ");
      }
    }
    fprintf(fp, "\n");
    // Write the best window
    fprintf(fp, "max-window:     %d\n", r.first);
    fprintf(fp, "max-window-sum: %.3f\n", r.second);
    fclose(fp);
    return 0;
  }
} // namespace se

