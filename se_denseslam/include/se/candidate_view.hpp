// SPDX-FileCopyrightText: 2019-2020 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2019 Anna Dai
// SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#ifndef __CANDIDATE_VIEW_HPP
#define __CANDIDATE_VIEW_HPP

#include <ptp/SafeFlightCorridorGenerator.hpp>
#include <se/entropy.hpp>
#include <se/image/image.hpp>
#include <se/object_utils.hpp>
#include <se/path.hpp>
#include <set>

namespace se {

struct CandidateConfig {
    // Utility settings
    /** The weights correspond to:
     * 0. Entropy gain
     * 1. Object distance gain
     * 2. Background distance gain
     */
    Eigen::VectorXf utility_weights;
    bool use_pose_history;
    // Raycasting parameters
    int raycast_width;
    int raycast_height;
    // MAV velocity
    float delta_t;
    float velocity_linear;
    float velocity_angular;
    // Goal settings
    float goal_xy_threshold;
    float goal_z_threshold;
    float goal_roll_pitch_threshold;
    float goal_yaw_threshold;
    // Path planner parameters
    ptp::PlanningParameter planner_config;

    CandidateConfig(const Configuration& c = Configuration(),
                    const Eigen::Matrix4f& T_MW = Eigen::Matrix4f::Identity());

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};



class CandidateView {
    public:
    /** \brief Create an invalid CandidateView.
     */
    CandidateView(const se::Octree<VoxelImpl::VoxelType>& map,
                  const Objects& objects,
                  const SensorImpl& sensor,
                  const Eigen::Matrix4f& T_BC);

    /** \brief Create a CandidateView and compute its utility.
     * \warning The map frame (M) MUST be z-up.
     */
    CandidateView(const se::Octree<VoxelImpl::VoxelType>& map,
                  ptp::SafeFlightCorridorGenerator& planner,
                  const std::vector<se::key_t>& frontiers,
                  const Objects& objects,
                  const SensorImpl& sensor,
                  const Eigen::Matrix4f& T_MB,
                  const Eigen::Matrix4f& T_BC,
                  const PoseHistory* T_MB_history,
                  const CandidateConfig& config);

    bool isValid() const;

    /** \brief Return the path to the candidate view.
     * The first path element is the T_MB supplied to the candidate view constructor.
     */
    const Path& path() const;

    float utility() const;

    float entropyUtility() const;

    float objectDistUtility() const;

    float bgDistUtility() const;

    std::string utilityStr() const;

    const Eigen::Matrix4f& goalT_MB() const;

    void computeIntermediateYaw(const PoseHistory* T_MB_history);

    Image<uint32_t> renderEntropy(const bool visualize_yaw = true) const;

    Image<uint32_t> renderBGDistGain(const bool visualize_yaw = true) const;

    Image<uint32_t> renderObjectDistGain(const bool visualize_yaw = true) const;

    Image<uint32_t> renderDepth(const bool visualize_yaw = true) const;

    void renderCurrentEntropyDepth(Image<uint32_t>& entropy,
                                   Image<uint32_t>& depth,
                                   const bool visualize_yaw = true) const;

    Image<Eigen::Vector3f> rays() const;

    bool writeEntropyData(const std::string& filename) const;

    /** ICRA 2020
     */
    static Path addIntermediateTranslation(const Eigen::Matrix4f& segment_start_M,
                                           const Eigen::Matrix4f& segment_end_M,
                                           float delta_t,
                                           float velocity_linear,
                                           float resolution);

    /** ICRA 2020
     */
    static Path addIntermediateYaw(const Eigen::Matrix4f& segment_start_M,
                                   const Eigen::Matrix4f& segment_end_M,
                                   float delta_t,
                                   float velocity_angular);

    /** ICRA 2020
     */
    static Path fuseIntermediatePaths(const Path& intermediate_translation,
                                      const Path& intermediate_yaw);

    /** ICRA 2020
     */
    static Path getFinalPath(const Path& path_M,
                             float delta_t,
                             float velocity_linear,
                             float velocity_angular,
                             float resolution);

    static Image<float> computeGainImage(const ImageVec<float>& gain_images,
                                         const Eigen::VectorXf& weights);

    /** The position initially desired for the candidate. The actual position can be different due
     * to partial path planning.
     */
    Eigen::Vector3f desired_t_MB_;
    /** The optimal yaw angle with respect to the information gain. */
    float yaw_M_;
    /** The index of the left edge of the optimal yaw window. */
    int window_idx_;
    /** The width of the optimal yaw window in pixels. */
    int window_width_;
    /** The time needed to complete the path. */
    float path_time_;
    /** The combined gain at the optimal yaw angle. */
    float gain_;
    /** The information gain for the map at the optimal yaw angle. */
    float entropy_gain_;
    /** The object distance gain at the optimal yaw angle. */
    float object_dist_gain_;
    /** The background distance gain at the optimal yaw angle. */
    float bg_dist_gain_;
    /** An image containing the combined gain produced by the 360 raycasting. It's a weighted sum
     * of entropy and level-of-detail gains. The yaw optimization is performed on this image.
     */
    Image<float> gain_image_;
    /** An image containing the information gain produced by the 360 raycasting. */
    Image<float> entropy_image_;
    /** An image containing the object distance gain produced by the 360 raycasting. */
    Image<float> object_dist_gain_image_;
    /** An image containing the background distance gain produced by the 360 raycasting. */
    Image<float> bg_dist_gain_image_;
    /** An image containing the points where entropy raycasting hit. */
    Image<Eigen::Vector3f> entropy_hits_M_;
    /** An image containing the per-pixel mask of frustum overlap with the candidate's neighbors. */
    Image<uint8_t> frustum_overlap_mask_;
    std::string utility_str_;
    std::string status_;
    const SensorImpl& sensor_;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    private:
    const se::Octree<VoxelImpl::VoxelType>& map_;
    const Objects& objects_;
    const Eigen::Matrix4f& T_BC_;
    /** std::vector of T_MB. */
    Path path_MB_;
    /** A function of entropy and path_time. */
    float utility_;
    float exploration_utility_;
    float object_dist_utility_;
    float bg_dist_utility_;
    CandidateConfig config_;
    Eigen::VectorXf weights_;

    /** \brief Perform a 360 degree raycast and compute the optimal yaw angle.
     */
    void entropyRaycast(const PoseHistory* T_MB_history);

    void computeUtility();

    /** \brief Create a rotation matrix C_MB from a yaw angle in the Map frame.
     */
    static Eigen::Matrix3f yawToC_MB(const float yaw_M);

    /** \brief Convert a path from the format returned by ptp to that used by semanticeight.
     */
    static Path convertPath(const ptp::Path<ptp::kDim>::Ptr ptp_path);

    /** \brief Change the positions of path vertices that are within radius of the first vertex to
     * the position of the first vertex.
     */
    static void removeSmallMovements(Path& path, const float radius);

    /** \brief Add intermmediate path waypoints so that yaw is performed before moving to the next
     * waypoint. This is used to avoid aggressive MAV flying.
     */
    static void yawBeforeMoving(Path& path);

    static void yawWhileMoving(Path& path, float velocity_linear, float velocity_angular);

    /** \brief Zero-out the roll and pitch angles for all poses in the path.
     */
    static void zeroRollPitch(Path& path_MB);

    /** \brief Compute the time required to complete the path by moving with the constant
     * velocities provided. The yaw time takes into account the intermmediate yaw values, not just
     * the start and and yaw angles.
     */
    static float pathTime(const Path& path, float velocity_linear, float velocity_angular);
};

std::ostream& operator<<(std::ostream& os, const CandidateView& c);

} // namespace se

#endif // __CANDIDATE_VIEW_HPP
