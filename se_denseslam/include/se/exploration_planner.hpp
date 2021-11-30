// SPDX-FileCopyrightText: 2021 Smart Robotics Lab
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#ifndef __EXPLORATION_PLANNER_HPP
#define __EXPLORATION_PLANNER_HPP

#include "se/single_path_exploration_planner.hpp"

namespace se {
/** The API uses Body poses in the World frame but internally everything is performed in the Map
   * frame. Body poses are used for sampling and path planning and they are converted to camera
   * poses for raycasting.
   *
   * The ExplorationPlanner keeps a copy of the path internally and removes vertices whenever
   * goalReached() succeeds.
   */
class ExplorationPlanner {
    public:
    ExplorationPlanner(const OctreePtr map,
                       const Eigen::Matrix4f& T_MW,
                       const Eigen::Matrix4f& T_BC,
                       const ExplorationConfig& config);

    void setT_WB(const Eigen::Matrix4f& T_WB);

    void setPlanningT_WB(const Eigen::Matrix4f& T_WB);

    Eigen::Matrix4f getT_WB() const;

    Eigen::Matrix4f getPlanningT_WB() const;

    Path getT_WBHistory() const;

    bool needsNewGoal() const;

    bool goalReached();

    bool goalT_WB(Eigen::Matrix4f& T_WB) const;

    /** This is only added as a hack to get the ICRA 2020 exploration to work.
       */
    void popGoalT_WB();

    /** Call the exploration planner and return the resulting camera path in the world frame.
       * The returned path is a series of T_WB.
       */
    Path computeNextPath_WB(const std::set<key_t>& frontiers,
                            const Objects& objects,
                            const SensorImpl& sensor);

    const std::vector<CandidateView>& candidateViews() const;

    const std::vector<CandidateView>& rejectedCandidateViews() const;

    const CandidateView& goalView() const;

    size_t goalViewIndex() const;

    Image<uint32_t> renderCurrentEntropy(const SensorImpl& sensor, const bool visualize_yaw = true);

    Image<uint32_t> renderCurrentEntropyDepth(const SensorImpl& sensor,
                                              const bool visualize_yaw = true);

    Image<uint32_t> renderMinScale(const SensorImpl& sensor);

    const PoseGridHistory& getPoseGridHistory() const;

    /** Write the T_WB history as a PLY file.
       */
    int writePathPLY(const std::string& filename) const;

    /** Write the T_WB history as a TSV file.
       */
    int writePathTSV(const std::string& filename) const;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
    typedef std::queue<Eigen::Matrix4f,
                       std::deque<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>>
        PathQueue;

    const OctreePtr map_;
    const ExplorationConfig config_;
    Eigen::Matrix4f T_MW_;
    Eigen::Matrix4f T_WM_;
    Eigen::Matrix4f T_BC_;
    Eigen::Matrix4f T_CB_;
    // The pose planning the next path will start from.
    Eigen::Matrix4f planning_T_MB_;
    // History of fusion poses.
    PoseGridHistory T_MB_grid_history_;
    PoseVectorHistory T_MB_history_;
    std::vector<CandidateView> candidate_views_;
    std::vector<CandidateView> rejected_candidate_views_;
    CandidateView goal_view_;
    size_t goal_view_idx_;
    PathQueue goal_path_T_MB_;

    static constexpr float goal_xy_threshold_ = 0.2f;
    static constexpr float goal_z_threshold_ = 0.2f;
    static constexpr float goal_yaw_threshold_ = math::deg_to_rad * 5.0f;
    static constexpr float goal_roll_pitch_threshold_ = math::deg_to_rad * 10.0f;
};
} // namespace se

#endif // __EXPLORATION_PLANNER_HPP
