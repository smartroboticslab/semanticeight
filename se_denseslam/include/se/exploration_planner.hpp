// SPDX-FileCopyrightText: 2021 Smart Robotics Lab
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#ifndef __EXPLORATION_PLANNER_HPP
#define __EXPLORATION_PLANNER_HPP

#include <mutex>

#include "se/DenseSLAMSystem.h"
#include "se/single_path_exploration_planner.hpp"

namespace se {
/** The API uses Body poses in the World frame but internally everything is performed in the Map
 * frame. Body poses are used for sampling and path planning and they are converted to camera
 * poses for raycasting.
 *
 * The ExplorationPlanner keeps a copy of the path internally and removes vertices whenever
 * goalReached() succeeds.
 *
 * \warning The map frame (M) MUST be z-up.
 */
class ExplorationPlanner {
    public:
    ExplorationPlanner(const DenseSLAMSystem& pipeline,
                       const SensorImpl& sensor,
                       const se::Configuration& config);

    void recordT_WB(const Eigen::Matrix4f& T_WB, const se::Image<float>& depth);

    Eigen::Matrix4f getT_WB() const;

    Path getT_WBHistory() const;

    bool needsNewGoal() const;

    math::PoseError goalCandidateError(const Eigen::Matrix4f& T_WB) const;

    bool inGoalCandidateThreshold(const Eigen::Matrix4f& T_WB) const;

    bool goalReached();

    bool goalT_WB(Eigen::Matrix4f& T_WB) const;

    /** This is only added as a hack to get the ICRA 2020 exploration to work.
     */
    void popGoalT_WB();

    /** Call the exploration planner and return the resulting camera path in the world frame.
     * The returned path is a series of T_WB. The first T_WB is the same as start_T_WB.
     */
    Path computeNextPath_WB(const std::set<key_t>& frontiers,
                            const Objects& objects,
                            const Eigen::Matrix4f& start_T_WB);

    const std::vector<CandidateView>& candidateViews() const;

    const std::vector<CandidateView>& rejectedCandidateViews() const;

    const CandidateView& goalView() const;

    size_t goalViewIndex() const;

    void renderCurrentEntropyDepth(Image<uint32_t>& entropy,
                                   Image<uint32_t>& depth,
                                   const bool visualize_yaw = true);

    const PoseGridHistory& getPoseGridHistory() const;

    const PoseMaskHistory& getPoseMaskHistory() const;

    /** Return the edge vertices of the sampling AABB in the map frame in an order that can be
     * directly passed to rviz as a LINE_LIST.
     */
    const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>&
    samplingAABBEdgesM() const;

    /** Write the T_WB history as a PLY file. Optionally transform from the world frame W to some
     * other frame F.
     */
    int writePathPLY(const std::string& filename,
                     const Eigen::Matrix4f& T_FW = Eigen::Matrix4f::Identity()) const;

    /** Write the T_WB history as a TSV file. Optionally transform from the world frame W to some
     * other frame F.
     */
    int writePathTSV(const std::string& filename,
                     const Eigen::Matrix4f& T_FW = Eigen::Matrix4f::Identity()) const;

    /** \brief Return true if the utility computation with and without the object gain resulted in
     * the same best candidate in the last planning iteration.
     */
    bool explorationDominant() const;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
    typedef std::queue<Eigen::Matrix4f,
                       std::deque<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>>
        PathQueue;

    mutable std::recursive_mutex mutex_;
    const ExplorationConfig config_;
    const OctreeConstPtr map_;
    const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>
        sampling_aabb_edges_M_;
    const Eigen::Matrix4f T_MW_;
    const Eigen::Matrix4f T_WM_;
    const Eigen::Matrix4f T_BC_;
    const Eigen::Matrix4f T_CB_;
    const SensorImpl sensor_;
    // History of fusion poses.
    PoseGridHistory T_MB_grid_history_;
    PoseMaskHistory T_MB_mask_history_;
    PoseVectorHistory T_MB_history_;
    std::vector<CandidateView> candidate_views_;
    std::vector<CandidateView> rejected_candidate_views_;
    size_t goal_view_idx_;
    PathQueue goal_path_T_MB_;
    bool exploration_dominant_;

    bool withinThreshold(const math::PoseError& error) const;
};
} // namespace se

#endif // __EXPLORATION_PLANNER_HPP
