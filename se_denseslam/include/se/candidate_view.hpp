// SPDX-FileCopyrightText: 2019-2020 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2019 Anna Dai
// SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#ifndef __CANDIDATE_VIEW_HPP
#define __CANDIDATE_VIEW_HPP

#include <set>

#include <se/entropy.hpp>
#include <se/image/image.hpp>
#include <se/object_utils.hpp>
#include <se/pose_history.hpp>
#include <ptp/OccupancyWorld.hpp>
#include <ptp/PlanningParameter.hpp>
#include <ptp/ProbCollisionChecker.hpp>
#include <ptp/SafeFlightCorridorGenerator.hpp>

namespace se {
  typedef std::shared_ptr<se::Octree<VoxelImpl::VoxelType>> OctreePtr;

  typedef std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> Path;

  struct CandidateConfig {
    // Raycasting parameters
    int raycast_width = 36;
    int raycast_height = 10;
    // MAV velocity
    float velocity_linear = 1.0f;
    float velocity_angular = 0.1f;
    // Path planner parameters
    ptp::PlanningParameter planner_config
        = {"", Eigen::Vector3f::Zero(), Eigen::Vector3f::Zero(), 0.3f, 0.0f, 0.1f, 0.05f, 0.1f};

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };



  class CandidateView {
    public:
      /** \brief Create an invalid CandidateView.
       */
      CandidateView();

      CandidateView(const Eigen::Vector3f& t_MC);

      /** \brief Create a CandidateView and compute its utility.
       */
      CandidateView(const OctreePtr&              map,
                    const std::vector<se::key_t>& frontiers,
                    const Objects&                objects,
                    const SensorImpl&             sensor,
                    const Eigen::Matrix4f&        T_MC,
                    const PoseHistory&            T_MC_history,
                    const CandidateConfig&        config);

      bool isValid() const;

      /** \brief Return the path to the candidate view.
       * The first path element is the T_MC supplied to the candidate view constructor.
       */
      Path path() const;

      float utility() const;

      Eigen::Matrix4f goal() const;

      void computeIntermediateYaw(const Octree<VoxelImpl::VoxelType>& map,
                                  const SensorImpl&                   sensor,
                                  const PoseHistory&                  T_MC_history);

      Image<uint32_t> renderEntropy(const Octree<VoxelImpl::VoxelType>& map,
                                    const SensorImpl&                   sensor,
                                    const bool                          visualize_yaw = true) const;

      Image<uint32_t> renderDepth(const Octree<VoxelImpl::VoxelType>& map,
                                  const SensorImpl&                   sensor,
                                  const bool                          visualize_yaw = true) const;

      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    private:
      /** The optimal yaw angle with respect to the information gain. */
      float yaw_M_;
      /** std::vector of T_MC. */
      Path path_MC_;
      /** The length of the path in metres. */
      float path_length_;
      /** The time needed to complete the path. */
      float path_time_;
      /** An image containing the information gain produced by the 360 raycasting. */
      Image<float> entropy_image_;
      /** An image containing the percentage of frustum overlap with the candidate's neighbors. */
      Image<float> frustum_overlap_image_;
      /** An image containing the minimum integration scale for each raycasted object. */
      Image<int8_t> min_scale_image_;
      /** The information gain for the map at the optimal yaw angle. */
      float entropy_;
      /** The object level-of-detail gain at the optimal yaw angle. */
      float lod_gain_;
      /** A function of entropy and path_time. */
      float utility_;
      CandidateConfig config_;

      /** \brief Perform a 360 degree raycast and compute the optimal yaw angle.
       */
      void entropyRaycast(const Octree<VoxelImpl::VoxelType>& map,
                          const SensorImpl&                   sensor,
                          const PoseHistory&                  T_MC_history);

      void computeUtility();

      /** \brief Create a rotation matrix C_MC from a yaw angle in the Map frame.
       */
      static Eigen::Matrix3f yawToC_MC(const float yaw_M);

      /** \brief Convert a path from the format returned by ptp to that used by semanticeight.
       */
      static Path convertPath(const ptp::Path<ptp::kDim>::Ptr ptp_path);

      /** \brief Compute the time required to complete the path by moving with the constant
       * velocities provided. The yaw time takes into account the intermmediate yaw values, not just
       * the start and and yaw angles.
       */
      static float pathTime(const Path& path, float velocity_linear, float velocity_angular);
  };
} // namespace se

#endif // __CANDIDATE_VIEW_HPP

