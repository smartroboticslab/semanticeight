/**
 * Probabilistic Trajectory Planning Probabilistc Collision Checker.
 *
 * Copyright (C) 2018 Imperial College London.
 * Copyright (C) 2018 ETH Zürich.
 *
 * @todo LICENSE
 *
 * @file file PlanningParameter.hpp
 * @author Nils Funk
 * @date June, 2018
 */

#include <ptp/ProbCollisionChecker.hpp>

#include "se/str_utils.hpp"


namespace ptp {

ProbCollisionChecker::ProbCollisionChecker(const ptp::OccupancyWorld& ow,
                                           const PlanningParameter& pp) :
        ow_(ow), pp_(pp), res_(ow.getMap().voxelDim())
{
}

bool ProbCollisionChecker::checkData(const MultiresOFusion::VoxelType::VoxelData& data,
                                     const bool finest) const
{
    if (data.x * data.y < free_threshold_ && (data.observed || (finest && data.y > 0))) {
        return true;
    }
    return false;
}



bool ProbCollisionChecker::checkPoint(const Eigen::Vector3f& point_M) const
{
    return ow_.isFreeAtPoint(point_M, free_threshold_);
}



bool ProbCollisionChecker::checkLine(const Eigen::Vector3f& start_point_M,
                                     const Eigen::Vector3f& connection_m,
                                     const int num_subpos) const
{
    Eigen::Vector3f point_M = start_point_M;
    if (!checkPoint(point_M)) {
        return false;
    }

    point_M = start_point_M + connection_m;
    if (!checkPoint(point_M)) {
        return false;
    }

    if (num_subpos > 0) {
        std::queue<std::pair<int, int>> line_pos;
        line_pos.push(std::make_pair(1, num_subpos));

        // Repeatedly subdivide the path segment in the middle (and check the middle)
        while (!line_pos.empty()) {
            std::pair<int, int> x = line_pos.front();

            int mid = (x.first + x.second) / 2;

            // Compute midpoint
            point_M = ((float) mid / (float) (num_subpos + 1) * connection_m + start_point_M);

            if (!checkPoint(point_M)) {
                return false;
            }

            line_pos.pop();

            if (x.first < mid)
                line_pos.push(std::make_pair(x.first, mid - 1));
            if (x.second > mid)
                line_pos.push(std::make_pair(mid + 1, x.second));
        }
    }

    return true;
}

static const Eigen::Matrix<float, 3, 6> sphere_h_v_sceleton_offset =
    (Eigen::Matrix<float, 3, 6>() << -1, 0, 0, 1, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, -1, 0, 0, -1)
        .finished();

static const Eigen::Matrix<float, 3, 8> sphere_d_sceleton_offset = 0.577
    * (Eigen::Matrix<float, 3, 8>() << -1,
       +1,
       -1,
       +1,
       -1,
       +1,
       -1,
       +1,
       -1,
       -1,
       +1,
       +1,
       -1,
       -1,
       +1,
       +1,
       -1,
       -1,
       -1,
       -1,
       +1,
       +1,
       +1,
       +1)
          .finished();


bool ProbCollisionChecker::checkSphereSkeleton(const Eigen::Vector3f& point_M,
                                               const float radius_m) const
{
    //=======================//
    // CHECK SPHERE SKELETON //
    //=======================//

    // Centre
    if (!checkPoint(point_M)) {
        return false;
    }

    // Horizontal and vertical
    for (int i = 0; i < 6; i++) {
        Eigen::Vector3f point_hv_M = point_M + radius_m * sphere_h_v_sceleton_offset.col(i);
        if (!checkPoint(point_hv_M)) {
            return false;
        }
    }
    // Diagonal
    for (int i = 0; i < 8; i++) {
        Eigen::Vector3f point_d_M = point_M + radius_m * sphere_d_sceleton_offset.col(i);
        if (!checkPoint(point_d_M)) {
            return false;
        }
    }

    return true;
}



bool ProbCollisionChecker::checkCylinderSkeleton(const Eigen::Vector3f& start_point_M,
                                                 const Eigen::Vector3f& end_point_M,
                                                 const float radius_m) const
{
    //=========================//
    // CHECK CYLINDER SKELETON //
    //=========================//

    float seg_prec =
        pp_.skeleton_sample_precision_; // Precision at which to sample the connection [m/voxel]
    Eigen::Vector3f start_cylinder_M =
        start_point_M; ///< Sphere check doesn't need to be so accurate as we extend the cylinder.
    Eigen::Vector3f end_cylinder_M = end_point_M;

    Eigen::Vector3f vec_seg_connection_M = end_cylinder_M
        - start_cylinder_M; // The vector in [m] connecting the start and end position
    Eigen::Vector3f vec_seg_direction_u =
        vec_seg_connection_M;        // The vector in [m] connecting the start and end position
    vec_seg_direction_u.normalize(); // The vector in [m] connecting the start and end position
    int num_axial_subpos =
        vec_seg_connection_M.norm() / seg_prec; // Number of sub points along the line to be checked

    if (!checkLine(start_cylinder_M,
                   vec_seg_connection_M,
                   num_axial_subpos)) { // Check if the line-of-sight connection is collision free
        return false;
    }

    /**
     * Get cylinder extrema in horizontal and vertical direction
     */
    Eigen::Vector3f vec_z_u = Eigen::Vector3f(0, 0, 1);
    Eigen::Vector3f vec_vertical_u = vec_seg_direction_u.cross(vec_z_u);
    Eigen::Vector3f vec_horizontal_u;
    if (vec_vertical_u == Eigen::Vector3f(0, 0, 0)) {
        vec_vertical_u = Eigen::Vector3f(1, 0, 0);
        vec_horizontal_u = Eigen::Vector3f(0, 1, 0);
    }
    else {
        vec_vertical_u.normalize();
        vec_horizontal_u = vec_seg_direction_u.cross(vec_vertical_u);
        vec_horizontal_u.normalize();
    }

    std::vector<Eigen::Vector3f> shell_hv_pos;
    shell_hv_pos = {start_cylinder_M + vec_horizontal_u * radius_m,
                    start_cylinder_M - vec_horizontal_u * radius_m,
                    start_cylinder_M + vec_vertical_u * radius_m,
                    start_cylinder_M - vec_vertical_u * radius_m};
    for (std::vector<Eigen::Vector3f>::iterator it = shell_hv_pos.begin(); it != shell_hv_pos.end();
         ++it) {
        if (!checkLine(*it, vec_seg_connection_M, num_axial_subpos)) {
            return false;
        }
    }

    Eigen::Vector3f vec_diag_1_u = (vec_horizontal_u + vec_vertical_u);
    vec_diag_1_u.normalize();
    Eigen::Vector3f vec_diag_2_u = (vec_horizontal_u - vec_vertical_u);
    vec_diag_2_u.normalize();

    std::vector<Eigen::Vector3f> shell_diag_pos;
    shell_diag_pos = {start_cylinder_M + vec_diag_1_u * radius_m,
                      start_cylinder_M - vec_diag_1_u * radius_m,
                      start_cylinder_M + vec_diag_2_u * radius_m,
                      start_cylinder_M - vec_diag_2_u * radius_m};
    for (std::vector<Eigen::Vector3f>::iterator it = shell_diag_pos.begin();
         it != shell_diag_pos.end();
         ++it) {
        if (!checkLine(*it, vec_seg_connection_M, num_axial_subpos)) {
            return false;
        }
    }

    return true;
}



bool ProbCollisionChecker::checkCorridorSkeleton(const Eigen::Vector3f& start_point_M,
                                                 const Eigen::Vector3f& end_point_M,
                                                 const float radius_m) const
{
    //=========================//
    // CHECK CORRIDOR SKELETON //
    //=========================//

    float seg_prec =
        pp_.skeleton_sample_precision_; // Precision at which to sample the connection [m/voxel]

    Eigen::Vector3f corridor_axis_u = (end_point_M - start_point_M);
    corridor_axis_u.normalize();

    Eigen::Vector3f start_corridor_mid_M = start_point_M
        - radius_m
            * corridor_axis_u; ///< Sphere check doesn't need to be so accurate as we extend the cylinder.
    Eigen::Vector3f end_corridor_mid_M = end_point_M + radius_m * corridor_axis_u;
    Eigen::Vector3f vec_seg_connection_mid_M = end_corridor_mid_M
        - start_corridor_mid_M; // The vector in [m] connecting the start and end position
    Eigen::Vector3f vec_seg_direction_u =
        vec_seg_connection_mid_M; // The vector in [m] connecting the start and end position
    vec_seg_direction_u.normalize();
    int num_axial_mid_subpos = vec_seg_connection_mid_M.norm()
        / seg_prec; // Number of sub points along the line to be checked

    if (!checkLine(
            start_corridor_mid_M,
            vec_seg_connection_mid_M,
            num_axial_mid_subpos)) { // Check if the line-of-sight connection is collision free
        return false;
    }

    /**
     * Get cylinder extrema in horizontal and vertical direction
     */
    Eigen::Vector3f vec_z_u = Eigen::Vector3f(0, 0, 1);
    Eigen::Vector3f vec_vertical_u = vec_seg_direction_u.cross(vec_z_u);
    Eigen::Vector3f vec_horizontal_u;
    if (vec_vertical_u == Eigen::Vector3f(0, 0, 0)) {
        vec_vertical_u = Eigen::Vector3f(1, 0, 0);
        vec_horizontal_u = Eigen::Vector3f(0, 1, 0);
    }
    else {
        vec_vertical_u.normalize();
        vec_horizontal_u = vec_seg_direction_u.cross(vec_vertical_u);
        vec_horizontal_u.normalize();
    }

    Eigen::Vector3f vec_seg_connection_outer_M =
        end_point_M - start_point_M; // The vector in [m] connecting the start and end position
    int num_axial_outer_subpos = vec_seg_connection_outer_M.norm()
        / seg_prec; // Number of sub points along the line to be checked

    std::vector<Eigen::Vector3f> shell_hv_pos;
    shell_hv_pos = {start_point_M + vec_horizontal_u * radius_m,
                    start_point_M - vec_horizontal_u * radius_m,
                    start_point_M + vec_vertical_u * radius_m,
                    start_point_M - vec_vertical_u * radius_m};
    for (std::vector<Eigen::Vector3f>::iterator it = shell_hv_pos.begin(); it != shell_hv_pos.end();
         ++it) {
        if (!checkLine(*it, vec_seg_connection_outer_M, num_axial_outer_subpos)) {
            return false;
        }
    }

    Eigen::Vector3f vec_diag_1_u = (vec_horizontal_u + vec_vertical_u);
    vec_diag_1_u.normalize();
    Eigen::Vector3f vec_diag_2_u = (vec_horizontal_u - vec_vertical_u);
    vec_diag_2_u.normalize();

    std::vector<Eigen::Vector3f> shell_diag_pos;
    shell_diag_pos = {start_point_M + vec_diag_1_u * radius_m,
                      start_point_M - vec_diag_1_u * radius_m,
                      start_point_M + vec_diag_2_u * radius_m,
                      start_point_M - vec_diag_2_u * radius_m};
    for (std::vector<Eigen::Vector3f>::iterator it = shell_diag_pos.begin();
         it != shell_diag_pos.end();
         ++it) {
        if (!checkLine(*it, vec_seg_connection_outer_M, num_axial_outer_subpos)) {
            return false;
        }
    }

    return true;
}



bool ProbCollisionChecker::checkNode(const Eigen::Vector3i& node_corner,
                                     const int node_size,
                                     bool& abort) const
{
    MultiresOFusion::VoxelType::VoxelData volume_data;
    Eigen::Vector3i volume_corner = Eigen::Vector3i(0, 0, 0);
    int volume_size = 0;
    bool is_finest = false;

    ow_.getMap().getThreshold(node_corner.x(),
                              node_corner.y(),
                              node_corner.z(),
                              free_threshold_,
                              node_size,
                              volume_data,
                              volume_size,
                              volume_corner,
                              is_finest);

    if (!volume_data.observed || volume_data.x * volume_data.y >= free_threshold_) {
        abort = (is_finest || (node_size == 1));
        return false;
    }

    return true;
}



bool ProbCollisionChecker::checkNode(const Eigen::Vector3i& node_corner, const int node_size) const
{
    MultiresOFusion::VoxelType::VoxelData volume_data;
    int volume_size = 0;
    Eigen::Vector3i volume_corner = Eigen::Vector3i(0, 0, 0);
    bool is_finest = false;

    ow_.getMap().getThreshold(node_corner.x(),
                              node_corner.y(),
                              node_corner.z(),
                              free_threshold_,
                              node_size,
                              volume_data,
                              volume_size,
                              volume_corner,
                              is_finest);

    if (!volume_data.observed || volume_data.x * volume_data.y >= free_threshold_) {
        return false;
    }

    return true;
}



bool ProbCollisionChecker::checkInSphereFree(const Eigen::Vector3i& node_corner_min,
                                             const int node_size,
                                             const Eigen::Vector3f& point_M,
                                             const float radius_m) const
{
    for (int i = 0; i < 8; i++) {
        Eigen::Vector3i node_corner = node_corner_min
            + Eigen::Vector3i::Constant(node_size).cwiseProduct(
                Eigen::Vector3i((i & 1) > 0, (i & 2) > 0, (i & 4) > 0));
        Eigen::Vector3i node_center = node_corner + Eigen::Vector3i::Constant(node_size / 2);
        if ((node_center.cast<float>() * res_ - point_M).norm()
            > radius_m + res_ * node_size * sqrt(3) / 2) {
            continue;
        }

        bool abort = false;
        if (checkNode(node_corner, node_size, abort)) {
            continue;
        }

        if (abort) {
            return false;
        }

        int num_inside = 0;
        std::vector<Eigen::Vector3i> test_positions;
        for (int j = 0; j < 8; j++) {
            Eigen::Vector3i sub_node_corner = node_corner
                + Eigen::Vector3i::Constant(node_size).cwiseProduct(
                    Eigen::Vector3i((j & 1) > 0, (j & 2) > 0, (j & 4) > 0));
            if ((sub_node_corner.cast<float>() * res_ - point_M).norm() < radius_m) {
                test_positions.push_back(sub_node_corner);
                num_inside++;
            }
        }

        if (num_inside == 8) {
            return false;
        }

        for (auto const& test_position : test_positions) {
            if (!ow_.isFree(test_position, free_threshold_)) {
                return false;
            }
        }

        if (!checkInSphereFree(node_corner, node_size / 2, point_M, radius_m)) {
            return false;
        }
    }
    return true;
}



bool ProbCollisionChecker::checkNodeInSphereFree(
    const se::Node<MultiresOFusion::VoxelType>* parent_node,
    const Eigen::Vector3i& parent_coord,
    const int node_size,
    const Eigen::Vector3f& point_M,
    const float radius_m) const
{
    for (int i = 0; i < 8; i++) {
        if (checkData(parent_node->childData(i))) { //< If free continue;
            continue;
        }

        const Eigen::Vector3i node_rel_step = Eigen::Vector3i::Constant(node_size).cwiseProduct(
            Eigen::Vector3i((i & 1) > 0, (i & 2) > 0, (i & 4) > 0));
        const Eigen::Vector3i node_coord = parent_coord + node_rel_step;
        const Eigen::Vector3f node_centre =
            node_coord.cast<float>() + Eigen::Vector3f::Constant((float) node_size / 2);
        if ((node_centre.cast<float>() * res_ - point_M).norm()
            > radius_m + res_ * node_size * sqrt(3) / 2) {
            continue;
        }

        const se::Node<MultiresOFusion::VoxelType>* node = parent_node->child(i);
        if (!node) {
            return false;
        }


        // TODO: Try without!
        int num_inside = 0;
        std::vector<Eigen::Vector3i> test_positions;
        for (int j = 0; j < 8; j++) {
            Eigen::Vector3i node_corner_offset = Eigen::Vector3i::Constant(node_size).cwiseProduct(
                Eigen::Vector3i((j & 1) > 0, (j & 2) > 0, (j & 4) > 0));
            Eigen::Vector3i node_corner = node_coord + node_corner_offset;
            if ((node_centre.cast<float>() * res_ - point_M).norm() > radius_m
                    + res_ * node_size * sqrt(3)
                        / 2) { ///< Check if all 8 corners are inside the cyclinder, causing guaranteed collision.
                num_inside++;
                test_positions.push_back(node_corner);
            }
        }

        if (num_inside
            == 8) { ///< Corridor is guaranteed to be in collision if node is fully contained.
            return false;
        }

        // TODO: Try without!
        for (auto const& test_position : test_positions) {
            if (!ow_.isFree(test_position, free_threshold_)) ///< Check if one of the contained corners is occupied.
                return false;
        }

        if (node->isBlock()) {
            const se::VoxelBlockSingleMax<MultiresOFusion::VoxelType>* block =
                static_cast<const se::VoxelBlockSingleMax<MultiresOFusion::VoxelType>*>(node);
            if (!checkBlockInSphereFree(block,
                                        node_coord,
                                        node_size / 2,
                                        point_M,
                                        radius_m)) { ///< Continue the search at a finer resolution
                return false;
            }
        }
        else {
            if (!checkNodeInSphereFree(node,
                                       node_coord,
                                       node_size / 2,
                                       point_M,
                                       radius_m)) { ///< Continue the search at a finer resolution
                return false;
            }
        }
    }
    return true;
}



bool ProbCollisionChecker::checkBlockInSphereFree(
    const se::VoxelBlockSingleMax<MultiresOFusion::VoxelType>* block,
    const Eigen::Vector3i& parent_coord,
    const int node_size,
    const Eigen::Vector3f& point_M,
    const float radius_m) const
{
    for (int i = 0; i < 8; i++) {
        const Eigen::Vector3i node_rel_step = Eigen::Vector3i::Constant(node_size).cwiseProduct(
            Eigen::Vector3i((i & 1) > 0, (i & 2) > 0, (i & 4) > 0));
        const Eigen::Vector3i node_coord = parent_coord + node_rel_step;
        const int scale = se::math::log2_const(node_size);

        if (checkData(block->maxData(node_coord, scale),
                      (scale == block->current_scale()))) { ///< If free continue
            continue;
        }


        const Eigen::Vector3f node_centre =
            node_coord.cast<float>() + Eigen::Vector3f::Constant((float) node_size / 2);
        if ((node_centre.cast<float>() * res_ - point_M).norm()
            > radius_m + res_ * node_size * sqrt(3) / 2) {
            continue;
        }

        if (scale == block->current_scale()) {
            return false;
        }


        // TODO: Try without!
        int num_inside = 0;
        std::vector<Eigen::Vector3i> test_positions;
        for (int j = 0; j < 8; j++) {
            Eigen::Vector3i node_corner_offset = Eigen::Vector3i::Constant(node_size).cwiseProduct(
                Eigen::Vector3i((j & 1) > 0, (j & 2) > 0, (j & 4) > 0));
            Eigen::Vector3i node_corner = node_coord + node_corner_offset;
            if ((node_corner.cast<float>() * res_ - point_M).norm()
                < radius_m) { ///< Check if all 8 corners are inside the cyclinder, causing guaranteed collision.
                num_inside++;
                test_positions.push_back(node_corner);
            }
        }

        if (num_inside
            == 8) { ///< Corridor is guaranteed to be in collision if node is fully contained.
            return false;
        }

        // TODO: Try without!
        for (auto const& test_position : test_positions) {
            if (!ow_.isFree(test_position, free_threshold_)) ///< Check if one of the contained corners is occupied.
                return false;
        }

        if (!checkBlockInSphereFree(block,
                                    node_coord,
                                    node_size / 2,
                                    point_M,
                                    radius_m)) { ///< Continue the search at a finer resolution
            return false;
        }
    }
    return true;
}



//

bool ProbCollisionChecker::checkSphere(const Eigen::Vector3f& point_M, const float radius_m) const
{
    //====================//
    // CHECK SPHERE DENSE //
    //====================//

    float radius_v = radius_m / res_;
    Eigen::Vector3i position_v = (point_M / res_).cast<int>();
    Eigen::Vector3i bb_corner_min = position_v - Eigen::Vector3i::Constant(radius_v);
    int node_size = 1 << (int) ceil(log2(radius_v));

    for (int i = 0; i < 8; i++) {
        Eigen::Vector3i bb_corner = bb_corner_min
            + Eigen::Vector3i::Constant(2 * radius_v)
                  .cwiseProduct(Eigen::Vector3i((i & 1) > 0, (i & 2) > 0, (i & 4) > 0));

        if (!checkNode(bb_corner, node_size)) {
            Eigen::Vector3i node_coord = node_size * (bb_corner / node_size);

            const se::Node<MultiresOFusion::VoxelType>* node = ow_.getNode(node_coord, node_size);
            if (!node) {
                return false;
            }

            if (node->isBlock()) {
                const se::VoxelBlockSingleMax<MultiresOFusion::VoxelType>* block =
                    static_cast<const se::VoxelBlockSingleMax<MultiresOFusion::VoxelType>*>(node);
                const int voxel_size = std::max(1 << block->current_scale(), node_size / 2);
                if (!checkBlockInSphereFree(block, node_coord, voxel_size, point_M, radius_m)) {
                    return false;
                }
            }
            else {
                if (!checkNodeInSphereFree(node, node_coord, node_size / 2, point_M, radius_m)) {
                    return false;
                }
            }
        }
    }
    return true; ///<< Sphere is free
}



/**
   * \brief The check approximates the node with a sphere and checks if the sphere would intersect with the cylinder
   *
   * \note The check is conservative and causes especially false positives for large node sizes. However the only consequence
   *       is a neglectable increase in checks.
   */
bool ProbCollisionChecker::checkCenterInCylinder(const Eigen::Vector3f& center_v,
                                                 const int node_size,
                                                 const Eigen::Vector3f& start_point_M,
                                                 const Eigen::Vector3f& end_point_M,
                                                 const Eigen::Vector3f& axis,
                                                 const float radius_m) const
{
    //    /**
    //     * Calculates the euclidean distance from a point to a line segment.
    //     *
    //     * @param v     the point
    //     * @param a     start of line segment
    //     * @param b     end of line segment
    //     * @return      distance from v to line segment [a,b]
    //     *
    //     * @author      Afonso Santos
    //     */
    //    public static
    //    float
    //    distanceToSegment( final R3 v, final R3 a, final R3 b )
    //    {
    //      final R3 ab  = b.sub( a ) ;
    //      final R3 av  = v.sub( a ) ;
    //
    //      if (av.dot(ab) <= 0.0)           // Point is lagging behind start of the segment, so perpendicular distance is not viable.
    //        return av.modulus( ) ;         // Use distance to start of segment instead.
    //
    //      final R3 bv  = v.sub( b ) ;
    //
    //      if (bv.dot(ab) >= 0.0)           // Point is advanced past the end of the segment, so perpendicular distance is not viable.
    //        return bv.modulus( ) ;         // Use distance to end of the segment instead.
    //
    //      return (ab.cross( av )).modulus() / ab.modulus() ;       // Perpendicular distance of point to segment.
    //    }

    Eigen::Vector3f center_m = center_v * res_;
    float extension = node_size * res_ * sqrt(3);
    Eigen::Vector3f center_start_point_M = center_m - (start_point_M - extension * axis);

    // Behind start layer
    if (center_start_point_M.dot(axis) < 0)
        return false;

    Eigen::Vector3f center_end_point_M = center_m - (end_point_M + extension * axis);

    // Infront end layer
    if (center_end_point_M.dot(axis) > 0)
        return false;

    // Compute point line distance
    float dist = (axis.cross(center_start_point_M)).norm();
    if (dist > (radius_m + extension))
        return false;
    return true;
}


/**
   * \brief Check if one of the eight node corners is contained in the cyclinder.
   *
   * \warning This check is dangerous if node is quite big in relation to the cylinder.
   *          In this case it could happen, that the cylinder surpasses the node without intersecting with
   *          the nodes corners. In the worse case the cyclinder could be fully contained in the node.
   */
bool ProbCollisionChecker::checkCornerInCylinder(const Eigen::Vector3i& corner_v,
                                                 const Eigen::Vector3f& start_point_M,
                                                 const Eigen::Vector3f& end_point_M,
                                                 const Eigen::Vector3f& axis,
                                                 const float radius_m) const
{
    //    /**
    //     * Calculates the euclidean distance from a point to a line segment.
    //     *
    //     * @param v     the point
    //     * @param a     start of line segment
    //     * @param b     end of line segment
    //     * @return      distance from v to line segment [a,b]
    //     *
    //     * @author      Afonso Santos
    //     */
    //    public static
    //    float
    //    distanceToSegment( final R3 v, final R3 a, final R3 b )
    //    {
    //      final R3 ab  = b.sub( a ) ;
    //      final R3 av  = v.sub( a ) ;
    //
    //      if (av.dot(ab) <= 0.0)           // Point is lagging behind start of the segment, so perpendicular distance is not viable.
    //        return av.modulus( ) ;         // Use distance to start of segment instead.
    //
    //      final R3 bv  = v.sub( b ) ;
    //
    //      if (bv.dot(ab) >= 0.0)           // Point is advanced past the end of the segment, so perpendicular distance is not viable.
    //        return bv.modulus( ) ;         // Use distance to end of the segment instead.
    //
    //      return (ab.cross( av )).modulus() / ab.modulus() ;       // Perpendicular distance of point to segment.
    //    }

    Eigen::Vector3f corner_m = corner_v.cast<float>() * res_;
    Eigen::Vector3f corner_start_point_M = corner_m - start_point_M;

    // Behind start layer
    if (corner_start_point_M.dot(axis) < 0)
        return false;

    Eigen::Vector3f corner_end_point_M = corner_m - end_point_M;

    // Infront end layer
    if (corner_end_point_M.dot(axis) > 0)
        return false;

    // Compute point line distance
    float dist = (axis.cross(corner_start_point_M)).norm();
    if (dist > radius_m)
        return false;
    return true;
}



bool ProbCollisionChecker::checkInCylinderFree(const Eigen::Vector3i& node_corner_min,
                                               const int node_size,
                                               const Eigen::Vector3f& start_point_M,
                                               const Eigen::Vector3f& end_point_M,
                                               const Eigen::Vector3f& axis,
                                               const float radius_m) const
{
    for (int i = 0; i < 8; i++) {
        Eigen::Vector3i node_corner = node_corner_min
            + Eigen::Vector3i::Constant(node_size).cwiseProduct(
                Eigen::Vector3i((i & 1) > 0, (i & 2) > 0, (i & 4) > 0));
        Eigen::Vector3f node_center =
            node_corner.cast<float>() + Eigen::Vector3f::Constant((float) node_size / 2);

        if (!checkCenterInCylinder(
                node_center,
                node_size,
                start_point_M,
                end_point_M,
                axis,
                radius_m)) { ///< Continue if the node does not intersect with the cylinder (conservative check).
            continue;
        }

        bool abort = false;
        if (checkNode(node_corner,
                      node_size,
                      abort)) { ///< Continue is the intersecting node is completely free.
            continue;
        }

        if (abort) {
            return false;
        }

        int num_inside = 0;
        std::vector<Eigen::Vector3i> test_positions;
        for (int j = 0; j < 8; j++) {
            Eigen::Vector3i sub_node_corner = node_corner
                + Eigen::Vector3i::Constant(node_size).cwiseProduct(
                    Eigen::Vector3i((j & 1) > 0, (j & 2) > 0, (j & 4) > 0));
            if (checkCornerInCylinder(
                    sub_node_corner,
                    start_point_M,
                    end_point_M,
                    axis,
                    radius_m)) { ///< Check if all 8 corners are inside the cyclinder, causing guaranteed collision.
                num_inside++;
                test_positions.push_back(sub_node_corner);
            }
        }

        if (num_inside
            == 8) { ///< Corridor is guaranteed to be in collision if node is fully contained.
            return false;
        }

        for (auto const& test_position : test_positions) {
            if (!ow_.isFree(test_position, free_threshold_)) ///< Check if one of the contained corners is occupied.
                return false;
        }

        if (!checkInCylinderFree(node_corner,
                                 node_size / 2,
                                 start_point_M,
                                 end_point_M,
                                 axis,
                                 radius_m)) { ///< Continue the search at a finer resolution
            return false;
        }
    }
    return true;
}

bool ProbCollisionChecker::checkNodeInCylinderFree(
    const se::Node<MultiresOFusion::VoxelType>* parent_node,
    const Eigen::Vector3i& parent_coord,
    const int node_size,
    const Eigen::Vector3f& start_point_M,
    const Eigen::Vector3f& end_point_M,
    const Eigen::Vector3f& axis,
    const float radius_m) const
{
    for (int i = 0; i < 8; i++) {
        if (checkData(parent_node->childData(i))) { //< If free continue;
            continue;
        }

        const Eigen::Vector3i node_rel_step = Eigen::Vector3i::Constant(node_size).cwiseProduct(
            Eigen::Vector3i((i & 1) > 0, (i & 2) > 0, (i & 4) > 0));
        const Eigen::Vector3i node_coord = parent_coord + node_rel_step;
        const Eigen::Vector3f node_centre =
            node_coord.cast<float>() + Eigen::Vector3f::Constant((float) node_size / 2);
        if (!checkCenterInCylinder(
                node_centre,
                node_size,
                start_point_M,
                end_point_M,
                axis,
                radius_m)) { ///< Continue if the node does not intersect with the cylinder (conservative check).
            continue;
        }

        const se::Node<MultiresOFusion::VoxelType>* node = parent_node->child(i);
        if (!node) {
            return false;
        }


        // TODO: Try without!
        int num_inside = 0;
        std::vector<Eigen::Vector3i> test_positions;
        for (int j = 0; j < 8; j++) {
            Eigen::Vector3i node_corner_offset = Eigen::Vector3i::Constant(node_size).cwiseProduct(
                Eigen::Vector3i((j & 1) > 0, (j & 2) > 0, (j & 4) > 0));
            Eigen::Vector3i node_corner = node_coord + node_corner_offset;
            if (checkCornerInCylinder(
                    node_corner,
                    start_point_M,
                    end_point_M,
                    axis,
                    radius_m)) { ///< Check if all 8 corners are inside the cyclinder, causing guaranteed collision.
                num_inside++;
                test_positions.push_back(node_corner);
            }
        }

        if (num_inside
            == 8) { ///< Corridor is guaranteed to be in collision if node is fully contained.
            return false;
        }

        // TODO: Try without!
        for (auto const& test_position : test_positions) {
            if (!ow_.isFree(test_position, free_threshold_)) ///< Check if one of the contained corners is occupied.
                return false;
        }

        if (node->isBlock()) {
            const se::VoxelBlockSingleMax<MultiresOFusion::VoxelType>* block =
                static_cast<const se::VoxelBlockSingleMax<MultiresOFusion::VoxelType>*>(node);
            if (!checkBlockInCylinderFree(
                    block,
                    node_coord,
                    node_size / 2,
                    start_point_M,
                    end_point_M,
                    axis,
                    radius_m)) { ///< Continue the search at a finer resolution
                return false;
            }
        }
        else {
            if (!checkNodeInCylinderFree(node,
                                         node_coord,
                                         node_size / 2,
                                         start_point_M,
                                         end_point_M,
                                         axis,
                                         radius_m)) { ///< Continue the search at a finer resolution
                return false;
            }
        }
    }
    return true;
}

bool ProbCollisionChecker::checkBlockInCylinderFree(
    const se::VoxelBlockSingleMax<MultiresOFusion::VoxelType>* block,
    const Eigen::Vector3i& parent_coord,
    const int node_size,
    const Eigen::Vector3f& start_point_M,
    const Eigen::Vector3f& end_point_M,
    const Eigen::Vector3f& axis,
    const float radius_m) const
{
    for (int i = 0; i < 8; i++) {
        const Eigen::Vector3i node_rel_step = Eigen::Vector3i::Constant(node_size).cwiseProduct(
            Eigen::Vector3i((i & 1) > 0, (i & 2) > 0, (i & 4) > 0));
        const Eigen::Vector3i node_coord = parent_coord + node_rel_step;
        const int scale = se::math::log2_const(node_size);

        if (checkData(block->maxData(node_coord, scale),
                      (scale == block->current_scale()))) { ///< If free continue
            continue;
        }


        const Eigen::Vector3f node_centre =
            node_coord.cast<float>() + Eigen::Vector3f::Constant((float) node_size / 2);
        if (!checkCenterInCylinder(
                node_centre,
                node_size,
                start_point_M,
                end_point_M,
                axis,
                radius_m)) { ///< Continue if the node does not intersect with the cylinder (conservative check).
            continue;
        }

        if (scale == block->current_scale()) {
            return false;
        }


        // TODO: Try without!
        int num_inside = 0;
        std::vector<Eigen::Vector3i> test_positions;
        for (int j = 0; j < 8; j++) {
            Eigen::Vector3i node_corner_offset = Eigen::Vector3i::Constant(node_size).cwiseProduct(
                Eigen::Vector3i((j & 1) > 0, (j & 2) > 0, (j & 4) > 0));
            Eigen::Vector3i node_corner = node_coord + node_corner_offset;
            if (checkCornerInCylinder(
                    node_corner,
                    start_point_M,
                    end_point_M,
                    axis,
                    radius_m)) { ///< Check if all 8 corners are inside the cyclinder, causing guaranteed collision.
                num_inside++;
                test_positions.push_back(node_corner);
            }
        }

        if (num_inside
            == 8) { ///< Corridor is guaranteed to be in collision if node is fully contained.
            return false;
        }

        // TODO: Try without!
        for (auto const& test_position : test_positions) {
            if (!ow_.isFree(test_position, free_threshold_)) ///< Check if one of the contained corners is occupied.
                return false;
        }

        if (!checkBlockInCylinderFree(block,
                                      node_coord,
                                      node_size / 2,
                                      start_point_M,
                                      end_point_M,
                                      axis,
                                      radius_m)) { ///< Continue the search at a finer resolution
            return false;
        }
    }
    return true;
}

inline bool cmpVecs(const Eigen::Vector3i& lhs, const Eigen::Vector3i& rhs)
{
    if (lhs.x() < rhs.x())
        return true;
    if (rhs.x() < lhs.x())
        return false;
    if (lhs.y() < rhs.y())
        return true;
    if (rhs.y() < lhs.y())
        return false;
    return lhs.z() < rhs.z();
}


inline void removeDuplicates(std::vector<Eigen::Vector3i>& con)
{
    std::sort(con.data(), con.data() + con.size(), cmpVecs);
    auto itr = std::unique(con.begin(), con.end());
    con.resize(itr - con.begin());
}


bool ProbCollisionChecker::checkSegmentFlightCorridor(const Eigen::Vector3f& start_point_M,
                                                      const Eigen::Vector3f& end_point_M,
                                                      const float radius_m) const
{
    // Don't test the segment start because:
    // 1) Testing the endpoint is enough to test every vertex in the path except the first one.
    // 2) We don't want to test the very first path vertex because it might have ended up slightly
    //    outside the map due to tracking inaccuracy. In that case we still want to be able to plan
    //    and not get stuck due to the start being occupied.

    if (!ow_.inMapBoundsMeter(end_point_M)) {
        return false;
    }

    /**
     * Check if the start and end position fullfill the requirements of the given flight corridor radius
     * TODO: Check if checkSphere works for radius 0
     */
    if (!checkCorridorSkeleton(start_point_M, end_point_M, radius_m)) {
        return false;
    }

    // Don't check the sphere at the segment start for the same reasons that the in-map test is
    // skipped.
    if (!checkSphere(end_point_M, radius_m)) {
        return false;
    }

    /**
     * Generate std::vector containing all voxels (Eigen::Vector3i) to checked that are within the flight corridor
     */
    const Eigen::Vector3f corridor_axis_u =
        (end_point_M - start_point_M) / (end_point_M - start_point_M).norm();

    const float radius_v = radius_m / res_;
    /// Cover the entire flight corridor in a bounding box and find the x, y, z min/max extrema.
    const Eigen::Vector3i bb_corner_min =
        (Eigen::Vector3f(std::min(start_point_M.x(), end_point_M.x()),
                         std::min(start_point_M.y(), end_point_M.y()),
                         std::min(start_point_M.z(), end_point_M.z()))
         / res_)
            .cast<int>()
        - Eigen::Vector3i::Constant(radius_v);
    const Eigen::Vector3i bb_corner_max =
        (Eigen::Vector3f(std::max(start_point_M.x(), end_point_M.x()),
                         std::max(start_point_M.y(), end_point_M.y()),
                         std::max(start_point_M.z(), end_point_M.z()))
         / res_)
            .cast<int>()
        + Eigen::Vector3i::Constant(radius_v);
    const Eigen::Vector3i bb_size =
        bb_corner_max - bb_corner_min;          ///< Compute the size of the bounding box.
    const int bb_max_size = bb_size.maxCoeff(); ///< Find the max side length.
    const int node_size =
        1 << (int) ceil(
            log2(bb_max_size)); ///< Find the node size that would cover the bounding box.

    std::vector<Eigen::Vector3i> bb_corners; ///< Compute the 8 corners of the bounding box.
    for (int i = 0; i < 8; i++) {
        bb_corners.push_back(
            bb_corner_min
            + bb_size.cwiseProduct(Eigen::Vector3i((i & 1) > 0, (i & 2) > 0, (i & 4) > 0)));
    }

    removeDuplicates(bb_corners); // TODO: What did this do? How can there be duplicates.

    for (auto const& bb_corner : bb_corners) {
        /// If the nodes are all free we can directly set the corridor free.
        /// The check is really conservative. Otherwise check further.
        if (!checkNode(bb_corner, node_size)) {
            const Eigen::Vector3i node_coord = node_size
                * (bb_corner
                   / node_size); ///< Compute the lower, front, left corner of the node containing the bb_corner.

            const se::Node<MultiresOFusion::VoxelType>* node = ow_.getNode(node_coord, node_size);
            if (!node) {
                return false;
            }

            if (node->isBlock()) {
                const se::VoxelBlockSingleMax<MultiresOFusion::VoxelType>* block =
                    static_cast<const se::VoxelBlockSingleMax<MultiresOFusion::VoxelType>*>(node);
                const int voxel_size = std::max(1 << block->current_scale(), node_size / 2);
                if (!checkBlockInCylinderFree(block,
                                              node_coord,
                                              voxel_size,
                                              start_point_M,
                                              end_point_M,
                                              corridor_axis_u,
                                              radius_m)) {
                    return false;
                }
            }
            else {
                if (!checkNodeInCylinderFree(node,
                                             node_coord,
                                             node_size / 2,
                                             start_point_M,
                                             end_point_M,
                                             corridor_axis_u,
                                             radius_m)) {
                    return false;
                }
            }
        }
    }

    return true;
}

} // namespace ptp
