/*

Copyright 2016 Emanuele Vespa, Imperial College London 

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 

*/

#include <gtest/gtest.h>
#include <se/geometry/aabb_collision.hpp>
#include <se/utils/math_utils.h>

TEST(AABBAABBTest, SquareOverlap)
{
    const Eigen::Vector3i a_coord = Eigen::Vector3i(5, 6, 9);
    const Eigen::Vector3i a_size = Eigen::Vector3i::Constant(3);
    const Eigen::Vector3i b_coord = Eigen::Vector3i(4, 4, 8);
    const Eigen::Vector3i b_size = Eigen::Vector3i::Constant(3);
    int overlaps = se::geometry::aabb_aabb_collision(a_coord, a_size, b_coord, b_size);
    ASSERT_EQ(overlaps, 1);
}

TEST(AABBAABBTest, SquareDisjoint1Axis)
{
    const Eigen::Vector3i a_coord = Eigen::Vector3i(5, 6, 9);
    const Eigen::Vector3i a_size = Eigen::Vector3i::Constant(3);
    const Eigen::Vector3i b_coord = Eigen::Vector3i(4, 4, 1);
    const Eigen::Vector3i b_size = Eigen::Vector3i::Constant(3);
    int overlaps = se::geometry::aabb_aabb_collision(a_coord, a_size, b_coord, b_size);
    ASSERT_EQ(overlaps, 0);
}

TEST(AABBAABBTest, SquareDisjoint2Axis)
{
    /* Disjoint on y and z */
    const Eigen::Vector3i a_coord = Eigen::Vector3i(5, 6, 9);
    const Eigen::Vector3i a_size = Eigen::Vector3i::Constant(3);
    const Eigen::Vector3i b_coord = Eigen::Vector3i(6, 22, 13);
    const Eigen::Vector3i b_size = Eigen::Vector3i::Constant(10);

    int overlapx = se::geometry::axis_overlap(a_coord.x(), a_size.x(), b_coord.x(), b_size.x());
    int overlapy = se::geometry::axis_overlap(a_coord.y(), a_size.y(), b_coord.y(), b_size.y());
    int overlapz = se::geometry::axis_overlap(a_coord.z(), a_size.z(), b_coord.z(), b_size.z());

    ASSERT_EQ(overlapx, 1);
    ASSERT_EQ(overlapy, 0);
    ASSERT_EQ(overlapz, 0);

    int overlaps = se::geometry::aabb_aabb_collision(a_coord, a_size, b_coord, b_size);
    ASSERT_EQ(overlaps, 0);
}

TEST(AABBAABBTest, SquareDisjoint)
{
    /* Disjoint on x, y and z */
    const Eigen::Vector3i a_coord = Eigen::Vector3i(5, 6, 9);
    const Eigen::Vector3i a_size = Eigen::Vector3i::Constant(4);
    const Eigen::Vector3i b_coord = Eigen::Vector3i(12, 22, 43);
    const Eigen::Vector3i b_size = Eigen::Vector3i::Constant(10);

    int overlapx = se::geometry::axis_overlap(a_coord.x(), a_size.x(), b_coord.x(), b_size.x());
    int overlapy = se::geometry::axis_overlap(a_coord.y(), a_size.y(), b_coord.y(), b_size.y());
    int overlapz = se::geometry::axis_overlap(a_coord.z(), a_size.z(), b_coord.z(), b_size.z());

    ASSERT_EQ(overlapx, 0);
    ASSERT_EQ(overlapy, 0);
    ASSERT_EQ(overlapz, 0);

    int overlaps = se::geometry::aabb_aabb_collision(a_coord, a_size, b_coord, b_size);
    ASSERT_EQ(overlaps, 0);
}

TEST(AABBAABBTest, SquareEnclosed)
{
    /* Disjoint on x, y and z */
    const Eigen::Vector3i a_coord = Eigen::Vector3i(5, 6, 9);
    const Eigen::Vector3i a_size = Eigen::Vector3i::Constant(10);
    const Eigen::Vector3i b_coord = Eigen::Vector3i(6, 7, 8);
    const Eigen::Vector3i b_size = Eigen::Vector3i::Constant(2);

    int overlapx = se::geometry::axis_overlap(a_coord.x(), a_size.x(), b_coord.x(), b_size.x());
    int overlapy = se::geometry::axis_overlap(a_coord.y(), a_size.y(), b_coord.y(), b_size.y());
    int overlapz = se::geometry::axis_overlap(a_coord.z(), a_size.z(), b_coord.z(), b_size.z());

    ASSERT_EQ(overlapx, 1);
    ASSERT_EQ(overlapy, 1);
    ASSERT_EQ(overlapz, 1);

    int overlaps = se::geometry::aabb_aabb_collision(a_coord, a_size, b_coord, b_size);
    ASSERT_EQ(overlaps, 1);
}

TEST(AABBAABBTest, Inclusion)
{
    const Eigen::Vector3i a_coord = Eigen::Vector3i(2, 1, 3);
    const Eigen::Vector3i a_size = Eigen::Vector3i::Constant(10);
    const Eigen::Vector3i b_coord = Eigen::Vector3i(3, 4, 5);
    const Eigen::Vector3i b_size = Eigen::Vector3i::Constant(2);
    int included = se::geometry::aabb_aabb_inclusion(a_coord, a_size, b_coord, b_size);
    ASSERT_EQ(included, 1);
}
