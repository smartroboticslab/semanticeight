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
#include <random>
#include <se/octree_defines.h>
#include <se/utils/math_utils.h>
#include <se/utils/morton_utils.hpp>

TEST(MortonCoding, RandomInts)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 4096);

    for (int i = 0; i < 1000; ++i) {
        const Eigen::Vector3i voxel_coord = {dis(gen), dis(gen), dis(gen)};
        const se::key_t code = compute_morton(voxel_coord.x(), voxel_coord.y(), voxel_coord.z());
        const Eigen::Vector3i decoded_voxel_coord = unpack_morton(code);
        ASSERT_EQ(decoded_voxel_coord.x(), voxel_coord.x());
        ASSERT_EQ(decoded_voxel_coord.y(), voxel_coord.y());
        ASSERT_EQ(decoded_voxel_coord.z(), voxel_coord.z());
    }
}

TEST(MortonCoding, ExhaustiveTest)
{
    for (int z = 2048; z < 4096; ++z)
        for (int y = 2048; y < 2050; ++y)
            for (int x = 0; x < 4096; ++x) {
                const Eigen::Vector3i voxel_coord = {x, y, z};
                const se::key_t code =
                    compute_morton(voxel_coord.x(), voxel_coord.y(), voxel_coord.z());
                const Eigen::Vector3i decoded_voxel_coord = unpack_morton(code);
                ASSERT_EQ(decoded_voxel_coord.x(), voxel_coord.x());
                ASSERT_EQ(decoded_voxel_coord.y(), voxel_coord.y());
                ASSERT_EQ(decoded_voxel_coord.z(), voxel_coord.z());
            }
}
