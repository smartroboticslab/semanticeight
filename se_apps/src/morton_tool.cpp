// SPDX-FileCopyrightText: 2021 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdio>
#include <cstdlib>
#include <se/octant_ops.hpp>
#include <string>

void usage()
{
    printf("Usage: se-morton-tool X Y Z DEPTH VOXEL_DEPTH|MORTON\n");
    printf("  Encode/decode supereight Morton codes.\n");
}

void decode_morton(char** const argv)
{
    try {
        const se::key_t code = std::stoull(std::string(argv[1]), nullptr, 0);
        const Eigen::Vector3i coord = se::keyops::decode(code);
        const int depth = se::keyops::depth(code);
        printf("Morton code: 0x%lx %lu\n", code, code);
        printf("Coordinates: %d, %d, %d\n", coord.x(), coord.y(), coord.z());
        printf("Depth:       %d\n", depth);
    }
    catch (const std::exception&) {
        usage();
        exit(EXIT_FAILURE);
    }
}

void encode_morton(char** const argv)
{
    try {
        const int x = std::stoull(std::string(argv[1]));
        const int y = std::stoull(std::string(argv[2]));
        const int z = std::stoull(std::string(argv[3]));
        const int depth = std::stoull(std::string(argv[4]));
        const int voxel_depth = std::stoull(std::string(argv[5]));
        const se::key_t code = se::keyops::encode(x, y, z, depth, voxel_depth);
        printf("Morton code: 0x%lx %lu\n", code, code);
        printf("Coordinates: %d, %d, %d\n", x, y, z);
        printf("Depth:       %d\n", depth);
        printf("Voxel depth: %d\n", voxel_depth);
    }
    catch (const std::exception&) {
        usage();
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv)
{
    switch (argc) {
    case 2:
        decode_morton(argv);
        break;
    case 6:
        encode_morton(argv);
        break;
    default:
        usage();
        exit(EXIT_FAILURE);
    }
    exit(EXIT_SUCCESS);
}
