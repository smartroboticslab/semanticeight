// SPDX-FileCopyrightText: 2021 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#include "se/exploration_utils.hpp"
#include "se/utils/math_utils.h"

namespace se {
  ExploredVolume::ExploredVolume(se::Octree<VoxelImpl::VoxelType>& map) {
    for (const auto& volume : map) {
      // The iterator will not return invalid (uninitialized) data so just focus on free and
      // occupied.
      if (se::math::cu(volume.dim) == 0.0f) {
        std::cout << "WAT?\n";
      }
      if (VoxelImpl::VoxelType::isFree(volume.data)) {
        free_volume += se::math::cu(volume.dim);
      } else {
        occupied_volume += se::math::cu(volume.dim);
      }
    }
    explored_volume = free_volume + occupied_volume;
  }
} // namespace se

