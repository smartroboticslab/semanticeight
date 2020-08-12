/*
 *
 * Copyright 2016 Emanuele Vespa, Imperial College London
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * */

#include "se/voxel_implementations/MultiresTSDF/MultiresTSDF.hpp"



// Initialize static data members.
constexpr bool MultiresTSDF::invert_normals;
float MultiresTSDF::mu_factor;
float MultiresTSDF::mu;
int   MultiresTSDF::max_weight;

void MultiresTSDF::configure(const float voxel_dim) {
  mu         = 8 * voxel_dim;
  max_weight = 100;
}

void MultiresTSDF::configure(YAML::Node yaml_config, const float voxel_dim) {
  configure(voxel_dim);
  if (yaml_config.IsNull()) {
    return;
  }

  if (yaml_config["mu_factor"]) {
    mu_factor = yaml_config["mu_factor"].as<float>();
    mu = mu_factor * voxel_dim;
  }
  if (yaml_config["max_weight"]) {
    max_weight = yaml_config["max_weight"].as<float>();
  }
  if (yaml_config["max_weight"]) {
    max_weight = yaml_config["max_weight"].as<float>();
  }
}

std::string MultiresTSDF::printConfig() {
  std::stringstream ss;
  ss << "========== VOXEL IMPL ========== " << "\n";
  ss << "Invert normals:                  " << (MultiresTSDF::invert_normals
                                                ? "true" : "false") << "\n";
  ss << "Mu factor:                       " << MultiresTSDF::mu_factor << "\n";
  ss << "Mu:                              " << MultiresTSDF::mu << "\n";
  ss << "Max weight:                      " << MultiresTSDF::max_weight << "\n";
  ss << "\n";
  return ss.str();
}

void MultiresTSDF::dumpMesh(OctreeType&                map,
                            std::vector<se::Triangle>& mesh) {
  auto inside = [](const VoxelData& data) {
    return data.x < 0.f;
  };

  auto select_value = [](const VoxelData& data) {
    return data.x;
  };

  se::algorithms::dual_marching_cube(map, select_value, inside, mesh);
}
