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
 *
 * */

#include "se/voxel_implementations/OFusion/OFusion.hpp"



// Initialize static data members.
constexpr bool OFusion::invert_normals;
float OFusion::surface_boundary;
float OFusion::min_occupancy;
float OFusion::max_occupancy;
float OFusion::tau;
float OFusion::k_sigma;

void OFusion::configure() {
  surface_boundary = 0.f;
  min_occupancy    = -1000;
  max_occupancy    =  1000;
  tau              = 4;
  k_sigma          = 0.01;
}

void OFusion::configure(YAML::Node yaml_config) {
  configure();

  if (yaml_config.IsNull()) {
    return;
  }

  if (yaml_config["surface_boundary"]) {
    surface_boundary = yaml_config["surface_boundary"].as<float>();
  }
  if (yaml_config["occupancy_min_max"]) {
    std::vector<float> occupancy_min_max = yaml_config["occupancy_min_max"].as<std::vector<float>>();
    min_occupancy = occupancy_min_max[0];
    max_occupancy = occupancy_min_max[1];
  }
  if (yaml_config["tau"]) {
    tau = yaml_config["tau"].as<float>();
  }
  if (yaml_config["k_sigma"]) {
    k_sigma = yaml_config["k_sigma"].as<float>();
  }
};

std::string OFusion::printConfig() {
  std::stringstream ss;
  ss << "========== VOXEL IMPL ========== " << "\n";

  ss << "Invert normals:                  " << (OFusion::invert_normals
                                                ? "true" : "false") << "\n";
  ss << "Surface boundary:                " << OFusion::surface_boundary << "\n";
  ss << "Min occupancy:                   " << OFusion::min_occupancy << "\n";
  ss << "Max occupancy:                   " << OFusion::max_occupancy << "\n";
  ss << "tau:                             " << OFusion::tau << "\n";
  ss << "k sigma:                         " << OFusion::k_sigma << "\n";
  ss << "\n";
  return ss.str();
}