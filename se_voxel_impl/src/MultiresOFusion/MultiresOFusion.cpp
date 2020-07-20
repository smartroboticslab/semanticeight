/*
 * Copyright 2019 Nils Funk, Imperial College London
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "se/voxel_implementations/MultiresOFusion/MultiresOFusion.hpp"



// Initialize static data members.
constexpr bool   MultiresOFusion::invert_normals;
float            MultiresOFusion::surface_boundary;
float            MultiresOFusion::min_occupancy;
float            MultiresOFusion::max_occupancy;
float            MultiresOFusion::max_weight;
int              MultiresOFusion::fs_integr_scale;
float            MultiresOFusion::factor;
float            MultiresOFusion::log_odd_min;
float            MultiresOFusion::log_odd_max;
bool             MultiresOFusion::const_surface_thickness;
float            MultiresOFusion::tau_min;
float            MultiresOFusion::tau_max;
float            MultiresOFusion::k_tau;
UncertaintyModel MultiresOFusion::uncertainty_model;
float            MultiresOFusion::sigma_min;
float            MultiresOFusion::sigma_max;
float            MultiresOFusion::k_sigma;

// Implement static member functions.
size_t MultiresOFusion::buildAllocationList(
    se::Octree<MultiresOFusion::VoxelType>&,
    const se::Image<float>&,
    const Eigen::Matrix4f&,
    const SensorImpl&,
    se::key_t*,
    size_t) {
  return 0;
}

