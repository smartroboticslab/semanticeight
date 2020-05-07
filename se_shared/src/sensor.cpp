// SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#include "se/sensor.hpp"

#include <cassert>

// Explicit template class instantiation
template class srl::projection::PinholeCamera<srl::projection::NoDistortion>;

// Used for initializing a PinholeCamera.
const srl::projection::NoDistortion _distortion;



se::PinholeCamera::PinholeCamera(const SensorConfig& c)
    : model(c.width, c.height, c.fx, c.fy, c.cx, c.cy, _distortion),
      left_hand_frame(c.left_hand_frame), near_plane(c.near_plane), far_plane(c.far_plane), mu(c.mu), scaled_pixel(1 / c.fx) {
  assert(c.width  > 0);
  assert(c.height > 0);
  assert(c.near_plane >= 0.f);
  assert(c.far_plane > c.near_plane);
  assert(c.mu > 0.f);
  assert(!isnan(c.fx));
  assert(!isnan(c.fy));
  assert(!isnan(c.cx));
  assert(!isnan(c.cy));
}

se::PinholeCamera::PinholeCamera(const PinholeCamera& pc, const float sf)
    : model(pc.model.imageWidth() * sf, pc.model.imageHeight() * sf,
            pc.model.focalLengthU() * sf, pc.model.focalLengthV() * sf,
            pc.model.imageCenterU() * sf, pc.model.imageCenterV() * sf, _distortion),
            left_hand_frame(pc.left_hand_frame), near_plane(pc.near_plane), far_plane(pc.far_plane), mu(pc.mu) {
}

/**
 * \brief Computes the scale corresponding to the back-projected pixel size
 * in voxel space
 * \param[in] depth distance from the camera to the voxel block centre
 * \param[out] scale scale from which propagate up voxel values
 */
int se::PinholeCamera::computeIntegrationScale(const float distance, const float voxel_dim) const {
  const float pv_ration = distance * scaled_pixel / voxel_dim;

  int scale = 0;
  if (pv_ration < 1.5)
    scale = 0;
  else if (pv_ration < 3)
    scale = 1;
  else if (pv_ration < 6)
    scale = 2;
  else
    scale = 3;

//      int scale = std::min(std::max(0, int(log2(pix_size / voxel_dim + 0.5f))),
//                           max_scale);
  return scale;
}


se::OusterLidar::OusterLidar(const SensorConfig& c)
    : model(c.width, c.height, c.beam_azimuth_angles, c.beam_elevation_angles),
      left_hand_frame(c.left_hand_frame), near_plane(c.near_plane), far_plane(c.far_plane), mu(c.mu) {
  assert(c.width  > 0);
  assert(c.height > 0);
  assert(c.near_plane >= 0.f);
  assert(c.far_plane > c.near_plane);
  assert(c.mu > 0.f);
  assert(c.beam_azimuth_angles.size()   > 0);
  assert(c.beam_elevation_angles.size() > 0);
}

se::OusterLidar::OusterLidar(const OusterLidar& ol, const float sf)
    : model(ol.model.imageWidth() * sf, ol.model.imageHeight() * sf,
            ol.model.beamAzimuthAngles(), ol.model.beamElevationAngles()), // TODO: Does the beam need to be scaled too?
            left_hand_frame(ol.left_hand_frame), near_plane(ol.near_plane), far_plane(ol.far_plane), mu(ol.mu) {
}

