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
      left_hand_frame(c.left_hand_frame), near_plane(c.near_plane), far_plane(c.far_plane), mu(c.mu) {
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

se::PinholeCamera::PinholeCamera(const PinholeCamera& pc, const int dsr)
    : model(pc.model.imageWidth() / dsr, pc.model.imageHeight() / dsr,
            pc.model.focalLengthU() / dsr, pc.model.focalLengthV() / dsr,
            pc.model.imageCenterU() / dsr, pc.model.imageCenterV() / dsr, _distortion),
            left_hand_frame(pc.left_hand_frame), near_plane(pc.near_plane), far_plane(pc.far_plane), mu(pc.mu) {
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

se::OusterLidar::OusterLidar(const OusterLidar& ol, const int dsr)
    : model(ol.model.imageWidth() / dsr, ol.model.imageHeight() / dsr,
            ol.model.beamAzimuthAngles(), ol.model.beamElevationAngles()), // TODO: Does the beam need to be scaled too?
            left_hand_frame(ol.left_hand_frame), near_plane(ol.near_plane), far_plane(ol.far_plane), mu(ol.mu) {
}
