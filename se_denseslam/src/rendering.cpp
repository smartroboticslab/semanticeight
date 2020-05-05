/*

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.


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

#include "se/rendering.hpp"

#include <cstring>


void renderRGBAKernel(uint8_t*                   output_RGBA,
                      const Eigen::Vector2i&     output_size,
                      const se::Image<uint32_t>& input_RGBA) {

  TICK();

  memcpy(output_RGBA, input_RGBA.data(), output_size.x() * output_size.y() * 4);

  TOCK("renderRGBAKernel", output_size.x() * output_size.y());
}



void renderDepthKernel(unsigned char*         out,
                       float*                 depth_image,
                       const Eigen::Vector2i& image_size,
                       const float            near_plane,
                       const float            far_plane) {

  TICK();

  const float range_scale = 1.f / (far_plane - near_plane);

#pragma omp parallel for
  for (int y = 0; y < image_size.y(); y++) {
    const int row_offset = y * image_size.x();
    for (int x = 0; x < image_size.x(); x++) {

      const unsigned int pixel_idx = row_offset + x;
      const unsigned int rgbw_idx = pixel_idx * 4;

      if (depth_image[pixel_idx] < near_plane) {
        out[rgbw_idx + 0] = 255;
        out[rgbw_idx + 1] = 255;
        out[rgbw_idx + 2] = 255;
        out[rgbw_idx + 3] = 255;
      } else if (depth_image[pixel_idx] > far_plane) {
        out[rgbw_idx + 0] = 0;
        out[rgbw_idx + 1] = 0;
        out[rgbw_idx + 2] = 0;
        out[rgbw_idx + 3] = 255;
      } else {
        const float depth_value = (depth_image[pixel_idx] - near_plane) * range_scale;
        unsigned char rgbw[4];
        gs2rgb(depth_value, rgbw);
        out[rgbw_idx + 0] = rgbw[0];
        out[rgbw_idx + 1] = rgbw[1];
        out[rgbw_idx + 2] = rgbw[2];
        out[rgbw_idx + 3] = rgbw[3];
      }
    }
  }
  TOCK("renderDepthKernel", image_size.x() * image_size.y());
}



void renderTrackKernel(unsigned char*         out,
                       const TrackData*       data,
                       const Eigen::Vector2i& out_size) {

  TICK();

#pragma omp parallel for
  for (int y = 0; y < out_size.y(); y++)
    for (int x = 0; x < out_size.x(); x++) {
      const int pixel_idx = x + out_size.x() * y;
      const int rgbw_idx = pixel_idx * 4;
      switch (data[pixel_idx].result) {
        case 1:
          out[rgbw_idx + 0] = 128;
          out[rgbw_idx + 1] = 128;
          out[rgbw_idx + 2] = 128;
          out[rgbw_idx + 3] = 255;
          break;
        case -1:
          out[rgbw_idx + 0] = 0;
          out[rgbw_idx + 1] = 0;
          out[rgbw_idx + 2] = 0;
          out[rgbw_idx + 3] = 255;
          break;
        case -2:
          out[rgbw_idx + 0] = 255;
          out[rgbw_idx + 1] = 0;
          out[rgbw_idx + 2] = 0;
          out[rgbw_idx + 3] = 255;
          break;
        case -3:
          out[rgbw_idx + 0] = 0;
          out[rgbw_idx + 1] = 255;
          out[rgbw_idx + 2] = 0;
          out[rgbw_idx + 3] = 255;
          break;
        case -4:
          out[rgbw_idx + 0] = 0;
          out[rgbw_idx + 1] = 0;
          out[rgbw_idx + 2] = 255;
          out[rgbw_idx + 3] = 255;
          break;
        case -5:
          out[rgbw_idx + 0] = 255;
          out[rgbw_idx + 1] = 255;
          out[rgbw_idx + 2] = 0;
          out[rgbw_idx + 3] = 255;
          break;
        default:
          out[rgbw_idx + 0] = 255;
          out[rgbw_idx + 1] = 128;
          out[rgbw_idx + 2] = 128;
          out[rgbw_idx + 3] = 255;
          break;
      }
    }
  TOCK("renderTrackKernel", out_size.x() * out_size.y());
}



inline void printNormals(const se::Image<Eigen::Vector3f>& normals,
                         const char*                       filename) {

  unsigned char* image = new unsigned char [normals.width() * normals.height() * 4];
  for (unsigned int y = 0; y < normals.height(); ++y) {
    for (unsigned int x = 0; x < normals.width(); ++x){
      const Eigen::Vector3f n = normals[x + normals.width() * y];
      image[4 * normals.width() * y + 4 * x + 0] = (n.x() / 2 + 0.5) * 255;
      image[4 * normals.width() * y + 4 * x + 1] = (n.y() / 2 + 0.5) * 255;
      image[4 * normals.width() * y + 4 * x + 2] = (n.z() / 2 + 0.5) * 255;
      image[4 * normals.width() * y + 4 * x + 3] = 255;
    }
  }
  lodepng_encode32_file(std::string(filename).append(".png").c_str(),
      image, normals.width(), normals.height());
}

