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

#ifndef __KERNEL_IMAGE
#define __KERNEL_IMAGE

#include <Eigen/Dense>
#include <se/image/image.hpp>
#include <iostream>
//#include "thirdparty/cutil_math.h"

namespace se {

  class KernelImage {
  public:
    using ValueT  = float;
    using StatusT = int;

    struct PixelT {
      ValueT  min;
      ValueT  max;

      // STATUS 1: Voxel image intersection
      // outside  := 2;
      // crossing := 1;
      // inside   := 0;
      StatusT status_1;

      // STATUS 2: Voxel content
      // unknown          := 2;
      // partially known  := 1;
      // known            := 0;
      StatusT status_2;

      PixelT() {};

      PixelT(ValueT min, ValueT max, StatusT status_1, StatusT status_2)
            : min(min), max(max), status_1(status_1), status_2(status_2) {};

      // Inside pixel //

      // Init known pixel
      static PixelT knownPixel() {
        PixelT knownPixel(std::numeric_limits<ValueT>::max(), std::numeric_limits<ValueT>::min(), 0, 0);
        return knownPixel;
      };

      // Init unknown pixel
      static PixelT unknownPixel() {
        PixelT unknownPixel(std::numeric_limits<ValueT>::max(), std::numeric_limits<ValueT>::min(), 0, 2);
        return unknownPixel;
      };

      // Crossing pixel //

      static PixelT crossingKnownPixel() {
        PixelT crossingPixel(std::numeric_limits<ValueT>::max(), std::numeric_limits<ValueT>::min(), 1, 0);
        return crossingPixel;
      };

      // Init crossing partially known pixel
      static PixelT crossingPartKnownPixel() {
        PixelT crossingPixel(std::numeric_limits<ValueT>::max(), std::numeric_limits<ValueT>::min(), 1, 1);
        return crossingPixel;
      };

      static PixelT crossingUnknownPixel() {
        PixelT crossingPixel(std::numeric_limits<ValueT>::max(), std::numeric_limits<ValueT>::min(), 1, 2);
        return crossingPixel;
      };

      // Outside pixel //

      // Init outside pixel
      static PixelT outsidePixelBatch() {
        PixelT outsidePixel(0, 0, 2, 2);
        return outsidePixel;
      };

    };

    using ImgT = std::vector<PixelT>;
    using PyramidT = std::vector<ImgT>;

    KernelImage(const se::Image<float>& depth_image);

    bool inImage(const int u, const int v) const;
    KernelImage::PixelT conservativeQuery(Eigen::Vector2i bb_min, Eigen::Vector2i bb_max) const;
    KernelImage::PixelT poolBoundingBox(int u_min, int u_max, int v_min, int v_max) const;

    size_t width() const {return image_width_;};
    size_t height() const {return image_heigth_;};
    ValueT maxValue() const {return image_max_value_;};
    size_t maxLevel() const {return image_max_level_;}
    
  private:
    size_t    image_max_level_;
    size_t    image_width_;
    size_t    image_heigth_;
    PyramidT  pyramid_image_;
    ValueT    image_max_value_;
  };

} // namespace se

#endif // __KERNEL_IMAGE