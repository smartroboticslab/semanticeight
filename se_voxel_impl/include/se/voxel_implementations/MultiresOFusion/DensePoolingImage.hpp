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

#ifndef DENSE_POOLING_IMAGE
#define DENSE_POOLING_IMAGE

#include "se/image/image.hpp"
#include "se/perfstats.h"
#include "se/timings.h"

#include <Eigen/Dense>
#include <iostream>

namespace se {

  class DensePoolingImage {
  public:
    using Value = float;
    using Status= int;

    struct Pixel {
      Value min;
      Value max;

      // STATUS Crossing: Voxel image intersection
      // outside  := 2;
      // crossing := 1;
      // inside   := 0;
      enum statusCrossing { inside = 0, crossing = 1, outside = 2 };

      // STATUS Known: Voxel content
      // unknown          := 2;
      // partially known  := 1;
      // known            := 0;
      enum statusKnown { known = 0, part_known = 1, unknown = 2 };

      statusCrossing status_crossing;
      statusKnown    status_known;

      Pixel() {};

      Pixel(Value min, Value max, statusCrossing status_crossing, statusKnown status_known)
            : min(min), max(max), status_crossing(status_crossing), status_known(status_known) {};

      // Inside pixel //

      // Init known pixel
      static Pixel knownPixel() {
        Pixel knownPixel(std::numeric_limits<Value>::max(), std::numeric_limits<Value>::min(),
            statusCrossing::inside, statusKnown::known);
        return knownPixel;
      };

      // Init unknown pixel
      static Pixel unknownPixel() {
        Pixel unknownPixel(std::numeric_limits<Value>::max(), std::numeric_limits<Value>::min(),
            statusCrossing::inside, statusKnown::known);
        return unknownPixel;
      };

      // Crossing pixel //

      static Pixel crossingKnownPixel() {
        Pixel crossingPixel(std::numeric_limits<Value>::max(), std::numeric_limits<Value>::min(),
            statusCrossing::crossing, statusKnown::known);
        return crossingPixel;
      };

      // Init crossing partially known pixel
      static Pixel crossingPartKnownPixel() {
        Pixel crossingPixel(std::numeric_limits<Value>::max(), std::numeric_limits<Value>::min(),
            statusCrossing::crossing, statusKnown::part_known);
        return crossingPixel;
      };

      static Pixel crossingUnknownPixel() {
        Pixel crossingPixel(std::numeric_limits<Value>::max(), std::numeric_limits<Value>::min(),
            statusCrossing::crossing, statusKnown::unknown);
        return crossingPixel;
      };

      // Outside pixel //

      // Init outside pixel
      static Pixel outsidePixelBatch() {
        Pixel outsidePixel(0, 0, statusCrossing::outside, statusKnown::unknown);
        return outsidePixel;
      };

    };

    using Img = std::vector<Pixel>;
    using Imgs = std::vector<Img>;

    DensePoolingImage(const se::Image<float>& depth_image);

    bool inImage(const int u, const int v) const;
    DensePoolingImage::Pixel conservativeQuery(const Eigen::Vector2i& bb_min, const Eigen::Vector2i& bb_max) const;
    DensePoolingImage::Pixel poolBoundingBox(int u_min, int u_max, int v_min, int v_max) const;

    int width() const {return image_width_;};
    int height() const {return image_height_;};
    Value maxValue() const {return image_max_value_;};
    int maxLevel() const {return image_max_level_;}

  private:
    int   image_max_level_;
    int   image_width_;
    int   image_height_;
    Imgs  pooling_image_;
    Value image_max_value_;
  };

} // namespace se

#endif // DENSE_POOLING_IMAGE
