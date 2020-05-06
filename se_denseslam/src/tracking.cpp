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

#include "se/tracking.hpp"



static inline Eigen::Matrix<float, 6, 6> makeJTJ(
    const Eigen::Matrix<float, 1, 21>& v) {

  Eigen::Matrix<float, 6, 6> C = Eigen::Matrix<float, 6, 6>::Zero();
  C.row(0) = v.segment(0, 6);
  C.row(1).segment(1, 5) = v.segment(6,  5);
  C.row(2).segment(2, 4) = v.segment(11, 4);
  C.row(3).segment(3, 3) = v.segment(15, 3);
  C.row(4).segment(4, 2) = v.segment(18, 2);
  C(5,5) = v(20);

  for (int r = 1; r < 6; ++r)
    for (int c = 0; c < r; ++c)
      C(r, c) = C(c, r);
  return C;
}



static inline Eigen::Matrix<float, 6, 1> solve(
    const Eigen::Matrix<float, 1, 27>& vals) {

  const Eigen::Matrix<float, 6, 1> b = vals.segment(0, 6);
  const Eigen::Matrix<float, 6, 6> C = makeJTJ(vals.segment(6, 21));
  Eigen::LLT <Eigen::Matrix<float, 6, 6> > llt;
  llt.compute(C);
  Eigen::Matrix<float, 6, 1> res = llt.solve(b);
  return llt.info() == Eigen::Success ? res : Eigen::Matrix<float, 6, 1>::Zero();
}



void new_reduce(int                    block_index,
                float*                 output_data,
                const Eigen::Vector2i& output_res,
                TrackData*             J_data,
                const Eigen::Vector2i& J_res) {

  float* sums = output_data + block_index * 32;

  for (unsigned int i = 0; i < 32; ++i)
    sums[i] = 0;

  float sums0, sums1, sums2, sums3, sums4, sums5, sums6, sums7, sums8, sums9,
        sums10, sums11, sums12, sums13, sums14, sums15, sums16, sums17,
        sums18, sums19, sums20, sums21, sums22, sums23, sums24, sums25,
        sums26, sums27, sums28, sums29, sums30, sums31;
  sums0 = 0.0f;
  sums1 = 0.0f;
  sums2 = 0.0f;
  sums3 = 0.0f;
  sums4 = 0.0f;
  sums5 = 0.0f;
  sums6 = 0.0f;
  sums7 = 0.0f;
  sums8 = 0.0f;
  sums9 = 0.0f;
  sums10 = 0.0f;
  sums11 = 0.0f;
  sums12 = 0.0f;
  sums13 = 0.0f;
  sums14 = 0.0f;
  sums15 = 0.0f;
  sums16 = 0.0f;
  sums17 = 0.0f;
  sums18 = 0.0f;
  sums19 = 0.0f;
  sums20 = 0.0f;
  sums21 = 0.0f;
  sums22 = 0.0f;
  sums23 = 0.0f;
  sums24 = 0.0f;
  sums25 = 0.0f;
  sums26 = 0.0f;
  sums27 = 0.0f;
  sums28 = 0.0f;
  sums29 = 0.0f;
  sums30 = 0.0f;
  sums31 = 0.0f;

#pragma omp parallel for reduction(+:sums0,sums1,sums2,sums3,sums4,sums5,sums6,sums7,sums8,sums9,sums10,sums11,sums12,sums13,sums14,sums15,sums16,sums17,sums18,sums19,sums20,sums21,sums22,sums23,sums24,sums25,sums26,sums27,sums28,sums29,sums30,sums31)
  for (int y = block_index; y < output_res.y(); y += 8) {
    for (int x = 0; x < output_res.x(); x++) {

      const TrackData & row = J_data[(x + y * J_res.x())]; // ...
      if (row.result < 1) {
        // accesses sums[28..31]
        /*(sums+28)[1]*/sums29 += row.result == -4 ? 1 : 0;
        /*(sums+28)[2]*/sums30 += row.result == -5 ? 1 : 0;
        /*(sums+28)[3]*/sums31 += row.result >  -4 ? 1 : 0;

        continue;
      }
      // Error part
      /*sums[0]*/sums0 += row.error * row.error;

      // JTe part
      /*for(int i = 0; i < 6; ++i)
        sums[i+1] += row.error * row.J[i];*/
      sums1 += row.error * row.J[0];
      sums2 += row.error * row.J[1];
      sums3 += row.error * row.J[2];
      sums4 += row.error * row.J[3];
      sums5 += row.error * row.J[4];
      sums6 += row.error * row.J[5];

      // JTJ part, unfortunatly the double loop is not unrolled well...
      /*(sums+7)[0]*/sums7  += row.J[0] * row.J[0];
      /*(sums+7)[1]*/sums8  += row.J[0] * row.J[1];
      /*(sums+7)[2]*/sums9  += row.J[0] * row.J[2];
      /*(sums+7)[3]*/sums10 += row.J[0] * row.J[3];

      /*(sums+7)[4]*/sums11 += row.J[0] * row.J[4];
      /*(sums+7)[5]*/sums12 += row.J[0] * row.J[5];

      /*(sums+7)[6]*/sums13 += row.J[1] * row.J[1];
      /*(sums+7)[7]*/sums14 += row.J[1] * row.J[2];
      /*(sums+7)[8]*/sums15 += row.J[1] * row.J[3];
      /*(sums+7)[9]*/sums16 += row.J[1] * row.J[4];

      /*(sums+7)[10]*/sums17 += row.J[1] * row.J[5];

      /*(sums+7)[11]*/sums18 += row.J[2] * row.J[2];
      /*(sums+7)[12]*/sums19 += row.J[2] * row.J[3];
      /*(sums+7)[13]*/sums20 += row.J[2] * row.J[4];
      /*(sums+7)[14]*/sums21 += row.J[2] * row.J[5];

      /*(sums+7)[15]*/sums22 += row.J[3] * row.J[3];
      /*(sums+7)[16]*/sums23 += row.J[3] * row.J[4];
      /*(sums+7)[17]*/sums24 += row.J[3] * row.J[5];

      /*(sums+7)[18]*/sums25 += row.J[4] * row.J[4];
      /*(sums+7)[19]*/sums26 += row.J[4] * row.J[5];

      /*(sums+7)[20]*/sums27 += row.J[5] * row.J[5];

      // extra info here
      /*(sums+28)[0]*/sums28 += 1;

    }
  }
  sums[0] = sums0;
  sums[1] = sums1;
  sums[2] = sums2;
  sums[3] = sums3;
  sums[4] = sums4;
  sums[5] = sums5;
  sums[6] = sums6;
  sums[7] = sums7;
  sums[8] = sums8;
  sums[9] = sums9;
  sums[10] = sums10;
  sums[11] = sums11;
  sums[12] = sums12;
  sums[13] = sums13;
  sums[14] = sums14;
  sums[15] = sums15;
  sums[16] = sums16;
  sums[17] = sums17;
  sums[18] = sums18;
  sums[19] = sums19;
  sums[20] = sums20;
  sums[21] = sums21;
  sums[22] = sums22;
  sums[23] = sums23;
  sums[24] = sums24;
  sums[25] = sums25;
  sums[26] = sums26;
  sums[27] = sums27;
  sums[28] = sums28;
  sums[29] = sums29;
  sums[30] = sums30;
  sums[31] = sums31;

}



void reduceKernel(float*                 output_data,
                  const Eigen::Vector2i& output_res,
                  TrackData*             J_data,
                  const Eigen::Vector2i& J_res) {

  TICK();
#ifdef OLDREDUCE
#pragma omp parallel for
#endif
  for (int block_index = 0; block_index < 8; block_index++) {
    new_reduce(block_index, output_data, output_res, J_data, J_res);
  }

  Eigen::Map<Eigen::Matrix<float, 8, 32, Eigen::RowMajor> > values(output_data);
  for (int j = 1; j < 8; ++j) {
    values.row(0) += values.row(j);
  }
  TOCK("reduceKernel", 512);
}



void trackKernel(TrackData*                        output_data,
                 const se::Image<Eigen::Vector3f>& input_point_cloud_C,
                 const se::Image<Eigen::Vector3f>& input_normals_C,
                 const se::Image<Eigen::Vector3f>& surface_point_cloud_M,
                 const se::Image<Eigen::Vector3f>& surface_normals_M,
                 const Eigen::Matrix4f&            T_MC,
                 const SensorImpl&                 sensor,
                 const float                       dist_threshold,
                 const float                       normal_threshold) {

  TICK();
  const Eigen::Vector2i input_res( input_point_cloud_C.width(),  input_point_cloud_C.height());
  const Eigen::Vector2i ref_res(surface_point_cloud_M.width(), surface_point_cloud_M.height());

#pragma omp parallel for
  for (int y = 0; y < input_res.y(); y++) {
    for (int x = 0; x < input_res.x(); x++) {
      const Eigen::Vector2i pixel(x, y);

      TrackData& row = output_data[pixel.x() + pixel.y() * ref_res.x()];

      if (input_normals_C[pixel.x() + pixel.y() * input_res.x()].x() == INVALID) {
        row.result = -1;
        continue;
      }

      const Eigen::Vector3f point_M = (T_MC *
          input_point_cloud_C[pixel.x() + pixel.y() * input_res.x()].homogeneous()).head<3>();

      const Eigen::Vector3f point_C = input_point_cloud_C[pixel.x() + pixel.y() * input_res.x()];
      
      Eigen::Vector2f proj_pixel;
      if (sensor.model.project(point_C, &proj_pixel) == srl::projection::ProjectionStatus::OutsideImage) {
        row.result = -2;
        continue;
      }

      proj_pixel += Eigen::Vector2f::Constant(0.5f);

      // sensor.model.project(point_C, &proj_pixel);
      // proj_pixel += Eigen::Vector2f::Constant(0.5f);
      // if (   proj_pixel.x() < 0.5 || proj_pixel.x() >= ref_res.x() - 1.5
      //     || proj_pixel.y() < 0.5 || proj_pixel.y() >= ref_res.y() - 1.5) {
      //   row.result = -2;
      //   continue;
      // }

      const Eigen::Vector2i ref_pixel = proj_pixel.cast<int>();
      const Eigen::Vector3f reference_normal
          = surface_normals_M[ref_pixel.x() + ref_pixel.y() * ref_res.x()];

      if (reference_normal.x() == INVALID) {
        row.result = -3;
        continue;
      }

      const Eigen::Vector3f diff = surface_point_cloud_M[ref_pixel.x() + ref_pixel.y() * ref_res.x()]
          - point_M;
      const Eigen::Vector3f proj_normal = T_MC.topLeftCorner<3, 3>()
          * input_normals_C[pixel.x() + pixel.y() * input_res.x()];

      if (diff.norm() > dist_threshold) {
        row.result = -4;
        continue;
      }
      if (proj_normal.dot(reference_normal) < normal_threshold) {
        row.result = -5;
        continue;
      }
      row.result = 1;
      row.error = reference_normal.dot(diff);
      row.J[0] = reference_normal.x();
      row.J[1] = reference_normal.y();
      row.J[2] = reference_normal.z();

      Eigen::Vector3f cross_res = point_M.cross(reference_normal);
      row.J[3] = cross_res.x();
      row.J[4] = cross_res.y();
      row.J[5] = cross_res.z();
    }
  }
  TOCK("trackKernel", input_res.x() * input_res.y());
}



bool updatePoseKernel(Eigen::Matrix4f& T_MC,
                      const float*     reduction_output_data,
                      float            icp_threshold) {

  bool res = false;
  TICK();
  Eigen::Map<const Eigen::Matrix<float, 8, 32, Eigen::RowMajor> > values(reduction_output_data);
  Eigen::Matrix<float, 6, 1> x = solve(values.row(0).segment(1, 27));
  Eigen::Matrix4f delta = Sophus::SE3<float>::exp(x).matrix();
  T_MC = delta * T_MC;

  if (x.norm() < icp_threshold)
    res = true;

  TOCK("updatePoseKernel", 1);
  return res;
}



bool checkPoseKernel(Eigen::Matrix4f&       T_MC,
                     Eigen::Matrix4f&       previous_T_MC,
                     const float*           reduction_output_data,
                     const Eigen::Vector2i& reduction_output_res,
                     float                  track_threshold) {

  // Check the tracking result, and go back to the previous camera position if necessary

  const Eigen::Matrix<float, 8, 32, Eigen::RowMajor> values(reduction_output_data);

  if ((std::sqrt(values(0, 0) / values(0, 28)) > 2e-2)
      || (values(0, 28) / (reduction_output_res.x() * reduction_output_res.y()) < track_threshold)) {
    T_MC = previous_T_MC;
    return false;
  } else {
    return true;
  }
}

