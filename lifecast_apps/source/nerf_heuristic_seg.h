// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include <string>
#include "logger.h"
#include "util_time.h"
#include "util_math.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "ceres/ceres.h"

namespace p11 { namespace nerf {

struct SmoothnessResidual {
  SmoothnessResidual() {}

  template <typename T>
  bool operator()(const T* const param1, const T* const param2, T* residual) const
  {
    constexpr double kWeight = 3.0;
    residual[0] = kWeight * (T(param1[0]) - param2[0]);
    return true;
  }
};


struct TargetResidual {
  double target;
  double weight;
  TargetResidual(double target, double weight) : target(target), weight(weight) {}

  template <typename T>
  bool operator()(const T* const param, T* residual) const
  {
    residual[0] = weight * (T(param[0]) - T(target)); 
    return true;
  }
};


inline int paramIndex(const int w, const int x, const int y)
{
  return y * w + x;
}

cv::Mat blendDepthBySegmentation(const cv::Mat& invdepth, const cv::Mat& seg, int target_layer)
{
  auto timer = time::now();

  XCHECK_EQ(invdepth.type(), CV_32F);
  XCHECK_EQ(seg.type(), CV_32FC3);
  XCHECK_EQ(invdepth.size(), seg.size());
  
  const int w = invdepth.cols;
  const int h = invdepth.rows;

  std::vector<double> params(w * h);
  ceres::Problem problem;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const float target = invdepth.at<float>(y, x);
      const float weight = seg.at<cv::Vec3f>(y, x)[target_layer];

      // TODO: optimize for pixels inside image circle?

      // Initialize with the inverse depth map as the guess
      params[paramIndex(w, x, y)] = target;

      problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction<TargetResidual, 1, 1>(
              new TargetResidual(target, weight)),
          nullptr,
          &params[paramIndex(w, x, y)]);

      // Make a loss for smoothness between this pixel and its neighbors
      if (x > 0) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<SmoothnessResidual, 1, 1, 1>(
                new SmoothnessResidual()),
            nullptr,
            &params[paramIndex(w, x, y)],
            &params[paramIndex(w, x - 1, y)]);
      }
      if (x < w - 1) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<SmoothnessResidual, 1, 1, 1>(
                new SmoothnessResidual()),
            nullptr,
            &params[paramIndex(w, x, y)],
            &params[paramIndex(w, x + 1, y)]);
      }
      if (y > 0) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<SmoothnessResidual, 1, 1, 1>(
                new SmoothnessResidual()),
            nullptr,
            &params[paramIndex(w, x, y)],
            &params[paramIndex(w, x, y - 1)]);
      }
      if (y < h - 1) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<SmoothnessResidual, 1, 1, 1>(
                new SmoothnessResidual()),
            nullptr,
            &params[paramIndex(w, x, y)],
            &params[paramIndex(w, x, y + 1)]);
      }
    }
  }

  // Run the solver
  ceres::Solver::Options options;
  options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.use_nonmonotonic_steps = true;
  options.gradient_tolerance = 1e-8;
  options.function_tolerance = 1e-8;
  options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 3;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;

  // Unpack the solution image
  cv::Mat result(invdepth.size(), CV_32F);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      result.at<float>(y, x) = math::clamp<float>(params[paramIndex(w, x, y)], 0.0f, 1.0f);
    }
  }

  XPLINFO << "smoothing time: " << time::timeSinceSec(timer);
  return result;
}

inline cv::Mat heuristic3LayerSegmentation(const cv::Mat& inv_depth)
{
  auto timer = time::now();
  cv::Size full_size = inv_depth.size();
  cv::Mat segmentation(full_size, CV_32FC3, cv::Scalar(0.0));

  // If the kernel is very large, we will first resize to a smaller image, then do the kernel, then
  // scale up the result.
  constexpr int kBigKernel = 50;
  const std::vector<int> kernel_sizes = {17, 31, 63, 127, 255};
  const int num_kernels = kernel_sizes.size();

  cv::Mat inv_depth_small;
  cv::Size small_size(inv_depth.cols / 4, inv_depth.rows / 4);

  cv::resize(inv_depth, inv_depth_small, small_size, 0, 0, cv::INTER_AREA);

  std::vector<cv::Mat> fgs, bgs, scale;
  for (int k = 0; k < num_kernels; ++k) {
    int ks = kernel_sizes[k];
    int ks2 = ks > kBigKernel ? ks / 4 : ks;
    cv::Mat invdepth_ks = ks > kBigKernel ? inv_depth_small : inv_depth;

    cv::Mat eroded, dilated;
    cv::erode(invdepth_ks, eroded, cv::Mat(), cv::Point(-1, -1), ks2);
    cv::dilate(invdepth_ks, dilated, cv::Mat(), cv::Point(-1, -1), ks2);
    cv::GaussianBlur(eroded, eroded, cv::Size(ks2, ks2), ks2 / 2.0);
    cv::GaussianBlur(dilated, dilated, cv::Size(ks2, ks2), ks2 / 2.0);

    cv::Mat fg = invdepth_ks - eroded;
    cv::Mat bg = dilated - invdepth_ks;
    cv::Mat sc = dilated - eroded;

    if (ks > kBigKernel) {
      cv::resize(fg, fg, full_size, 0, 0, cv::INTER_LINEAR);
      cv::resize(bg, bg, full_size, 0, 0, cv::INTER_LINEAR);
      cv::resize(sc, sc, full_size, 0, 0, cv::INTER_LINEAR);
    }

    fgs.push_back(fg);
    bgs.push_back(bg);
    scale.push_back(sc);
  }

  for (int y = 0; y < segmentation.rows; ++y) {
    for (int x = 0; x < segmentation.cols; ++x) {
      float sum_fg = 0;
      float sum_bg = 0;
      float sum_mid = 3.0; // HACK: bias
      for (int k = 0; k < num_kernels; ++k) {
        float s = scale[k].at<float>(y, x) + 0.01;
        const float fg = fgs[k].at<float>(y, x);
        const float bg = bgs[k].at<float>(y, x);

        sum_fg += fg / s;
        sum_bg += bg / s;
      }

      constexpr float kSteepness = 4.0;
      sum_fg = std::exp(kSteepness * sum_fg);
      sum_bg = std::exp(kSteepness * sum_bg);
      sum_mid = std::exp(kSteepness * sum_mid);

      float sum_all = sum_fg + sum_bg + sum_mid + 1e-6;
      float score_fg = sum_fg / sum_all;
      float score_bg = sum_bg / sum_all;
      float score_mid = sum_mid / sum_all;

      //                                // layer   0         1          2
      //                                //         blue      green      red
      segmentation.at<cv::Vec3f>(y, x) = cv::Vec3f(score_bg, score_mid, score_fg);
    }
  }

  XPLINFO << "segmentation time: " << time::timeSinceSec(timer);
  return segmentation;
}

}}  // namespace p11::nerf
