// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

/*
Our own inhouse inpainting/outpainting designed to produce a smooth blur.
The goal is to avoid the "ridge" artifacts of the Telea method.
To make this fast, we'll solve the optimization problem using ceres.
Also, this should work with 1, 2, or 3 channel images, which could be useful for (1--depth, 2--flow,
3--color)
*/
#pragma once

#include <string>
#include "logger.h"
#include "util_time.h"
#include "util_math.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "ceres/ceres.h"

namespace p11 { namespace inpaint {

struct SmoothnessResidual {
  int dim;
  SmoothnessResidual(const int dim) : dim(dim) {}

  template <typename T>
  bool operator()(const T* const param1, const T* const param2, T* residual) const
  {
    for (int i = 0; i < dim; ++i) {
      residual[i] = T(param1[i]) - param2[i];
    }
    return true;
  }
};

struct UppperBoundResidual {
  double b;
  UppperBoundResidual(const double b) : b(b) {}

  template <typename T>
  bool operator()(const T* const param, T* residual) const
  {
    constexpr double kMargin = 2.0 / 255.0;
    residual[0] = ceres::fmax(T(0.0), T(kMargin) + param[0] - T(b));
    return true;
  }
};

inline int paramIndex(const int w, const int dim, const int x, const int y)
{
  return (y * w + x) * dim;
}

template <int dim>
cv::Mat inpaintWithCeres(
    const cv::Mat& image, const cv::Mat& mask, const cv::Mat& upper_bound = cv::Mat())
{
  if (dim == 1) XCHECK_EQ(image.type(), CV_32FC1);
  if (dim == 2) XCHECK_EQ(image.type(), CV_32FC2);
  if (dim == 3) XCHECK_EQ(image.type(), CV_32FC3);
  XCHECK(image.channels() != 4) << "TODO";
  XCHECK_EQ(mask.type(), CV_8U);

  const int w = image.cols;
  const int h = image.rows;

  // Build an optimization problem where the solution is the inpainted image result, and the cost
  // function encodes consistency with the source image outside the inpainting mask, and smoothness.
  std::vector<double> params(w * h * dim);
  // TODO: better initialization!?! may not make any difference but maybe

  ceres::Problem problem;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      std::vector<double> pixel_channels;
      if (dim == 1) {
        pixel_channels.push_back(image.at<float>(y, x));
      }
      if (dim == 2) {
        pixel_channels.push_back(image.at<cv::Vec2f>(y, x)[0]);
        pixel_channels.push_back(image.at<cv::Vec2f>(y, x)[1]);
      }
      if (dim == 3) {
        pixel_channels.push_back(image.at<cv::Vec3f>(y, x)[0]);
        pixel_channels.push_back(image.at<cv::Vec3f>(y, x)[1]);
        pixel_channels.push_back(image.at<cv::Vec3f>(y, x)[2]);
      }
      // Initialize with parameter vector with the input image
      for (int c = 0; c < dim; ++c) {
        params[paramIndex(w, dim, x, y) + c] = pixel_channels[c];
      }

      if (mask.at<uint8_t>(y, x) < 128) {
        // Lock this pixel so it can't change

        problem.AddParameterBlock(&params[paramIndex(w, dim, x, y)], dim);
        problem.SetParameterBlockConstant(&params[paramIndex(w, dim, x, y)]);

      } else {
        // Make a loss for smoothness between this pixel and its neighbors
        if (x > 0) {
          problem.AddResidualBlock(
              new ceres::AutoDiffCostFunction<SmoothnessResidual, dim, dim, dim>(
                  new SmoothnessResidual(dim)),
              nullptr,
              &params[paramIndex(w, dim, x, y)],
              &params[paramIndex(w, dim, x - 1, y)]);
        }
        if (x < w - 1) {
          problem.AddResidualBlock(
              new ceres::AutoDiffCostFunction<SmoothnessResidual, dim, dim, dim>(
                  new SmoothnessResidual(dim)),
              nullptr,
              &params[paramIndex(w, dim, x, y)],
              &params[paramIndex(w, dim, x + 1, y)]);
        }
        if (y > 0) {
          problem.AddResidualBlock(
              new ceres::AutoDiffCostFunction<SmoothnessResidual, dim, dim, dim>(
                  new SmoothnessResidual(dim)),
              nullptr,
              &params[paramIndex(w, dim, x, y)],
              &params[paramIndex(w, dim, x, y - 1)]);
        }
        if (y < h - 1) {
          problem.AddResidualBlock(
              new ceres::AutoDiffCostFunction<SmoothnessResidual, dim, dim, dim>(
                  new SmoothnessResidual(dim)),
              nullptr,
              &params[paramIndex(w, dim, x, y)],
              &params[paramIndex(w, dim, x, y + 1)]);
        }

        // If we have an upper bound, add a loss for that
        if (dim == 1 && !upper_bound.empty()) {
          const float b = upper_bound.at<float>(y, x);
          problem.AddResidualBlock(
              new ceres::AutoDiffCostFunction<UppperBoundResidual, 1, 1>(
                  new UppperBoundResidual(b)),
              nullptr,
              &params[paramIndex(w, dim, x, y)]);
        }
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
  cv::Mat result(image.size(), image.type());
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      std::vector<float> pix_vals;
      for (int c = 0; c < dim; ++c) {
        pix_vals.push_back(params[paramIndex(w, dim, x, y) + c]);
      }

      if (dim == 1) {
        result.at<float>(y, x) = math::clamp(pix_vals[0], 0.0f, 1.0f);
      }
      if (dim == 2) {
        result.at<cv::Vec2f>(y, x)[0] = math::clamp(pix_vals[0], 0.0f, 1.0f);
        result.at<cv::Vec2f>(y, x)[1] = math::clamp(pix_vals[1], 0.0f, 1.0f);
      }
      if (dim == 3) {
        result.at<cv::Vec3f>(y, x)[0] = math::clamp(pix_vals[0], 0.0f, 1.0f);
        result.at<cv::Vec3f>(y, x)[1] = math::clamp(pix_vals[1], 0.0f, 1.0f);
        result.at<cv::Vec3f>(y, x)[2] = math::clamp(pix_vals[2], 0.0f, 1.0f);
      }
    }
  }
  return result;
}

// Downscale, run at small size. Returns a small image. The caller is responsible for upscaling if
// necessary.
template <int dim>
cv::Mat inpaintWithCeresSmallSize(
    const int scale,
    const cv::Mat& image,
    const cv::Mat& mask,
    const cv::Mat& upper_bound = cv::Mat())
{
  cv::Mat result, small_image, small_mask, small_upper_bound;
  cv::resize(
      image,
      small_image,
      cv::Size(image.cols / scale, image.rows / scale),
      0.0,
      0.0,
      cv::INTER_AREA);
  cv::resize(
      mask, small_mask, cv::Size(image.cols / scale, image.rows / scale), 0.0, 0.0, cv::INTER_AREA);

  if (upper_bound.empty()) {
    result = inpaintWithCeres<dim>(small_image, small_mask);
  } else {
    cv::resize(
        upper_bound,
        small_upper_bound,
        cv::Size(image.cols / scale, image.rows / scale),
        0.0,
        0.0,
        cv::INTER_AREA);
    result = inpaintWithCeres<dim>(small_image, small_mask, small_upper_bound);
  }

  return result;
}

}}  // namespace p11::inpaint
