// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ldi_segmentation2.h"

#include <algorithm>
#include <random>

#include "logger.h"
#include "check.h"
#include "util_math.h"
#include "util_time.h"
#include "util_opencv.h"

namespace p11 { namespace ldi {

cv::Mat segment3layerWithHeuristic(
    const std::string debug_dir,
    const cv::Mat& R_inv_depth_ftheta,
    const cv::Mat prior_segmentation,
    const cv::Mat prior_weight)
{
  auto timer = time::now();
  cv::Size full_size = R_inv_depth_ftheta.size();
  cv::Mat segmentation(full_size, CV_32FC3, cv::Scalar(0.0));

  // If the kernel is very large, we will first resize to a smaller image, then do the kernel, then
  // scale up the result.
  static constexpr int kBigKernel = 50;
  const std::vector<int> kernel_sizes = {17, 31, 63, 127, 255};
  const int num_kernels = kernel_sizes.size();

  cv::Mat inv_depth_small;
  cv::Size small_size(R_inv_depth_ftheta.cols / 4, R_inv_depth_ftheta.rows / 4);

  cv::resize(R_inv_depth_ftheta, inv_depth_small, small_size, 0, 0, cv::INTER_AREA);

  std::vector<cv::Mat> fgs, bgs, mds, scale;
  for (int k = 0; k < num_kernels; ++k) {
    int ks = kernel_sizes[k];
    int ks2 = ks > kBigKernel ? ks / 4 : ks;
    cv::Mat invdepth_ks = ks > kBigKernel ? inv_depth_small : R_inv_depth_ftheta;

    cv::Mat eroded, dilated;
    cv::erode(invdepth_ks, eroded, cv::Mat(), cv::Point(-1, -1), ks2);
    cv::dilate(invdepth_ks, dilated, cv::Mat(), cv::Point(-1, -1), ks2);
    cv::GaussianBlur(eroded, eroded, cv::Size(ks2, ks2), ks2 / 2.0);
    cv::GaussianBlur(dilated, dilated, cv::Size(ks2, ks2), ks2 / 2.0);

    cv::Mat md = invdepth_ks - eroded;
    cv::Mat bg = dilated - invdepth_ks;
    cv::Mat sc = dilated - eroded;

    // Eroding reduces artifats where edges are treated as a thin region of middle
    static constexpr int kErodeMid = 2;
    cv::Mat mde, bge;
    cv::erode(md, mde, cv::Mat(), cv::Point(-1, -1), kErodeMid);
    cv::erode(bg, bge, cv::Mat(), cv::Point(-1, -1), kErodeMid);

    cv::Mat fg(bg.size(), CV_32F);
    for (int y = 0; y < fg.rows; ++y) {
      for (int x = 0; x < fg.cols; ++x) {
        fg.at<float>(y, x) = std::sqrt(mde.at<float>(y, x) * bge.at<float>(y, x) + 1e-3);
      }
    }

    if (ks > kBigKernel) {
      cv::resize(fg, fg, full_size, 0, 0, cv::INTER_LINEAR);
      cv::resize(bg, bg, full_size, 0, 0, cv::INTER_LINEAR);
      cv::resize(md, md, full_size, 0, 0, cv::INTER_LINEAR);
      cv::resize(sc, sc, full_size, 0, 0, cv::INTER_LINEAR);
    }

    // cv::imwrite(debug_dir + "/fg_" + std::to_string(k) + ".png", fg * 255.0f);
    // cv::imwrite(debug_dir + "/bg_" + std::to_string(k) + ".png", bg * 255.0f);
    // cv::imwrite(debug_dir + "/md_" + std::to_string(k) + ".png", md * 10 * 255.0f);

    fgs.push_back(fg);
    bgs.push_back(bg);
    mds.push_back(md);
    scale.push_back(sc);
  }

  for (int y = 0; y < segmentation.rows; ++y) {
    for (int x = 0; x < segmentation.cols; ++x) {
      float sum_fg = 0;
      float sum_bg = 0;
      float sum_mid = 0;
      for (int k = 0; k < num_kernels; ++k) {
        const float far_awayness =
            std::max(0.0f, 1.0f - 20.0f * R_inv_depth_ftheta.at<float>(y, x));

        float s = scale[k].at<float>(y, x) + 0.01;
        const float fg = fgs[k].at<float>(y, x);
        const float bg = bgs[k].at<float>(y, x);
        const float md = mds[k].at<float>(y, x);

        sum_fg += fg * (0.01 + 0.3 / s);
        sum_bg += bg * (0.6 + 0.6 / s) + far_awayness * 0.5;
        sum_mid += md * (0.5 + 0.5 / s);
      }

      static constexpr float kSteepness = 2.0;
      sum_fg = std::exp(kSteepness * sum_fg);
      sum_bg = std::exp(kSteepness * sum_bg);
      sum_mid = std::exp(kSteepness * sum_mid);

      float sum_all = sum_fg + sum_bg + sum_mid;
      float score_fg = sum_fg / sum_all;
      float score_bg = sum_bg / sum_all;
      float score_mid = sum_mid / sum_all;

      //                                // layer   2         1          0
      //                                //         blue      green      red
      segmentation.at<cv::Vec3f>(y, x) = cv::Vec3f(score_fg, score_mid, score_bg);
    }
  }

  cv::Size double_size(full_size.width * 2, full_size.height * 2);
  cv::resize(segmentation, segmentation, double_size, 0.0, 0.0, cv::INTER_LINEAR);
  XPLINFO << "segmentation time: " << time::timeSinceSec(timer);
  return segmentation;
}

}}  // namespace p11::ldi
